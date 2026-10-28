"""Household: people, Hermes profiles, bound phones/devices, per-person memory and profile choice."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gaja.household import (
    UNKNOWN,
    Household,
    HouseholdStore,
    Identity,
    Person,
    Profile,
    explicit_profile,
)
from gaja.llm.client import TextDelta
from gaja.llm.router import Router, build_profile_prompt, option_masses
from gaja.memory import HOME, MemoryStore, set_memory_store
from gaja.pipeline import Session, _HermesJob, is_new_topic
from gaja.server import create_app
from gaja.tools.base import current_session_context
from gaja.tools.memory_tool import forget, remember
from gaja.tts.base import EnginePool
from tests.conftest import FakeHermes, FakeLLM, RecordingTTS
from tests.test_ws_protocol import is_listening, is_type, read_until

GAJA = Profile(id="default", label="Gaja", description="od wszystkiego")
SERZANT = Profile(id="serzant", label="Sierżant", description="lista zadań, plan dnia", aliases=["sierżancie"])
DOM = Profile(id="dom", label="Dom", description="sprawy domowe")
FINANSE = Profile(id="finanse", label="Finanse", description="budżet", sensitive=True)


def test_person_default_profile_follows_the_id_pattern() -> None:
    # F11b: default_profile builds <hermes_home>/profiles/<id> for the Spotify broker — an id
    # like "../x" or an absolute path would escape the profiles directory.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Person(id="x", name="X", default_profile="../x")
    with pytest.raises(ValidationError):
        Person(id="x", name="X", default_profile="/abs")
    with pytest.raises(ValidationError):
        Person(id="x", name="X", default_profile="")
    Person(id="x", name="X", default_profile="dom-2_")  # the ID_RE shape stays valid


def household() -> Household:
    return Household(
        people=[
            Person(id="tymek", name="Tymek", default_profile="default", profiles=["serzant", "finanse"]),
            Person(id="ola", name="Ola", default_profile="dom"),
        ],
        profiles=[GAJA, SERZANT, DOM, FINANSE],
        general_profile="dom",
        note="Mieszkamy w Katowicach.",
    )


def store(tmp_path: Path) -> HouseholdStore:
    s = HouseholdStore(tmp_path / "household.json")
    s.replace(household())
    return s


# ---- model and store --------------------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"general_profile": "nope"},
        {"people": [{"id": "x", "name": "X", "default_profile": "nope"}]},
        {"people": [{"id": "guest", "name": "Gość", "default_profile": "dom"}]},
        {"clients": {"abcdefgh": {"person": "nobody"}}},
        {"profiles": [GAJA.model_dump(), GAJA.model_dump()], "general_profile": "default"},
    ],
)
def test_household_rejects_bad_references(change: dict[str, Any]) -> None:
    data = household().model_dump() | change
    with pytest.raises(ValueError):
        Household.model_validate(data)


def test_household_rejects_a_sensitive_general_profile() -> None:
    # The general profile serves unknown voices: a sensitive one would hand them its data.
    data = household().model_dump() | {"general_profile": "finanse"}
    with pytest.raises(ValueError, match="czuły"):
        Household.model_validate(data)


def test_stored_sensitive_general_profile_still_loads(tmp_path: Path) -> None:
    # A file saved before the rule must not be dropped (the next save would then wipe it).
    path = tmp_path / "household.json"
    path.write_text(json.dumps(household().model_dump() | {"general_profile": "finanse"}), encoding="utf-8")
    assert [p.id for p in HouseholdStore(path).data.people] == ["tymek", "ola"]


def test_default_household_has_one_general_profile() -> None:
    h = Household()
    assert [p.id for p in h.profiles] == ["default"] and h.general_profile == "default"


def test_store_persists_and_keeps_client_timestamps(tmp_path: Path) -> None:
    s = store(tmp_path)
    s.touch_client("phone-tymek-1", "Pixel", 100.0)
    s.touch_client("phone-tymek-1", "Other label", 200.0)
    client = s.data.clients["phone-tymek-1"]
    assert (client.label, client.first_seen, client.last_seen) == ("Pixel", 100.0, 200.0)

    calls: list[int] = []
    s.on_change.append(lambda: calls.append(1))
    edited = s.data.model_dump()
    edited["clients"]["phone-tymek-1"] = {"person": "tymek", "label": "Telefon Tymka", "first_seen": 0, "last_seen": 0}
    s.replace(Household.model_validate(edited))
    assert calls == [1]
    reloaded = HouseholdStore(tmp_path / "household.json").data.clients["phone-tymek-1"]
    assert (reloaded.person, reloaded.label, reloaded.first_seen, reloaded.last_seen) == ("tymek", "Telefon Tymka", 100.0, 200.0)


def test_broken_file_starts_empty(tmp_path: Path) -> None:
    path = tmp_path / "household.json"
    path.write_text("{not json")
    assert HouseholdStore(path).data.people == []


def test_identify_by_client_then_device(tmp_path: Path) -> None:
    s = store(tmp_path)
    data = s.data.model_dump()
    data["clients"] = {"phone-tymek-1": {"person": "tymek"}, "phone-unbound": {}}
    data["devices"] = {"atom-kuchnia": {"name": "Kuchnia", "person": "ola"}, "atom-salon": {"name": "Salon"}}
    s.replace(Household.model_validate(data))
    assert s.identify(client_id="phone-tymek-1") == Identity(s.data.people[0], "client")
    assert s.identify(device_id="atom-kuchnia").person.id == "ola"
    assert s.identify(device_id="atom-kuchnia").source == "device"
    assert s.identify(client_id="phone-unbound") is UNKNOWN
    assert s.identify(device_id="atom-salon") is UNKNOWN
    assert s.identify() is UNKNOWN


def test_usable_profiles_hide_sensitive_ones_from_voice(tmp_path: Path) -> None:
    s = store(tmp_path)
    tymek, ola = s.data.people
    ids = lambda profiles: [p.id for p in profiles]  # noqa: E731
    assert ids(s.usable_profiles(Identity(tymek, "client"), voice=False)) == ["default", "serzant", "finanse"]
    assert ids(s.usable_profiles(Identity(tymek, "client"), voice=True)) == ["default", "serzant"]
    assert ids(s.usable_profiles(Identity(tymek, "device"), voice=False)) == ["default", "serzant"]
    assert ids(s.usable_profiles(Identity(ola, "device"), voice=True)) == ["dom"]
    assert ids(s.usable_profiles(UNKNOWN, voice=True)) == ["dom"]


def test_person_with_only_sensitive_profiles_falls_back_to_general(tmp_path: Path) -> None:
    data = household().model_dump()
    data["people"].append({"id": "kuba", "name": "Kuba", "default_profile": "finanse"})
    s = HouseholdStore(tmp_path / "h.json")
    s.replace(Household.model_validate(data))
    assert [p.id for p in s.usable_profiles(Identity(s.data.person("kuba"), "device"), voice=True)] == ["dom"]


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("Sierżancie, co mam dziś na liście?", "serzant"),
        ("zapytaj sierżanta o plan", "serzant"),
        ("Sierzant dodaj zadanie", "serzant"),
        ("Hej Gaja, jaka jest pogoda?", None),
        ("Gajo, sprawdź pogodę", None),
        ("co mamy w domu do jedzenia", None),
        ("dom, włącz światło", "dom"),
    ],
)
def test_explicit_profile(utterance: str, expected: str | None) -> None:
    hit = explicit_profile(utterance, [GAJA, SERZANT, DOM])
    assert (hit.id if hit else None) == expected


LISTA = Profile(id="lista", label="Lista", description="lista zadań, przypomnienia")
PLAN = Profile(id="plan", label="Plan", description="plan dnia")


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        # A 5-letter name's old stem ("Lista" -> "lis") must not hijack ordinary words.
        ("Co się dzieje w listopadzie?", None),
        ("Ile dni ma listopad?", None),
        ("Czy w ogrodzie jest lis?", None),
        ("Jaka jest pogoda?", None),
        # Real whole-word mentions (and their inflected forms) still match.
        ("Pokaż listę zadań", "lista"),
        ("Sprawdź listę", "lista"),
        ("co jest na liście?", "lista"),
        ("Jaki mam plan na dziś?", "plan"),
    ],
)
def test_explicit_profile_matches_only_the_names_forms(utterance: str, expected: str | None) -> None:
    hit = explicit_profile(utterance, [LISTA, PLAN, SERZANT])
    assert (hit.id if hit else None) == expected


# ---- router ----------------------------------------------------------------------------------


def test_profile_prompt_lists_options() -> None:
    prompt = build_profile_prompt("dodaj zakupy", [("Gaja", "od wszystkiego"), ("Sierżant", "lista zadań")])
    assert "1. Gaja: od wszystkiego\n2. Sierżant: lista zadań" in prompt
    assert prompt.endswith("Numer:")


def test_option_masses_normalises_digits() -> None:
    completion = {"completion_probabilities": [{"top_logprobs": [
        {"token": "2", "logprob": math.log(0.6)},
        {"token": " 1", "logprob": math.log(0.2)},
        {"token": "7", "logprob": math.log(0.1)},
        {"token": "x", "logprob": math.log(0.1)},
    ]}]}
    assert option_masses(completion, 2) == pytest.approx([0.25, 0.75])
    assert option_masses({}, 4) == [0.25] * 4


class DigitLLM(FakeLLM):
    """Answers the profile-choice prompt with fixed digit probabilities; otherwise a plain fake."""

    def __init__(self, digits: dict[str, float], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.digits = digits

    async def complete(self, prompt: str, *, timeout_s: float, slot: int | None = None) -> dict[str, Any]:
        if prompt.endswith("Numer:"):
            self.prompts.append(prompt)
            top = [{"token": k, "logprob": math.log(v)} for k, v in self.digits.items()]
            return {"completion_probabilities": [{"top_logprobs": top}]}
        return await super().complete(prompt, timeout_s=timeout_s, slot=slot)


@pytest.mark.asyncio
async def test_router_pick_profile() -> None:
    router = Router(DigitLLM({"2": 0.9, "1": 0.1}), "sys", threshold=0.9, history_turns=2, timeout_s=1.0)
    index, probability = await router.pick_profile("dodaj zadanie", [("Gaja", ""), ("Sierżant", "")])
    assert index == 1 and probability == pytest.approx(0.9)


# ---- memory scopes ---------------------------------------------------------------------------


def test_memory_scopes_and_prompt_block(tmp_path: Path) -> None:
    mem = MemoryStore(tmp_path / "memory.json")
    mem.add_fact("Kot ma na imię Mruczek")
    mem.add_fact("Tymek lubi kawę", scope="tymek")
    mem.add_fact("Ola gra na skrzypcach", scope="ola")
    block = mem.format_prompt_block("tymek")
    assert block.index("Fakty o rozmówcy:") < block.index("Fakty o domu:")
    assert "kawę" in block and "Mruczek" in block and "skrzypcach" not in block
    guest = mem.format_prompt_block()
    assert "Mruczek" in guest and "kawę" not in guest
    assert [f["text"] for f in mem.list_facts({"ola"})] == ["Ola gra na skrzypcach"]
    assert mem.forget_by_query("kawę", scopes={HOME, "ola"}) == []
    assert len(mem.forget_by_query("kawę", scopes={HOME, "tymek"})) == 1


def test_legacy_facts_count_as_household(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    path.write_text(json.dumps([{"id": "a1", "text": "Stary fakt", "created_at": "2026-01-01T00:00:00"}]))
    mem = MemoryStore(path)
    assert mem.list_facts({HOME})[0]["text"] == "Stary fakt"


@pytest.mark.asyncio
async def test_memory_tool_uses_the_speaker_scope(tmp_path: Path) -> None:
    mem = MemoryStore(tmp_path / "memory.json")
    set_memory_store(mem)
    token = current_session_context.set({"person": "tymek", "utterance": "zapamiętaj że lubię kawę"})
    try:
        await remember("lubię kawę")
        await remember("dla wszystkich, że kot ma na imię Mruczek")
        current_session_context.set({"person": "tymek", "utterance": "zapamiętaj dla wszystkich, że śmieci są we wtorek"})
        await remember("śmieci są we wtorek")
    finally:
        current_session_context.reset(token)
    scopes = {f["text"]: f["scope"] for f in mem.list_facts()}
    assert scopes == {"Użytkownik lubi kawę": "tymek", "Kot ma na imię Mruczek": HOME, "Śmieci są we wtorek": HOME}

    token = current_session_context.set({"person": "ola", "utterance": "zapomnij o kawie"})
    try:
        result = await forget("kawę")
    finally:
        current_session_context.reset(token)
    assert result.data["deleted"] == [] and len(mem.list_facts()) == 3


# ---- pipeline --------------------------------------------------------------------------------


def test_new_topic_intent() -> None:
    assert is_new_topic("Gaja, nowy temat")
    assert is_new_topic("nowy temat.")
    assert is_new_topic("hej gaju zacznijmy od nowa")
    assert is_new_topic("new topic")
    assert not is_new_topic("opowiedz mi o czymś na nowy temat")


def _hello(ws, client_id: str | None = None) -> None:
    msg: dict[str, Any] = {"type": "hello", "lang": "pl"}
    if client_id:
        msg |= {"client_id": client_id, "client_label": "Telefon"}
    ws.send_text(json.dumps(msg))
    read_until(ws, is_listening)


def _handoff(ws, text: str) -> dict[str, Any]:
    ws.send_text(json.dumps({"type": "text", "text": text}))
    msgs = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "done")
    return next(m for m in msgs if isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "started")


def _bound_services(make_services, tmp_path: Path, llm: FakeLLM, hermes: FakeHermes):
    services = make_services(llm=llm, hermes=hermes, router=True, cfg={"household": {"profile_threshold": 0.6}})
    s = store(tmp_path)
    data = s.data.model_dump()
    data["clients"] = {"phone-tymek-1": {"person": "tymek"}}
    s.replace(Household.model_validate(data))
    services.household = s
    services.memory_store = MemoryStore(tmp_path / "memory.json")
    services.memory_store.add_fact("Tymek lubi kawę", scope="tymek")
    return services


def test_bound_phone_picks_profiles(make_services, tmp_path: Path) -> None:
    llm = DigitLLM({"2": 0.95, "1": 0.05}, p_hermes=0.99)
    hermes = FakeHermes(result="Gotowe.")
    services = _bound_services(make_services, tmp_path, llm, hermes)
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        _hello(ws, "phone-tymek-1")
        started = _handoff(ws, "dodaj do listy zakupy")
        assert started["profile"] == "Sierżant"
        llm.digits = {"1": 0.5, "2": 0.5}
        _handoff(ws, "sprawdź pogodę w Katowicach")
        _handoff(ws, "Sierżancie, co mam dziś?")
    assert hermes.profiles == ["serzant", "default", "serzant"]
    assert all(i.person.id == "tymek" and i.source == "client" for i in hermes.identities)
    assert "Tymek" in hermes.instructions[0] and "Katowicach" in hermes.instructions[0]
    system = llm.calls[-1][0][0]["content"] if llm.calls else ""
    assert system == "" or ("Rozmawiasz z: Tymek." in system and "kawę" in system)


def test_unknown_client_uses_general_profile(make_services, tmp_path: Path) -> None:
    llm = DigitLLM({"2": 0.95}, p_hermes=0.99)
    hermes = FakeHermes(result="Gotowe.")
    services = _bound_services(make_services, tmp_path, llm, hermes)
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        _hello(ws, "phone-somebody")
        _handoff(ws, "Sierżancie, dodaj zadanie")
    assert hermes.profiles == ["dom"]
    assert hermes.identities[0] is UNKNOWN
    assert "phone-somebody" in services.household.data.clients
    assert not any(p.endswith("Numer:") for p in llm.prompts)


def _sensitive_general_services(make_services, tmp_path: Path, llm: FakeLLM | None = None, hermes: FakeHermes | None = None):
    """A household whose general profile is sensitive: an untrusted speaker may use no profile."""
    services = make_services(llm=llm, hermes=hermes, router=True, cfg={"household": {"profile_threshold": 0.6}})
    # A household saved before the general profile had to be non-sensitive: it still loads.
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(store(tmp_path).data.model_dump() | {"general_profile": "finanse"}), encoding="utf-8")
    services.household = HouseholdStore(path)
    assert services.household.data.general_profile == "finanse"
    return services


@pytest.mark.asyncio
async def test_choose_profile_returns_none_when_no_profile_is_usable(make_services, tmp_path: Path) -> None:
    """usable_profiles() is empty when every profile this speaker may use (the general one
    included) is sensitive and the turn is not trusted text from a bound phone: then no profile
    may be chosen, not the sensitive general one."""

    class NoSendOutbox:
        async def send_json(self, payload: dict[str, Any]) -> None: ...

        async def send_bytes(self, data: bytes) -> None: ...

    services = _sensitive_general_services(make_services, tmp_path)
    session = Session(services, NoSendOutbox())
    try:
        job = _HermesJob(id="j", task="zapytaj o saldo", language="pl")  # UNKNOWN speaker, typed turn
        assert services.household.usable_profiles(job.identity, voice=job.voice) == []
        assert await session._choose_profile(job) is None
        job.voice = True  # a voice turn from an unknown speaker is refused the same way
        assert await session._choose_profile(job) is None
    finally:
        await session.close()


def test_sensitive_general_profile_refuses_the_handoff(make_services, tmp_path: Path) -> None:
    """A turn with no usable profile must be refused (handoff error, nothing submitted to
    Hermes) instead of silently running the sensitive general profile."""
    llm = DigitLLM({"1": 0.5, "2": 0.5}, p_hermes=0.99)
    hermes = FakeHermes(result="Gotowe.")
    services = _sensitive_general_services(make_services, tmp_path, llm=llm, hermes=hermes)
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        _hello(ws, "phone-guest")
        ws.send_text(json.dumps({"type": "text", "text": "zapytaj o saldo"}))
        msgs = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "error")
    assert hermes.profiles == []  # the turn never reached Hermes
    error = next(m for m in msgs if isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "error")
    assert "profil" in error["text"].lower()


def test_new_topic_resets_hermes_sessions(make_services, tmp_path: Path) -> None:
    hermes = FakeHermes()
    services = _bound_services(make_services, tmp_path, FakeLLM(), hermes)
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        _hello(ws, "phone-tymek-1")
        ws.send_text(json.dumps({"type": "text", "text": "Gaja, nowy temat"}))
        msgs = read_until(ws, is_type("metrics"))
    assert hermes.resets == ["tymek"]
    assert any(isinstance(m, dict) and m.get("text") == "Dobrze, zaczynamy nowy temat." for m in msgs)


def test_without_household_hermes_gets_default_profile(make_services) -> None:
    llm = FakeLLM(p_hermes=0.99)
    hermes = FakeHermes(result="Gotowe.")
    app = create_app(services=make_services(llm=llm, hermes=hermes, router=True))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        _hello(ws)
        started = _handoff(ws, "wyślij maila")
    assert "profile" not in started
    assert hermes.profiles == ["default"] and hermes.instructions == [None]


# ---- per-person voices --------------------------------------------------------------------


def _voice_services(
    make_services,
    tmp_path: Path,
    *,
    llm: FakeLLM | None = None,
    hermes: FakeHermes | None = None,
    general_voice: str | None = None,
    tymek_tts: str | None = "fake-darkman",
    darkman_id: str = "fake-darkman",
) -> tuple[Any, RecordingTTS, RecordingTTS]:
    """Two fake engines ("fake" = the global default, `darkman_id` = Polish only) plus a household
    with Tymek (tts=`tymek_tts`) and Ola (no voice), Tymek's and Ola's phones bound."""
    services = make_services(llm=llm or FakeLLM([[TextDelta("Cześć!")], [TextDelta("Cześć!")]]), hermes=hermes or FakeHermes(result="Gotowe."), router=True)
    default = RecordingTTS("fake")
    darkman = RecordingTTS(darkman_id, languages=frozenset({"pl"}))
    services.tts = EnginePool({"fake": lambda: default, darkman_id: lambda: darkman})
    store = HouseholdStore(tmp_path / "household.json")
    data = household().model_dump()
    data["people"][0]["tts"] = tymek_tts
    data["general_voice"] = general_voice
    data["clients"] = {"phone-tymek-1": {"person": "tymek"}, "phone-ola-1": {"person": "ola"}}
    store.replace(Household.model_validate(data))
    services.household = store
    return services, default, darkman


def _text_turn(ws, text: str = "Dzień dobry") -> None:
    ws.send_text(json.dumps({"type": "text", "text": text}))
    read_until(ws, is_type("metrics"))


def _hello_as(ws, client_id: str, lang: str = "pl") -> None:
    ws.send_text(json.dumps({"type": "hello", "lang": lang, "client_id": client_id, "client_label": "Telefon"}))
    read_until(ws, is_listening)


def test_person_voice_is_used_for_their_turns(make_services, tmp_path: Path) -> None:
    services, default, darkman = _voice_services(make_services, tmp_path)
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        _hello_as(ws, "phone-tymek-1")
        _text_turn(ws)
        _hello_as(ws, "phone-nobody")
        _text_turn(ws)
    # Tymek's turn speaks with his voice, the unknown voice with the global default engine.
    assert darkman.spoken == [("Cześć!", "pl")]
    assert default.spoken == [("Cześć!", "pl")]


def test_general_voice_serves_unknown_and_voiceless_people(make_services, tmp_path: Path) -> None:
    services, default, darkman = _voice_services(make_services, tmp_path, general_voice="fake-darkman")
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        _hello_as(ws, "phone-nobody")
        _text_turn(ws)
        _hello_as(ws, "phone-ola-1")  # Ola has no voice of her own
        _text_turn(ws)
    assert darkman.spoken == [("Cześć!", "pl"), ("Cześć!", "pl")]
    assert default.spoken == []


def test_hermes_result_is_spoken_with_the_persons_voice(make_services, tmp_path: Path) -> None:
    llm = FakeLLM(p_hermes=0.99)
    hermes = FakeHermes(result="Pogoda: słonecznie.")
    services, default, darkman = _voice_services(make_services, tmp_path, llm=llm, hermes=hermes)
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        _hello_as(ws, "phone-tymek-1")
        _handoff(ws, "jaka jest pogoda?")
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "metrics" and m.get("kind") == "result")
    assert ("Pogoda: słonecznie.", "pl") in darkman.spoken
    assert default.spoken == []


def test_voice_that_cannot_speak_the_turn_language_falls_back(make_services, tmp_path: Path) -> None:
    # fake-darkman speaks Polish only (like the real piper-darkman the catalog describes).
    services, default, darkman = _voice_services(make_services, tmp_path, darkman_id="piper-darkman", tymek_tts="piper-darkman")
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        _hello_as(ws, "phone-tymek-1", lang="en")
        _text_turn(ws)
    assert darkman.spoken == []
    assert default.spoken == [("Cześć!", "en")]


def test_language_fallback_warns_once_per_voice_and_language(make_services, tmp_path: Path, caplog) -> None:
    import logging

    services, default, darkman = _voice_services(make_services, tmp_path, darkman_id="piper-darkman", tymek_tts="piper-darkman")
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        _hello_as(ws, "phone-tymek-1", lang="en")
        with caplog.at_level(logging.WARNING, logger="gaja.pipeline"):
            _text_turn(ws)
            _text_turn(ws)  # every sentence asks again; the journal hears it once
    assert default.spoken == [("Cześć!", "en"), ("Cześć!", "en")]
    assert len([r for r in caplog.records if "cannot speak" in r.getMessage()]) == 1


def test_voice_missing_from_the_engine_pool_falls_back(make_services, tmp_path: Path) -> None:
    # The household entry can be stale (config changed); the turn must not stall on it.
    services, default, darkman = _voice_services(make_services, tmp_path, tymek_tts="ghost")
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        _hello_as(ws, "phone-tymek-1")
        _text_turn(ws)
    assert darkman.spoken == []
    assert default.spoken == [("Cześć!", "pl")]


def test_voice_that_fails_to_build_falls_back_and_is_remembered(make_services, tmp_path: Path, caplog) -> None:
    """A household voice whose engine cannot build (e.g. a missing model file) must not kill the
    turn: fall back to the normal engine, log once (class name only -- the exception text can carry
    paths), and do not retry the build on every sentence."""
    import logging

    services = make_services(llm=FakeLLM([[TextDelta("Cześć!")], [TextDelta("Cześć!")]]))
    default = RecordingTTS("fake")
    builds: list[int] = []

    def broken() -> Any:
        builds.append(1)
        raise FileNotFoundError("supertonic model not downloaded")

    services.tts = EnginePool({"fake": lambda: default, "supertonic": broken})
    store = HouseholdStore(tmp_path / "household.json")
    store.replace(
        Household.model_validate(
            {"people": [{"id": "ala", "name": "Ala", "default_profile": "default", "tts": "supertonic"}],
             "clients": {"phone-ala": {"person": "ala"}}}
        )
    )
    services.household = store
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        _hello_as(ws, "phone-ala")
        with caplog.at_level(logging.WARNING, logger="gaja.pipeline"):
            _text_turn(ws)  # the turn is answered by the default engine despite the broken build
            _text_turn(ws)  # ...and the failed voice is not rebuilt for the next turn
    assert default.spoken == [("Cześć!", "pl"), ("Cześć!", "pl")]
    assert builds == [1]
    warned = [r for r in caplog.records if "could not be built" in r.getMessage()]
    assert len(warned) == 1
    assert "FileNotFoundError" in warned[0].getMessage()  # the class name is safe to log...
    assert "supertonic model not downloaded" not in warned[0].getMessage()  # ...the message is not
