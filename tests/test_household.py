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
from gaja.llm.router import Router, build_profile_prompt, option_masses
from gaja.memory import HOME, MemoryStore, set_memory_store
from gaja.pipeline import is_new_topic
from gaja.server import create_app
from gaja.tools.base import current_session_context
from gaja.tools.memory_tool import forget, remember
from tests.conftest import FakeHermes, FakeLLM
from tests.test_ws_protocol import is_listening, is_type, read_until

GAJA = Profile(id="default", label="Gaja", description="od wszystkiego")
SERZANT = Profile(id="serzant", label="Sierżant", description="lista zadań, plan dnia", aliases=["sierżancie"])
DOM = Profile(id="dom", label="Dom", description="sprawy domowe")
FINANSE = Profile(id="finanse", label="Finanse", description="budżet", sensitive=True)


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
