from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gaja.llm.client import LLMUnavailable, TextDelta
from gaja.llm.router import (
    Router,
    SPOTIFY_DEVICE_TRANSFER,
    TOOL_CUES,
    home_person_words,
    set_home_device_words,
    set_home_person_words,
    addresses_hermes,
    build_router_prompt,
    class_masses,
    media_handoff,
    p_hermes,
    parse_call,
    promises_handoff,
    recent_spotify_turn,
    spotify_fast_path,
    tool_grammar,
    volume_fast_path,
)
from gaja.pipeline import ACKS, Session
from gaja.server import create_app
from gaja.tools.registry import ToolRegistry, build_runtime_registry, default_registry
from gaja.tools.spotify_tool import set_spotify_playing
from tests.conftest import FakeHermes, FakeLLM
from tests.test_ws_protocol import is_listening, is_type, read_until


@pytest.mark.parametrize(
    "text",
    [
        "Przekażę to Hermesowi.",
        "przekazuję to hermesowi",
        "Przekażę to zadanie Hermesowi, chwilkę.",
        "Przekażę sprawę do Hermesa.",
        "I will hand this over to Hermes.",
        "Passing it to Hermes now.",
    ],
)
def test_safety_net_catches_handoff_promises(text: str) -> None:
    assert promises_handoff(text)


@pytest.mark.parametrize(
    "text",
    [
        "Kim jest Hermes w mitologii greckiej?",
        "Przekazuję informacje o Hermesie.",
        "Hermes to posłaniec bogów.",
        "Przekazałem plik.",
        "Who is Hermes in Greek mythology?",
    ],
)
def test_safety_net_ignores_mentions(text: str) -> None:
    assert not promises_handoff(text)


def test_build_router_prompt_appends_history_then_utterance() -> None:
    history = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    prompt = build_router_prompt("ROUTER", "cześć", history[1:])
    assert prompt.startswith("<|im_start|>system\nROUTER<|im_end|>\n")
    assert "<|im_start|>user\nZapytanie: a<|im_end|>\n<|im_start|>assistant\nb<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>user\nZapytanie: cześć<|im_end|>\n<|im_start|>assistant\nKategoria:")


def test_build_router_prompt_shortens_a_giant_utterance_like_history() -> None:
    # The utterance is spliced raw no longer: a paste of thousands of characters must not blow the
    # one-token classification prompt.
    from gaja.llm.router import ROUTER_MESSAGE_CHARS

    prompt = build_router_prompt("ROUTER", "a" * 1000, [])
    assert f"Zapytanie: {'a' * (ROUTER_MESSAGE_CHARS - 1)}…<|im_end|>" in prompt
    assert "a" * ROUTER_MESSAGE_CHARS not in prompt


@pytest.mark.asyncio
async def test_router_prime_without_history_sends_the_system_prefix_and_user_header() -> None:
    llm = FakeLLM()
    router = Router(llm, "ROUTER", threshold=0.97, history_turns=4, timeout_s=1.0)
    await router.prime()
    assert llm.primed == ["<|im_start|>system\nROUTER<|im_end|>\n<|im_start|>user\n"]
    assert llm.prompts == []
    assert build_router_prompt("ROUTER", "cześć", []).startswith(llm.primed[0])


@pytest.mark.asyncio
async def test_router_prime_with_history_is_an_exact_prefix_of_the_next_classify_prompt() -> None:
    llm = FakeLLM()
    router = Router(llm, "ROUTER", threshold=0.97, history_turns=2, timeout_s=1.0)
    history = [
        {"role": "user", "content": "stare"},
        {"role": "assistant", "content": "stara odpowiedź"},
        {"role": "user", "content": "jak się masz?"},
        {"role": "assistant", "content": "dobrze " * 80},
    ]
    await router.prime(history)
    await router.classify("a pogoda?", history)
    assert llm.prompts[0].startswith(llm.primed[0])
    assert "stare" not in llm.primed[0]  # only the last history_turns messages
    assert "Zapytanie: jak się masz?" in llm.primed[0]


def test_p_hermes_normalises_h_and_l_mass() -> None:
    completion = {
        "completion_probabilities": [
            {
                "top_logprobs": [
                    {"token": " Hermes", "logprob": math.log(0.97)},
                    {"token": " Lokalnie", "logprob": math.log(0.03)},
                    {"token": " Kat", "logprob": -9.0},
                ]
            }
        ]
    }
    assert p_hermes(completion) == pytest.approx(0.97)


def test_p_hermes_without_candidates_is_zero() -> None:
    assert p_hermes({"completion_probabilities": [{"top_logprobs": []}]}) == 0.0


def test_router_threshold_is_inclusive() -> None:
    router = Router(FakeLLM(), "sys", threshold=0.97, history_turns=2, timeout_s=2.0)
    assert router.hands_off(0.97)
    assert not router.hands_off(0.969)


def test_router_p_hermes_uses_recent_history_only() -> None:
    llm = FakeLLM(p_hermes=0.99)
    router = Router(llm, "sys", threshold=0.97, history_turns=2, timeout_s=2.0)
    history = [{"role": "user", "content": "stare"}, {"role": "assistant", "content": "odpowiedź"}, {"role": "user", "content": "nowe"}]
    probability = asyncio.run(router.p_hermes("pogoda", history))
    assert probability == pytest.approx(0.99)
    assert "stare" not in llm.prompts[0]
    assert "Zapytanie: pogoda<|im_end|>" in llm.prompts[0]


def test_router_routes_to_hermes_without_local_reply(make_services) -> None:
    llm = FakeLLM(p_hermes=0.99)
    hermes = FakeHermes(result="Pogoda: dwanaście stopni.")
    app = create_app(services=make_services(llm=llm, hermes=hermes, router=True))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "sprawdź pogodę w Krakowie"}))
        messages = read_until(ws, is_type("metrics"))
        spoken = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "audio_end" and m["id"] == 2)
    assert llm.calls == []
    assert hermes.tasks == ["sprawdź pogodę w Krakowie"]
    utterances = [m["text"] for m in messages + spoken if isinstance(m, dict) and m.get("type") == "utterance"]
    assert utterances[0] in ACKS["pl"]
    assert utterances[-1] == "Pogoda: dwanaście stopni."
    assert messages[-1]["router_ms"] is not None


def test_router_below_threshold_keeps_local_reply(make_services) -> None:
    llm = FakeLLM([[TextDelta("Dwa plus dwa to cztery.")]], p_hermes=0.2)
    hermes = FakeHermes()
    app = create_app(services=make_services(llm=llm, hermes=hermes, router=True))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "ile to dwa plus dwa"}))
        messages = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "metrics")
    assert [allow for _, allow in llm.calls] == [True]
    assert hermes.tasks == []
    utterances = [m["text"] for m in messages if isinstance(m, dict) and m.get("type") == "utterance"]
    assert utterances == ["Dwa plus dwa to cztery."]


def test_router_outage_falls_back_to_local_reply(make_services) -> None:
    llm = FakeLLM([[TextDelta("Dobrze.")]], down=True)
    app = create_app(services=make_services(llm=llm, router=True))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "cześć"}))
        messages = read_until(ws, is_listening)
    errors = [m for m in messages if isinstance(m, dict) and m.get("type") == "error"]
    assert errors == [{"type": "error", "message": "language model unavailable"}]


def test_safety_net_turns_spoken_promise_into_handoff(make_services) -> None:
    llm = FakeLLM([[TextDelta("Przekażę to Hermesowi, chwilkę. ")]])
    hermes = FakeHermes(result="Gotowe.")
    app = create_app(services=make_services(llm=llm, hermes=hermes))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "sprawdź dla mnie coś"}))
        messages = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "done")
        spoken = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "audio_end" and m["id"] == 2)
    utterances = [m["text"] for m in messages + spoken if isinstance(m, dict) and m.get("type") == "utterance"]
    assert utterances[0] in ACKS["pl"]
    assert not any(u.startswith("Przekażę") for u in utterances)
    assert hermes.tasks == ["sprawdź dla mnie coś"]
    assert utterances[-1] == "Gotowe."


def test_ack_rotates_across_handoffs(make_services) -> None:
    session = Session(make_services(), None)  # type: ignore[arg-type]
    seen = [session._next_ack("pl") for _ in range(len(ACKS["pl"]) + 1)]
    assert seen[: len(ACKS["pl"])] == list(ACKS["pl"])
    assert seen[-1] == ACKS["pl"][0]


def test_router_raises_when_llm_is_down() -> None:
    router = Router(FakeLLM(down=True), "sys", threshold=0.97, history_turns=2, timeout_s=2.0)
    with pytest.raises(LLMUnavailable):
        asyncio.run(router.p_hermes("cześć", []))


def _completion(local: float, tool: float, hermes: float) -> dict:
    top = [{"token": t, "logprob": math.log(p)} for t, p in ((" LOCAL", local), (" TOOL", tool), (" HERMES", hermes)) if p > 0]
    return {"completion_probabilities": [{"top_logprobs": top}]}


def test_class_masses_split_three_ways() -> None:
    local, tool, hermes = class_masses(_completion(0.2, 0.3, 0.5))
    assert (local, tool, hermes) == pytest.approx((0.2, 0.3, 0.5))
    assert class_masses({}) == (1.0, 0.0, 0.0)


TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}},
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "when": {"type": "string"}}},
        },
    },
]


def _router(llm: FakeLLM, tools=TOOL_SCHEMAS) -> Router:
    return Router(llm, "sys", threshold=0.9, history_turns=2, timeout_s=2.0, tools=tools, tool_threshold=0.5)


@pytest.mark.parametrize(
    "text",
    ["zapytaj Hermesa jaki jest dziś dzień tygodnia", "Hermes, sprawdź pocztę", "ask Hermes about my calendar", "niech Hermes to zrobi",
     "Sprawdź w Hermesie, tylko odczytaj, pogodę w Krakowie na jutro.", "zrób to przez Hermesa", "check it via Hermes"],
)
def test_explicit_hermes_request_skips_the_model(text: str) -> None:
    llm = FakeLLM(p_hermes=0.0)
    route = asyncio.run(_router(llm).classify(text, []))
    assert route.kind == "hermes" and route.reason == "explicit"
    assert llm.prompts == []


@pytest.mark.parametrize(
    "text",
    ["kim jest Hermes w mitologii", "kim był Hermes", "opowiedz o Hermesie, bogu greckim", "who is Hermes in Greek mythology"],
)
def test_mythology_hermes_is_never_a_handoff(text: str) -> None:
    llm = FakeLLM(p_hermes=0.99)
    route = asyncio.run(_router(llm).classify(text, []))
    assert route.kind != "hermes"
    assert not addresses_hermes(text)
    assert not promises_handoff(f"{text}? To był posłaniec bogów, przekażę to Hermesowi")


def test_classify_three_classes() -> None:
    assert asyncio.run(_router(FakeLLM(p_tool=0.8)).classify("która godzina", [])).kind == "tool"
    assert asyncio.run(_router(FakeLLM(p_tool=0.3)).classify("cześć", [])).kind == "local"
    assert asyncio.run(_router(FakeLLM(p_hermes=0.95)).classify("wyślij maila", [])).kind == "hermes"


def test_tool_mass_counts_as_local_without_tools() -> None:
    route = asyncio.run(_router(FakeLLM(p_tool=0.8), tools=None).classify("która godzina", []))
    assert route.kind == "local"
    assert route.p_local == pytest.approx(1.0)


def test_tool_grammar_lists_tools_and_parameters() -> None:
    grammar = tool_grammar(TOOL_SCHEMAS)
    assert 'call ::= t0 | t1' in grammar
    assert 't0 ::= "get_time()"' in grammar
    assert 'a1 ::= ("location=" (str | num)) | ("when=" (str | num))' in grammar


def test_tool_grammar_restricts_enum_parameters() -> None:
    schemas = [{"function": {"name": "w", "parameters": {"properties": {"when": {"type": "string", "enum": ["now", "today"]}}}}}]
    assert 'a0 ::= ("when=" ("\\"now\\"" | "\\"today\\""))' in tool_grammar(schemas)


def test_parse_call_accepts_only_known_tools_with_literal_arguments() -> None:
    allowed = {"get_weather", "get_time"}
    call = parse_call(' get_weather(location="Kraków", when="tomorrow")', allowed)
    assert call is not None and call.name == "get_weather"
    assert call.arguments == {"location": "Kraków", "when": "tomorrow"}
    assert parse_call("get_time()", allowed).arguments == {}
    assert parse_call("rm_rf()", allowed) is None
    assert parse_call("get_weather(location=x)", allowed) is None
    assert parse_call("get_weather(", allowed) is None


def test_select_tool_continues_the_router_prompt_with_a_grammar() -> None:
    llm = FakeLLM(tool_call=' get_weather(location="Gdańsk")')
    call = asyncio.run(_router(llm).select_tool("pogoda w Gdańsku", []))
    assert call is not None and call.arguments == {"location": "Gdańsk"}
    assert llm.generated[0].endswith("Zapytanie: pogoda w Gdańsku<|im_end|>\n<|im_start|>assistant\nKategoria: TOOL\nWywołanie:")


def test_router_history_messages_are_shortened() -> None:
    long = "słowo " * 200
    prompt = build_router_prompt("sys", "a teraz?", [{"role": "assistant", "content": long}])
    assistant = prompt.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0]
    assert len(assistant) <= 240
    assert assistant.endswith("…")


@pytest.mark.parametrize(
    "text",
    ["napisz w pythonie funkcję która liczy silnię", "włącz klimatyzację w sypialni", "przypomnij mi o spotkaniu", "write a python script that renames files"],
)
def test_action_requests_go_to_hermes_even_when_the_model_says_tool(text: str) -> None:
    route = asyncio.run(_router(FakeLLM(p_tool=0.95, p_hermes=0.04)).classify(text, []))
    assert route.kind == "hermes" and route.reason == "action"


@pytest.mark.parametrize("text", ["napisz krótki wierszyk o kocie", "przypomnij mi co to jest fotosynteza", "opowiedz dowcip"])
def test_creative_and_recall_requests_are_not_actions(text: str) -> None:
    route = asyncio.run(_router(FakeLLM(p_tool=0.1)).classify(text, []))
    assert route.kind == "local"


def test_messenger_is_not_mythology() -> None:
    llm = FakeLLM(p_hermes=0.99)
    assert asyncio.run(_router(llm).classify("napisz do Ani na messengerze", [])).kind == "hermes"


@pytest.mark.parametrize(
    ("call", "text", "history", "ok"),
    [
        ('get_weather(location="Polska")', "kiedy była bitwa pod Grunwaldem", [], False),
        ('get_weather(location="Kraków")', "a w Krakowie?", [{"role": "user", "content": "jaka jest pogoda w Gdańsku"}], True),
        ('get_weather(location="Zakopane", when="today")', "jak będzie w Zakopanem w weekend", [], True),
        (
            'get_weather(location="Berlin")',
            "a w Berlinie?",
            [
                {"role": "user", "content": "a w Gdańsku jutro?"},
                {"role": "assistant", "content": "Gdańsk, jutro: od dziesięciu do piętnastu stopni, pochmurno."},
            ],
            True,
        ),
        ("get_time()", "ile wydałem w zeszłym miesiącu", [], False),
        ('calculate(expression="20 + 33")', "policz dwadzieścia plus trzydzieści pięć", [], False),
        ('calculate(expression="17 * 23")', "ile to 17 razy 23", [], True),
        ('exchange_rate(currency="USD")', "ile kosztuje subskrypcja netflixa", [], False),
        ("server_status()", "jak się czuje serwer", [], True),
    ],
)
def test_implausible_tool_calls_are_dropped(call: str, text: str, history: list, ok: bool) -> None:
    llm = FakeLLM(tool_call=call)
    result = asyncio.run(_router(llm, tools=TOOL_SCHEMAS + [
        {"function": {"name": n, "parameters": {"properties": {"expression": {}, "currency": {}}}}}
        for n in ("calculate", "exchange_rate", "server_status")
    ]).select_tool(text, history))
    assert (result is not None) == ok


def test_home_sensors_go_to_hermes() -> None:
    route = asyncio.run(_router(FakeLLM(p_tool=0.98)).classify("jaka jest temperatura w salonie", []))
    assert route.kind == "hermes"


@pytest.mark.asyncio
async def test_select_tool_skips_generation_when_no_tool_cue_matches() -> None:
    llm = FakeLLM()
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"properties": {"location": {"type": "string"}}}}}]
    router = Router(llm, "ROUTER", threshold=0.5, history_turns=2, timeout_s=1.0, tools=tools)
    assert await router.select_tool("Podróżni zwiedzający państwa obłożone wysokimi górami", []) is None
    assert llm.generated == []


@pytest.mark.parametrize(
    "text",
    ["Zlecę to Hermesowi.", "Poproszę Hermesa, żeby to sprawdził.", "Hermes się tym zajmie.", "Już to sprawdza Hermes."],
)
def test_promise_phrasings_from_review_m8(text: str) -> None:
    from gaja.llm.router import promises_handoff

    assert promises_handoff(text)


@pytest.mark.parametrize("text", ["Hermes był posłańcem bogów.", "Kim jest Hermes w mitologii greckiej?", "Sprawdza się w praktyce."])
def test_promise_negatives(text: str) -> None:
    from gaja.llm.router import promises_handoff

    assert not promises_handoff(text)


def test_weather_cue_for_jak_bedzie_needs_a_place_name_and_a_recent_weather_turn() -> None:
    from gaja.llm.client import ToolCall
    from gaja.llm.router import plausible

    call = ToolCall(name="get_weather", arguments={})
    assert plausible(call, "jak będzie w Zakopanem w weekend", [])
    assert not plausible(call, "jak jest w pracy u ciebie", [])
    weather = [{"role": "user", "content": "jaka pogoda"}, {"role": "assistant", "content": "Słonecznie, 20 stopni."}]
    other = [{"role": "user", "content": "kim był Kopernik"}, {"role": "assistant", "content": "Astronomem."}]
    assert plausible(call, "a w Gdańsku?", weather)
    assert not plausible(call, "a w Gdańsku?", weather + other)


def test_spotify_fast_path_and_volume_disambiguation() -> None:
    set_spotify_playing(False)
    try:
        # Bare volume stays on the Atom speaker when there was no recent Spotify turn
        assert spotify_fast_path("głośniej", []) is None
        assert volume_fast_path("głośniej") is not None
        assert volume_fast_path("głośniej").name == "set_speaker_volume"

        # Explicit music/Spotify volume goes to spotify
        assert spotify_fast_path("muzykę głośniej", []).arguments == {"action": "volume_up"}
        assert spotify_fast_path("Spotify głośniej", []).arguments == {"action": "volume_up"}
        assert spotify_fast_path("weź muzykę trochę ciszej", []).arguments == {"action": "volume_down"}
        assert spotify_fast_path("głośność muzyki na 45", []).arguments == {"action": "volume_set", "level": 45}

        # Bare "głośniej"/"ciszej" right after a local spotify tool turn (its reply carries the
        # "Spotify:" history marker) goes to spotify
        spot_hist = [
            {"role": "user", "content": "puść Dawida Podsiadło na Spotify"},
            {"role": "assistant", "content": "Spotify: Włączyłam Dawid Podsiadło."},
        ]
        llm = FakeLLM(p_tool=0.9)
        router = Router(
            llm, "sys", threshold=0.9, history_turns=2, timeout_s=2.0,
            tools=default_registry.schemas(), tool_threshold=0.5, spotify_ready=lambda: True,
        )
        route = asyncio.run(router.classify("głośniej", spot_hist))
        assert route.kind == "tool" and route.reason == "fast-path"
        selected = asyncio.run(router.select_tool("głośniej", spot_hist))
        assert selected is not None and selected.name == "spotify" and selected.arguments == {"action": "volume_up"}

        # Without recent Spotify turn, router picks set_speaker_volume for bare "głośniej"
        selected_atom = asyncio.run(router.select_tool("głośniej", []))
        assert selected_atom is not None and selected_atom.name == "set_speaker_volume"

        # "dalej" requires active Spotify playback or recent Spotify turn
        assert spotify_fast_path("dalej", [], playing=False) is None
        assert spotify_fast_path("dalej", [], playing=True).arguments == {"action": "next"}
        assert spotify_fast_path("pauza", []).arguments == {"action": "pause"}
        assert spotify_fast_path("następna piosenka", []).arguments == {"action": "next"}

        # Device transfer goes to Hermes, not spotify_fast_path
        assert spotify_fast_path("przełącz Spotify na głośnik w salonie", []) is None
    finally:
        set_spotify_playing(False)


def test_genre_words_are_whole_word_metallica_is_an_artist() -> None:
    """NEW-7: GENERIC_PLAY_REQUEST used 'metall?\\w*', so 'puść Metallicę' read as a *genre* job and
    went to Hermes although the local play fast path can serve a named artist. Whole-word 'metal'
    keeps the genre hand-off while artist names ride the fast path (main@7741180 behaviour)."""
    schemas = default_registry.schemas()
    ready = Router(
        FakeLLM(p_tool=0.92, p_hermes=0.05), "sys", threshold=0.9, history_turns=2, timeout_s=2.0,
        tools=schemas, tool_threshold=0.5, spotify_ready=lambda: True,
    )
    route = asyncio.run(ready.classify("puść Metallicę", []))
    assert route.kind == "tool" and route.reason == "fast-path", route
    call = asyncio.run(ready.select_tool("puść Metallicę", []))
    assert call is not None and call.name == "spotify"
    assert call.arguments["action"] == "play" and call.arguments["query"].lower().startswith("metallic")
    # The genre itself stays a generic job: Spotify search cannot honour "puść metal" as a mood.
    assert asyncio.run(ready.classify("puść metal", [])).kind == "hermes"


@pytest.mark.parametrize(
    "text",
    [
        "zamknij się",
        "otwórz się sezamie",
        "otworz sie",
    ],
)
def test_reflexive_close_and_open_stay_local(text: str) -> None:
    """NEW-12: the merged branch added 'zamknij|otwórz' to HERMES_ACTION (blinds/doors are Hermes
    jobs), which dragged reflexive 'zamknij się'/'otwórz się sezamie' to Hermes too. main@7741180
    (no zamknij/otwórz at all) answered these locally; the 'się' lookahead restores that."""
    router = _router(FakeLLM(p_hermes=0.05, p_tool=0.05))
    route = asyncio.run(router.classify(text, []))
    assert route.kind == "local", (text, route)


def test_close_and_open_objects_still_go_to_hermes() -> None:
    router = _router(FakeLLM(p_hermes=0.05, p_tool=0.05))
    for text in ("zamknij rolety w salonie", "otwórz bramę garażową"):
        route = asyncio.run(router.classify(text, []))
        assert route.kind == "hermes" and route.reason == "action", (text, route)


@pytest.mark.parametrize(
    "text",
    [
        "puść film",
        "puść serial",
        "puść bajkę",
        "puść radio",
        "puść budzik",
        "puść alarm",
        "puść minutnik",
        "puść timer",
        "puść wiadomość",
        "puść przypomnienie",
        "puść film na youtube",
    ],
)
def test_play_fast_path_rejects_non_music_objects(text: str) -> None:
    """NEW-4: the play fast path accepted any object after 'puść', so 'puść film'/'puść budzik'
    became spotify search jobs. Video, radio, fairy-tale, alarm, timer and message nouns must go
    through the normal router (or Hermes), never to spotify search."""
    assert spotify_fast_path(text) is None, text


def test_m3_play_fast_path_word_boundaries_and_artists() -> None:
    """M3: NON_MUSIC_PLAY_OBJECT must not prefix-match music artists (Radiohead, Alarm Will Sound,
    Disney) while rejecting non-music objects with inflections (nagranie, bajeczkę)."""
    for utt, query in [
        ("puść Radiohead", "Radiohead"),
        ("puść Alarm Will Sound", "Alarm Will Sound"),
        ("puść Disney", "Disney"),
    ]:
        call = spotify_fast_path(utt)
        assert call is not None and call.arguments == {"action": "play", "query": query}, (utt, call)

    for utt in ["puść nagranie", "puść bajeczkę"]:
        assert spotify_fast_path(utt) is None, utt


def test_play_fast_path_still_accepts_music_and_artists() -> None:
    call = spotify_fast_path("puść Dawida Podsiadło")
    assert call is not None and call.arguments == {"action": "play", "query": "Dawida Podsiadło"}
    call = spotify_fast_path("puść muzykę")
    assert call is not None and call.arguments["action"] == "play"


@pytest.mark.parametrize(
    "text, query",
    [("puść film", "film"), ("puść serial", "serial"), ("puść bajkę", "bajkę"), ("puść budzik", "budzik")],
)
def test_model_selected_spotify_play_rejects_non_music_objects(text: str, query: str) -> None:
    """F7: the NON_MUSIC_PLAY_OBJECT veto lived only in spotify_fast_path. The model path reaches
    the same spotify(play, ...) through plausible()'s "puść" cue, so "puść film" became a spotify
    search when the small model picked the tool. The veto must apply to the selected call too."""
    llm = FakeLLM(tool_call=f' spotify(action="play", query="{query}")')
    router = Router(
        llm, "sys", threshold=0.9, history_turns=2, timeout_s=2.0,
        tools=default_registry.schemas(), tool_threshold=0.5, spotify_ready=lambda: True,
    )
    assert asyncio.run(router.select_tool(text, [])) is None, text


def test_model_selected_spotify_play_keeps_music_objects() -> None:
    """The veto is about the object, not the path: a music query selected by the model still runs."""
    llm = FakeLLM(tool_call=' spotify(action="play", query="muzyka relaksacyjna")')
    router = Router(
        llm, "sys", threshold=0.9, history_turns=2, timeout_s=2.0,
        tools=default_registry.schemas(), tool_threshold=0.5, spotify_ready=lambda: True,
    )
    call = asyncio.run(router.select_tool("włącz muzykę", []))
    assert call is not None and call.arguments == {"action": "play", "query": "muzyka relaksacyjna"}


@pytest.mark.parametrize(
    "text",
    [
        "napisz maila do Anny o klimatyzacji",
        "napisz maila o telewizorze",
        "wyślij sms o klimie w salonie",
        "ustaw przypomnienie o włączeniu klimy",
        "ustaw budzik na siódmą u Ani",
        "odpisz na wiadomość o amplitunerze",
    ],
)
def test_message_alarm_guard_beats_home_device_cues(text: str) -> None:
    """NEW-5: the HERMES_MESSAGE_ALARM guard is load-bearing. A message/alarm object in one
    utterance with a home device noun ('o klimatyzacji', 'klimy', 'u Ani') must stay a Hermes
    action: TOOL_CUES['home'] and HOME_OVERRIDE_DEVICE both match, so without the guard the merged
    branch would route the turn to the home tool instead of sending the message."""
    from gaja.llm.router import local_tool_override

    assert not local_tool_override(text, set(default_registry.names())), text
    router = _router(FakeLLM(p_hermes=0.05, p_tool=0.92), tools=default_registry.schemas())
    route = asyncio.run(router.classify(text, []))
    assert route.kind == "hermes" and route.reason == "action", (text, route)


@pytest.mark.parametrize(
    "text",
    [
        "puść Muse na komputerze",
        "puść Dawida Podsiadło na komputerze",
        "włącz Spotify na laptopie",
        "puść coś na telefonie",
        "puść film na pececie",
        "włącz muzykę na pc",
        "przełącz spotify na słuchawki",
        "puść playlistę na telefonie",
        "odtwórz Muse na komputerze",
        "wlacz spotify na laptopie",
    ],
)
def test_pc_and_phone_playback_targets_go_to_hermes(text: str) -> None:
    """NEW-1: an explicit non-household playback target (computer, PC, laptop, phone, headphones)
    is Hermes' job even with a ready local spotify tool: the local tool can only play on the
    household device. Before, 'puść X na komputerze' hit the play fast path and played there."""
    schemas = default_registry.schemas()
    ready = Router(
        FakeLLM(p_hermes=0.05, p_tool=0.92), "sys", threshold=0.9, history_turns=2, timeout_s=2.0,
        tools=schemas, tool_threshold=0.5, spotify_ready=lambda: True,
    )
    route = asyncio.run(ready.classify(text, []))
    assert route.kind == "hermes", (text, route)
    assert spotify_fast_path(text) is None, text
    from gaja.llm.router import local_tool_override

    assert not local_tool_override(text, {"spotify", "home"}), text
    # Ready or not: without a local tool every request is Hermes' anyway.
    not_ready = _router(FakeLLM(p_hermes=0.05, p_tool=0.92), tools=schemas)
    assert asyncio.run(not_ready.classify(text, [])).kind == "hermes", text


@pytest.mark.parametrize(
    "text",
    [
        "puść Dawida Podsiadło na moim komputerze",
        "puść Muse na swoim telefonie",
        "puść Muse na desktopie",
        "puść Muse na kompie",
        "puść Muse na macbooku",
        "puść Muse na iPhonie",
        "puść Muse na tablecie",
        "puść Muse na AirPodsach",
    ],
)
def test_new1_device_transfer_leak_targets_and_possessives(text: str) -> None:
    """NEW-1 leak: non-household playback targets with possessive pronouns, comp, macbook,
    desktop, iphone, tablet, airpods must route to Hermes and bypass spotify fast path."""
    schemas = default_registry.schemas()
    ready = Router(
        FakeLLM(p_hermes=0.05, p_tool=0.92), "sys", threshold=0.9, history_turns=2, timeout_s=2.0,
        tools=schemas, tool_threshold=0.5, spotify_ready=lambda: True,
    )
    route = asyncio.run(ready.classify(text, []))
    assert route.kind == "hermes", (text, route)
    assert spotify_fast_path(text) is None, text


def test_kompozycje_chopina_not_pc_target() -> None:
    """NEW-1: komp\\w* was too broad (matching kompozycja/kompakt). Whole-word komp inflections
    ensure 'puść kompozycje Chopina' is not treated as a PC target."""
    assert not SPOTIFY_DEVICE_TRANSFER.search("puść kompozycje Chopina")
    call = spotify_fast_path("puść kompozycje Chopina")
    assert call is not None and call.arguments == {"action": "play", "query": "kompozycje Chopina"}


def test_new1_bare_targets_never_trigger_the_spotify_fast_path() -> None:
    """NEW-1: the bare target phrases themselves must never trigger the spotify fast path (only a
    play/media phrase around them may hand off via SPOTIFY_DEVICE_TRANSFER)."""
    for text in ("na desktopie", "na kompie", "na macbooku", "na iPhonie", "na tablecie", "na AirPodsach"):
        assert spotify_fast_path(text) is None, text
        assert not SPOTIFY_DEVICE_TRANSFER.search(text), text


def test_household_speakers_still_transfer_on_the_existing_cues() -> None:
    assert SPOTIFY_DEVICE_TRANSFER.search("przełącz spotify na głośnik w salonie")
    assert SPOTIFY_DEVICE_TRANSFER.search("puść muzykę w salonie")


# Fixed table imported from main@7741180's media_handoff: `git show main:gaja/llm/router.py`
# was loaded as a module and its media_handoff run over this corpus to record every value below.
# NEW-2: with
# spotify_ready=False the merged branch must answer exactly like main — the wider PLAYBACK_VERB
# (zatrzymaj/wstrzymaj/pusc) and the bare "playback verb => Hermes" rule regressed timer/alarm/
# film/message/game utterances to Hermes while main kept them local.
MAIN_MEDIA_HANDOFF: dict[str, bool] = {
    "zatrzymaj minutnik": False,
    "wstrzymaj timer": False,
    "wznów minutnik": False,
    "pauza minutnika": False,
    "pauza": False,
    "wznów": False,
    "puść film": False,
    "zagraj w szachy": False,
    "odtwórz wiadomość": False,
    "puść budzik": False,
    "zagraj mi coś na gitarze": False,
    "play": False,
    "zatrzymaj": False,
    "zatrzymaj muzykę": False,
    "puść bajkę": False,
    "puść minutnik": False,
    "odtwórz film": False,
    "pauza na spotify": True,
    "puść muzykę": True,
    "puść coś na Spotify": True,
    "puść Muse na Spotify": True,
    "wycisz Spotify": True,
    "puść muzykę głośniej": True,
    "zagraj coś głośniej": True,
    "puść radio": True,
    "pauza muzyki": True,
    "wznów muzykę": True,
    "odtwórz playlistę": True,
    "pusc muzyke": True,
    "puść mi jakąś muzykę": True,
}


@pytest.mark.parametrize("text,expected", sorted(MAIN_MEDIA_HANDOFF.items()))
def test_media_handoff_matches_main_when_spotify_not_ready(text: str, expected: bool) -> None:
    """NEW-2: spotify_ready=False is the live state until the first Spotify login and the state
    the router-down path used unconditionally; media_handoff must keep main@7741180's answers."""
    assert media_handoff(text, [], spotify_ready=False) is expected, text


def test_media_handoff_hermes_result_followup_matches_main() -> None:
    hermes_result = [{"role": "user", "content": "Wynik zadania od Hermesa:\nPuszczam Muse na Desktopie."}]
    assert media_handoff("daj to głośniej", hermes_result, spotify_ready=False) is True
    assert media_handoff("puść muzykę głośniej", hermes_result, spotify_ready=False) is True
    assert media_handoff("daj to głośniej", [], spotify_ready=False) is False


@pytest.mark.parametrize(
    "text",
    [
        "muzyka do pracy",
        "spokojną muzykę",
        "włącz jazz",
        "puść jazz",
        "puść mi jakąś muzykę na spacer",
    ],
)
def test_generic_music_requests_are_the_only_new_not_ready_handoffs(text: str) -> None:
    """NEW-2 exception: the only utterances that may hand off with no ready local tool where main
    said False are the generic/mood/genre requests that genuinely name music. Indefinite 'coś'
    without a music noun ('puść coś spokojnego', 'zagraj mi coś na gitarze') is not one."""
    assert media_handoff(text, [], spotify_ready=False) is True, text
    assert media_handoff("puść coś spokojnego", [], spotify_ready=False) is False


@pytest.mark.parametrize(
    "text,expected",
    [
        ("puść film Rocky", False),
        ("puść film rockowy", False),
        ("zagraj w punkty", False),
        ("włącz klimę na punkt", False),
        ("włącz telewizor indie", False),
        ("włącz jazz", True),
        ("puść rocka", True),
        ("puść hip-hop", True),
        ("muzyka do pracy", True),
        ("włącz energetyczną muzykę", True),
        ("puść spokojny jazz", True),
        ("wlacz jazz", True),
        ("pusc rocka", True),
        ("pusc hip-hop", True),
        ("wlacz energetyczna muzyke", True),
        ("pusc spokojny jazz", True),
    ],
)
def test_m4_genre_regex_not_ready_handoff(text: str, expected: bool) -> None:
    """M4: genre regex in not-ready handoff must not match across words ('puść film Rocky'
    vs 'puść rocka'), and must match inflected genres and ASCII variants directly behind play verbs."""
    assert media_handoff(text, [], spotify_ready=False) is expected, text


def test_router_down_deterministic_fast_paths_serve_media_without_the_llm(make_services) -> None:
    """The deterministic media fast paths never touch the LLM, so a router outage does not touch
    them: with a ready local spotify tool 'pauza' is still served locally (classify's fast path
    fires before llm.complete could raise)."""
    from gaja.tools.base import ToolResult
    from gaja.tools.registry import ToolSpec

    seen: list[dict[str, Any]] = []

    async def fake_spotify(**kwargs: Any):
        seen.append(dict(kwargs))
        return ToolResult(data={"ok": True}, speak="Zatrzymane.")

    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="spotify",
            description_pl="Spotify",
            description_en="Spotify",
            parameters={"type": "object", "properties": {"action": {"type": "string"}}},
            handler=fake_spotify,
        )
    )
    llm = FakeLLM(down=True)
    hermes = FakeHermes(result="Hermes puszcza muzykę.")
    services = make_services(llm=llm, hermes=hermes)
    services.tools = reg
    services.router = Router(
        llm, "sys", threshold=0.9, history_turns=2, timeout_s=2.0,
        tools=reg.schemas(), tool_threshold=0.5, spotify_ready=lambda: True,
    )
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "pauza"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "audio_end" and m["id"] == 1)
    assert seen and seen[0]["action"] == "pause"
    assert hermes.tasks == []


@pytest.mark.parametrize("spotify_ready", [False, True])
def test_router_down_media_jobs_still_go_to_hermes_ready_or_not(
    make_services, spotify_ready: bool
) -> None:
    """NEW-2 decision (pipeline router-down path): it must NOT pass the real Spotify readiness.
    Whatever the ready local tool could serve deterministically already ran in classify (the fast
    paths precede the LLM call); what reaches router-down and matches the not-ready media rules
    ('zatrzymaj muzykę głośniej' — not a whole-utterance fast-path form) has only two homes:
    Hermes (works) or the local chat LLM (dead with the router). Handing it to Hermes is correct
    ready or not — passing readiness=True would leave the turn broken."""
    llm = FakeLLM(down=True)
    hermes = FakeHermes(result="Hermes załatwił.")
    services = make_services(llm=llm, hermes=hermes, router=True, tools=ToolRegistry())
    services.router = Router(
        llm, "sys", threshold=0.9, history_turns=2, timeout_s=2.0,
        tools=ToolRegistry().schemas(), tool_threshold=0.5,
        spotify_ready=(lambda: True) if spotify_ready else None,
    )
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "zatrzymaj muzykę głośniej"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "done")
    assert hermes.tasks == ["zatrzymaj muzykę głośniej"]


def test_router_down_timer_questions_stay_local(make_services) -> None:
    """NEW-2 (main parity): during a router outage 'zatrzymaj minutnik' must not become a Hermes
    media handoff; main kept it local. The LLM is down, so the local reply speaks the outage
    error — but Hermes gets no task."""
    llm = FakeLLM(down=True)
    hermes = FakeHermes(result="Hermes się wtrącił.")
    services = make_services(llm=llm, hermes=hermes, router=True, tools=ToolRegistry())
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "zatrzymaj minutnik"}))
        read_until(ws, is_listening)
    assert hermes.tasks == []


def test_bare_volume_and_dalej_follow_the_spotify_tool_marker() -> None:
    """M1/M2: bare 'głośniej'/'ciszej' is the Atom speaker *unless* Gaja's own previous turn was a
    local spotify tool call — recognised by the 'Spotify:' history marker the pipeline records, not
    by free-text mentions. 'dalej' likewise only right after such a turn (or a fresh playing state)."""
    schemas = default_registry.schemas()
    ready = Router(
        FakeLLM(p_tool=0.9), "sys", threshold=0.9, history_turns=2, timeout_s=2.0,
        tools=schemas, tool_threshold=0.5, spotify_ready=lambda: True,
    )
    spotify_turn = [
        {"role": "user", "content": "puść Dawida Podsiadło na Spotify"},
        {"role": "assistant", "content": "Spotify: Włączyłam Dawida Podsiadło."},
    ]
    assert asyncio.run(ready.classify("głośniej", spotify_turn)).reason == "fast-path"
    assert asyncio.run(ready.select_tool("głośniej", spotify_turn)).arguments == {"action": "volume_up"}
    assert asyncio.run(ready.select_tool("ciszej", spotify_turn)).arguments == {"action": "volume_down"}
    assert asyncio.run(ready.select_tool("dalej", spotify_turn)).arguments == {"action": "next"}

    # A free-text Spotify mention without the marker is an old turn: bare volume stays on the speaker.
    free_text = [
        {"role": "user", "content": "puść jazz na Spotify"},
        {"role": "assistant", "content": "Włączyłam Jazz Classics."},
    ]
    call = asyncio.run(ready.select_tool("głośniej", free_text))
    assert call is not None and call.name == "set_speaker_volume"

    # After a non-Spotify turn 'dalej' is never hijacked, whatever an older turn mentioned.
    later = spotify_turn + [
        {"role": "user", "content": "jaka pogoda w Krakowie"},
        {"role": "assistant", "content": "Kraków, obecnie 15 stopni, pogodnie."},
    ]
    assert spotify_fast_path("dalej", later, playing=False) is None

    # Without a ready local Spotify, classify's fast path for bare volume is the Atom speaker; the
    # r4 readiness fix keeps the marker path off too (an unusable tool must not swallow the
    # command): the speaker serves "głośniej" and Hermes gets anything the speaker cannot.
    not_ready = _router(FakeLLM(p_tool=0.9), tools=schemas)
    assert asyncio.run(not_ready.classify("głośniej", spotify_turn)).reason == "fast-path"
    assert asyncio.run(not_ready.select_tool("głośniej", spotify_turn)).name == "set_speaker_volume"
    selected_pauza = asyncio.run(not_ready.select_tool("pauza", spotify_turn))
    assert selected_pauza is None or selected_pauza.name != "spotify"


def test_not_ready_spotify_is_never_plausible_even_when_the_llm_emits_a_call() -> None:
    """Readiness leak, r4d follow-up: with Spotify NOT ready, select_tool must not return a spotify
    call even when the grammar-constrained LLM emits one and the history carries the "Spotify:"
    marker — both the fast-path plausibility and the cue/history plausibility must ignore an
    unusable tool, so the turn follows main@7741180 (local reply / Hermes) instead of a doomed
    spotify call that then has to hand off."""
    schemas = default_registry.schemas()
    emitting = FakeLLM(p_tool=0.92, p_hermes=0.05, tool_call='spotify(action="pause")')
    not_ready = Router(
        emitting, "sys", threshold=0.9, history_turns=2, timeout_s=2.0,
        tools=schemas, tool_threshold=0.5, spotify_ready=None,
    )
    spotify_turn = [
        {"role": "user", "content": "puść Muse"},
        {"role": "assistant", "content": "Spotify: Włączyłam Muse."},
    ]
    assert asyncio.run(not_ready.select_tool("pauza", spotify_turn)) is None
    assert asyncio.run(not_ready.select_tool("dalej", spotify_turn)) is None
    # The ready counterpart still serves the marker turn through the spotify tool.
    ready = Router(
        FakeLLM(p_tool=0.92, p_hermes=0.05, tool_call='spotify(action="pause")'), "sys",
        threshold=0.9, history_turns=2, timeout_s=2.0,
        tools=schemas, tool_threshold=0.5, spotify_ready=lambda: True,
    )
    call = asyncio.run(ready.select_tool("pauza", spotify_turn))
    assert call is not None and call.name == "spotify"


def test_not_ready_spotify_selection_routes_to_hermes_as_tool_unavailable() -> None:
    """r4e: with the spotify tool registered but NOT ready, a TOOL-classified turn that is about
    spotify goes to Hermes with reason "tool-unavailable" — the twin of the H4 unregistered-tool
    rule. select_tool would refuse the call (plausible() rejects an unusable tool, r4d) and the
    turn used to fall to a local chat reply that hallucinates the action ("Zatrzymałam…"), which
    is worse than Hermes. Router-level half of test_spotify.py::
    test_missing_token_file_hands_off_silently_without_extra_sentence."""
    schemas = default_registry.schemas()
    router = Router(
        FakeLLM(p_tool=0.92, p_hermes=0.05, tool_call='spotify(action="pause")'), "sys",
        threshold=0.9, history_turns=2, timeout_s=2.0,
        tools=schemas, tool_threshold=0.5, spotify_ready=None,
    )
    route = asyncio.run(router.classify("zatrzymaj muzykę na Spotify", []))
    assert route.kind == "hermes" and route.reason == "tool-unavailable"

    # Whole-utterance commands without a music noun keep main@7741180's local route: the
    # classify-level cue check deliberately ignores the spotify fast path, so "pauza" classifies
    # as TOOL but select_tool returns no call and the turn ends local (parity table below).
    assert asyncio.run(router.classify("pauza", [])).kind == "tool"
    assert asyncio.run(router.select_tool("pauza", [])) is None

    # With Spotify ready nothing changes: the same utterance is still the local tool's job.
    ready = Router(
        FakeLLM(p_tool=0.92, p_hermes=0.05, tool_call='spotify(action="pause")'), "sys",
        threshold=0.9, history_turns=2, timeout_s=2.0,
        tools=schemas, tool_threshold=0.5, spotify_ready=lambda: True,
    )
    ready_route = asyncio.run(ready.classify("zatrzymaj muzykę na Spotify", []))
    assert ready_route.kind == "tool"


def test_select_tool_raises_tool_unavailable_when_the_model_picks_spotify_not_ready() -> None:
    """r4e: when the cue check in classify could not catch it (another tool's cue matched, no
    spotify cue) and the grammar-constrained model still emits a spotify call with Spotify not
    ready, select_tool raises ToolUnavailable so the pipeline hands the turn to Hermes instead of
    the plausible()-rejected None -> local fallback."""
    from gaja.llm.router import ToolUnavailable

    schemas = default_registry.schemas()
    router = Router(
        FakeLLM(p_tool=0.9, tool_call='spotify(action="pause")'), "sys",
        threshold=0.9, history_turns=2, timeout_s=2.0,
        tools=schemas, tool_threshold=0.5, spotify_ready=None,
    )
    # "włącz klimę w salonie" cues home, not spotify: classify says TOOL, the model emits spotify.
    assert asyncio.run(router.classify("włącz klimę w salonie", [])).kind == "tool"
    with pytest.raises(ToolUnavailable):
        asyncio.run(router.select_tool("włącz klimę w salonie", []))





MAIN_NOT_READY_CLASSIFY_EXPECTED: dict[str, str] = {
    # Media playback (readiness fix: without a ready local spotify tool,
    # "włącz muzykę"/"włącz Spotify" keep main@7741180's hermes:action route)
    "włącz muzykę": "hermes",
    "wyłącz muzykę": "hermes",
    "włącz Spotify": "hermes",
    "wyłącz spotify": "hermes",
    "puść muzykę": "hermes",
    "puść coś na Spotify": "hermes",
    "puść Muse na Spotify": "hermes",
    "zatrzymaj muzykę": "local",
    "wznów muzykę": "hermes",
    "odtwórz playlistę": "hermes",
    "pusc muzyke": "hermes",
    "puść radio": "hermes",
    # Deliberate exceptions (NEW-2 / M4): generic music/genre requests that genuinely name music
    # hand off to Hermes when not ready (media_handoff -> True), whereas main routed them to local.
    "puść rocka": "hermes",
    "włącz jazz": "hermes",
    "puść hip-hop": "hermes",
    "muzyka do pracy": "hermes",
    "włącz energetyczną muzykę": "hermes",
    "puść spokojny jazz": "hermes",
    # Pause / control (readiness fix: "pauza" stays local as on main when not ready,
    # never hitting the spotify fast path)
    "pauza": "local",
    "zatrzymaj": "local",
    "wznów": "local",
    "dalej": "local",
    "poprzedni utwór": "local",
    "następna piosenka": "local",
    # Volume utterances
    "głośniej": "tool",
    "ciszej": "tool",
    "dużo głośniej": "tool",
    "trochę ciszej": "tool",
    "wycisz": "tool",
    "ustaw głośność na 50": "tool",
    "głośność 30": "tool",
    "głośność muzyki na 45": "hermes",
    "daj to głośniej": "local",
    "puść muzykę głośniej": "hermes",
    "wycisz Spotify": "hermes",
    "zagraj coś głośniej": "hermes",
    # Timers / alarms / non-music
    "ustaw budzik na 7:00": "hermes",
    "ustaw minutnik na 5 minut": "local",
    "anuluj minutnik": "local",
    "ile zostało do końca minutnika": "local",
    "zatrzymaj minutnik": "local",
    "wstrzymaj timer": "local",
    "wznów minutnik": "local",
    "pauza minutnika": "local",
    "puść film": "local",
    "puść budzik": "local",
    "puść serial": "local",
    "puść bajkę": "local",
}


@pytest.mark.parametrize("text,expected", sorted(MAIN_NOT_READY_CLASSIFY_EXPECTED.items()))
def test_not_ready_router_classify_matches_main_parity(text: str, expected: str) -> None:
    """Readiness leak (r4): compare classify kind of main's Router with branch (not ready).
    With Spotify not ready, no spotify override or fast path fires, so Router.classify
    matches main@7741180 for media utterances."""
    schemas = default_registry.schemas()
    router = Router(
        FakeLLM(p_tool=0.05, p_hermes=0.05), "sys", threshold=0.9, history_turns=2, timeout_s=2.0,
        tools=schemas, tool_threshold=0.5, spotify_ready=None,
    )
    route = asyncio.run(router.classify(text, []))
    assert route.kind == expected, f"{text}: expected {expected}, got {route.kind} (reason={route.reason})"


def test_pipeline_records_spotify_tool_marker_in_history() -> None:
    """The pipeline prefixes a local spotify tool reply with 'Spotify:' so the next bare command
    rides the spotify fast path (source of the marker checked by recent_spotify_turn)."""
    from gaja.llm.router import SPOTIFY_TURN_PREFIX

    assert SPOTIFY_TURN_PREFIX == "Spotify:"
    assert recent_spotify_turn(
        [
            {"role": "user", "content": "puść Muse"},
            {"role": "assistant", "content": f"{SPOTIFY_TURN_PREFIX} Włączyłam Muse."},
        ]
    )
    assert not recent_spotify_turn(
        [
            {"role": "user", "content": "puść jazz na Spotify"},
            {"role": "assistant", "content": "Włączyłam Jazz Classics."},
        ]
    )
    assert not recent_spotify_turn([])


def test_home_and_spotify_route_locally_while_lights_stay_hermes(monkeypatch: pytest.MonkeyPatch) -> None:
    from gaja.config import Settings

    schemas = default_registry.schemas()
    llm = FakeLLM(p_tool=0.92, p_hermes=0.05)
    ready = Router(
        llm, "sys", threshold=0.9, history_turns=2, timeout_s=2.0, tools=schemas,
        tool_threshold=0.5, spotify_ready=lambda: True,
    )
    not_ready = _router(FakeLLM(p_tool=0.92, p_hermes=0.05), tools=schemas)
    assert asyncio.run(ready.classify("włącz klimę w salonie", [])).kind == "tool"
    assert asyncio.run(ready.classify("ustaw klimatyzację w sypialni na 22 stopnie", [])).kind == "tool"
    assert asyncio.run(ready.classify("wyłącz telewizor w salonie", [])).kind == "tool"
    assert asyncio.run(ready.classify("puść Sanah na Spotify", [])).kind == "tool"
    # Without a logged-in local Spotify the same request is Hermes' (the tool would hand it off).
    assert asyncio.run(not_ready.classify("puść Sanah na Spotify", [])).kind == "hermes"

    # Lights and blinds are not covered by home -> stay Hermes
    assert asyncio.run(ready.classify("włącz światło w salonie", [])).kind == "hermes"
    assert asyncio.run(ready.classify("zamknij rolety w całym domu", [])).kind == "hermes"

    # Runtime registry disables home when HASS_URL / HASS_TOKEN are absent
    monkeypatch.delenv("HASS_URL", raising=False)
    monkeypatch.delenv("HASS_TOKEN", raising=False)
    reg_disabled = build_runtime_registry(Settings())
    assert reg_disabled.get("home") is None
    assert reg_disabled.get("spotify") is not None

    monkeypatch.setenv("HASS_URL", "http://127.0.0.1:8123")
    monkeypatch.setenv("HASS_TOKEN", "test-token")
    reg_enabled = build_runtime_registry(Settings())
    assert reg_enabled.get("home") is not None


def test_media_routing_depends_on_spotify_readiness() -> None:
    """B1: a ready local spotify tool serves simple playback/volume; Hermes stays the fallback.

    Hermes always keeps the indefinite/mood jobs ("coś"), a volume command right after a Hermes
    result, and device transfers; everything else musical goes local once Spotify is logged in.
    """
    schemas = default_registry.schemas()
    ready = Router(
        FakeLLM(p_tool=0.92, p_hermes=0.05), "sys", threshold=0.9, history_turns=2, timeout_s=2.0,
        tools=schemas, tool_threshold=0.5, spotify_ready=lambda: True,
    )
    not_ready = _router(FakeLLM(p_tool=0.92, p_hermes=0.05), tools=schemas)

    # "puść coś na Spotify": indefinite "coś" is a mood/generic job -> Hermes, ready or not.
    assert asyncio.run(not_ready.classify("puść coś na Spotify", [])).reason == "media"
    assert asyncio.run(ready.classify("puść coś na Spotify", [])).reason == "media"

    # "puść Muse": named playback. Ready -> the deterministic play fast path serves it locally;
    # not ready -> main@7741180's media_handoff says False (NEW-2): the cue-driven TOOL route is
    # upgraded to Hermes "tool-unavailable" (r4e) instead of ending in the tool's not_configured
    # handoff (covered in test_spotify.py::test_missing_token_file_hands_off_silently_without_extra_sentence).
    ready_muse = asyncio.run(ready.classify("puść Muse", []))
    assert ready_muse.kind == "tool" and ready_muse.reason == "fast-path"
    call = asyncio.run(ready.select_tool("puść Muse", []))
    assert call is not None and call.name == "spotify"
    assert call.arguments["action"] == "play" and "muse" in str(call.arguments.get("query", "")).lower()
    assert media_handoff("puść Muse", [], spotify_ready=False) is False
    muse_not_ready = asyncio.run(not_ready.classify("puść Muse", []))
    assert muse_not_ready.kind == "hermes" and muse_not_ready.reason == "tool-unavailable"
    # A music object still hands off with no ready tool, before the model masses.
    assert asyncio.run(not_ready.classify("puść muzykę", [])).reason == "media"

    # "puść muzykę głośniej": volume for the music, not the Atom speaker. Ready -> volume_up;
    # not ready -> Hermes (the sound plays where Hermes put it).
    ready_vol = asyncio.run(ready.classify("puść muzykę głośniej", []))
    assert ready_vol.kind == "tool" and ready_vol.reason == "fast-path"
    assert asyncio.run(ready.select_tool("puść muzykę głośniej", [])).arguments == {"action": "volume_up"}
    assert asyncio.run(not_ready.classify("puść muzykę głośniej", [])).reason == "media"

    # B1 fast paths: "zagraj coś głośniej" -> volume_up, "wycisz muzykę"/"wycisz Spotify" -> volume 0.
    assert asyncio.run(ready.select_tool("zagraj coś głośniej", [])).arguments == {"action": "volume_up"}
    assert asyncio.run(ready.select_tool("wycisz muzykę", [])).arguments == {"action": "volume_set", "level": 0}
    assert asyncio.run(ready.select_tool("wycisz Spotify", [])).arguments == {"action": "volume_set", "level": 0}

    # Hermes always keeps device transfers and a volume follow-up right after a Hermes result.
    assert asyncio.run(ready.classify("puść Spotify na głośniku w salonie", [])).kind == "hermes"
    hermes_result = [{"role": "user", "content": "Wynik zadania od Hermesa:\nPuszczam Muse na Desktopie."}]
    assert asyncio.run(ready.classify("daj to głośniej", hermes_result)).reason == "media"
    assert asyncio.run(ready.classify("puść muzykę głośniej", hermes_result)).reason == "media"


def test_hermes_only_imperatives_are_not_hijacked_by_home_or_spotify_cues() -> None:
    """H3: home/spotify cues must not defeat the Hermes action rule or steal unrelated verbs.

    Every utterance below was measured as a hijack on the merged branch: the cue match made
    local_tool_override True, which skipped the Hermes action rule and left the turn on a tool
    (or on a boiler read-only refusal).
    """
    from gaja.llm.client import ToolCall
    from gaja.llm.router import TOOL_CUES, local_tool_override, plausible

    schemas = default_registry.schemas()
    router = _router(FakeLLM(p_tool=0.92, p_hermes=0.05), tools=schemas)
    for text in (
        "napisz wiadomość u Ani",
        "ustaw budzik u Ani na szóstą",
        "włącz muzykę w salonie",
        "wyłącz ogrzewanie w całym domu",
    ):
        route = asyncio.run(router.classify(text, []))
        assert route.kind == "hermes" and route.reason == "action", text

    # Word boundaries: climate talk and baking are not the AC or the boiler.
    home_call = ToolCall(name="home", arguments={"action": "status"})
    assert not TOOL_CUES["home"].search("co sądzisz o klimacie Polski")
    assert not TOOL_CUES["home"].search("jakie są zmiany klimatu")
    assert not plausible(home_call, "co sądzisz o klimacie Polski", [])
    assert not plausible(home_call, "jakie są zmiany klimatu", [])
    assert not plausible(home_call, "czy trzeba piec ciasto", [])
    assert TOOL_CUES["home"].search("jaki stan pieca")
    assert plausible(home_call, "czy piec grzeje", [])

    # Spotify bare verbs no longer steal unrelated objects ("pomiń reklamy", "odtwórz wiadomość",
    # "zagraj w szachy"), while real music objects keep matching.
    spotify_call = ToolCall(name="spotify", arguments={"action": "play"})
    assert not plausible(spotify_call, "pomiń reklamy", [])
    assert not plausible(spotify_call, "odtwórz wiadomość", [])
    assert not plausible(spotify_call, "zagraj w szachy", [])
    assert plausible(spotify_call, "pomiń piosenkę", [])
    assert plausible(spotify_call, "odtwórz album", [])
    assert plausible(spotify_call, "zatrzymaj muzykę", [])
    assert plausible(spotify_call, "puść Muse", [])

    # Local home/spotify control keeps winning where it should (explicit device noun present).
    # The "u <person>" places come from the household aliases (NEW-6), injected here as the
    # server does at start-up.
    set_home_person_words(("ani", "bartka"))
    try:
        for text in (
            "włącz klimę u Ani",
            "ustaw klimatyzację w sypialni na 22 stopni",
            "wyłącz telewizor w salonie",
            "włącz grzanie u Bartka",
        ):
            assert local_tool_override(text, set(default_registry.names())), text
    finally:
        set_home_person_words(None)


def test_home_questions_go_to_hermes_when_home_tool_unregistered(monkeypatch: pytest.MonkeyPatch) -> None:
    """H4: without HASS_URL/HASS_TOKEN the home tool is unregistered; a home question the router
    prompt teaches as TOOL must reach Hermes, not the local chat model (or the weather tool)."""
    from pathlib import Path

    from gaja.config import Settings
    from gaja.llm.prompt import load_text_block
    from gaja.llm.router import template_router_prompt

    monkeypatch.delenv("HASS_URL", raising=False)
    monkeypatch.delenv("HASS_TOKEN", raising=False)
    schemas = build_runtime_registry(Settings()).schemas()
    names = {s["function"]["name"] for s in schemas}
    assert "home" not in names and "spotify" in names

    router = _router(FakeLLM(p_tool=0.92, p_hermes=0.05), tools=schemas)
    set_home_person_words(("ani",))  # the household's own "u Ani" cue, as server.py injects it
    try:
        for text in (
            "jaka jest temperatura w sypialni",
            "jaka temperatura u Ani",
            "czy klima w sypialni chodzi",
            "włącz klimę w salonie",
        ):
            route = asyncio.run(router.classify(text, []))
            assert route.kind == "hermes", text
        # "u Ani" matches no AT_HOME/action rule: only the unavailable-tool rule sends it to Hermes.
        route = asyncio.run(router.classify("jaka jest temperatura u Ani", []))
        assert route.reason == "tool-unavailable"
    finally:
        set_home_person_words(None)

    # The prompt the server builds drops the home teachings (cheap templating, see server.py).
    system = load_text_block(Path("prompts/router3_pl.md"))
    templated = template_router_prompt(system, names)
    assert "home(action=" not in templated
    assert "Wywołanie: home(" not in templated
    assert "klimatyzacja, temperatura w pokojach" not in templated
    assert "home(action=" in system  # the untemplated prompt does teach it
    # Spotify stays taught while it is registered.
    assert "spotify(action=" in templated
    # And with both registered the prompt is passed through untouched.
    assert template_router_prompt(system, names | {"home"}) == system


def test_home_person_cues_match_ascii_folded() -> None:
    """L3: Home person cues from home_person_words must also match the ASCII-folded form,
    e.g. 'temperatura u rodzicow' when the alias is 'u rodziców'."""
    import gaja.llm.router as router_module
    words = home_person_words({"u rodziców": "climate.rodzice", "u Michała": "climate.michal"})
    set_home_person_words(words)
    try:
        assert TOOL_CUES["home"].search("temperatura u rodzicow")
        assert TOOL_CUES["home"].search("temperatura u rodziców")
        assert TOOL_CUES["home"].search("klima u michala")
        assert router_module.HOME_OVERRIDE_DEVICE.search("włącz grzanie u rodzicow")
    finally:
        set_home_person_words(None)


def test_home_person_cues_come_from_the_household_aliases_not_the_repo() -> None:
    import gaja.llm.router as router_module

    """NEW-6: no person names are hard-coded in the router. The "u <person>" home cues match only
    after the household's alias words are injected (as server.py does at start-up), and the words
    themselves are derived from the loaded aliases."""
    # Default (no household file): a person place never cues home, and no write override fires.
    set_home_person_words(None)
    assert not TOOL_CUES["home"].search("jaka jest temperatura u Ani")
    assert not router_module.HOME_OVERRIDE_DEVICE.search("włącz grzanie u Ani")

    # Derivation mirrors the household file's aliases; injection arms exactly those words.
    words = home_person_words({"klima u Ani": "climate.ania", "u rodziców": "climate.sypialnia", "sypialnia": "climate.sypialnia"})
    assert words == ("ani", "rodziców")
    set_home_person_words(words)
    try:
        assert TOOL_CUES["home"].search("jaka jest temperatura u Ani")
        assert TOOL_CUES["home"].search("temperatura u rodziców")
        assert router_module.HOME_OVERRIDE_DEVICE.search("włącz grzanie u Ani")
        # Someone else's place still matches nothing.
        assert not TOOL_CUES["home"].search("temperatura u Bartka")
        assert not router_module.HOME_OVERRIDE_DEVICE.search("włącz grzanie u Bartka")
        # Room-based cues are unaffected.
        assert TOOL_CUES["home"].search("jaka jest temperatura w sypialni")
        assert router_module.HOME_OVERRIDE_DEVICE.search("włącz klimę w salonie")
    finally:
        set_home_person_words(None)
    assert not TOOL_CUES["home"].search("jaka jest temperatura u Ani")


# NEW-1 round 5: possessives/demonstratives and more non-household playback targets.
@pytest.mark.parametrize(
    "text",
    [
        "puść Muse na moich słuchawkach",
        "puść Muse na swoich słuchawkach",
        "puść Muse na tych słuchawkach",
        "puść Muse na moim nowym laptopie",
        "puść Muse na moim starym telefonie",
        "puść Muse na tym telefonie",
        "puść Muse w moich słuchawkach",
        "puść Muse przez słuchawki",
        "puść Muse na moich głośnikach",
        "puść Muse na smartfonie",
        "puść Muse na komórce",
        "puść Muse na iPadzie",
        "puść Muse na maku",
        "puść Muse na lapku",
        "puść Muse na Windowsie",
        "puść Muse na Chromebooku",
    ],
)
def test_new1_possessives_and_new_pc_targets_go_to_hermes(text: str) -> None:
    """NEW-1 r5: 'na <possessive(s)> <target>' (up to 2 words between) is Hermes' ready or not,
    and the local spotify fast path never fires for it."""
    schemas = default_registry.schemas()
    ready = Router(
        FakeLLM(p_hermes=0.05, p_tool=0.92), "sys", threshold=0.9, history_turns=2, timeout_s=2.0,
        tools=schemas, tool_threshold=0.5, spotify_ready=lambda: True,
    )
    route = asyncio.run(ready.classify(text, []))
    assert route.kind == "hermes", (text, route)
    assert spotify_fast_path(text) is None, text

def test_kompozycje_chopina_still_not_a_pc_target() -> None:
    """NEW-1 r5 guard: the widened target list and 2-word gap must not swallow real artists."""
    assert not SPOTIFY_DEVICE_TRANSFER.search("puść kompozycje Chopina")
    call = spotify_fast_path("puść kompozycje Chopina")
    assert call is not None and call.arguments == {"action": "play", "query": "kompozycje Chopina"}


# M4 round 5: genre requests with fillers/adjectives between the play verb and the genre, and the
# adjectival genre forms, must hand off to Hermes ready or not — while film/serial/game/device
# objects stay non-genre.
@pytest.mark.parametrize("text", [
    "puść mi jazz",
    "puść mi trochę jazzu",
    "puść jakiś jazz",
    "puść stary rock",
    "puść polski rap",
    "puść dobry jazz",
    "puść ulubiony jazz",
    "puść coś jazzowego",
    "puść mi coś rockowego",
])
@pytest.mark.parametrize("ready", [False, True])
def test_m4_genre_with_fillers_and_adjectives_hands_off_ready_or_not(text: str, ready: bool) -> None:
    """M4 r5: up to 2 arbitrary words between the play verb and a whole-word genre (plus the
    jazzow/rockow/rapow/metalow/popow adjective forms) is a Hermes media job, with the local
    spotify tool ready or not."""
    assert media_handoff(text, [], spotify_ready=ready) is True, text


@pytest.mark.parametrize("text", [
    "puść film Rocky",
    "puść film rockowy",
    "zagraj w punkty",
    "włącz klimę na punkt",
    "włącz telewizor indie",
    "puść bajkę",
    "puść serial",
])
@pytest.mark.parametrize("ready", [False, True])
def test_m4_film_serial_game_device_objects_stay_non_genre(text: str, ready: bool) -> None:
    """M4 r5 guard: a gap word that is the utterance's real object (film/serial/bajka/game/device
    noun) blocks the genre match — "włącz telewizor indie" must never become a genre handoff."""
    from gaja.llm.router import GENERIC_MUSIC_PLAY_REQUEST as gmpr
    assert not gmpr.search(text), text
    assert media_handoff(text, [], spotify_ready=False) is False, text


def test_home_override_device_matches_ascii_folded_device_nouns() -> None:
    """L3 round 5 follow-up (6d): 'wlacz klime u rodzicow' must reach the same local home-tool
    override as 'włącz klimę u rodziców' — the device-noun stems in the home cue and the
    HOME_OVERRIDE_DEVICE pattern accept their ASCII-folded spellings."""
    import gaja.llm.router as router_module
    from gaja.llm.router import local_tool_override

    router_module.set_home_person_words(("rodziców",))
    try:
        for ascii_form, diacritic_form in [
            ("wlacz klime u rodzicow", "włącz klimę u rodziców"),
            ("wlacz klime u rodzicow", "włącz klimę u rodziców"),
            ("wlacz telewizor w kuchni", "włącz telewizor w kuchni"),
            ("wlacz kociol w piecu", "włącz kocioł w piecu"),
        ]:
            assert router_module.TOOL_CUES["home"].search(ascii_form) is not None, ascii_form
            assert router_module.HOME_OVERRIDE_DEVICE.search(ascii_form) is not None, ascii_form
            assert local_tool_override(ascii_form, {"home"}), ascii_form
            assert local_tool_override(diacritic_form, {"home"}), diacritic_form
    finally:
        router_module.set_home_person_words(None)


# ---- Home-status fast path and household device words (routing_home) --------------------------------

def _arm_example_household_words() -> None:
    """Arm the router exactly as server.py/main() do, from the example household file (the same
    source the routing_home eval dataset uses)."""
    import tomllib
    from pathlib import Path

    from gaja.llm.router import home_device_words, home_person_words

    example = Path(__file__).resolve().parent.parent / "docs" / "home.example.toml"
    aliases = tomllib.loads(example.read_text(encoding="utf-8"))["aliases"]
    set_home_person_words(home_person_words(aliases))
    set_home_device_words(home_device_words(aliases))


def _example_aliases() -> dict[str, str]:
    import tomllib
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / "docs" / "home.example.toml"
    return tomllib.loads(example.read_text(encoding="utf-8"))["aliases"]


def test_home_device_words_derive_from_the_aliases_and_filter_persons_and_rooms() -> None:
    """The home cue/override/fast-path device words are the household's own alias words. Person
    places (folded or not), single-word person aliases sharing a "u <X>" target, and generic room
    words never become device words — "czy rodzice wrócili" is not a boiler question."""
    import gaja.llm.router as router_module
    from gaja.llm.router import home_device_words

    words = home_device_words(_example_aliases())
    assert "palnik" in words and "drukarka" in words and "woda" in words
    assert "konsola" in words and "ps5" in words and "dysk" in words and "wilgotność" in words
    # Person words never leak in (raw, folded or stem forms).
    for person in ("ani", "ania", "rodzice", "rodzicow", "rodziców"):
        assert person not in words, person
    # Rooms stay places, not devices.
    assert "salon" not in words and "sypialnia" not in words
    _arm_example_household_words()
    try:
        assert router_module.TOOL_CUES["home"].search("czy palnik pracuje")
        assert router_module.TOOL_CUES["home"].search("jaka wilgotność w salonie")
        # ASCII-typed utterances cue through the folded form.
        assert router_module.TOOL_CUES["home"].search("jaka wilgotnosc w salonie")
        assert router_module.TOOL_CUES["home"].search("podaj cisnienie w instalacji")
        assert router_module.HOME_OVERRIDE_DEVICE.search("włącz palnik")
        assert router_module.HOME_OVERRIDE_DEVICE.search("wyłącz drukarkę")
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


def test_home_device_words_never_derive_the_pronoun_nas() -> None:
    """A bare/dotted "nas" alias would make "czy u nas działa" a device question (the Polish
    pronoun); the household file must say "dysk sieciowy" and the router must never cue on "nas"."""
    from gaja.llm.router import home_device_words

    assert "nas" not in home_device_words(_example_aliases())
    assert "nas" not in home_device_words({"dysk sieciowy": "binary_sensor.nas", "serwer nas": "binary_sensor.nas"})
    _arm_example_household_words()
    try:
        assert TOOL_CUES["home"].search("czy u nas działa") is None
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


@pytest.mark.parametrize(
    "text",
    [
        "czy palnik pracuje",
        "czy palnik pali",
        "jaki jest stan pieca",
        "czy piec grzeje",
        "czy kocioł działa",
        "sprawdź stan kotła",
        "czy jest ciepła woda",
        "czy woda jest ciepła",
        "jaka jest temperatura ciepłej wody",
        "jakie jest ciśnienie w instalacji",
        "podaj ciśnienie w instalacji",
        "jaka wilgotność w salonie",
        "sprawdź wilgotność w salonie",
        "czy drukarka drukuje",
        "jaki jest stan drukarki",
        "czy konsola działa",
        "czy ps5 jest włączone",
        "czy dysk sieciowy działa",
        "sprawdź dysk sieciowy",
        "czy amplituner jest włączony",
        "sprawdź temperaturę u Ani",
        "jaka temperatura jest u Ani",
        "sprawdz czy drukarka działa",  # ASCII spelling
    ],
)
def test_home_status_fast_path_serves_read_only_status_questions(text: str) -> None:
    """Read-only status questions over a household-configured device ride the deterministic
    home-status fast path — including exactly the live miss 'czy palnik pracuje', which the small
    router model filed under LOCAL."""
    from gaja.llm.router import home_status_route

    _arm_example_household_words()
    try:
        assert home_status_route(text), text
        # The cue matches too, so select_tool may generate the home call and the
        # tool-unavailable rule still catches a missing home tool.
        assert TOOL_CUES["home"].search(text), text
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


@pytest.mark.parametrize(
    "text",
    [
        # General chat / shopping / knowledge about a "piec", "telewizor", "konsola", "woda".
        "jaki piec na pellet kupić",
        "ile kosztuje nowy telewizor",
        "czy telewizor 4k warto kupić",
        "czy warto wymienić piec na pompę ciepła",
        "jak działa telewizor plazmowy",
        "czy woda w morzu jest słona",
        "jaki jest najlepszy telewizor do konsoli",
        "gdzie kupić tanią klimatyzację",
        "opowiedz o piecach kaflowych",
        "czy wodomierz się kręci",
        # Music stays Hermes'.
        "puść muzykę w salonie",
        "włącz muzykę w sypialni",
        "co teraz gra na spotify",
        # Weather keeps the weather tool.
        "jaka jest temperatura w Zakopanem",
        "jaka będzie jutro temperatura w górach",
        "czy będzie dziś deszcz w Krakowie",
        # Timers and alarms stay on their own routes.
        "ustaw minutnik na 10 minut",
        "ustaw budzik na szóstą",
        "ile zostało do minutnika",
        "zatrzymaj minutnik",
        "przypomnij mi o 30 stopniach u Ani",
        # Messages are Hermes' even when a device noun appears.
        "napisz wiadomość o palniku",
        # Unsupported home devices (lights, doors) and bare "u <person>" checks.
        "zapal światło w kuchni",
        "zamknij drzwi do garażu",
        "sprawdź u Ani czy masz wiadomości",
        "czy u nas działa",
        # F2: chat/web requests that merely mention a device noun are knowledge questions —
        # price, recipe, shop, internet, reviews, how-to — never a spoken sensor state.
        "sprawdź w internecie cenę pieca",
        "podaj przepis na chleb z pieca",
        "podaj cenę telewizora w sklepie",
        "sprawdź opinie o piecach na pellet",
        "sprawdź recenzje telewizorów",
        "podaj cenę bojlera",
        "sprawdź cenę pieca w internecie",
        "podaj instrukcję obsługi pieca",
        "jak włączyć piec",
        "jak podłączyć termostat",
        "ile kosztuje nowy bojler",
        # Writes keep the model + override path, never the status fast path.
        "włącz palnik",
        "ustaw klimę w salonie na 22 stopnie",
    ],
)
def test_home_status_fast_path_never_fires_on_chat_music_weather_writes_or_messages(text: str) -> None:
    from gaja.llm.router import home_status_route

    _arm_example_household_words()
    try:
        assert not home_status_route(text), text
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


@pytest.mark.parametrize(
    "text",
    [
        # F2 control: a check/report imperative about a device's own state keeps the fast path.
        "sprawdź dysk sieciowy",
        "sprawdź stan pieca w piwnicy",
        "podaj ciśnienie w instalacji",
        "sprawdź temperaturę u Ani",
        "podaj stan drukarki",
    ],
)
def test_home_status_fast_path_keeps_check_verbs_about_a_devices_state(text: str) -> None:
    """Control for F2: 'sprawdź/podaj' over a household device word stays a deterministic home
    status call — only chat/web knowledge requests (price, recipe, shop, internet, reviews,
    how-to) are turned away."""
    from gaja.llm.router import home_status_route

    _arm_example_household_words()
    try:
        assert home_status_route(text), text
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


def test_home_status_fast_path_needs_household_words() -> None:
    """Without the household's derived words (no data/home.toml) the fast path is off: the model
    and the tool-unavailable rule keep the pre-existing behaviour."""
    from gaja.llm.router import home_status_route

    set_home_person_words(None)
    set_home_device_words(None)
    assert not home_status_route("czy palnik pracuje")
    assert not home_status_route("jaki jest stan drukarki")


def test_home_status_fast_path_routes_even_when_the_model_says_local() -> None:
    """The live miss: 'czy palnik pracuje' classified LOCAL, the local chat promised a check and
    the safety net handed the turn to Hermes. With the household words armed the deterministic
    route serves the home tool before the model masses are even asked."""
    schemas = default_registry.schemas()
    _arm_example_household_words()
    try:
        from gaja.config import Settings

        router = Router(
            FakeLLM(p_tool=0.05, p_hermes=0.05, tool_call='home(action="status", device="palnik")'),
            "sys", threshold=0.9, history_turns=2, timeout_s=2.0,
            tools=schemas, tool_threshold=0.5,
        )
        route = asyncio.run(router.classify("czy palnik pracuje", []))
        assert route.kind == "tool" and route.reason == "home-status"
        call = asyncio.run(router.select_tool("czy palnik pracuje", []))
        assert call is not None and call.name == "home"
        assert call.arguments.get("action") == "status"

        # And the exact same utterance with the home tool unregistered goes to Hermes.
        monkey_free = build_runtime_registry(Settings())  # no HASS_* env: home missing
        names = {s["function"]["name"] for s in monkey_free.schemas()}
        assert "home" not in names
        unregistered = Router(
            FakeLLM(p_tool=0.05, p_hermes=0.05), "sys", threshold=0.9, history_turns=2, timeout_s=2.0,
            tools=monkey_free.schemas(), tool_threshold=0.5,
        )
        off_route = asyncio.run(unregistered.classify("czy palnik pracuje", []))
        assert off_route.kind == "hermes" and off_route.reason == "tool-unavailable"
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


@pytest.mark.parametrize(
    "text",
    [
        "jaka jest temperatura w sypialni",
        "ile stopni jest w salonie",
        "jak jest temperatura w salonie",
        "jaka temperatura jest w domu",
        "jaka wilgotność w salonie",
    ],
)
def test_home_status_room_quantity_questions_route_tool(text: str) -> None:
    """Room places join the household gate for sensor-quantity status questions (the small model
    handed these to Hermes); the household file must be present so aliases can resolve the room."""
    from gaja.llm.router import home_status_route

    _arm_example_household_words()
    try:
        assert home_status_route(text), text
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


@pytest.mark.parametrize(
    "text",
    [
        "jaka temperatura powinna być w pokoju dziecka",
        "jaka wilgotność jest optymalna w sypialni",
        "jaka temperatura w pokoju jest zalecana",
        "jaka jest najlepsza temperatura w sypialni",
        # F4: the ordinary advice synonyms — chat, not a sensor read.
        "jaka temperatura jest odpowiednia w pokoju dziecka",
        "jaka wilgotność jest zdrowa w sypialni",
        "jaka temperatura jest idealna w salonie",
        "jaka wilgotność jest komfortowa w sypialni",
        "jaka temperatura jest bezpieczna w pokoju dziecka",
        "jaka temperatura jest właściwa w salonie",
        "jaka temperatura jest dobra dla niemowlaka w sypialni",
        # F4: natural water bodies and geography stay out (no household sensor there).
        "jaka jest temperatura wody w morzu",
        "jaka jest temperatura w jeziorze",
        "jaka jest temperatura w rzece",
        "jaka jest temperatura w oceanie",
        "jaka jest temperatura wody w Bałtyku",
        "czy woda w morzu jest ciepła",
        "ile stopni ma woda w basenie",
    ],
)
def test_home_status_advice_questions_never_route_tool(text: str) -> None:
    """Advice/shopping questions about room climate are chat, not a sensor read — even though the
    room+quantity shape and a household device word both match. F4: the closed word list is
    widened to the ordinary synonyms (odpowiednia, zdrowa, idealna, komfortowa, bezpieczna,
    właściwa, "dobra dla") and natural water bodies/geography (morze, jezioro, rzeka, ocean,
    Bałtyk) stay out — the example household has no device word for any of them."""
    from gaja.llm.router import home_status_route

    _arm_example_household_words()
    try:
        assert not home_status_route(text), text
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


def test_home_status_outdoors_kept_when_the_household_has_a_device_word_for_it() -> None:
    """F4 carve-out: a household that aliases the outdoors thing ("woda w basenie" ->
    sensor.basen_temp, invented words) keeps its pool questions; the sea/lake/ocean words stay
    out for that very household, and "temperatura na zewnątrz" stays a home question when the
    household has an outdoor sensor."""
    from gaja.config import Settings

    from gaja.llm.router import home_device_words, home_person_words, home_status_route

    aliases = {
        "woda w basenie": "sensor.basen_temp",
        "temperatura na zewnątrz": "sensor.outdoor",
    }
    set_home_person_words(home_person_words(aliases))
    set_home_device_words(home_device_words(aliases))
    try:
        assert home_status_route("ile stopni ma woda w basenie")
        assert home_status_route("jaka jest temperatura wody w basenie")
        assert home_status_route("jaka jest temperatura na zewnątrz")
        # Nature words stay out even for this household.
        assert not home_status_route("jaka jest temperatura wody w morzu")
        assert not home_status_route("jaka jest temperatura w jeziorze")
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


def test_home_status_room_quantity_needs_the_household_file() -> None:
    """Without the household's device words (no data/home.toml) the room+quantity route stays off:
    the pre-existing model route (Hermes masses / tool-unavailable) is kept."""
    from gaja.llm.router import home_status_route

    set_home_person_words(None)
    set_home_device_words(None)
    assert not home_status_route("jaka jest temperatura w sypialni")


@pytest.mark.parametrize(
    ("text", "action", "extra"),
    [
        ("podgłośnij telewizor w salonie", "volume_up", {}),
        ("ustaw głośność telewizora w salonie na 30", "volume_set", {"level": 30}),
        ("ustaw głośność amplitunera na 40", "volume_set", {"level": 40}),
        ("wycisz amplituner", "mute", {}),
        ("ścisz tv w salonie", "volume_down", {}),
        ("jaka głośność amplitunera", "status", {}),
    ],
)
def test_home_volume_route_serves_household_device_volume_commands(
    text: str, action: str, extra: dict
) -> None:
    """Media-player volume commands over a household device word are deterministic home calls —
    the small model filed them under LOCAL or picked the Atom speaker (the wrong device)."""
    from gaja.llm.router import home_volume_call, home_volume_route

    _arm_example_household_words()
    try:
        assert home_volume_route(text), text
        call = home_volume_call(text)
        assert call is not None and call.name == "home", text
        assert call.arguments["action"] == action, (text, call.arguments)
        for key, value in extra.items():
            assert call.arguments[key] == value, (text, call.arguments)
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


@pytest.mark.parametrize(
    "text",
    [
        "podgłośnij spotify",
        "ścisz radio",
        "wycisz muzykę",
        "daj to głośniej",
        "ustaw głośność na 50",  # the bare Atom speaker fast path, no household device
        "wycisz",
        "głośność 30",
    ],
)
def test_home_volume_route_never_takes_music_or_bare_speaker_volume(text: str) -> None:
    """Music stays on the spotify/Hermes media rules; bare whole-utterance volume keeps controlling
    the Atom speaker (volume_fast_path)."""
    from gaja.llm.router import home_volume_route

    _arm_example_household_words()
    try:
        assert not home_volume_route(text), text
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


def test_home_volume_select_tool_returns_the_home_call_not_the_speaker() -> None:
    """select_tool must not let the grammar-constrained model pick set_speaker_volume for a
    household device-volume command: 'ustaw głośność amplitunera na 40' used to land on the Atom
    speaker (the wrong device)."""
    schemas = default_registry.schemas()
    _arm_example_household_words()
    try:
        router = Router(
            FakeLLM(p_tool=0.9, tool_call='set_speaker_volume(action="set", level=40)'),
            "sys", threshold=0.9, history_turns=2, timeout_s=2.0,
            tools=schemas, tool_threshold=0.5,
        )
        call = asyncio.run(router.select_tool("ustaw głośność amplitunera na 40", []))
        assert call is not None and call.name == "home"
        assert call.arguments["action"] == "volume_set" and call.arguments["level"] == 40
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


@pytest.mark.parametrize(
    ("text", "action"),
    [
        ("włącz amplituner", "turn_on"),
        ("wyłącz tv w salonie", "turn_off"),
        ("włącz klimę u Ani", "turn_on"),
        ("włącz palnik", "turn_on"),  # the local tool refuses it read-only; Hermes must not try
    ],
)
def test_home_imperative_route_serves_household_on_off_writes(text: str, action: str) -> None:
    """On/off imperatives over a household device route to the local home tool deterministically —
    the small model files some under LOCAL ('włącz amplituner'), and a HERMES route would let the
    general agent touch the entities the local tool guards ('włącz palnik' must reach the local
    read-only refusal)."""
    from gaja.llm.router import home_imperative_call, home_imperative_route

    _arm_example_household_words()
    try:
        assert home_imperative_route(text), text
        call = home_imperative_call(text)
        assert call is not None and call.name == "home", text
        assert call.arguments["action"] == action, (text, call.arguments)
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


@pytest.mark.parametrize(
    "text",
    [
        "włącz muzykę",  # music stays Hermes'/spotify's
        "puść muzykę w salonie",
        "włącz Spotify",
        "ustaw budzik na szóstą u Ani",  # message/alarm objects stay Hermes'
        "napisz wiadomość do Ani",
        "zapal światło w kuchni",  # lights/doors stay Hermes'
        "zamknij drzwi do garażu",
        "ustaw minutnik na 10 minut",  # timers stay local/manage_timer
        "włącz klimatyzację gdzieś indziej",  # no household device word
        "wyłącz się",
    ],
)
def test_home_imperative_route_never_takes_music_alarms_lights_or_timers(text: str) -> None:
    from gaja.llm.router import home_imperative_route

    _arm_example_household_words()
    try:
        assert not home_imperative_route(text), text
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


@pytest.mark.parametrize(
    "text",
    [
        # An unknown device noun must never ride the bare "u <person>" place into the tool
        # (F1): the household's AC at that place is NOT a dishwasher/coffee machine/oven.
        "włącz zmywarkę u Ani",
        "wyłącz ekspres do kawy u Ani",
        "włącz piekarnik u rodziców",
        "włącz pralkę u Ani",
    ],
)
def test_home_imperative_route_never_lets_a_person_place_stand_in_for_an_unknown_device(text: str) -> None:
    """F1: 'włącz zmywarkę u Ani' used to take the imperative fast path unconditionally, and the
    tool then resolved the bare 'u Ani' alias onto the AC of that room — a wrong write."""
    from gaja.llm.router import home_imperative_call, home_imperative_route

    _arm_example_household_words()
    try:
        assert not home_imperative_route(text), text
        assert home_imperative_call(text) is None, text
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


@pytest.mark.parametrize(
    "text",
    [
        "włącz klimę u Ani",
        "wyłącz klimę u Ani",
        "włącz chłodzenie u rodziców",
        # A person place next to a full multi-word device name: the device words carry the
        # command even though a room word ("piwnicy") is not a person-path filler (F5 shape).
        "wyłącz kocioł gazowy w piwnicy i klimę u Ani",
    ],
)
def test_home_imperative_route_keeps_person_place_commands_over_known_devices(text: str) -> None:
    """Control for F1: the person place fast path keeps firing when nothing but known device
    words, fillers and the place remain ('klima' is a household device word, 'chłodzenie' an
    HVAC mode value word)."""
    from gaja.llm.router import home_imperative_call, home_imperative_route

    _arm_example_household_words()
    try:
        assert home_imperative_route(text), text
        assert home_imperative_call(text) is not None, text
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


@pytest.mark.parametrize(
    "text",
    [
        "czy u Ani działa pralka",
        "czy pralka u Ani działa",
        "jaka jest temperatura pralki u Ani",
    ],
)
def test_home_status_route_never_lets_a_person_place_stand_in_for_an_unknown_device(text: str) -> None:
    """F3: a status question about an unknown device at a person place must not become the AC's
    status; the person place alone may only carry status words, not another device noun."""
    from gaja.llm.router import home_status_route

    _arm_example_household_words()
    try:
        assert not home_status_route(text), text
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


@pytest.mark.parametrize(
    "text",
    [
        "jaka jest temperatura u Ani",
        "sprawdź temperaturę u Ani",
        "czy klima u Ani działa",
        "czy u Ani działa",
    ],
)
def test_home_status_route_keeps_person_place_status_questions(text: str) -> None:
    """Control for F3: bare person-place status questions (quantity words, known device words
    or just the status verb) keep the deterministic status fast path."""
    from gaja.llm.router import home_status_route

    _arm_example_household_words()
    try:
        assert home_status_route(text), text
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


def test_eval_mirror_decides_home_fast_paths_like_classify() -> None:
    """F7: bench/router3_eval.py's decide() must mirror Router.classify's guards exactly:
    classify guards only the STATUS home path with MYTHOLOGY (router.py), volume/imperative
    fire regardless — a mythology-bearing home imperative is a tool route in production, and
    the eval mirror must score it identically instead of handing it to Hermes."""
    from bench.router3_eval import decide

    _arm_example_household_words()
    try:
        imperative = {
            "masses": [0.05, 0.05, 0.9],
            "text": "włącz klimę w salonie i opowiedz o Zeusie",
            "item": {"history": []},
        }
        assert decide(imperative, "v3", 0.5, 0.6) == "tool"
        # The status path keeps its mythology guard in classify — and in the mirror: no home
        # fast path fires and the mythology guard keeps the hermes masses out (local reply).
        status = {
            "masses": [0.05, 0.05, 0.9],
            "text": "czy klima w salonie działa, kim jest bóg grecki",
            "item": {"history": []},
        }
        assert decide(status, "v3", 0.5, 0.6) == "local"
    finally:
        set_home_person_words(None)
        set_home_device_words(None)


def test_select_tool_returns_deterministic_home_calls_for_status_and_imperatives() -> None:
    """Status questions and on/off imperatives over household devices never ride the grammar
    generation: the model sometimes picks a cue-overlapping tool ('dysk' -> server_status) or
    nothing plausible; the home tool owns these turns and hands off itself on no_match."""
    schemas = default_registry.schemas()
    _arm_example_household_words()
    try:
        router = Router(
            FakeLLM(p_tool=0.9, tool_call='server_status()'),
            "sys", threshold=0.9, history_turns=2, timeout_s=2.0,
            tools=schemas, tool_threshold=0.5,
        )
        call = asyncio.run(router.select_tool("jaki jest stan dysku sieciowego", []))
        assert call is not None and call.name == "home" and call.arguments["action"] == "status"
        call = asyncio.run(router.select_tool("włącz amplituner", []))
        assert call is not None and call.name == "home" and call.arguments["action"] == "turn_on"
    finally:
        set_home_person_words(None)
        set_home_device_words(None)
