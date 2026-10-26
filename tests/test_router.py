from __future__ import annotations

import asyncio
import json
import math

import pytest
from fastapi.testclient import TestClient

from gaja.llm.client import LLMUnavailable, TextDelta
from gaja.llm.router import (
    Router,
    addresses_hermes,
    build_router_prompt,
    class_masses,
    p_hermes,
    parse_call,
    promises_handoff,
    tool_grammar,
)
from gaja.pipeline import ACKS, Session
from gaja.server import create_app
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
