""""Mówi, a nie robi": the chat model announces a tool call (or a done action) as plain text.

Seen in bench/llm_noisy_eval.py ("Jasne, przekazuję to Hermesowi… Wywołaj narzędzie
handoff_to_hermes" or "Zrobiłem to teraz", with no tool call behind them). The pipeline must
turn an announced handoff into the real handoff_to_hermes call, and drop a first-person
action claim, before the false sentence reaches TTS (Session._enqueue, local replies only).
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from gaja.llm.client import TextDelta
from gaja.server import create_app
from gaja.text.claims import claims_action, mentions_tool_call
from tests.conftest import FakeHermes, FakeLLM
from tests.test_ws_protocol import is_listening, is_type, read_until


# ---- unit level -------------------------------------------------------------------------------------


def test_tool_call_text_catches_announced_calls() -> None:
    for text in (
        "Jasne, przekazuję to Hermesowi… Wywołaj narzędzie handoff_to_hermes",
        'handoff_to_hermes(task="puść muzykę", reason="media")',
        '<|tool_call_start|>[handoff_to_hermes(task="x")]<|tool_call_end|>',
        'spotify(action="play", query="Muse")',
        "Wywołuję narzędzie get_weather.",
    ):
        assert mentions_tool_call(text), f"missed a tool-call announcement: {text}"


def test_tool_call_text_ignores_normal_speech() -> None:
    for text in (
        "Puść Muse na Spotify.",
        "Pamiętaj, żeby wziąć parasol.",
        "Zrobiłem to teraz.",
        "Dwa plus dwa to cztery.",
        "Wikipedia zna odpowiedź na to pytanie.",
    ):
        assert not mentions_tool_call(text), f"false positive on normal speech: {text}"


def test_action_claims_catch_first_person_past_claims() -> None:
    for text in (
        "Zrobiłem to teraz.",
        "Zrobiłam to.",
        "Usunąłem to z pamięci.",
        "Usunęłam notatkę.",
        "Zapisałam nową notatkę.",
        "Anulowałem minutnik.",
        "Już wyłączyłem światło w kuchni.",
        "I've sent the message.",
        "It's done!",
    ):
        assert claims_action(text), f"missed an action claim: {text}"


def test_action_claims_leave_normal_speech_alone() -> None:
    for text in (
        "Nie zrobiłam nic złego.",
        "Nic nie usunąłem.",
        "Przypomniałem sobie o spotkaniu.",
        "Dwa plus dwa to cztery.",
        "Jaka jest pogoda w Krakowie?",
        "Włączam klimatyzację w sypialni.",
        "Zaraz sprawdzam, chwilkę.",
        "Zapamiętasz to na przyszłość?",
    ):
        assert not claims_action(text), f"false positive on normal speech: {text}"


# ---- pipeline level ---------------------------------------------------------------------------------


def _utterances(messages: list[Any]) -> list[str]:
    return [
        m["text"]
        for m in messages
        if isinstance(m, dict) and m.get("type") == "utterance"
    ]


def test_announced_handoff_text_becomes_a_real_handoff(make_services) -> None:
    llm = FakeLLM([[TextDelta("Dobrze. Wywołaj narzędzie handoff_to_hermes.")]])
    hermes = FakeHermes(result="Nie powinno dojść do Hermesa.")
    services = make_services(llm=llm, hermes=hermes, router=True)
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "włącz światło w salonie"}))
        done = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "done")
        read_until(ws, is_type("metrics"))
    assert hermes.tasks == ["włącz światło w salonie"]
    spoken = _utterances(done)
    assert not any("handoff" in u.lower() for u in spoken)  # the announcement was never spoken


def test_claimed_done_action_is_never_spoken_and_hands_off(make_services) -> None:
    llm = FakeLLM([[TextDelta("Zrobiłem to teraz.")]])
    hermes = FakeHermes(result="Zapomniałam.")
    services = make_services(llm=llm, hermes=hermes, router=True)
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "zapomnij o tym"}))
        done = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "done")
        read_until(ws, is_type("metrics"))
    assert hermes.tasks == ["zapomnij o tym"]  # the claimed action is done for real
    assert not any("Zrobiłem" in u for u in _utterances(done))  # and the false claim never spoken


def test_claim_after_a_spoken_sentence_still_hands_off(make_services) -> None:
    llm = FakeLLM([[TextDelta("Nie mam takiej funkcji. Wysłałem maila.")]])
    hermes = FakeHermes(result="Mail wysłany.")
    services = make_services(llm=llm, hermes=hermes, router=True)
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "wyślij maila do szefa"}))
        done = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "done")
        read_until(ws, is_type("metrics"))
    assert hermes.tasks == ["wyślij maila do szefa"]
    assert not any("Wysłałem" in u for u in _utterances(done))
