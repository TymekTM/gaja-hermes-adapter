"""The local chat model must not answer by repeating the previous (Hermes) answer (device log 2026-10-26).

After a Hermes result the history holds "Powiedz to krótko…" (user) plus the result text (assistant),
so the small model's last job was to relay Hermes. On a new turn ("Daj to głośniej.", a stray "PU.")
it sometimes replays that answer instead of acting. The guard drops a streaming reply that is nearly
the previous assistant message and hands the turn to Hermes — unless the user deliberately asked the
same thing again.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from gaja.llm.client import TextDelta
from gaja.pipeline import (
    RESULT_PROMPT,
    previous_exchange,
    repeats_previous_reply,
    same_request,
)
from gaja.server import create_app
from tests.conftest import FakeHermes, FakeLLM
from tests.test_ws_protocol import is_listening, is_type, read_until

HERMES_RESULT = "Puszczam Muse na Desktopie, gra Uprising. Jakby nie podszedło, mów coś innego."
HERMES_PARROT = "Puszczę Muse na Desktopie, gra Uprising. Jakby nie podszedło, mów coś innego."
HERMES_VOLUME = "Podkręciłem z 57 na 82 procent. Głośniej już tylko na maksa, jak chcesz, to mów."
HERMES_VOLUME_PARROT = "Hermes, podkręciłem z 57 na 82 procent. Głośniej już tylko na maksa…"


# ---- unit level -------------------------------------------------------------------------------------


def test_repeats_previous_reply_catches_the_live_parrots() -> None:
    assert repeats_previous_reply(HERMES_PARROT, HERMES_RESULT)
    assert repeats_previous_reply(HERMES_VOLUME_PARROT, HERMES_VOLUME)


def test_repeats_previous_reply_ignores_short_and_different_replies() -> None:
    assert not repeats_previous_reply("Dobrze.", HERMES_RESULT)  # too short to judge
    assert not repeats_previous_reply("", HERMES_RESULT)
    assert not repeats_previous_reply(HERMES_PARROT, "")  # nothing to compare with
    assert not repeats_previous_reply("Dwa plus dwa to cztery.", HERMES_RESULT)


def test_same_request_stands_down_for_a_deliberate_repeat() -> None:
    assert same_request("ile to dwa plus dwa", "Ile to dwa plus dwa?")
    assert same_request("jaka jest pogoda w krakowie", "jaka jest pogoda dzisiaj w Krakowie")
    assert not same_request("Daj to głośniej.", "Puść na moim komputerze coś na Spotify, według swojego uznania.")


def test_previous_exchange_skips_the_current_user_turn() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "puść coś"},
        {"role": "assistant", "content": HERMES_RESULT},
        {"role": "user", "content": "daj to głośniej"},
    ]
    assert previous_exchange(messages) == (HERMES_RESULT, "puść coś")
    assert previous_exchange([{"role": "system", "content": "sys"}, {"role": "user", "content": "cześć"}]) == ("", "")


# ---- pipeline level ---------------------------------------------------------------------------------


def _history_after_hermes_result(session: Any) -> None:
    """Exactly what a delivered Hermes result turn leaves behind (Session._result_turn)."""
    session._history.add("user", RESULT_PROMPT.format(result=HERMES_RESULT))
    session._history.add("assistant", HERMES_RESULT)


def test_local_reply_that_parrots_the_hermes_result_hands_off_instead(make_services) -> None:
    llm = FakeLLM([[TextDelta(HERMES_PARROT)]])
    hermes = FakeHermes(result="Gotowe, głośniej.")
    services = make_services(llm=llm, hermes=hermes, router=True)
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        _history_after_hermes_result(next(iter(services.sessions)))
        ws.send_text(json.dumps({"type": "text", "text": "Daj to głośniej."}))
        done = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "done")
        spoken = read_until(ws, is_type("metrics"))
    assert hermes.tasks == ["Daj to głośniej."]
    utterances = [
        m["text"]
        for m in done + spoken
        if isinstance(m, dict) and m.get("type") == "utterance"
    ]
    assert not any("Muse" in u for u in utterances)  # the parroted reply was never spoken
    assert utterances[-1] == "Gotowe, głośniej."


def test_a_repeated_question_still_gets_its_repeated_answer(make_services) -> None:
    llm = FakeLLM([[TextDelta("Dwa plus dwa to cztery.")]])
    hermes = FakeHermes(result="Nie powinno dojść do Hermesa.")
    services = make_services(llm=llm, hermes=hermes, router=True)
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        session = next(iter(services.sessions))
        session._history.add("user", "ile to dwa plus dwa")
        session._history.add("assistant", "Dwa plus dwa to cztery.")
        ws.send_text(json.dumps({"type": "text", "text": "ile to dwa plus dwa"}))
        messages = read_until(ws, is_type("metrics"))
    utterances = [m["text"] for m in messages if isinstance(m, dict) and m.get("type") == "utterance"]
    assert utterances == ["Dwa plus dwa to cztery."]
    assert hermes.tasks == []
