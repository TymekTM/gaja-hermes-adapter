"""Media (music/Spotify/PC) volume must go to Hermes, never be claimed locally (device log 2026-10-26).

The local volume fast path drives only the Atom speaker. "Puść muzykę głośniej." routed local and the
chat model answered "Puszczę muzykę głośniej." — a false action claim. The rule: bare whole-utterance
commands stay on the fast path; a volume word plus music/Spotify/playback, or an object pronoun right
after a Hermes result, is media and goes to Hermes.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from gaja.llm.router import Router, media_handoff, recent_hermes_result
from gaja.server import create_app
from tests.conftest import FakeHermes, FakeLLM
from tests.test_ws_protocol import is_listening, is_type, read_until

LIVE_MEDIA_HISTORY = [
    {"role": "user", "content": "Puść na moim komputerze coś na Spotify, według swojego uznania."},
    {"role": "assistant", "content": "Jasne, przekazuję to Hermesowi."},
    {
        "role": "user",
        "content": "Wynik zadania od Hermesa:\nPuszczam Muse na Desktopie, gra Uprising. "
        "Jakby nie podszedło, mów coś innego.\n\nPowiedz to krótko, w 1 do 3 zdaniach.",
    },
    {"role": "assistant", "content": "Puszczam Muse na Desktopie, gra Uprising. Jakby nie podszedło, mów coś innego."},
]

VOLUME_TOOL = [
    {
        "function": {
            "name": "set_speaker_volume",
            "parameters": {
                "type": "object",
                "properties": {"action": {"type": "string"}, "delta": {"type": "integer"}, "level": {"type": "integer"}},
            },
        }
    }
]


def _router(llm: FakeLLM) -> Router:
    return Router(llm, "sys", threshold=0.9, history_turns=2, timeout_s=2.0, tools=VOLUME_TOOL, tool_threshold=0.5)


# ---- the rule ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "Puść muzykę głośniej.",
        "Puść na moim komputerze coś na Spotify, według swojego uznania.",
        "puść coś na spotify",
        "podgłośnij spotify",
        "wycisz muzykę",
        "zwiększ głośność muzyki",
        "ścisz radio",
        "zagraj coś głośniej",
    ],
)
def test_media_requests_go_to_hermes_before_the_model(utterance: str) -> None:
    # Even a strong TOOL mass must not win: the local speaker is the wrong device for these.
    llm = FakeLLM(p_tool=0.95, p_hermes=0.0)
    route = asyncio.run(_router(llm).classify(utterance, []))
    assert route.kind == "hermes" and route.reason == "media"
    assert llm.prompts == []  # decided by the rule, not by the small router


def test_object_volume_right_after_a_hermes_result_goes_to_hermes() -> None:
    llm = FakeLLM(p_tool=0.95, p_hermes=0.0)
    router = _router(llm)
    route = asyncio.run(router.classify("Daj to głośniej.", LIVE_MEDIA_HISTORY))
    assert route.kind == "hermes" and route.reason == "media"
    assert llm.prompts == []


@pytest.mark.parametrize("utterance", ["Daj to głośniej.", "podkręć to", "ścisz to"])
def test_media_handoff_needs_a_hermes_result_for_the_object_pronoun(utterance: str) -> None:
    assert media_handoff(utterance, LIVE_MEDIA_HISTORY)
    assert not media_handoff(utterance, [])
    assert not recent_hermes_result([])


def test_recent_hermes_result_needs_to_be_the_last_user_turn() -> None:
    assert recent_hermes_result(LIVE_MEDIA_HISTORY)
    older = LIVE_MEDIA_HISTORY + [{"role": "user", "content": "a co tam słychać?"}]
    assert not recent_hermes_result(older)


# ---- the guard rails --------------------------------------------------------------------------------


@pytest.mark.parametrize("utterance", ["głośniej", "dużo ciszej", "głośność 50", "na maksa", "Gaja, ciszej"])
def test_bare_volume_commands_still_take_the_speaker_fast_path(utterance: str) -> None:
    llm = FakeLLM(p_tool=0.0)
    route = asyncio.run(_router(llm).classify(utterance, LIVE_MEDIA_HISTORY))
    assert route.kind == "tool" and route.reason == "fast-path"
    assert llm.prompts == []


@pytest.mark.parametrize(
    ("utterance", "history"),
    [
        ("lubisz muzykę", []),
        ("zagrajmy w zgadywankę zwierząt", []),
        ("nie lubię tej muzyki", LIVE_MEDIA_HISTORY),  # opinion, no volume word
        ("głośno było na weselu", []),  # not a media request at all
        ("Która jest godzina?", LIVE_MEDIA_HISTORY),  # "to" absent, plain tool question
    ],
)
def test_media_words_alone_do_not_hand_off(utterance: str, history: list[dict[str, str]]) -> None:
    llm = FakeLLM(p_hermes=0.0)
    route = asyncio.run(_router(llm).classify(utterance, history))
    assert route.kind == "local"
    assert media_handoff(utterance, history) is False


# ---- pipeline: the rule also holds when the router is down ------------------------------------------


def test_media_volume_hands_off_even_when_the_router_is_down(make_services) -> None:
    llm = FakeLLM(down=True)
    hermes = FakeHermes(result="Podkręciłem muzykę.")
    app = create_app(services=make_services(llm=llm, hermes=hermes, router=True))
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "puść muzykę głośniej"}))
        read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "done")
    assert hermes.tasks == ["puść muzykę głośniej"]
