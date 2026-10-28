"""Voice volume control: deterministic fast path (router), local tool, spoken confirmation."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from gaja.llm.parse import ToolCall
from gaja.llm.router import Router, TOOL_CUES, volume_fast_path
from gaja.tools.base import current_session_context
from gaja.tools.registry import ToolRegistry
from gaja.tools.volume_tool import set_speaker_volume
from gaja.devices.volume import DEFAULT_STEP, STEP_FLOOR, VolumeStore, set_volume_store

from tests.conftest import FakeLLM


# ---- fast-path phrase matching ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        # Polish, with diacritics
        ("ciszej", {"action": "step", "delta": -DEFAULT_STEP}),
        ("Ciszej!", {"action": "step", "delta": -DEFAULT_STEP}),
        ("trochę ciszej", {"action": "step", "delta": -6}),
        ("dużo głośniej", {"action": "step", "delta": 24}),
        ("dużo ciszej", {"action": "step", "delta": -24}),
        ("bardzo głośniej", {"action": "step", "delta": 24}),
        ("głośniej", {"action": "step", "delta": DEFAULT_STEP}),
        ("przycisz", {"action": "step", "delta": -DEFAULT_STEP}),
        ("przycisz o 10", {"action": "step", "delta": -10}),
        ("ścisz o 5", {"action": "step", "delta": -5}),
        ("podgłośnij", {"action": "step", "delta": DEFAULT_STEP}),
        ("podgłośnij o 15", {"action": "step", "delta": 15}),
        ("ciszej o 20", {"action": "step", "delta": -20}),
        ("głośniej o 10 procent", {"action": "step", "delta": 10}),
        ("głośność 50", {"action": "set", "level": 50}),
        ("głośność na 50", {"action": "set", "level": 50}),
        ("głośność 50 procent", {"action": "set", "level": 50}),
        ("ustaw głośność na 30", {"action": "set", "level": 30}),
        ("ustaw głośność na 30 procent", {"action": "set", "level": 30}),
        ("ustaw głośność na 30%", {"action": "set", "level": 30}),
        ("wycisz", {"action": "mute"}),
        ("wyciszaj", {"action": "mute"}),
        ("na maksa", {"action": "max"}),
        ("na maksimum", {"action": "max"}),
        ("maksymalna głośność", {"action": "max"}),
        ("głośność na maks", {"action": "max"}),
        ("ustaw głośność na maksimum", {"action": "max"}),
        # wake / name prefixes (web text input; voice turns arrive with the wake phrase stripped)
        ("hej gaja, ciszej", {"action": "step", "delta": -DEFAULT_STEP}),
        ("Gaja, głośność 40", {"action": "set", "level": 40}),
        # Polish typed without diacritics
        ("glosnosc 50", {"action": "set", "level": 50}),
        ("ustaw glosnosc na 30", {"action": "set", "level": 30}),
        ("glosniej", {"action": "step", "delta": DEFAULT_STEP}),
        ("duzo glosniej", {"action": "step", "delta": 24}),
        ("maksymalna glosnosc", {"action": "max"}),
        # English
        ("quieter", {"action": "step", "delta": -DEFAULT_STEP}),
        ("louder", {"action": "step", "delta": DEFAULT_STEP}),
        ("a bit quieter", {"action": "step", "delta": -6}),
        ("much louder", {"action": "step", "delta": 24}),
        ("volume up", {"action": "step", "delta": DEFAULT_STEP}),
        ("volume down", {"action": "step", "delta": -DEFAULT_STEP}),
        ("volume down by 20", {"action": "step", "delta": -20}),
        ("volume 50", {"action": "set", "level": 50}),
        ("volume 50%", {"action": "set", "level": 50}),
        ("set volume to 30", {"action": "set", "level": 30}),
        ("mute", {"action": "mute"}),
        ("max volume", {"action": "max"}),
        ("full volume", {"action": "max"}),
    ],
)
def test_volume_fast_path_matches_deterministic_commands(utterance: str, expected: dict[str, Any]) -> None:
    call = volume_fast_path(utterance)
    assert call is not None, utterance
    assert call.name == "set_speaker_volume"
    assert call.arguments == expected, utterance


@pytest.mark.parametrize(
    "utterance",
    [
        "",  # empty
        "cisza",  # the word "silence" is not a mute command
        "jaka jest pogoda",  # unrelated
        "ustaw minutnik na 10 minut",  # "ustaw ... na 10" without a volume keyword
        "opowiedz coś głośno",  # mid-sentence word, anchored match must refuse
        "1000",  # out of range
        "głośność 150",  # out of range
        "ciszej o 150",  # out of range
        "zapytaj Hermesa żeby był ciszej",  # explicitly addressed to Hermes
        "hermes, ciszej",
        "kim jest Hermes",  # mythology question mentioning nothing volume-like anyway
    ],
)
def test_volume_fast_path_refuses_non_commands(utterance: str) -> None:
    assert volume_fast_path(utterance) is None, utterance


# ---- router integration: no LLM round trip ---------------------------------------------------------


def _router_with_volume_tool() -> Router:
    return Router(
        FakeLLM(),  # must never be called for fast-path utterances
        "router-sys",
        threshold=0.97,
        history_turns=2,
        timeout_s=2.0,
        tools=ToolRegistry().schemas(),
    )


@pytest.mark.asyncio
async def test_router_classifies_volume_commands_without_touching_the_llm() -> None:
    router = _router_with_volume_tool()
    route = await router.classify("ciszej", [])
    assert route.kind == "tool"
    assert route.reason == "fast-path"

    call = await router.select_tool("dużo głośniej", [])
    assert call is not None and call.name == "set_speaker_volume"
    assert call.arguments == {"action": "step", "delta": 24}


@pytest.mark.asyncio
async def test_router_still_answers_hermes_address_before_the_volume_fast_path() -> None:
    router = _router_with_volume_tool()
    route = await router.classify("hermes, ciszej", [])
    assert route.kind == "hermes"


def test_volume_tool_has_a_selection_cue() -> None:
    cue = TOOL_CUES["set_speaker_volume"]
    assert cue.search("przycisz trochę muzykę")
    assert cue.search("podkręć głośność")
    assert cue.search("turn the volume down")


# ---- local tool handler ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_sets_level_and_confirms_at_the_new_level() -> None:
    store = VolumeStore(None)
    set_volume_store(store)
    try:
        token = current_session_context.set({"device_id": "atom-pokoj", "session_id": "s1"})
        try:
            res = await set_speaker_volume(action="step", delta=-DEFAULT_STEP, lang="pl")
            assert res.error is None
            assert res.data["level"] == 100 - DEFAULT_STEP
            assert res.data["device_id"] == "atom-pokoj"
            assert "osiemdziesiąt osiem" in res.speak and "procent" in res.speak  # digits normalized for TTS

            res = await set_speaker_volume(action="mute", lang="pl")
            assert res.data["level"] == 0 and "Wycisz" in res.speak

            res = await set_speaker_volume(action="max", lang="pl")
            assert res.data["level"] == 100 and "maks" in res.speak.lower()

            res = await set_speaker_volume(action="set", level=30, lang="pl")
            assert res.data["level"] == 30 and "trzydzieści" in res.speak

            res = await set_speaker_volume(action="set", level=30, lang="en")
            assert res.data["level"] == 30 and "percent" in res.speak
        finally:
            current_session_context.reset(token)
    finally:
        set_volume_store(None)


@pytest.mark.asyncio
async def test_volume_percent_noun_stays_procent_after_a_numeral() -> None:
    store = VolumeStore(None)
    set_volume_store(store)
    try:
        token = current_session_context.set({"device_id": "atom-pokoj"})
        try:
            for level in (2, 3, 4):
                res = await set_speaker_volume(action="set", level=level, lang="pl")
                assert res.error is None
                assert "procenty" not in res.speak
                assert res.speak.rstrip(".").endswith("procent")
        finally:
            current_session_context.reset(token)
    finally:
        set_volume_store(None)


@pytest.mark.asyncio
async def test_tool_stepping_stops_at_the_floor_and_says_so() -> None:
    store = VolumeStore(None)
    set_volume_store(store)
    try:
        token = current_session_context.set({"device_id": "atom-pokoj"})
        try:
            await set_speaker_volume(action="set", level=8, lang="pl")
            res = await set_speaker_volume(action="step", delta=-DEFAULT_STEP, lang="pl")
            assert res.data["level"] == STEP_FLOOR
            res = await set_speaker_volume(action="step", delta=-DEFAULT_STEP, lang="pl")
            assert res.data["level"] == STEP_FLOOR
            assert "minimum" in res.speak
        finally:
            current_session_context.reset(token)
    finally:
        set_volume_store(None)


@pytest.mark.asyncio
async def test_tool_without_a_device_speaks_a_graceful_refusal() -> None:
    set_volume_store(VolumeStore(None))
    try:
        token = current_session_context.set({"session_id": "web-1"})
        try:
            res = await set_speaker_volume(action="set", level=40, lang="pl")
            assert res.error is None
            assert "głośnik" in res.speak.lower()
            res_en = await set_speaker_volume(action="set", level=40, lang="en")
            assert "device" in res_en.speak.lower()
        finally:
            current_session_context.reset(token)
    finally:
        set_volume_store(None)


@pytest.mark.asyncio
async def test_tool_without_a_store_reports_unavailable() -> None:
    set_volume_store(None)
    token = current_session_context.set({"device_id": "atom-pokoj"})
    try:
        res = await set_speaker_volume(action="set", level=40, lang="pl")
        assert res.error == "volume_unavailable"
    finally:
        current_session_context.reset(token)


@pytest.mark.asyncio
async def test_registry_executes_the_tool_and_speaks_the_new_level() -> None:
    store = VolumeStore(None)
    set_volume_store(store)
    try:
        registry = ToolRegistry()
        assert "set_speaker_volume" in registry.names()
        token = current_session_context.set({"device_id": "atom-pokoj"})
        try:
            result = await registry.execute("set_speaker_volume", {"action": "set", "level": 60}, lang="pl")
        finally:
            current_session_context.reset(token)
        assert result.error is None
        assert result.data["level"] == 60
        assert "sześćdziesiąt" in result.speak
    finally:
        set_volume_store(None)


# ---- pipeline router-down fallback -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_down_fallback_executes_the_volume_fast_path(make_services) -> None:
    """With the router LLM unavailable, "ciszej" still works (same fallback as the clock)."""
    from gaja.devices.registry import DeviceInfo
    from gaja.pipeline import Session
    from tests.test_device_protocol import FAST, HELLO, TOKEN
    from tests.test_device_volume_gain import RecordingOutbox

    store = VolumeStore(None)
    set_volume_store(store)
    llm = FakeLLM(down=True)
    services = make_services(llm=llm, router=True, tools=ToolRegistry(), cfg={"devices": {**FAST}})
    services.volume_store = store
    try:
        raw_out = RecordingOutbox()
        session = Session(services, raw_out, device=DeviceInfo("atom-pokoj"))
        await session.handle_text(json.dumps({**HELLO, "device_id": "atom-pokoj", "token": TOKEN}))
        raw_out.sent.clear()
        await session.handle_text(json.dumps({"type": "text", "text": "ciszej"}))
        if session._active is not None:  # the turn runs as a task; wait for it to finish
            await asyncio.gather(session._active, return_exceptions=True)
        transcripts = [m for m in raw_out.sent if isinstance(m, dict) and m.get("type") == "transcript"]
        assert store.level("atom-pokoj") == 100 - DEFAULT_STEP
        tool_events = [m for m in raw_out.sent if isinstance(m, dict) and m.get("type") == "tool"]
        assert tool_events and tool_events[0]["name"] == "set_speaker_volume"
        assert transcripts, "the command should have been transcribed and answered"
        await session.close()
    finally:
        set_volume_store(None)
