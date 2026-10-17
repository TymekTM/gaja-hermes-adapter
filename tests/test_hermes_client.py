from __future__ import annotations

import asyncio
import sys
from typing import Any

import pytest

from gaja.hermes.base import HermesEvent
from gaja.hermes.client import (
    HermesClient,
    SubprocessAcpTransport,
    clean_spoken_hint,
    detect_language,
    format_voice_prompt,
    pick_allow_option,
    with_context,
)


class FakeAcpTransport:
    """In-memory fake ACP transport for unit testing."""

    def __init__(self) -> None:
        self.is_started = False
        self.closed = False
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.cancels: list[str] = []
        self.update_handlers: dict[str, Any] = {}
        self.next_session_id = 1
        self.fail_on_prompt = False
        self.prompt_delay = 0.02

    def is_alive(self) -> bool:
        return self.is_started and not self.closed

    async def start(self) -> None:
        self.is_started = True

    async def close(self) -> None:
        self.closed = True

    def register_update_handler(self, session_id: str, handler: Any) -> None:
        self.update_handlers[session_id] = handler

    def unregister_update_handler(self, session_id: str) -> None:
        self.update_handlers.pop(session_id, None)

    async def send_cancel(self, session_id: str) -> None:
        self.cancels.append(session_id)

    async def request(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.requests.append((method, params))

        if method == "initialize":
            return {
                "protocolVersion": 1,
                "agentInfo": {"name": "hermes-agent", "version": "fake-1.0"},
            }

        elif method == "session/new":
            sid = f"fake-hermes-sid-{self.next_session_id}"
            self.next_session_id += 1
            return {"sessionId": sid}

        elif method == "session/prompt":
            sid = params["sessionId"]
            if self.fail_on_prompt:
                raise RuntimeError("Simulated model failure")

            # Simulate streaming updates to handler
            handler = self.update_handlers.get(sid)
            if handler:
                handler({
                    "sessionUpdate": "step",
                    "step": "analyzing",
                })
                handler({
                    "sessionUpdate": "tool_call",
                    "title": "search_db",
                })
                handler({
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Wynik to "},
                })
                handler({
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "391."},
                })

            await asyncio.sleep(self.prompt_delay)
            return {"stopReason": "end_turn"}

        elif method == "session/cancel":
            self.cancels.append(params.get("sessionId", ""))
            return {}

        return {}


def test_protocol_conformance() -> None:
    transport = FakeAcpTransport()
    client = HermesClient(transport=transport)
    assert hasattr(client, "submit") and callable(client.submit)
    assert hasattr(client, "health") and callable(client.health)


def test_language_detection() -> None:
    assert detect_language("Ile to jest 17 razy 23?") == "pl"
    assert detect_language("Zrób podsumowanie spotkania") == "pl"
    assert detect_language("Jaka jest dzisiaj pogoda?") == "pl"
    assert detect_language("What is the weather today?") == "en"
    assert detect_language("Calculate 17 times 23", [{"role": "user", "content": "Dzień dobry"}]) == "pl"
    assert detect_language("Hello world") == "en"


def test_clean_spoken_hint() -> None:
    # Markdown text should be cleaned for TTS
    md_text = "### Wynik\nOto obliczenia: **391**.\n- punkt 1\n- punkt 2\n```code```"
    cleaned = clean_spoken_hint(md_text)
    assert cleaned is not None
    assert "#" not in cleaned
    assert "**" not in cleaned
    assert "```" not in cleaned
    assert "391" in cleaned

    # Simple plain text without markdown needs no special spoken hint
    simple_text = "Wynik to 391."
    assert clean_spoken_hint(simple_text) is None


def test_format_voice_prompt() -> None:
    prompt_pl = format_voice_prompt("Ile to jest 17*23?", "pl")
    assert "Rozmowa głosowa" in prompt_pl
    assert "polski" in prompt_pl
    assert "Ile to jest 17*23?" in prompt_pl

    prompt_en = format_voice_prompt("What is 17*23?", "en")
    assert "voice conversation" in prompt_en
    assert "English" in prompt_en


@pytest.mark.asyncio
async def test_submit_lifecycle_and_streaming() -> None:
    transport = FakeAcpTransport()
    client = HermesClient(transport=transport)

    events: list[HermesEvent] = []
    async for event in client.submit("Ile to jest 17*23?", context=[], session_id="gaja-turn-1"):
        events.append(event)

    assert len(events) >= 3
    # First event: accepted
    assert events[0].kind == "accepted"
    assert "session" in events[0].text
    assert events[0].data["session_id"] == "gaja-turn-1"
    assert events[0].data["hermes_session_id"] == "fake-hermes-sid-1"

    # Progress events
    progress_events = [e for e in events if e.kind == "progress"]
    assert len(progress_events) >= 2
    step_events = [e for e in progress_events if e.data.get("type") == "step"]
    tool_events = [e for e in progress_events if e.data.get("type") == "tool_call"]
    delta_events = [e for e in progress_events if e.data.get("type") == "text_delta"]
    assert len(step_events) >= 1
    assert len(tool_events) >= 1
    assert len(delta_events) >= 2

    # Final event: result
    result_event = events[-1]
    assert result_event.kind == "result"
    assert result_event.text == "Wynik to 391."
    assert result_event.data["raw_text"] == "Wynik to 391."
    assert result_event.data["session_id"] == "gaja-turn-1"

    await client.aclose()


@pytest.mark.asyncio
async def test_session_continuity() -> None:
    transport = FakeAcpTransport()
    client = HermesClient(transport=transport)

    # Turn 1
    t1_events = [e async for e in client.submit("Turn 1", context=[], session_id="session-xyz")]
    assert t1_events[0].kind == "accepted"
    sid_turn1 = t1_events[0].data["hermes_session_id"]

    # Turn 2 with SAME gaja session_id
    t2_events = [e async for e in client.submit("Turn 2", context=[], session_id="session-xyz")]
    assert t2_events[0].kind == "accepted"
    sid_turn2 = t2_events[0].data["hermes_session_id"]

    # Ensure hermes session ID was reused!
    assert sid_turn1 == sid_turn2

    # session/new should have been called EXACTLY ONCE
    new_sess_calls = [r for r in transport.requests if r[0] == "session/new"]
    assert len(new_sess_calls) == 1

    # session/prompt should have been called twice with the same session ID
    prompt_calls = [r for r in transport.requests if r[0] == "session/prompt"]
    assert len(prompt_calls) == 2
    assert prompt_calls[0][1]["sessionId"] == sid_turn1
    assert prompt_calls[1][1]["sessionId"] == sid_turn1

    await client.aclose()


@pytest.mark.asyncio
async def test_new_topic_starts_a_fresh_hermes_session_without_cancelling() -> None:
    transport = FakeAcpTransport()
    client = HermesClient(transport=transport)

    t1 = [e async for e in client.submit("Turn 1", context=[], session_id="session-xyz")]
    client.new_topic("session-xyz")
    t2 = [e async for e in client.submit("Turn 2", context=[], session_id="session-xyz")]

    assert t1[0].data["hermes_session_id"] != t2[0].data["hermes_session_id"]
    assert [r[0] for r in transport.requests].count("session/new") == 2
    assert "session/cancel" not in [r[0] for r in transport.requests]
    await client.aclose()


@pytest.mark.asyncio
async def test_distinct_sessions() -> None:
    transport = FakeAcpTransport()
    client = HermesClient(transport=transport)

    t1_events = [e async for e in client.submit("Turn 1", context=[], session_id="session-a")]
    t2_events = [e async for e in client.submit("Turn 2", context=[], session_id="session-b")]

    sid_a = t1_events[0].data["hermes_session_id"]
    sid_b = t2_events[0].data["hermes_session_id"]

    assert sid_a != sid_b
    new_sess_calls = [r for r in transport.requests if r[0] == "session/new"]
    assert len(new_sess_calls) == 2

    await client.aclose()


@pytest.mark.asyncio
async def test_error_handling() -> None:
    transport = FakeAcpTransport()
    transport.fail_on_prompt = True
    client = HermesClient(transport=transport)

    events: list[HermesEvent] = []
    async for event in client.submit("Will fail", context=[], session_id="err-sess"):
        events.append(event)

    assert events[0].kind == "accepted"
    assert events[-1].kind == "error"
    assert "Simulated model failure" in events[-1].text

    await client.aclose()


@pytest.mark.asyncio
async def test_cancellation() -> None:
    transport = FakeAcpTransport()
    transport.prompt_delay = 0.5
    client = HermesClient(transport=transport)

    gen = client.submit("Will cancel", context=[], session_id="cancel-sess")
    ev1 = await gen.__anext__()
    assert ev1.kind == "accepted"

    # Read progress event so prompt_task has started
    ev2 = await gen.__anext__()
    assert ev2.kind == "progress"

    # Close/cancel the generator mid-turn
    await gen.aclose()
    await asyncio.sleep(0.01)

    assert len(transport.cancels) >= 1
    assert "fake-hermes-sid-1" in transport.cancels
    await client.aclose()


@pytest.mark.asyncio
async def test_health() -> None:
    transport = FakeAcpTransport()
    client = HermesClient(transport=transport)

    assert await client.health() is True
    await client.aclose()


ECHO_ACP = r'''
import json, sys, time
for line in sys.stdin:
    frame = json.loads(line)
    if frame.get("method") == "slow":
        time.sleep(0.3)
    print(json.dumps({"jsonrpc": "2.0", "id": frame["id"], "result": {"method": frame["method"]}}), flush=True)
'''


@pytest.mark.asyncio
async def test_cancelled_request_does_not_stop_the_reader(tmp_path) -> None:
    transport = SubprocessAcpTransport([sys.executable, "-c", ECHO_ACP], tmp_path)
    await transport.start()
    try:
        slow = asyncio.create_task(transport.request("slow", {}, timeout=10.0))
        await asyncio.sleep(0.05)
        slow.cancel()
        with pytest.raises(asyncio.CancelledError):
            await slow
        await asyncio.sleep(0.5)
        result = await transport.request("ping", {}, timeout=5.0)
        assert result == {"method": "ping"}
        assert transport._pending_requests == {}
    finally:
        await transport.close()


RECORDING_ACP = r"""
import json, sys
log = open(sys.argv[1], "a")
for line in sys.stdin:
    frame = json.loads(line)
    log.write(line)
    log.flush()
    method = frame.get("method")
    if "id" not in frame:
        continue
    if method == "big":
        result = {"blob": "x" * frame["params"]["size"]}
    elif method == "oversized_then_ok":
        print("z" * frame["params"]["size"], flush=True)
        result = {"ok": True}
    elif method == "die":
        sys.exit(0)
    else:
        result = {"method": method}
    print(json.dumps({"jsonrpc": "2.0", "id": frame["id"], "result": result}), flush=True)
"""


def recording_transport(tmp_path) -> tuple[SubprocessAcpTransport, Any]:
    log = tmp_path / "frames.jsonl"
    transport = SubprocessAcpTransport([sys.executable, "-c", RECORDING_ACP, str(log)], tmp_path)
    return transport, log


@pytest.mark.asyncio
async def test_cancel_is_sent_as_a_notification(tmp_path) -> None:
    import json

    transport, log = recording_transport(tmp_path)
    await transport.start()
    try:
        await transport.send_cancel("sid-1")
        assert await transport.request("ping", {}, timeout=5.0) == {"method": "ping"}
        frames = [json.loads(line) for line in log.read_text().splitlines()]
    finally:
        await transport.close()
    cancel = next(f for f in frames if f["method"] == "session/cancel")
    assert "id" not in cancel
    assert cancel["params"] == {"sessionId": "sid-1"}
    assert transport._pending_requests == {}


@pytest.mark.asyncio
async def test_frames_over_64_kib_are_read(tmp_path) -> None:
    transport, _ = recording_transport(tmp_path)
    await transport.start()
    try:
        result = await transport.request("big", {"size": 300_000}, timeout=5.0)
        assert len(result["blob"]) == 300_000
        assert await transport.request("ping", {}, timeout=5.0) == {"method": "ping"}
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_oversized_frame_is_dropped_and_reader_survives(tmp_path, monkeypatch) -> None:
    import gaja.hermes.client as client_mod

    monkeypatch.setattr(client_mod, "STREAM_LIMIT", 4096)
    transport, _ = recording_transport(tmp_path)
    await transport.start()
    try:
        assert await transport.request("oversized_then_ok", {"size": 20_000}, timeout=5.0) == {"ok": True}
        assert transport.is_alive()
        assert await transport.request("ping", {}, timeout=5.0) == {"method": "ping"}
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_dead_worker_fails_pending_and_respawns(tmp_path) -> None:
    transport, _ = recording_transport(tmp_path)
    await transport.start()
    try:
        first = transport.proc
        with pytest.raises(RuntimeError):
            await transport.request("die", {}, timeout=5.0)
        await asyncio.sleep(0.1)
        assert not transport.is_alive()
        assert await transport.request("ping", {}, timeout=5.0) == {"method": "ping"}
        assert transport.proc is not first
        assert transport.is_alive()
    finally:
        await transport.close()


def test_with_context_prepends_recent_turns() -> None:
    context = [
        {"role": "user", "content": "stare pytanie"},
        {"role": "assistant", "content": "stara odpowiedź"},
        {"role": "user", "content": "jaka jest pogoda w Gdańsku"},
        {"role": "assistant", "content": "Sprawdzę to."},
        {"role": "user", "content": "a jutro?"},
        {"role": "assistant", "content": "Zaraz sprawdzę."},
    ]
    prompt = with_context("sprawdź pogodę tam jutro", context, "pl")
    assert prompt.startswith("Kontekst rozmowy:\n")
    assert "Użytkownik: jaka jest pogoda w Gdańsku" in prompt
    assert "stare pytanie" not in prompt
    assert prompt.endswith("---\nZadanie: sprawdź pogodę tam jutro")
    assert with_context("task", [], "en") == "task"
    assert with_context("task", [{"role": "user", "content": "hi"}], "en").startswith("Conversation context:\nUser: hi")


class TimeoutAcpTransport(FakeAcpTransport):
    async def request(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        if method == "session/prompt":
            self.requests.append((method, params))
            await asyncio.sleep(timeout + 0.05)
            raise TimeoutError(f"ACP request {method} timed out after {timeout}s")
        return await super().request(method, params, timeout)


@pytest.mark.asyncio
async def test_timeout_sends_cancel_and_reports_error() -> None:
    transport = TimeoutAcpTransport()
    client = HermesClient(transport=transport, timeout_s=0.05)
    events = [e async for e in client.submit("slow task", context=[], session_id="t")]
    assert events[-1].kind == "error"
    assert transport.cancels == ["fake-hermes-sid-1"]
    await client.aclose()


@pytest.mark.asyncio
async def test_submit_sends_context_and_language() -> None:
    transport = FakeAcpTransport()
    client = HermesClient(transport=transport)
    context = [{"role": "user", "content": "jaka jest pogoda w Gdańsku"}]
    [e async for e in client.submit("a jutro", context=context, session_id="s", language="pl")]
    prompt = next(p for m, p in transport.requests if m == "session/prompt")
    text = prompt["prompt"][0]["text"]
    assert "Użytkownik: jaka jest pogoda w Gdańsku" in text
    assert "Zadanie: a jutro" in text
    await client.aclose()


@pytest.mark.asyncio
async def test_parallel_tasks_get_separate_acp_sessions() -> None:
    transport = FakeAcpTransport()
    transport.prompt_delay = 0.2
    client = HermesClient(transport=transport)

    async def run(task: str) -> list[HermesEvent]:
        return [e async for e in client.submit(task, context=[], session_id="same")]

    first, second = await asyncio.gather(run("one"), run("two"))
    sids = {first[0].data["hermes_session_id"], second[0].data["hermes_session_id"]}
    assert len(sids) == 2
    assert first[-1].kind == second[-1].kind == "result"
    await client.aclose()


PERMISSION_ACP = r"""
import json, sys
options = json.loads(sys.argv[1])
for line in sys.stdin:
    frame = json.loads(line)
    if frame.get("method") != "trigger":
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": 900, "method": "session/request_permission",
                      "params": {"sessionId": "s1", "options": options}}), flush=True)
    reply = json.loads(sys.stdin.readline())
    print(json.dumps({"jsonrpc": "2.0", "id": frame["id"], "result": reply}), flush=True)
"""


@pytest.mark.parametrize(
    ("options", "outcome"),
    [
        (
            [{"optionId": "allow_once", "kind": "allow_once"}, {"optionId": "allow_session", "kind": "allow_always"},
             {"optionId": "deny", "kind": "reject_once"}],
            {"outcome": "selected", "optionId": "allow_once"},
        ),
        ([{"optionId": "deny", "kind": "reject_once"}], {"outcome": "cancelled"}),
    ],
)
@pytest.mark.asyncio
async def test_permission_requests_are_auto_approved(tmp_path, options, outcome) -> None:
    import json

    transport = SubprocessAcpTransport([sys.executable, "-c", PERMISSION_ACP, json.dumps(options)], tmp_path)
    updates: list[dict[str, Any]] = []
    try:
        await transport.start()
        transport.register_update_handler("s1", updates.append)
        reply = await transport.request("trigger", {}, timeout=5.0)
    finally:
        await transport.close()
    assert reply == {"jsonrpc": "2.0", "id": 900, "result": {"outcome": outcome}}
    expected = "permission_approved" if outcome["outcome"] == "selected" else "permission_cancelled"
    assert [u["sessionUpdate"] for u in updates] == [expected]


def test_pick_allow_option() -> None:
    assert pick_allow_option([{"optionId": "a", "kind": "allow_always"}, {"optionId": "b", "kind": "allow_once"}]) == "b"
    assert pick_allow_option([{"optionId": "a", "kind": "allow_always"}]) == "a"
    assert pick_allow_option([{"optionId": "d", "kind": "reject_once"}]) is None
    assert pick_allow_option([]) is None
