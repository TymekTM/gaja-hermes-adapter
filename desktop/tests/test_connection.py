"""Session-flow and pc_call tests against the in-process fake server (no audio hardware, ephemeral ports)."""

import asyncio
import base64
import json
import socket
import time

import pytest

from gaja_desktop.audio import FRAME_BYTES
from gaja_desktop.config import ALL_TOOLS, DesktopConfig
from gaja_desktop.connection import DeviceClient, backoff_delay, pc_os_name
from gaja_desktop.tools import TEXT_LIMIT, Tool, ToolResult
from gaja_desktop.tools.screenshot import B64_LIMIT, encode_jpeg_under_limit
from gaja_desktop.tools.shell import ShellTool

from fake_server import FakeDeviceServer


class FakeCapture:
    def __init__(self, spec: str, callback) -> None:
        self.spec = spec
        self.callback = callback
        self.started = False
        self.stopped = False
        self.error = None

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakePlayback:
    def __init__(self, specs: list[str], rate: int) -> None:
        self.specs = specs
        self.rate = rate
        self.written: list[bytes] = []
        self.busy_flag = False
        self.closed = False
        self.started = False

    def start(self) -> None:
        self.started = True

    def play(self, pcm: bytes) -> None:
        self.written.append(pcm)

    def busy(self) -> bool:
        return self.busy_flag

    def cancel(self) -> None:
        self.written.clear()

    def close(self) -> None:
        self.closed = True


class FakeWake:
    def __init__(self, fire_after: int = 3) -> None:
        self.armed = True
        self.count = 0
        self.fire_after = fire_after
        self.resets = 0
        self.score = 0.97

    def process(self, frame: bytes) -> bool:
        self.count += 1
        if self.armed and self.count >= self.fire_after:
            self.armed = False
            return True
        return False

    def reset(self) -> None:
        self.resets += 1
        self.armed = True
        self.count = 0


class FakeTool(Tool):
    def __init__(self, name: str, result: ToolResult | None = None, delay: float = 0.0) -> None:
        self._name = name
        self._result = result
        self._delay = delay
        self.calls: list[tuple[dict, float]] = []

    @property
    def name(self) -> str:  # instance-level override on purpose
        return self._name

    def available(self) -> bool:
        return True

    def run(self, args: dict, timeout_s: float) -> ToolResult:
        self.calls.append((args, timeout_s))
        if self._delay:
            time.sleep(self._delay)
        if self._result is not None:
            return self._result
        return ToolResult(ok=True, text=f"{self._name} done")


def make_frame(i: int = 0) -> bytes:
    return bytes([i % 256]) * FRAME_BYTES


def make_client(server: FakeDeviceServer, cfg: DesktopConfig | None = None, **kwargs) -> DeviceClient:
    cfg = cfg or DesktopConfig(url=server.url, device_id="pc-test", token="tok-123")
    cfg.url = server.url
    if "tools" not in kwargs:
        kwargs["tools"] = [FakeTool(name) for name in ALL_TOOLS]
    kwargs.setdefault("capture_factory", lambda spec, cb: FakeCapture(spec, cb))
    kwargs.setdefault("playback_factory", lambda specs, rate: FakePlayback(specs, rate))
    kwargs.setdefault("wake_factory", lambda: FakeWake())
    return DeviceClient(cfg, **kwargs)


async def run_until_idle(client: DeviceClient, timeout: float = 5.0) -> asyncio.Task:
    task = asyncio.create_task(client.run())
    await asyncio.wait_for(client._ready_event.wait(), timeout)
    assert client.state == "idle"
    return task


@pytest.fixture
async def server():
    s = FakeDeviceServer()
    await s.start()
    yield s
    await s.stop()


@pytest.fixture
async def running(server):
    client = make_client(server)
    task = await run_until_idle(client)
    yield client
    client.stop()
    try:
        await asyncio.wait_for(task, 3)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()


# ---- hello -------------------------------------------------------------------------------


async def test_hello_caps_content(server) -> None:
    cfg = DesktopConfig(url=server.url, device_id="pc-tymek", token="tok-abc", lang="pl",
                        preroll_ms=1500, out_rates=[48000, 24000, 16000])
    client = make_client(server, cfg, pc_enabled=True)
    task = await run_until_idle(client)
    client.stop()
    await asyncio.wait_for(task, 3)

    hello = server.hellos[0]
    assert hello["type"] == "hello" and hello["client"] == "device"
    assert hello["device_id"] == "pc-tymek" and hello["token"] == "tok-abc"
    assert hello["fw"].startswith("desktop-")
    assert hello["lang"] == "pl"
    caps = hello["caps"]
    assert caps["aec"] is False  # half duplex by default
    assert caps["out_rates"] == [48000, 24000, 16000]  # 48000 offered first
    assert caps["preroll_ms"] == 1500
    assert caps["ptt"] is True
    assert caps["pc"]["os"] == pc_os_name() == "linux"
    assert caps["pc"]["host"] == socket.gethostname()[:64]
    assert caps["pc"]["tools"] == ALL_TOOLS  # only tools enabled and available


async def test_hello_without_pc_when_toggled_off(server) -> None:
    client = make_client(server, pc_enabled=False)
    task = await run_until_idle(client)
    client.stop()
    await asyncio.wait_for(task, 3)
    caps = server.hellos[0]["caps"]
    assert caps["pc"] is None  # absent (None) -> the server treats it as a plain voice device


# ---- wake -> preroll -> frames -----------------------------------------------------------


async def test_wake_preroll_frames_and_reject(server, running) -> None:
    client = running
    # feed 1.5 s of audio; the fake detector fires on the 3rd frame
    for _ in range(75):
        client.on_mic_frame(make_frame())
    wake = await server.wait_event("wake", 5)
    assert wake["word"] == "hej_gaja"
    assert wake["preroll_ms"] == 1500  # the whole ring goes out as the pre-roll
    assert wake["score"] == pytest.approx(0.97)
    await server.wait_frames(75, 5)  # the pre-roll burst arrived

    # live frames stream in real-time pace: a few in the next 200 ms, not a burst
    await asyncio.sleep(0.02)
    count_now = len(server.binary)
    for _ in range(10):
        client.on_mic_frame(make_frame())
    await asyncio.sleep(0.08)
    mid = len(server.binary)
    assert 0 < mid - count_now < 20  # paced, not bursted

    await server.send({"type": "wake_rejected", "reason": "no_phrase"})
    await server.wait_for(lambda: client.state == "idle", 3, "back to idle")
    client._wake.armed = False  # keep the (re-armed) fake detector quiet for this check
    count = len(server.binary)
    for _ in range(10):
        client.on_mic_frame(make_frame())
    await asyncio.sleep(0.15)
    assert len(server.binary) == count  # streaming stopped after the rejection
    assert client._wake.resets >= 1  # the detector re-armed


async def test_push_to_talk_wake(server, running) -> None:
    client = running
    await client.push_to_talk()
    wake = await server.wait_event("wake", 5)
    assert wake["word"] == "push_to_talk"
    assert wake["score"] == 1.0
    assert wake["preroll_ms"] == 0  # the press is the intent, no pre-roll


# ---- full session: audio -> follow_up -> listen_end ---------------------------------------


async def test_session_audio_follow_up_listen_end(server, running) -> None:
    client = running
    playback = client._playback
    for _ in range(10):
        client.on_mic_frame(make_frame())
    await server.wait_event("wake", 5)
    await server.wait_frames(10, 5)
    streamed = len(server.binary)
    assert streamed >= 10

    # the server answers: audio plays (half duplex -> the mic pauses)
    await server.send({"type": "audio_start", "id": "u1", "sample_rate": server.rate})
    await server.send_audio(0.3)
    await server.send({"type": "audio_end", "id": "u1"})
    await server.wait_for(lambda: client.state == "speaking", 3, "speaking state")
    assert len(playback.written) >= 2
    count = len(server.binary)
    for _ in range(5):
        client.on_mic_frame(make_frame())
    await asyncio.sleep(0.1)
    assert len(server.binary) == count  # half duplex: no upload while Gaja speaks

    await server.send({"type": "follow_up", "ms": 2000})
    # once the audio has drained, the follow-up window opens the mic without a wake
    await server.wait_for(lambda: client.state == "follow_up", 3, "follow_up state")
    frames_before = len(server.binary)
    for _ in range(5):
        client.on_mic_frame(make_frame())
    await server.wait_frames(frames_before + 5, 3)  # streaming again, no wake in between
    assert server.event("wake")["word"] == "hej_gaja"  # no new wake message was sent

    await server.send({"type": "listen_end"})
    await server.wait_for(lambda: client.state == "idle", 3, "idle after listen_end")
    client._wake.armed = False  # keep the (re-armed) fake detector quiet for this check
    count = len(server.binary)
    for _ in range(5):
        client.on_mic_frame(make_frame())
    await asyncio.sleep(0.1)
    assert len(server.binary) == count  # mic closed again


async def test_playback_stop_cancels_audio(server, running) -> None:
    client = running
    playback = client._playback
    for _ in range(5):
        client.on_mic_frame(make_frame())
    await server.wait_event("wake", 5)
    await server.send_audio(0.3)
    await server.wait_for(lambda: client.state == "speaking", 3, "speaking")
    playback.busy_flag = True
    await server.send({"type": "playback_stop"})
    await server.wait_for(lambda: not playback.busy_flag or not playback.written, 2, "playback dropped")
    await server.wait_for(lambda: client.state == "idle", 3, "idle after playback_stop")
    assert client._wake.armed  # wake word listens again


# ---- reconnect ---------------------------------------------------------------------------


async def test_reconnect_after_drop(server) -> None:
    client = make_client(server)
    task = await run_until_idle(client)
    await server.close_connections()
    # the client reconnects (first attempt ~0.5 s +- 15%) and gets a ready again
    await server.wait_for(lambda: len(server.hellos) >= 2, 5, "second hello")
    await server.wait_for(lambda: client.state == "idle", 5, "idle after reconnect")
    client.stop()
    await asyncio.wait_for(task, 3)


async def test_backoff_delay_bounds() -> None:
    for attempt, base in [(0, 0.5), (1, 0.85), (2, 1.445), (5, 7.08), (6, 10.0), (20, 10.0)]:
        for _ in range(20):
            delay = backoff_delay(attempt)
            assert base * 0.85 <= delay <= min(base, 10.0) * 1.15 + 1e-9


# ---- pc tools over the protocol -----------------------------------------------------------


async def test_pc_call_each_tool(server, running) -> None:
    client = running
    shell = FakeTool("shell")
    screenshot = FakeTool("screenshot")
    client.tools = {"shell": shell, "screenshot": screenshot}

    await server.send({"type": "pc_call", "id": "0a1b2c3d4e5f", "tool": "shell",
                       "args": {"command": "echo hi"}, "timeout_s": 5})
    result = await server.wait_for(lambda: next((r for r in server.results if r["id"] == "0a1b2c3d4e5f"), None),
                                   5, "shell result")
    assert result["ok"] is True and result["text"] == "shell done"
    assert shell.calls and shell.calls[0][0] == {"command": "echo hi"}
    assert shell.calls[0][1] == 5

    await server.send({"type": "pc_call", "id": "ffffffffffff", "tool": "screenshot", "args": {}})
    await server.wait_for(lambda: any(r["id"] == "ffffffffffff" for r in server.results), 5, "screenshot result")
    assert screenshot.calls[0][1] == 60.0  # the server's default timeout was applied


async def test_pc_call_real_shell_and_timeout(server, running) -> None:
    client = running
    client.tools = {"shell": ShellTool()}
    started = time.monotonic()
    await server.send({"type": "pc_call", "id": "abc123def456", "tool": "shell",
                       "args": {"command": "printf %s hello"}, "timeout_s": 10})
    result = await server.wait_for(lambda: next((r for r in server.results if r["id"] == "abc123def456"), None),
                                   8, "shell result")
    assert result["ok"] is True and result["text"] == "hello"
    assert result["data"]["exit_code"] == 0

    await server.send({"type": "pc_call", "id": "abc123def457", "tool": "shell",
                       "args": {"command": "sleep 30"}, "timeout_s": 1})
    result = await server.wait_for(lambda: next((r for r in server.results if r["id"] == "abc123def457"), None),
                                   8, "timeout result")
    assert result["ok"] is False and result["error"] == "timeout"
    assert time.monotonic() - started < 6  # the client enforced the timeout itself


async def test_pc_call_output_truncated(server, running) -> None:
    client = running
    client.tools = {"shell": ShellTool()}
    await server.send({"type": "pc_call", "id": "aaa000bbb111", "tool": "shell",
                       "args": {"command": "head -c 200000 /dev/zero | tr '\\0' 'a'"}, "timeout_s": 15})
    result = await server.wait_for(lambda: next((r for r in server.results if r["id"] == "aaa000bbb111"), None),
                                   15, "truncated result")
    assert result["ok"] is True
    assert len(result["text"]) <= TEXT_LIMIT and "truncated" in result["text"]


async def test_pc_call_screenshot_frame_under_wire_limit(server, running) -> None:
    """The screenshot's b64 stays <= 900 KiB so the whole frame survives the 1 MiB ws_max_size."""
    client = running
    from PIL import Image

    def big_screenshot(args: dict, timeout_s: float) -> ToolResult:
        img = Image.effect_noise((4000, 2500), 60).convert("RGB")
        b64, width, height = encode_jpeg_under_limit(img, max_width=args.get("max_width", 1600))
        return ToolResult(ok=True, text=f"{width}x{height}",
                          image={"mime": "image/jpeg", "b64": b64, "width": width, "height": height})

    client.tools = {"screenshot": FakeTool("screenshot")}
    client.tools["screenshot"].run = big_screenshot  # type: ignore[method-assign]
    await server.send({"type": "pc_call", "id": "ccc000ddd222", "tool": "screenshot", "args": {}})
    result = await server.wait_for(lambda: next((r for r in server.results if r["id"] == "ccc000ddd222"), None),
                                   15, "screenshot result")
    assert result["ok"] is True
    image = result["image"]
    assert image["mime"] == "image/jpeg"
    assert len(image["b64"]) <= B64_LIMIT
    assert base64.b64decode(image["b64"])[:3] == b"\xff\xd8\xff"
    frame_bytes = len(json.dumps(result).encode())
    assert frame_bytes < 1 << 20  # survives the listener's ws_max_size


async def test_pc_call_unknown_and_unadvertised_tool(server, running) -> None:
    client = running
    client.tools = {"shell": FakeTool("shell")}  # screenshot was never advertised
    await server.send({"type": "pc_call", "id": "aaa111aaa111", "tool": "screenshot", "args": {}})
    result = await server.wait_for(lambda: next((r for r in server.results if r["id"] == "aaa111aaa111"), None),
                                   5, "unadvertised tool result")
    assert result["ok"] is False and result["error"].startswith("bad_request:")

    await server.send({"type": "pc_call", "id": "aaa111aaa112", "tool": "format_disk", "args": {}})
    result = await server.wait_for(lambda: next((r for r in server.results if r["id"] == "aaa111aaa112"), None),
                                   5, "unknown tool result")
    assert result["ok"] is False and result["error"].startswith("bad_request:")


async def test_pc_calls_run_concurrently_and_audio_flows(server, running) -> None:
    client = running
    slow = FakeTool("shell", delay=0.8)
    fast = FakeTool("screenshot")
    client.tools = {"shell": slow, "screenshot": fast}
    started = time.monotonic()
    await server.send({"type": "pc_call", "id": "ddd000eee111", "tool": "shell", "args": {}})
    await server.send({"type": "pc_call", "id": "ddd000eee222", "tool": "screenshot", "args": {}})
    await server.wait_for(lambda: len(server.results) >= 2, 5, "both results")
    elapsed = time.monotonic() - started
    assert elapsed < 1.6  # concurrent: not 0.8 + tool time, serialized

    # and the mic stream is not stalled by a slow tool
    for _ in range(3):
        client.on_mic_frame(make_frame())
    await server.wait_event("wake", 3)
    count = len(server.binary)
    for _ in range(5):
        client.on_mic_frame(make_frame())
    await server.wait_frames(count + 3, 3)


async def test_pc_call_ignored_without_pc_cap(server) -> None:
    s = FakeDeviceServer(ready_pc=False)  # an old server that did not accept caps.pc
    await s.start()
    try:
        client = make_client(s)
        task = await run_until_idle(client)
        assert client.saw_pc is False
        await s.send({"type": "pc_call", "id": "eee111eee111", "tool": "shell", "args": {}})
        await asyncio.sleep(0.4)
        assert all(r["id"] != "eee111eee111" for r in s.results)
        client.stop()
        await asyncio.wait_for(task, 3)
    finally:
        await s.stop()


async def test_app_ping_gets_pong(server, running) -> None:
    await server.send({"type": "ping"})
    pong = await server.wait_event("pong", 5)
    assert pong["type"] == "pong"


async def test_unauthorized_stops_the_client(server) -> None:
    async def refuse(ws, hello) -> None:
        await ws.send(json.dumps({"type": "error", "message": "unauthorized"}))

    s = FakeDeviceServer(on_hello=refuse)
    await s.start()
    try:
        client = make_client(s)
        task = asyncio.create_task(client.run())
        await asyncio.wait_for(task, 5)  # run() exits by itself: a bad token will not fix itself
        assert client._stopping.is_set()
    finally:
        await s.stop()
