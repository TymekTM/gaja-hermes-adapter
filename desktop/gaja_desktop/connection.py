"""The device-protocol client (v1.13 flow + v1.14 caps): asyncio websocket, session state machine,
reconnect with backoff, and the `pc_call` executor.

Session flow (docs/DESKTOP.md "Desktop client", docs/DEVICE.md): idle -> `wake` (+ pre-roll burst)
-> stream 20 ms frames until `wake_rejected` / `listen_end` / the reply ends; `follow_up{ms}`
re-opens the mic without a wake for `ms`; drops reconnect with backoff 0.5 s x 1.7^n, max 10 s,
±15 % jitter (reset on `ready`).

Audio never blocks the loop: capture runs in its own thread and hands frames to `on_mic_frame`;
playback runs in its own threads. PC tool calls run concurrently in worker threads, each answered
exactly once, `timeout_s` enforced by the client itself.

Everything hardware-ish (capture, playback, wake detector) is injected through factories so tests
run against fakes on a headless machine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import socket
import threading
import time
from collections import deque
from random import uniform
from typing import Any, Callable

import websockets

from . import FW
from .audio import FRAME_MS, MultiOutput
from .config import DesktopConfig
from .tools import TEXT_LIMIT, Tool, ToolResult, create_tools, truncate_middle

log = logging.getLogger(__name__)

#: hello must arrive first and stay small (the listener enforces both).
HELLO_TIMEOUT_S = 5.0
#: A missing `ready` counts as a failed attempt (the firmware uses 6 s too).
READY_TIMEOUT_S = 6.0
#: Server-side calls run 1..300 s; the client clamps, the executor adds a small grace.
MAX_TIMEOUT_S = 300.0
DEFAULT_TIMEOUT_S = 60.0
#: Belt-and-braces grace over the tool's own timeout before the outer answer goes out.
EXECUTOR_GRACE_S = 1.0
#: Live mic frames in flight at most (20 ms each); beyond this the link is effectively stalled.
MIC_QUEUE_LIMIT = 500
#: A `pc_call` id, per the protocol.
PC_ID_CHARS = set("0123456789abcdef")


def pc_os_name() -> str:
    """platform.system() in the hello's vocabulary."""
    name = platform.system().lower()
    return {"darwin": "macos"}.get(name, name)


class ProtocolError(RuntimeError):
    """The server spoke nonsense or refused us (unauthorized); the reconnect loop treats it like a drop."""


class DeviceClient:
    """One client owning a connection, the audio threads and the tool executor."""

    def __init__(
        self,
        cfg: DesktopConfig,
        *,
        tools: list[Tool] | None = None,
        capture_factory: Callable[[str, Callable[[bytes], None]], Any] | None = None,
        playback_factory: Callable[[list[str], int], Any] | None = None,
        wake_factory: Callable[[], Any] | None = None,
        pc_enabled: bool = True,
        on_state: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.pc_enabled = pc_enabled  # the "PC actions" tray switch
        self.muted = False  # the mute-listening tray switch
        self.on_state = on_state
        self._tools_list = tools if tools is not None else create_tools(cfg.tools)
        self.tools = {t.name: t for t in self._tools_list}
        self._capture_factory = capture_factory
        self._playback_factory = playback_factory
        self._wake_factory = wake_factory
        self._wake = wake_factory() if wake_factory else None

        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws: Any = None
        self._stopping = threading.Event()
        self._stop_evt: asyncio.Event | None = None
        self._force_reconnect: asyncio.Event | None = None
        self._send_lock: asyncio.Lock | None = None  # created per connection
        self._attempt = 0
        self._unauthorized = False

        # session state
        self.state = "connecting"
        self.out_rate = cfg.out_rates[0]
        self.saw_pc = False  # ready.device.pc: only then may pc_result be sent
        self.follow_up_ms = 0
        self._follow_until = 0.0
        self._speaking = False
        self._last_audio_t = 0.0
        self._mic_queue: deque[bytes] = deque(maxlen=MIC_QUEUE_LIMIT)
        self._frames_ready: asyncio.Event | None = None
        self._ring: deque[bytes] = deque()
        self._ring_lock = threading.Lock()
        self._upload_live = False  # mic thread: frames go to the server, not the wake detector
        self._playback: Any = None
        self._capture: Any = None
        self._pc_calls: dict[str, bool] = {}  # answered-once guard per call id
        self._tasks: set[asyncio.Task] = set()
        self._ready_event: asyncio.Event | None = None  # set for tests/startup

    # ---- lifecycle -------------------------------------------------------------------------

    async def run(self) -> None:
        """Run until `stop()`: connect, serve the session, reconnect with backoff on every drop."""
        self._loop = asyncio.get_running_loop()
        self._frames_ready = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._stop_evt = asyncio.Event()
        self._force_reconnect = asyncio.Event()
        self._start_audio()
        try:
            while not self._stopping.is_set():
                self._set_state("connecting")
                try:
                    await self._connect_and_run()
                except ProtocolError as exc:
                    log.warning("server refused us: %s", exc)
                except (OSError, websockets.WebSocketException, TimeoutError, RuntimeError) as exc:
                    log.info("connection lost: %s: %s", type(exc).__name__, exc)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - never let the client die silently
                    log.exception("unexpected error in the device session: %s", exc)
                if self._stopping.is_set():
                    break
                self._attempt += 1
                delay = min(0.5 * 1.7 ** (self._attempt - 1), 10.0) * uniform(0.85, 1.15)
                log.info("reconnecting in %.1f s (attempt %d)", delay, self._attempt)
                await self._sleep_or_event(delay)
        finally:
            self._stop_audio()

    async def _sleep_or_event(self, delay: float) -> None:
        """The backoff wait, ended early by stop() or request_reconnect()."""
        stop_task = asyncio.create_task(self._stop_evt.wait())
        force_task = asyncio.create_task(self._force_reconnect.wait())
        done, pending = await asyncio.wait({stop_task, force_task}, timeout=delay,
                                           return_when=asyncio.FIRST_COMPLETED)
        for task in (*done, *pending):
            task.cancel()
        self._force_reconnect.clear()

    def stop(self) -> None:
        """Thread-safe shutdown request (tray Quit, tests)."""
        self._stopping.set()
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._wake_waiters)

    def _wake_waiters(self) -> None:
        if self._stop_evt is not None:
            self._stop_evt.set()
        if self._force_reconnect is not None:
            self._force_reconnect.set()
        self._close_ws()

    def request_reconnect(self) -> None:
        """Re-open the session now (the PC-actions toggle changed the hello caps)."""
        self._attempt = 0
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._close_ws)

    def _close_ws(self) -> None:
        if self._ws is not None:
            try:
                asyncio.create_task(self._ws.close())
            except RuntimeError:
                pass

    # ---- hello / caps ----------------------------------------------------------------------

    def hello_caps(self) -> dict[str, Any]:
        """The `caps` object of the hello (v1.14: ptt and pc)."""
        caps: dict[str, Any] = {
            "aec": not self.cfg.half_duplex,  # full-duplex mode means "we may stream while playing"
            "out_rates": list(self.cfg.out_rates),
            "preroll_ms": self.cfg.preroll_ms,
        }
        if self.cfg.ptt_enabled:
            caps["ptt"] = True
        if self.pc_enabled:
            caps["pc"] = {
                "os": pc_os_name(),
                "host": socket.gethostname()[:64],
                "tools": [t.name for t in self._tools_list],  # only enabled and runnable tools
            }
        else:
            caps["pc"] = None
        return caps

    async def _connect_and_run(self) -> None:
        async with websockets.connect(self.cfg.url, max_size=1 << 20) as ws:
            self._ws = ws
            self._send_lock = asyncio.Lock()
            self._pc_calls = {}
            self._unauthorized = False
            hello = {"type": "hello", "client": "device", "device_id": self.cfg.device_id,
                     "token": self.cfg.token, "fw": FW, "lang": self.cfg.lang,
                     "caps": self.hello_caps()}
            await ws.send(json.dumps(hello))
            log.info("hello sent: device_id=%s tools=%s pc=%s", self.cfg.device_id,
                     [t.name for t in self._tools_list], self.pc_enabled)
            self._set_state("connecting")
            ready = json.loads(await asyncio.wait_for(ws.recv(), HELLO_TIMEOUT_S))
            if ready.get("type") != "ready":
                raise ProtocolError(f"expected ready, got {ready.get('type')}")
            self._on_ready(ready)
            receiver = asyncio.create_task(self._receive(ws))
            sender = asyncio.create_task(self._mic_sender(ws))
            monitor = asyncio.create_task(self._playback_monitor())
            try:
                await receiver  # returns when the socket drops
            finally:
                for task in (sender, monitor):
                    task.cancel()
                    with contextlib_suppress():
                        await task
                self._reset_session()
            if self._unauthorized:
                raise ProtocolError("unauthorized")

    def _on_ready(self, ready: dict[str, Any]) -> None:
        self._attempt = 0  # a ready resets the backoff ladder
        device = ready.get("device") or {}
        rate = int(ready.get("sample_rate") or device.get("out_rate") or self.out_rate)
        self.saw_pc = bool(device.get("pc"))
        self.follow_up_ms = int(device.get("follow_up_ms") or 0)
        self._set_playback_rate(rate)
        self._set_state("idle")
        self._arm_wake()
        if self._ready_event is not None:
            self._ready_event.set()
        log.info("ready: rate=%d pc=%s follow_up=%d ms", rate, self.saw_pc, self.follow_up_ms)

    def _reset_session(self) -> None:
        """Back to a fresh pre-session state (connection lost)."""
        self._upload_live = False
        self._speaking = False
        self._follow_until = 0.0
        self._pc_calls = {}
        with self._ring_lock:
            self._ring.clear()
        self._mic_queue.clear()
        if self._frames_ready is not None:
            self._frames_ready.set()  # wake the sender so it can exit

    # ---- state -----------------------------------------------------------------------------

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            log.debug("state -> %s", state)
            if self.on_state is not None:
                try:
                    self.on_state(state)
                except Exception:  # noqa: BLE001 - UI callbacks must not kill the client
                    log.exception("state callback failed")

    # ---- audio threads ---------------------------------------------------------------------

    def _start_audio(self) -> None:
        if self._playback_factory is None:
            self._playback_factory = lambda specs, rate: MultiOutput(specs, rate)
        if self._capture_factory is None:
            from .audio import Capture  # noqa: PLC0415

            self._capture_factory = lambda spec, cb: Capture(spec, cb)
        self._set_playback_rate(self.cfg.out_rates[0])
        self._capture = self._capture_factory(self.cfg.input, self.on_mic_frame)
        if self._wake_factory is not None:
            self._wake = self._wake_factory()
        self._capture.start()

    def _stop_audio(self) -> None:
        if self._capture is not None:
            self._capture.stop()
            self._capture = None
        if self._playback is not None:
            self._playback.close()
            self._playback = None

    def _set_playback_rate(self, rate: int) -> None:
        if self._playback is not None and getattr(self._playback, "rate", None) == rate:
            return
        if self._playback is not None:
            self._playback.close()
        self.out_rate = rate
        self._playback = self._playback_factory(list(self.cfg.outputs), rate)
        if hasattr(self._playback, "start"):
            self._playback.start()
        if getattr(self._playback, "errors", None):
            for err in self._playback.errors:
                log.warning("playback device problem: %s", err)

    # ---- mic frames (capture thread context) ----------------------------------------------

    def on_mic_frame(self, frame: bytes) -> None:
        """Called from the capture thread for every 20 ms frame. Cheap, never blocking."""
        if self.muted:
            return
        with self._ring_lock:  # pre-roll ring: keep the last preroll_ms even while speaking
            self._ring.append(frame)
            while len(self._ring) > self.cfg.preroll_ms // FRAME_MS + 1:
                self._ring.popleft()
        loop = self._loop
        if loop is None or self._ws is None:
            return
        if self.cfg.half_duplex and self._speaking:
            return  # Gaja is talking: no upload, no wake detection
        if self._upload_live:
            self._mic_queue.append(frame)  # bounded: when the link stalls, the oldest audio falls out
            if self._frames_ready is not None:
                loop.call_soon_threadsafe(self._frames_ready.set)
            return
        if self.state == "idle" and self.cfg.wake_enabled and self._wake is not None:
            try:
                fired = self._wake.process(frame)
            except Exception:  # noqa: BLE001 - a broken detector must not stop the audio
                log.exception("wake detector failed")
                fired = False
            if fired:
                loop.call_soon_threadsafe(self._wake_fired, "hej_gaja",
                                          getattr(self._wake, "score", 0.97))

    def _wake_fired(self, word: str, score: float) -> None:
        asyncio.create_task(self.send_wake(word, score))

    # ---- session flow ----------------------------------------------------------------------

    async def send_wake(self, word: str = "hej_gaja", score: float = 0.97, *, preroll: bool = True) -> None:
        """Open the mic window: `wake` message, pre-roll burst, then paced live frames."""
        ws = self._ws
        if ws is None or self._stopping.is_set():
            return
        if self.state not in ("idle", "follow_up", "speaking"):
            return  # already streaming
        speaking = self._speaking
        if speaking:  # a wake barges in: drop the playback like the firmware does
            self._cancel_playback()
        self._disarm_wake_switch(False)  # frames stay in the ring until the burst is out
        with self._ring_lock:
            ring = list(self._ring)
            self._ring.clear()
        # When our own playback was audible the pre-roll only holds Gaja's voice: skip it.
        skip_preroll = speaking and self.cfg.half_duplex
        frames = [] if (speaking or not preroll or skip_preroll) else ring
        preroll_ms = len(frames) * FRAME_MS
        self._set_state("listening")
        await self._send_json({"type": "wake", "word": word, "score": score, "preroll_ms": preroll_ms})
        log.info("wake sent: word=%s score=%.2f preroll=%d ms", word, score, preroll_ms)
        for frame in frames:  # the pre-roll goes out as a burst, the live audio is paced by the sender
            await self._send_bytes(frame)
        self._disarm_wake_switch(True)  # from here on new frames stream (paced)

    def _disarm_wake_switch(self, live: bool) -> None:
        """Route mic frames: False = keep filling the pre-roll ring, True = stream to the server."""
        self._upload_live = live

    async def push_to_talk(self) -> None:
        """The PTT hotkey: a wake that the server never verifies (caps.ptt)."""
        await self.send_wake("push_to_talk", 1.0, preroll=False)

    def _arm_wake(self) -> None:
        self._upload_live = False
        if self._wake is not None:
            try:
                self._wake.reset()
            except Exception:  # noqa: BLE001
                pass

    def _disarm_wake(self) -> None:
        self._upload_live = True

    async def _mic_sender(self, ws: Any) -> None:
        """Stream queued live frames at real-time pace (the pre-roll was already bursted)."""
        next_slot = time.monotonic()
        while True:
            if not self._mic_queue:
                if self._frames_ready is None:
                    await asyncio.sleep(0.005)
                    continue
                self._frames_ready.clear()
                await self._frames_ready.wait()
                continue
            now = time.monotonic()
            if now < next_slot:
                await asyncio.sleep(next_slot - now)
            elif now - next_slot > 1.0:
                next_slot = now  # we fell far behind (laptop sleep): drop the backlog instead of bursting
            frame = self._mic_queue.popleft()
            await self._send_bytes(frame)
            next_slot = max(next_slot + FRAME_MS / 1000, time.monotonic())

    async def _receive(self, ws: Any) -> None:
        async for message in ws:
            if isinstance(message, bytes):
                self._on_audio(message)
            else:
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    log.warning("non-JSON text frame ignored")
                    continue
                self._on_event(event)

    def _on_audio(self, data: bytes) -> None:
        if not data:
            return
        self._speaking = True
        self._last_audio_t = time.monotonic()
        if self.state in ("listening", "follow_up"):
            self._set_state("speaking")  # audio while the mic was open: the server started talking
        if self._playback is not None:
            self._playback.play(data)

    def _on_event(self, evt: dict[str, Any]) -> None:
        kind = evt.get("type")
        if kind == "wake_accepted":
            log.info("wake accepted")
        elif kind == "wake_rejected":
            log.info("wake rejected: %s", evt.get("reason"))
            self._stop_streaming("idle")
        elif kind == "listen_end":
            log.debug("listen_end")
            self._stop_streaming("idle")
        elif kind == "follow_up":
            ms = int(evt.get("ms") or 0)
            log.debug("follow_up %d ms", ms)
            self.follow_up_ms = ms
            if self.state == "speaking":
                self._follow_until = 0.0  # opens when the playback drains (see _playback_monitor)
            else:
                self._open_follow_up(ms)
        elif kind == "audio_start":
            pass  # binary frames drive the half-duplex gating; nothing to do
        elif kind == "audio_end":
            self._last_audio_t = time.monotonic()
        elif kind == "playback_stop":
            log.info("playback cancelled by the server (barge-in)")
            self._cancel_playback()
            if self.state == "speaking":
                self._set_state("idle")
                self._arm_wake()
        elif kind == "ping":
            asyncio.create_task(self._send_json({"type": "pong"}))
        elif kind == "error":
            message = str(evt.get("message") or evt)
            log.warning("server error: %s", message)
            if message == "unauthorized":
                self._unauthorized = True
                self._stopping.set()  # a bad token will not fix itself
                self._close_ws()
        elif kind == "pc_call":
            self._spawn_pc_call(evt)
        elif kind in ("state", "transcript", "utterance", "metrics", "vad", "notice", "timer",
                      "record_start", "record_stop"):
            pass  # ignored: this client runs no recordings and shows no metrics
        else:
            log.debug("unhandled message type %r", kind)

    def _stop_streaming(self, then: str) -> None:
        """Close the mic window (wake_rejected / listen_end) and re-arm the wake word."""
        self._upload_live = False
        self._mic_queue.clear()
        self._follow_until = 0.0
        self._set_state(then)
        if then == "idle":
            self._arm_wake()

    def _open_follow_up(self, ms: int) -> None:
        """Mic open for `ms` without a wake (the reply just ended)."""
        self._upload_live = True
        self._mic_queue.clear()
        self._follow_until = time.monotonic() + ms / 1000
        self._set_state("follow_up")

    async def _playback_monitor(self) -> None:
        """Half-duplex bookkeeping: when the reply's audio has drained, open the follow-up window."""
        while True:
            await asyncio.sleep(0.05)
            if self._speaking and self._playback is not None:
                drained = not self._playback.busy() and time.monotonic() - self._last_audio_t > 0.2
                if drained:
                    self._speaking = False
                    if self.state == "speaking":
                        if self.follow_up_ms > 0:
                            self._open_follow_up(self.follow_up_ms)
                        else:
                            self._stop_streaming("idle")
            if self.state == "follow_up" and self._follow_until and time.monotonic() > self._follow_until:
                # The server usually sends listen_end first; this is the local safety net.
                self._stop_streaming("idle")

    def _cancel_playback(self) -> None:
        """Drop everything queued (barge-in / playback_stop): silence must stop at once."""
        if self._playback is not None and hasattr(self._playback, "cancel"):
            self._playback.cancel()
        elif self._playback is not None:
            self._playback.close()
            self._playback = None
            self._set_playback_rate(self.out_rate)
        self._speaking = False

    # ---- send helpers ----------------------------------------------------------------------

    async def _send_json(self, payload: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None:
            return
        async with self._send_lock:
            await ws.send(json.dumps(payload))

    async def _send_bytes(self, data: bytes) -> None:
        ws = self._ws
        if ws is None:
            return
        async with self._send_lock:
            await ws.send(data)

    # ---- pc tools --------------------------------------------------------------------------

    def _spawn_pc_call(self, evt: dict[str, Any]) -> None:
        if not self.saw_pc:
            log.debug("pc_call ignored: the server did not accept caps.pc")
            return
        call_id = str(evt.get("id") or "")
        if not call_id or not set(call_id) <= PC_ID_CHARS or len(call_id) > 32:
            log.warning("pc_call with a malformed id ignored: %r", call_id)
            return
        if call_id in self._pc_calls:
            log.debug("duplicate pc_call %s ignored", call_id)
            return
        self._pc_calls[call_id] = False
        task = asyncio.create_task(self._run_pc_call(evt))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_pc_call(self, evt: dict[str, Any]) -> None:
        call_id = str(evt.get("id") or "")
        tool_name = evt.get("tool")
        args = evt.get("args")
        timeout = evt.get("timeout_s", DEFAULT_TIMEOUT_S)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= MAX_TIMEOUT_S:
            timeout = min(max(float(timeout) if isinstance(timeout, (int, float)) else DEFAULT_TIMEOUT_S,
                              1.0), MAX_TIMEOUT_S)
        if not isinstance(args, dict):
            args = {}
        tool = self.tools.get(tool_name) if isinstance(tool_name, str) else None
        if tool is None:
            result = ToolResult.bad_request(
                f"unknown tool {tool_name!r} (advertised: {', '.join(self.tools)})")
        else:
            try:
                # Worker thread: a slow tool must never stall the audio loop or other calls,
                # and several calls may be in flight at once (docs/DESKTOP.md).
                result = await asyncio.wait_for(asyncio.to_thread(tool.run, args, timeout),
                                                timeout + EXECUTOR_GRACE_S)
            except TimeoutError:
                result = ToolResult(ok=False, error="timeout")
            except Exception as exc:  # noqa: BLE001 - a crashing tool still answers, once
                log.exception("tool %s crashed", tool_name)
                result = ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        self._pc_calls[call_id] = True
        await self._send_pc_result(call_id, result)

    async def _send_pc_result(self, call_id: str, result: ToolResult) -> None:
        frame: dict[str, Any] = {"type": "pc_result", "id": call_id, "ok": result.ok}
        text = truncate_middle(result.text or "", TEXT_LIMIT)
        if text:
            frame["text"] = text
        if result.data:
            frame["data"] = result.data
        if result.error:
            frame["error"] = result.error
        if result.image:
            frame["image"] = result.image
        try:
            await self._send_json(frame)
            log.info("pc_result sent: id=%s ok=%s tool_text=%d chars", call_id, result.ok, len(text))
        except (OSError, websockets.WebSocketException) as exc:
            log.warning("pc_result %s could not be sent: %s", call_id, exc)

    # ---- tray toggles -----------------------------------------------------------------------

    def set_muted(self, muted: bool) -> None:
        self.muted = muted
        log.info("listening %s", "muted" if muted else "unmuted")

    def set_pc_enabled(self, enabled: bool) -> None:
        if enabled == self.pc_enabled:
            return
        self.pc_enabled = enabled
        self.saw_pc = False
        self.request_reconnect()  # the hello must change, so re-open the session
