"""An in-process fake of the adapter's `/device` listener (protocol v1.14) for client tests.

It records everything the client sends (hello, wake, binary frames, pc_result, …) and lets the test
drive the server side: send events, send audio, close connections. Everything runs on ephemeral
ports; no real adapter is ever contacted.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable

import websockets


class FakeDeviceServer:
    def __init__(self, *, ready_pc: bool = True, rate: int = 48000, follow_up_ms: int = 6000,
                 verify_wake: bool = True, on_hello: Callable[[Any, dict[str, Any]], Any] | None = None) -> None:
        self.ready_pc = ready_pc
        self.rate = rate
        self.follow_up_ms = follow_up_ms
        self.verify_wake = verify_wake
        self.on_hello = on_hello  # custom greeting (e.g. unauthorized)
        self.port = 0
        self.url = ""
        self.hellos: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []  # every JSON message the client sent (incl. hello/wake)
        self.results: list[dict[str, Any]] = []  # pc_result messages
        self.binary: list[bytes] = []  # every binary frame in order
        self.clients: list[Any] = []
        self._server: Any = None
        self._closed = asyncio.Event()

    async def start(self) -> None:
        self._server = await websockets.serve(self._handler, "127.0.0.1", 0)
        self.port = int(self._server.sockets[0].getsockname()[1])
        self.url = f"ws://127.0.0.1:{self.port}/device"

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        for ws in list(self.clients):
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    # ---- handler ---------------------------------------------------------------------------

    async def _handler(self, ws: Any) -> None:
        self.clients.append(ws)
        try:
            first = await asyncio.wait_for(ws.recv(), 5)
            hello = json.loads(first)
        except Exception:  # noqa: BLE001
            return
        self.hellos.append(hello)
        self.events.append(hello)
        reader = asyncio.create_task(self._read(ws))
        try:
            if self.on_hello is not None:
                await self.on_hello(ws, hello)
            else:
                await ws.send(json.dumps(self.ready_payload(hello)))
            await reader
        except Exception:  # noqa: BLE001 - abrupt test closes end the handler
            pass
        finally:
            reader.cancel()

    async def _read(self, ws: Any) -> None:
        async for message in ws:
            if isinstance(message, bytes):
                self.binary.append(message)
                continue
            evt = json.loads(message)
            self.events.append(evt)
            if evt.get("type") == "pc_result":
                self.results.append(evt)

    def ready_payload(self, hello: dict[str, Any]) -> dict[str, Any]:
        caps = hello.get("caps") or {}
        device: dict[str, Any] = {
            "id": hello.get("device_id"), "aec": caps.get("aec", False), "out_rate": self.rate,
            "verify_wake": self.verify_wake, "verify_ms": 2500, "follow_up_ms": self.follow_up_ms,
            "ack": "spoken", "logs": True,
        }
        if self.ready_pc:
            device["pc"] = True
        return {"type": "ready", "session_id": "fake", "tts": "fake", "sample_rate": self.rate,
                "device": device}

    # ---- test-side helpers -----------------------------------------------------------------

    @property
    def ws(self) -> Any:
        return self.clients[-1] if self.clients else None

    async def send(self, obj: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(obj))

    async def send_raw(self, data: bytes) -> None:
        await self.ws.send(data)

    async def send_audio(self, seconds: float, rate: int | None = None, chunk_ms: int = 100) -> None:
        """Server -> device binary: silence chunks, like the outbox would send."""
        rate = rate or self.rate
        chunk = bytes(int(rate * chunk_ms / 1000) * 2)
        for _ in range(max(1, round(seconds * 1000 / chunk_ms))):
            await self.send_raw(chunk)

    async def wait_for(self, pred: Callable[[Any], bool], timeout: float = 10.0,
                       what: str = "condition") -> Any:
        """Poll the recorded traffic until pred() holds; returns pred's truthy value."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = pred()
            if value:
                return value
            await asyncio.sleep(0.01)
        raise TimeoutError(f"fake server: {what} not met after {timeout}s")

    def event(self, kind: str) -> dict[str, Any] | None:
        return next((e for e in self.events if e.get("type") == kind), None)

    async def wait_event(self, kind: str, timeout: float = 10.0) -> dict[str, Any]:
        return await self.wait_for(lambda: self.event(kind), timeout, f"event {kind}")

    async def wait_frames(self, count: int, timeout: float = 10.0) -> int:
        return await self.wait_for(lambda: len(self.binary) if len(self.binary) >= count else 0,
                                   timeout, f">= {count} binary frames")

    async def close_connections(self) -> None:
        """Abruptly drop every client connection (reconnect test)."""
        for ws in list(self.clients):
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass
