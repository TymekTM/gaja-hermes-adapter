"""The device-only listener: WS /device with per-device token auth, and no web UI.

A second uvicorn server inside the adapter process (same event loop, same Services), started from the main
app's lifespan only when ``[devices] enabled``. It is a separate FastAPI app that serves nothing but ``/device``.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.status import WS_1008_POLICY_VIOLATION, WS_1013_TRY_AGAIN_LATER

from gaja.config import DevicesConfig
from gaja.devices.registry import DeviceRegistry
from gaja.pipeline import Services, Session

log = logging.getLogger(__name__)

AUTH_FAIL_DELAY_S = 1.0


def peer_allowed(host: str | None, networks: list[str]) -> bool:
    if host is None:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return any(address in ipaddress.ip_network(n, strict=False) for n in networks if _same_family(address, n))


def _same_family(address: ipaddress.IPv4Address | ipaddress.IPv6Address, network: str) -> bool:
    return (":" in network) == isinstance(address, ipaddress.IPv6Address)


class _Outbox:
    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._lock = asyncio.Lock()

    async def send_json(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            await self._websocket.send_json(payload)

    async def send_bytes(self, data: bytes) -> None:
        async with self._lock:
            await self._websocket.send_bytes(data)


def create_device_app(services: Services, registry: DeviceRegistry) -> FastAPI:
    cfg: DevicesConfig = services.cfg.devices
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.services = services
    app.state.registry = registry
    active: dict[str, WebSocket] = {}
    app.state.active = active

    @app.websocket("/device")
    async def device(websocket: WebSocket) -> None:
        peer = websocket.client.host if websocket.client else None
        if not peer_allowed(peer, cfg.allowed_networks):
            log.warning("device connection from %s refused (network not allowed)", peer)
            await websocket.close(code=WS_1008_POLICY_VIOLATION)
            return
        await websocket.accept()
        hello = await _read_hello(websocket, cfg.hello_timeout_s)
        info = registry.authenticate(hello.get("device_id"), hello.get("token")) if hello is not None else None
        if hello is None or info is None:
            # Same answer for an unknown device and a wrong token; a delay slows brute-force attempts.
            log.warning("device auth failed from %s (id=%.64r)", peer, (hello or {}).get("device_id"))
            await asyncio.sleep(AUTH_FAIL_DELAY_S)
            with contextlib.suppress(Exception):
                await websocket.send_json({"type": "error", "message": "unauthorized"})
                await websocket.close(code=WS_1008_POLICY_VIOLATION)
            return
        hello.pop("token", None)
        previous = active.pop(info.id, None)
        if previous is not None:
            # The device reconnected (a half-open socket is still around): the new connection wins.
            with contextlib.suppress(Exception):
                await previous.close(code=1000)
        if len(active) >= cfg.max_connections:
            await websocket.send_json({"type": "error", "message": "too many devices"})
            await websocket.close(code=WS_1013_TRY_AGAIN_LATER)
            return
        active[info.id] = websocket
        log.info("device %s connected from %s (fw=%.32r)", info.id, peer, hello.get("fw"))
        session = Session(services, _Outbox(websocket), device=info)
        try:
            await session.handle_text(json.dumps(hello))
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await session.handle_audio(message["bytes"])
                elif message.get("text") is not None:
                    await session.handle_text(message["text"])
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if active.get(info.id) is websocket:
                del active[info.id]
            await session.close()
            log.info("device %s disconnected", info.id)

    return app


async def _read_hello(websocket: WebSocket, timeout_s: float) -> dict[str, Any] | None:
    try:
        message = await asyncio.wait_for(websocket.receive(), timeout_s)
    except (TimeoutError, WebSocketDisconnect, RuntimeError):
        return None
    text = message.get("text")
    if text is None or len(text) > 4096:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and data.get("type") == "hello" else None


class _EmbeddedServer(uvicorn.Server):
    """uvicorn server that leaves signal handling to the main server."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


class DeviceListener:
    def __init__(self, services: Services, registry: DeviceRegistry | None = None, *, port: int | None = None) -> None:
        cfg = services.cfg.devices
        self.registry = registry or DeviceRegistry(cfg.tokens_file)
        self.app = create_device_app(services, self.registry)
        ssl: dict[str, Any] = {}
        if cfg.tls_certfile and cfg.tls_keyfile:
            ssl = {"ssl_certfile": str(Path(cfg.tls_certfile).expanduser()), "ssl_keyfile": str(Path(cfg.tls_keyfile).expanduser())}
        config = uvicorn.Config(
            self.app,
            host=cfg.listen_host,
            port=cfg.port if port is None else port,
            log_level="info",
            lifespan="off",
            access_log=False,
            server_header=False,
            ws_max_size=1 << 20,
            **ssl,
        )
        self._server = _EmbeddedServer(config)
        self._task: asyncio.Task[None] | None = None

    @property
    def port(self) -> int | None:
        servers = getattr(self._server, "servers", None)
        if not servers or not servers[0].sockets:
            return None
        return servers[0].sockets[0].getsockname()[1]

    @property
    def connected(self) -> list[str]:
        return sorted(self.app.state.active)

    async def _run(self) -> None:
        try:
            await self._server.serve()
        except SystemExit:  # uvicorn exits when the port cannot be bound; never take the adapter down with it
            log.error("device listener could not start (port in use?)")

    async def start(self) -> bool:
        self._task = asyncio.create_task(self._run())
        for _ in range(200):
            if self._server.started:
                self.registry.count()  # logs how many devices are configured
                log.info("device listener on %s:%s (/device only)", self._server.config.host, self.port)
                return True
            if self._task.done():
                break
            await asyncio.sleep(0.05)
        log.error("device listener did not start; the adapter continues without it")
        await self.stop()
        return False

    async def stop(self) -> None:
        if self._task is None:
            return
        self._server.should_exit = True
        try:
            await asyncio.wait_for(self._task, 10)
        except (TimeoutError, asyncio.CancelledError):
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
