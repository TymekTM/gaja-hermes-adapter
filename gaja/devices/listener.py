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
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
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
# A send that does not complete in this long means the peer vanished (Wi-Fi gone, socket still "open"):
# the connection is dropped instead of wedging the session and every server -> device announcement.
SEND_TIMEOUT_S = 5.0
# Bound on a superseded connection's polite close frame; the old handler is cancelled regardless.
SUPERSEDE_CLOSE_S = 2.0


NetworkType = ipaddress.IPv4Network | ipaddress.IPv6Network


def parse_networks(networks: list[str] | list[NetworkType]) -> list[NetworkType]:
    parsed: list[NetworkType] = []
    for n in networks:
        if isinstance(n, (ipaddress.IPv4Network, ipaddress.IPv6Network)):
            parsed.append(n)
        else:
            try:
                parsed.append(ipaddress.ip_network(n, strict=False))
            except ValueError:
                log.warning("invalid network in allowed_networks: %s", n)
    return parsed


def peer_allowed(host: str | None, networks: list[str] | list[NetworkType]) -> bool:
    if host is None:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    for n in networks:
        net = n if isinstance(n, (ipaddress.IPv4Network, ipaddress.IPv6Network)) else ipaddress.ip_network(n, strict=False)
        if address.version == net.version and address in net:
            return True
    return False


def _same_family(address: ipaddress.IPv4Address | ipaddress.IPv6Address, network: str | NetworkType) -> bool:
    if isinstance(network, (ipaddress.IPv4Network, ipaddress.IPv6Network)):
        return address.version == network.version
    return (":" in network) == isinstance(address, ipaddress.IPv6Address)


class _Outbox:
    """Serialised, time-bounded sends to one device.

    A send that fails or times out marks the outbox dead, raises ``ConnectionError`` (every later send fails
    at once, so nothing queues behind a stuck socket) and calls ``on_dead`` once so the connection handler
    ends and its session is torn down.
    """

    def __init__(self, websocket: WebSocket, *, timeout_s: float | None = None, on_dead: Callable[[], None] | None = None) -> None:
        self._websocket = websocket
        self._lock = asyncio.Lock()
        self._timeout_s = timeout_s
        self._on_dead = on_dead
        self.dead = False

    def _kill(self, reason: str) -> None:
        if self.dead:
            return
        self.dead = True
        log.warning("device send failed (%s); dropping the connection", reason)
        if self._on_dead is not None:
            self._on_dead()

    async def _send(self, send: Callable[[], Awaitable[None]]) -> None:
        async with self._lock:
            if self.dead:
                raise ConnectionError("device connection is closed")
            try:
                await asyncio.wait_for(send(), SEND_TIMEOUT_S if self._timeout_s is None else self._timeout_s)
            except TimeoutError:
                self._kill("timeout")
                raise ConnectionError("device send timed out") from None
            except (WebSocketDisconnect, RuntimeError, OSError) as exc:
                self._kill(type(exc).__name__)
                raise ConnectionError("device connection is closed") from exc

    async def send_json(self, payload: dict[str, Any]) -> None:
        await self._send(lambda: self._websocket.send_json(payload))

    async def send_bytes(self, data: bytes) -> None:
        await self._send(lambda: self._websocket.send_bytes(data))


@dataclass
class _Conn:
    websocket: WebSocket
    task: asyncio.Task[Any]
    superseded: bool = False
    closing: bool = False  # the handler is already unwinding: never cancel it again


def _abort(task: asyncio.Task[Any]) -> None:
    """End a connection handler from outside; a handler never cancels itself (it unwinds by exception)."""
    if task is not asyncio.current_task() and not task.done():
        task.cancel()


def create_device_app(services: Services, registry: DeviceRegistry) -> FastAPI:
    cfg: DevicesConfig = services.cfg.devices
    allowed_networks = parse_networks(cfg.allowed_networks)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.services = services
    app.state.registry = registry
    app.state.allowed_networks = allowed_networks
    active: dict[str, WebSocket] = {}
    app.state.active = active
    conns: dict[str, _Conn] = {}
    pending = 0  # connections accepted but not yet authenticated

    async def supersede(previous: _Conn) -> None:
        """The device reconnected: end its old (possibly half-open) connection without waiting for it."""
        previous.superseded = True
        _abort(previous.task)  # the old handler unwinds, closes its session and returns

    @app.websocket("/device")
    async def device(websocket: WebSocket) -> None:
        nonlocal pending
        peer = websocket.client.host if websocket.client else None
        if not peer_allowed(peer, allowed_networks):
            log.warning("device connection from %s refused (network not allowed)", peer)
            await websocket.close(code=WS_1008_POLICY_VIOLATION)
            return
        if pending >= cfg.max_connections:
            # Unauthenticated sockets are bounded: a flood of silent connections cannot starve the handler.
            log.warning("device connection from %s refused (too many pending handshakes)", peer)
            await websocket.close(code=WS_1013_TRY_AGAIN_LATER)
            return
        pending += 1
        try:
            await websocket.accept()
            hello = await _read_hello(websocket, cfg.hello_timeout_s)
            info = registry.authenticate(hello.get("device_id"), hello.get("token")) if hello is not None else None
        finally:
            pending -= 1
        if hello is None or info is None:
            # Same answer for an unknown device and a wrong token; a delay slows brute-force attempts.
            log.warning("device auth failed from %s (id=%.64r)", peer, (hello or {}).get("device_id"))
            await asyncio.sleep(AUTH_FAIL_DELAY_S)
            with contextlib.suppress(Exception):
                await websocket.send_json({"type": "error", "message": "unauthorized"})
                await websocket.close(code=WS_1008_POLICY_VIOLATION)
            return
        hello.pop("token", None)
        previous = conns.pop(info.id, None)
        active.pop(info.id, None)
        if previous is not None:
            # The device reconnected (a half-open socket is still around): the new connection wins.
            await supersede(previous)
        if len(active) >= cfg.max_connections:
            await websocket.send_json({"type": "error", "message": "too many devices"})
            await websocket.close(code=WS_1013_TRY_AGAIN_LATER)
            return
        task = asyncio.current_task()
        assert task is not None
        conn = conns[info.id] = _Conn(websocket, task)
        active[info.id] = websocket
        outbox = _Outbox(websocket, on_dead=lambda: None if conn.closing else _abort(task))
        log.info("device %s connected from %s (fw=%.32r)", info.id, peer, hello.get("fw"))
        session = Session(services, outbox, device=info)
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
                if conn.superseded or outbox.dead:
                    break  # a wait_for inside a handler can swallow the cancellation that ended this connection
        except asyncio.CancelledError:
            if not (outbox.dead or conn.superseded):
                raise  # server shutdown: let it propagate
            task.uncancel()
        except (WebSocketDisconnect, RuntimeError, ConnectionError):
            pass
        finally:
            conn.closing = True
            if active.get(info.id) is websocket:
                del active[info.id]
            if conns.get(info.id) is conn:
                del conns[info.id]
            await session.close()
            log.info("device %s disconnected", info.id)
            if outbox.dead or conn.superseded:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await asyncio.wait_for(websocket.close(code=1000), SUPERSEDE_CLOSE_S)

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
