"""PC tool relay (device protocol v1.14, docs/DESKTOP.md): ``pc_call`` / ``pc_result``.

A device whose hello carries ``caps.pc`` executes tools on its PC for the agent. The relay keeps
one entry per connected PC device (id, host, os, advertised tools) and a map of pending call
futures keyed by the 12-hex call id. ``DeviceController`` registers/unregisters it and routes
``pc_result`` messages here; the HTTP API (``gaja/api/pc.py``) and the MCP server
(``gaja.mcp_pc``) call ``PcRelay.call``.

Everything on the device->server path is synchronous and O(1): a ``pc_result`` is one dict lookup
and one ``set_result`` — it can never block or slow the audio path. Sends go through the device
outbox (serialized with the audio frames, bounded by its send timeout).
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from gaja.devices.audio import PcCaps

log = logging.getLogger(__name__)

# The server waits timeout_s + this grace before failing a call itself: the client enforces its
# own timeout_s and answers "timeout", so the server-side one only fires when that answer is lost.
GRACE_S = 5.0
# Protocol: text is at most 65536 characters (the client truncates the middle); clamp defensively.
MAX_TEXT_CHARS = 65536
MAX_ERROR_CHARS = 512
ID_RE = re.compile(r"^[0-9a-f]{12}$")

SendFn = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class PcResult:
    """One answered ``pc_result`` (or a relay-generated failure)."""

    ok: bool = False
    text: str = ""
    data: dict[str, Any] | None = None
    error: str = ""
    image: dict[str, Any] | None = None

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": self.ok, "text": self.text, "data": self.data, "error": self.error}
        if self.image is not None:
            out["image"] = self.image
        return out


def _clamp(value: Any, limit: int) -> str:
    return value[:limit] if isinstance(value, str) else ""


@dataclass
class _Entry:
    device_id: str
    host: str
    os: str
    tools: tuple[str, ...]
    send: SendFn
    owner: Any  # the DeviceController that registered; a late close of a superseded one is ignored
    connected_at: float = field(default_factory=time.monotonic)
    pending: dict[str, asyncio.Future[PcResult]] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.device_id,
            "host": self.host,
            "os": self.os,
            "tools": list(self.tools),
            "connected_s": round(time.monotonic() - self.connected_at, 1),
        }

    def fail_all(self, error: str) -> None:
        for call_id, fut in list(self.pending.items()):
            if not fut.done():
                fut.set_result(PcResult(ok=False, error=error))
        self.pending.clear()


class PcRelay:
    """Registry of connected PC devices and their pending tool calls (one per Services)."""

    def __init__(self) -> None:
        self._devices: dict[str, _Entry] = {}

    # ---- device lifecycle (called by DeviceController) --------------------------------------

    def register(self, owner: Any, device_id: str, caps: PcCaps, send: SendFn) -> None:
        """A hello with accepted ``caps.pc``: this connection becomes the device's PC entry.

        Replaces any previous entry of the same device id (a reconnect can win the race against
        the old connection's close): the replaced entry's pending calls fail at once — their
        ``pc_call`` went over a socket that is no longer the live one.
        """
        old = self._devices.get(device_id)
        if old is not None and old is not owner:
            old.fail_all("disconnected")
        self._devices[device_id] = _Entry(
            device_id=device_id,
            host=caps.host,
            os=caps.os,
            tools=tuple(caps.tools),
            send=send,
            owner=owner,
        )
        log.info("pc device %s connected (os=%s host=%s tools=%s)", device_id, caps.os, caps.host, caps.tools)

    def unregister(self, owner: Any, device_id: str) -> None:
        """The connection that registered this device went away: fail its pending calls."""
        entry = self._devices.get(device_id)
        if entry is None or entry.owner is not owner:
            return  # a newer connection already replaced the entry; leave it alone
        del self._devices[device_id]
        pending = len(entry.pending)
        entry.fail_all("disconnected")
        if pending:
            log.info("pc device %s disconnected: %d pending call(s) failed", device_id, pending)
        else:
            log.info("pc device %s disconnected", device_id)

    # ---- message routing (from the device) --------------------------------------------------

    def on_result(self, device_id: str, msg: dict[str, Any]) -> None:
        """Route one ``pc_result`` to its pending call. Unknown/finished/timed-out ids are ignored."""
        entry = self._devices.get(device_id)
        if entry is None:
            log.debug("pc_result from %s with no relay entry; ignored", device_id)
            return
        call_id = msg.get("id")
        if not isinstance(call_id, str) or not ID_RE.match(call_id):
            log.debug("pc_result from %s with malformed id %r; ignored", device_id, call_id)
            return
        fut = entry.pending.pop(call_id, None)
        if fut is None or fut.done():
            log.debug("pc_result from %s for unknown/finished call %s; ignored", device_id, call_id)
            return
        fut.set_result(
            PcResult(
                ok=msg.get("ok") is True,
                text=_clamp(msg.get("text"), MAX_TEXT_CHARS),
                data=msg.get("data") if isinstance(msg.get("data"), dict) else None,
                error=_clamp(msg.get("error"), MAX_ERROR_CHARS),
                image=msg.get("image") if isinstance(msg.get("image"), dict) else None,
            )
        )

    # ---- calls (from the HTTP API / MCP) ------------------------------------------------------

    def devices(self) -> list[dict[str, Any]]:
        return [entry.describe() for _, entry in sorted(self._devices.items())]

    def get(self, device_id: str) -> _Entry | None:
        return self._devices.get(device_id)

    def has_tool(self, device_id: str, tool: str) -> bool:
        entry = self._devices.get(device_id)
        return entry is not None and tool in entry.tools

    async def call(self, device_id: str, tool: str, args: dict[str, Any], timeout_s: float) -> PcResult:
        """Send one ``pc_call`` and await its ``pc_result`` (timeout_s + GRACE_S server-side).

        Never raises for device-level outcomes: a missing/vanished device answers
        ``error:"disconnected"``, a lost answer ``error:"timeout"``.
        """
        entry = self._devices.get(device_id)
        if entry is None:
            return PcResult(ok=False, error="disconnected")
        call_id = secrets.token_hex(6)
        fut: asyncio.Future[PcResult] = asyncio.get_running_loop().create_future()
        entry.pending[call_id] = fut
        try:
            await entry.send({"type": "pc_call", "id": call_id, "tool": tool, "args": args, "timeout_s": timeout_s})
        except asyncio.CancelledError:
            entry.pending.pop(call_id, None)
            raise
        except Exception as exc:  # the link died under the send: same as a disconnect
            entry.pending.pop(call_id, None)
            log.info("pc_call %s to %s could not be sent (%s)", call_id, device_id, type(exc).__name__)
            return PcResult(ok=False, error="disconnected")
        try:
            return await asyncio.wait_for(fut, timeout_s + GRACE_S)
        except asyncio.TimeoutError:
            entry.pending.pop(call_id, None)
            return PcResult(ok=False, error="timeout")
        except asyncio.CancelledError:
            entry.pending.pop(call_id, None)
            raise
