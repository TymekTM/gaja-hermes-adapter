"""Stdio MCP server exposing Gaja voice-device speaker tools to Hermes.

Runnable as::

    python -m gaja.mcp_speaker

Talks to the local Gaja adapter HTTP API on loopback (default ``http://127.0.0.1:8790``,
configurable via ``GAJA_API_URL``) with no ``Origin`` header.
"""

from __future__ import annotations

import os
import re
from typing import Annotated, Any

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from gaja.api.devices import list_sounds as _list_allowlisted_sounds
from gaja.devices.quiet_hours import format_quiet_hours_error

DEFAULT_API_URL = "http://127.0.0.1:8790"
# /say and /play answer once the audio is sent (TTS + transfer, not playback). The device buffers ~44 s and
# pushes back when full, so 500 characters (~40 s of speech) can take most of that; 90 s leaves room.
DEFAULT_TIMEOUT_S = 90.0
_DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

mcp = FastMCP("gaja_speaker")


def get_api_url() -> str:
    """Return the configured Gaja adapter base URL."""
    return (os.environ.get("GAJA_API_URL") or DEFAULT_API_URL).rstrip("/")


async def _request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    base_url: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    url_base = (base_url or get_api_url()).rstrip("/")
    try:
        if client is not None:
            resp = await client.request(method, path, json=json_body)
        else:
            async with httpx.AsyncClient(base_url=url_base, timeout=DEFAULT_TIMEOUT_S) as http:
                resp = await http.request(method, path, json=json_body)
    except httpx.HTTPError as exc:
        return {"ok": False, "error": f"adapter request failed: {exc}"}

    try:
        payload = resp.json()
    except Exception:
        payload = {"detail": resp.text}

    if resp.status_code >= 400:
        detail = payload.get("detail") if isinstance(payload, dict) else str(payload)
        return {"ok": False, "status": resp.status_code, "error": str(detail or f"HTTP {resp.status_code}")}

    if isinstance(payload, dict):
        return payload
    return {"ok": True, "data": payload}


@mcp.tool()
async def list_speakers() -> dict[str, Any]:
    """List configured voice-device speakers (e.g. 'atom-pokoj'), whether each is online, and quiet hours status."""
    return await _request("GET", "/api/devices")


@mcp.tool()
async def list_sounds() -> dict[str, Any]:
    """List allow-listed short sound names that can be played with speaker_play_sound."""
    return {"sounds": _list_allowlisted_sounds()}


@mcp.tool()
async def speaker_say(
    device_id: str,
    text: str,
    force: Annotated[
        bool,
        Field(description="Bypass quiet hours if the user explicitly asked for playback right now."),
    ] = False,
) -> dict[str, Any]:
    """Speak a short announcement via TTS on a connected voice-device speaker (1-500 chars)."""
    dev = (device_id or "").strip()
    if not _DEVICE_ID_RE.match(dev):
        return {"ok": False, "error": "invalid device_id"}
    cleaned = (text or "").strip()
    if not (1 <= len(cleaned) <= 500):
        return {"ok": False, "error": "text must be between 1 and 500 characters"}
    payload: dict[str, Any] = {"text": cleaned}
    if force:
        payload["force"] = True
    res = await _request("POST", f"/api/devices/{dev}/say", json_body=payload)
    if res.get("ok") is True:
        return {"ok": True, "device_id": dev, "text": cleaned}
    if res.get("status") == 409:
        res["error"] = format_quiet_hours_error(res.get("error", ""))
    return res


@mcp.tool()
async def speaker_play_sound(
    device_id: str,
    sound: str,
    force: Annotated[
        bool,
        Field(description="Bypass quiet hours if the user explicitly asked for playback right now."),
    ] = False,
) -> dict[str, Any]:
    """Play an allow-listed sound file (see list_sounds, e.g. 'chime', 'alert', 'success') on a speaker."""
    dev = (device_id or "").strip()
    if not _DEVICE_ID_RE.match(dev):
        return {"ok": False, "error": "invalid device_id"}
    snd = (sound or "").strip()
    if not snd:
        return {"ok": False, "error": "sound must not be empty"}
    payload: dict[str, Any] = {"sound": snd}
    if force:
        payload["force"] = True
    res = await _request("POST", f"/api/devices/{dev}/play", json_body=payload)
    if res.get("ok") is True:
        return {"ok": True, "device_id": dev, "sound": snd}
    if res.get("status") == 409:
        res["error"] = format_quiet_hours_error(res.get("error", ""))
    return res


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
