"""Stdio MCP server exposing Gaja voice-device speaker tools to Hermes.

Runnable as::

    python -m gaja.mcp_speaker

Talks to the local Gaja adapter HTTP API on loopback (default ``http://127.0.0.1:8790``,
configurable via ``GAJA_API_URL``) with no ``Origin`` header.
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path
from typing import Annotated, Any

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from gaja.api.devices import list_sounds as _list_allowlisted_sounds
from gaja.config import Settings, resolve
from gaja.devices.quiet_hours import format_quiet_hours_error

DEFAULT_API_URL = "http://127.0.0.1:8790"
# /say and /play answer once the audio is sent (TTS + transfer, not playback). The device buffers only
# a few dozen seconds and pushes back, so the downlink is paced: 500 characters (~40 s of speech) took
# most of the original 90 s budget. Long morning briefings (up to 4000 chars, ~5 min of speech) get a
# proportionally larger budget via say_timeout_s.
DEFAULT_TIMEOUT_S = 90.0
# Fixed per-request overhead plus ~2x the speech estimate of 12.5 chars/s (500 chars ≈ 40 s of speech).
SAY_TIMEOUT_BASE_S = 30.0
SAY_TIMEOUT_PER_CHAR_S = 0.16
# /play is paced at playback speed too: the longest user sound (gaja.api.devices.MAX_SOUND_SECONDS, 300 s)
# plus decode and transfer slack.
PLAY_TIMEOUT_S = 360.0
_DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

mcp = FastMCP("gaja_speaker")


def get_api_url() -> str:
    """Return the configured Gaja adapter base URL."""
    return (os.environ.get("GAJA_API_URL") or DEFAULT_API_URL).rstrip("/")


@functools.cache
def _user_sounds_dir() -> Path:
    """The user sounds dir configured under [devices] sounds_dir, resolved once per process."""
    try:
        return resolve(Settings().devices.sounds_dir)
    except Exception:
        return resolve(Path("data/sounds"))


def say_timeout_s(text: str) -> float:
    """HTTP budget for one /say call: the default for short texts, scaling for long briefings."""
    return max(DEFAULT_TIMEOUT_S, SAY_TIMEOUT_BASE_S + SAY_TIMEOUT_PER_CHAR_S * len(text))


async def _request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    base_url: str | None = None,
    client: httpx.AsyncClient | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    url_base = (base_url or get_api_url()).rstrip("/")
    try:
        if client is not None:
            resp = await client.request(method, path, json=json_body)
        else:
            async with httpx.AsyncClient(base_url=url_base, timeout=timeout_s or DEFAULT_TIMEOUT_S) as http:
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
    """List playable sound names: built-ins (e.g. 'chime', 'alert') plus the user's own sounds (user names win)."""
    return {"sounds": _list_allowlisted_sounds(user_dir=_user_sounds_dir())}


@mcp.tool()
async def speaker_say(
    device_id: str,
    text: str,
    force: Annotated[
        bool,
        Field(
            description=(
                "Bypass quiet hours. Set true only when the user explicitly asked for playback now,"
                " or for a job the user scheduled themselves (e.g. an alarm)."
            )
        ),
    ] = False,
) -> dict[str, Any]:
    """Speak text via TTS on a connected voice-device speaker (1-4000 chars; a morning briefing is fine)."""
    dev = (device_id or "").strip()
    if not _DEVICE_ID_RE.match(dev):
        return {"ok": False, "error": "invalid device_id"}
    cleaned = (text or "").strip()
    if not (1 <= len(cleaned) <= 4000):
        return {"ok": False, "error": "text must be between 1 and 4000 characters"}
    payload: dict[str, Any] = {"text": cleaned}
    if force:
        payload["force"] = True
    res = await _request("POST", f"/api/devices/{dev}/say", json_body=payload, timeout_s=say_timeout_s(cleaned))
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
        Field(
            description=(
                "Bypass quiet hours. Set true only when the user explicitly asked for playback now,"
                " or for a job the user scheduled themselves (e.g. an alarm)."
            )
        ),
    ] = False,
) -> dict[str, Any]:
    """Play a sound by name from list_sounds (built-in or the user's own) on a connected speaker."""
    dev = (device_id or "").strip()
    if not _DEVICE_ID_RE.match(dev):
        return {"ok": False, "error": "invalid device_id"}
    snd = (sound or "").strip()
    if not snd:
        return {"ok": False, "error": "sound must not be empty"}
    payload: dict[str, Any] = {"sound": snd}
    if force:
        payload["force"] = True
    res = await _request("POST", f"/api/devices/{dev}/play", json_body=payload, timeout_s=PLAY_TIMEOUT_S)
    if res.get("ok") is True:
        return {"ok": True, "device_id": dev, "sound": snd}
    if res.get("status") == 409:
        res["error"] = format_quiet_hours_error(res.get("error", ""))
    return res


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
