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
import soundfile as sf
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from gaja.api.devices import SOUNDS_DIR, available_sounds, list_sounds as _list_allowlisted_sounds
from gaja.config import Settings, resolve
from gaja.devices.quiet_hours import format_quiet_hours_error
from gaja.devices.volume import DEFAULT_STEP as VOLUME_STEP
from gaja.hermes.client import HermesClientConfig

DEFAULT_API_URL = "http://127.0.0.1:8790"
# /say and /play answer once the audio is sent (TTS + transfer, not playback). The device buffers only
# a few dozen seconds and pushes back, so the downlink is paced: 500 characters (~40 s of speech) took
# most of the original 90 s budget. Long morning briefings get a proportionally larger budget via
# say_timeout_s, up to one Hermes turn (turn_timeout_cap_s).
DEFAULT_TIMEOUT_S = 90.0
# Fixed per-request overhead plus ~2x the speech estimate of 12.5 chars/s (500 chars ≈ 40 s of speech).
SAY_TIMEOUT_BASE_S = 30.0
SAY_TIMEOUT_PER_CHAR_S = 0.16
# The speech rate behind those numbers: speech is sent at ~playback speed, so a text can only be
# spoken inside one turn when len(text) / SPEECH_CHARS_PER_SECOND fits the turn cap.
SPEECH_CHARS_PER_SECOND = 12.5
# /play is paced at playback speed too: the longest user sound (gaja.api.devices.MAX_SOUND_SECONDS, 300 s)
# plus decode and transfer slack. Capped to one Hermes turn at the call site.
PLAY_TIMEOUT_S = 360.0
# One Hermes turn ([hermes] timeout_s, 300 s by default) bounds the whole session/prompt, tool calls
# included: a tool budget above it can only outlive a turn that is already dead. Cap every per-request
# budget just below the turn timeout; playback itself keeps running on the device regardless.
TURN_MARGIN_S = 5.0
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


@functools.cache
def turn_timeout_cap_s() -> float:
    """Largest per-request HTTP budget that still fits inside one Hermes turn (read once per process)."""
    try:
        return Settings().hermes.timeout_s - TURN_MARGIN_S
    except Exception:
        return HermesClientConfig().timeout_s - TURN_MARGIN_S


def max_say_chars() -> int:
    """Longest text whose speech still fits one Hermes turn (3687 chars at the default 300 s turn)."""
    return int(turn_timeout_cap_s() * SPEECH_CHARS_PER_SECOND)


def max_sound_seconds() -> float:
    """Longest sound whose paced send still fits one Hermes turn (295 s at the default 300 s turn)."""
    return turn_timeout_cap_s()


def speech_seconds(text: str) -> float:
    """Rough speech time of ``text`` at the pacing rate above."""
    return len(text) / SPEECH_CHARS_PER_SECOND


def sound_duration_s(name: str) -> float | None:
    """Length of an allow-listed sound in seconds, or None when it cannot be determined."""
    path = available_sounds(SOUNDS_DIR, _user_sounds_dir()).get(name)
    if path is None:
        return None
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        return None


def _say_description() -> str:
    """Tool description: the model must only be promised what one Hermes turn can actually send."""
    return (
        "Speak text via TTS on a connected voice-device speaker (1-"
        f"{max_say_chars()} characters: the longest text whose speech still fits one Hermes turn;"
        " split a longer text into parts)."
    )


def say_timeout_s(text: str) -> float:
    """HTTP budget for one /say call: the default for short texts, scaling for long briefings,
    never more than one Hermes turn."""
    return min(max(DEFAULT_TIMEOUT_S, SAY_TIMEOUT_BASE_S + SAY_TIMEOUT_PER_CHAR_S * len(text)), turn_timeout_cap_s())


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
    return await _request("GET", "/api/devices", timeout_s=min(DEFAULT_TIMEOUT_S, turn_timeout_cap_s()))


@mcp.tool()
async def list_sounds() -> dict[str, Any]:
    """List playable sound names: built-ins (e.g. 'chime', 'alert') plus the user's own sounds (user names win)."""
    # No HTTP call: this only scans the two local sound directories.
    return {"sounds": _list_allowlisted_sounds(user_dir=_user_sounds_dir())}


@mcp.tool(description=_say_description())
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
    """Speak text via TTS on a connected voice-device speaker.

    The limit is one Hermes turn of speech: at most ``max_say_chars()`` characters (3687 at the
    default 300 s turn), because a tool call cannot outlive its turn. A longer text is refused with
    the current limit and must be split into parts.
    """
    dev = (device_id or "").strip()
    if not _DEVICE_ID_RE.match(dev):
        return {"ok": False, "error": "invalid device_id"}
    cleaned = (text or "").strip()
    if not (1 <= len(cleaned) <= 4000):
        return {"ok": False, "error": "text must be between 1 and 4000 characters"}
    limit = max_say_chars()
    if len(cleaned) > limit:
        return {
            "ok": False,
            "error": (
                f"text is too long for one turn: {len(cleaned)} characters is about"
                f" {speech_seconds(cleaned):.0f} s of speech, but a turn allows at most"
                f" {turn_timeout_cap_s():.0f} s ({limit} characters);"
                f" split it into parts of at most {limit} characters"
            ),
        }
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
    """Play a sound by name from list_sounds (built-in or the user's own) on a connected speaker.

    A sound longer than one Hermes turn can send (at most ``max_sound_seconds()`` s, 295 s at the
    default 300 s turn) is refused: the paced send would outlive the turn that asked for it.
    """
    dev = (device_id or "").strip()
    if not _DEVICE_ID_RE.match(dev):
        return {"ok": False, "error": "invalid device_id"}
    snd = (sound or "").strip()
    if not snd:
        return {"ok": False, "error": "sound must not be empty"}
    duration = sound_duration_s(snd)
    if duration is not None and duration > max_sound_seconds():
        return {
            "ok": False,
            "error": (
                f"sound {snd!r} is {duration:.0f} s long, but one turn can send at most"
                f" {max_sound_seconds():.0f} s of playback; pick a shorter sound"
            ),
        }
    payload: dict[str, Any] = {"sound": snd}
    if force:
        payload["force"] = True
    res = await _request("POST", f"/api/devices/{dev}/play", json_body=payload, timeout_s=min(PLAY_TIMEOUT_S, turn_timeout_cap_s()))
    if res.get("ok") is True:
        return {"ok": True, "device_id": dev, "sound": snd}
    if res.get("status") == 409:
        res["error"] = format_quiet_hours_error(res.get("error", ""))
    return res


@mcp.tool()
async def speaker_get_volume(device_id: str) -> dict[str, Any]:
    """Get the speaker volume level (0-100; 100 is full volume, the default) of a voice-device speaker."""
    dev = (device_id or "").strip()
    if not _DEVICE_ID_RE.match(dev):
        return {"ok": False, "error": "invalid device_id"}
    return await _request(
        "GET", f"/api/devices/{dev}/volume", timeout_s=min(DEFAULT_TIMEOUT_S, turn_timeout_cap_s())
    )


@mcp.tool()
async def speaker_set_volume(
    device_id: str,
    level: Annotated[
        int | None,
        Field(ge=0, le=100, description="Absolute volume level in percent: 0 mutes, 100 is full volume."),
    ] = None,
    step: Annotated[
        int | None,
        Field(ge=-8, le=8, description="Relative change in steps; one step is ~12 percent (negative = quieter)."),
    ] = None,
) -> dict[str, Any]:
    """Set the volume of a voice-device speaker, absolutely or relatively.

    Use for requests like Polish "ustaw głośność na 30" / "głośność 50" -> level=30/50;
    "wycisz" -> level=0; "maksymalna głośność" / "na maks" -> level=100;
    "ciszej" / "przycisz" -> step=-1; "głośniej" / "podgłośnij" -> step=1;
    "dużo ciszej" -> step=-2; English "volume 40", "mute", "max volume", "louder", "quieter" likewise.
    Exactly one of ``level`` (absolute percent) or ``step`` (relative, each step ~12 percent) must be
    given. The new level is persisted and returned (fields ``level`` and ``muted``); audio already sent
    keeps its old loudness, everything after plays at the new volume. Works offline too (applies when
    the device reconnects) and is not blocked by quiet hours.
    """
    dev = (device_id or "").strip()
    if not _DEVICE_ID_RE.match(dev):
        return {"ok": False, "error": "invalid device_id"}
    if (level is None) == (step is None):
        return {"ok": False, "error": "pass exactly one of level or step"}
    if step is not None:
        if not -8 <= int(step) <= 8:
            return {"ok": False, "error": "step must be between -8 and 8"}
        body: dict[str, Any] = {"delta": int(step) * VOLUME_STEP}
    else:
        assert level is not None
        if not 0 <= int(level) <= 100:
            return {"ok": False, "error": "level must be between 0 and 100"}
        body = {"level": int(level)}
    return await _request(
        "PUT", f"/api/devices/{dev}/volume", json_body=body, timeout_s=min(DEFAULT_TIMEOUT_S, turn_timeout_cap_s())
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
