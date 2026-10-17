"""Device HTTP API: GET /api/devices, GET /api/devices/sounds, POST /api/devices/{device_id}/say, POST /api/devices/{device_id}/play.

Returns configured and connected voice devices with status, last seen, and last wake times,
and triggers TTS speech or allow-listed WAV sound playback on a connected device.
"""

from __future__ import annotations

import logging
import re
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from gaja.config import resolve
from gaja.devices.quiet_hours import is_quiet_hour

log = logging.getLogger(__name__)

SOUNDS_DIR = resolve(Path("assets/sounds"))
_SOUND_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# In-memory tracking for device events: device_id -> {"last_seen": ts, "last_wake": ts}
_DEVICE_TIMESTAMPS: dict[str, dict[str, float]] = {}


def record_device_activity(device_id: str, is_wake: bool = False) -> None:
    now = time.time()
    entry = _DEVICE_TIMESTAMPS.setdefault(device_id, {"last_seen": now, "last_wake": 0.0})
    entry["last_seen"] = now
    if is_wake:
        entry["last_wake"] = now


def available_sounds(sounds_dir: Path = SOUNDS_DIR) -> dict[str, Path]:
    """Return a mapping of allow-listed sound names to their WAV file paths in ``sounds_dir``."""
    if not sounds_dir.is_dir():
        return {}
    allowed: dict[str, Path] = {}
    for entry in sorted(sounds_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() == ".wav" and _SOUND_NAME_RE.match(entry.stem):
            allowed[entry.stem] = entry
    return allowed


def list_sounds(sounds_dir: Path = SOUNDS_DIR) -> list[str]:
    """Return sorted allow-listed sound names from ``sounds_dir``."""
    return sorted(available_sounds(sounds_dir))


def load_sound_wav(name: str, sounds_dir: Path = SOUNDS_DIR) -> tuple[bytes, int] | None:
    """Load an allow-listed mono PCM16 WAV file by name.

    Never builds a path from ``name``; only looks up ``name`` in the pre-built
    dictionary from ``available_sounds(sounds_dir)``.
    Returns ``(pcm16_bytes, sample_rate)`` or ``None`` if ``name`` is not allow-listed.
    """
    allowed = available_sounds(sounds_dir)
    wav_path = allowed.get(name)
    if wav_path is None:
        return None
    with wave.open(str(wav_path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            raise ValueError(f"sound {name!r} must be mono 16-bit PCM WAV")
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    return pcm, sample_rate


class SayRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    force: bool = Field(
        default=False,
        description="Bypass quiet hours if the user explicitly asked for playback right now.",
    )


class PlayRequest(BaseModel):
    sound: str = Field(min_length=1, max_length=64)
    force: bool = Field(
        default=False,
        description="Bypass quiet hours if the user explicitly asked for playback right now.",
    )


def create_devices_router(
    origin_ok: Callable[[str | None], bool],
    sounds_dir: Path = SOUNDS_DIR,
    quiet_hours: str | Callable[[], str] | None = None,
) -> APIRouter:
    def check_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        if not origin_ok(origin):
            log.warning("rejected devices request from origin %r", origin)
            raise HTTPException(403, "origin not allowed")

    router = APIRouter(prefix="/api/devices", dependencies=[Depends(check_origin)])

    def _resolve_quiet_hours(request: Request) -> str:
        if callable(quiet_hours):
            return str(quiet_hours())
        if quiet_hours is not None:
            return str(quiet_hours)
        app = getattr(request, "app", None)
        if app is not None:
            if hasattr(app.state, "quiet_hours"):
                return str(app.state.quiet_hours)
            services = getattr(app.state, "services", None)
            if services is not None and getattr(services, "cfg", None):
                devices_cfg = getattr(services.cfg, "devices", None)
                if devices_cfg is not None and hasattr(devices_cfg, "quiet_hours"):
                    return str(devices_cfg.quiet_hours)
            listener = getattr(app.state, "device_listener", None)
            if listener is not None and hasattr(listener, "quiet_hours"):
                return str(listener.quiet_hours)
        return ""

    def _is_quiet(request: Request) -> tuple[bool, str]:
        qh = _resolve_quiet_hours(request)
        app = getattr(request, "app", None)
        clock = getattr(getattr(app, "state", None), "quiet_hours_now", None)
        if callable(clock):
            now_val = clock()
        else:
            now_val = clock
        return is_quiet_hour(qh, now=now_val), qh

    @router.get("")
    async def list_devices(request: Request) -> dict[str, Any]:
        app = request.app
        listener = getattr(app.state, "device_listener", None)
        connected_ids = set(listener.connected) if listener else set()

        registry = getattr(listener, "registry", None)
        if registry is None:
            # Check if services config has devices enabled and tokens_file
            services = getattr(app.state, "services", None)
            if services is not None and getattr(services.cfg, "devices", None):
                from gaja.devices.registry import DeviceRegistry

                registry = DeviceRegistry(services.cfg.devices.tokens_file)

        devices: list[dict[str, Any]] = []
        if registry is not None:
            registry._reload()
            for dev_id, (_tok, info) in sorted(registry._entries.items()):
                online = dev_id in connected_ids
                activity = _DEVICE_TIMESTAMPS.get(dev_id, {})
                last_seen = activity.get("last_seen")
                if online and not last_seen:
                    last_seen = time.time()
                devices.append({
                    "id": info.id,
                    "name": info.name or info.id,
                    "online": online,
                    "last_seen": last_seen,
                    "last_wake": activity.get("last_wake"),
                })

        quiet_now, window = _is_quiet(request)
        return {
            "devices": devices,
            "quiet_now": quiet_now,
            "quiet_hours": window,
            "window": window,
        }

    @router.get("/sounds")
    async def get_sounds() -> dict[str, Any]:
        """List allow-listed sound names available for playback."""
        return {"sounds": list_sounds(sounds_dir)}

    @router.post("/{device_id}/say")
    async def say(device_id: str, body: SayRequest, request: Request) -> dict[str, Any]:
        """Speak a sentence on a connected device (speaker / downlink test). Returns once it was played."""
        quiet_now, window = _is_quiet(request)
        if quiet_now and not body.force:
            raise HTTPException(409, f"Quiet hours active ({window})")
        listener = getattr(request.app.state, "device_listener", None)
        if listener is None or not await listener.say(device_id, body.text.strip()):
            raise HTTPException(404, "device not connected")
        return {"ok": True}

    @router.post("/{device_id}/play")
    async def play(device_id: str, body: PlayRequest, request: Request) -> dict[str, Any]:
        """Play an allow-listed WAV sound on a connected device. Returns once it was played."""
        quiet_now, window = _is_quiet(request)
        if quiet_now and not body.force:
            raise HTTPException(409, f"Quiet hours active ({window})")
        sound_name = body.sound.strip()
        loaded = load_sound_wav(sound_name, sounds_dir)
        if loaded is None:
            raise HTTPException(400, f"unknown sound {sound_name!r}")
        pcm, sample_rate = loaded
        listener = getattr(request.app.state, "device_listener", None)
        if listener is None or not await listener.play(device_id, pcm, sample_rate, name=sound_name):
            raise HTTPException(404, "device not connected")
        return {"ok": True, "sound": sound_name}

    return router
