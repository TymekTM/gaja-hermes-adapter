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


class PlayRequest(BaseModel):
    sound: str = Field(min_length=1, max_length=64)


def create_devices_router(
    origin_ok: Callable[[str | None], bool],
    sounds_dir: Path = SOUNDS_DIR,
) -> APIRouter:
    def check_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        if not origin_ok(origin):
            log.warning("rejected devices request from origin %r", origin)
            raise HTTPException(403, "origin not allowed")

    router = APIRouter(prefix="/api/devices", dependencies=[Depends(check_origin)])

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

        return {"devices": devices}

    @router.get("/sounds")
    async def get_sounds() -> dict[str, Any]:
        """List allow-listed sound names available for playback."""
        return {"sounds": list_sounds(sounds_dir)}

    @router.post("/{device_id}/say")
    async def say(device_id: str, body: SayRequest, request: Request) -> dict[str, Any]:
        """Speak a sentence on a connected device (speaker / downlink test). Returns once it was played."""
        listener = getattr(request.app.state, "device_listener", None)
        if listener is None or not await listener.say(device_id, body.text.strip()):
            raise HTTPException(404, "device not connected")
        return {"ok": True}

    @router.post("/{device_id}/play")
    async def play(device_id: str, body: PlayRequest, request: Request) -> dict[str, Any]:
        """Play an allow-listed WAV sound on a connected device. Returns once it was played."""
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
