"""Device HTTP API: GET /api/devices, GET /api/devices/sounds, POST /api/devices/{device_id}/say, POST /api/devices/{device_id}/play, GET/PUT /api/devices/{device_id}/volume.

Returns configured and connected voice devices with status, last seen, and last wake times,
triggers TTS speech or allow-listed sound playback (built-in and user sounds) on a connected device,
and reads/sets the per-device speaker volume (server-side gain; gaja/devices/volume.py).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from gaja.config import resolve
from gaja.devices.quiet_hours import is_quiet_hour
from gaja.devices.registry import DeviceRegistry
from gaja.devices.volume import VolumeStore, get_volume_store

log = logging.getLogger(__name__)

SOUNDS_DIR = resolve(Path("assets/sounds"))
_SOUND_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SOUND_SUFFIXES = frozenset({".wav", ".mp3", ".ogg", ".flac"})
# A sound is refused with 400 (clear error) when the file or its decoded PCM16 form is over these.
MAX_SOUND_FILE_BYTES = 20 * 1024 * 1024  # 20 MB on disk (~3 min of stereo 44.1 kHz MP3 at 320 kbit/s)
MAX_SOUND_SECONDS = 300.0  # 5 minutes: decoded and resampled in RAM (the box is short on memory)

# In-memory tracking for device events: device_id -> {"last_seen": ts, "last_wake": ts}
_DEVICE_TIMESTAMPS: dict[str, dict[str, float]] = {}


def record_device_activity(device_id: str, is_wake: bool = False) -> None:
    now = time.time()
    entry = _DEVICE_TIMESTAMPS.setdefault(device_id, {"last_seen": now, "last_wake": 0.0})
    entry["last_seen"] = now
    if is_wake:
        entry["last_wake"] = now


def available_sounds(sounds_dir: Path = SOUNDS_DIR, user_dir: Path | None = None) -> dict[str, Path]:
    """Map allow-listed sound names to sound files (WAV/MP3/OGG/FLAC).

    Built-in files come from ``sounds_dir``; ``user_dir`` (when given) holds the user's own sounds and
    wins on a name clash. Both directories are re-scanned on every call, so files can be dropped in
    without a restart. A file is allow-listed only when its stem matches ``_SOUND_NAME_RE`` (no path
    traversal, ``load_sound`` never builds a path from a requested name) and its suffix is known.
    """
    allowed: dict[str, Path] = {}
    for directory in (sounds_dir, user_dir):
        if directory is None or not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            if entry.is_file() and entry.suffix.lower() in _SOUND_SUFFIXES and _SOUND_NAME_RE.match(entry.stem):
                allowed[entry.stem] = entry
    return allowed


def list_sounds(sounds_dir: Path = SOUNDS_DIR, user_dir: Path | None = None) -> list[str]:
    """Return sorted allow-listed sound names from ``sounds_dir`` and the optional ``user_dir``."""
    return sorted(available_sounds(sounds_dir, user_dir))


def load_sound(name: str, sounds_dir: Path = SOUNDS_DIR, user_dir: Path | None = None) -> tuple[bytes, int] | None:
    """Load an allow-listed sound by name, decoded to mono PCM16 via libsndfile.

    Never builds a path from ``name``; only looks up ``name`` in the pre-built
    dictionary from ``available_sounds(sounds_dir, user_dir)``.
    Returns ``(pcm16_bytes, sample_rate)`` or ``None`` if ``name`` is not allow-listed.
    Raises ``ValueError`` with a clear message when the file is too large, too long or not audio
    (the API turns that into a 400). Stereo files are mixed down to mono.
    """
    allowed = available_sounds(sounds_dir, user_dir)
    path = allowed.get(name)
    if path is None:
        return None
    size = path.stat().st_size
    if size > MAX_SOUND_FILE_BYTES:
        raise ValueError(f"sound {name!r} is too large ({size} bytes; the limit is {MAX_SOUND_FILE_BYTES})")
    try:
        info = sf.info(str(path))
        duration_s = info.frames / info.samplerate
        if duration_s > MAX_SOUND_SECONDS:
            raise ValueError(
                f"sound {name!r} is too long ({duration_s:.0f} s; the limit is {MAX_SOUND_SECONDS:.0f} s)"
            )
        data, sample_rate = sf.read(str(path), dtype="int16", always_2d=True)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise ValueError(f"sound {name!r} could not be decoded as audio: {exc}") from exc
    if data.shape[0] == 0:
        raise ValueError(f"sound {name!r} contains no audio")
    if data.shape[1] > 1:
        mono = data.astype(np.float32).mean(axis=1)
        pcm = np.clip(mono, -32768, 32767).astype("<i2")
    else:
        pcm = data.reshape(-1).astype("<i2", copy=False)
    if len(pcm) / sample_rate > MAX_SOUND_SECONDS:  # belt and braces: the decoded form itself
        raise ValueError(f"sound {name!r} is too long (the limit is {MAX_SOUND_SECONDS:.0f} s)")
    return pcm.tobytes(), int(sample_rate)


class SayRequest(BaseModel):
    # Long morning briefings are spoken sentence by sentence through the normal TTS path (Session._speak_text).
    text: str = Field(min_length=1, max_length=4000)
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


class VolumeRequest(BaseModel):
    """Exactly one of ``level`` (absolute) or ``delta`` (relative) must be present."""

    level: int | None = Field(default=None, ge=0, le=100, description="Absolute level 0-100; 0 mutes.")
    delta: int | None = Field(default=None, ge=-100, le=100, description="Relative change in level points (negative = quieter).")


def create_devices_router(
    origin_ok: Callable[[str | None], bool],
    sounds_dir: Path = SOUNDS_DIR,
    user_sounds_dir: Path | None = None,
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
        """List allow-listed sound names available for playback (built-ins plus the user's own sounds)."""
        return {"sounds": list_sounds(sounds_dir, user_sounds_dir)}

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
        """Play an allow-listed sound (built-in or user WAV/MP3/OGG/FLAC) on a connected device. Returns once it was played."""
        quiet_now, window = _is_quiet(request)
        if quiet_now and not body.force:
            raise HTTPException(409, f"Quiet hours active ({window})")
        sound_name = body.sound.strip()
        try:
            # Off the event loop: decoding a long MP3 would stall every device socket meanwhile.
            loaded = await asyncio.to_thread(load_sound, sound_name, sounds_dir, user_sounds_dir)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if loaded is None:
            raise HTTPException(400, f"unknown sound {sound_name!r}")
        pcm, sample_rate = loaded
        listener = getattr(request.app.state, "device_listener", None)
        if listener is None or not await listener.play(device_id, pcm, sample_rate, name=sound_name):
            raise HTTPException(404, "device not connected")
        return {"ok": True, "sound": sound_name}

    # ---- speaker volume --------------------------------------------------------------------

    def _registry_entries(request: Request) -> dict[str, Any] | None:
        """Configured device ids, or None when no registry can be built (tests without services)."""
        listener = getattr(request.app.state, "device_listener", None)
        registry = getattr(listener, "registry", None)
        services = getattr(request.app.state, "services", None)
        if registry is None and services is not None and getattr(services.cfg, "devices", None):
            registry = DeviceRegistry(services.cfg.devices.tokens_file)
        if registry is None:
            return None
        registry._reload()
        return registry._entries

    def _volume_store(request: Request) -> VolumeStore:
        services = getattr(request.app.state, "services", None)
        store = getattr(services, "volume_store", None) or get_volume_store()
        if store is None:
            raise HTTPException(503, "volume control is not available")
        return store

    def _apply_to_live_sessions(request: Request, device_id: str, store: VolumeStore) -> None:
        """Push the stored gain into every live session of this device (idempotent with on_change)."""
        services = getattr(request.app.state, "services", None)
        if services is None:
            return
        gain = store.gain(device_id)
        for session in list(getattr(services, "sessions", ()) or ()):
            if session.device_id == device_id:
                session.set_volume_gain(gain)

    @router.get("/{device_id}/volume")
    async def get_volume(device_id: str, request: Request) -> dict[str, Any]:
        """The device's speaker volume level (0-100; 100 = full, default until the user changes it)."""
        entries = _registry_entries(request)
        if entries is not None and device_id not in entries:
            raise HTTPException(404, "unknown device")
        store = _volume_store(request)
        payload = store.payload(device_id)
        payload["ok"] = True
        return payload

    @router.put("/{device_id}/volume")
    async def put_volume(device_id: str, body: VolumeRequest, request: Request) -> dict[str, Any]:
        """Set the speaker volume: ``level`` (absolute 0-100, 0 mutes) or ``delta`` (relative points).

        Not subject to quiet hours: changing the volume is not playback. The level is persisted and
        applied to the live session at once, so running audio changes volume from its next chunk.
        """
        entries = _registry_entries(request)
        if entries is not None and device_id not in entries:
            raise HTTPException(404, "unknown device")
        if (body.level is None) == (body.delta is None):
            raise HTTPException(400, "pass exactly one of level or delta")
        store = _volume_store(request)
        if body.level is not None:
            store.set_level(device_id, body.level)
        else:
            assert body.delta is not None
            store.step_level(device_id, body.delta)
        _apply_to_live_sessions(request, device_id, store)
        payload = store.payload(device_id)
        payload["ok"] = True
        return payload

    return router
