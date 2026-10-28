"""Per-device speaker volume: a perceptual server-side gain, persisted across restarts.

The user level (0-100 %) is set by voice ("Hej Gaja, ciszej"), Hermes (``speaker_set_volume``)
or ``PUT /api/devices/{id}/volume``. It is applied as a linear amplitude gain on every PCM
frame the adapter sends to that device (see ``ResamplingOutbox.send_bytes``), so it affects
TTS replies, ``say``, ``play`` and the server earcon alike -- no firmware change needed.

Mapping: perceptual (dB-linear), not linear PCM gain, so equal level steps sound like equal
changes. 100 % is unity gain -- today's loudness, so nothing changes until the user asks.
0 % is silence, reachable only by an explicit "wycisz"/``level = 0``; stepping down from any
audible level stops at ``STEP_FLOOR``, so mashing "ciszej" can never silently mute the speaker.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

DEFAULT_LEVEL = 100  # unity gain: today's loudness
STEP_FLOOR = 5  # "ciszej" steps stop here (~-43 dB, quiet but audible); only explicit 0 mutes
DEFAULT_STEP = 12  # one "ciszej"/"głośniej" step, in level points (12 x 0.45 dB = 5.4 dB)
MAX_ATTENUATION_DB = 45.0  # level 0 would sit here; 0 % is silence instead
_DEVICE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def gain_for_level(level: int) -> float:
    """Linear amplitude factor for a 0-100 level: 100 -> 1.0, 0 -> 0.0, dB-linear in between."""
    level = max(0, min(100, int(level)))
    if level >= 100:
        return 1.0
    if level <= 0:
        return 0.0
    attenuation_db = (100 - level) / 100.0 * MAX_ATTENUATION_DB
    return 10 ** (-attenuation_db / 20.0)


def _valid_id(device_id: str) -> bool:
    return bool(_DEVICE_ID.match(device_id))


class VolumeStore:
    """Levels per device id, kept in a small JSON file (same pattern as TimerManager).

    ``file_path=None`` keeps levels in memory only (tests, or callers that persist elsewhere).
    """

    def __init__(self, file_path: Path | str | None = "data/volumes.json") -> None:
        self._path = Path(file_path) if file_path is not None else None
        self._levels: dict[str, int] = {}
        # Called as on_change(device_id, level, gain) after every persisted change; the server
        # wires it so live device sessions pick the new gain up at once (voice fast path).
        self.on_change: Callable[[str, int, float], None] | None = None
        self._load()

    # ---- reads --------------------------------------------------------------------------

    def level(self, device_id: str) -> int:
        """The stored level, or ``DEFAULT_LEVEL`` for a device the user never changed."""
        return self._levels.get(device_id, DEFAULT_LEVEL)

    def gain(self, device_id: str) -> float:
        return gain_for_level(self.level(device_id))

    def is_muted(self, device_id: str) -> bool:
        return self.level(device_id) <= 0

    def payload(self, device_id: str) -> dict[str, Any]:
        level = self.level(device_id)
        return {
            "device_id": device_id,
            "level": level,
            "gain": self.gain(device_id),
            "muted": level <= 0,
            "default_level": DEFAULT_LEVEL,
            "step": DEFAULT_STEP,
            "floor": STEP_FLOOR,
        }

    # ---- writes -------------------------------------------------------------------------

    def set_level(self, device_id: str, level: int) -> int:
        """Persist an absolute level (0-100, clamped) and notify; returns the stored level."""
        level = max(0, min(100, int(level)))
        self._levels[device_id] = level
        self._save()
        self._notify(device_id)
        return level

    def step_level(self, device_id: str, delta: int) -> int:
        """Move the level by ``delta`` points, but never step below ``STEP_FLOOR``.

        Stepping up from mute gives an audible level again; only an explicit ``set_level(0)``
        silences the speaker.
        """
        level = self.level(device_id)
        if level <= 0:
            new = max(STEP_FLOOR, min(100, int(delta))) if delta > 0 else 0
        else:
            new = max(STEP_FLOOR, min(100, level + int(delta)))
        if new == level:
            return level
        return self.set_level(device_id, new)

    def _notify(self, device_id: str) -> None:
        if self.on_change is None:
            return
        try:
            self.on_change(device_id, self.level(device_id), self.gain(device_id))
        except Exception:
            log.exception("volume on_change callback failed for %s", device_id)

    # ---- file -----------------------------------------------------------------------------

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            levels = data.get("levels", {})
            if isinstance(levels, dict):
                self._levels = {
                    str(k): max(0, min(100, int(v)))
                    for k, v in levels.items()
                    if isinstance(v, (int, float)) and _valid_id(str(k))
                }
        except (OSError, ValueError, TypeError) as exc:
            log.warning("Could not read volume file %s: %s", self._path, exc)

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"version": 1, "levels": self._levels}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self._path)


_module_store: VolumeStore | None = None


def get_volume_store() -> VolumeStore | None:
    """The process-wide store (tools read it like ``get_timer_manager``); None until wired."""
    return _module_store


def set_volume_store(store: VolumeStore | None) -> None:
    global _module_store
    _module_store = store
