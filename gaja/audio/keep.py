"""Keeps the audio of device turns that decoded to nothing, so a live "Gaja did not hear me" can be replayed.

Off unless ``[metrics] empty_turn_audio_keep`` > 0. Files are 16 kHz mono PCM16 WAVs named
``<UTC time>-<device>-turn<N>-<reason>.wav``; only the newest ``keep`` stay. The directory holds household
speech, so it lives under the gitignored ``logs/`` and is never uploaded anywhere.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def keep_turn_audio(directory: Path, keep: int, audio: np.ndarray, *, device: str, turn: int, reason: str) -> Path | None:
    """Write ``audio`` (float32, 16 kHz) and drop all but the newest ``keep`` files; sync, run it in a thread."""
    if keep <= 0 or audio.size == 0:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%f")[:-3]
    path = directory / f"{stamp}-{_UNSAFE.sub('_', device)}-turn{turn}-{_UNSAFE.sub('_', reason)}.wav"
    sf.write(path, np.clip(audio, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")
    for old in sorted(directory.glob("*.wav"))[:-keep]:
        old.unlink(missing_ok=True)
    return path
