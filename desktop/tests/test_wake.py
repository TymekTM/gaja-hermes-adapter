"""The bundled hej_gaja model loads and runs through pymicro-wakeword (CPU only, no audio hardware).

Skipped when the tflite shared library cannot be loaded on this platform — everything else in the
suite covers the detector through fakes.
"""

import numpy as np
import pytest

from gaja_desktop.audio import FRAME_BYTES
from gaja_desktop.wake import DEFAULT_MODEL_JSON, WakeWord


def _loadable() -> bool:
    try:
        WakeWord.load()
        return True
    except Exception:  # noqa: BLE001 - missing/broken tflite backend
        return False


pytestmark = pytest.mark.skipif(not _loadable(), reason="pymicro-wakeword backend unavailable")


def test_bundled_model_loads() -> None:
    w = WakeWord.load()
    assert w.name == "Hej Gaja"
    assert 0 < w.cutoff < 1
    assert w.step_ms == 10  # the micro config's feature step size
    assert DEFAULT_MODEL_JSON.exists() and DEFAULT_MODEL_JSON.with_suffix(".tflite").exists()


def test_silence_never_fires() -> None:
    w = WakeWord.load()
    quiet = (np.sin(np.arange(320) * 0.05) * 200).astype("<i2").tobytes()
    assert len(quiet) == FRAME_BYTES
    fired = False
    for _ in range(300):  # 6 s of near-silence
        fired |= w.process(quiet)
    assert not fired


def test_reset_restarts_the_window() -> None:
    w = WakeWord.load()
    tone = (np.sin(np.arange(320) * 0.5) * 8000).astype("<i2").tobytes()
    for _ in range(100):
        w.process(tone)
    w.reset()
    # after reset the sliding window restarts: no immediate detection without new audio
    assert not w.process(tone)
