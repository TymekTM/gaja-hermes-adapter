"""Wake word ("Hej Gaja") using pymicro-wakeword with the bundled microWakeWord model.

The model files live in `gaja_desktop/models/` (copied from `firmware/esphome/models/` at build
time so PyInstaller bundles them); a custom path can be set in the config. Everything is 16 kHz
16-bit mono — exactly what the capture pipeline produces.

`pymicro_wakeword` is imported lazily: an install where it does not work only disables the wake
word (PTT keeps working), it must not break the client.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
# models/ is git-ignored (filled by the PyInstaller spec); a run from a repo checkout uses the
# firmware's copy, which is the one tracked in git.
_REPO_MODEL_JSON = Path(__file__).resolve().parents[2] / "firmware" / "esphome" / "models" / "hej_gaja.json"
DEFAULT_MODEL_JSON = MODELS_DIR / "hej_gaja.json" if (MODELS_DIR / "hej_gaja.json").exists() else _REPO_MODEL_JSON


class WakeWord:
    """Streaming detector: feed 20 ms PCM16 frames, `process()` returns True when the phrase fired."""

    def __init__(self, detector: object, features: object, model_path: Path) -> None:
        self._detector = detector  # pymicro_wakeword.MicroWakeWord (duck-typed for tests)
        self._features = features  # pymicro_wakeword.MicroWakeWordFeatures
        self.model_path = model_path
        self.name = str(getattr(detector, "wake_word", "hej_gaja"))
        self.cutoff = float(getattr(detector, "probability_cutoff", 0.9))
        self._lock = threading.Lock()  # process() runs on the mic thread, reset() on the loop thread

    @classmethod
    def load(cls, model_json: str | Path = "") -> "WakeWord":
        """Load from a model .json (its .tflite sits next to it); default: the bundled hej_gaja."""
        from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures  # noqa: PLC0415 (lazy)

        path = Path(model_json) if model_json else DEFAULT_MODEL_JSON
        if not path.exists():
            path = DEFAULT_MODEL_JSON
        detector = MicroWakeWord.from_config(str(path))
        return cls(detector, MicroWakeWordFeatures(), path)

    @property
    def step_ms(self) -> int:
        """Feature step size from the model (10 ms normally); the detector wants frames of this."""
        cfg = self.model_path.with_suffix(".json")
        try:
            return int(json.loads(cfg.read_text(encoding="utf-8"))["micro"]["feature_step_size"])
        except Exception:  # noqa: BLE001 - a broken json only means the default
            return 10

    def process(self, pcm16_frame: bytes) -> bool:
        """Feed one chunk of 16 kHz PCM16 audio; True when the sliding window fired."""
        detected = False
        with self._lock:
            for features in self._features.process_streaming(pcm16_frame):
                if self._detector.process_streaming(features):
                    detected = True
        return detected

    def reset(self) -> None:
        with self._lock:
            self._detector.reset()
            self._features.reset()
