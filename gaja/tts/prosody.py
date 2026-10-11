from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from gaja.text.style import Style

PIPER_NOISE_SCALE = 0.667
PIPER_NOISE_W_SCALE = 0.8
PIPER_LENGTH_EXPONENT = 1.5
_EMOTION_NOISE_SCALE_DELTA = {
    "neutral": 0.0,
    "happy": 0.08,
    "excited": 0.05,
    "calm": -0.07,
    "sad": -0.12,
    "serious": -0.07,
}
_PAUSE_BEFORE_S = {"sigh": 0.25}
_PAUSE_AFTER_S = {"sad": 0.15, "calm": 0.08, "serious": 0.05, "laugh": 0.12, "sigh": 0.20}
_PITCH_EPSILON = 0.05
_SOFT_KNEE = 0.8
# Piper output is already normalised to ~full scale, so a gain > 1 must not push it into clipping.
_PEAK = 0.95
_pitch_shift_enabled = False


def configure(pitch_shift: bool) -> None:
    """Process-wide switch for the post-hoc pitch shift (config [style].pitch_shift)."""
    global _pitch_shift_enabled
    _pitch_shift_enabled = bool(pitch_shift)
_WSOLA_FRAME_AT_22K = 1024


def engine_params(engine_name: str, style: Style) -> dict:
    if engine_name == "piper":
        noise_scale = PIPER_NOISE_SCALE + _EMOTION_NOISE_SCALE_DELTA.get(style.emotion, 0.0) + 0.1 * (style.energy - 1.0)
        return {
            "length_scale": (1.0 / style.rate) ** PIPER_LENGTH_EXPONENT,
            "noise_scale": min(max(noise_scale, 0.4), 0.95),
            "noise_w_scale": PIPER_NOISE_W_SCALE,
        }
    if engine_name in ("paradee", "supertonic"):
        return {"speed": style.rate}
    return {}


def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if abs(semitones) < _PITCH_EPSILON or len(x) == 0:
        return x.copy()
    frame = max(64, int(round(_WSOLA_FRAME_AT_22K * sr / 22050)) // 2 * 2)
    if len(x) < 4 * frame:
        return x.copy()
    ratio = 2.0 ** (semitones / 12.0)
    stretched = _wsola_stretch(x, ratio, frame=frame, search=frame // 8)
    f = Fraction(ratio).limit_denominator(64)
    shifted = resample_poly(stretched, up=f.denominator, down=f.numerator)
    if len(shifted) >= len(x):
        return shifted[: len(x)].astype(np.float32)
    return np.pad(shifted, (0, len(x) - len(shifted))).astype(np.float32)


def _wsola_stretch(x: np.ndarray, alpha: float, frame: int, search: int) -> np.ndarray:
    hs = frame // 2
    overlap = frame - hs
    n_out = int(round(len(x) * alpha))
    n_frames = int(np.ceil(n_out / hs)) + 1
    xp = np.pad(x, (search, frame + search))
    last = len(xp) - frame
    win = np.hanning(frame + 1)[:-1].astype(np.float32)
    out = np.zeros(n_frames * hs + frame, dtype=np.float32)
    off = search
    out[:frame] += xp[off: off + frame] * win
    for t in range(1, n_frames):
        centre = min(max(int(round(t * hs / alpha)) + search, 0), last)
        lo = max(0, centre - search)
        hi = min(last, centre + search)
        template = xp[off + hs: off + hs + overlap]
        corr = np.correlate(xp[lo: hi + overlap], template, mode="valid")
        off = lo + int(np.argmax(corr))
        out[t * hs: t * hs + frame] += xp[off: off + frame] * win
    return out[:n_out]


def _soft_clip(x: np.ndarray) -> np.ndarray:
    mag = np.abs(x)
    over = mag > _SOFT_KNEE
    if not over.any():
        return x
    span = 1.0 - _SOFT_KNEE
    y = x.copy()
    y[over] = np.sign(x[over]) * (_SOFT_KNEE + span * np.tanh((mag[over] - _SOFT_KNEE) / span))
    return y


def apply_post(audio: np.ndarray, sr: int, style: Style) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    if _pitch_shift_enabled:
        x = pitch_shift(x, sr, style.pitch_semitones)
    if style.energy != 1.0:
        x = x * np.float32(style.energy)
    # clean limiter: scale the whole sentence down instead of squashing its peaks (audible distortion)
    peak = float(np.max(np.abs(x)))
    if peak > _PEAK:
        x = x * np.float32(_PEAK / peak)
    before = sum(_PAUSE_BEFORE_S.get(e, 0.0) for e in style.events)
    after = _PAUSE_AFTER_S.get(style.emotion, 0.0) + sum(_PAUSE_AFTER_S.get(e, 0.0) for e in style.events)
    lead = np.zeros(int(round(before * sr)), dtype=np.float32)
    trail = np.zeros(int(round(after * sr)), dtype=np.float32)
    return np.concatenate([lead, x, trail]).astype(np.float32)
