"""Generate models/boot_chime.wav: the power-on chime (a quick rising bell arpeggio, drone-style).

    uv run python firmware/esphome/make_boot_chime.py [out.wav]

48 kHz mono PCM16 (the announcement pipeline's rate). Everything sits above ~800 Hz: the Atom's
tiny speaker has no low end and distorts on it. Peak -6 dBFS, played at the 0 dB DAC full scale.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

RATE = 48000
# A major arpeggio climbing an octave and a half, then a held top note: (onset s, Hz, decay s, gain).
NOTES = [
    (0.000, 880.00, 0.16, 0.70),  # A5
    (0.085, 1108.73, 0.16, 0.70),  # C#6
    (0.170, 1318.51, 0.16, 0.75),  # E6
    (0.255, 1760.00, 0.55, 1.00),  # A6, rings out
    (0.255, 2217.46, 0.40, 0.25),  # C#7 shimmer on top of the last note
]
LENGTH_S = 1.05
PEAK = 0.5


def bell(freq: float, decay: float, n: int) -> np.ndarray:
    t = np.arange(n) / RATE
    attack = np.minimum(1.0, t / 0.004)  # 4 ms: a soft click-free onset
    env = attack * np.exp(-t / decay)
    # Slightly detuned pair (a gentle beat) plus a quiet octave: bright but not harsh.
    tone = np.sin(2 * np.pi * freq * t) + 0.6 * np.sin(2 * np.pi * (freq * 1.003) * t)
    tone += 0.18 * np.sin(2 * np.pi * 2 * freq * t) * np.exp(-t / (decay * 0.4))
    return env * tone


def chime() -> np.ndarray:
    out = np.zeros(int(RATE * LENGTH_S))
    for onset, freq, decay, gain in NOTES:
        start = int(RATE * onset)
        out[start:] += gain * bell(freq, decay, len(out) - start)
    fade = int(RATE * 0.08)
    out[-fade:] *= np.linspace(1.0, 0.0, fade)
    return out / np.abs(out).max() * PEAK


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "models" / "boot_chime.wav"
    pcm = (chime() * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    print(f"wrote {path} ({len(pcm) / RATE:.2f} s)")


if __name__ == "__main__":
    main()
