"""Synthesize a 16 kHz mono test utterance with the Polish Piper voice (default gosia, from models/)."""

from __future__ import annotations

import argparse
import io
import wave
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from piper import PiperVoice

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOICE = ROOT / "models/pl_PL-gosia-medium.onnx"
DEFAULT_TEXT = "Dzień dobry, chciałbym sprawdzić pogodę na jutro w Krakowie. Czy będzie padać deszcz po południu?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--voice", type=Path, default=DEFAULT_VOICE)
    parser.add_argument("--out", type=Path, default=ROOT / "bench/out/input_pl_16k.wav")
    args = parser.parse_args()

    voice = PiperVoice.load(args.voice)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(args.text, wf)
    buf.seek(0)
    audio, sr = sf.read(buf, dtype="float32")
    audio16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, np.asarray(audio16k, dtype=np.float32), 16000, subtype="PCM_16")
    print(f"wrote {args.out} sr_in={sr} dur={len(audio16k) / 16000:.2f}s text={args.text!r}")


if __name__ == "__main__":
    main()
