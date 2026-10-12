#!/usr/bin/env python3
"""Generates the mixed PL/EN technical test set for STT evaluation.

Reads bench/sentences_mixed.txt and synthesizes 16 kHz mono WAV audio
alternating between models/pl_PL-gosia-medium.onnx and models/pl_PL-darkman-medium.onnx.
Outputs:
  bench/data/stt/mixed/wer/*.wav
  bench/data/stt/mixed/refs.tsv
  bench/data/stt/mixed/latency/*.wav
  bench/data/stt/mixed/latency.tsv
"""

import csv
import io
import os
import re
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS = REPO_ROOT / "models"
DATA_MIXED = REPO_ROOT / "bench" / "data" / "stt" / "mixed"
SENTENCES_FILE = REPO_ROOT / "bench" / "sentences_mixed.txt"


def clean_ref(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def synthesize_mixed():
    from piper import PiperVoice, SynthesisConfig

    DATA_MIXED.mkdir(parents=True, exist_ok=True)
    wer_dir = DATA_MIXED / "wer"
    lat_dir = DATA_MIXED / "latency"
    wer_dir.mkdir(parents=True, exist_ok=True)
    lat_dir.mkdir(parents=True, exist_ok=True)

    gosia_model = MODELS / "pl_PL-gosia-medium.onnx"
    darkman_model = MODELS / "pl_PL-darkman-medium.onnx"

    print("Loading Piper voices...")
    voice_gosia = PiperVoice.load(str(gosia_model), config_path=str(gosia_model) + ".json")
    voice_darkman = PiperVoice.load(str(darkman_model), config_path=str(darkman_model) + ".json")

    sentences = [line.strip() for line in SENTENCES_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(sentences)} mixed sentences from {SENTENCES_FILE}")

    refs_rows = []
    lat_rows = []

    for idx, raw_sentence in enumerate(sentences):
        use_gosia = (idx % 2 == 0)
        voice = voice_gosia if use_gosia else voice_darkman
        vname = "gosia" if use_gosia else "darkman"
        utt_id = f"mixed_{idx:03d}"
        wav_name = f"{utt_id}_{vname}.wav"
        wav_path = wer_dir / wav_name

        # Synthesize audio
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            voice.synthesize(raw_sentence, wav_file)

        buf.seek(0)
        audio, sample_rate = sf.read(buf, dtype="float32")
        if sample_rate != 16000:
            audio = soxr.resample(audio, sample_rate, 16000, quality="HQ")
            sample_rate = 16000

        sf.write(str(wav_path), audio, 16000, subtype="PCM_16")
        dur_s = round(len(audio) / 16000.0, 3)
        ref_norm = clean_ref(raw_sentence)

        refs_rows.append({
            "utt_id": utt_id,
            "file": f"wer/{wav_name}",
            "duration_s": dur_s,
            "ref": ref_norm,
            "raw_ref": raw_sentence,
            "voice": vname,
        })

        # Cut 2-5s latency clips if duration >= 2s
        if dur_s >= 2.0:
            clip_dur = min(dur_s, 3.0)
            clip_samples = int(clip_dur * 16000)
            clip_name = f"mixed_lat_{idx:03d}_{clip_dur:.2f}s.wav"
            clip_path = lat_dir / clip_name
            sf.write(str(clip_path), audio[:clip_samples], 16000, subtype="PCM_16")
            lat_rows.append({
                "clip": f"latency/{clip_name}",
                "duration_s": clip_dur,
                "source": f"wer/{wav_name}",
                "cut_target": 3.0,
            })

    # Write refs.tsv
    with open(DATA_MIXED / "refs.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utt_id", "file", "duration_s", "ref", "raw_ref", "voice"], delimiter="\t")
        writer.writeheader()
        writer.writerows(refs_rows)

    # Write latency.tsv
    with open(DATA_MIXED / "latency.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "duration_s", "source", "cut_target"], delimiter="\t")
        writer.writeheader()
        writer.writerows(lat_rows)

    print(f"Generated {len(refs_rows)} mixed test audio files and metadata in {DATA_MIXED}")


if __name__ == "__main__":
    synthesize_mixed()
