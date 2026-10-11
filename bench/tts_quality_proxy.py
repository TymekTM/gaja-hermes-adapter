#!/usr/bin/env python3
"""Objective quality proxies for the Polish voice candidates. Nothing here is a listening test.

For each voice, the main set (12 sentences) and the probe set (12 sentences) are transcribed by
faster-whisper `small` (CPU, int8, beam 5, language forced to pl) and scored with WER after the same
normalisation as tts_bench.py. Each wav also gets clipping, level and internal-silence stats.

Inputs: bench/results/tts_<tag>.json (written by tts_bench.py; the wav paths and texts come from there).
Output: bench/results/tts_quality_proxy.json and a printed table.

Timed-free, but CPU-heavy, so run it under the shared lock:
  flock /tmp/gaja-bench.lock .venv/bin/python bench/tts_quality_proxy.py
"""
import json
import os
import statistics
import sys
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import jiwer
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from tts_bench import normalize  # noqa: E402

RESULTS = ROOT / "bench" / "results"
OUT_PATH = RESULTS / "tts_quality_proxy.json"
WHISPER_MODEL = "small"
WHISPER_COMPUTE = "int8"
WHISPER_BEAM = 5
SILENCE_DBFS = -45.0
FRAME_S = 0.01
PAUSE_MIN_S = 0.3

VOICES = [
    {"key": "piper_final_gosia", "label": "piper gosia (production default)", "tags": ["final_gosia", "final_gosia_probe"]},
    {"key": "piper_final_darkman", "label": "piper darkman", "tags": ["final_darkman", "final_darkman_probe"]},
    {"key": "piper_final_mc_speech", "label": "piper mc_speech", "tags": ["final_mc_speech", "final_mc_speech_probe"]},
    {"key": "piper_final_bass-high", "label": "piper bass-high", "tags": ["final_bass-high", "final_bass-high_probe"]},
    {"key": "supertonic_final", "label": "supertonic M4 6 steps, 22.05 kHz (production config)",
     "tags": ["final", "final_probe"], "result_prefix": "supertonic_"},
    {"key": "supertonic_native_M4", "label": "supertonic M4 6 steps, native 44.1 kHz",
     "tags": ["native_M4", "native_M4_probe"], "result_prefix": "supertonic_"},
    {"key": "mms_final_cpu", "label": "mms-tts-pol (CPU, CC-BY-NC)", "tags": ["final_cpu", "final_cpu_probe"],
     "result_prefix": "mms_"},
]


def load_items(voice):
    """Return ({set_name: [item,...]}, contaminated) from the result JSON files."""
    prefix = voice.get("result_prefix", "piper_")
    sets, contaminated = {}, False
    for tag in voice["tags"]:
        doc = json.loads((RESULTS / f"tts_{prefix}{tag}.json").read_text(encoding="utf-8"))
        contaminated |= bool(doc.get("contaminated", True))
        sets["probe" if tag.endswith("probe") else "main"] = doc["langs"]["pl"]["items"]
    return sets, contaminated


def load_mono16k(path):
    x, sr = sf.read(ROOT / path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        f = Fraction(16000, sr)
        x = resample_poly(x, f.numerator, f.denominator).astype(np.float32)
    return x, sr


def level_stats(path):
    x, sr = sf.read(ROOT / path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms_db = 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9)
    frame = max(1, int(FRAME_S * sr))
    n = len(x) // frame
    frames_db = 20 * np.log10(np.sqrt(np.mean(np.square(x[: n * frame].reshape(n, frame)), axis=1)) + 1e-9)
    speech = np.where(frames_db > SILENCE_DBFS)[0]
    if speech.size:
        core = frames_db[speech[0]: speech[-1] + 1]
        silent = core <= SILENCE_DBFS
        runs, run = [], 0
        for s in silent:
            if s:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        max_pause = max(runs, default=0) * FRAME_S
        n_pauses = sum(1 for r in runs if r * FRAME_S >= PAUSE_MIN_S)
    else:
        max_pause, n_pauses = 0.0, 0
    return {
        "sr": sr,
        "peak_dbfs": round(20 * np.log10(peak + 1e-9), 2),
        "rms_dbfs": round(float(rms_db), 2),
        "clip_frac_999": round(float(np.mean(np.abs(x) >= 0.999)), 6),
        "max_internal_pause_s": round(max_pause, 3),
        "n_internal_pauses_ge_0_3s": int(n_pauses),
    }


def transcribe_all(model, items):
    rows = []
    for it in items:
        audio, _ = load_mono16k(it["wav_path"])
        segments, _ = model.transcribe(audio, language="pl", beam_size=WHISPER_BEAM)
        hyp = " ".join(s.text for s in segments)
        ref_n, hyp_n = normalize(it["text"], "pl"), normalize(hyp, "pl")
        wer = 1.0 if not hyp_n else round(jiwer.wer(ref_n, hyp_n), 4)
        rows.append({"id": it["id"], "wer_fw": wer, "hyp": hyp, "ref_norm": ref_n, "hyp_norm": hyp_n,
                     **level_stats(it["wav_path"])})
    return rows


def main():
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type=WHISPER_COMPUTE)
    report = {"whisper": {"model": WHISPER_MODEL, "device": "cpu", "compute_type": WHISPER_COMPUTE,
                          "beam_size": WHISPER_BEAM, "language": "pl"},
              "normalisation": "tts_bench.normalize (lowercase, numbers as words, punctuation stripped)",
              "silence_dbfs": SILENCE_DBFS, "pause_min_s": PAUSE_MIN_S, "voices": {}}
    for voice in VOICES:
        sets, contaminated = load_items(voice)
        rows_by_set = {name: transcribe_all(model, items) for name, items in sets.items()}
        allrows = [r for rows in rows_by_set.values() for r in rows]
        report["voices"][voice["key"]] = {
            "label": voice["label"],
            "n": len(allrows),
            "wer_fw_mean": round(statistics.mean(r["wer_fw"] for r in allrows), 3),
            "wer_fw_median": round(statistics.median(r["wer_fw"] for r in allrows), 3),
            "wer_fw_mean_main": round(statistics.mean(r["wer_fw"] for r in rows_by_set["main"]), 3),
            "wer_fw_mean_probe": round(statistics.mean(r["wer_fw"] for r in rows_by_set["probe"]), 3),
            "peak_dbfs_max": max(r["peak_dbfs"] for r in allrows),
            "clip_frac_999_max": max(r["clip_frac_999"] for r in allrows),
            "rms_dbfs_median": round(statistics.median(r["rms_dbfs"] for r in allrows), 2),
            "max_internal_pause_s_max": max(r["max_internal_pause_s"] for r in allrows),
            "contaminated_source": contaminated,
            "sample_rate": allrows[0]["sr"],
            "items": rows_by_set,
        }
        v = report["voices"][voice["key"]]
        print(f"{voice['key']:24s} n={v['n']} wer_fw mean={v['wer_fw_mean']:.3f} main={v['wer_fw_mean_main']:.3f} "
              f"probe={v['wer_fw_mean_probe']:.3f} peak_max={v['peak_dbfs_max']} clip={v['clip_frac_999_max']} "
              f"max_pause={v['max_internal_pause_s_max']}s", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
