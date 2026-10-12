#!/usr/bin/env python3
"""Smart Turn v3 semantic endpointing evaluation for Gaja.

Measures:
1. Polish test set precision, recall, F1 for "complete" turn prediction
2. CPU inference latency (p50, p90, mean, loadavg) under flock /tmp/gaja-bench.lock
3. Simulated endpointing metrics:
   - Mean commit delay after true end of speech
   - Number of premature commits on hesitation pauses (400-700 ms)
   compared against the current baseline (250 ms spec / 500 ms min_silence).
"""

from __future__ import annotations

import asyncio
import fcntl
import glob
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal
import soundfile as sf

from gaja.audio.smart_turn import SmartTurn, resolve_smart_turn_model
from gaja.audio.vad import WINDOW, WINDOW_MS, Endpointer, VadEvent
from gaja.config import VADConfig
from gaja.text.style import Style
from gaja.tts.piper_tts import PiperTTS

SAMPLE_RATE = 16000
RESULTS_DIR = Path("bench/results")
SENTENCES_FILE = Path("bench/sentences_pl.txt")
PIPER_MODEL = Path("models/pl_PL-gosia-medium.onnx")
LOCK_FILE = "/tmp/gaja-bench.lock"


def get_loadavg() -> list[float]:
    with open("/proc/loadavg") as f:
        parts = f.read().strip().split()
        return [float(parts[0]), float(parts[1]), float(parts[2])]


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == SAMPLE_RATE:
        return audio.astype(np.float32)
    # E.g. Piper 22050 -> 16000: gcd is 50 -> 320 / 441
    gcd = math.gcd(SAMPLE_RATE, orig_sr)
    up = SAMPLE_RATE // gcd
    down = orig_sr // gcd
    resampled = scipy.signal.resample_poly(audio, up, down)
    return resampled.astype(np.float32)


def build_test_dataset() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build Polish test sets: complete, cut mid-phrase (incomplete), and hesitation pause."""
    complete_items: list[dict[str, Any]] = []

    # 1. Synthesize sentences from bench/sentences_pl.txt using Piper
    if PIPER_MODEL.is_file() and SENTENCES_FILE.is_file():
        tts = PiperTTS("piper", PIPER_MODEL)
        with open(SENTENCES_FILE) as f:
            sentences = [line.strip() for line in f if line.strip()]
        for idx, text in enumerate(sentences):
            audio_22k = asyncio.run(tts.synthesize(text, Style(), "pl"))
            audio_16k = resample_to_16k(audio_22k, tts.sample_rate)
            complete_items.append({
                "id": f"piper_{idx:02d}",
                "text": text,
                "audio": audio_16k,
                "source": "piper",
            })

    # 2. Add natural clips from bench/data/stt/pl/latency
    lat_files = sorted(glob.glob("bench/data/stt/pl/latency/*.wav"))
    for idx, path in enumerate(lat_files):
        audio, sr = sf.read(path, dtype="float32")
        audio_16k = resample_to_16k(audio, sr)
        complete_items.append({
            "id": f"lat_{idx:02d}",
            "text": Path(path).stem,
            "audio": audio_16k,
            "source": "latency_clip",
        })

    # 3. Create incomplete (cut mid-phrase at word boundary / low energy point)
    incomplete_items: list[dict[str, Any]] = []
    hesitation_items: list[dict[str, Any]] = []

    for item in complete_items:
        audio = item["audio"]
        n = len(audio)
        if n < int(1.0 * SAMPLE_RATE):
            continue

        # Find a low energy split point around 40-55% through the audio
        search_start = int(n * 0.35)
        search_end = int(n * 0.60)
        chunk_len = 512
        min_energy = float("inf")
        split_idx = int(n * 0.45)
        for i in range(search_start, search_end, chunk_len):
            energy = float(np.mean(audio[i : i + chunk_len] ** 2))
            if energy < min_energy:
                min_energy = energy
                split_idx = i

        part1 = audio[:split_idx]
        part2 = audio[split_idx:]

        # Incomplete item: cut mid-phrase
        incomplete_items.append({
            "id": f"{item['id']}_cut",
            "text": f"{item['text']} [CUT]",
            "audio": part1,
            "source": item["source"],
        })

        # Hesitation pause item: part1 + 600 ms pause + part2
        pause_ms = 600
        pause_samples = int(pause_ms * SAMPLE_RATE / 1000)
        hesitation_audio = np.concatenate([part1, np.zeros(pause_samples, dtype=np.float32), part2])
        hesitation_items.append({
            "id": f"{item['id']}_hesitation",
            "text": f"{item['text']} [HESITATION 600ms]",
            "audio": hesitation_audio,
            "part1_samples": len(part1),
            "pause_samples": pause_samples,
            "part2_samples": len(part2),
            "source": item["source"],
        })

    return complete_items, incomplete_items, hesitation_items


def evaluate_classifier(
    model: SmartTurn,
    complete_items: list[dict[str, Any]],
    incomplete_items: list[dict[str, Any]],
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]

    complete_probs = [model.predict_probability(it["audio"]) for it in complete_items]
    incomplete_probs = [model.predict_probability(it["audio"]) for it in incomplete_items]

    results_by_thresh: dict[str, Any] = {}
    for th in thresholds:
        tp = sum(1 for p in complete_probs if p >= th)
        fn = sum(1 for p in complete_probs if p < th)
        fp = sum(1 for p in incomplete_probs if p >= th)
        tn = sum(1 for p in incomplete_probs if p < th)

        total = tp + fn + fp + tn
        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        results_by_thresh[f"{th:.2f}"] = {
            "threshold": th,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    return {
        "n_complete": len(complete_items),
        "n_incomplete": len(incomplete_items),
        "complete_prob_mean": round(float(np.mean(complete_probs)), 4),
        "complete_prob_median": round(float(np.median(complete_probs)), 4),
        "incomplete_prob_mean": round(float(np.mean(incomplete_probs)), 4),
        "incomplete_prob_median": round(float(np.median(incomplete_probs)), 4),
        "by_threshold": results_by_thresh,
    }


def benchmark_latency(model: SmartTurn, dummy_audio: np.ndarray, iterations: int = 50) -> tuple[dict[str, Any], list[float], list[float]]:
    # Warm-up
    for _ in range(5):
        model.predict_probability(dummy_audio)

    print(f"Acquiring {LOCK_FILE} for timing measurement...")
    with open(LOCK_FILE, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            loadavg_start = get_loadavg()
            dts: list[float] = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                model.predict_probability(dummy_audio)
                dts.append((time.perf_counter() - t0) * 1000)
            loadavg_end = get_loadavg()
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    dts.sort()
    return {
        "iterations": iterations,
        "mean_ms": round(float(np.mean(dts)), 2),
        "p50_ms": round(float(np.median(dts)), 2),
        "p90_ms": round(dts[int(len(dts) * 0.90)], 2),
        "p95_ms": round(dts[int(len(dts) * 0.95)], 2),
        "min_ms": round(min(dts), 2),
        "max_ms": round(max(dts), 2),
    }, loadavg_start, loadavg_end


def simulate_endpointing(
    cfg: VADConfig,
    smart_turn_model: SmartTurn | None,
    complete_items: list[dict[str, Any]],
    hesitation_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Simulate audio stream through Endpointer and record commit timings."""
    # Energy-based VAD for speech simulation
    def energy_prob(window: np.ndarray) -> float:
        return 1.0 if float(np.abs(window).max()) > 0.03 else 0.0

    trailing_silence_s = 2.5
    trailing_silence = np.zeros(int(trailing_silence_s * SAMPLE_RATE), dtype=np.float32)

    # 1. Complete utterances: measure commit delay after speech ends
    commit_delays_ms: list[float] = []
    for item in complete_items:
        audio = item["audio"]
        full_stream = np.concatenate([audio, trailing_silence])
        speech_len_samples = len(audio)

        ep = Endpointer(energy_prob, cfg, smart_turn=smart_turn_model)
        samples_fed = 0
        committed_at_sample: int | None = None

        for start in range(0, len(full_stream), WINDOW):
            frame = full_stream[start : start + WINDOW]
            samples_fed += len(frame)
            events = ep.feed(frame)
            for ev in events:
                if ev.kind == "speech_end":
                    committed_at_sample = samples_fed
                    break
            if committed_at_sample is not None:
                break

        if committed_at_sample is not None:
            delay_samples = committed_at_sample - speech_len_samples
            delay_ms = max(0.0, delay_samples * 1000.0 / SAMPLE_RATE)
            commit_delays_ms.append(delay_ms)

    # 2. Hesitation pause utterances: check for premature commit during the middle pause
    premature_commits = 0
    hesitation_delays_ms: list[float] = []

    for item in hesitation_items:
        stream = np.concatenate([item["audio"], trailing_silence])
        part1_end = item["part1_samples"]
        pause_end = part1_end + item["pause_samples"]
        speech_total_end = len(item["audio"])

        ep = Endpointer(energy_prob, cfg, smart_turn=smart_turn_model)
        samples_fed = 0
        committed_at_sample: int | None = None
        premature = False

        for start in range(0, len(stream), WINDOW):
            frame = stream[start : start + WINDOW]
            samples_fed += len(frame)
            events = ep.feed(frame)
            for ev in events:
                if ev.kind == "speech_end":
                    committed_at_sample = samples_fed
                    if samples_fed <= pause_end:
                        premature = True
                    break
            if committed_at_sample is not None:
                break

        if premature:
            premature_commits += 1
        elif committed_at_sample is not None:
            delay_ms = max(0.0, (committed_at_sample - speech_total_end) * 1000.0 / SAMPLE_RATE)
            hesitation_delays_ms.append(delay_ms)

    n_hesitation = len(hesitation_items)
    premature_pct = (premature_commits / n_hesitation * 100.0) if n_hesitation > 0 else 0.0

    return {
        "complete_commit_delay_mean_ms": round(float(np.mean(commit_delays_ms)), 1) if commit_delays_ms else None,
        "complete_commit_delay_median_ms": round(float(np.median(commit_delays_ms)), 1) if commit_delays_ms else None,
        "complete_commit_delay_p90_ms": round(float(np.percentile(commit_delays_ms, 90)), 1) if commit_delays_ms else None,
        "hesitation_n": n_hesitation,
        "hesitation_premature_commits": premature_commits,
        "hesitation_premature_pct": round(premature_pct, 1),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "smart_turn_eval.json"

    print("Building Polish test dataset...")
    complete_items, incomplete_items, hesitation_items = build_test_dataset()
    print(f"Dataset: {len(complete_items)} complete, {len(incomplete_items)} incomplete, {len(hesitation_items)} hesitation.")

    model_path = resolve_smart_turn_model()
    print(f"Loading Smart Turn model from {model_path}...")
    st_model = SmartTurn(model_path, threads=1)

    print("Evaluating classifier precision/recall...")
    classifier_metrics = evaluate_classifier(st_model, complete_items, incomplete_items)

    print("Measuring CPU inference latency (50 iterations, 1 thread)...")
    bench_audio = complete_items[0]["audio"]
    latency_metrics, loadavg_start, loadavg_end = benchmark_latency(st_model, bench_audio, iterations=50)

    print("Simulating endpointing outcomes...")
    # Config A: Baseline (no Smart Turn, 250 ms spec, 500 ms min_silence)
    cfg_baseline = VADConfig(
        smart_turn=False,
        spec_silence_ms=250,
        min_silence_ms=500,
    )
    sim_baseline = simulate_endpointing(cfg_baseline, None, complete_items, hesitation_items)

    # Config B: Smart Turn at 250 ms check
    cfg_st250 = VADConfig(
        smart_turn=True,
        smart_turn_check_ms=250,
        smart_turn_threshold=0.5,
        max_incomplete_silence_ms=1200,
        spec_silence_ms=250,
        min_silence_ms=500,
    )
    sim_st250 = simulate_endpointing(cfg_st250, st_model, complete_items, hesitation_items)

    # Config C: Smart Turn at 200 ms check
    cfg_st200 = VADConfig(
        smart_turn=True,
        smart_turn_check_ms=200,
        smart_turn_threshold=0.5,
        max_incomplete_silence_ms=1200,
        spec_silence_ms=200,
        min_silence_ms=500,
    )
    sim_st200 = simulate_endpointing(cfg_st200, st_model, complete_items, hesitation_items)

    results = {
        "model_path": str(model_path),
        "threads": 1,
        "loadavg_start": loadavg_start,
        "loadavg_end": loadavg_end,
        "test_dataset": {
            "complete_count": len(complete_items),
            "incomplete_count": len(incomplete_items),
            "hesitation_count": len(hesitation_items),
        },
        "classifier": classifier_metrics,
        "latency_ms": latency_metrics,
        "simulation": {
            "baseline_500ms": sim_baseline,
            "smart_turn_check250ms": sim_st250,
            "smart_turn_check200ms": sim_st200,
        },
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
