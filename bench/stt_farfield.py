#!/usr/bin/env python3
"""Reproducible far-field benchmark for Parakeet STT on Atom-like audio.

Evaluates 44 Polish FLEURS utterances under:
  - Variants:
      1) quiet: scaled to -36 dBFS RMS
      2) room: convolved with synthetic RIR (RT60 ~0.4s, fixed seed), scaled to -36 dBFS RMS
      3) room_noise: room + noise at 20 dB SNR
  - Gain configs:
      1) none
      2) RMS to -26 dBFS
      3) RMS to -20 dBFS
      4) peak to 0.5
  - Hotwords:
      1) none
      2) tech list (score 3.0)
      3) household list (score 3.0)

Reports:
  - WER (normalized case & punctuation)
  - Empty-transcript rate
  - Mean decode time (ms)

Output:
  bench/results/stt_farfield_<YYYYMMDD>.json
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import datetime
import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bench.eval_hotwords_cost import calc_wer, normalize_text, read_tsv  # noqa: E402
from gaja.stt.parakeet_stt import ParakeetSTT  # noqa: E402

log = logging.getLogger("stt_farfield")

# The clips are gitignored: a worktree points GAJA_STT_DATA at the checkout that has them.
DATA_DIR = Path(os.environ.get("GAJA_STT_DATA", REPO_ROOT / "bench" / "data" / "stt" / "pl"))
REFS_PATH = DATA_DIR / "refs.tsv"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

TECH_HOTWORDS = (
    "Gaja, Hermes, Kubernetes, Docker, Git, Grafana, Tailscale, "
    "Home Assistant, TypeScript, API, GPU, endpoint, backend, frontend, "
    "deploy, commit, merge, pull request, build, release"
)
HOUSEHOLD_HOTWORDS = "Gaja, Ania, Ani, klimatyzacja, pogoda"

HOTWORDS_CONFIGS: dict[str, str | None] = {
    "none": None,
    "tech": TECH_HOTWORDS,
    "household": HOUSEHOLD_HOTWORDS,
}

GAIN_CONFIGS = ["none", "-26dBFS", "-20dBFS", "peak0.5"]
VARIANTS = ["quiet", "room", "room_noise"]


# --- Audio math and degradation helpers (numpy only) ------------------------------------------------


def calc_rms(audio: np.ndarray) -> float:
    """Calculates RMS amplitude of a 1D float array."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def calc_dbfs(audio: np.ndarray) -> float:
    """Calculates dBFS relative to full scale 1.0."""
    rms = calc_rms(audio)
    if rms <= 1e-12:
        return -120.0
    return float(20.0 * np.log10(rms))


def scale_to_rms_dbfs(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Scales audio so its RMS matches target_dbfs. Returns copy of silence if energy is zero."""
    rms = calc_rms(audio)
    if rms <= 1e-12:
        return audio.copy().astype(np.float32)
    target_rms = 10.0 ** (target_dbfs / 20.0)
    gain = target_rms / rms
    return (audio * gain).astype(np.float32)


def peak_normalise(audio: np.ndarray, target_peak: float = 0.5) -> np.ndarray:
    """Scales audio so its peak absolute amplitude matches target_peak."""
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak <= 1e-12:
        return audio.copy().astype(np.float32)
    gain = target_peak / peak
    return (audio * gain).astype(np.float32)


def make_room_ir(rt60: float = 0.4, sample_rate: int = 16000, seed: int = 42) -> np.ndarray:
    """Synthesizes a room impulse response with exponential decay matching rt60.

    Amplitude decay drops by 60 dB (10^-3) in rt60 seconds.
    """
    rng = np.random.RandomState(seed)
    num_samples = int(rt60 * sample_rate)
    decay = np.exp(-3.0 * np.log(10.0) * np.arange(num_samples) / num_samples)
    noise = rng.standard_normal(num_samples).astype(np.float32)
    ir = noise * decay.astype(np.float32)
    ir[0] += 1.0  # direct path arrival
    energy = float(np.dot(ir, ir))
    if energy > 1e-12:
        ir = ir / np.sqrt(energy)
    return ir.astype(np.float32)


def degrade_quiet(audio: np.ndarray) -> np.ndarray:
    """Scales audio to -36 dBFS RMS."""
    return scale_to_rms_dbfs(audio, -36.0)


def degrade_room(audio: np.ndarray, ir: np.ndarray | None = None, seed: int = 42) -> np.ndarray:
    """Convolves audio with synthetic RIR (RT60 ~0.4s) and scales to -36 dBFS RMS."""
    if ir is None:
        ir = make_room_ir(rt60=0.4, sample_rate=16000, seed=seed)
    if audio.size == 0 or calc_rms(audio) <= 1e-12:
        return audio.copy().astype(np.float32)
    convolved = np.convolve(audio, ir)[: audio.size].astype(np.float32)
    return scale_to_rms_dbfs(convolved, -36.0)


def degrade_room_noise(
    audio: np.ndarray,
    ir: np.ndarray | None = None,
    snr_db: float = 20.0,
    seed: int = 42,
) -> np.ndarray:
    """Takes room-degraded audio (-36 dBFS) and adds noise at snr_db SNR."""
    room_audio = degrade_room(audio, ir=ir, seed=seed)
    rms_signal = calc_rms(room_audio)
    if rms_signal <= 1e-12:
        return room_audio
    target_noise_rms = rms_signal / (10.0 ** (snr_db / 20.0))
    rng = np.random.RandomState(seed + 1)
    noise = rng.standard_normal(room_audio.size).astype(np.float32)
    noise_rms = calc_rms(noise)
    if noise_rms > 1e-12:
        noise = noise * (target_noise_rms / noise_rms)
    return (room_audio + noise).astype(np.float32)


def apply_gain(audio: np.ndarray, gain_config: str) -> np.ndarray:
    """Applies the specified gain setting."""
    if gain_config == "none":
        return audio.copy()
    if gain_config == "-26dBFS":
        return scale_to_rms_dbfs(audio, -26.0)
    if gain_config == "-20dBFS":
        return scale_to_rms_dbfs(audio, -20.0)
    if gain_config == "peak0.5":
        return peak_normalise(audio, 0.5)
    raise ValueError(f"Unknown gain config: {gain_config}")


# --- Benchmark execution ----------------------------------------------------------------------------


def check_ram_or_abort(min_mb: int = 2000) -> int:
    """Checks MemAvailable via free -m. Stops execution if under min_mb to protect production."""
    free_output = subprocess.check_output(["free", "-m"], text=True)
    available_mb = None
    for line in free_output.splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[0] == "Mem:":
            available_mb = int(parts[6])
            break
    if available_mb is None:
        raise RuntimeError("Failed to parse free -m output")
    if available_mb < min_mb:
        raise SystemExit(
            f"ERROR: Available memory ({available_mb} MB) is under {min_mb} MB threshold. "
            "Aborting to protect production."
        )
    return available_mb


def load_dataset(refs_path: Path, data_dir: Path) -> list[dict[str, Any]]:
    rows = read_tsv(refs_path)
    dataset = []
    for r in rows:
        audio_path = data_dir / r["file"]
        audio, rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        assert rate == 16000, f"Expected 16kHz audio, got {rate} in {audio_path}"
        dataset.append({
            "utt_id": r["utt_id"],
            "raw_audio": audio,
            "ref": r["ref"],
            "duration_s": float(r["duration_s"]),
        })
    return dataset


async def run_benchmark(
    variants: list[str],
    gains: list[str],
    hotwords_keys: list[str],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    # Lower process priority to nice 19 as requested
    try:
        os.nice(19 - os.nice(0))
    except (OSError, AttributeError):
        pass

    avail_mb = check_ram_or_abort(2000)
    print(f"Available memory before benchmark: {avail_mb} MB")

    if not REFS_PATH.exists():
        raise FileNotFoundError(f"Missing refs.tsv at {REFS_PATH}")

    print(f"Loading references and audio from {DATA_DIR}...")
    dataset = load_dataset(REFS_PATH, DATA_DIR)
    if limit is not None:
        dataset = dataset[:limit]
    print(f"Loaded {len(dataset)} clips.")

    # Pre-generate degraded variants in memory
    print("Generating degraded variants in memory...")
    shared_ir = make_room_ir(rt60=0.4, sample_rate=16000, seed=42)
    degraded_audio: dict[str, list[np.ndarray]] = {"quiet": [], "room": [], "room_noise": []}

    for item in dataset:
        raw = item["raw_audio"]
        if "quiet" in variants:
            degraded_audio["quiet"].append(degrade_quiet(raw))
        if "room" in variants:
            degraded_audio["room"].append(degrade_room(raw, ir=shared_ir, seed=42))
        if "room_noise" in variants:
            degraded_audio["room_noise"].append(degrade_room_noise(raw, ir=shared_ir, snr_db=20.0, seed=42))

    new_results: list[dict[str, Any]] = []

    # Sequential recognizer loading per hotword config
    for hw_key in hotwords_keys:
        hw_str = HOTWORDS_CONFIGS[hw_key]
        check_ram_or_abort(2000)
        print(f"\n--- Loading ParakeetSTT recognizer for hotwords={hw_key} (score=3.0) ---")

        # Instantiate ONE model instance with num_threads=2
        stt = ParakeetSTT(
            num_threads=2,
            hotwords_score=3.0,
            max_active_paths=4,
            prefix_below_s=2.6,
        )
        await stt.warm_up()

        try:
            for variant in variants:
                audio_list = degraded_audio[variant]
                for gain in gains:
                    print(f"Evaluating variant={variant}, gain={gain}, hotwords={hw_key}...", flush=True)
                    tally = {"S": 0, "D": 0, "I": 0, "H": 0, "N": 0}
                    empty_count = 0
                    decode_times_ms: list[float] = []

                    for idx, item in enumerate(dataset):
                        audio = audio_list[idx]
                        processed_audio = apply_gain(audio, gain)

                        t0 = time.perf_counter()
                        transcript = await stt.transcribe(
                            processed_audio,
                            language="pl",
                            hotwords=hw_str,
                        )
                        dt_ms = (time.perf_counter() - t0) * 1000
                        decode_times_ms.append(dt_ms)

                        hyp_raw = transcript.text
                        if not hyp_raw.strip():
                            empty_count += 1

                        hyp_norm = normalize_text(hyp_raw, "pl")
                        ref_norm = normalize_text(item["ref"], "pl")
                        err = calc_wer(ref_norm, hyp_norm)

                        tally["S"] += err["S"]
                        tally["D"] += err["D"]
                        tally["I"] += err["I"]
                        tally["H"] += err["H"]
                        tally["N"] += err["N"]

                    total_errors = tally["S"] + tally["D"] + tally["I"]
                    total_words = max(1, tally["N"])
                    wer = round(total_errors / total_words, 4)
                    empty_rate = round(empty_count / len(dataset), 4)
                    mean_decode_ms = round(float(np.mean(decode_times_ms)), 1)

                    print(
                        f"  -> WER: {wer * 100:.2f}% | Empty: {empty_count}/{len(dataset)} "
                        f"({empty_rate * 100:.1f}%) | Mean decode: {mean_decode_ms} ms"
                    )

                    new_results.append({
                        "variant": variant,
                        "gain": gain,
                        "hotwords": hw_key,
                        "wer": wer,
                        "wer_pct": round(wer * 100, 2),
                        "empty_rate": empty_rate,
                        "empty_count": empty_count,
                        "total_clips": len(dataset),
                        "mean_decode_ms": mean_decode_ms,
                        "substitutions": tally["S"],
                        "deletions": tally["D"],
                        "insertions": tally["I"],
                        "total_words": tally["N"],
                    })
        finally:
            # Free model and threadpool before next instance
            stt._executor.shutdown(wait=True)
            del stt
            gc.collect()

    return new_results


def save_merged_results(out_path: Path, new_results: list[dict[str, Any]]) -> None:
    existing_data: dict[str, Any] = {}
    if out_path.exists():
        try:
            existing_data = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing_data = {}

    results_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in existing_data.get("results", []):
        results_map[(r["variant"], r["gain"], r["hotwords"])] = r
    for r in new_results:
        results_map[(r["variant"], r["gain"], r["hotwords"])] = r

    # Sort results for deterministic order
    ordered_results = sorted(
        results_map.values(),
        key=lambda x: (
            VARIANTS.index(x["variant"]) if x["variant"] in VARIANTS else 99,
            list(HOTWORDS_CONFIGS.keys()).index(x["hotwords"]) if x["hotwords"] in HOTWORDS_CONFIGS else 99,
            GAIN_CONFIGS.index(x["gain"]) if x["gain"] in GAIN_CONFIGS else 99,
        ),
    )

    data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_configs": len(ordered_results),
        "total_clips_per_config": ordered_results[0]["total_clips"] if ordered_results else 44,
        "output_file": str(out_path),
        "results": ordered_results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(ordered_results)} configs to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Far-field benchmark for Parakeet STT")
    parser.add_argument("--variant", choices=["all"] + VARIANTS, default="all")
    parser.add_argument("--gain", choices=["all"] + GAIN_CONFIGS, default="all")
    parser.add_argument("--hotwords", choices=["all", "none", "tech", "household"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of clips for smoke testing")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    variants = VARIANTS if args.variant == "all" else [args.variant]
    gains = GAIN_CONFIGS if args.gain == "all" else [args.gain]
    hw_keys = list(HOTWORDS_CONFIGS.keys()) if args.hotwords == "all" else [args.hotwords]

    date_str = datetime.date.today().strftime("%Y%m%d")
    out_path = args.out or (RESULTS_DIR / f"stt_farfield_{date_str}.json")

    new_results = asyncio.run(run_benchmark(variants, gains, hw_keys, limit=args.limit))
    save_merged_results(out_path, new_results)


if __name__ == "__main__":
    main()
