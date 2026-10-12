#!/usr/bin/env python3
"""Measures latency, WER, term accuracy, and hallucination/insertion cost of STT hotwords.

Configurations evaluated on faster-whisper small int8 CUDA beam 1:
(a) No hotwords (baseline)
(b) Default vocab hotwords (all terms, up to 50)
(c) Top 15 terms hotwords

Each configuration runs in a clean subprocess to prevent CTranslate2 CUDA memory fragmentation.
Results saved to bench/results/hotwords_cost.json.
"""

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import soundfile as sf
from num2words import num2words

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from gaja.stt.vocab import VocabularyManager

DATA = REPO_ROOT / "bench" / "data" / "stt"
RESULTS = REPO_ROOT / "bench" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TECH_TERMS = [
    "git", "deploy", "docker", "kubernetes", "pull request", "merge",
    "commit", "backend", "frontend", "api", "endpoint", "gpu",
    "grafana", "home assistant", "tailscale", "python", "typescript",
    "build", "release"
]


def normalize_text(text: str, lang: str = "pl") -> str:
    text = text.lower()
    text = re.sub(r"\d+", lambda m: f" {num2words(int(m.group()), lang=lang)} ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in handle]


def load_audio(path: str) -> np.ndarray:
    data, rate = sf.read(path, dtype="float32", always_2d=False)
    assert rate == 16000, f"Expected 16kHz audio, got {rate} in {path}"
    return data


def calc_wer(ref: str, hyp: str) -> dict[str, int]:
    r_words, h_words = ref.split(), hyp.split()
    if not h_words:
        return {"S": 0, "D": len(r_words), "I": 0, "H": 0, "N": len(r_words)}
    if not r_words:
        return {"S": 0, "D": 0, "I": len(h_words), "H": 0, "N": 0}
    out = jiwer.process_words(ref, hyp)
    return {"S": out.substitutions, "D": out.deletions, "I": out.insertions, "H": out.hits, "N": len(r_words)}


def check_term_accuracy(raw_ref: str, hyp: str) -> tuple[int, int]:
    ref_lower = raw_ref.lower()
    hyp_lower = hyp.lower()
    total_in_ref = 0
    matched = 0
    for term in TECH_TERMS:
        pattern = r"\b" + re.escape(term)
        if re.search(pattern, ref_lower):
            total_in_ref += 1
            stem = "grafan" if term == "grafana" else term
            if re.search(r"\b" + re.escape(stem), hyp_lower) or term in hyp_lower:
                matched += 1
    return matched, total_in_ref


def count_vocab_insertions(vocab_terms: list[str], raw_ref: str, hyp: str) -> list[str]:
    ref_lower = raw_ref.lower()
    hyp_lower = hyp.lower()
    insertions = []
    for term in vocab_terms:
        term_clean = term.lower().strip()
        if not term_clean or len(term_clean) < 3:
            continue
        if not re.search(r"\b" + re.escape(term_clean), ref_lower):
            if re.search(r"\b" + re.escape(term_clean), hyp_lower):
                insertions.append(term)
    return insertions


def run_single_config(config_name: str, out_json: Path):
    from faster_whisper import WhisperModel

    vm = VocabularyManager(
        default_path=REPO_ROOT / "prompts/vocab_default.toml",
        user_path=REPO_ROOT / "data/vocab.toml",
    )
    all_terms = [t.term for t in vm.all_terms]

    if config_name == "none":
        hotwords_str = None
    elif config_name == "all":
        hotwords_str = vm.hotwords(max_terms=50)
    elif config_name == "top15":
        hotwords_str = vm.hotwords(max_terms=15)
    else:
        raise ValueError(f"Unknown config: {config_name}")

    print(f"\n=== Running config: {config_name} (hotwords: {hotwords_str!r}) ===", flush=True)

    pl_refs = read_tsv(DATA / "pl" / "refs.tsv")
    mixed_refs = read_tsv(DATA / "mixed" / "refs.tsv")
    pl_lat = [row for row in read_tsv(DATA / "pl" / "latency.tsv") if 2.0 <= float(row["duration_s"]) <= 5.0]

    import gc

    def get_model():
        return WhisperModel("small", device="cuda", compute_type="int8", cpu_threads=4, local_files_only=True)

    # 1. Polish FLEURS WER set
    print("Loading model for PL set...", flush=True)
    model = get_model()
    # Warmup
    warmup_audio = load_audio(str(DATA / "pl" / pl_refs[0]["file"]))
    model.transcribe(warmup_audio, language="pl", beam_size=1, vad_filter=False, condition_on_previous_text=False, without_timestamps=True, hotwords=hotwords_str, temperature=0.0)

    pl_err_s = pl_err_d = pl_err_i = pl_err_n = 0
    total_insertions = 0
    insertion_details = []

    for row in pl_refs:
        path = str(DATA / "pl" / row["file"])
        audio = load_audio(path)
        segments, _ = model.transcribe(audio, language="pl", beam_size=1, vad_filter=False, condition_on_previous_text=False, without_timestamps=True, hotwords=hotwords_str, temperature=0.0)
        hyp_raw = " ".join(s.text.strip() for s in segments).strip()
        hyp_norm = normalize_text(hyp_raw, "pl")
        ref_norm = normalize_text(row["ref"], "pl")

        err = calc_wer(ref_norm, hyp_norm)
        pl_err_s += err["S"]
        pl_err_d += err["D"]
        pl_err_i += err["I"]
        pl_err_n += err["N"]

        ins = count_vocab_insertions(all_terms, row["ref"], hyp_raw)
        if ins:
            total_insertions += len(ins)
            insertion_details.append({"utt_id": row["utt_id"], "inserted": ins, "hyp": hyp_raw})

    pl_wer = (pl_err_s + pl_err_d + pl_err_i) / max(1, pl_err_n)
    del model
    gc.collect()

    # 2. Mixed PL/EN technical set
    print("Loading fresh model for Mixed set...", flush=True)
    model = get_model()
    mix_err_s = mix_err_d = mix_err_i = mix_err_n = 0
    matched_tot = ref_tot = 0

    for row in mixed_refs:
        path = str(DATA / "mixed" / row["file"])
        audio = load_audio(path)
        segments, _ = model.transcribe(audio, language="pl", beam_size=1, vad_filter=False, condition_on_previous_text=False, without_timestamps=True, hotwords=hotwords_str, temperature=0.0)
        hyp_raw = " ".join(s.text.strip() for s in segments).strip()
        hyp_norm = normalize_text(hyp_raw, "pl")
        ref_norm = normalize_text(row["ref"], "pl")

        err = calc_wer(ref_norm, hyp_norm)
        mix_err_s += err["S"]
        mix_err_d += err["D"]
        mix_err_i += err["I"]
        mix_err_n += err["N"]

        m, r = check_term_accuracy(row["raw_ref"], hyp_raw)
        matched_tot += m
        ref_tot += r

    mixed_wer = (mix_err_s + mix_err_d + mix_err_i) / max(1, mix_err_n)
    term_acc = matched_tot / max(1, ref_tot)
    del model
    gc.collect()

    # 3. Latency on PL 2-5s utterances (sample 20 clips)
    print("Loading fresh model for Latency set...", flush=True)
    model = get_model()
    latencies = []
    for row in pl_lat[:20]:
        path = str(DATA / "pl" / row["clip"])
        audio = load_audio(path)
        t0 = time.perf_counter()
        segments, _ = model.transcribe(audio, language="pl", beam_size=1, vad_filter=False, condition_on_previous_text=False, without_timestamps=True, hotwords=hotwords_str, temperature=0.0)
        _ = " ".join(s.text.strip() for s in segments)
        latencies.append(time.perf_counter() - t0)

    p50_s = float(np.percentile(latencies, 50))
    p90_s = float(np.percentile(latencies, 90))
    del model
    gc.collect()

    res = {
        "config": config_name,
        "hotwords": hotwords_str,
        "pl_wer": pl_wer,
        "mixed_wer": mixed_wer,
        "term_accuracy": term_acc,
        "term_stats": {"matched": matched_tot, "total": ref_tot},
        "latency_p50_ms": round(p50_s * 1000, 1),
        "latency_p90_ms": round(p90_s * 1000, 1),
        "vocab_term_insertions_on_pl": total_insertions,
        "insertion_details": insertion_details,
    }

    print(f"Results for {config_name}:", flush=True)
    print(f"  PL WER: {pl_wer * 100:.2f}%", flush=True)
    print(f"  Mixed WER: {mixed_wer * 100:.2f}%", flush=True)
    print(f"  Term Accuracy: {term_acc * 100:.1f}% ({matched_tot}/{ref_tot})", flush=True)
    print(f"  Latency p50: {res['latency_p50_ms']} ms | p90: {res['latency_p90_ms']} ms", flush=True)
    print(f"  Vocab Term Insertions on PL set: {total_insertions}", flush=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-config", choices=["none", "all", "top15"])
    parser.add_argument("--worker-out", type=Path)
    args = parser.parse_args()

    if args.worker_config:
        run_single_config(args.worker_config, args.worker_out)
        return

    # Master orchestrator
    lock_file = open("/tmp/gaja-bench.lock", "w")
    print("Acquiring /tmp/gaja-bench.lock...")
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    try:
        tmp_dir = Path("/tmp/gaja_hw_bench")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        configs = ["none", "all", "top15"]
        results = {}

        for cfg in configs:
            out_f = tmp_dir / f"{cfg}.json"
            cmd = [sys.executable, __file__, "--worker-config", cfg, "--worker-out", str(out_f)]
            print(f"Launching worker for config: {cfg}...")
            res = subprocess.run(cmd, check=True)
            with open(out_f, encoding="utf-8") as f:
                results[cfg] = json.load(f)

        base_p50 = results["none"]["latency_p50_ms"]
        all_p50 = results["all"]["latency_p50_ms"]
        delta_p50 = round(all_p50 - base_p50, 1)

        summary = {
            "model": "faster-whisper small",
            "compute_type": "int8",
            "device": "cuda",
            "beam_size": 1,
            "latency_overhead_all_vs_none_ms": delta_p50,
            "results": results,
        }

        out_path = RESULTS / "hotwords_cost.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n=== Benchmark Complete ===")
        print(f"Latency Overhead (All vs None): {delta_p50:+.1f} ms p50")
        print(f"Saved hotwords cost report to: {out_path}")
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


if __name__ == "__main__":
    main()
