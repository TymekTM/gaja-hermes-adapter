#!/usr/bin/env python3
"""Pełna ewaluacja STT: LFM3-Audio-2B vs referencje (parakeet 12.6% / whisper 18.69%).

Metodologia 1:1 z bench/eval_parakeet.py (normalize_text + calc_wer z
bench/eval_hotwords_cost, p50/p90 z numpy). Zbiory:
- PL WER: bench/data/stt/pl/wer (44× FLEURS pl_pl) — benchmark produkcyjny,
- PL latency: bench/data/stt/pl/latency (114 klipów 2-5 s),
- mixed WER: bench/data/stt/mixed/wer (PL/EN techniczne, synteza gosia),
  w dwóch trybach: language=pl i auto (bez pola).

Serwer: llama-server -m LFM3-Audio-2B-Q4_K_M + mmproj Q8_0, port 8794,
transkrypcja przez /v1/audio/transcriptions (kompatybilne z Whisper API).

  BENCH_LLM_PORT=8794 uv run python bench/lfm3_audio_stt_eval.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_hotwords_cost import calc_wer, normalize_text, read_tsv  # noqa: E402
from lfm3_audio_stt_smoke import gguf_path, S  # noqa: E402

MAIN = Path("/home/ttm-server/.t3/projects/gaja-hermes-adapter/bench/data/stt")
RESULTS = S.ROOT / "bench/results"
PORT = int(__import__("os").environ.get("BENCH_LLM_PORT", "8794"))
URL = f"http://127.0.0.1:{PORT}"


def start_server():
    model = gguf_path("models--LiquidAI--LFM3-Audio-2B-GGUF", "LFM3-Audio-2B-Q4_K_M.gguf")
    mmproj = gguf_path("models--LiquidAI--LFM3-Audio-2B-GGUF", "mmproj-LFM3-Audio-2B-Q8_0.gguf")
    cmd = [str(S.BIN), "-m", str(model), "--mmproj", str(mmproj),
           "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99", "-c", "8192",
           "-np", "1", "--jinja", "-fa", "on"]
    log = open(S.ROOT / "bench/logs/audio_stt_eval_server.log", "w")
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(240):
        try:
            if requests.get(f"{URL}/health", timeout=1).status_code == 200:
                return proc, time.perf_counter() - t0
        except requests.RequestException:
            pass
        if proc.poll() is not None:
            raise SystemExit("serwer padł: bench/logs/audio_stt_eval_server.log")
        time.sleep(0.5)
    raise SystemExit("serwer nie wstał w 120 s")


def transcribe(wav: Path, language: str | None):
    data = {"response_format": "json"}
    if language:
        data["language"] = language
    t0 = time.perf_counter()
    with wav.open("rb") as f:
        r = requests.post(f"{URL}/v1/audio/transcriptions",
                          files={"file": (wav.name, f, "audio/wav")}, data=data, timeout=300)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return r.json().get("text", ""), dt


def wer_set(rows, language):
    agg = {"S": 0, "D": 0, "I": 0, "H": 0, "N": 0}
    out = []
    for row in rows:
        wav = MAIN / row["_setdir"] / row["file"]
        hyp, dt = transcribe(wav, language)
        c = calc_wer(normalize_text(row["ref"]), normalize_text(hyp))
        for k in agg:
            agg[k] += c[k]
        out.append({"id": row["utt_id"], "ref": row["ref"], "hyp": hyp,
                    "dur_s": float(row.get("duration_s", 0)), "wall_s": round(dt, 3)})
    wer = (agg["S"] + agg["D"] + agg["I"]) / agg["N"] if agg["N"] else 1.0
    return round(wer, 4), out


def latency_set(rows, language="pl"):
    lats, durs = [], []
    out = []
    for row in rows:
        wav = MAIN / "pl" / row["clip"]
        hyp, dt = transcribe(wav, language)
        lats.append(dt)
        durs.append(float(row["duration_s"]))
        out.append({"clip": row["clip"], "dur_s": float(row["duration_s"]), "wall_s": round(dt, 3)})
    return {
        "n": len(lats),
        "latency_p50_ms": round(float(np.percentile(lats, 50)) * 1000),
        "latency_p90_ms": round(float(np.percentile(lats, 90)) * 1000),
        "rtf_corpus": round(sum(durs) / sum(lats), 2),
        "clips": out,
    }


def main():
    proc, load_s = start_server()
    try:
        time.sleep(2)
        vram = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.strip())
        pl_rows = [dict(r, _setdir="pl") for r in read_tsv(MAIN / "pl/refs.tsv")]
        mixed_rows = [dict(r, _setdir="mixed") for r in read_tsv(MAIN / "mixed/refs.tsv")]

        print(f"== PL WER (44 FLEURS), language=pl ...", flush=True)
        pl_wer, pl_out = wer_set(pl_rows, "pl")
        print(f"   LFM3-Audio WER: {pl_wer:.2%}  (parakeet 12.6%, faster-whisper 18.69%)", flush=True)

        print("== mixed PL/EN WER, language=pl ...", flush=True)
        mx_wer, mx_out = wer_set(mixed_rows, "pl")
        print(f"   WER: {mx_wer:.2%} (parakeet 19.0%)", flush=True)
        print("== mixed PL/EN WER, auto ...", flush=True)
        mx_auto_wer, mx_auto_out = wer_set(mixed_rows, None)
        print(f"   WER: {mx_auto_wer:.2%}", flush=True)

        print("== PL latency (114 klipów 2-5 s) ...", flush=True)
        lat = latency_set(read_tsv(MAIN / "pl/latency.tsv"))
        print(f"   p50={lat['latency_p50_ms']} ms p90={lat['latency_p90_ms']} ms RTF={lat['rtf_corpus']}", flush=True)

        result = {
            "model": "LFM3-Audio-2B-Q4_K_M + mmproj-Q8_0 (llama-server, /v1/audio/transcriptions)",
            "load_time_s": round(load_s, 1), "vram_mib_total": vram,
            "pl_wer": pl_wer, "pl_wer_reference": {"parakeet_cpu": 0.126, "faster_whisper_small_gpu": 0.1869},
            "mixed_wer_lang_pl": mx_wer, "mixed_wer_auto": mx_auto_wer,
            "mixed_wer_reference": {"parakeet_cpu": 0.190},
            "latency": {k: v for k, v in lat.items() if k != "clips"},
            "latency_reference": {"faster_whisper_small_p50_ms": 395, "whistle_pl_p50_ms": 266},
            "rows": {"pl_wer": pl_out, "mixed_pl": mx_out, "mixed_auto": mx_auto_out,
                     "latency": lat["clips"]},
        }
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "lfm3_audio_stt_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                                          encoding="utf-8")
        print("wrote bench/results/lfm3_audio_stt_eval.json", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
