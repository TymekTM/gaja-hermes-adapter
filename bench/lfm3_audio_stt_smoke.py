#!/usr/bin/env python3
"""Smoke-test STT: LFM3-Audio-2B (llama-server + mmproj) vs referencje FLEURS pl.

Feasibility check "czy Audio-2B zastąpi whispera": kilka klipów z
bench/data/stt/pl/wer (główny checkout), prompt do transkrypcji, proste WER
po normalizacji (ta sama co bench/stt_bench.py).

  flock /tmp/gaja-bench.lock uv run python bench/lfm3_audio_stt_smoke.py --n 8
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_server as S  # noqa: E402

try:
    from num2words import num2words
except ImportError:  # venv bez num2words -> tylko lowercase/cyfry
    num2words = None

RESULTS = S.ROOT / "bench/results"
# Gitignored data and builds live in the main checkout; a worktree points at it with GAJA_MAIN_CHECKOUT.
MAIN_CHECKOUT = Path(os.environ.get("GAJA_MAIN_CHECKOUT") or Path(__file__).resolve().parent.parent)
REFS = MAIN_CHECKOUT / "bench/data/stt/pl/refs.tsv"
WER_DIR = MAIN_CHECKOUT / "bench/data/stt/pl/wer"

HF = Path.home() / ".cache/huggingface/hub"


def gguf_path(repo_dir, name):
    matches = list((HF / repo_dir / "snapshots").glob(f"*/{name}"))
    if not matches:
        raise SystemExit(f"brak {name} w {repo_dir} (pobierz najpierw)")
    return matches[0]


def normalize(text):
    text = text.lower()
    if num2words:
        text = re.sub(r"\d+", lambda m: f" {num2words(int(m.group()), lang='pl')} ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(h) + 1):
            cur = min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, dp[j] = dp[j], cur
    return dp[-1] / len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--model", default="LFM3-Audio-2B-Q4_K_M.gguf")
    ap.add_argument("--mmproj", default="mmproj-LFM3-Audio-2B-Q8_0.gguf")
    ap.add_argument("--port", type=int, default=8793)
    args = ap.parse_args()

    rows = []
    with REFS.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            rows.append(dict(zip(header, line.rstrip("\n").split("\t"))))
    rows = rows[:args.n]

    model = gguf_path("models--LiquidAI--LFM3-Audio-2B-GGUF", args.model)
    mmproj = gguf_path("models--LiquidAI--LFM3-Audio-2B-GGUF", args.mmproj)
    url = f"http://127.0.0.1:{args.port}"
    cmd = [str(S.BIN), "-m", str(model), "--mmproj", str(mmproj),
           "--host", "127.0.0.1", "--port", str(args.port), "-ngl", "99", "-c", "8192",
           "-np", "1", "--jinja", "-fa", "on"]
    print("server:", " ".join(cmd), flush=True)
    log = open(S.ROOT / "bench/logs/audio_smoke_server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            try:
                if requests.get(f"{url}/health", timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if proc.poll() is not None:
                raise SystemExit(f"serwer padł, patrz bench/logs/audio_smoke_server.log")
            time.sleep(0.5)

        out_rows = []
        for row in rows:
            wav = WER_DIR / Path(row["file"]).name
            audio_b64 = base64.b64encode(wav.read_bytes()).decode()
            body = {
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Przepisz dokładnie usłyszany tekst po polsku. "
                                             "Odpowiedz samą transkrypcją, bez komentarza."},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                ]}],
                "temperature": 0, "max_tokens": 256, "stream": False,
            }
            t0 = time.perf_counter()
            resp = requests.post(f"{url}/v1/chat/completions", json=body, timeout=300)
            dt = time.perf_counter() - t0
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            content = (msg.get("content") or "").split("</think>")[-1].strip()
            hyp_n, ref_n = normalize(content), normalize(row["ref"])
            e = wer(ref_n, hyp_n)
            out_rows.append({"utt": row["utt_id"], "wer": round(e, 4), "latency_s": round(dt, 2),
                             "ref": row["ref"], "hyp": content})
            print(f"{row['utt_id']}: WER={e:.2%} {dt:.1f}s -> {content[:90]!r}", flush=True)
        mean_wer = sum(r["wer"] for r in out_rows) / len(out_rows)
        summary = {"model": args.model, "mmproj": args.mmproj, "n": len(out_rows),
                   "wer_mean": round(mean_wer, 4), "rows": out_rows}
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "lfm3_audio_stt_smoke.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nWER mean: {mean_wer:.2%} (baseline faster-whisper small: 18.69%)", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
