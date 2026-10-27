#!/usr/bin/env python3
"""Właściwy test LFM3-Audio-2B: STT przez /v1/audio/transcriptions + TTS przez /tts."""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfm3_audio_stt_smoke import normalize, wer, REFS, WER_DIR, gguf_path, S  # noqa: E402

PORT = 8793
URL = f"http://127.0.0.1:{PORT}"


def start_server():
    model = gguf_path("models--LiquidAI--LFM3-Audio-2B-GGUF", "LFM3-Audio-2B-Q4_K_M.gguf")
    mmproj = gguf_path("models--LiquidAI--LFM3-Audio-2B-GGUF", "mmproj-LFM3-Audio-2B-Q8_0.gguf")
    cmd = [str(S.BIN), "-m", str(model), "--mmproj", str(mmproj),
           "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99", "-c", "8192",
           "-np", "1", "--jinja", "-fa", "on"]
    log = open(S.ROOT / "bench/logs/audio_e2e_server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(240):
        try:
            if requests.get(f"{URL}/health", timeout=1).status_code == 200:
                return proc
        except requests.RequestException:
            pass
        if proc.poll() is not None:
            raise SystemExit("serwer padł: bench/logs/audio_e2e_server.log")
        time.sleep(0.5)
    raise SystemExit("serwer nie wstał w 120 s")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    proc = start_server()
    try:
        # --- STT przez właściwy endpoint ---
        rows = []
        with REFS.open(encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            data = [dict(zip(header, l.rstrip("\n").split("\t"))) for l in fh]
        for row in data[:n]:
            wav = WER_DIR / Path(row["file"]).name
            t0 = time.perf_counter()
            with wav.open("rb") as f:
                r = requests.post(f"{URL}/v1/audio/transcriptions",
                                  files={"file": (wav.name, f, "audio/wav")},
                                  data={"language": "pl", "response_format": "json"},
                                  timeout=600)
            dt = time.perf_counter() - t0
            r.raise_for_status()
            hyp = r.json().get("text", "")
            e = wer(normalize(row["ref"]), normalize(hyp))
            rows.append({"utt": row["utt_id"], "wer": round(e, 4), "latency_s": round(dt, 2),
                         "ref": row["ref"], "hyp": hyp})
            print(f"{row['utt_id']}: WER={e:.2%} {dt:.1f}s -> {hyp[:110]!r}", flush=True)
        mean_wer = sum(r["wer"] for r in rows) / len(rows)
        print(f"\nSTT mean WER: {mean_wer:.2%} (faster-whisper small: 18.69%)", flush=True)

        out = {"model": "LFM3-Audio-2B-Q4_K_M + mmproj Q8_0", "n": len(rows),
               "wer_mean": round(mean_wer, 4), "rows": rows, "tts": {}}

        # --- TTS przez /tts ---
        caps = requests.get(f"{URL}/props").json()
        print("\nprops keys:", [k for k in caps.keys()], flush=True)
        texts = {
            "pl": "Cześć! Jestem Gaja, twoja asystentka głosowa. Miło cię słyszeć!",
            "pl2": "Dwadzieścia pięć procent z dwustu czterdziestu to sześćdziesiąt.",
        }
        for tag, text in texts.items():
            try:
                r = requests.post(f"{URL}/tts", json={"input": text, "language": "pl"},
                                  timeout=600)
                print(f"tts[{tag}] status={r.status_code} ctype={r.headers.get('content-type')}", flush=True)
                if r.status_code == 200:
                    outdir = S.ROOT / "bench/results"
                    if "json" in (r.headers.get("content-type") or ""):
                        d = r.json()
                        # szukaj base64 audio w json
                        cand = None
                        def walk(o):
                            nonlocal cand
                            if isinstance(o, dict):
                                for k, v in o.items():
                                    if isinstance(v, str) and len(v) > 1000 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", v):
                                        cand = v
                                    else:
                                        walk(v)
                            elif isinstance(o, list):
                                for v in o:
                                    walk(v)
                        walk(d)
                        print("tts json keys:", json.dumps(d)[:200], flush=True)
                        if cand:
                            import base64
                            p = outdir / f"lfm3_audio_tts_{tag}.wav"
                            p.write_bytes(base64.b64decode(cand))
                            print("zapisano", p, p.stat().st_size, "B", flush=True)
                            out["tts"][tag] = str(p)
                    else:
                        p = outdir / f"lfm3_audio_tts_{tag}.bin"
                        p.write_bytes(r.content)
                        print("zapisano raw", p, len(r.content), "B", flush=True)
                        out["tts"][tag] = str(p)
                else:
                    print("tts error:", r.text[:300], flush=True)
                    out["tts"][tag] = f"error {r.status_code}"
            except Exception as e:  # noqa: BLE001
                print("tts exception:", repr(e)[:300], flush=True)
                out["tts"][tag] = repr(e)[:200]

        (S.ROOT / "bench/results/lfm3_audio_e2e.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\nwrote bench/results/lfm3_audio_e2e.json", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
