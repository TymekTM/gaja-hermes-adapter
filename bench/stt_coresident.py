"""Co-residency on one GPU: LFM llama-server (127.0.0.1:8791) vs faster-whisper, alone and together.

Modes: llm (LLM loop only), stt (STT loop only), both (STT worker process + LLM loop in parallel).
The STT loop runs in a child process so the LLM client's GIL use does not skew STT latency.
"""
import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests
import soundfile as sf

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH))

from stt_bench import DATA, RESULTS, GpuSampler, read_tsv  # noqa: E402

LLM_URL = "http://127.0.0.1:8791/v1/chat/completions"
PROMPT = "Opowiedz krótko, co warto zobaczyć w Krakowie w jeden dzień."
MAX_TOKENS = 150


def system_text():
    md = (ROOT / "prompts" / "system_pl.md").read_text(encoding="utf-8")
    return md.split("```text\n", 1)[1].split("\n```", 1)[0]


def latency_clips():
    rows = read_tsv(DATA / "pl" / "latency.tsv")
    return [(str(DATA / "pl" / row["clip"]), float(row["duration_s"])) for row in rows
            if 2.0 <= float(row["duration_s"]) <= 5.0]


def loadavg():
    return open("/proc/loadavg", encoding="utf-8").read().strip()


def llm_once(system):
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": PROMPT}],
            "temperature": 0, "max_tokens": MAX_TOKENS, "stream": False}
    started = time.perf_counter()
    resp = requests.post(LLM_URL, json=body, timeout=300)
    resp.raise_for_status()
    wall = time.perf_counter() - started
    data = resp.json()
    timings = data["timings"]
    return {"wall_s": wall, "predicted_n": timings["predicted_n"], "predicted_ms": timings["predicted_ms"],
            "predicted_per_second": timings["predicted_per_second"], "prompt_ms": timings["prompt_ms"],
            "finish_reason": data["choices"][0].get("finish_reason"), "chars": len(data["choices"][0]["message"]["content"])}


def llm_loop(seconds):
    system = system_text()
    records, retries = [], 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            records.append(llm_once(system))
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            retries += 1
            time.sleep(2)
    return records, retries


def summarize_llm(records, retries):
    if not records:
        return {"n": 0, "retries": retries}
    tokens = sum(r["predicted_n"] for r in records)
    ms = sum(r["predicted_ms"] for r in records)
    return {"n": len(records), "retries": retries, "predicted_tokens_total": tokens,
            "tokens_per_s_aggregate": tokens / (ms / 1000),
            "tokens_per_s_median_request": statistics.median(r["predicted_per_second"] for r in records),
            "tokens_per_request_median": statistics.median(r["predicted_n"] for r in records),
            "prefill_ms_median": statistics.median(r["prompt_ms"] for r in records),
            "wall_s_median": statistics.median(r["wall_s"] for r in records),
            "finish_reasons": sorted({r["finish_reason"] for r in records}, key=str)}


def worker_main(args):
    from gaja.stt.faster_whisper_stt import FasterWhisperSTT

    gpu = GpuSampler(args.vram_cap_mb)
    gpu.start()
    stt = FasterWhisperSTT(model=args.model, device="cuda", compute_type="int8", beam_size=args.beam)

    async def run():
        await stt.warm_up()
        await stt.warm_up()
        clips = latency_clips()
        audio = [sf.read(path, dtype="float32")[0] for path, _ in clips]
        print("READY", flush=True)
        sys.stdin.readline()
        records = []
        started = time.perf_counter()
        i = 0
        while time.perf_counter() - started < args.seconds:
            path, dur = clips[i % len(clips)]
            out = await stt.transcribe(audio[i % len(clips)], "pl")
            records.append({"id": Path(path).stem, "duration_s": dur, "wall_s": out.duration_ms / 1000, "text": out.text})
            i += 1
        return records

    records = asyncio.run(run())
    stats = gpu.stop()
    Path(args.out).write_text(json.dumps({"records": records, "gpu": stats}, ensure_ascii=False), encoding="utf-8")


def summarize_stt(records):
    if not records:
        return {"n": 0}
    wall = [r["wall_s"] for r in records]
    cuts = sorted(wall)
    return {"n": len(records), "latency_p50_s": statistics.median(wall),
            "latency_p90_s": cuts[int(0.9 * (len(cuts) - 1))], "latency_mean_s": statistics.fmean(wall),
            "latency_max_s": max(wall)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stt-worker", action="store_true")
    p.add_argument("--mode", choices=["llm", "stt", "both"], required=False)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--model", default="small")
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--vram-cap-mb", type=float, default=3900)
    p.add_argument("--out", default=None)
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    if args.stt_worker:
        return worker_main(args)

    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.mode}_{args.model}_b{args.beam}"
    worker_out = RESULTS / f".coresident_worker_{tag}.json"
    gpu_before = GpuSampler(args.vram_cap_mb).snapshot()[0]
    load_before = loadavg()

    proc = None
    if args.mode in ("stt", "both"):
        cmd = [sys.executable, __file__, "--stt-worker", "--seconds", str(args.seconds), "--model", args.model,
               "--beam", str(args.beam), "--vram-cap-mb", str(args.vram_cap_mb), "--out", str(worker_out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        line = proc.stdout.readline().strip()
        if line != "READY":
            proc.kill()
            raise SystemExit(f"STT worker failed to start: {line!r}")
        proc.stdin.write("GO\n")
        proc.stdin.flush()

    llm_records, retries = [], 0
    if args.mode in ("llm", "both"):
        llm_records, retries = llm_loop(args.seconds)
    elif args.mode == "stt":
        time.sleep(args.seconds)

    stt_summary = None
    if proc is not None:
        proc.stdin.close()
        if proc.wait(timeout=120) != 0:
            raise SystemExit(f"STT worker exited {proc.returncode}")
        worker = json.loads(worker_out.read_text(encoding="utf-8"))
        worker_out.unlink()
        stt_summary = summarize_stt(worker["records"])
        stt_summary["gpu_worker"] = worker["gpu"]

    result = {
        "tag": tag, "mode": args.mode, "seconds": args.seconds,
        "llm": {"url": LLM_URL, "prompt": PROMPT, "max_tokens": MAX_TOKENS, "stream": False,
                "system_prompt": "prompts/system_pl.md (text section)",
                "summary": summarize_llm(llm_records, retries)} if args.mode in ("llm", "both") else None,
        "stt": {"model": args.model, "device": "cuda", "compute_type": "int8", "beam_size": args.beam,
                "language": "pl", "summary": stt_summary} if proc is not None else None,
        "gpu_total_mb_before": gpu_before, "gpu_total_mb_after": GpuSampler(args.vram_cap_mb).snapshot()[0],
        "loadavg_before": load_before, "loadavg_after": loadavg(),
        "llm_requests": llm_records,
    }
    (RESULTS / f"stt_coresident_{tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("tag", "llm", "stt")}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
