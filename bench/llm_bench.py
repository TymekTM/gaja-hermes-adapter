#!/usr/bin/env python3
"""Latency profile of LFM2.5 on llama-server: TTFT warm/cold, first sentence, decode speed, VRAM, knob sweep."""
import json
import re
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_server as S  # noqa: E402
import routing_eval as R  # noqa: E402

RESULTS = S.ROOT / "bench/results"
OUT = RESULTS / "llm_latency.json"
URL = S.URL + "/v1/chat/completions"
QUALITY = S.ROOT / "bench/data/quality.jsonl"
PROMPT_IDS = ["pl-q07", "pl-q12", "pl-q16", "pl-q20", "pl-q24"]
REPS = 3
HISTORY_POOL = [
    ("jak się masz", "Dobrze, dziękuję, a ty?"),
    ("co słychać", "Wszystko w porządku, dzięki za pytanie."),
    ("lubisz muzykę", "Lubię spokojne brzmienia, najbardziej wieczorem."),
    ("powiedz coś miłego", "Każdy dzień ma jakiś dobry moment, warto go zauważyć."),
    ("co robisz w weekend", "Porozmawiam z tobą, to mi wystarczy."),
    ("masz jakieś hobby", "Lubię rozmowy, bo każda jest trochę inna."),
    ("jaka jest twoja ulubiona pora dnia", "Chyba wieczór, jest wtedy spokojniej."),
    ("dzięki za rozmowę", "Zawsze chętnie."),
    ("lubisz gotować", "Nie gotuję, ale chętnie słucham o przepisach."),
    ("co myślisz o spacerach", "Spacer dobrze robi na głowę."),
]
BUDGET0 = ["--reasoning-budget", "0"]
NOTHINK = ["--chat-template-file", str(S.TEMPLATES / "lfm_nothink.jinja")]
LFM3_THINK = ["--chat-template-file", str(S.TEMPLATES / "LFM3.jinja")]
LFM3_NOTHINK = ["--chat-template-file", str(S.TEMPLATES / "LFM3.jinja"), "--reasoning-budget", "0"]

CONFIGS = [
    ("baseline_budget0", ["-c", "4096", "-fa", "on"], BUDGET0),
    ("fa_off", ["-c", "4096", "-fa", "off"], BUDGET0),
    ("ub128", ["-c", "4096", "-fa", "on", "-b", "512", "-ub", "128"], BUDGET0),
    ("ub1024", ["-c", "4096", "-fa", "on", "-b", "2048", "-ub", "1024"], BUDGET0),
    ("kv_q8", ["-c", "4096", "-fa", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"], BUDGET0),
    ("ctx8192", ["-c", "8192", "-fa", "on"], BUDGET0),
    ("threads4", ["-c", "4096", "-fa", "on", "-t", "4"], BUDGET0),
    ("threads8", ["-c", "4096", "-fa", "on", "-t", "8"], BUDGET0),
    ("nothink_template", ["-c", "4096", "-fa", "on"], NOTHINK),
]

# LFM3-1.7B: ten sam zestaw gałek, szablon nothink wg models/templates/LFM3.jinja
CONFIGS_LFM3 = [
    ("lfm3_baseline", ["-c", "4096", "-fa", "on"], LFM3_NOTHINK),
    ("lfm3_fa_off", ["-c", "4096", "-fa", "off"], LFM3_NOTHINK),
    ("lfm3_ub128", ["-c", "4096", "-fa", "on", "-b", "512", "-ub", "128"], LFM3_NOTHINK),
    ("lfm3_ub1024", ["-c", "4096", "-fa", "on", "-b", "2048", "-ub", "1024"], LFM3_NOTHINK),
    ("lfm3_kv_q8", ["-c", "4096", "-fa", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"], LFM3_NOTHINK),
    ("lfm3_ctx8192", ["-c", "8192", "-fa", "on"], LFM3_NOTHINK),
    ("lfm3_think", ["-c", "4096", "-fa", "on"], LFM3_THINK),
    ("lfm3_norp", ["-c", "4096", "-fa", "on", "--no-reasoning-preserve"], LFM3_NOTHINK),
    ("lfm3_swafull", ["-c", "4096", "-fa", "on", "--swa-full"], LFM3_NOTHINK),
]


def vram_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip().splitlines()[0])


def median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def tmin(samples):
    xs = [s["ttft_s"] for s in samples if s["ttft_s"]]
    return round(min(xs), 3) if xs else None


def tmax(samples):
    xs = [s["ttft_s"] for s in samples if s["ttft_s"]]
    return round(max(xs), 3) if xs else None


def base_system():
    persona, policy, handoff, tool = R.load_prompt()
    return "\n\n".join([persona, policy, handoff]), tool


def history_msgs(n):
    out = []
    for user_text, answer in HISTORY_POOL[:n]:
        out += [{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}]
    return out


def stream_once(msgs, tool, max_tokens=120):
    body = {"messages": msgs, "tools": [tool], "temperature": 0, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft = sentence = None
    text = ""
    timings = usage = None
    with requests.post(URL, json=body, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            ev = json.loads(payload)
            timings = ev.get("timings") or timings
            usage = ev.get("usage") or usage
            choices = ev.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or ""
            # tryb think: pierwszy token rozumowania też jest pierwszym bajtem strumienia
            if (piece or delta.get("tool_calls") or delta.get("reasoning_content")) and ttft is None:
                ttft = time.perf_counter() - t0
            if piece:
                text += piece
                if sentence is None and len(text.split()) >= 3 and re.search(r"[.!?]", text):
                    sentence = time.perf_counter() - t0
    total = time.perf_counter() - t0
    if timings and timings.get("predicted_per_second"):
        tps = timings["predicted_per_second"]
    else:
        n_out = (usage or {}).get("completion_tokens", 0)
        tps = n_out / (total - ttft) if ttft and total > ttft and n_out else None
    return {"ttft_s": ttft, "sentence_s": sentence, "total_s": total, "decode_tps": tps,
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "think_leak": "</think>" in text,
            "text": text}


def token_count(text):
    resp = requests.post(S.URL + "/tokenize", json={"content": text}, timeout=60)
    resp.raise_for_status()
    return len(resp.json()["tokens"])


def prime(system, tool, history):
    msgs = [{"role": "system", "content": system}, *history]
    body = {"messages": msgs, "tools": [tool], "temperature": 0, "max_tokens": 1, "stream": False}
    resp = requests.post(URL, json=body, timeout=300)
    resp.raise_for_status()


def measure(system, tool, history_n, warm):
    samples = []
    for item_id in PROMPT_IDS:
        item = ITEMS[item_id]
        for _ in range(REPS):
            if warm:
                prime(system, tool, history_msgs(history_n))
                sys_text = system
            else:
                sys_text = f"[sesja {uuid.uuid4().hex}]\n" + system
            msgs = [{"role": "system", "content": sys_text}, *history_msgs(history_n),
                    {"role": "user", "content": item["text"]}]
            samples.append({"id": item_id, **stream_once(msgs, tool)})
    return samples


def summarize_samples(samples):
    return {
        "n": len(samples),
        "ttft_median_s": median([s["ttft_s"] for s in samples]),
        "ttft_min_s": tmin(samples),
        "ttft_max_s": tmax(samples),
        "sentence_median_s": median([s["sentence_s"] for s in samples]),
        "decode_tps_median": median([s["decode_tps"] for s in samples]),
        "prompt_tokens_median": median([s["prompt_tokens"] for s in samples]),
        "think_leaks": sum(1 for s in samples if s["think_leak"]),
    }


def run_config(name, args, extra):
    print(f"--- {name}: {' '.join(args + extra)}", flush=True)
    load_start = S.loadavg_1min()
    vram_before = vram_used_mib()
    proc = S.start([], f"llm_bench_{name}.log", base_args=["-ngl", "99", "-np", "1", "--jinja", *args, *extra])
    try:
        time.sleep(2)
        vram_loaded = vram_used_mib()
        system, tool = base_system()
        cold = measure(system, tool, 0, warm=False)
        warm = measure(system, tool, 0, warm=True)
        result = {"name": name, "args": args + extra, "load_1min_start": load_start,
                  "vram_mib_before_start": vram_before, "vram_mib_loaded": vram_loaded,
                  "vram_mib_delta": vram_loaded - vram_before,
                  "cold": summarize_samples(cold), "warm": summarize_samples(warm),
                  "cold_samples": cold, "warm_samples": warm}
        if name == "baseline_budget0":
            result["system_tokens"] = token_count(system)
            result["tool_tokens"] = token_count(json.dumps(tool, ensure_ascii=False))
            result["history"] = {str(n): summarize_samples(measure(system, tool, n, warm=True)) for n in (1, 5, 10)}
        result["load_1min_end"] = S.loadavg_1min()
    finally:
        S.stop(proc)
    print(f"    ttft warm={result['warm']['ttft_median_s']}s cold={result['cold']['ttft_median_s']}s "
          f"sentence={result['warm']['sentence_median_s']}s tps={result['warm']['decode_tps_median']} "
          f"vram_delta={result['vram_mib_delta']}MiB load={load_start:.2f}", flush=True)
    return result


def load_items():
    return {json.loads(line)["id"]: json.loads(line) for line in QUALITY.read_text(encoding="utf-8").splitlines()
            if line.strip()}


ITEMS = load_items()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="lfm2.5", choices=["lfm2.5", "lfm3"],
                    help="który zestaw CONFIGS uruchomić")
    ap.add_argument("--only", default="", help="filtr po nazwie konfiguracji (podstring)")
    ap.add_argument("--out", default="", help="nadpisz ścieżkę wyniku (domyślnie wg --set)")
    args = ap.parse_args()
    configs = CONFIGS if args.set == "lfm2.5" else CONFIGS_LFM3
    if args.only:
        configs = [c for c in configs if args.only in c[0]]
    out = Path(args.out) if args.out else (OUT if args.set == "lfm2.5"
                                           else RESULTS / f"llm_latency_{args.set}.json")
    RESULTS.mkdir(parents=True, exist_ok=True)
    results = [run_config(name, args_, extra) for name, args_, extra in configs]
    out.write_text(json.dumps({"gguf": S.GGUF.name, "reps": REPS, "prompt_ids": PROMPT_IDS,
                               "configs": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
