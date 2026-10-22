#!/usr/bin/env python3
"""Gaja-shaped latency profile of llama-server configs (llm-speed workstream).

Measures what the adapter actually sends to :8791:
  - chat: POST /v1/chat/completions {system(prompts/system_pl.md)+history+user, tools, stream,
    temperature 0.3, max_tokens 200, cache_prompt, id_slot 0}  (gaja/llm/client.py stream())
  - router: POST /completion {n_predict 1, n_probs 10, temperature 0, cache_prompt, id_slot 1}
    on the prompts/router_pl.md prompt (gaja/llm/router.py classify())
  - raw prefill/decode probe for pp t/s and tg t/s from the server's own timings.

Cold = fresh session marker in the system message (full prefill). Warm = repeated prompt
(llama-server prefix cache hit, the state after prime_chat()/prime_after_turn).

The script never starts or stops any systemd unit; wrap invocations in
`flock /tmp/gaja-bench.lock` yourself and only point --port at 8791 while the production
gaja-llm unit is down.

Usage: flock /tmp/gaja-bench.lock uv run python bench/llm_speed_sweep.py \
          --configs prod_baseline,ub128 --port 8791 --tag w1
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MAIN_CHECKOUT = Path("/home/ttm-server/.t3/projects/gaja-hermes-adapter")
BIN = Path(
    MAIN_CHECKOUT / "third_party/llama.cpp/build/bin/llama-server"
)  # same binary the production unit runs
GGUF_PATH_FILE = ROOT / "bench/logs/lfm_gguf_path.txt"
if not GGUF_PATH_FILE.exists():
    GGUF_PATH_FILE = MAIN_CHECKOUT / "bench/logs/lfm_gguf_path.txt"
GGUF = Path(GGUF_PATH_FILE.read_text().strip())
TEMPLATES = ROOT / "deploy"
RESULTS = ROOT / "bench/results/llm-speed"

# Fixed on every config: the production template + reasoning flags (docs/LLM_RESULTS.md section 9).
COMMON = ["-ngl", "99", "--jinja", "--chat-template-file", str(TEMPLATES / "lfm_nothink.jinja"),
          "--reasoning-budget", "0"]

CONFIGS: dict[str, list[str]] = {
    # Production unit ExecStart (systemctl --user cat gaja-llm, verified 2026-10-22).
    "prod_baseline": ["-c", "8192", "-np", "2", "--no-kv-unified", "-fa", "on"],
    # logical/physical batch sweep (server defaults: -b 2048 -ub 512).
    "ub128": ["-c", "8192", "-np", "2", "--no-kv-unified", "-fa", "on", "-b", "2048", "-ub", "128"],
    "ub256": ["-c", "8192", "-np", "2", "--no-kv-unified", "-fa", "on", "-b", "2048", "-ub", "256"],
    "ub1024": ["-c", "8192", "-np", "2", "--no-kv-unified", "-fa", "on", "-b", "2048", "-ub", "1024"],
    "ub2048_b4096": ["-c", "8192", "-np", "2", "--no-kv-unified", "-fa", "on", "-b", "4096", "-ub", "2048"],
    "b1024_ub512": ["-c", "8192", "-np", "2", "--no-kv-unified", "-fa", "on", "-b", "1024", "-ub", "512"],
    # slots / context
    "np1_c8192": ["-c", "8192", "-np", "1", "--no-kv-unified", "-fa", "on"],
    "np1_c4096": ["-c", "4096", "-np", "1", "--no-kv-unified", "-fa", "on"],
    "np2_c4096": ["-c", "4096", "-np", "2", "--no-kv-unified", "-fa", "on"],
    # KV cache types (need -fa on)
    "kv_q8": ["-c", "8192", "-np", "2", "--no-kv-unified", "-fa", "on",
              "-ctk", "q8_0", "-ctv", "q8_0"],
    "kv_q8_c4096": ["-c", "4096", "-np", "2", "--no-kv-unified", "-fa", "on",
                    "-ctk", "q8_0", "-ctv", "q8_0"],
    # attention
    "fa_off": ["-c", "8192", "-np", "2", "--no-kv-unified", "-fa", "off"],
    # CPU threads (generation is GPU-bound; threads only feed the remaining CPU work)
    "t4": ["-c", "8192", "-np", "2", "--no-kv-unified", "-fa", "on", "-t", "4", "-tb", "4"],
    "t2": ["-c", "8192", "-np", "2", "--no-kv-unified", "-fa", "on", "-t", "2", "-tb", "2"],
    # prompt-cache reuse across prefix shifts (history trims)
    "cache_reuse": ["-c", "8192", "-np", "2", "--no-kv-unified", "-fa", "on", "--cache-reuse", "256"],
    # RAM pinning (this build: --load-mode; the old --mlock/--no-mmap flags are rejected)
    "mlock": ["-c", "8192", "-np", "2", "--no-kv-unified", "-fa", "on", "-lm", "mmap+mlock"],
    "no_mmap": ["-c", "8192", "-np", "2", "--no-kv-unified", "-fa", "on", "-lm", "none"],
    # n-gram speculative decoding (no draft model needed; lossless vs greedy target)
    "spec_ngram": ["-c", "8192", "-np", "2", "--no-kv-unified", "-fa", "on",
                   "--spec-type", "ngram-simple"],
}

# drift/variance control: identical args to prod_baseline under a distinct key so the
# resume logic re-runs it in a later window
CONFIGS["baseline_repeat"] = list(CONFIGS["prod_baseline"])

URL = "http://127.0.0.1:8797"  # overridden by --port


def set_url(port: int) -> None:
    global URL
    URL = f"http://127.0.0.1:{port}"


def vram_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip().splitlines()[0])


def loadavg_1min() -> float:
    return float(Path("/proc/loadavg").read_text().split()[0])


def start_server(extra: list[str], log_name: str, port: int) -> subprocess.Popen:
    cmd = [str(BIN), "-m", str(GGUF), "--host", "127.0.0.1", "--port", str(port), *COMMON, *extra]
    log_path = ROOT / "bench/logs" / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    for _ in range(240):
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early, see {log_path}")
        try:
            if requests.get(f"{url}/health", timeout=1).status_code == 200:
                return proc
        except requests.RequestException:
            pass
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("llama-server did not become healthy")


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def load_system_prompt() -> tuple[str, dict]:
    text = (ROOT / "prompts/system_pl.md").read_text(encoding="utf-8")
    system = re.search(r"## System\n\n```text\n(.*?)\n```", text, re.S).group(1)
    tool = json.loads(re.search(r"## Narzędzie\n\n```json\n(.*?)\n```", text, re.S).group(1))
    return system, tool


def load_router_system() -> str:
    text = (ROOT / "prompts/router_pl.md").read_text(encoding="utf-8")
    return re.search(r"## System\n\n```text\n(.*?)\n```", text, re.S).group(1)


def load_quality_items() -> list[dict]:
    items = [json.loads(line) for line in
             (ROOT / "bench/data/quality.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {i["id"]: i for i in items}
    # same short-voice prompts llm_bench.py uses, plus a weather/hermes-shaped question for realism
    return [by_id[i] for i in ("pl-q07", "pl-q12", "pl-q16", "pl-q20", "pl-q24")]


HISTORY_POOL = [
    ("jak się masz", "Dobrze, dziękuję, a ty?"),
    ("co słychać", "Wszystko w porządku, dzięki za pytanie."),
    ("lubisz muzykę", "Lubię spokojne brzmienia, najbardziej wieczorem."),
    ("powiedz coś miłego", "Każdy dzień ma jakiś dobry moment, warto go zauważyć."),
    ("co robisz w weekend", "Porozmawiam z tobą, to mi wystarczy."),
    ("masz jakieś hobby", "Lubię rozmowy, bo każda jest trochę inna."),
    ("jaka jest twoja ulubiona pora dnia", "Chyba wieczór, jest wtedy spokojniej."),
    ("dzięki za rozmowę", "Zawsze chętnie."),
]


def history_msgs(n: int) -> list[dict]:
    out = []
    for user_text, answer in HISTORY_POOL[:n]:
        out += [{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}]
    return out


def chat_stream(msgs: list[dict], tool: dict, slot: int, max_tokens: int = 200) -> dict:
    """Exactly the payload gaja/llm/client.py stream() sends (minus 'model')."""
    body = {"model": "lfm2.5-2.6b", "messages": msgs, "stream": True, "temperature": 0.3,
            "max_tokens": max_tokens, "cache_prompt": True, "id_slot": slot, "tools": [tool]}
    t0 = time.perf_counter()
    ttft = sentence = None
    text = ""
    timings = usage = None
    with requests.post(f"{URL}/v1/chat/completions", json=body, stream=True, timeout=300) as resp:
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
            if (piece or delta.get("tool_calls")) and ttft is None:
                ttft = time.perf_counter() - t0
            if piece:
                text += piece
                if sentence is None and len(text.split()) >= 3 and re.search(r"[.!?]", text):
                    sentence = time.perf_counter() - t0
    total = time.perf_counter() - t0
    pred_tps = timings.get("predicted_per_second") if timings else None
    prompt_tps = timings.get("prompt_per_second") if timings else None
    # streaming responses carry no usage chunk (client.py does not send stream_options),
    # so token counts come from the server's own timings
    return {"ttft_s": ttft, "sentence_s": sentence, "total_s": total,
            "decode_tps": pred_tps, "prefill_tps": prompt_tps,
            "prompt_tokens": (usage or {}).get("prompt_tokens") or (timings or {}).get("prompt_n"),
            "completion_tokens": (usage or {}).get("completion_tokens") or (timings or {}).get("predicted_n"),
            "tool_call": "tool_calls" in text, "text": text}


def router_probe(system: str, slot: int, reps: int = 20, history: list[dict] | None = None) -> list[dict]:
    """gaja/llm/router.py classify(): one 1-token /completion with n_probs on the router slot."""
    parts = [f"<|startoftext|><|im_start|>system\n{system}<|im_end|>\n"]
    for m in history or []:
        role = m["role"]
        parts.append(f"<|im_start|>{role}\n" + (f"Zapytanie: {m['content']}" if role == "user" else m["content"])
                     + "<|im_end|>\n")
    parts.append("<|im_start|>user\nZapytanie: jaka jest dzisiaj pogoda w Lublinie<|im_end|>\n"
                 "<|im_start|>assistant\nKategoria:")
    body = {"prompt": "".join(parts), "n_predict": 1, "n_probs": 10, "temperature": 0,
            "cache_prompt": True, "id_slot": slot}
    requests.post(f"{URL}/completion", json=body, timeout=60)  # warm the cache once
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = requests.post(f"{URL}/completion", json=body, timeout=60)
        r.raise_for_status()
        dt = time.perf_counter() - t0
        data = r.json()
        out.append({"dt_ms": dt * 1000, "prompt_tokens": data.get("tokens_evaluated"),
                    "timings": data.get("timings")})
    return out


def raw_probe(slot: int, max_prompt_tokens: int = 1400) -> dict:
    """Prefill (pp) and decode (tg) t/s from the server's own timings on raw /completion calls."""
    filler = ("Pogoda w Polsce zmienia się często. Wiatr wieje od zachodu, czasem pada deszcz. "
              "Latem bywa gorąco, a zimą śnieg. Ludzie rozmawiają o temperaturze codziennie. ")
    repeats = max(4, min(40, (max_prompt_tokens - 60) // 31))
    prompt = "<|startoftext|><|im_start|>system\n" + filler * repeats + "<|im_end|>\n<|im_start|>user\n"
    body = {"prompt": prompt, "n_predict": 1, "temperature": 0, "cache_prompt": False, "id_slot": slot}
    pp = []
    for _ in range(3):
        r = requests.post(f"{URL}/completion", json=body, timeout=120)
        r.raise_for_status()
        t = r.json().get("timings") or {}
        if t.get("prompt_per_second"):
            pp.append(t["prompt_per_second"])
    body2 = {"prompt": prompt, "n_predict": 64, "temperature": 0, "cache_prompt": True,
             "id_slot": slot, "ignore_eos": True}
    r = requests.post(f"{URL}/completion", json=body2, timeout=120)
    r.raise_for_status()
    t = r.json().get("timings") or {}
    return {"pp_tps": statistics.median(pp) if pp else None,
            "pp_runs": pp, "tg_tps": t.get("predicted_per_second"),
            "tg_tokens": t.get("predicted_n"), "probe_prompt_tokens": t.get("prompt_n")}


def pct(vals: list[float | None], q: float) -> float | None:
    xs = sorted(v for v in vals if v is not None)
    if not xs:
        return None
    idx = min(len(xs) - 1, round(q * (len(xs) - 1)))
    return round(xs[idx], 3)


def summarize_chat(samples: list[dict]) -> dict:
    return {
        "n": len(samples),
        "ttft_p50_s": pct([s["ttft_s"] for s in samples], 0.5),
        "ttft_p90_s": pct([s["ttft_s"] for s in samples], 0.9),
        "total_p50_s": pct([s["total_s"] for s in samples], 0.5),
        "total_p90_s": pct([s["total_s"] for s in samples], 0.9),
        "sentence_p50_s": pct([s["sentence_s"] for s in samples], 0.5),
        "decode_tps_median": round(statistics.median([s["decode_tps"] for s in samples if s["decode_tps"]]), 2)
        if any(s["decode_tps"] for s in samples) else None,
        "completion_tokens_median": pct([s["completion_tokens"] for s in samples], 0.5),
        "prompt_tokens_median": pct([s["prompt_tokens"] for s in samples], 0.5),
    }


def summarize_router(probes: list[dict]) -> dict:
    dts = [p["dt_ms"] for p in probes]
    return {"n": len(dts), "p50_ms": round(statistics.median(dts), 2),
            "p95_ms": round(sorted(dts)[int(0.95 * (len(dts) - 1))], 2),
            "min_ms": round(min(dts), 2), "max_ms": round(max(dts), 2)}


def measure_config(name: str, extra: list[str], port: int, log_name: str) -> dict:
    print(f"--- {name}: {' '.join(COMMON + extra)}", flush=True)
    vram_before = vram_used_mib()
    proc = start_server(extra, log_name, port)
    try:
        time.sleep(2)
        vram_loaded = vram_used_mib()
        props = requests.get(f"{URL}/props", timeout=10).json()
        n_ctx_slot = int(props.get("default_generation_settings", {}).get("n_ctx_slot") or 0)
        system, tool = load_system_prompt()
        router_system = load_router_system()
        router_slot = 1 if "-np" not in extra or extra[extra.index("-np") + 1] != "1" else 0
        chat_slot = 0
        warmup = [{"role": "system", "content": system}, {"role": "user", "content": "Dzień dobry."}]
        chat_stream(warmup, tool, chat_slot, max_tokens=8)  # engine warm-up, discard

        raw = raw_probe(chat_slot, max_prompt_tokens=min(1400, n_ctx_slot - 250) if n_ctx_slot else 1400)
        router = summarize_router(router_probe(router_system, router_slot))

        items = load_quality_items()
        cold_samples, warm_samples = [], []
        for item in items:
            marker = f"[sesja {uuid.uuid4().hex}]\n"
            msgs_cold = [{"role": "system", "content": marker + system},
                         *history_msgs(0), {"role": "user", "content": item["text"]}]
            cold_samples.append(chat_stream(msgs_cold, tool, chat_slot))
            msgs_warm = [{"role": "system", "content": system},
                         *history_msgs(0), {"role": "user", "content": item["text"]}]
            for _ in range(3):
                warm_samples.append(chat_stream(msgs_warm, tool, chat_slot))

        # longer history shape (8 turns ~= mid-conversation, still under the 24-turn cap)
        hist_cold, hist_warm = [], []
        for item in items[:2]:
            marker = f"[sesja {uuid.uuid4().hex}]\n"
            msgs = [{"role": "system", "content": marker + system}, *history_msgs(8),
                    {"role": "user", "content": item["text"]}]
            hist_cold.append(chat_stream(msgs, tool, chat_slot))
            msgs = [{"role": "system", "content": system}, *history_msgs(8),
                    {"role": "user", "content": item["text"]}]
            for _ in range(3):
                hist_warm.append(chat_stream(msgs, tool, chat_slot))

        result = {"name": name, "args": COMMON + extra, "port": port,
                  "n_ctx_slot": n_ctx_slot,
                  "vram_mib_before_start": vram_before, "vram_mib_loaded": vram_loaded,
                  "vram_mib_delta": vram_loaded - vram_before, "load_1min": loadavg_1min(),
                  "raw": raw, "router": router,
                  "chat_cold": summarize_chat(cold_samples), "chat_warm": summarize_chat(warm_samples),
                  "chat_hist8_cold": summarize_chat(hist_cold),
                  "chat_hist8_warm": summarize_chat(hist_warm),
                  "cold_samples": cold_samples, "warm_samples": warm_samples,
                  "hist8_cold_samples": hist_cold, "hist8_warm_samples": hist_warm,
                  "router_probes": router}
    finally:
        stop_server(proc)
    r = result
    print(f"    pp={r['raw']['pp_tps']:.1f} tg={r['raw']['tg_tps']:.1f} "
          f"ttft cold={r['chat_cold']['ttft_p50_s']} warm={r['chat_warm']['ttft_p50_s']} "
          f"total warm p50={r['chat_warm']['total_p50_s']} router p50={r['router']['p50_ms']}ms "
          f"vram_delta={r['vram_mib_delta']}MiB", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="comma-separated CONFIGS keys")
    ap.add_argument("--port", type=int, default=8797)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    set_url(args.port)
    RESULTS.mkdir(parents=True, exist_ok=True)
    names = args.configs.split(",")
    out_path = RESULTS / f"sweep_{args.tag}.json"
    results = []
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))["configs"]
        done = {r["name"] for r in results}
        names = [n for n in names if n not in done]
    for name in names:
        results.append(measure_config(name, CONFIGS[name], args.port, f"speed_{args.tag}_{name}.log"))
        out_path.write_text(json.dumps({"gguf": GGUF.name, "port": args.port, "configs": results},
                                       ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
