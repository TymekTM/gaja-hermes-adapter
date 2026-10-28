#!/usr/bin/env python3
"""Run the 30 quality prompts through llama-server with a given system prompt and record outputs and automatic checks.

Manual 1-5 scores are added afterwards in bench/results/llm_quality_scores_<tag>.json.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_server as S  # noqa: E402

RESULTS = S.ROOT / "bench/results"
QUALITY = S.ROOT / "bench/data/quality.jsonl"
URL = S.URL + "/v1/chat/completions"

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F000-\U0001F2FF]")
MARKDOWN = re.compile(r"(\*\*|__|^#{1,6}\s|^\s*[-*+]\s|^\s*\d+[.)]\s|`|\[[^\]]+\]\([^)]+\))", re.M)
DIGITS = re.compile(r"\d")


def load_prompt(path):
    text = Path(path).read_text(encoding="utf-8")
    system = re.search(r"## System\n\n```text\n(.*?)\n```", text, re.S).group(1)
    tool = json.loads(re.search(r"## Narzędzie\n\n```json\n(.*?)\n```", text, re.S).group(1))
    return system, tool


def sentence_count(text):
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return len(parts)


def checks(text):
    return {
        "sentences": sentence_count(text),
        "markdown": bool(MARKDOWN.search(text)),
        "emoji": bool(EMOJI.search(text)),
        "digits": bool(DIGITS.search(text)),
        "think_leak": "</think>" in text or "<think>" in text,
        "chars": len(text),
    }


def chat(system, tool, item, temp, top_k=None, repeat_penalty=None, max_tokens=200):
    msgs = [{"role": "system", "content": system}, *item["history"], {"role": "user", "content": item["text"]}]
    body = {"messages": msgs, "tools": [tool], "temperature": temp, "seed": 42, "max_tokens": max_tokens, "stream": False}
    if top_k:
        body["top_k"] = top_k
    if repeat_penalty:
        body["repeat_penalty"] = repeat_penalty
    t0 = time.time()
    resp = requests.post(URL, json=body, timeout=300)
    dt = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").split("</think>")[-1].strip()
    calls = [c["function"] for c in (msg.get("tool_calls") or [])]
    reasoning = msg.get("reasoning_content") or ""
    return {"content": content, "reasoning_chars": len(reasoning), "tool_calls": calls, "latency_s": round(dt, 3),
            "completion_tokens": data["usage"]["completion_tokens"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompt", default=str(S.ROOT / "prompts/system_pl.md"))
    ap.add_argument("--variant", default="budget0", choices=list(S.VARIANTS))
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--repeat-penalty", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=200,
                    help="limit generacji; przy budgetach reasoning >0 ustaw ~200+budget")
    args = ap.parse_args()

    import os
    reuse = os.environ.get("BENCH_REUSE_URL", "")
    global URL
    if reuse:
        URL = reuse.rstrip("/") + "/v1/chat/completions"

    system, tool = load_prompt(args.prompt)
    items = [json.loads(line) for line in QUALITY.read_text(encoding="utf-8").splitlines() if line.strip()]
    RESULTS.mkdir(parents=True, exist_ok=True)
    server_extra = S.VARIANTS[args.variant]
    proc = None
    if not reuse:
        proc = S.start(server_extra, f"quality_server_{args.tag}.log")
    rows = []
    try:
        for item in items:
            out = chat(system, tool, item, args.temp, args.top_k, args.repeat_penalty, args.max_tokens)
            rows.append({"id": item["id"], "lang": item["lang"], "category": item["category"],
                         "text": item["text"], **out, "checks": checks(out["content"])})
            print(f"{item['id']}: {out['content'][:90]!r} tools={len(out['tool_calls'])}", flush=True)
    finally:
        if proc is not None:
            S.stop(proc)
    summary = {
        "n": len(rows),
        "tool_calls": sum(1 for r in rows if r["tool_calls"]),
        "think_leaks": sum(1 for r in rows if r["checks"]["think_leak"]),
        "markdown": sum(1 for r in rows if r["checks"]["markdown"]),
        "emoji": sum(1 for r in rows if r["checks"]["emoji"]),
        "digits": sum(1 for r in rows if r["checks"]["digits"]),
        "over_3_sentences": sum(1 for r in rows if r["checks"]["sentences"] > 3),
        "empty": sum(1 for r in rows if not r["content"] and not r["tool_calls"]),
        "reasoning_channel": sum(1 for r in rows if r["reasoning_chars"]),
        "hit_max_tokens": sum(1 for r in rows if r["completion_tokens"] >= args.max_tokens),
    }
    out = {"tag": args.tag, "prompt": str(Path(args.prompt).resolve()), "summary": summary,
           "temp": args.temp, "seed": 42, "server_args": S.BASE_ARGS + server_extra, "gguf": S.GGUF.name,
           "loadavg_1min_end": S.loadavg_1min(), "rows": rows}
    (RESULTS / f"llm_quality_{args.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
