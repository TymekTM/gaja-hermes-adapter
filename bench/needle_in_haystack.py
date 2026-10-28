#!/usr/bin/env python3
"""Needle-in-a-haystack po polsku dla modelu z llama-server (domyślnie LFM3-1.7B).

Siatka: dlugosc kontekstu x glebokosc igly x typ igly. Haystack to prawdziwa
polska proza (Wolne Lektury, bench/data/niah/*.txt). Wyniki: JSON + heatmap HTML.

Uruchamiać pod flock /tmp/gaja-bench.lock (konwencja repo), np.:

  flock /tmp/gaja-bench.lock uv run python bench/needle_in_haystack.py \
      --tag lfm3_q4 --ctx-list 4096,8192,16384,32768

Serwer dla kazdego rozmiaru kontekstu startuje osobno (-c <ctx>) na porcie 8794.
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_server as S  # noqa: E402

RESULTS = S.ROOT / "bench/results"
FILLER_DIR = S.ROOT / "bench/data/niah"
URL = S.URL + "/v1/chat/completions"

SYSTEM = ("Jesteś precyzyjnym asystentem archiwum. Odpowiadaj wyłącznie na podstawie "
          "podanego dokumentu, krótko, po polsku.")

WORDS = ["bursztynowy", "wiklinowy", "szmaragdowy", "gołębi", "termiczny", "purpurowy",
         "lniany", "koralowy", "atramentowy", "miedziany"]

# (typ, szablon igly, pytanie, generator wartosci, ekstraktor odpowiedzi)
def gen_needle(ctype, rng):
    if ctype == "code":
        val = f"XK-{rng.randint(1000, 9999)}"
        needle = f"Wiadomość administracyjna: sekretny kod dostępu do serwera to {val}."
        q = "Jaki jest sekretny kod dostępu do serwera? Podaj go w formacie XK-0000."
    elif ctype == "task":
        val = f"{rng.randint(10, 23):02d}:{rng.choice(['00', '15', '30', '45'])}"
        needle = f"Przypomnienie z harmonogramu: o godzinie {val} uruchom kopię zapasową bazy danych."
        q = "O której godzinie uruchomić kopię zapasową bazy danych? Zapisz cyframi w formacie GG:MM."
    else:  # word
        val = rng.choice(WORDS)
        needle = f"Uwaga: hasło dzisiejszego rajdu nocnego to «{val}»."
        q = "Jakie jest hasło dzisiejszego rajdu nocnego? Podaj samo hasło."
    return needle, q, val


def load_filler():
    parts = []
    for path in sorted(FILLER_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r"\s+", " ", text)
        parts.extend(s.strip() for s in re.split(r"(?<=[.!?]) ", text) if 40 < len(s.strip()) < 400)
    return parts


def tokenize_count(text):
    resp = requests.post(S.URL + "/tokenize", json={"content": text}, timeout=300)
    resp.raise_for_status()
    return len(resp.json()["tokens"])


def build_haystack(sentences, target_tokens, rng):
    """Zwija korpus w ciag o dlugosci ~target_tokens tokenow, zaczynajac od losowego offsetu."""
    out, n = [], 0
    idx = rng.randrange(len(sentences))
    step = 0
    while n < target_tokens and step < len(sentences) * 2:
        s = sentences[(idx + step) % len(sentences)]
        out.append(s)
        n += max(1, len(s) // 4)  # szacunek ~4 znaki/token dla polskiego
        step += 1
    text = " ".join(out)
    # korekta do celu po fakcie (dokladna tokenizacja)
    for _ in range(40):
        real = tokenize_count(text)
        if real <= target_tokens:
            break
        # ucinaj ~nadmiar proporcjonalnie
        cut = int(len(text) * (real - target_tokens) / real) + 8
        text = text[:-cut].rsplit(" ", 1)[0] + "."
    return text


def chat_stream(messages, max_tokens=64):
    body = {"messages": messages, "temperature": 0, "top_k": 1, "seed": 42,
            "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    text, timings = "", None
    with requests.post(URL, json=body, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            ev = json.loads(payload)
            timings = ev.get("timings") or timings
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                text += piece
    return text.split("</think>")[-1].strip(), timings, time.perf_counter() - t0


def vram_used_mib():
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def run_ctx(ctx, depths, types, reps, sentences, variant_args, kv_args, tag, done=None):
    print(f"=== ctx {ctx} ===", flush=True)
    if done is not None and all((ctx, d, t, r) in done for d in depths for t in types for r in range(reps)):
        print("  (komplet w JSONL — pomijam)", flush=True)
        return [], None
    base = ["-ngl", "99", "-c", str(ctx), "-np", "1", "--jinja", "-fa", "on", *kv_args]
    proc = S.start(variant_args, f"niah_server_{ctx}.log", base_args=base)
    rows, vram_loaded = [], None
    jsonl = RESULTS / f"niah_{tag}.jsonl"
    try:
        time.sleep(2)
        vram_loaded = vram_used_mib()
        overhead = 220  # system+pytanie+szablon+rezerva na odpowiedz
        target = ctx - overhead
        for depth in depths:
            for ctype in types:
                for rep in range(reps):
                    if done is not None and (ctx, depth, ctype, rep) in done:
                        continue
                    ctype_idx = ["code", "task", "word"].index(ctype)
                    rng = random.Random(42 + rep * 1000 + int(depth) * 10 + ctype_idx * 7)
                    needle, q, val = gen_needle(ctype, rng)
                    hay = build_haystack(sentences, target - tokenize_count(needle), rng)
                    pos = int(len(hay) * depth / 100)
                    if pos > 0:
                        pos = hay.find(" ", pos) + 1 or pos
                    doc = hay[:pos] + " " + needle + " " + hay[pos:] if pos else needle + " " + hay
                    messages = [{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": doc + "\n\n" + q}]
                    prompt_tokens = tokenize_count(doc)
                    answer, timings, wall = chat_stream(messages)
                    hit = val.lower() in answer.lower()
                    row = {"ctx": ctx, "depth_pct": depth, "type": ctype, "rep": rep,
                           "value": val, "hit": hit, "answer": answer[:200],
                           "prompt_tokens": prompt_tokens,
                           "prompt_ms": round((timings or {}).get("prompt_ms", 0), 1),
                           "prompt_tps": round((timings or {}).get("prompt_per_second", 0), 1),
                           "wall_s": round(wall, 2)}
                    rows.append(row)
                    with jsonl.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"  ctx={ctx} d={depth:>3}% {ctype:<4} val={val:<10} "
                          f"{'HIT ' if hit else 'MISS'} prompt={prompt_tokens} "
                          f"prefill={row['prompt_tps']} t/s", flush=True)
    finally:
        S.stop(proc)
    return rows, vram_loaded


def summarize(rows):
    grid = {}
    for r in rows:
        key = (r["ctx"], r["depth_pct"])
        grid.setdefault(key, []).append(r["hit"])
    return {f"{c}|{d}": {"hits": sum(v), "n": len(v), "acc": round(sum(v) / len(v), 3)}
            for (c, d), v in sorted(grid.items())}


def heatmap_html(summary, types_acc, tag, meta):
    ctxs = sorted({int(k.split("|")[0]) for k in summary})
    depths = sorted({int(k.split("|")[1]) for k in summary})
    cells = ""
    for c in ctxs:
        row = f"<tr><th>{c//1024}K</th>"
        for d in depths:
            s = summary.get(f"{c}|{d}", {})
            acc = s.get("acc", 0)
            color = f"hsl({int(acc*120)},70%,45%)" if s else "#333"
            row += (f'<td style="background:{color};color:#fff;text-align:center;'
                    f'padding:6px 10px">{s.get("hits",0)}/{s.get("n",0)}</td>')
        cells += row + "</tr>"
    head = "".join(f"<th>{d}%</th>" for d in depths)
    types = "".join(f"<li>{t}: {a:.0%}</li>" for t, a in types_acc.items())
    return f"""<!doctype html><html lang="pl"><head><meta charset="utf-8">
<title>NIAH {tag}</title><style>body{{font-family:system-ui;margin:24px}}
table{{border-collapse:collapse}}th,td{{border:1px solid #999}}</style></head><body>
<h1>Needle-in-a-haystack — {tag}</h1>
<p>{meta}</p>
<h2>Trafność wg typu igły</h2><ul>{types}</ul>
<h2>Heatmapa (kontekst × głębokość)</h2>
<table><tr><th>ctx \\ depth</th>{head}</tr>{cells}</table>
<p>Zielony = pełne trafienia, czerwony = pudła. Komórka: trafienia/liczba prób.</p>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ctx-list", default="4096,8192,16384,32768")
    ap.add_argument("--depths", default="0,10,25,50,75,90,100")
    ap.add_argument("--types", default="code,task,word")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--variant", default="lfm3_nothink", choices=list(S.VARIANTS))
    ap.add_argument("--kv", default="", help="np. --kv q8 (cache-type-k/v q8_0)")
    args = ap.parse_args()

    kv_args = []
    if args.kv == "q8":
        kv_args = ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]

    sentences = load_filler()
    print(f"filler: {len(sentences)} zdań", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    # resume: wczytaj wiersze z przerwanego przebiegu
    jsonl = RESULTS / f"niah_{args.tag}.jsonl"
    all_rows, done = [], set()
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                all_rows.append(r)
                done.add((r["ctx"], r["depth_pct"], r["type"], r["rep"]))
        print(f"resume: {len(all_rows)} wierszy z {jsonl.name}", flush=True)
    vram = {}
    for ctx in [int(x) for x in args.ctx_list.split(",")]:
        rows, vl = run_ctx(ctx, [int(d) for d in args.depths.split(",")], args.types.split(","),
                           args.reps, sentences, S.VARIANTS[args.variant], kv_args, args.tag, done)
        all_rows.extend(rows)
        vram[str(ctx)] = vl

    total = len(all_rows)
    hits = sum(1 for r in all_rows if r["hit"])
    types_acc = {}
    for t in args.types.split(","):
        sub = [r for r in all_rows if r["type"] == t]
        types_acc[t] = sum(1 for r in sub if r["hit"]) / len(sub) if sub else 0
    summary = summarize(all_rows)
    out = {"tag": args.tag, "gguf": S.GGUF.name, "variant": args.variant, "kv": args.kv,
           "reps": args.reps, "total": total, "hits": hits,
           "accuracy": round(hits / total, 3) if total else 0,
           "types_accuracy": types_acc, "grid": summary, "vram_mib": vram, "rows": all_rows,
           "server_args": S.BASE_ARGS}
    (RESULTS / f"niah_{args.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
    meta = (f"model: {S.GGUF.name}, variant: {args.variant}, kv: {args.kv or 'f16'}, "
            f"trafność ogólna: {hits}/{total}")
    (RESULTS / f"niah_{args.tag}.html").write_text(heatmap_html(summary, types_acc, args.tag, meta),
                                                   encoding="utf-8")
    print(json.dumps({"accuracy": out["accuracy"], "types": types_acc}, ensure_ascii=False), flush=True)
    print(f"wrote bench/results/niah_{args.tag}.json / .html", flush=True)


if __name__ == "__main__":
    main()
