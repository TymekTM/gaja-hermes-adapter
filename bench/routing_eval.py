#!/usr/bin/env python3
"""Routing eval: decide local vs hermes with LFM2.5 under strategies A-D.

A  native tool call handoff_to_hermes(task, reason)
B  grammar-constrained classifier {"route": ...}, then a local answer call if local
C  one call, grammar forces the reply to start with [LOCAL] or [HERMES]
D  strategy A with few-shot examples in the system prompt
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_server as S  # noqa: E402

ROOT = S.ROOT
DATA = ROOT / "bench/data/routing.jsonl"
PROMPT_MD = ROOT / "prompts/system_pl.md"
RESULTS = ROOT / "bench/results"
URL = S.URL + "/v1/chat/completions"

FEWSHOT = (
    "Przykłady (nie są częścią rozmowy):\n"
    "Rozmówca: ile to jest 9 razy 6. Gaja odpowiada sama: 54.\n"
    "Rozmówca: jak się masz. Gaja odpowiada sama.\n"
    "Rozmówca: jaka pogoda w Lublinie. Gaja wywołuje handoff_to_hermes.\n"
    "Rozmówca: wyślij Kacprowi sms, że się spóźnię. Gaja wywołuje handoff_to_hermes.\n"
    "Rozmówca: opowiedz dowcip. Gaja odpowiada sama.\n"
    "Rozmówca: uruchom kopię zapasową. Gaja wywołuje handoff_to_hermes."
)

C_MARKER = (
    "Każdą odpowiedź zacznij od znacznika. [LOCAL] oznacza, że odpowiadasz sama: po znaczniku "
    "napisz odpowiedź na głos. [HERMES] oznacza, że zadanie trzeba przekazać Hermesowi: po znaczniku "
    "napisz jedno pełne zdanie z opisem zadania, bez odpowiedzi dla rozmówcy. Nie pisz nic przed znacznikiem."
)
CLS_HEADER = (
    "Jesteś klasyfikatorem dla asystentki głosowej Gaja. Dla ostatniej wiadomości użytkownika zdecyduj: "
    "local, jeśli Gaja odpowiada sama, albo hermes, jeśli trzeba przekazać zadanie Hermesowi. "
    "Odpowiedz wyłącznie w formacie JSON."
)
CLS_SCHEMA = {
    "type": "object",
    "properties": {"route": {"type": "string", "enum": ["local", "hermes"]}},
    "required": ["route"],
}
GRAMMAR_C = r'root ::= ("[LOCAL]" | "[HERMES]") [^\x00]*'

PRETEND_PL = re.compile(
    r"(wysłał|wysłałam|wysłałem|ustawi[łl]|ustawiłam|zapisa[łl]|dodał[ae]?m|przypomnę|przypomniałam|"
    r"włączy[łl]|wyłączy[łl]|zamknął|zamknęł|zrobi[łl]|zrobiłam|sprawdzi[łl]|sprawdziłam|gotowe|"
    r"wysłane|załatwi|zapamiętam|zapamiętałam|zapisałem|zapisałam|wysyłam|ustawiam)",
    re.I,
)
PRETEND_EN = re.compile(
    r"(\bI'?ve (sent|set|added|turned|checked)|\bI (sent|set|added|turned|checked)\b|reminder is set|"
    r"\bsent\b|\bdone\b|turned (on|off)|added to your calendar)",
    re.I,
)
DATA_PL = re.compile(
    r"(\d+\s*(stopni|°|zł|procent|dolar|dolarów|USD|GB|%)|jest (słonecznie|pochmurno|deszczowo)|"
    r"\bpada\b|wynosi|kosztuje|kurs)",
    re.I,
)
DATA_EN = re.compile(r"(\d+\s*(degrees|°|\$|%|dollars)|sunny|cloudy|raining|costs|currently)", re.I)


def load_prompt():
    text = PROMPT_MD.read_text(encoding="utf-8")
    system = re.search(r"## System\n\n```text\n(.*?)\n```", text, re.S).group(1)
    tool = json.loads(re.search(r"## Narzędzie\n\n```json\n(.*?)\n```", text, re.S).group(1))
    parts = system.split("\n\n")
    return parts[0], parts[1], "\n\n".join(parts[2:]), tool


def load_items(subset, limit):
    items = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    if subset == "borderline":
        items = [i for i in items if i["borderline"]]
    return items[:limit] if limit else items


def chat(messages, temp, max_tokens, seed, tools=None, grammar=None, response_format=None):
    body = {"messages": messages, "temperature": temp, "max_tokens": max_tokens, "seed": seed, "stream": False}
    if tools:
        body["tools"] = tools
    if grammar:
        body["grammar"] = grammar
    if response_format:
        body["response_format"] = response_format
    t0 = time.time()
    resp = requests.post(URL, json=body, timeout=600)
    dt = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    return data, dt


def split_think(content):
    if "</think>" in content:
        return True, content.split("</think>")[-1].strip()
    return False, content.strip()


PROMISE_PL = re.compile(r"(przekaż|przekażę|przekażem|Hermes)", re.I)
PROMISE_EN = re.compile(r"(\bhand(ing)? (it |this )?(off|over)|\bHermes\b|\bpass(ing)? (it |this )?to Hermes)", re.I)


def flags(answer, lang):
    pretend = bool((PRETEND_PL if lang == "pl" else PRETEND_EN).search(answer))
    invented = bool((DATA_PL if lang == "pl" else DATA_EN).search(answer))
    return pretend, invented


def run_item(strategy, item, persona, policy, handoff, tool, temp, seed):
    history = item["history"]
    text = item["text"]
    user = {"role": "user", "content": text}
    out = {"pred_strict": None, "pred_lenient": None, "handoff_format": "none", "task": None,
           "answer": "", "think_leak": False, "latency_s": 0.0, "route_latency_s": None,
           "prompt_tokens": 0, "completion_tokens": 0, "error": None}
    try:
        if strategy in ("A", "D"):
            system = "\n\n".join([persona, policy, handoff]) + ("\n\n" + FEWSHOT if strategy == "D" else "")
            msgs = [{"role": "system", "content": system}, *history, user]
            data, dt = chat(msgs, temp, 200, seed, tools=[tool])
            m = data["choices"][0]["message"]
            leak, answer = split_think(m.get("content") or "")
            calls = [c for c in (m.get("tool_calls") or []) if c["function"]["name"] == "handoff_to_hermes"]
            if calls:
                out["pred_strict"] = "hermes"
                out["handoff_format"] = "tool_call"
                out["task"] = json.loads(calls[0]["function"]["arguments"]).get("task")
            else:
                out["pred_strict"] = "local"
            text_handoff = bool(re.search(r"handoff_to_hermes", answer, re.I))
            out["pred_lenient"] = "hermes" if (calls or text_handoff) else "local"
            if text_handoff and not calls:
                out["handoff_format"] = "text"
            out["answer"] = answer
            out["reasoning_chars"] = len(m.get("reasoning_content") or "")
            out["route_latency_s"] = dt
            out["latency_s"] = dt
            out["think_leak"] = leak
            out["prompt_tokens"] = data["usage"]["prompt_tokens"]
            out["completion_tokens"] = data["usage"]["completion_tokens"]
        elif strategy == "B":
            cls_system = CLS_HEADER + "\n\n" + policy + "\n\n" + handoff
            cls_msgs = [{"role": "system", "content": cls_system}, *history, user]
            rf = {"type": "json_schema", "json_schema": {"schema": CLS_SCHEMA}}
            data, dt1 = chat(cls_msgs, 0.0 if temp == 0 else temp, 16, seed, response_format=rf)
            route = json.loads(data["choices"][0]["message"]["content"])["route"]
            out["route_latency_s"] = dt1
            out["latency_s"] = dt1
            out["pred_strict"] = out["pred_lenient"] = route
            out["handoff_format"] = "json"
            out["prompt_tokens"] += data["usage"]["prompt_tokens"]
            out["completion_tokens"] += data["usage"]["completion_tokens"]
            if route == "local":
                ans_msgs = [{"role": "system", "content": persona + "\n\n" + policy}, *history, user]
                data2, dt2 = chat(ans_msgs, temp, 200, seed)
                leak, answer = split_think(data2["choices"][0]["message"].get("content") or "")
                out["answer"] = answer
                out["think_leak"] = leak
                out["latency_s"] = dt1 + dt2
                out["prompt_tokens"] += data2["usage"]["prompt_tokens"]
                out["completion_tokens"] += data2["usage"]["completion_tokens"]
        elif strategy == "C":
            c_system = persona + "\n\n" + policy + "\n\n" + C_MARKER
            msgs = [{"role": "system", "content": c_system}, *history, user]
            data, dt = chat(msgs, temp, 200, seed, grammar=GRAMMAR_C)
            content = (data["choices"][0]["message"].get("content") or "").strip()
            out["route_latency_s"] = dt
            out["latency_s"] = dt
            out["prompt_tokens"] = data["usage"]["prompt_tokens"]
            out["completion_tokens"] = data["usage"]["completion_tokens"]
            if content.startswith("[HERMES]"):
                out["pred_strict"] = out["pred_lenient"] = "hermes"
                out["handoff_format"] = "marker"
                out["task"] = content[len("[HERMES]"):].strip()
            elif content.startswith("[LOCAL]"):
                out["pred_strict"] = out["pred_lenient"] = "local"
                out["handoff_format"] = "marker"
                out["answer"] = content[len("[LOCAL]"):].strip()
            else:
                out["error"] = "no marker"
                out["pred_strict"] = out["pred_lenient"] = "local"
                out["answer"] = content
        else:
            raise ValueError(f"unknown strategy {strategy}")
    except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["pred_strict"] = out["pred_lenient"] = "error"
    pretend, invented = flags(out["answer"], item["lang"])
    out["promised_no_call"] = bool(out["pred_strict"] != "hermes" and (PROMISE_PL if item["lang"] == "pl" else PROMISE_EN).search(out["answer"]))
    out.update({"claims_action": pretend, "claims_data": invented})
    return out


def metrics(rows, key):
    tp = sum(1 for r in rows if r["label"] == "hermes" and r[key] == "hermes")
    fp = sum(1 for r in rows if r["label"] == "local" and r[key] == "hermes")
    fn = sum(1 for r in rows if r["label"] == "hermes" and r[key] != "hermes")
    tn = sum(1 for r in rows if r["label"] == "local" and r[key] != "hermes")
    n = len(rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(prec, 3),
            "recall": round(rec, 3), "f1": round(f1, 3), "accuracy": round((tp + tn) / n, 3) if n else 0.0}


def summarize(rows):
    def med(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.median(vals), 3) if vals else None

    summary = {}
    for key in ("pred_strict", "pred_lenient"):
        summary[key] = {
            "all": metrics(rows, key),
            "pl": metrics([r for r in rows if r["lang"] == "pl"], key),
            "en": metrics([r for r in rows if r["lang"] == "en"], key),
            "borderline": metrics([r for r in rows if r["borderline"]], key) if any(r["borderline"] for r in rows) else None,
        }
    summary["confusion_strict"] = {
        "label_local_pred_local": sum(1 for r in rows if r["label"] == "local" and r["pred_strict"] != "hermes"),
        "label_local_pred_hermes": sum(1 for r in rows if r["label"] == "local" and r["pred_strict"] == "hermes"),
        "label_hermes_pred_local": sum(1 for r in rows if r["label"] == "hermes" and r["pred_strict"] != "hermes"),
        "label_hermes_pred_hermes": sum(1 for r in rows if r["label"] == "hermes" and r["pred_strict"] == "hermes"),
    }
    summary["latency_s_median"] = {
        "all": med([r["latency_s"] for r in rows]),
        "pred_local": med([r["latency_s"] for r in rows if r["pred_strict"] != "hermes"]),
        "pred_hermes": med([r["latency_s"] for r in rows if r["pred_strict"] == "hermes"]),
        "route_decision": med([r["route_latency_s"] for r in rows]),
    }
    summary["errors"] = sum(1 for r in rows if r["error"])
    summary["think_leaks"] = sum(1 for r in rows if r["think_leak"])
    summary["false_local_claims_action"] = sum(1 for r in rows if r["label"] == "hermes" and r["pred_strict"] != "hermes" and r["claims_action"])
    summary["false_local_claims_data"] = sum(1 for r in rows if r["label"] == "hermes" and r["pred_strict"] != "hermes" and r["claims_data"])
    summary["false_local_total"] = sum(1 for r in rows if r["label"] == "hermes" and r["pred_strict"] != "hermes")
    summary["claims_action_any"] = sum(1 for r in rows if r["claims_action"])
    summary["claims_data_any"] = sum(1 for r in rows if r["claims_data"])
    summary["promised_no_call_any"] = sum(1 for r in rows if r["promised_no_call"])
    summary["reasoning_channel"] = sum(1 for r in rows if r.get("reasoning_chars"))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="budget0", choices=list(S.VARIANTS))
    ap.add_argument("--strategies", default="A,B,C,D")
    ap.add_argument("--temps", default="0")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--subset", default="all", choices=["all", "borderline"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    persona, policy, handoff, tool = load_prompt()
    items = load_items(args.subset, args.limit)
    RESULTS.mkdir(parents=True, exist_ok=True)
    proc = S.start(S.VARIANTS[args.variant], f"routing_server_{args.variant}.log")
    try:
        for strategy in args.strategies.split(","):
            for temp in [float(t) for t in args.temps.split(",")]:
                load_start = S.loadavg_1min()
                rows = []
                t_start = time.time()
                for rep in range(args.repeats):
                    for item in items:
                        seed = 1000 + rep
                        res = run_item(strategy, item, persona, policy, handoff, tool, temp, seed)
                        rows.append({"id": item["id"], "lang": item["lang"], "label": item["label"],
                                     "borderline": item["borderline"], "category": item["category"],
                                     "text": item["text"], "rep": rep, **res})
                load_end = S.loadavg_1min()
                summary = summarize(rows)
                name = f"routing_{args.variant}_{strategy}_t{temp}_r{args.repeats}_{args.subset}.json"
                payload = {
                    "meta": {"strategy": strategy, "variant": args.variant, "temp": temp, "repeats": args.repeats,
                             "subset": args.subset, "n_items": len(items), "loadavg_1min_start": load_start,
                             "loadavg_1min_end": load_end, "wall_s": round(time.time() - t_start, 1),
                             "server_args": S.BASE_ARGS + S.VARIANTS[args.variant], "gguf": S.GGUF.name},
                    "summary": summary,
                    "rows": rows,
                }
                (RESULTS / name).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
                s = summary["pred_strict"]["all"]
                print(f"{name}: acc={s['accuracy']} P={s['precision']} R={s['recall']} F1={s['f1']} "
                      f"FL={summary['false_local_total']} (action={summary['false_local_claims_action']}, "
                      f"data={summary['false_local_claims_data']}) median_s={summary['latency_s_median']['all']} "
                      f"errors={summary['errors']} think_leaks={summary['think_leaks']}", flush=True)
    finally:
        S.stop(proc)


if __name__ == "__main__":
    main()
