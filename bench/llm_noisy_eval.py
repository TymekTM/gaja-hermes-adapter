#!/usr/bin/env python3
"""Evaluation of Gaja's local LLM and router on clean and ASR-noisy Polish speech turns.

Measures:
- Route accuracy (local/reply vs tool vs hermes)
- Tool selection accuracy (tool name + required arguments)
- Polish-only rate (heuristic test for English intrusion or drift)
- Think-leak rate (<think> / </think> tag leakage)
- Max-token hit rate (finish_reason == "length")
- Latency and token statistics

Saves output to bench/results/llm_noisy_<date>.json (gitignored).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import os
import re
import statistics
import sys
import time
import tomllib
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Safe dummy credentials for HA REST API probe (rules-oc-3: never call live APIs)
os.environ.setdefault("HASS_URL", "http://127.0.0.1:1")
os.environ.setdefault("HASS_TOKEN", "dummy_probe_token")

from gaja.config import Settings
from gaja.llm.client import LLMClient
from gaja.llm.prompt import load_system_prompt, load_text_block
from gaja.llm.router import (
    Router,
    home_device_words,
    home_person_words,
    set_home_device_words,
    set_home_person_words,
    template_router_prompt,
)
from gaja.text.normalize import normalize_for_tts
from gaja.tools.registry import build_runtime_registry
from gaja.tools.time_tool import PL_MONTHS, PL_WEEKDAYS

log = logging.getLogger("llm_noisy_eval")

DATA_PATH = ROOT / "bench/data/llm_pl_noisy.jsonl"
RESULTS_DIR = ROOT / "bench/results"
HOME_EXAMPLE = ROOT / "docs/home.example.toml"

# English stop words for the language detection heuristic
EN_STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "all", "an", "and", "any", "are", "as", "at", "be", "been", "but", "by",
    "can", "could", "day", "did", "do", "does", "down", "each", "else", "find", "first",
    "for", "from", "get", "go", "had", "has", "have", "he", "help", "her", "here", "him",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "know", "let", "like",
    "look", "make", "many", "may", "me", "more", "most", "my", "need", "no", "not", "now",
    "number", "of", "on", "one", "or", "other", "out", "over", "people", "please", "said",
    "see", "she", "should", "side", "so", "some", "sorry", "sure", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "thing", "this", "time", "to", "two",
    "up", "use", "was", "water", "way", "we", "were", "what", "when", "which", "who", "will",
    "with", "word", "would", "write", "yeah", "yes", "you", "your"
})

# Polish function words
PL_STOPWORDS: frozenset[str] = frozenset({
    "i", "w", "z", "ze", "na", "do", "nie", "to", "że", "się", "o", "a", "jak", "ale", "co",
    "po", "dla", "od", "przez", "tak", "czy", "za", "jest", "są", "być", "będzie", "będą",
    "ma", "mają", "mi", "mnie", "cię", "go", "ją", "nas", "was", "ich", "mu", "jej", "im",
    "ten", "ta", "te", "to", "tym", "tego", "tej", "tych", "tam", "tu", "tutaj", "teraz",
    "już", "jeszcze", "tylko", "bardzo", "może", "gdy", "kiedy", "jeśli", "jeżeli", "albo",
    "lub", "ani", "oraz", "więc", "dlatego", "bo", "żeby", "aby", "proszę", "przepraszam",
    "jasne", "dobrze", "chwilę", "chwilkę", "mam", "masz", "mamy", "macie", "mogę", "możesz"
})

PL_CHARS: frozenset[str] = frozenset("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")


def is_english_sentence(sentence: str) -> bool:
    """Classify whether an individual sentence is English using stop-word ratio.
    
    Tokens are extracted (alphabetic + Polish diacritics). If tokens < 2, returns False.
    Computes English stop-word ratio and compares against Polish stop-words and Polish diacritics.
    """
    tokens = re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", sentence.lower())
    if len(tokens) < 2:
        return False
    has_pl_char = any(any(c in PL_CHARS for c in t) for t in tokens)
    en_matches = sum(1 for t in tokens if t in EN_STOPWORDS)
    pl_matches = sum(1 for t in tokens if t in PL_STOPWORDS)
    en_ratio = en_matches / len(tokens)
    
    if has_pl_char:
        return en_ratio >= 0.6 and en_matches > pl_matches * 2
    return en_matches >= 2 and en_ratio >= 0.35 and en_matches > pl_matches


def contains_english_sentence(text: str) -> bool:
    """Returns True if any sentence in text is classified as English.
    
    Strips XML-style tags (such as <think> and </think>) before splitting into sentences.
    """
    cleaned = re.sub(r"</?[a-zA-Z0-9_-]+>", " ", text).strip()
    sentences = re.split(r"[.!?\n]+", cleaned)
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if is_english_sentence(s):
            return True
    return False


def build_production_system_prompt(prompt_path: Path) -> str:
    """Construct production system prompt with today's date formatted as in Services.get_system_prompt."""
    p = load_system_prompt(prompt_path)
    base = p.text
    now = datetime.datetime.now(ZoneInfo("Europe/Warsaw"))
    today = f"{PL_WEEKDAYS[now.weekday()]}, {now.day} {PL_MONTHS[now.month]} {now.year} roku"
    spoken_today = normalize_for_tts(today, "pl")
    return base + f"\n\nDzisiejsza data (Warszawa): {today} ({spoken_today})."


def load_dataset(path: Path) -> list[dict[str, Any]]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def evaluate_case_accuracy(case: dict[str, Any], result: dict[str, Any]) -> tuple[bool, bool]:
    """Returns (route_correct, tool_correct)."""
    expect = case["expect"]
    expected_kind = expect["kind"]  # "tool" | "reply" | "hermes"
    route_chosen = result["route_chosen"]  # "tool" | "reply" | "hermes" | "local"
    
    route_correct = False
    if expected_kind == "tool":
        route_correct = (route_chosen == "tool")
    elif expected_kind == "reply":
        route_correct = (route_chosen in ("reply", "local"))
    elif expected_kind == "hermes":
        route_correct = (route_chosen == "hermes")

    tool_correct = False
    if expected_kind == "tool":
        expected_tool = expect.get("tool")
        called_tool = result.get("tool_called")
        called_args = result.get("tool_arguments") or {}
        if called_tool == expected_tool:
            expected_args = expect.get("args_contain") or {}
            if all(called_args.get(k) == v for k, v in expected_args.items()):
                tool_correct = True
    else:
        # For non-tool cases, tool is "correct" (i.e. no false tool call) if no tool called
        tool_correct = (result.get("tool_called") is None)
        
    return route_correct, tool_correct


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "count": 0,
            "route_accuracy": 0.0,
            "tool_accuracy": 0.0,
            "tool_cases_count": 0,
            "false_tool_rate": 0.0,
            "polish_only_rate": 0.0,
            "think_leak_rate": 0.0,
            "max_token_hit_rate": 0.0,
            "mean_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "mean_completion_tokens": 0.0,
        }
    n = len(results)
    route_correct_count = sum(1 for r in results if r["route_correct"])
    
    tool_cases = [r for r in results if r["case"]["expect"]["kind"] == "tool"]
    n_tool_cases = len(tool_cases)
    tool_correct_count = sum(1 for r in tool_cases if r["tool_correct"])
    
    non_tool_cases = [r for r in results if r["case"]["expect"]["kind"] != "tool"]
    false_tool_count = sum(1 for r in non_tool_cases if r.get("tool_called") is not None)
    
    polish_only_count = sum(1 for r in results if not r["has_english"])
    think_leak_count = sum(1 for r in results if r["has_think"])
    max_token_count = sum(1 for r in results if r["finish_reason"] == "length")
    
    latencies = [r["latency_ms"] for r in results]
    tokens = [r["completion_tokens"] for r in results]
    
    return {
        "count": n,
        "route_accuracy": round(route_correct_count / n * 100.0, 1),
        "tool_accuracy": round(tool_correct_count / n_tool_cases * 100.0, 1) if n_tool_cases else 100.0,
        "tool_cases_count": n_tool_cases,
        "false_tool_rate": round(false_tool_count / len(non_tool_cases) * 100.0, 1) if non_tool_cases else 0.0,
        "polish_only_rate": round(polish_only_count / n * 100.0, 1),
        "think_leak_rate": round(think_leak_count / n * 100.0, 1),
        "max_token_hit_rate": round(max_token_count / n * 100.0, 1),
        "mean_latency_ms": round(statistics.mean(latencies), 1),
        "median_latency_ms": round(statistics.median(latencies), 1),
        "p95_latency_ms": round(float(statistics.quantiles(latencies, n=20)[18]) if len(latencies) >= 20 else max(latencies), 1),
        "mean_completion_tokens": round(statistics.mean(tokens), 1),
    }


async def run_evaluation(
    data_path: Path,
    output_path: Path | None = None,
    limit: int | None = None,
    pause_s: float = 0.5,
) -> dict[str, Any]:
    settings = Settings()
    
    # Load example household aliases
    if HOME_EXAMPLE.exists():
        with open(HOME_EXAMPLE, "rb") as f:
            example_data = tomllib.load(f)
        settings.home.aliases = example_data.get("aliases", {})
        
    set_home_person_words(home_person_words(settings.home.aliases))
    set_home_device_words(home_device_words(settings.home.aliases))

    registry = build_runtime_registry(settings)
    tool_schemas = registry.schemas()
    system_prompt = build_production_system_prompt(ROOT / settings.llm.system_prompt_path)

    llm_client = LLMClient(settings.llm, tool_schemas)
    
    router = Router(
        llm_client,
        template_router_prompt(
            load_text_block(ROOT / settings.router.prompt_path),
            set(registry.names()),
        ),
        threshold=settings.router.threshold,
        history_turns=settings.router.history_turns,
        timeout_s=settings.router.timeout_s,
        slot=settings.llm.router_slot,
        tools=tool_schemas,
        tool_threshold=settings.router.tool_threshold,
        spotify_ready=lambda: True,
    )

    cases = load_dataset(data_path)
    if limit is not None:
        cases = cases[:limit]

    print(f"Loaded {len(cases)} cases from {data_path}")
    print(f"Base URL: {settings.llm.base_url}, Model: {settings.llm.model}")
    print(f"Pause between requests: {pause_s}s (concurrency 1)")

    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(base_url=settings.llm.base_url, timeout=settings.llm.timeout_s) as http:
        for idx, case in enumerate(cases, 1):
            utterance = case["utterance"]
            
            # 1. Run router classification & tool selection
            router_call = None
            try:
                route_res = await router.classify(utterance, [])
                router_kind = route_res.kind
                if router_kind == "tool":
                    try:
                        router_call = await router.select_tool(utterance, [])
                    except Exception as exc:
                        log.warning("select_tool error on case %s: %s", case["id"], exc)
                        router_call = None
            except Exception as exc:
                log.warning("Router classify failed on case %s: %s", case["id"], exc)
                router_kind = "local"
                router_call = None

            # Determine initial router route
            if router_kind == "tool" and router_call is not None:
                router_route = "tool"
                router_tool = router_call.name
                router_args = router_call.arguments
            elif router_kind == "hermes":
                router_route = "hermes"
                router_tool = None
                router_args = None
            else:
                router_route = "local"
                router_tool = None
                router_args = None

            # 2. Run LLM non-streaming /v1/chat/completions (production payload)
            payload: dict[str, Any] = {
                "model": settings.llm.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": utterance},
                ],
                "stream": False,
                "temperature": 0,
                "max_tokens": settings.llm.max_tokens,
                "cache_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
                "tools": tool_schemas,
            }
            if settings.llm.chat_slot >= 0:
                payload["id_slot"] = settings.llm.chat_slot

            t0 = time.perf_counter()
            resp = await http.post("/v1/chat/completions", json=payload)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

            resp_data = resp.json()
            choice = resp_data["choices"][0]
            finish_reason = choice.get("finish_reason", "stop")
            message = choice.get("message", {})
            content = message.get("content") or ""
            llm_tool_calls = message.get("tool_calls") or []
            usage = resp_data.get("usage") or {}
            completion_tokens = usage.get("completion_tokens", 0)

            # Analyze LLM tool calls
            llm_tool_called = None
            llm_tool_args = None
            if llm_tool_calls:
                first_call = llm_tool_calls[0].get("function", {})
                llm_tool_called = first_call.get("name")
                raw_args = first_call.get("arguments", "{}")
                try:
                    llm_tool_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    llm_tool_args = {"raw": raw_args}

            # Content checks
            # With enable_thinking off LFM3 starts every reply with a stray "</think>", which the client's
            # ThinkFilter drops; only an opened block or a further close would reach the speaker.
            has_think = "<think>" in content or content.count("</think>") > int(content.startswith("</think>"))
            has_english = contains_english_sentence(content)

            # Determine end-to-end production route and tool:
            # If router chose hermes -> hermes
            # If router chose tool with valid call -> tool
            # If router chose local -> LLM decides: if LLM calls handoff_to_hermes -> hermes,
            # if LLM calls a tool -> tool, else reply/local.
            if router_route == "hermes":
                route_chosen = "hermes"
                tool_called = None
                tool_args = None
            elif router_route == "tool":
                route_chosen = "tool"
                tool_called = router_tool
                tool_args = router_args
            else:
                # Local reply path in pipeline
                if llm_tool_called == "handoff_to_hermes":
                    route_chosen = "hermes"
                    tool_called = None
                    tool_args = None
                elif llm_tool_called is not None:
                    route_chosen = "tool"
                    tool_called = llm_tool_called
                    tool_args = llm_tool_args
                else:
                    route_chosen = "reply"
                    tool_called = None
                    tool_args = None

            case_result: dict[str, Any] = {
                "id": case["id"],
                "utterance": utterance,
                "clean": case["clean"],
                "noise": case["noise"],
                "case": case,
                "route_chosen": route_chosen,
                "tool_called": tool_called,
                "tool_arguments": tool_args,
                "router_route": router_route,
                "router_tool": router_tool,
                "llm_tool_called": llm_tool_called,
                "llm_tool_arguments": llm_tool_args,
                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
                "content": content,
                "has_think": has_think,
                "has_english": has_english,
                "latency_ms": round(latency_ms, 1),
            }

            route_correct, tool_correct = evaluate_case_accuracy(case, case_result)
            case_result["route_correct"] = route_correct
            case_result["tool_correct"] = tool_correct
            results.append(case_result)

            if idx % 20 == 0 or idx == len(cases):
                print(f"[{idx}/{len(cases)}] Case {case['id']}: route={route_chosen} (correct={route_correct}), tool={tool_called} (correct={tool_correct}), latency={latency_ms:.0f}ms")

            # Pause between requests (concurrency 1)
            if pause_s > 0:
                await asyncio.sleep(pause_s)

    # Compute aggregate metrics
    clean_results = [r for r in results if r["noise"] == "clean"]
    asr_results = [r for r in results if r["noise"] == "asr"]

    summary = {
        "date": datetime.date.today().isoformat(),
        "timestamp": datetime.datetime.now(ZoneInfo("Europe/Warsaw")).isoformat(),
        "model": settings.llm.model,
        "overall": compute_metrics(results),
        "clean": compute_metrics(clean_results),
        "asr": compute_metrics(asr_results),
        "results": results,
    }

    if output_path is None:
        today_str = datetime.date.today().isoformat()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULTS_DIR / f"llm_noisy_{today_str}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved evaluation results to {output_path}")
    print("\n--- Summary ---")
    print(f"Overall cases: {summary['overall']['count']}")
    print(f"Route Accuracy:   overall={summary['overall']['route_accuracy']}%, clean={summary['clean']['route_accuracy']}%, asr={summary['asr']['route_accuracy']}%")
    print(f"Tool Accuracy:    overall={summary['overall']['tool_accuracy']}%, clean={summary['clean']['tool_accuracy']}%, asr={summary['asr']['tool_accuracy']}%")
    print(f"Polish-Only Rate: overall={summary['overall']['polish_only_rate']}%, clean={summary['clean']['polish_only_rate']}%, asr={summary['asr']['polish_only_rate']}%")
    print(f"Think-Leak Rate:  overall={summary['overall']['think_leak_rate']}%, clean={summary['clean']['think_leak_rate']}%, asr={summary['asr']['think_leak_rate']}%")
    print(f"Max-Token Rate:   overall={summary['overall']['max_token_hit_rate']}%, clean={summary['clean']['max_token_hit_rate']}%, asr={summary['asr']['max_token_hit_rate']}%")
    print(f"Mean Latency:     overall={summary['overall']['mean_latency_ms']}ms, clean={summary['clean']['mean_latency_ms']}ms, asr={summary['asr']['mean_latency_ms']}ms")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM on noisy Polish speech turns.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to evaluation jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Path to save json result")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases (for testing)")
    parser.add_argument("--pause", type=float, default=0.5, help="Pause in seconds between requests (default 0.5s)")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.data, output_path=args.output, limit=args.limit, pause_s=args.pause))


if __name__ == "__main__":
    main()
