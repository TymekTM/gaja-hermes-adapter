#!/usr/bin/env python3
"""Regression smoke test sending ~25 typical Polish tool questions over WebSocket.

Uses the same text-over-WS protocol as clients/cli_wav.py.
Reports: query, route, tool called, arguments, spoken reply, e2e latency.
Cleans up any created timers and memories via HTTP API.
Saves results to bench/results/tools_live_smoke.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import websockets

SMOKE_QUERIES = [
    # 1. Clock / time
    {"id": "smoke_01", "category": "clock_time", "query": "Która jest godzina?"},
    {"id": "smoke_02", "category": "clock_time", "query": "Która godzina?"},
    {"id": "smoke_03", "category": "clock_date", "query": "Jaki dziś dzień?"},
    {"id": "smoke_04", "category": "clock_date", "query": "Jaka jest dzisiaj data?"},
    {"id": "smoke_05", "category": "clock_date", "query": "Jaki mamy dzień tygodnia?"},
    # 2. Holidays & calendar
    {"id": "smoke_06", "category": "holiday", "query": "Ile dni zostało do Bożego Narodzenia?"},
    {"id": "smoke_07", "category": "holiday", "query": "Ile dni do świąt Bożego Narodzenia?"},
    {"id": "smoke_08", "category": "holiday", "query": "Kiedy jest najbliższe święto?"},
    {"id": "smoke_09", "category": "time_target", "query": "Jaki dzień tygodnia będzie 3 listopada?"},
    # 3. Weather
    {"id": "smoke_10", "category": "weather", "query": "Jaka jest teraz pogoda w Katowicach?"},
    {"id": "smoke_11", "category": "weather", "query": "Czy jutro będzie padać w Krakowie?"},
    {"id": "smoke_12", "category": "weather", "query": "Jaka jest dzisiaj pogoda w Warszawie?"},
    # 4. Sun
    {"id": "smoke_13", "category": "sun", "query": "O której wschodzi słońce w Katowicach?"},
    {"id": "smoke_14", "category": "sun", "query": "Kiedy zachodzi słońce w Warszawie?"},
    # 5. Calculation
    {"id": "smoke_15", "category": "calculate", "query": "Ile to jest 15 procent z 240?"},
    {"id": "smoke_16", "category": "calculate", "query": "Ile to jest 17 razy 23?"},
    # 6. Unit conversion
    {"id": "smoke_17", "category": "convert", "query": "Przelicz 10 kilogramów na funty."},
    {"id": "smoke_18", "category": "convert", "query": "Ile mililitrów to 2 szklanki?"},
    {"id": "smoke_19", "category": "convert", "query": "Ile bajtów to jeden kilobajt?"},
    # 7. Currency exchange
    {"id": "smoke_20", "category": "currency", "query": "Jaki jest kurs euro?"},
    {"id": "smoke_21", "category": "currency", "query": "Ile kosztuje dolar w złotówkach?"},
    # 8. Wikipedia
    {"id": "smoke_22", "category": "wiki", "query": "Kim był Mikołaj Kopernik?"},
    # 9. Server status
    {"id": "smoke_23", "category": "system", "query": "Jak się czuje serwer?"},
    # 10. Timers (with cleanup)
    {"id": "smoke_24", "category": "timer", "query": "Nastaw minutnik na 5 minut."},
    # 11. Memory (with cleanup)
    {"id": "smoke_25", "category": "memory", "query": "Zapamiętaj że lubię herbatę z cytryną."},
]


async def query_live_adapter(
    ws_url: str,
    text: str,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    """Execute one turn over WebSocket and collect tool call, reply, and metrics."""
    result: dict[str, Any] = {
        "query": text,
        "tool": None,
        "arguments": None,
        "tool_ok": None,
        "tool_speak": None,
        "reply": None,
        "route": None,
        "latency_e2e_ms": None,
        "error": None,
    }

    t0 = time.perf_counter()
    for attempt in range(2):
        try:
            async with websockets.connect(ws_url, close_timeout=2.0) as ws:
                # 1. Hello
                await ws.send(json.dumps({"type": "hello", "lang": "pl", "tts": None, "client": "cli"}))

                # 2. Ready
                first = await asyncio.wait_for(ws.recv(), timeout=5.0)
                ready = json.loads(first)
                if ready.get("type") != "ready":
                    result["error"] = f"Expected ready, got {first}"
                    return result

                # 3. Send text
                await ws.send(json.dumps({"type": "text", "text": text}))

                # 4. Receive events until turn finished
                replies = []
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
                    except asyncio.TimeoutError:
                        result["error"] = "timeout"
                        break

                    if isinstance(msg, bytes):
                        continue

                    evt = json.loads(msg)
                    evt_type = evt.get("type")

                    if evt_type == "tool":
                        result["tool"] = evt.get("name")
                        result["arguments"] = evt.get("arguments")
                        result["tool_ok"] = evt.get("ok")
                        result["tool_speak"] = evt.get("text")
                    elif evt_type == "utterance":
                        replies.append(evt.get("text", ""))
                    elif evt_type == "handoff":
                        result["route"] = "hermes"
                        replies.append(evt.get("text", ""))
                    elif evt_type == "metrics":
                        result["route"] = evt.get("route")
                        if evt.get("tool"):
                            result["tool"] = evt.get("tool")
                        e2e = evt.get("e2e_first_audio_ms")
                        if e2e is not None:
                            result["latency_e2e_ms"] = round(float(e2e), 1)
                        break
                    elif evt_type == "error":
                        result["error"] = evt.get("message")
                        break

                if result["latency_e2e_ms"] is None:
                    result["latency_e2e_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)

                if replies:
                    result["reply"] = " ".join(replies).strip()
                elif result["tool_speak"]:
                    result["reply"] = result["tool_speak"]

            break
        except Exception as exc:
            result["error"] = str(exc)
            if attempt == 0:
                await asyncio.sleep(1.0)
                continue
            if result["latency_e2e_ms"] is None:
                result["latency_e2e_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)

    return result


async def cleanup_timers(api_url: str, initial_timer_ids: set[str]) -> int:
    """Delete all timers that were created during the test run."""
    deleted = 0
    try:
        async with httpx.AsyncClient(base_url=api_url, timeout=5.0) as client:
            resp = await client.get("/api/timers")
            if resp.status_code == 200:
                raw = resp.json()
                timers = raw.get("timers", raw) if isinstance(raw, dict) else raw
                for t in timers:
                    t_id = t.get("id") if isinstance(t, dict) else None
                    if t_id and t_id not in initial_timer_ids:
                        del_resp = await client.delete(f"/api/timers/{t_id}")
                        if del_resp.status_code == 200:
                            deleted += 1
    except Exception as exc:
        print(f"[WARN] Failed to cleanup timers: {exc}", file=sys.stderr)
    return deleted


async def cleanup_memory(api_url: str, initial_fact_ids: set[str]) -> int:
    """Delete all facts that were created during the test run."""
    deleted = 0
    try:
        async with httpx.AsyncClient(base_url=api_url, timeout=5.0) as client:
            resp = await client.get("/api/memory")
            if resp.status_code == 200:
                raw = resp.json()
                facts = raw.get("facts", raw) if isinstance(raw, dict) else raw
                for f in facts:
                    f_id = f.get("id") if isinstance(f, dict) else None
                    if f_id and f_id not in initial_fact_ids:
                        del_resp = await client.delete(f"/api/memory/{f_id}")
                        if del_resp.status_code == 200:
                            deleted += 1
    except Exception as exc:
        print(f"[WARN] Failed to cleanup memory: {exc}", file=sys.stderr)
    return deleted


async def get_initial_ids(api_url: str) -> tuple[set[str], set[str]]:
    timer_ids: set[str] = set()
    fact_ids: set[str] = set()
    try:
        async with httpx.AsyncClient(base_url=api_url, timeout=5.0) as client:
            r_timers = await client.get("/api/timers")
            if r_timers.status_code == 200:
                raw_t = r_timers.json()
                t_list = raw_t.get("timers", raw_t) if isinstance(raw_t, dict) else raw_t
                timer_ids = {t["id"] for t in t_list if isinstance(t, dict) and "id" in t}
            r_mem = await client.get("/api/memory")
            if r_mem.status_code == 200:
                raw_m = r_mem.json()
                m_list = raw_m.get("facts", raw_m) if isinstance(raw_m, dict) else raw_m
                fact_ids = {f["id"] for f in m_list if isinstance(f, dict) and "id" in f}
    except Exception as exc:
        print(f"[WARN] Initial state check failed: {exc}", file=sys.stderr)
    return timer_ids, fact_ids


async def run_smoke(ws_url: str, api_url: str, out_path: Path, timeout_s: float = 15.0) -> int:
    print(f"=== Gaja Tools Live Smoke Test ===")
    print(f"WebSocket URL: {ws_url}")
    print(f"HTTP API URL:  {api_url}")
    print(f"Queries count: {len(SMOKE_QUERIES)}")
    print("=" * 60)

    initial_timers, initial_facts = await get_initial_ids(api_url)

    results = []
    latencies = []

    for item in SMOKE_QUERIES:
        q_id = item["id"]
        cat = item["category"]
        query = item["query"]

        print(f"[{q_id}] ({cat}) -> {query} ... ", end="", flush=True)
        res = await query_live_adapter(ws_url, query, timeout_s=timeout_s)
        res["id"] = q_id
        res["category"] = cat

        tool_str = f"{res['tool']}" if res["tool"] else f"route:{res['route']}"
        lat = res["latency_e2e_ms"]
        if lat is not None:
            latencies.append(lat)
        print(f"[{tool_str}] ({lat} ms)")
        if res.get("reply"):
            print(f"    Reply: {res['reply'][:90]}...")
        if res.get("error"):
            print(f"    Error: {res['error']}")

        results.append(res)
        await asyncio.sleep(0.1)

    # Cleanup
    print("\nCleaning up test artifacts...")
    del_timers = await cleanup_timers(api_url, initial_timers)
    del_facts = await cleanup_memory(api_url, initial_facts)
    print(f"Cleaned up {del_timers} timers, {del_facts} facts.")

    # Save output JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ws_url": ws_url,
        "count": len(results),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "cleaned_timers": del_timers,
        "cleaned_facts": del_facts,
        "results": results,
    }
    out_path.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved results to {out_path}")

    # Summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'ID':<10} | {'Kategoria':<12} | {'Narzędzie':<15} | {'E2E (ms)':<9} | Odpowiedź")
    print("-" * 80)
    for r in results:
        t_name = r["tool"] or f"({r['route']})"
        lat = f"{r['latency_e2e_ms']}" if r['latency_e2e_ms'] else "N/A"
        rep = (r["reply"] or "").replace("\n", " ")[:40]
        print(f"{r['id']:<10} | {r['category']:<12} | {t_name:<15} | {lat:<9} | {rep}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Live smoke evaluation of Gaja tools")
    parser.add_argument("--url", default="ws://127.0.0.1:8790/ws", help="WebSocket URL")
    parser.add_argument("--api-url", default="http://127.0.0.1:8790", help="HTTP API URL")
    parser.add_argument("--out", default="bench/results/tools_live_smoke.json", help="Output JSON path")
    parser.add_argument("--timeout", type=float, default=15.0, help="Per-turn timeout in seconds")

    args = parser.parse_args()
    out_path = Path(args.out)
    exit_code = asyncio.run(run_smoke(args.url, args.api_url, out_path, timeout_s=args.timeout))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
