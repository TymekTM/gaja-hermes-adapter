#!/usr/bin/env python3
"""Gaja's LLM latency while another client loads the same llama-server.

Probes the two Gaja call shapes on their pinned slots, the way the adapter makes them:
  chat   - a ~3k-token chat history already cached in the chat slot plus one new user message,
           streamed; the number is the time to the first content token (TTFT).
  router - a 1-token logprob completion over a ~1.1k-token prompt cached in the router slot.

Each scenario runs the probes while a background "external" client loops one kind of load:
  idle     - no external load
  decode   - short prompt, long streamed generation (summaries, rewrites)
  prefill  - a fresh ~5k-token prompt each time, few output tokens (extraction over a document)

    uv run python scripts/bench_llm_share.py --external http://127.0.0.1:8790/llm
    uv run python scripts/bench_llm_share.py --external http://127.0.0.1:8791   # straight to llama-server

Like the adapter, each probe first primes its slot with the cached prefix: LFM3's recurrent layers
can only resume a prompt cache from a checkpoint, so Gaja re-primes after every turn too. The prime
is reported separately as "slot ready": the time until Gaja's slot holds its prefix again, which is
long when an external request sits in (or evicted) that slot.
The probes overwrite the chat and router slot caches; the adapter re-primes them on its next turn.
Hold /tmp/gaja-bench.lock while it runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent

FILLER_PL = (
    "Użytkownik pytał o pogodę, o muzykę i o światło w pokoju. Gaja odpowiadała krótko i rzeczowo, "
    "podając temperaturę, nazwę utworu albo stan lampy. "
)


def _history(n_turns: int) -> list[dict[str, str]]:
    system = (ROOT / "prompts" / "system_lfm3_exp.md").read_text(encoding="utf-8")
    messages = [{"role": "system", "content": system}]
    for i in range(n_turns):
        messages.append({"role": "user", "content": f"Pytanie numer {i}: {FILLER_PL * 2}"})
        messages.append({"role": "assistant", "content": f"Odpowiedź numer {i}. {FILLER_PL}"})
    return messages


def _document(n_words: int, rng: random.Random) -> str:
    words = FILLER_PL.split()
    return " ".join(rng.choice(words) for _ in range(n_words))


PRIME_MARK = "\u2063gaja-prime\u2063"  # gaja/llm/client.py


async def _chat_prefix(http: httpx.AsyncClient, history: list[dict[str, str]]) -> str:
    body = {
        "messages": [*history, {"role": "user", "content": PRIME_MARK}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    prompt = (await http.post("/apply-template", json=body)).json()["prompt"]
    return prompt[: prompt.rfind(PRIME_MARK)]


async def _prime(http: httpx.AsyncClient, prompt: str, slot: int) -> float:
    """Re-primes the slot; the time is how long Gaja's slot was unavailable (queued or evicted)."""
    body = {"prompt": prompt, "n_predict": 0, "cache_prompt": True, "id_slot": slot}
    started = time.perf_counter()
    (await http.post("/completion", json=body)).raise_for_status()
    return (time.perf_counter() - started) * 1000


async def _ttft_chat(http: httpx.AsyncClient, history: list[dict[str, str]], slot: int, i: int) -> float:
    payload = {
        "messages": [*history, {"role": "user", "content": f"Jaka będzie jutro pogoda? ({i})"}],
        "stream": True,
        "max_tokens": 8,
        "temperature": 0.3,
        "cache_prompt": True,
        "id_slot": slot,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    async with http.stream("POST", "/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            delta = json.loads(line[5:])["choices"][0].get("delta") or {}
            if delta.get("content"):
                return (time.perf_counter() - started) * 1000
    return float("nan")


async def _router(http: httpx.AsyncClient, prompt: str, slot: int, i: int) -> float:
    payload = {
        "prompt": f"{prompt}\nWypowiedź: puść muzykę ({i})\nDecyzja:",
        "n_predict": 1,
        "n_probs": 10,
        "temperature": 0,
        "cache_prompt": True,
        "id_slot": slot,
    }
    started = time.perf_counter()
    response = await http.post("/completion", json=payload)
    response.raise_for_status()
    return (time.perf_counter() - started) * 1000


async def _external_loop(base: str, headers: dict[str, str], kind: str, stop: asyncio.Event, stats: dict) -> None:
    rng = random.Random(7)
    async with httpx.AsyncClient(base_url=base, headers=headers, timeout=600) as http:
        while not stop.is_set():
            if kind == "decode":
                body = {
                    "messages": [{"role": "user", "content": "Napisz długie opowiadanie o lesie."}],
                    "max_tokens": 1024,
                    "stream": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            else:
                body = {
                    "messages": [
                        {"role": "user", "content": f"Wypisz trzy słowa kluczowe:\n{_document(2500, rng)}"}
                    ],
                    "max_tokens": 16,
                    "stream": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            try:
                async with http.stream("POST", "/v1/chat/completions", json=body) as response:
                    stats.setdefault(response.status_code, 0)
                    stats[response.status_code] += 1
                    async for _ in response.aiter_bytes():
                        if stop.is_set():
                            break
            except httpx.HTTPError as exc:
                stats.setdefault(type(exc).__name__, 0)
                stats[type(exc).__name__] += 1
                await asyncio.sleep(0.5)


def _summary(values: list[float]) -> str:
    values = sorted(values)
    p90 = values[min(len(values) - 1, int(round(0.9 * (len(values) - 1))))]
    return f"median {statistics.median(values):7.0f} ms  p90 {p90:7.0f} ms  max {values[-1]:7.0f} ms"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llama", default="http://127.0.0.1:8791")
    parser.add_argument("--external", required=True, help="base URL the external load goes to")
    parser.add_argument("--token-file", type=Path, default=None)
    parser.add_argument("--chat-slot", type=int, default=0)
    parser.add_argument("--router-slot", type=int, default=1)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--scenarios", default="idle,decode,prefill")
    args = parser.parse_args()
    headers = {}
    if args.token_file:
        headers["Authorization"] = f"Bearer {args.token_file.expanduser().read_text().strip()}"

    history = _history(20)
    router_prompt = (ROOT / "prompts" / "system_lfm3_exp.md").read_text(encoding="utf-8")
    async with httpx.AsyncClient(base_url=args.llama, timeout=120) as http:
        chat_prefix = await _chat_prefix(http, history)
        n_chat = len((await http.post("/tokenize", json={"content": chat_prefix})).json()["tokens"])
        n_router = len((await http.post("/tokenize", json={"content": router_prompt})).json()["tokens"])
        print(f"chat prefix {n_chat} tokens, router prompt {n_router} tokens, {args.probes} probes per scenario")
        for name in args.scenarios.split(","):
            stop = asyncio.Event()
            stats: dict = {}
            loader = None
            if name != "idle":
                loader = asyncio.create_task(_external_loop(args.external, headers, name, stop, stats))
                await asyncio.sleep(2.0)
            chat, router, prime = [], [], []
            for i in range(args.probes):
                prime.append(await _prime(http, chat_prefix, args.chat_slot))
                chat.append(await _ttft_chat(http, history, args.chat_slot, i))
                prime.append(await _prime(http, router_prompt, args.router_slot))
                router.append(await _router(http, router_prompt, args.router_slot, i))
                await asyncio.sleep(random.uniform(0.3, 1.2))
            stop.set()
            if loader is not None:
                await loader
            print(f"{name:8s} chat TTFT   {_summary(chat)}")
            print(f"{name:8s} router      {_summary(router)}   external responses {stats or '-'}")
            print(f"{name:8s} slot ready  {_summary(prime)}")


if __name__ == "__main__":
    asyncio.run(main())
