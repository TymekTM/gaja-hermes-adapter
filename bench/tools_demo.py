#!/usr/bin/env python3
"""Interactive demo running 10 real prompts end-to-end against llama-server :8791.

Executes real tool calls with real network APIs (Open-Meteo, NBP, Wikipedia, local OS).
Prints: question -> tool call -> raw result -> direct speak (Variant C) -> final LLM spoken text (Variant B).
Saves transcript to bench/results/tools_demo.json with start/end loadavg.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx

from gaja.tools import default_registry

RESULTS_DIR = ROOT / "bench/results"
SERVER_URL = "http://127.0.0.1:8791"

DEMO_PROMPTS = [
    {"id": "demo_1", "lang": "pl", "query": "Która jest teraz godzina w Polsce?"},
    {"id": "demo_2", "lang": "pl", "query": "Jaka jest jutro pogoda w Krakowie?"},
    {"id": "demo_3", "lang": "pl", "query": "Ile to jest 15 procent z 240?"},
    {"id": "demo_4", "lang": "pl", "query": "Przelicz 100 kilometrów na mile."},
    {"id": "demo_5", "lang": "pl", "query": "Jaki jest aktualny kurs euro w złotówkach?"},
    {"id": "demo_6", "lang": "pl", "query": "Kim był Mikołaj Kopernik według Wikipedii?"},
    {"id": "demo_7", "lang": "pl", "query": "Jaki jest aktualny stan tego serwera i zużycie RAM?"},
    {"id": "demo_8", "lang": "pl", "query": "Kiedy wypada najbliższe święto i dzień wolny od pracy?"},
    {"id": "demo_9", "lang": "pl", "query": "Cześć Gaja, jak się dzisiaj czujesz?"},
    {"id": "demo_10", "lang": "pl", "query": "Kim jest Hermes w mitologii greckiej?"},
]

ROUTER_V3_SYSTEM = """Jesteś klasyfikatorem zapytań dla asystentki Gaja. Przypisz zapytanie do jednej z trzech kategorii:
- LOCAL: zwykła rozmowa, powitanie, small-talk, żarty, dowcipy, opinie i upodobania ("co sądzisz o muzyce klasycznej"), prośby o czynności fizyczne ("zrób herbatę z cytryną"), stałe fakty i wiedza ogólna ("ile godzin ma doba", "kiedy skończyła się druga wojna"), ceny produktów komercyjnych ("ile kosztuje subskrypcja Netfliksa"), wydarzenia historyczne i pogoda z przeszłości (sprzed lat).
- TOOL: szybkie lokalne narzędzia asystentki:
  * aktualna data i czas (która godzina, jaki dzień)
  * aktualna i jutrzejsza prognoza pogody (tylko teraz i jutro, nie historyczna)
  * przeliczanie jednostek miar i temperatur (km/h na m/s, stopnie Celsjusza na Fahrenheita, funty/pounds [waga] na kg)
  * obliczenia matematyczne i procenty z liczb (np. 15% z 240, 12*5, pierwiastki)
  * kursy walut z NBP (np. kurs euro, dolarów, funtów GBP na PLN)
  * encyklopedia Wikipedia (biografie, pojęcia, miasta, zjawiska)
  * stan serwera (uptime, obciążenie, RAM, GPU)
  * najbliższe polskie święto i dni wolne od pracy
- HERMES: zadania zewnętrzne i złożone: wysyłanie e-maili i wiadomości do ludzi, kalendarz, przypomnienia, budziki, smart home, pisanie kodu i skryptów, operacje na plikach, wieloetapowy research w internecie.

Przykłady:
Zapytanie: cześć co tam słychać
Kategoria: LOCAL

Zapytanie: co sądzisz o muzyce klasycznej
Kategoria: LOCAL

Zapytanie: opowiedz mi jakiś śmieszny dowcip
Kategoria: LOCAL

Zapytanie: tell me a funny programming joke
Kategoria: LOCAL

Zapytanie: zrób mi proszę herbatę z cytryną
Kategoria: LOCAL

Zapytanie: ile godzin ma doba
Kategoria: LOCAL

Zapytanie: ile kosztuje miesięczna subskrypcja netflixa
Kategoria: LOCAL

Zapytanie: jaka była pogoda w warszawie w lipcu 1990 roku
Kategoria: LOCAL

Zapytanie: która jest godzina
Kategoria: TOOL

Zapytanie: jaka jest pogoda jutro w krakowie
Kategoria: TOOL

Zapytanie: ile to 15 procent z 240
Kategoria: TOOL

Zapytanie: ile to jest 20 stopni celsjusza w fahrenheitach
Kategoria: TOOL

Zapytanie: przelicz 50 kilometrów na godzinę na metry na sekundę
Kategoria: TOOL

Zapytanie: how many pounds is 10 kilograms
Kategoria: TOOL

Zapytanie: jaki jest kurs euro
Kategoria: TOOL

Zapytanie: kim był Mikołaj Kopernik
Kategoria: TOOL

Zapytanie: jaki jest stan serwera
Kategoria: TOOL

Zapytanie: kiedy jest najbliższe święto
Kategoria: TOOL

Zapytanie: wyślij maila do Kasi
Kategoria: HERMES

Zapytanie: sprawdź mój kalendarz na jutro
Kategoria: HERMES

Zapytanie: włącz światło w kuchni
Kategoria: HERMES

Zapytanie: przypomnij mi o lekarzu o 18
Kategoria: HERMES

Zapytanie: kim jest Hermes w mitologii greckiej
Kategoria: TOOL

Zapytanie: dziękuję bardzo za pomoc
Kategoria: LOCAL

Zapytanie: zrób dokładny research na temat fuzji jądrowej
Kategoria: HERMES"""

SYSTEM_PROMPT_TOOL = """Jesteś asystentem Gaja. Wybierz odpowiednie narzędzie i podaj jego parametry:
- convert_units: przeliczanie jednostek miary (długość: km, mi, m; masa: kg, funty/pounds [waga]; prędkość: km/h na m/s; temperatura: stopnie celsjusza na fahrenheity; objętość; czas). Używaj do przeliczania jednostek miar, wagi w funtach oraz przeliczania temperatur.
- calculate: czyste operacje arytmetyczne (+, -, *, /, procenty z liczby np. 15% z 240, pierwiastki). NIE używaj do przeliczania jednostek miar ani temperatur.
- exchange_rate: kursy walut NBP (np. euro, dolary, funty brytyjskie jako waluta GBP). NIE używaj do przeliczania wagi (pounds jako waga to convert_units).
- get_weather: aktualna i jutrzejsza prognoza pogody.
- wiki_summary: Wikipedia, biografie, pojęcia, miasta."""

SYSTEM_PROMPT_CHAT = "Jesteś Gaja, asystentka głosowa. Odpowiadaj krótko i naturalnie po polsku, 1-2 zdaniami, bez markdown."


def get_loadavg() -> list[float]:
    try:
        return list(os.getloadavg())
    except Exception:
        return [0.0, 0.0, 0.0]


def classify_router_category(top_logprobs: list[dict[str, Any]]) -> str:
    masses = {"local": 0.0, "tool": 0.0, "hermes": 0.0}
    for item in top_logprobs:
        tok = str(item.get("token", "")).strip().lower()
        prob = math.exp(float(item.get("logprob", -99.0)))
        if tok.startswith("to") or tok == "tool":
            masses["tool"] += prob
        elif tok.startswith("h") or tok == "herm":
            masses["hermes"] += prob
        elif tok.startswith("l") or tok == "loc":
            masses["local"] += prob
    total = sum(masses.values())
    if total > 0:
        for k in masses:
            masses[k] /= total
    return max(masses, key=masses.get)


async def run_demo() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tools_schemas = default_registry.schemas("pl")
    demo_records = []
    loadavg_start = get_loadavg()

    print("=" * 80)
    print("GAJA LOCAL TOOLS END-TO-END DEMO (LiquidAI LFM2.5-2.6B on :8791)")
    print(f"Start loadavg: {loadavg_start}")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=30.0) as client:
        for idx, item in enumerate(DEMO_PROMPTS, 1):
            q = item["query"]
            print(f"\n[{idx}/10] Pytanie: \"{q}\"")
            t_start = time.perf_counter()

            # 1. Router classification (no duplicate BOS)
            prompt_r = (
                f"<|im_start|>system\n{ROUTER_V3_SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\nZapytanie: {q}<|im_end|>\n"
                f"<|im_start|>assistant\nKategoria:"
            )
            r_resp = await client.post(
                f"{SERVER_URL}/completion",
                json={"prompt": prompt_r, "n_predict": 1, "n_probs": 10, "temperature": 0.0, "cache_prompt": True},
            )
            top = (r_resp.json().get("completion_probabilities") or [{}])[0].get("top_logprobs") or []
            category = classify_router_category(top)
            router_ms = (time.perf_counter() - t_start) * 1000

            tool_name = None
            tool_args = {}
            raw_result = None
            direct_speak = ""
            final_spoken = ""
            tool_call_latency_ms = 0.0
            second_pass_ms = 0.0

            if category == "local":
                print(f"  -> Kategoria: LOCAL (router: {router_ms:.1f} ms)")
                c_resp = await client.post(
                    f"{SERVER_URL}/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT_CHAT},
                            {"role": "user", "content": q},
                        ],
                        "temperature": 0.0,
                        "stream": False,
                    },
                )
                final_spoken = c_resp.json()["choices"][0]["message"].get("content", "").strip()
                direct_speak = final_spoken
                print(f"  -> Bezpośrednia odpowiedź: \"{final_spoken}\"")
            elif category == "tool":
                print(f"  -> Kategoria: TOOL (router: {router_ms:.1f} ms)")
                t_tc = time.perf_counter()
                tc_resp = await client.post(
                    f"{SERVER_URL}/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT_TOOL},
                            {"role": "user", "content": q},
                        ],
                        "tools": tools_schemas,
                        "temperature": 0.0,
                        "stream": False,
                    },
                )
                tool_call_latency_ms = (time.perf_counter() - t_tc) * 1000
                msg = tc_resp.json()["choices"][0]["message"]
                calls = msg.get("tool_calls", [])
                if calls:
                    fn = calls[0]["function"]
                    tool_name = fn.get("name")
                    try:
                        tool_args = json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        tool_args = {}
                    print(f"  -> Wywołanie narzędzia ({tool_call_latency_ms:.1f} ms): {tool_name}({tool_args})")

                    # Real tool execution
                    t_exec = time.perf_counter()
                    tool_res = await default_registry.execute(tool_name, tool_args, lang=item["lang"])
                    exec_ms = (time.perf_counter() - t_exec) * 1000
                    raw_result = tool_res.to_dict()
                    direct_speak = tool_res.speak
                    print(f"  -> Wynik surowy ({exec_ms:.1f} ms): {json.dumps(raw_result, ensure_ascii=False)}")
                    print(f"  -> [Wariant C] Mowa bezpośrednia (0 ms extra): \"{direct_speak}\"")

                    # Second pass LLM generation (Variant B)
                    t_pass2 = time.perf_counter()
                    pass2_resp = await client.post(
                        f"{SERVER_URL}/v1/chat/completions",
                        json={
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT_CHAT},
                                {"role": "user", "content": q},
                                {"role": "assistant", "content": "", "tool_calls": calls},
                                {
                                    "role": "tool",
                                    "tool_call_id": calls[0].get("id", "call_1"),
                                    "content": json.dumps(raw_result, ensure_ascii=False),
                                },
                            ],
                            "temperature": 0.0,
                            "stream": False,
                        },
                    )
                    second_pass_ms = (time.perf_counter() - t_pass2) * 1000
                    final_spoken = pass2_resp.json()["choices"][0]["message"].get("content", "").strip()
                    print(f"  -> [Wariant B] Wypowiedź 2nd-pass ({second_pass_ms:.1f} ms): \"{final_spoken}\"")
                else:
                    final_spoken = msg.get("content", "").strip()
                    direct_speak = final_spoken
                    print(f"  -> Odpowiedź tekstowa: \"{final_spoken}\"")
            else:
                print(f"  -> Kategoria: HERMES (router: {router_ms:.1f} ms)")
                final_spoken = "Przekazuję zadanie do Hermesa."
                direct_speak = final_spoken
                print(f"  -> Akcja: handoff_to_hermes")

            total_ms = (time.perf_counter() - t_start) * 1000
            variant_c_ms = router_ms + tool_call_latency_ms + (exec_ms if category == "tool" and calls else 0.0)
            print(f"  -> Latencja E2E Wariant C: {variant_c_ms:.1f} ms | Wariant B: {total_ms:.1f} ms")

            demo_records.append({
                "id": item["id"],
                "question": q,
                "category": category,
                "tool_called": tool_name,
                "tool_arguments": tool_args,
                "raw_result": raw_result,
                "direct_speak_text_variant_c": direct_speak,
                "final_spoken_text_variant_b": final_spoken,
                "router_ms": round(router_ms, 1),
                "variant_c_latency_ms": round(variant_c_ms, 1),
                "variant_b_total_latency_ms": round(total_ms, 1),
            })

    loadavg_end = get_loadavg()
    out_file = RESULTS_DIR / "tools_demo.json"
    demo_output = {
        "loadavg_start": loadavg_start,
        "loadavg_end": loadavg_end,
        "records": demo_records,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(demo_output, f, ensure_ascii=False, indent=2)
    print(f"\nTranskrypcja demo zapisana w {out_file} (end loadavg: {loadavg_end})")


if __name__ == "__main__":
    asyncio.run(run_demo())
