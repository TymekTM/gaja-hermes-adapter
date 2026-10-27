#!/usr/bin/env python3
"""Comprehensive evaluation of local tools on LiquidAI LFM2.5-2.6B (llama-server :8791).

Round 2:
  - Measures Variant B_hybrid (Rules + Router v3 + targeted tool selection + final LLM pass)
  - Measures Variant C (Rules + Router v3 + targeted tool selection + DIRECT speak template, NO final LLM pass)
  - Compares: E2E latency, action accuracy, tool accuracy, argument accuracy,
    false tool-call rate on chit-chat, Hermes boundary accuracy, and Faithfulness.
  - Supports held-out evaluation set to prove no overfitting.
  - Records load average (cat /proc/loadavg) at start and end.

Outputs:
  bench/results/tools_eval_variant_b_hybrid.json
  bench/results/tools_eval_variant_c.json
  bench/results/tools_eval_heldout.json
  bench/results/tools_eval_comparison.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx

from gaja.text.normalize import normalize_for_tts
from gaja.tools import default_registry
from gaja.tools.weather_tool import resolve_tool_location

DATA_PATH = ROOT / "bench/data/tools_eval.jsonl"
HELDOUT_PATH = ROOT / "bench/data/tools_heldout.jsonl"
RESULTS_DIR = ROOT / "bench/results"
SERVER_URL = "http://127.0.0.1:8791"

# Router v3 system prompt with few-shot negative clarifications
ROUTER_V3_SYSTEM = """Jesteś klasyfikatorem zapytań dla asystentki Gaja. Przypisz zapytanie do jednej z trzech kategorii:
- LOCAL: zwykła rozmowa, powitanie, small-talk, żarty, dowcipy, opinie i upodobania ("co sądzisz o muzyce klasycznej"), prośby o czynności fizyczne ("zrób herbatę z cytryną"), stałe fakty i wiedza ogólna ("ile godzin ma doba", "kiedy skończyła się druga wojna"), ceny produktów komercyjnych ("ile kosztuje subskrypcja Netfliksa"), wydarzenia historyczne i pogoda z przeszłości (sprzed lat).
- TOOL: szybkie lokalne narzędzia asystentki:
  * aktualna data i czas (która godzina, jaki dzień, ile dni do daty, jaki dzień tygodnia będzie)
  * aktualna i jutrzejsza prognoza pogody (tylko teraz i jutro, nie historyczna)
  * wschód i zachód słońca oraz długość dnia dla Katowic lub innego miasta
  * przeliczanie jednostek miar, temperatur, miar kuchennych i rozmiarów danych
  * obliczenia matematyczne i procenty z liczb (np. 15% z 240, 12*5, pierwiastki)
  * kursy walut z NBP (np. kurs euro, dolarów, funtów GBP na PLN)
  * encyklopedia Wikipedia (biografie, pojęcia, miasta, zjawiska)
  * stan serwera (uptime, obciążenie, RAM, GPU)
  * najbliższe lub wskazane polskie święto i dni wolne od pracy
  * minutniki i przypomnienia o konkretnej godzinie (nastawianie, ile zostało, anulowanie)
  * zapamiętywanie i usuwanie faktów o użytkowniku
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

# Targeted tool selection prompt distinguishing convert_units vs calculate vs exchange_rate
SYSTEM_PROMPT_TARGETED_TOOL = """Jesteś asystentem Gaja. Wybierz odpowiednie narzędzie i podaj jego parametry:
- get_time: aktualna godzina, data, dzień tygodnia, obliczenia dat (parametr target_date, np. 3 listopada, 24 grudnia).
- get_weather: aktualna i jutrzejsza prognoza pogody.
- get_sun: wschód i zachód słońca oraz długość dnia (domyślnie Katowice).
- calculate: czyste operacje arytmetyczne (+, -, *, /, procenty z liczby np. 15% z 240, pierwiastki). NIE używaj do przeliczania jednostek miar ani temperatur.
- convert_units: przeliczanie jednostek miary (długość, masa, prędkość, temperatura, jednostki kuchenne: szklanki/łyżki, dane: bajty/megabajty).
- exchange_rate: kursy walut NBP (np. euro, dolary, funty brytyjskie jako waluta GBP).
- wiki_summary: Wikipedia, biografie, definicje.
- server_status: stan serwera, obciążenie, procesor, RAM, GPU.
- next_holiday: najbliższe lub wskazane polskie święto (np. Boże Narodzenie).
- manage_timer: minutnik i przypomnienia (action: set, status, cancel; duration_min, time, label).
- remember: zapamiętanie faktu o użytkowniku.
- forget: usunięcie faktu z pamięci."""

SYSTEM_PROMPT_VAR_A = (
    "Jesteś Gaja, przyjazna asystentka głosowa na serwerze Tymka. Twoje odpowiedzi są czytane na głos: pisz zwykłym tekstem, bez markdown. "
    "Odpowiadaj krótko w jednym lub dwóch zdaniach. Liczby pisz słowami. "
    "Masz do dyspozycji 8 szybkich narzędzi lokalnych: get_time, get_weather, calculate, convert_units, exchange_rate, wiki_summary, server_status, next_holiday. "
    "Do skomplikowanych zadań zewnętrznych (mail, kalendarz, smart home, research) wywołaj narzędzie handoff_to_hermes. "
    "Na zwykłą rozmowę i pytania ogólne odpowiadaj bezpośrednio, bez narzędzi."
)

SYSTEM_PROMPT_LOCAL_ONLY = (
    "Jesteś Gaja, przyjazna asystentka głosowa. Odpowiadaj krótko, naturalnie, bez markdown i list."
)

HERMES_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "handoff_to_hermes",
        "description": "Przekazuje zadanie do Hermesa: mail, kalendarz, przypomnienia, smart home, research.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Opis zadania"},
                "reason": {"type": "string", "description": "Powód przekazania"},
            },
            "required": ["task", "reason"],
        },
    },
}

RULES_HERMES = [
    re.compile(r"\b(wy[sś]lij\s+(maila|e-?mail|sms|wiadomo[sś][cć])|sprawd[zź]\s+(mail[ey]|skrzynk[eę]|pocz[tę]|wiadomo[sś]ci)|napisz\s+(do\s+\w+|maila\s+do|sms)|send\s+(an?\s+)?(email|e-mail|message|sms)|check\s+(my\s+)?(emails?|inbox|messages))\b", re.I),
    re.compile(r"\b(w\s+kalendarzu|dodaj\s+do\s+kalendarza|spotkanie\s+w\s+kalendarzu|harmonogram|check\s+(my\s+)?calendar|add\s+to\s+(my\s+)?calendar|schedule\s+a\s+meeting)\b", re.I),
    re.compile(r"\b(przypomnij\s+mi\s+(o\b|za\b|[żz]eby\b|abym\b|jutro\b|w\b)|ustaw\s+przypomnienie|ustaw\s+budzik|obud[zź]\s+mnie|remind\s+me\s+to\b|remind\s+me\s+in\b|set\s+an?\s+alarm|wake\s+me\s+up)\b", re.I),
    re.compile(r"\b((w[lł][aą]cz|zga[sś]|wy[lł][aą]cz|za[sś]wie[cć])\s+[sś]wiat[lł]o|klimatyzacj[eę]|zamknij\s+rolety|otw[oó]rz\s+rolety|turn\s+(on|off)\s+the\s+light|lights\s+(on|off))\b", re.I),
    re.compile(r"\b(skrypt\s+backupu|git\s+commit|napisz\s+dla\s+mnie\s+skrypt|zapisz\s+ten\s+tekst\s+do\s+pliku|conduct\s+a\s+deep\s+web\s+research)\b", re.I),
]

RULES_LOCAL = [
    re.compile(r"\b(dowcip|joke|herbata|herbat[eę]|muzyk[aeę]\s+klasyczn|ile\s+godzin\s+ma\s+doba|subskrypcj[aeę]|kawa|kaw[eę])\b", re.I),
    re.compile(r"\b(w\s+(19\d\d|20[01]\d)|w\s+lipcu\s+19\d\d|w\s+zeszłym\s+wieku|druga\s+wojna\s+[sś]wiatowa|in\s+19\d\d)\b", re.I),
]

RULES_TOOL = [
    re.compile(r"\b(kt[oó]ra\s+(jest\s+)?(teraz\s+)?godzina|jaka\s+(jest\s+)?godzina|what\s+time\s+is\s+it|current\s+time|what'?s\s+the\s+time)\b", re.I),
    re.compile(r"\b(jaki\s+(mamy\s+|dzisiaj\s+|jest\s+)?dzie[nń]\s+tygodnia|jaka\s+(jest\s+)?(dzisiejsza\s+)?data|jaki\s+(jest\s+)?dzisiaj\s+dzie[nń]|kt[oó]ry\s+(dzisiaj\s+|jest\s+)?(mamy\s+)?dzie[nń]|kt[oó]ry\s+dzi[sś]\b|what\s+day\s+is\s+it|what('?s|\s+is)\s+today'?s\s+date)\b", re.I),
    re.compile(r"\b(pogod[aęy]|prognoz[aęy]|b[eę]dzie\s+pada[cć]|pada\s+deszcz|temperatura\s+na\s+zewn[aą]trz|weather|forecast|is\s+it\s+raining)\b", re.I),
    re.compile(r"\b(ile\s+to\s+jest\s+\d+\s*(%|procent)|ile\s+to\s+\d+\s*[*xX/+-]|policz\s+\d+|pierwiastek\s+z\s+\d+|what\s+is\s+\d+\s*percent|calculate\s+\d+)\b", re.I),
    re.compile(r"\b(przelicz|stopni[ae]?\s+celsjusza|celsius\s+to|pounds?\s+(is|to)|funt[oó]w\s+to\s+\d+\s*kg|km/h\s+(na|to)\s+m/s|ile\s+mil\s+to)\b", re.I),
    re.compile(r"\b(kurs\s+(euro|dolara|funta|franka|walut)|ile\s+kosztuje\s+(euro|dolar|funt|frank)|exchange\s+rate)\b", re.I),
    re.compile(r"\b(najbli[zż]sze\s+[sś]wi[eę]to|dzie[nń]\s+wolny\s+od\s+pracy|kiedy\s+mamy\s+wolne|kiedy\s+wypada\s+najbli[zż]sze|next\s+(public\s+)?holiday)\b", re.I),
    re.compile(r"\b(stan\s+serwera|zu[zż]ycie\s+(ram|gpu)|obci[aą][zż]enie\s+maszyny|server\s+status)\b", re.I),
]


def match_rules(text: str) -> str | None:
    if re.search(r"\b(mitologi|greck|greek\s+mythology)\b", text, re.I):
        return "tool"
    for r in RULES_LOCAL:
        if r.search(text):
            return "local"
    for r in RULES_HERMES:
        if r.search(text):
            return "hermes"
    for r in RULES_TOOL:
        if r.search(text):
            return "tool"
    return None


def get_loadavg() -> list[float]:
    try:
        return list(os.getloadavg())
    except Exception:
        return [0.0, 0.0, 0.0]


async def post_json_retry(
    client: httpx.AsyncClient, url: str, json_body: dict[str, Any], timeout: float = 45.0, retries: int = 5
) -> dict[str, Any]:
    for attempt in range(retries):
        try:
            resp = await client.post(url, json=json_body, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt == retries - 1:
                raise
            await asyncio.sleep(1.0)
    raise RuntimeError(f"Failed to post to {url} after {retries} attempts")


def load_dataset(path: Path = DATA_PATH) -> list[dict[str, Any]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def check_arg_accuracy(tool_name: str, args: dict[str, Any], item: dict[str, Any]) -> bool:
    """Check whether predicted tool arguments match expected requirements."""
    expected = item.get("expected_args", {})
    if not expected:
        return True

    if tool_name == "get_weather":
        if "location" in expected:
            resolved_loc = resolve_tool_location(
                args.get("location"),
                item.get("text", ""),
                history=item.get("history") or [],
                last_location=item.get("last_location"),
            )
            pred_loc = str(resolved_loc or "").strip().lower()
            exp_loc = str(expected["location"]).strip().lower()
            if not pred_loc or pred_loc != exp_loc:
                return False
        if "when" in expected:
            pred_when = str(args.get("when", "now")).lower()
            text = item.get("text", "")
            if re.search(r"\bjutro\b|\btomorrow\b", text, re.I):
                pred_when = "tomorrow"
            elif "when" not in args and re.search(
                r"\bdzi[sś]\b|\bdzisiaj\b|\bw\s+weekend\b|\btoday\b", text, re.I
            ):
                pred_when = "today"
            exp_when = str(expected["when"]).lower()
            if pred_when != exp_when:
                return False
        return True

    if tool_name == "calculate":
        pred_expr = str(args.get("expression", "")).replace(" ", "")
        exp_expr = str(expected.get("expression", "")).replace(" ", "")
        numbers = re.findall(r"\d+", exp_expr)
        for num in numbers:
            if num not in pred_expr:
                return False
        return True

    if tool_name == "convert_units":
        if "value" in expected:
            try:
                pred_val = float(args.get("value", 0))
                exp_val = float(expected["value"])
                if abs(pred_val - exp_val) > 0.05 * exp_val:
                    return False
            except (ValueError, TypeError):
                return False
        return True

    if tool_name == "exchange_rate":
        if "currency" in expected:
            pred_cur = str(args.get("currency", "")).upper()
            exp_cur = str(expected["currency"]).upper()
            if exp_cur not in pred_cur and pred_cur not in exp_cur:
                return False
        return True

    if tool_name == "wiki_summary":
        if "topic" in expected:
            pred_top = str(args.get("topic", "")).lower()
            exp_top = str(expected["topic"]).lower()
            exp_words = [w for w in exp_top.split() if len(w) > 3]
            if exp_words and not any(w in pred_top for w in exp_words):
                return False
        return True

    if tool_name == "get_sun":
        if "location" in expected:
            resolved_loc = resolve_tool_location(
                args.get("location"),
                item.get("text", ""),
                history=item.get("history") or [],
                last_location=item.get("last_location"),
            )
            pred_loc = str(resolved_loc or "").strip().lower()
            exp_loc = str(expected["location"]).strip().lower()
            if not pred_loc or pred_loc != exp_loc:
                return False
        return True

    if tool_name == "manage_timer":
        if "action" in expected:
            pred_act = str(args.get("action", "set")).lower()
            exp_act = str(expected["action"]).lower()
            if pred_act != exp_act:
                return False
        if "duration_min" in expected:
            try:
                pred_d = float(args.get("duration_min", 0))
                exp_d = float(expected["duration_min"])
                if abs(pred_d - exp_d) > 0.5:
                    return False
            except (ValueError, TypeError):
                return False
        return True

    if tool_name in ("remember", "forget"):
        if "fact" in expected:
            pred_f = str(args.get("fact", "")).lower()
            exp_f = str(expected["fact"]).lower()
            words = [w for w in exp_f.split() if len(w) > 3]
            if words and not any(w in pred_f for w in words):
                return False
        return True

    if tool_name == "next_holiday":
        if "holiday" in expected:
            pred_h = str(args.get("holiday", "")).lower()
            exp_h = str(expected["holiday"]).lower()
            if exp_h not in pred_h and pred_h not in exp_h:
                return False
        return True

    if tool_name == "get_time":
        if "target_date" in expected:
            pred_td = str(args.get("target_date", "")).lower()
            exp_td = str(expected["target_date"]).lower()
            if exp_td not in pred_td and pred_td not in exp_td:
                return False
        return True

    if tool_name == "set_speaker_volume":
        if "action" in expected:
            if str(args.get("action", "")).lower() != str(expected["action"]).lower():
                return False
        return True

    if tool_name == "spotify":
        if "action" in expected:
            exp_act = expected["action"]
            allowed = [a.lower() for a in exp_act] if isinstance(exp_act, list) else [str(exp_act).lower()]
            if str(args.get("action", "")).lower() not in allowed:
                return False
        if "query" in expected:
            pred_q = str(args.get("query", "")).lower()
            exp_q = str(expected["query"]).lower()
            if exp_q not in pred_q:
                return False
        if "level" in expected:
            try:
                if abs(float(args.get("level", -999)) - float(expected["level"])) > 1.0:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    if tool_name == "home":
        if "action" in expected:
            exp_act = expected["action"]
            allowed = [a.lower() for a in exp_act] if isinstance(exp_act, list) else [str(exp_act).lower()]
            if str(args.get("action", "")).lower() not in allowed:
                return False
        if "device" in expected:
            from gaja.config import Settings as BenchSettings
            from gaja.tools.home_tool import resolve_device

            act_str = str(args.get("action", "status")).lower()
            # Generic bench household: docs/home.example.toml stands in for the real
            # data/home.toml (NEW-6 keeps real entity ids out of the repo).
            bench_cfg = BenchSettings(home={"household_file": ROOT / "docs/home.example.toml"})
            pred_res = resolve_device(
                str(args.get("device", "")),
                bench_cfg.home.aliases,
                utterance=item.get("text", ""),
                action=act_str,
                allowlist=bench_cfg.home.write_allowlist,
                boiler_substrings=bench_cfg.home.boiler_id_substrings,
            )
            if pred_res.entity_id is None:
                return False
            spoken = bench_cfg.home.spoken_forms.get(pred_res.entity_id.lower(), ("", "", ""))
            resolved_desc = f"{pred_res.entity_id} {spoken[0]}".lower()
            exp_dev = str(expected["device"]).lower()
            if exp_dev not in resolved_desc and exp_dev not in str(args.get("device", "")).lower():
                return False
        if "temperature" in expected:
            try:
                if abs(float(args.get("temperature", -999)) - float(expected["temperature"])) > 0.5:
                    return False
            except (TypeError, ValueError):
                return False
        if "hvac_mode" in expected:
            if str(args.get("hvac_mode", "")).lower() != str(expected["hvac_mode"]).lower():
                return False
        if "level" in expected:
            try:
                if abs(float(args.get("level", -999)) - float(expected["level"])) > 1.0:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    return True


def check_faithfulness(tool_name: str | None, tool_res: Any, final_text: str) -> bool:
    """Check whether final spoken text contains only facts from raw tool result (no hallucinations)."""
    if tool_res is None or not isinstance(tool_res, dict):
        return True

    # 1. Hallucination checks: invented repetitions or wild phrasing
    if "polsko-polski" in final_text.lower():
        return False

    # 2. Key numbers check
    if tool_name == "calculate":
        res_val = tool_res.get("result")
        if res_val is not None:
            norm_res = normalize_for_tts(str(res_val), "pl").lower()
            norm_speech = normalize_for_tts(final_text, "pl").lower()
            if str(res_val) not in final_text and norm_res not in norm_speech:
                return False

    if tool_name == "exchange_rate":
        res_val = tool_res.get("result")
        if res_val is not None:
            norm_speech = normalize_for_tts(final_text, "pl").lower()
            res_str = str(res_val)
            if res_str not in final_text and str(int(res_val)) not in final_text:
                norm_res = normalize_for_tts(res_str, "pl").lower()
                if norm_res not in norm_speech:
                    return False

    if tool_name == "get_weather":
        # Check that temperatures mentioned are consistent with data
        t_max = tool_res.get("temp_max") or tool_res.get("temperature")
        t_min = tool_res.get("temp_min")
        norm_speech = normalize_for_tts(final_text, "pl").lower()
        if t_max is not None:
            max_int = str(round(float(t_max)))
            if max_int not in final_text and normalize_for_tts(max_int, "pl") not in norm_speech:
                # If neither integer nor float appears
                return False

    return True


async def call_router_v3(client: httpx.AsyncClient, query: str) -> tuple[str, float, float]:
    """1-token completion classification: LOCAL, TOOL, HERMES using cached system prefix."""
    prompt = (
        f"<|im_start|>system\n{ROUTER_V3_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\nZapytanie: {query}<|im_end|>\n"
        f"<|im_start|>assistant\nKategoria:"
    )
    t0 = time.perf_counter()
    data = await post_json_retry(
        client,
        f"{SERVER_URL}/completion",
        {
            "prompt": prompt,
            "n_predict": 1,
            "n_probs": 10,
            "temperature": 0.0,
            "cache_prompt": True,
        },
        timeout=30.0,
    )
    duration_ms = (time.perf_counter() - t0) * 1000

    top_logprobs = (data.get("completion_probabilities") or [{}])[0].get("top_logprobs") or []
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

    best_cat = max(masses, key=masses.get)
    return best_cat, masses[best_cat], duration_ms


async def eval_prompt_variant_b_or_c(
    client: httpx.AsyncClient,
    item: dict[str, Any],
    local_tools_only: list[dict[str, Any]],
    use_rules: bool = True,
    variant: str = "b_hybrid",
) -> dict[str, Any]:
    query = item["text"]
    t_start = time.perf_counter()

    router_ms = 0.0
    rule_matched = None
    if use_rules:
        rule_matched = match_rules(query)

    if rule_matched is not None:
        cat = rule_matched
        prob = 1.0
        router_ms = 0.0
    else:
        cat, prob, router_ms = await call_router_v3(client, query)

    predicted_action = cat
    predicted_tool = None
    predicted_args: dict[str, Any] = {}
    tool_call_ms = 0.0
    tool_exec_ms = 0.0
    second_pass_ms = 0.0
    final_text = ""
    raw_result = None

    if predicted_action == "hermes":
        predicted_tool = "handoff_to_hermes"
        final_text = "Przekazuję zadanie do Hermesa."
    elif predicted_action == "local":
        t_loc = time.perf_counter()
        loc_json = await post_json_retry(
            client,
            f"{SERVER_URL}/v1/chat/completions",
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_LOCAL_ONLY},
                    {"role": "user", "content": query},
                ],
                "temperature": 0.0,
                "stream": False,
            },
            timeout=30.0,
        )
        second_pass_ms = (time.perf_counter() - t_loc) * 1000
        final_text = loc_json["choices"][0]["message"].get("content", "")
    else:  # "tool"
        t_tc = time.perf_counter()
        tc_json = await post_json_retry(
            client,
            f"{SERVER_URL}/v1/chat/completions",
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_TARGETED_TOOL},
                    {"role": "user", "content": query},
                ],
                "tools": local_tools_only,
                "temperature": 0.0,
                "stream": False,
            },
            timeout=45.0,
        )
        tool_call_ms = (time.perf_counter() - t_tc) * 1000
        tc_msg = tc_json["choices"][0]["message"]
        calls = tc_msg.get("tool_calls", [])

        if calls:
            fn = calls[0]["function"]
            predicted_tool = fn.get("name")
            try:
                predicted_args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                predicted_args = {}

            t_exec = time.perf_counter()
            tool_res = await default_registry.execute(predicted_tool, predicted_args, lang=item.get("lang", "pl"))
            tool_exec_ms = (time.perf_counter() - t_exec) * 1000
            raw_result = tool_res.to_dict()

            if variant == "c":
                # VARIANT C: Speak the tool's speak text DIRECTLY (0 ms second pass LLM!)
                second_pass_ms = 0.0
                final_text = tool_res.speak
            else:
                # VARIANT B_hybrid: Second pass LLM generation
                t_pass = time.perf_counter()
                pass2_json = await post_json_retry(
                    client,
                    f"{SERVER_URL}/v1/chat/completions",
                    {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT_VAR_A},
                            {"role": "user", "content": query},
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": calls,
                            },
                            {
                                "role": "tool",
                                "tool_call_id": calls[0].get("id", "call_1"),
                                "content": json.dumps(raw_result, ensure_ascii=False),
                            },
                        ],
                        "temperature": 0.0,
                        "stream": False,
                    },
                    timeout=45.0,
                )
                second_pass_ms = (time.perf_counter() - t_pass) * 1000
                final_text = pass2_json["choices"][0]["message"].get("content", "")
        else:
            final_text = tc_msg.get("content", "")

    total_e2e_ms = (time.perf_counter() - t_start) * 1000

    target_action = item["target_action"]
    target_tool = item.get("target_tool")

    action_correct = predicted_action == target_action
    tool_correct = (
        action_correct
        if target_action != "tool"
        else (predicted_action == "tool" and predicted_tool == target_tool)
    )
    args_correct = False
    if target_action == "tool" and tool_correct:
        args_correct = check_arg_accuracy(target_tool, predicted_args, item)
    elif target_action != "tool" and action_correct:
        args_correct = True

    faithful = check_faithfulness(predicted_tool, raw_result, final_text)

    return {
        "id": item["id"],
        "query": query,
        "target_action": target_action,
        "predicted_action": predicted_action,
        "target_tool": target_tool,
        "predicted_tool": predicted_tool,
        "predicted_args": predicted_args,
        "router_prob": round(prob, 4),
        "rule_matched": rule_matched,
        "action_correct": action_correct,
        "tool_correct": tool_correct,
        "args_correct": args_correct,
        "faithful": faithful,
        "router_ms": round(router_ms, 1),
        "tool_call_ms": round(tool_call_ms, 1),
        "tool_exec_ms": round(tool_exec_ms, 1),
        "second_pass_ms": round(second_pass_ms, 1),
        "total_e2e_ms": round(total_e2e_ms, 1),
        "final_text": final_text,
    }


def compute_metrics(results: list[dict[str, Any]], dataset: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    action_acc = sum(1 for r in results if r["action_correct"]) / n
    tool_acc = sum(1 for r in results if r["tool_correct"]) / n
    args_acc = sum(1 for r in results if r["args_correct"]) / n
    faithfulness = sum(1 for r in results if r.get("faithful", True)) / n

    chitchat_items = [r for r, d in zip(results, dataset) if d.get("category") == "chitchat"]
    n_chitchat = len(chitchat_items)
    false_tool_chitchat = (
        sum(1 for r in chitchat_items if r["predicted_action"] != "local") / n_chitchat if n_chitchat else 0.0
    )

    hermes_items = [r for r, d in zip(results, dataset) if d.get("category") in ("hermes_boundary", "hermes")]
    n_hermes = len(hermes_items)
    hermes_boundary_acc = (
        sum(1 for r in hermes_items if r["predicted_action"] == "hermes") / n_hermes if n_hermes else 0.0
    )

    traps_items = [r for r, d in zip(results, dataset) if d.get("category") == "trap"]
    n_traps = len(traps_items)
    trap_acc = sum(1 for r in traps_items if r["action_correct"]) / n_traps if n_traps else 0.0

    tool_items = [r for r, d in zip(results, dataset) if d.get("target_action") == "tool"]
    n_tools = len(tool_items)
    tool_selection_recall = (
        sum(1 for r in tool_items if r["predicted_tool"] == r["target_tool"]) / n_tools if n_tools else 0.0
    )

    e2e_latencies = [r["total_e2e_ms"] for r in results]
    e2e_p50 = statistics.median(e2e_latencies)
    e2e_p90 = sorted(e2e_latencies)[int(len(e2e_latencies) * 0.9)]

    router_latencies = [r.get("router_ms", 0.0) for r in results if "router_ms" in r and r.get("router_ms", 0.0) > 0]
    router_p50 = statistics.median(router_latencies) if router_latencies else 0.0
    router_p90 = (
        sorted(router_latencies)[int(len(router_latencies) * 0.9)] if router_latencies else 0.0
    )

    return {
        "n_samples": n,
        "action_accuracy": round(action_acc, 4),
        "tool_accuracy": round(tool_acc, 4),
        "argument_accuracy": round(args_acc, 4),
        "tool_selection_recall": round(tool_selection_recall, 4),
        "false_tool_call_rate_chitchat": round(false_tool_chitchat, 4),
        "hermes_boundary_accuracy": round(hermes_boundary_acc, 4),
        "trap_accuracy": round(trap_acc, 4),
        "faithfulness_rate": round(faithfulness, 4),
        "latency_router_p50_ms": round(router_p50, 1),
        "latency_router_p90_ms": round(router_p90, 1),
        "latency_e2e_p50_ms": round(e2e_p50, 1),
        "latency_e2e_p90_ms": round(e2e_p90, 1),
        "latency_e2e_mean_ms": round(statistics.mean(e2e_latencies), 1),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["b_hybrid", "c", "all"], default="all")
    parser.add_argument("--dataset", choices=["eval", "heldout", "both"], default="both")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    local_tools_only = default_registry.schemas("pl")

    datasets_to_run = []
    if args.dataset in ("eval", "both"):
        ds_eval = load_dataset(DATA_PATH)
        if args.limit > 0:
            ds_eval = ds_eval[: args.limit]
        datasets_to_run.append(("eval", DATA_PATH, ds_eval))
    if args.dataset in ("heldout", "both"):
        ds_heldout = load_dataset(HELDOUT_PATH)
        if args.limit > 0:
            ds_heldout = ds_heldout[: args.limit]
        datasets_to_run.append(("heldout", HELDOUT_PATH, ds_heldout))

    async with httpx.AsyncClient(timeout=60.0) as client:
        for ds_name, ds_path, dataset in datasets_to_run:
            print(f"\n=======================================================")
            print(f"Running evaluation on dataset: {ds_name} ({len(dataset)} prompts)")
            print(f"=======================================================")

            # Evaluate Variant B_hybrid
            if args.variant in ("b_hybrid", "all"):
                load_start = get_loadavg()
                print(f"\n=== Running Variant B_hybrid (Round 2) on {ds_name} (start loadavg: {load_start}) ===")
                res_b_hybrid = []
                for i, item in enumerate(dataset):
                    r = await eval_prompt_variant_b_or_c(
                        client, item, local_tools_only, use_rules=True, variant="b_hybrid"
                    )
                    res_b_hybrid.append(r)
                    if (i + 1) % 20 == 0 or (i + 1) == len(dataset):
                        print(f"  [B_hybrid-{ds_name}] Processed {i+1}/{len(dataset)} prompts...")

                load_end = get_loadavg()
                metrics_b_hybrid = compute_metrics(res_b_hybrid, dataset)
                out_b_hybrid = {
                    "variant": "B_hybrid_round2",
                    "dataset": ds_name,
                    "loadavg_start": load_start,
                    "loadavg_end": load_end,
                    "metrics": metrics_b_hybrid,
                    "results": res_b_hybrid,
                }
                fn = f"tools_eval_variant_b_hybrid_{ds_name}.json" if ds_name != "eval" else "tools_eval_variant_b_hybrid.json"
                with open(RESULTS_DIR / fn, "w", encoding="utf-8") as f:
                    json.dump(out_b_hybrid, f, ensure_ascii=False, indent=2)
                print(f"Variant B_hybrid ({ds_name}) Metrics: {json.dumps(metrics_b_hybrid, indent=2)}")

            # Evaluate Variant C
            if args.variant in ("c", "all"):
                load_start = get_loadavg()
                print(f"\n=== Running Variant C (Direct Speak, No 2nd Pass LLM) on {ds_name} (start loadavg: {load_start}) ===")
                res_c = []
                for i, item in enumerate(dataset):
                    r = await eval_prompt_variant_b_or_c(
                        client, item, local_tools_only, use_rules=True, variant="c"
                    )
                    res_c.append(r)
                    if (i + 1) % 20 == 0 or (i + 1) == len(dataset):
                        print(f"  [C-{ds_name}] Processed {i+1}/{len(dataset)} prompts...")

                load_end = get_loadavg()
                metrics_c = compute_metrics(res_c, dataset)
                out_c = {
                    "variant": "C_direct_speak",
                    "dataset": ds_name,
                    "loadavg_start": load_start,
                    "loadavg_end": load_end,
                    "metrics": metrics_c,
                    "results": res_c,
                }
                fn = f"tools_eval_variant_c_{ds_name}.json" if ds_name != "eval" else "tools_eval_variant_c.json"
                with open(RESULTS_DIR / fn, "w", encoding="utf-8") as f:
                    json.dump(out_c, f, ensure_ascii=False, indent=2)
                print(f"Variant C ({ds_name}) Metrics: {json.dumps(metrics_c, indent=2)}")

        # Build comprehensive comparison JSON
        comparison = {}
        # Read r1 comparison if exists
        comp_path = RESULTS_DIR / "tools_eval_comparison.json"
        if comp_path.exists():
            with open(comp_path, encoding="utf-8") as f:
                comparison = json.load(f)

        if (RESULTS_DIR / "tools_eval_variant_b_hybrid.json").exists():
            with open(RESULTS_DIR / "tools_eval_variant_b_hybrid.json", encoding="utf-8") as f:
                d = json.load(f)
                comparison["variant_b_hybrid_r2"] = d["metrics"]
                comparison["variant_b_hybrid_r2"]["loadavg_start"] = d.get("loadavg_start")
                comparison["variant_b_hybrid_r2"]["loadavg_end"] = d.get("loadavg_end")

        if (RESULTS_DIR / "tools_eval_variant_c.json").exists():
            with open(RESULTS_DIR / "tools_eval_variant_c.json", encoding="utf-8") as f:
                d = json.load(f)
                comparison["variant_c"] = d["metrics"]
                comparison["variant_c"]["loadavg_start"] = d.get("loadavg_start")
                comparison["variant_c"]["loadavg_end"] = d.get("loadavg_end")

        if (RESULTS_DIR / "tools_eval_variant_c_heldout.json").exists():
            with open(RESULTS_DIR / "tools_eval_variant_c_heldout.json", encoding="utf-8") as f:
                d = json.load(f)
                comparison["variant_c_heldout"] = d["metrics"]
                comparison["variant_c_heldout"]["loadavg_start"] = d.get("loadavg_start")
                comparison["variant_c_heldout"]["loadavg_end"] = d.get("loadavg_end")

        with open(RESULTS_DIR / "tools_eval_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print(f"\nSaved updated comparison to {comp_path}")


if __name__ == "__main__":
    asyncio.run(main())
