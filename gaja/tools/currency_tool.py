from __future__ import annotations

import time
import unicodedata
from typing import Any

import httpx

from gaja.text.normalize import normalize_for_tts
from gaja.tools.base import (
    ToolResult,
    format_number_pl,
    inflect_number_noun,
    polish_declension,
)

CURRENCY_ALIASES: dict[str, str] = {
    "eur": "EUR",
    "euro": "EUR",
    "usd": "USD",
    "dolar": "USD",
    "dolary": "USD",
    "dolarow": "USD",
    "dollar": "USD",
    "dollars": "USD",
    "gbp": "GBP",
    "funt": "GBP",
    "funty": "GBP",
    "funtow": "GBP",
    "pound": "GBP",
    "pounds": "GBP",
    "chf": "CHF",
    "frank": "CHF",
    "franki": "CHF",
    "frankow": "CHF",
    "czk": "CZK",
    "koronaczeska": "CZK",
    "koronacz": "CZK",
    "nok": "NOK",
    "sek": "SEK",
    "jpy": "JPY",
    "jen": "JPY",
    "jeny": "JPY",
    "jenow": "JPY",
    "yen": "JPY",
    "aud": "AUD",
    "cad": "CAD",
    "pln": "PLN",
    "zl": "PLN",
    "zloty": "PLN",
    "zlote": "PLN",
    "zlotych": "PLN",
}

CURRENCY_NAMES_PL = {
    "EUR": ("euro", "euro", "euro"),
    "USD": ("dolar", "dolary", "dolarów"),
    "GBP": ("funt", "funty", "funtów"),
    "CHF": ("frank", "franki", "franków"),
    "CZK": ("korona", "korony", "koron"),
    "NOK": ("korona", "korony", "koron"),
    "SEK": ("korona", "korony", "koron"),
    "JPY": ("jen", "jeny", "jenów"),
    "AUD": ("dolar", "dolary", "dolarów"),
    "CAD": ("dolar", "dolary", "dolarów"),
    "PLN": ("złoty", "złote", "złotych"),
}

CURRENCY_NOUN_MAP: dict[str, str] = {
    "EUR": "euro",
    "USD": "dolar",
    "GBP": "funt",
    "CHF": "frank",
    "CZK": "korona",
    "NOK": "korona",
    "SEK": "korona",
    "JPY": "jen",
    "AUD": "dolar",
    "CAD": "dolar",
    "PLN": "złoty",
}

_NBP_CACHE: tuple[float, dict[str, float], str] | None = None
CACHE_TTL_S = 3600.0  # 1 hour


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _resolve_currency(name: str) -> str:
    norm = _strip_accents(name.lower().strip().replace(" ", ""))
    if norm in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[norm]
    orig_norm = name.lower().strip().replace(" ", "")
    if orig_norm in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[orig_norm]
    return name.upper().strip()


async def _get_nbp_rates(client: httpx.AsyncClient) -> tuple[dict[str, float], str]:
    """Fetch current mid rates from NBP Table A with 1-hour cache."""
    global _NBP_CACHE
    now = time.time()
    if _NBP_CACHE is not None:
        cached_time, cached_rates, cached_date = _NBP_CACHE
        if now - cached_time < CACHE_TTL_S:
            return cached_rates, cached_date

    url = "https://api.nbp.pl/api/exchangerates/tables/A/?format=json"
    res = await client.get(url, timeout=2.5)
    if res.status_code != 200:
        raise RuntimeError(f"NBP API returned HTTP {res.status_code}")

    data = res.json()
    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError("Invalid NBP API response structure")

    table_data = data[0]
    date_str = table_data.get("effectiveDate", "")
    items = table_data.get("rates", [])

    rates: dict[str, float] = {"PLN": 1.0}
    for item in items:
        code = item.get("code")
        mid = item.get("mid")
        if code and mid:
            rates[code.upper()] = float(mid)

    _NBP_CACHE = (now, rates, date_str)
    return rates, date_str


async def exchange_rate(
    currency: str,
    amount: float | int = 1.0,
    to: str = "PLN",
    lang: str = "pl",
    *,
    client: httpx.AsyncClient | None = None,
) -> ToolResult:
    """Check exchange rates via NBP API (table A) with 1h cache."""
    from_code = _resolve_currency(currency)
    to_code = _resolve_currency(to)

    own_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=2.5)
        own_client = True

    try:
        rates, date_str = await _get_nbp_rates(client)

        if from_code not in rates:
            err = f"Nieznana waluta: {currency} ({from_code})"
            speak = f"Nie mam aktualnego kursu dla waluty {currency}." if lang == "pl" else f"Unknown currency {currency}."
            return ToolResult(data={"error": err}, speak=speak, error=err)

        if to_code not in rates:
            err = f"Nieznana waluta docelowa: {to} ({to_code})"
            speak = f"Nie mam kursu dla waluty docelowej {to}." if lang == "pl" else f"Unknown target currency {to}."
            return ToolResult(data={"error": err}, speak=speak, error=err)

        amt = float(amount)
        rate_from = rates[from_code]
        rate_to = rates[to_code]

        total = (amt * rate_from) / rate_to
        rounded_total = round(total, 2)
        single_rate = round(rate_from / rate_to, 4)

        data = {
            "amount": amt,
            "currency_from": from_code,
            "currency_to": to_code,
            "rate": single_rate,
            "result": rounded_total,
            "effective_date": date_str,
        }

        if lang == "en":
            raw_speak = f"{amt} {from_code} is {rounded_total} {to_code} (rate: {single_rate})."
        else:
            noun_from = CURRENCY_NOUN_MAP.get(from_code, from_code)
            amt_num = int(amt) if amt.is_integer() else amt
            from_str = inflect_number_noun(amt_num, noun_from, case="nom")

            if to_code == "PLN":
                zlotych_int = int(rounded_total)
                groszy_int = int(round((rounded_total - zlotych_int) * 100))
                zl_str = inflect_number_noun(zlotych_int, "złoty", case="nom")
                if groszy_int > 0:
                    gr_str = inflect_number_noun(groszy_int, "grosz", case="nom")
                    raw_speak = f"{from_str} to {zl_str} i {gr_str}."
                else:
                    raw_speak = f"{from_str} to {zl_str}."
            else:
                noun_to = CURRENCY_NOUN_MAP.get(to_code, to_code)
                to_str = inflect_number_noun(rounded_total, noun_to, case="nom")
                raw_speak = f"{from_str} to około {to_str}."

        speak = normalize_for_tts(raw_speak, lang)
        return ToolResult(data=data, speak=speak)
    except Exception as exc:
        err = f"Błąd NBP: {exc}"
        speak = "Nie udało się pobrać kursu walut." if lang == "pl" else "Could not fetch exchange rates."
        return ToolResult(data={"error": err}, speak=speak, error=err)
    finally:
        if own_client:
            await client.aclose()
