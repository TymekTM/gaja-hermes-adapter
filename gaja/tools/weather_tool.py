from __future__ import annotations

import os
import time
import urllib.parse
from typing import Any

import httpx

from gaja.text.normalize import normalize_for_tts
from gaja.tools.base import ToolResult, polish_declension

DEFAULT_LOCATION = os.environ.get("GAJA_TOOLS__DEFAULT_LOCATION", "Katowice")

WMO_CODES_PL = {
    0: "słonecznie",
    1: "głównie bezchmurnie",
    2: "częściowo pochmurno",
    3: "pochmurno",
    45: "mgła",
    48: "mgła osadzająca szadź",
    51: "lekka mżawka",
    53: "umiarkowana mżawka",
    55: "gęsta mżawka",
    56: "marznąca mżawka",
    57: "gęsta marznąca mżawka",
    61: "słaby deszcz",
    63: "umiarkowany deszcz",
    65: "silny deszcz",
    66: "marznący deszcz",
    67: "silny marznący deszcz",
    71: "słaby śnieg",
    73: "umiarkowany śnieg",
    75: "silny śnieg",
    77: "ziarna śniegu",
    80: "lekki przelotny deszcz",
    81: "umiarkowany przelotny deszcz",
    82: "gwałtowny przelotny deszcz",
    85: "lekki przelotny śnieg",
    86: "obfity przelotny śnieg",
    95: "burza",
    96: "burza z lekkim gradem",
    99: "burza z silnym gradem",
}

WMO_CODES_EN = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}

_WEATHER_CACHE: dict[tuple[str, str, str], tuple[float, ToolResult]] = {}
CACHE_TTL_S = 600.0  # 10 minutes


def _degree_word_pl(val: int | float) -> str:
    n = round(val)
    return polish_declension(n, "stopień", "stopnie", "stopni")


async def get_weather(
    location: str | None = None,
    when: str = "now",
    lang: str = "pl",
    *,
    client: httpx.AsyncClient | None = None,
) -> ToolResult:
    """Get weather forecast from Open-Meteo with 10-minute cache."""
    loc = (location or DEFAULT_LOCATION).strip()
    if not loc:
        loc = DEFAULT_LOCATION

    when_norm = when.lower().strip() if when else "now"
    if when_norm not in ("now", "today", "tomorrow"):
        when_norm = "now"

    cache_key = (loc.lower(), when_norm, lang)
    now_ts = time.time()
    if cache_key in _WEATHER_CACHE:
        cached_time, cached_result = _WEATHER_CACHE[cache_key]
        if now_ts - cached_time < CACHE_TTL_S:
            return cached_result

    own_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=2.5)
        own_client = True

    try:
        geo_url = (
            f"https://geocoding-api.open-meteo.com/v1/search?"
            f"name={urllib.parse.quote(loc)}&count=1&language={lang}&format=json"
        )
        geo_res = await client.get(geo_url)
        if geo_res.status_code != 200:
            err = f"Open-Meteo geocoding error HTTP {geo_res.status_code}"
            return ToolResult(data={"error": err}, speak="Nie udało się pobrać informacji o pogodzie.", error=err)

        geo_data = geo_res.json()
        results = geo_data.get("results")
        if not results:
            err = f"Nie znaleziono lokalizacji '{loc}'"
            speak = f"Nie znalazłam lokalizacji {loc}." if lang == "pl" else f"Could not find location {loc}."
            return ToolResult(data={"error": err}, speak=speak, error=err)

        matched_loc = results[0].get("name", loc)
        lat = results[0]["latitude"]
        lon = results[0]["longitude"]

        forecast_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&"
            f"current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m&"
            f"daily=weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,"
            f"apparent_temperature_min,precipitation_probability_max,wind_speed_10m_max,sunrise,sunset&"
            f"timezone=auto"
        )
        f_res = await client.get(forecast_url)
        if f_res.status_code != 200:
            err = f"Open-Meteo forecast error HTTP {f_res.status_code}"
            return ToolResult(data={"error": err}, speak="Nie udało się pobrać informacji o pogodzie.", error=err)

        f_data = f_res.json()
        current = f_data.get("current", {})
        daily = f_data.get("daily", {})

        if when_norm == "now":
            temp_raw = current.get("temperature_2m", 0.0)
            feels_raw = current.get("apparent_temperature", temp_raw)
            temp = round(temp_raw, 1)
            feels_like = round(feels_raw, 1)
            temp_int = round(temp_raw)
            feels_int = round(feels_raw)
            w_code = current.get("weather_code", 0)
            wind = round(current.get("wind_speed_10m", 0.0))

            cond_text = (
                WMO_CODES_PL.get(w_code, "umiarkowane warunki")
                if lang == "pl"
                else WMO_CODES_EN.get(w_code, "moderate conditions")
            )

            data = {
                "location": matched_loc,
                "when": "now",
                "temperature": temp,
                "apparent_temperature": feels_like,
                "conditions": cond_text,
                "wind_speed_kmh": wind,
            }

            if lang == "en":
                raw_speak = (
                    f"{matched_loc}, currently: {temp_int} degrees, feels like {feels_int} degrees, "
                    f"{cond_text}, wind {wind} km/h."
                )
            else:
                deg_temp = _degree_word_pl(temp_int)
                deg_feels = _degree_word_pl(feels_int)
                raw_speak = (
                    f"{matched_loc}, obecnie: {temp_int} {deg_temp}, odczuwalna {feels_int} {deg_feels}, "
                    f"{cond_text}, wiatr {wind} km/h."
                )
        else:
            day_idx = 1 if when_norm == "tomorrow" else 0
            times = daily.get("time", [])
            if day_idx >= len(times):
                day_idx = 0

            target_date = times[day_idx] if day_idx < len(times) else ""
            t_max_raw = daily.get("temperature_2m_max", [0])[day_idx]
            t_min_raw = daily.get("temperature_2m_min", [0])[day_idx]
            t_max = round(t_max_raw, 1)
            t_min = round(t_min_raw, 1)
            t_max_int = round(t_max_raw)
            t_min_int = round(t_min_raw)
            w_code = daily.get("weather_code", [0])[day_idx]
            precip_prob = round(daily.get("precipitation_probability_max", [0])[day_idx])
            wind_max = round(daily.get("wind_speed_10m_max", [0])[day_idx])

            sunrise_raw = daily.get("sunrise", [""])[day_idx]
            sunset_raw = daily.get("sunset", [""])[day_idx]
            sunrise = sunrise_raw.split("T")[-1] if "T" in sunrise_raw else sunrise_raw
            sunset = sunset_raw.split("T")[-1] if "T" in sunset_raw else sunset_raw

            cond_text = (
                WMO_CODES_PL.get(w_code, "umiarkowane warunki")
                if lang == "pl"
                else WMO_CODES_EN.get(w_code, "moderate conditions")
            )

            data = {
                "location": matched_loc,
                "when": when_norm,
                "date": target_date,
                "temp_min": t_min,
                "temp_max": t_max,
                "conditions": cond_text,
                "precipitation_probability": precip_prob,
                "wind_speed_kmh": wind_max,
                "sunrise": sunrise,
                "sunset": sunset,
            }

            day_word_pl = "jutro" if when_norm == "tomorrow" else "dzisiaj"
            day_word_en = "tomorrow" if when_norm == "tomorrow" else "today"

            if lang == "en":
                raw_speak = (
                    f"{matched_loc}, {day_word_en}: temperature from {t_min_int} to {t_max_int} degrees, "
                    f"{cond_text}, chance of rain {precip_prob} percent, sunrise at {sunrise}, sunset at {sunset}."
                )
            else:
                deg_max = _degree_word_pl(t_max_int)
                raw_speak = (
                    f"{matched_loc}, {day_word_pl}: od {t_min_int} do {t_max_int} {deg_max}, "
                    f"{cond_text}, szansa na opady {precip_prob} procent, wschód słońca o {sunrise}, zachód o {sunset}."
                )

        speak = normalize_for_tts(raw_speak, lang)
        result = ToolResult(data=data, speak=speak)
        _WEATHER_CACHE[cache_key] = (now_ts, result)
        return result
    except Exception as exc:
        err = f"Błąd pobierania pogody: {exc}"
        return ToolResult(data={"error": err}, speak="Nie udało się pobrać informacji o pogodzie.", error=err)
    finally:
        if own_client:
            await client.aclose()
