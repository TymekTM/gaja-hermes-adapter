from __future__ import annotations

import datetime
import math
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from gaja.text.normalize import normalize_for_tts
from gaja.tools.base import PLACEHOLDER_LOCATIONS, ToolResult, get_default_location, polish_declension
from gaja.tools.weather_tool import geocode_location, normalize_location

DEFAULT_LAT = 50.2649
DEFAULT_LON = 19.0238
DEFAULT_CITY = get_default_location()

CITY_COORDINATES: dict[str, tuple[float, float]] = {
    "katowice": (50.2649, 19.0238),
    "warszawa": (52.2297, 21.0122),
    "krakow": (50.0647, 19.9450),
    "kraków": (50.0647, 19.9450),
    "lodz": (51.7592, 19.4560),
    "łódź": (51.7592, 19.4560),
    "wroclaw": (51.1079, 17.0385),
    "wrocław": (51.1079, 17.0385),
    "poznan": (52.4064, 16.9252),
    "poznań": (52.4064, 16.9252),
    "gdansk": (54.3520, 18.6466),
    "gdańsk": (54.3520, 18.6466),
    "szczecin": (53.4285, 14.5528),
    "bydgoszcz": (53.1235, 18.0084),
    "lublin": (51.2465, 22.5684),
    "bialystok": (53.1325, 23.1688),
    "białystok": (53.1325, 23.1688),
    "gdynia": (54.5189, 18.5305),
    "sopot": (54.4416, 18.5601),
    "czestochowa": (50.8118, 19.1203),
    "częstochowa": (50.8118, 19.1203),
    "radom": (51.4027, 21.1471),
    "sosnowiec": (50.2863, 19.1041),
    "torun": (53.0138, 18.5984),
    "toruń": (53.0138, 18.5984),
    "kielce": (50.8661, 20.6286),
    "gliwice": (50.2945, 18.6714),
    "zabrze": (50.3249, 18.7857),
    "bytom": (50.3480, 18.9328),
    "bielsko-biala": (49.8224, 19.0469),
    "bielsko-biała": (49.8224, 19.0469),
    "olsztyn": (53.7784, 20.4801),
    "rzeszow": (50.0412, 21.9991),
    "rzeszów": (50.0412, 21.9991),
    "rybnik": (50.0961, 18.5417),
    "tychy": (50.1231, 18.9868),
    "opole": (50.6751, 17.9213),
    "zielona gora": (51.9356, 15.5062),
    "zielona góra": (51.9356, 15.5062),
    "jelenia gora": (50.9044, 15.7194),
    "jelenia góra": (50.9044, 15.7194),
    "zakopane": (49.2992, 19.9496),
    "nowy targ": (49.4775, 20.0323),
    "nowy sacz": (49.6175, 20.7153),
    "nowy sącz": (49.6175, 20.7153),
    "kolobrzeg": (54.1759, 15.5833),
    "kołobrzeg": (54.1759, 15.5833),
    "swinoujscie": (53.9105, 14.2471),
    "świnoujście": (53.9105, 14.2471),
    "londyn": (51.5074, -0.1278),
    "london": (51.5074, -0.1278),
    "paryz": (48.8566, 2.3522),
    "paryż": (48.8566, 2.3522),
    "paris": (48.8566, 2.3522),
    "berlin": (52.5200, 13.4050),
    "rzym": (41.9028, 12.4964),
    "rome": (41.9028, 12.4964),
    "madryt": (40.4168, -3.7038),
    "madrid": (40.4168, -3.7038),
    "wieden": (48.2082, 16.3738),
    "wiedeń": (48.2082, 16.3738),
    "vienna": (48.2082, 16.3738),
    "praga": (50.0755, 14.4378),
    "prague": (50.0755, 14.4378),
    "nowy jork": (40.7128, -74.0060),
    "new york": (40.7128, -74.0060),
    "tokio": (35.6762, 139.6503),
    "tokyo": (35.6762, 139.6503),
}


def calculate_sun_times(
    date: datetime.date,
    lat: float,
    lon: float,
    tz_name: str = "Europe/Warsaw",
) -> tuple[datetime.time | None, datetime.time | None, float]:
    """NOAA solar calculation for sunrise, sunset and day length in seconds."""
    tz = ZoneInfo(tz_name)
    day_of_year = date.timetuple().tm_yday
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1)

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )

    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )

    lat_rad = math.radians(lat)
    zenith = math.radians(90.833)  # Official solar zenith including refraction

    cos_ha = math.cos(zenith) / (math.cos(lat_rad) * math.cos(decl)) - math.tan(lat_rad) * math.tan(decl)

    if cos_ha > 1.0:
        return None, None, 0.0
    if cos_ha < -1.0:
        return None, None, 86400.0

    ha = math.acos(cos_ha)
    ha_deg = math.degrees(ha)

    solar_noon_utc_min = 720.0 - 4.0 * lon - eqtime
    sunrise_utc_min = solar_noon_utc_min - ha_deg * 4.0
    sunset_utc_min = solar_noon_utc_min + ha_deg * 4.0
    day_length_sec = ha_deg * 8.0 * 60.0

    sunrise_utc_dt = datetime.datetime.combine(
        date, datetime.time.min, tzinfo=datetime.timezone.utc
    ) + datetime.timedelta(minutes=sunrise_utc_min)
    sunset_utc_dt = datetime.datetime.combine(
        date, datetime.time.min, tzinfo=datetime.timezone.utc
    ) + datetime.timedelta(minutes=sunset_utc_min)

    sunrise_local = sunrise_utc_dt.astimezone(tz)
    sunset_local = sunset_utc_dt.astimezone(tz)

    return sunrise_local.time(), sunset_local.time(), day_length_sec


async def get_sun(
    location: str | None = None,
    date: str | None = None,
    lang: str = "pl",
    kind: str = "all",
    *,
    client: httpx.AsyncClient | None = None,
    **_extra: Any,
) -> ToolResult:
    """Compute offline sunrise, sunset, and day length for the configured default or specified location."""
    default_city = get_default_location()
    raw_city = (location or "").strip()
    if not raw_city or raw_city.lower() in PLACEHOLDER_LOCATIONS:
        city_name = normalize_location(default_city) or default_city
    else:
        city_name = normalize_location(raw_city) or raw_city

    city_key = city_name.lower().strip()
    matched_name = city_name
    coords = CITY_COORDINATES.get(city_key)
    if coords is not None:
        lat, lon = coords
    else:
        # Not in the fast lookup table: geocode the requested city instead of silently using the
        # default city's coordinates (a wrong city by hundreds of km, ~38 min of sunrise in June).
        # Same Open-Meteo geocoding as the weather tool, within the 4 s tool timeout budget.
        own_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=2.5)
            own_client = True
        try:
            hit, geo_err = await geocode_location(city_name, lang, client=client)
            if geo_err is not None:
                local = "Nie udało się ustalić lokalizacji." if lang == "pl" else "Could not resolve the location."
                return ToolResult(data={"error": geo_err}, speak=local, error=geo_err)
            if hit is None:
                err = f"Nie znaleziono lokalizacji '{city_name}'"
                speak = f"Nie znalazłam lokalizacji {city_name}." if lang == "pl" else f"Could not find location {city_name}."
                return ToolResult(data={"error": err}, speak=speak, error=err)
            lat = float(hit["latitude"])
            lon = float(hit["longitude"])
            matched_name = str(hit.get("name")) or city_name
        except Exception as exc:
            err = f"Błąd geokodowania lokalizacji: {exc!r}"
            local = "Nie udało się ustalić lokalizacji." if lang == "pl" else "Could not resolve the location."
            return ToolResult(data={"error": err}, speak=local, error=err)
        finally:
            if own_client:
                await client.aclose()

    tz_name = "Europe/Warsaw"
    tz = ZoneInfo(tz_name)
    target_date = datetime.datetime.now(tz).date()

    if date and date not in ("today", "dzis", "dziś", "now"):
        try:
            target_date = datetime.date.fromisoformat(date)
        except Exception:
            pass

    sunrise, sunset, day_sec = calculate_sun_times(target_date, lat, lon, tz_name)

    if sunrise is None or sunset is None:
        err = "Noc polarna lub dzień polarny"
        return ToolResult(data={"error": err}, speak="W tym miejscu występuje noc polarna lub dzień polarny.", error=err)

    sr_str = sunrise.strftime("%H:%M")
    ss_str = sunset.strftime("%H:%M")
    total_minutes = int(round(day_sec / 60.0))
    hours = total_minutes // 60
    minutes = total_minutes % 60

    if lang == "en":
        h_word = "hour" if hours == 1 else "hours"
        m_word = "minute" if minutes == 1 else "minutes"
        day_len_str = f"{hours} {h_word} {minutes} {m_word}" if minutes else f"{hours} {h_word}"
        if kind == "sunrise":
            raw_speak = f"In {matched_name}, sunrise is at {sr_str}."
        elif kind == "sunset":
            raw_speak = f"In {matched_name}, sunset is at {ss_str}."
        elif kind == "day_length":
            raw_speak = f"In {matched_name}, day length is {day_len_str}."
        else:
            raw_speak = (
                f"In {matched_name}, sunrise is at {sr_str} and sunset is at {ss_str}. "
                f"Day length is {day_len_str}."
            )
    else:
        h_word = polish_declension(hours, "godzina", "godziny", "godzin")
        m_word = polish_declension(minutes, "minuta", "minuty", "minut")
        day_len_str = f"{hours} {h_word} i {minutes} {m_word}" if minutes else f"{hours} {h_word}"
        if kind == "sunrise":
            raw_speak = f"W miejscowości {matched_name} słońce wschodzi o godzinie {sr_str}."
        elif kind == "sunset":
            raw_speak = f"W miejscowości {matched_name} słońce zachodzi o godzinie {ss_str}."
        elif kind == "day_length":
            raw_speak = f"W miejscowości {matched_name} dzień trwa {day_len_str}."
        else:
            raw_speak = (
                f"W miejscowości {matched_name} słońce wschodzi o godzinie {sr_str}, a zachodzi o {ss_str}. "
                f"Dzień trwa {day_len_str}."
            )

    speak = normalize_for_tts(raw_speak, lang)
    data = {
        "location": matched_name,
        "date": target_date.isoformat(),
        "sunrise": sr_str,
        "sunset": ss_str,
        "day_length_hours": round(day_sec / 3600.0, 2),
        "day_length": day_len_str,
        "kind": kind,
    }
    return ToolResult(data=data, speak=speak)
