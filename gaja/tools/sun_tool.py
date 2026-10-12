from __future__ import annotations

import datetime
import math
from zoneinfo import ZoneInfo
from typing import Any

from gaja.text.normalize import normalize_for_tts
from gaja.tools.base import ToolResult, polish_declension

DEFAULT_LAT = 50.2649
DEFAULT_LON = 19.0238
DEFAULT_CITY = "Katowice"

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

    cos_ha = (math.cos(zenith) / (math.cos(lat_rad) * math.cos(decl)) - math.tan(lat_rad) * math.tan(decl))

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
) -> ToolResult:
    """Compute offline sunrise, sunset, and day length for Katowice or specified location."""
    city_name = (location or DEFAULT_CITY).strip()
    if not city_name or city_name.lower() in {"miasto", "city", "tutaj", "here"}:
        city_name = DEFAULT_CITY

    city_key = city_name.lower().strip()
    coords = CITY_COORDINATES.get(city_key)
    if coords is None:
        # Fallback to Katowice if location is unknown offline
        lat, lon = DEFAULT_LAT, DEFAULT_LON
    else:
        lat, lon = coords

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
        raw_speak = (
            f"In {city_name}, sunrise is at {sr_str} and sunset is at {ss_str}. "
            f"Day length is {day_len_str}."
        )
    else:
        h_word = polish_declension(hours, "godzina", "godziny", "godzin")
        m_word = polish_declension(minutes, "minuta", "minuty", "minut")
        day_len_str = f"{hours} {h_word} i {minutes} {m_word}" if minutes else f"{hours} {h_word}"
        raw_speak = (
            f"W miejscowości {city_name} słońce wschodzi o godzinie {sr_str}, a zachodzi o {ss_str}. "
            f"Dzień trwa {day_len_str}."
        )

    speak = normalize_for_tts(raw_speak, lang)
    data = {
        "location": city_name,
        "date": target_date.isoformat(),
        "sunrise": sr_str,
        "sunset": ss_str,
        "day_length_hours": round(day_sec / 3600.0, 2),
        "day_length": day_len_str,
    }
    return ToolResult(data=data, speak=speak)
