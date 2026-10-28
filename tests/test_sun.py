from __future__ import annotations

import datetime
import httpx
import pytest
from zoneinfo import ZoneInfo

from gaja.tools.sun_tool import (
    calculate_sun_times,
    get_sun,
    DEFAULT_LAT,
    DEFAULT_LON,
    DEFAULT_CITY,
)


def test_calculate_sun_times_katowice_solstices():
    # Katowice: lat 50.2649 N, lon 19.0238 E
    # Summer solstice (2026-06-21, CEST UTC+2):
    # Sunrise ~ 04:32, Sunset ~ 20:57, Day length ~ 16.4 hours
    sr_s, ss_s, dl_s = calculate_sun_times(datetime.date(2026, 6, 21), DEFAULT_LAT, DEFAULT_LON)
    assert sr_s is not None and ss_s is not None
    assert sr_s.hour == 4
    assert 30 <= sr_s.minute <= 35
    assert ss_s.hour == 20
    assert 55 <= ss_s.minute <= 59
    assert 16.3 <= (dl_s / 3600.0) <= 16.5

    # Winter solstice (2026-12-21, CET UTC+1):
    # Sunrise ~ 07:40, Sunset ~ 15:42, Day length ~ 8.0 hours
    sr_w, ss_w, dl_w = calculate_sun_times(datetime.date(2026, 12, 21), DEFAULT_LAT, DEFAULT_LON)
    assert sr_w is not None and ss_w is not None
    assert sr_w.hour == 7
    assert 38 <= sr_w.minute <= 43
    assert ss_w.hour == 15
    assert 40 <= ss_w.minute <= 45
    assert 7.9 <= (dl_w / 3600.0) <= 8.1


@pytest.mark.asyncio
async def test_get_sun_default_katowice():
    res = await get_sun()
    assert res.error is None
    assert res.data["location"] == "Katowice"
    assert "sunrise" in res.data
    assert "sunset" in res.data
    assert "day_length" in res.data
    assert "Katowice" in res.speak
    assert "wschodzi o godzinie" in res.speak


@pytest.mark.asyncio
async def test_get_sun_default_from_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GAJA_TOOLS__DEFAULT_LOCATION", "Wrocław")
    res = await get_sun()
    assert res.error is None
    assert res.data["location"] == "Wrocław"
    assert "Wrocław" in res.speak


@pytest.mark.asyncio
async def test_get_sun_specified_and_locative_locations():
    res_w = await get_sun(location="Warszawa", date="2026-06-21")
    assert res_w.error is None
    assert res_w.data["location"] == "Warszawa"
    assert res_w.data["sunrise"] in ("04:13", "04:14")

    res_l = await get_sun(location="Londyn", date="2026-06-21")
    assert res_l.error is None
    assert res_l.data["location"] == "Londyn"

    res_loc = await get_sun(location="Gdańsku", date="2026-06-21", kind="sunset")
    assert res_loc.error is None
    assert res_loc.data["location"] == "Gdańsk"


@pytest.mark.asyncio
async def test_get_sun_geocodes_a_city_absent_from_the_table() -> None:
    # Suwałki (~54.11 N) vs the default city Katowice (~50.26 N): sunrise differs by ~38 min in June.
    suwalki = (54.1116, 22.9309)

    def handler(request: httpx.Request) -> httpx.Response:
        assert "geocoding-api.open-meteo.com" in str(request.url)
        return httpx.Response(
            200,
            json={"results": [{"name": "Suwałki", "latitude": suwalki[0], "longitude": suwalki[1]}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await get_sun(location="Suwałki", date="2026-06-21", client=client)
    assert res.error is None
    assert res.data["location"] == "Suwałki"
    sr, _ss, _dl = calculate_sun_times(datetime.date(2026, 6, 21), *suwalki)
    assert sr is not None
    assert res.data["sunrise"] == sr.strftime("%H:%M")


@pytest.mark.asyncio
async def test_get_sun_admits_it_when_geocoding_finds_nothing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await get_sun(location="Atlantyda", client=client)
    assert res.error is not None
    assert "Nie znalazłam lokalizacji Atlantyda" in res.speak
    assert "wschodzi o godzinie" not in res.speak  # never speaks the default city's times instead


@pytest.mark.asyncio
async def test_get_sun_uses_the_locations_own_timezone() -> None:
    """Sunrise in London is London local time, not Europe/Warsaw wall time."""
    when = datetime.date(2026, 10, 27)
    london = await get_sun(location="Londyn", date=when.isoformat())
    sr, ss, _ = calculate_sun_times(when, 51.5074, -0.1278, "Europe/London")
    assert london.error is None
    assert london.data["timezone"] == "Europe/London"
    assert london.data["sunrise"] == sr.strftime("%H:%M")
    assert london.data["sunset"] == ss.strftime("%H:%M")

    tokyo = await get_sun(location="Tokio", date=when.isoformat())
    sr_t, ss_t, _ = calculate_sun_times(when, 35.6762, 139.6503, "Asia/Tokyo")
    assert tokyo.error is None
    assert tokyo.data["timezone"] == "Asia/Tokyo"
    assert tokyo.data["sunrise"] == sr_t.strftime("%H:%M")
    assert tokyo.data["sunset"] == ss_t.strftime("%H:%M")


@pytest.mark.asyncio
async def test_get_sun_uses_the_geocoders_timezone_for_unknown_cities() -> None:
    when = datetime.date(2026, 10, 27)
    coords = (-33.8679, 151.2073)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "Sydney",
                        "latitude": coords[0],
                        "longitude": coords[1],
                        "timezone": "Australia/Sydney",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await get_sun(location="Sydney", date=when.isoformat(), client=client)
    sr, ss, _ = calculate_sun_times(when, *coords, "Australia/Sydney")
    assert res.error is None
    assert res.data["timezone"] == "Australia/Sydney"
    assert res.data["sunrise"] == sr.strftime("%H:%M")
    assert res.data["sunset"] == ss.strftime("%H:%M")


@pytest.mark.asyncio
async def test_get_sun_honours_relative_and_dotted_dates() -> None:
    today = datetime.datetime.now(ZoneInfo("Europe/Warsaw")).date()
    cases = {
        "jutro": today + datetime.timedelta(days=1),
        "pojutrze": today + datetime.timedelta(days=2),
        "28.10.2026": datetime.date(2026, 10, 28),
    }
    for text, expected in cases.items():
        res = await get_sun(location="Katowice", date=text)
        assert res.error is None, text
        assert res.data["date"] == expected.isoformat(), text


@pytest.mark.asyncio
async def test_get_sun_spoken_text_follows_kind() -> None:
    res_rise = await get_sun(kind="sunrise")
    assert "wschodzi o godzinie" in res_rise.speak
    assert "zachodzi" not in res_rise.speak

    res_set = await get_sun(kind="sunset")
    assert "zachodzi" in res_set.speak
    assert "wschodzi" not in res_set.speak

    res_day = await get_sun(kind="day_length")
    assert "dzień trwa" in res_day.speak.lower()
    assert "wschodzi" not in res_day.speak and "zachodzi" not in res_day.speak
