from __future__ import annotations

import datetime
import pytest

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
