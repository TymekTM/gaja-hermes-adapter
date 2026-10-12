from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from gaja.tools import (
    ToolRegistry,
    ToolResult,
    calculate,
    convert_units,
    exchange_rate,
    get_time,
    get_weather,
    next_holiday,
    server_status,
    wiki_summary,
)
from gaja.tools.holiday_tool import calculate_easter, get_polish_holidays


# 1. get_time
@pytest.mark.asyncio
async def test_get_time_pl():
    res = await get_time("Europe/Warsaw", lang="pl")
    assert res.error is None
    assert "date" in res.data
    assert "time" in res.data
    assert "weekday" in res.data
    assert res.data["timezone"] == "Europe/Warsaw"
    assert "Jest " in res.speak
    assert "godzina " in res.speak


@pytest.mark.asyncio
async def test_get_time_en():
    res = await get_time("Europe/Warsaw", lang="en")
    assert res.error is None
    assert "It is " in res.speak


# 2. get_weather (mocked)
@pytest.mark.asyncio
async def test_get_weather_now_mocked():
    mock_geo_resp = {
        "results": [{"name": "Warszawa", "latitude": 52.23, "longitude": 21.01}]
    }
    mock_forecast_resp = {
        "current": {
            "temperature_2m": 12.4,
            "apparent_temperature": 9.1,
            "weather_code": 3,
            "wind_speed_10m": 15.0,
        },
        "daily": {
            "time": ["2026-10-09", "2026-10-10"],
            "temperature_2m_max": [15.0, 16.0],
            "temperature_2m_min": [8.0, 7.0],
            "weather_code": [3, 2],
            "precipitation_probability_max": [10, 20],
            "wind_speed_10m_max": [20, 18],
            "sunrise": ["2026-10-09T06:50", "2026-10-10T06:52"],
            "sunset": ["2026-10-09T18:05", "2026-10-10T18:03"],
        },
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)

    async def mock_get(url: str, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "geocoding-api" in url:
            resp.json.return_value = mock_geo_resp
        else:
            resp.json.return_value = mock_forecast_resp
        return resp

    mock_client.get = mock_get

    res = await get_weather("Warszawa", "now", "pl", client=mock_client)
    assert res.error is None
    assert res.data["location"] == "Warszawa"
    assert res.data["temperature"] == 12.4
    assert res.data["conditions"] == "pochmurno"
    assert "dwanaście stopni" in res.speak or "12" in res.speak


@pytest.mark.asyncio
async def test_get_weather_tomorrow_mocked():
    mock_geo_resp = {
        "results": [{"name": "Kraków", "latitude": 50.06, "longitude": 19.94}]
    }
    mock_forecast_resp = {
        "daily": {
            "time": ["2026-10-09", "2026-10-10"],
            "temperature_2m_max": [14.0, 16.5],
            "temperature_2m_min": [6.0, 5.0],
            "weather_code": [2, 1],
            "precipitation_probability_max": [0, 5],
            "wind_speed_10m_max": [15, 12],
            "sunrise": ["2026-10-09T06:48", "2026-10-10T06:50"],
            "sunset": ["2026-10-09T18:08", "2026-10-10T18:06"],
        }
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)

    async def mock_get(url: str, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "geocoding-api" in url:
            resp.json.return_value = mock_geo_resp
        else:
            resp.json.return_value = mock_forecast_resp
        return resp

    mock_client.get = mock_get

    res = await get_weather("Kraków", "tomorrow", "pl", client=mock_client)
    assert res.error is None
    assert res.data["location"] == "Kraków"
    assert res.data["temp_max"] == 16.5
    assert "Kraków, jutro:" in res.speak and "stopni" in res.speak


@pytest.mark.asyncio
async def test_get_weather_default_location_katowice():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_geo_resp = {
        "results": [{"name": "Katowice", "latitude": 50.26, "longitude": 19.02}]
    }
    mock_forecast_resp = {
        "current": {
            "temperature_2m": 15.0,
            "apparent_temperature": 14.5,
            "precipitation": 0.0,
            "weather_code": 0,
            "wind_speed_10m": 10.0,
        }
    }

    async def mock_get(url, params=None):
        resp = MagicMock()
        resp.status_code = 200
        if "geocoding-api" in url:
            resp.json.return_value = mock_geo_resp
        else:
            resp.json.return_value = mock_forecast_resp
        return resp

    mock_client.get = mock_get

    res = await get_weather(None, "now", "pl", client=mock_client)
    assert res.error is None
    assert res.data["location"] == "Katowice"
    assert "Katowice" in res.speak


@pytest.mark.asyncio
async def test_get_weather_unknown_location():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"results": []}
    mock_client.get = AsyncMock(return_value=resp)

    res = await get_weather("NibylandiaX99", "now", "pl", client=mock_client)
    assert res.error is not None
    assert "Nie znaleziono" in res.data["error"]


# 3. calculate
@pytest.mark.asyncio
async def test_calculate_basic_and_percentages():
    res = await calculate("15 * 240 / 100")
    assert res.data["result"] == 36

    res_pct = await calculate("15% z 240")
    assert res_pct.data["result"] == 36

    res_pct_pl = await calculate("20 procent z 150")
    assert res_pct_pl.data["result"] == 30

    res_sqrt = await calculate("sqrt(144)")
    assert res_sqrt.data["result"] == 12

    res_pow = await calculate("2 ** 8")
    assert res_pow.data["result"] == 256


@pytest.mark.asyncio
async def test_calculate_safety():
    res_import = await calculate("__import__('os').system('ls')")
    assert res_import.error is not None

    res_zero = await calculate("10 / 0")
    assert res_zero.error is not None
    assert "Dzielenie przez zero" in res_zero.error


# 4. convert_units
@pytest.mark.asyncio
async def test_convert_units_length_and_mass():
    res_len = await convert_units(10, "km", "mile")
    assert res_len.error is None
    assert pytest.approx(res_len.data["result"], 0.01) == 6.214

    res_mass = await convert_units(5, "kg", "funty")
    assert res_mass.error is None
    assert pytest.approx(res_mass.data["result"], 0.01) == 11.023


@pytest.mark.asyncio
async def test_convert_units_temperature():
    res_temp = await convert_units(20, "C", "F")
    assert res_temp.error is None
    assert res_temp.data["result"] == 68

    res_temp_rev = await convert_units(68, "F", "C")
    assert res_temp_rev.error is None
    assert res_temp_rev.data["result"] == 20


@pytest.mark.asyncio
async def test_convert_units_incompatible():
    res = await convert_units(10, "km", "kg")
    assert res.error is not None
    assert "kategorie" in res.data["error"]


@pytest.mark.asyncio
async def test_convert_units_speak_pl_precision_and_grammar():
    res = await convert_units(100, "km", "mi", lang="pl")
    assert res.error is None
    # Exact value in data
    assert pytest.approx(res.data["result"], 1e-4) == 62.1371
    # Nominative and natural spoken fraction
    assert "sto kilometrów to około sześćdziesiąt dwie i jedna dziesiąta mili." in res.speak

    # Fractions and plurals of mila
    res_1 = await convert_units(1.609344, "km", "mi", lang="pl")
    assert "jedna mila" in res_1.speak

    res_2 = await convert_units(3.218688, "km", "mi", lang="pl")
    assert "dwie mile" in res_2.speak

    res_5 = await convert_units(8.04672, "km", "mi", lang="pl")
    assert "pięć mil" in res_5.speak


@pytest.mark.asyncio
async def test_convert_units_speak_en():
    res = await convert_units(100, "km", "mi", lang="en")
    assert res.error is None
    assert pytest.approx(res.data["result"], 1e-4) == 62.1371
    assert "one hundred kilometers is about sixty-two point one miles." in res.speak

    res_1 = await convert_units(1, "km", "mi", lang="en")
    assert "one kilometer is about zero point six miles." in res_1.speak


@pytest.mark.asyncio
async def test_convert_units_m3_declension():
    res_1 = await convert_units(1, "m3", "l", lang="pl")
    assert "jeden metr sześcienny to tysiąc litrów." in res_1.speak

    res_2 = await convert_units(2, "m3", "l", lang="pl")
    assert "dwa metry sześcienne to dwa tysiące litrów." in res_2.speak

    res_5 = await convert_units(5, "m3", "l", lang="pl")
    assert "pięć metrów sześciennych to pięć tysięcy litrów." in res_5.speak


# 5. exchange_rate (mocked)
@pytest.mark.asyncio
async def test_exchange_rate_mocked():
    mock_nbp_resp = [
        {
            "table": "A",
            "effectiveDate": "2026-10-09",
            "rates": [
                {"code": "EUR", "mid": 4.30},
                {"code": "USD", "mid": 3.95},
                {"code": "GBP", "mid": 5.10},
            ],
        }
    ]
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = mock_nbp_resp
    mock_client.get = AsyncMock(return_value=resp)

    res = await exchange_rate("EUR", 100, "PLN", "pl", client=mock_client)
    assert res.error is None
    assert res.data["result"] == 430.0
    assert "czterysta trzydzieści złotych" in res.speak or "430" in res.speak


# 6. wiki_summary (mocked)
@pytest.mark.asyncio
async def test_wiki_summary_mocked():
    mock_resp = {
        "title": "Mikołaj Kopernik",
        "extract": "Mikołaj Kopernik – polski astronom, kanonik, prawnik. Opracował heliocentryczny model Układu Słonecznego. Jego dzieło zrewolucjonizowało naukę.",
    }
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = mock_resp
    mock_client.get = AsyncMock(return_value=resp)

    res = await wiki_summary("Mikołaj Kopernik", "pl", client=mock_client)
    assert res.error is None
    assert res.data["title"] == "Mikołaj Kopernik"
    assert "heliocentryczny" in res.speak


@pytest.mark.asyncio
async def test_wiki_summary_404_resolves_title_via_search():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/search/title"):
            assert request.url.params["q"] == "Hermes"  # parenthetical stripped
            return httpx.Response(200, json={"pages": [{"key": "Hermes"}]})
        if path.endswith("/page/summary/Hermes"):
            return httpx.Response(200, json={"title": "Hermes", "extract": "Hermes – w mitologii greckiej bóg dróg."})
        return httpx.Response(404, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await wiki_summary("Hermes (mitologia)", "pl", client=client)
    assert res.error is None
    assert res.data["title"] == "Hermes"
    assert "mitologii" in res.speak


@pytest.mark.asyncio
async def test_wiki_summary_404_without_search_hit():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/search/title"):
            return httpx.Response(200, json={"pages": []})
        return httpx.Response(404, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await wiki_summary("xqzv", "pl", client=client)
    assert res.error is not None
    assert "Nie znalazłam" in res.speak


# 7. server_status
@pytest.mark.asyncio
async def test_server_status():
    res = await server_status("pl")
    assert res.error is None
    assert "loadavg_1m" in res.data
    assert "ram_total_gb" in res.data
    assert "services" in res.data
    assert "Serwer działa od" in res.speak


# 8. holiday_tool
def test_easter_and_holidays():
    easter_2024 = calculate_easter(2024)
    assert easter_2024 == datetime.date(2024, 3, 31)

    easter_2025 = calculate_easter(2025)
    assert easter_2025 == datetime.date(2025, 4, 20)

    holidays = get_polish_holidays(2026)
    assert len(holidays) == 13
    holiday_names = [h[1] for h in holidays]
    assert "Wszystkich Świętych" in holiday_names
    assert "Boże Ciało" in holiday_names


@pytest.mark.asyncio
async def test_next_holiday():
    res = await next_holiday("pl")
    assert res.error is None
    assert "holiday" in res.data
    assert "days_until" in res.data
    assert "Najbliższe święto to" in res.speak


# 9. ToolRegistry
@pytest.mark.asyncio
async def test_registry_schemas_and_execution():
    registry = ToolRegistry()
    names = registry.names()
    assert len(names) == 12
    assert "get_weather" in names
    assert "calculate" in names
    assert "get_sun" in names
    assert "manage_timer" in names
    assert "remember" in names
    assert "forget" in names

    schemas = registry.schemas("pl")
    assert len(schemas) == 12
    assert all("function" in s for s in schemas)

    # Test execution
    res = await registry.execute("calculate", {"expression": "2 + 2"}, lang="pl")
    assert res.error is None
    assert res.data["result"] == 4

    # Test unknown tool
    res_unknown = await registry.execute("unknown_tool", {}, lang="pl")
    assert res_unknown.error is not None


# 10. Optional live network smoke test
@pytest.mark.network
@pytest.mark.asyncio
async def test_live_network_tools():
    res_w = await get_weather("Warszawa", "now", "pl")
    assert res_w.error is None
    res_c = await exchange_rate("EUR", 1, "PLN", "pl")
    assert res_c.error is None
    res_k = await wiki_summary("Warszawa", "pl")
    assert res_k.error is None
