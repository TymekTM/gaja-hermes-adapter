from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo
import pytest

from gaja.tools.convert_tool import convert_units
from gaja.tools.holiday_tool import next_holiday
from gaja.tools.time_tool import get_time


@pytest.mark.asyncio
async def test_next_holiday_default_and_specific():
    # Default next holiday
    res_default = await next_holiday()
    assert res_default.error is None
    assert "holiday" in res_default.data
    assert "days_until" in res_default.data
    assert res_default.data["days_until"] >= 0
    assert "Najbliższe święto to" in res_default.speak or "Dzisiaj jest" in res_default.speak

    # Specific holiday: Boże Narodzenie
    res_xmas = await next_holiday(holiday="Boże Narodzenie")
    assert res_xmas.error is None
    assert "Bożego Narodzenia" in res_xmas.data["holiday"]
    assert "dni" in res_xmas.speak or "dzień" in res_xmas.speak

    # Specific holiday: Wielkanoc
    res_easter = await next_holiday(holiday="Wielkanoc")
    assert res_easter.error is None
    assert "Wielkanoc" in res_easter.data["holiday"]


@pytest.mark.asyncio
async def test_get_time_date_arithmetic():
    # Regular time
    res_now = await get_time()
    assert res_now.error is None
    assert "time" in res_now.data

    # Target date arithmetic: 3 listopada
    res_target = await get_time(target_date="3 listopada")
    assert res_target.error is None
    assert "target_date" in res_target.data
    assert "weekday" in res_target.data
    assert "days_until" in res_target.data
    assert res_target.data["month"] == 11
    assert res_target.data["day"] == 3
    assert "listopada" in res_target.speak

    # Relative target date: jutro
    res_tomorrow = await get_time(target_date="jutro")
    assert res_tomorrow.error is None
    assert res_tomorrow.data["days_until"] == 1
    assert "jutro" in res_tomorrow.speak


@pytest.mark.asyncio
async def test_convert_cooking_units():
    # 2 szklanki to ml (1 szklanka = 0.25 l = 250 ml -> 2 szklanki = 500 ml)
    res_cup = await convert_units(2, "szklanki", "ml")
    assert res_cup.error is None
    assert res_cup.data["result"] == 500.0

    # 3 łyżki to ml (1 tbsp = 15 ml -> 45 ml)
    res_tbsp = await convert_units(3, "łyżki", "ml")
    assert res_tbsp.error is None
    assert res_tbsp.data["result"] == 45.0

    # 4 łyżeczki to ml (1 tsp = 5 ml -> 20 ml)
    res_tsp = await convert_units(4, "łyżeczki", "ml")
    assert res_tsp.error is None
    assert res_tsp.data["result"] == 20.0


@pytest.mark.asyncio
async def test_convert_data_units():
    # 1024 bajty to kB (1 kB = 1024 B)
    res_kb = await convert_units(1024, "bajt", "kB")
    assert res_kb.error is None
    assert res_kb.data["result"] == 1.0

    # 1 GB to MB (1024 MB)
    res_gb = await convert_units(1, "GB", "MB")
    assert res_gb.error is None
    assert res_gb.data["result"] == 1024.0

    # 1 TB to GB (1024 GB)
    res_tb = await convert_units(1, "TB", "GB")
    assert res_tb.error is None
    assert res_tb.data["result"] == 1024.0

    # 8 bitów to bajt (1 bajt)
    res_bit = await convert_units(8, "bit", "bajt")
    assert res_bit.error is None
    assert res_bit.data["result"] == 1.0


def test_holiday_cues_match_boze_narodzenie():
    from gaja.llm.router import TOOL_CUES
    cue = TOOL_CUES["next_holiday"]
    assert cue.search("ile dni zostało do Bożego Narodzenia?")
    assert cue.search("ile dni do świąt Bożego Narodzenia")
    assert cue.search("kiedy jest Boże Narodzenie")
    assert cue.search("ile dni zostało do Nowego Roku")
    assert cue.search("ile dni do Wielkanocy")


def test_system_prompt_injects_current_date():
    from gaja.pipeline import Services, SystemPrompt
    from gaja.config import Settings
    services = Services(
        cfg=Settings(),
        stt=None,
        tts=None,
        llm=None,
        prompt=SystemPrompt(text="Jesteś Gaja.", tools=[]),
        hermes=None,
        metrics=None,
        new_vad=None,
    )
    prompt = services.get_system_prompt()
    assert "Aktualny czas i data (Warszawa):" in prompt
    assert "roku" in prompt
