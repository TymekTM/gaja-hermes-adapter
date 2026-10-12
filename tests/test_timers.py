from __future__ import annotations

import asyncio
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pytest

from gaja.timers import TimerManager, WARSAW_TZ
from gaja.tools.timer_tool import manage_timer, set_timer_manager


@pytest.mark.asyncio
async def test_timer_manager_persistence(tmp_path: Path):
    timers_file = tmp_path / "timers.json"
    mgr1 = TimerManager(timers_file)
    now = datetime.datetime.now(WARSAW_TZ)
    t1 = mgr1.add_timer("pizza", now + datetime.timedelta(minutes=10), kind="timer")
    t2 = mgr1.add_timer("leki", now + datetime.timedelta(hours=2), kind="reminder")

    assert len(mgr1.list_timers()) == 2
    assert timers_file.exists()

    # Reload from file
    mgr2 = TimerManager(timers_file)
    items = mgr2.list_timers()
    assert len(items) == 2
    assert any(i["label"] == "pizza" for i in items)
    assert any(i["label"] == "leki" for i in items)


@pytest.mark.asyncio
async def test_timer_manager_cancel_and_delete(tmp_path: Path):
    timers_file = tmp_path / "timers.json"
    mgr = TimerManager(timers_file)
    now = datetime.datetime.now(WARSAW_TZ)
    t1 = mgr.add_timer("herbata", now + datetime.timedelta(minutes=5), kind="timer")
    t2 = mgr.add_timer("kawa", now + datetime.timedelta(minutes=10), kind="timer")
    t3 = mgr.add_timer("spotkanie", now + datetime.timedelta(hours=1), kind="reminder")

    cancelled = mgr.cancel_timers(label="herbata")
    assert cancelled == [t1["id"]]
    assert len(mgr.list_timers()) == 2

    assert mgr.delete_timer(t2["id"]) is True
    assert len(mgr.list_timers()) == 1
    assert mgr.delete_timer("nonexistent") is False


@pytest.mark.asyncio
async def test_timer_manager_get_status(tmp_path: Path):
    timers_file = tmp_path / "timers.json"
    mgr = TimerManager(timers_file)
    now = datetime.datetime.now(WARSAW_TZ)
    mgr.add_timer("szybki", now + datetime.timedelta(seconds=60), kind="timer")
    mgr.add_timer("wolny", now + datetime.timedelta(seconds=300), kind="timer")

    status = mgr.get_status()
    assert len(status) == 2
    assert status[0]["label"] == "szybki"
    assert status[0]["remaining_seconds"] <= 60
    assert status[1]["label"] == "wolny"


@pytest.mark.asyncio
async def test_timer_manager_past_due_on_startup(tmp_path: Path):
    timers_file = tmp_path / "timers.json"
    mgr1 = TimerManager(timers_file)
    now = datetime.datetime.now(WARSAW_TZ)
    # Add past-due timer
    mgr1.add_timer("dawny", now - datetime.timedelta(minutes=5), kind="timer")
    # Add upcoming timer
    mgr1.add_timer("przyszly", now + datetime.timedelta(minutes=30), kind="timer")

    fired: list[tuple[dict, str | None]] = []

    async def fake_notify(timer: dict, note: str | None = None):
        fired.append((timer, note))

    mgr2 = TimerManager(timers_file, notify_fn=fake_notify)
    await mgr2.start()
    try:
        assert len(fired) == 1
        assert fired[0][0]["label"] == "dawny"
        assert fired[0][1] == "restart"
        # Only upcoming remains in active list
        assert len(mgr2.list_timers()) == 1
        assert mgr2.list_timers()[0]["label"] == "przyszly"
    finally:
        await mgr2.stop()


@pytest.mark.asyncio
async def test_timer_manager_fires_when_due(tmp_path: Path):
    timers_file = tmp_path / "timers.json"
    fired: list[dict] = []

    async def fake_notify(timer: dict, note: str | None = None):
        fired.append(timer)

    mgr = TimerManager(timers_file, notify_fn=fake_notify)
    now = datetime.datetime.now(WARSAW_TZ)
    mgr.add_timer("krotki", now + datetime.timedelta(seconds=0.05), kind="timer")

    await mgr.start()
    try:
        await asyncio.sleep(0.15)
        assert len(fired) == 1
        assert fired[0]["label"] == "krotki"
        assert len(mgr.list_timers()) == 0
    finally:
        await mgr.stop()


@pytest.mark.asyncio
async def test_manage_timer_tool(tmp_path: Path):
    mgr = TimerManager(tmp_path / "timers.json")
    set_timer_manager(mgr)

    # Set timer
    res_set = await manage_timer(action="set", duration_min=5, label="jajka")
    assert res_set.error is None
    assert "Ustawiłam minutnik" in res_set.speak
    assert len(mgr.list_timers()) == 1

    # Check status
    res_stat = await manage_timer(action="status")
    assert res_stat.error is None
    assert "zostało" in res_stat.speak

    # Cancel timer
    res_cancel = await manage_timer(action="cancel", label="jajka")
    assert res_cancel.error is None
    assert "Anulowałam minutnik" in res_cancel.speak
    assert len(mgr.list_timers()) == 0


@pytest.mark.asyncio
async def test_manage_timer_stores_session_context(tmp_path: Path):
    from gaja.tools.base import current_session_context
    mgr = TimerManager(tmp_path / "timers.json")
    set_timer_manager(mgr)
    tok = current_session_context.set({"session_id": "sess_123", "device_id": "dev_456"})
    try:
        res = await manage_timer(action="set", duration_min=5, label="makaron")
        assert res.data["session_id"] == "sess_123"
        assert res.data["device_id"] == "dev_456"
        t = mgr.list_timers()[0]
        assert t["session_id"] == "sess_123"
        assert t["device_id"] == "dev_456"
    finally:
        current_session_context.reset(tok)


@pytest.mark.asyncio
async def test_timer_label_plural_declensions(tmp_path: Path):
    mgr = TimerManager(tmp_path / "timers.json")
    set_timer_manager(mgr)

    res1 = await manage_timer(action="set", duration_min=1)
    assert res1.data["label"] == "1 minuta"
    assert "jedną minutę" in res1.speak

    res2 = await manage_timer(action="set", duration_min=2)
    assert res2.data["label"] == "2 minuty"
    assert "dwie minuty" in res2.speak

    res5 = await manage_timer(action="set", duration_min=5)
    assert res5.data["label"] == "5 minut"
    assert "pięć minut" in res5.speak

    res22 = await manage_timer(action="set", duration_min=22)
    assert res22.data["label"] == "22 minuty"
    assert "dwadzieścia dwie minuty" in res22.speak


def test_format_timer_announcement_natural():
    from gaja.pipeline import format_timer_announcement

    a2 = format_timer_announcement({"label": "2 minuty", "kind": "timer"})
    assert a2 == "Minął minutnik na dwie minuty."

    a1 = format_timer_announcement({"label": "1 minuta", "kind": "timer"})
    assert a1 == "Minął minutnik na jedną minutę."

    a5 = format_timer_announcement({"label": "5 minut", "kind": "timer"})
    assert a5 == "Minął minutnik na pięć minut."

    a22 = format_timer_announcement({"label": "22 minuty", "kind": "timer"})
    assert a22 == "Minął minutnik na dwadzieścia dwie minuty."

    a_custom = format_timer_announcement({"label": "herbata", "kind": "timer"})
    assert a_custom == "Minął minutnik: herbata."

    a_restart = format_timer_announcement({"label": "2 minuty", "kind": "timer"}, note="restart")
    assert "podczas restartu" in a_restart
    assert "na dwie minuty" in a_restart

    a_rem = format_timer_announcement({"label": "lekarz o 18", "kind": "reminder"})
    assert a_rem == "Przypomnienie: lekarz o osiemnaście."
