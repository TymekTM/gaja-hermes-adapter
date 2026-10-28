from __future__ import annotations

import asyncio
import datetime
import json
import logging
from pathlib import Path
from zoneinfo import ZoneInfo
import pytest

from gaja.timers import TimerManager, WARSAW_TZ
from gaja.tools.timer_tool import manage_timer, set_timer_manager


def _freeze_timer_clock(monkeypatch: pytest.MonkeyPatch, moment: datetime.datetime) -> None:
    """Pin ``datetime.datetime.now`` inside timer_tool while keeping the rest of datetime."""
    import types

    import gaja.tools.timer_tool as timer_tool

    class FrozenDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return moment.astimezone(tz) if tz is not None else moment.replace(tzinfo=None)

    monkeypatch.setattr(
        timer_tool,
        "datetime",
        types.SimpleNamespace(
            datetime=FrozenDatetime,
            timedelta=datetime.timedelta,
            timezone=datetime.timezone,
        ),
    )


@pytest.mark.asyncio
async def test_duration_timer_measures_real_time_across_dst_fall_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = TimerManager(tmp_path / "timers.json")
    set_timer_manager(mgr)
    # 02:30 CEST on the fall-back night; the clock repeats 02:00-03:00 an hour later.
    moment = datetime.datetime(2026, 10, 25, 2, 30, tzinfo=WARSAW_TZ, fold=0)
    _freeze_timer_clock(monkeypatch, moment)

    res = await manage_timer(action="set", duration_min=60, label="herbata")
    due = datetime.datetime.fromisoformat(res.data["due"])
    elapsed = (due.astimezone(datetime.timezone.utc) - moment.astimezone(datetime.timezone.utc)).total_seconds()
    assert elapsed == 3600


@pytest.mark.asyncio
async def test_reminder_label_digits_are_not_parsed_as_an_hour(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = TimerManager(tmp_path / "timers.json")
    set_timer_manager(mgr)
    moment = datetime.datetime(2026, 10, 27, 12, 0, tzinfo=WARSAW_TZ)
    _freeze_timer_clock(monkeypatch, moment)

    for label in ("kupić 2 bilety", "zapłać 500 zł"):
        res = await manage_timer(action="set", kind="reminder", label=label)
        due = datetime.datetime.fromisoformat(res.data["due"])
        assert due.hour == 12 and due.minute == 30, label
        assert due.date() == moment.date(), label
        assert label in res.data["label"]

    # An explicit time expression still sets the hour.
    res = await manage_timer(action="set", kind="reminder", time="18:30", label="leki")
    due = datetime.datetime.fromisoformat(res.data["due"])
    assert (due.hour, due.minute) == (18, 30)


@pytest.mark.asyncio
async def test_reminder_honours_duration_min(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = TimerManager(tmp_path / "timers.json")
    set_timer_manager(mgr)
    moment = datetime.datetime(2026, 10, 27, 12, 0, tzinfo=WARSAW_TZ)
    _freeze_timer_clock(monkeypatch, moment)

    res = await manage_timer(action="set", kind="reminder", label="zadzwonić do mamy", duration_min=10)
    due = datetime.datetime.fromisoformat(res.data["due"])
    assert due == moment + datetime.timedelta(minutes=10)
    assert "dwunastej dziesięć" in res.speak


@pytest.mark.asyncio
async def test_cancel_respects_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = TimerManager(tmp_path / "timers.json")
    set_timer_manager(mgr)
    _freeze_timer_clock(monkeypatch, datetime.datetime(2026, 10, 27, 12, 0, tzinfo=WARSAW_TZ))
    await manage_timer(action="set", duration_min=5, label="herbata")
    await manage_timer(action="set", kind="reminder", time="18:30", label="leki")

    res = await manage_timer(action="cancel", kind="reminder")
    assert [t["label"] for t in mgr.list_timers()] == ["herbata"]
    assert "przypomnienie" in res.speak

    res = await manage_timer(action="cancel", kind="timer")
    assert mgr.list_timers() == []
    assert "minutnik" in res.speak


@pytest.mark.asyncio
async def test_due_timer_survives_until_a_session_receives_it(tmp_path: Path) -> None:
    timers_file = tmp_path / "timers.json"
    delivered: list[tuple[str, str | None]] = []
    connected = False

    async def notify(timer: dict, note: str | None = None):
        if not connected:
            return False  # notify_sessions: no sessions, nobody heard it
        delivered.append((timer["label"], note))
        return True

    mgr = TimerManager(timers_file, notify_fn=notify)
    mgr.add_timer("leki", datetime.datetime.now(WARSAW_TZ) - datetime.timedelta(minutes=1), kind="reminder")
    await mgr.start()
    try:
        assert delivered == []
    finally:
        await mgr.stop()

    # A restart with nobody connected must not lose it either; the next start that has a
    # session announces it with the restart note.
    connected = True
    mgr2 = TimerManager(timers_file, notify_fn=notify)
    await mgr2.start()
    try:
        assert delivered == [("leki", "restart")]
        assert mgr2.pending_timers() == []
    finally:
        await mgr2.stop()

    assert TimerManager(timers_file).pending_timers() == []


@pytest.mark.asyncio
async def test_timer_firing_without_a_session_stays_pending(tmp_path: Path) -> None:
    timers_file = tmp_path / "timers.json"
    delivered: list[dict] = []
    connected = False

    async def notify(timer: dict, note: str | None = None):
        if not connected:
            return False
        delivered.append(timer)
        return True

    mgr = TimerManager(timers_file, notify_fn=notify)
    await mgr.start()
    try:
        mgr.add_timer("krotki", datetime.datetime.now(WARSAW_TZ) + datetime.timedelta(seconds=0.05))
        await asyncio.sleep(0.15)
        assert delivered == []
    finally:
        await mgr.stop()

    connected = True
    mgr2 = TimerManager(timers_file, notify_fn=notify)
    await mgr2.start()
    try:
        assert [t["label"] for t in delivered] == ["krotki"]
        assert mgr2.pending_timers() == []
    finally:
        await mgr2.stop()


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
    assert "zosta" in res_stat.speak

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


@pytest.mark.asyncio
async def test_malformed_entries_are_skipped_not_fatal(tmp_path: Path, caplog) -> None:
    """A malformed entry in timers.json (hand-edited file, old format) must not crash
    start() -- and with it the adapter boot. Malformed entries are skipped and logged;
    the valid entries around them still work."""
    timers_file = tmp_path / "timers.json"
    timers_file.write_text(
        json.dumps(
            {
                "timers": [
                    None,  # not even a dict
                    {"id": "t_no_due", "label": "bez terminu"},  # no readable 'due'
                    {"label": "bez id", "due": "2099-01-01T10:00:00+01:00"},  # no id to schedule
                    {"id": "t_ok", "label": "herbata", "kind": "timer", "due": "2099-01-01T10:00:00+01:00"},
                ],
                "pending": [
                    None,  # crashed flush_pending with AttributeError before the fix
                    {"note": None},  # no timer object at all
                    {"timer": "leki", "note": None},  # timer is not a dict
                    {"timer": {"label": "bez id"}, "note": None},  # timer without an id
                    {"timer": {"id": "p1", "label": "leki", "kind": "reminder"}, "note": "restart"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    announced: list[str] = []

    async def notify(timer: dict, note: str | None = None):
        announced.append(timer["label"])
        return True

    with caplog.at_level(logging.WARNING, logger="gaja.timers"):
        mgr = TimerManager(timers_file, notify_fn=notify)
        await mgr.start()  # used to raise AttributeError / TypeError
    try:
        await mgr.stop()
        assert announced == ["leki"], "the one valid pending entry must still be announced"
        assert any(t.get("id") == "t_ok" for t in mgr.list_timers())
        assert mgr.pending_timers() == []
        assert any("Skipping malformed" in r.getMessage() for r in caplog.records)
    finally:
        await mgr.stop()


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        # Dosage counts: "po <1-6>" is an hour only when a time-of-day word follows.
        ("po 2 tabletkach", None),
        ("po 2 tabletki", None),
        ("po 2 i pół tabletki", None),
        ("po 2 na raz", None),
        ("po 2 do każdego posiłku", None),
        ("po 2 w opakowaniu", None),
        ("po 2 za każdym razem", None),
        ("po 2 łyżki", None),
        ("po 2 x dziennie", None),
        ("po 2 godziny", None),
        ("co 2 godziny", None),
        # A decimal comma is a number, never a time.
        ("po 1,5 tabletki", None),
        ("po 2,5 tabletki", None),
        # "o <number> <noun>" was already a count and stays one.
        ("o 2 tabletkach", None),
        # Durations stay durations.
        ("po 15 minut", None),
        # Real times keep working.
        ("przypomnij mi po 18", (18, 0)),
        ("po 8", (8, 0)),
        ("po 2 rano", (2, 0)),
        ("po 2 wieczorem", (2, 0)),
        ("o 18", (18, 0)),
        ("o 22 w nocy", (22, 0)),
        ("leki o 7 rano", (7, 0)),
        ("o 18:30", (18, 30)),
        ("wziąć lek po 14:30", (14, 30)),
        ("zadzwonić po 18 do mamy", (18, 0)),
        ("weź po 2 tabletki o 20", (20, 0)),
        # Numbers without a cue are not times.
        ("kupić 2 bilety", None),
        ("zapłać 500 zł", None),
        ("remind me at 5 pm", None),
    ],
)
def test_time_phrases_from_the_review(phrase: str, expected: tuple[int, int] | None) -> None:
    """F3: after "po" (not "o") an hour 1-6 without :MM is a time only when followed by a
    time-of-day word; function words (w|we|na|do|za|i|oraz) must not license a count as a
    time, and a following ",<digit>" is always a number."""
    from gaja.tools.timer_tool import _parse_time_str

    assert _parse_time_str(phrase) == expected


def test_time_expression_stripping_agrees_with_parsing() -> None:
    """The label cleaner must remove exactly the spans the parser treats as a time."""
    from gaja.tools.timer_tool import _TIME_EXPRESSION_RE, _parse_time_str

    for phrase, expected in (
        ("przypomnij mi po 18", "przypomnij mi"),
        ("zadzwonić po 18 do mamy", "zadzwonić do mamy"),
        ("weź po 2 tabletki o 20", "weź po 2 tabletki"),
        ("przypomnij o 2 tabletkach", "przypomnij o 2 tabletkach"),
        ("po 2 na raz", "po 2 na raz"),
        ("po 1,5 tabletki", "po 1,5 tabletki"),
    ):
        assert " ".join(_TIME_EXPRESSION_RE.sub("", phrase).split()) == expected, phrase


@pytest.mark.asyncio
async def test_reminder_po_hour_sets_the_hour(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = TimerManager(tmp_path / "timers.json")
    set_timer_manager(mgr)
    moment = datetime.datetime(2026, 10, 27, 12, 0, tzinfo=WARSAW_TZ)
    _freeze_timer_clock(monkeypatch, moment)

    res = await manage_timer(action="set", kind="reminder", label="przypomnij mi po 18")
    due = datetime.datetime.fromisoformat(res.data["due"])
    assert (due.hour, due.minute) == (18, 0)
    assert res.data["label"] == "przypomnij mi"

    # "po <number> <noun>" is a count: no hour, default duration, label kept intact
    res2 = await manage_timer(action="set", kind="reminder", label="po 2 tabletkach")
    due2 = datetime.datetime.fromisoformat(res2.data["due"])
    assert (due2.hour, due2.minute) == (12, 30)
    assert res2.data["label"] == "po 2 tabletkach"

    # a mid-phrase cue is stripped from the label without leaving a double space
    res3 = await manage_timer(action="set", kind="reminder", label="zadzwonić po 18 do mamy")
    assert res3.data["label"] == "zadzwonić do mamy"


@pytest.mark.asyncio
async def test_two_concurrent_flushes_announce_a_timer_once(tmp_path: Path) -> None:
    """Two hellos racing (both observe has_pending() before either flush runs) must announce
    the pending timer exactly once -- notify now awaits, so flush has suspension points."""
    timers_file = tmp_path / "timers.json"
    announced: list[str] = []

    async def notify(timer: dict, note: str | None = None):
        await asyncio.sleep(0.01)  # any suspension opens the double-announce window
        announced.append(timer["label"])
        return True

    mgr = TimerManager(timers_file, notify_fn=notify)
    await mgr.start()
    try:
        mgr._pending.append({"timer": {"id": "t1", "label": "leki", "kind": "reminder"}, "note": None})
        mgr._save()
        assert mgr.has_pending()
        first = asyncio.create_task(mgr.flush_pending())
        second = asyncio.create_task(mgr.flush_pending())
        await asyncio.gather(first, second)
        assert announced == ["leki"], f"announced twice: {announced}"
        assert mgr.pending_timers() == []
    finally:
        await mgr.stop()


@pytest.mark.asyncio
async def test_fire_racing_a_flush_announces_once(tmp_path: Path) -> None:
    """A timer firing while a hello flush is in flight must be announced once, not twice."""
    timers_file = tmp_path / "timers.json"
    release = asyncio.Event()
    announced: list[str] = []

    async def notify(timer: dict, note: str | None = None):
        await release.wait()
        announced.append(timer["label"])
        return True

    mgr = TimerManager(timers_file, notify_fn=notify)
    await mgr.start()
    try:
        timer = {"id": "t1", "label": "zupa", "kind": "timer", "due": datetime.datetime.now(WARSAW_TZ).isoformat()}
        mgr._timers.append(timer)
        mgr._pending.append({"timer": dict(timer), "note": None})
        fire = asyncio.create_task(mgr._fire("t1"))
        flush = asyncio.create_task(mgr.flush_pending())
        await asyncio.sleep(0.05)  # both are now inside notify, blocked on the event
        assert announced == []
        release.set()
        await asyncio.gather(fire, flush)
        assert announced == ["zupa"], f"announced {announced}"
        assert mgr.pending_timers() == []
    finally:
        await mgr.stop()


@pytest.mark.asyncio
async def test_flushed_pending_timer_gets_the_late_wording(tmp_path: Path) -> None:
    """F6: a timer announced long after it came due must not sound like it just fired
    ("Minął minutnik!"). The flush passes the late note -- like the restart path -- so
    the announcement says it happened earlier ("podczas restartu")."""
    from gaja.pipeline import format_timer_announcement

    timers_file = tmp_path / "timers.json"
    notes: list[str | None] = []

    async def notify(timer: dict, note: str | None = None):
        notes.append(note)
        return True

    mgr = TimerManager(timers_file, notify_fn=notify)
    await mgr.start()
    try:
        # entry written by a live fire that nobody heard: no note on disk
        mgr._pending.append({"timer": {"id": "t1", "label": "2 minuty", "kind": "timer"}, "note": None})
        mgr._save()
        await mgr.flush_pending()
        assert notes == ["restart"], f"flush must deliver the late note, got {notes}"
        assert mgr.pending_timers() == []
    finally:
        await mgr.stop()

    assert "podczas restartu" in format_timer_announcement({"label": "2 minuty", "kind": "timer"}, "restart")


# --------------------------------------------------------------------------
# F1: a flush must iterate a snapshot of _pending and remove exactly the
# entries it delivered. A different timer firing during the await must be
# neither announced twice nor dropped from memory/timers.json.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fire_of_another_timer_during_a_flush_announces_both_once(tmp_path: Path) -> None:
    """P1 of the round-2 review: the flush iterated the live list, so a timer that fired
    (and was delivered) while the flush was awaiting the notifier was announced again."""
    timers_file = tmp_path / "timers.json"
    announced: list[str] = []
    gate = asyncio.Event()
    release = asyncio.Event()

    async def notify(timer: dict, note: str | None = None):
        if timer["id"] == "A":
            gate.set()  # the flush is now parked inside notify(A)
            await release.wait()
        announced.append(timer["id"])
        return True

    mgr = TimerManager(timers_file, notify_fn=notify)
    await mgr.start()
    try:
        mgr._pending.append({"timer": {"id": "A", "label": "A", "kind": "timer"}, "note": "restart"})
        mgr._timers.append(
            {"id": "B", "label": "B", "kind": "timer", "due": datetime.datetime.now(WARSAW_TZ).isoformat()}
        )
        flush = asyncio.create_task(mgr.flush_pending())
        await gate.wait()
        await asyncio.create_task(mgr._fire("B"))  # live fire of a *different* timer
        release.set()
        await flush

        assert announced.count("B") == 1, f"B announced {announced.count('B')}x: {announced}"
        assert announced.count("A") == 1, f"A announced {announced.count('A')}x: {announced}"
        assert mgr.pending_timers() == []
        assert json.loads(timers_file.read_text(encoding="utf-8"))["pending"] == []
    finally:
        await mgr.stop()


@pytest.mark.asyncio
async def test_flush_keeps_an_entry_appended_while_it_awaited_the_notifier(tmp_path: Path) -> None:
    """P3 of the round-2 review: after a live fire of timer B rebound _pending, timer C was
    appended to the new list and (unheard) must stay there and in timers.json."""
    timers_file = tmp_path / "timers.json"
    announced: list[str] = []
    gate = asyncio.Event()
    release = asyncio.Event()

    async def notify(timer: dict, note: str | None = None):
        announced.append(timer["id"])
        if timer["id"] == "A":
            gate.set()
            await release.wait()
            return False  # A unheard
        if timer["id"] == "B":
            return True  # B heard live
        return False  # C unheard

    mgr = TimerManager(timers_file, notify_fn=notify)
    await mgr.start()
    try:
        due = datetime.datetime.now(WARSAW_TZ).isoformat()
        mgr._pending.append({"timer": {"id": "A", "label": "A", "kind": "timer"}, "note": "restart"})
        for tid in ("B", "C"):
            mgr._timers.append({"id": tid, "label": tid, "kind": "timer", "due": due})
        flush = asyncio.create_task(mgr.flush_pending())
        await gate.wait()
        await asyncio.create_task(mgr._fire("B"))  # delivered -> rebinds _pending
        await asyncio.create_task(mgr._fire("C"))  # appended to the new list, unheard
        release.set()
        await flush

        pending_ids = [t["id"] for t in mgr.pending_timers()]
        on_disk = [e["timer"]["id"] for e in json.loads(timers_file.read_text(encoding="utf-8"))["pending"]]
        assert "C" in pending_ids, f"unheard C dropped from memory: {pending_ids}"
        assert "C" in on_disk, f"unheard C dropped from timers.json: {on_disk}"
        assert "A" in pending_ids, f"A was not delivered and must stay pending: {pending_ids}"
        assert announced.count("B") == 1
    finally:
        await mgr.stop()


# --------------------------------------------------------------------------
# F4: a fired-but-unheard timer must be cancellable, deletable and visible as
# overdue instead of appearing in no list while still being spoken later.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_timer_is_overdue_in_status_and_list(tmp_path: Path) -> None:
    timers_file = tmp_path / "timers.json"
    heard: list[str] = []
    online = False

    async def notify(timer: dict, note: str | None = None):
        if not online:
            return False
        heard.append(timer["label"])
        return True

    mgr = TimerManager(timers_file, notify_fn=notify)
    await mgr.start()
    try:
        timer = mgr.add_timer(
            "leki", datetime.datetime.now(WARSAW_TZ) + datetime.timedelta(seconds=0.05), kind="reminder"
        )
        await asyncio.sleep(0.15)
        assert [t["label"] for t in mgr.pending_timers()] == ["leki"]

        status = mgr.get_status()
        assert [s["id"] for s in status] == [timer["id"]]
        assert status[0]["overdue"] is True
        assert status[0]["remaining_seconds"] == 0
        # field naming is the same as for a scheduled timer
        scheduled = mgr.add_timer(
            "herbata", datetime.datetime.now(WARSAW_TZ) + datetime.timedelta(minutes=5), kind="timer"
        )
        active = [s for s in mgr.get_status() if s["id"] == scheduled["id"]][0]
        assert set(active) - set(status[0]) == set() and active["overdue"] is False

        listed = mgr.list_timers()
        assert {t["id"] for t in listed} == {timer["id"], scheduled["id"]}
        assert {t["id"]: t["overdue"] for t in listed} == {timer["id"]: True, scheduled["id"]: False}
    finally:
        await mgr.stop()


@pytest.mark.asyncio
async def test_cancel_removes_a_pending_timer_so_it_is_never_spoken(tmp_path: Path) -> None:
    timers_file = tmp_path / "timers.json"
    heard: list[str] = []
    online = False

    async def notify(timer: dict, note: str | None = None):
        if not online:
            return False
        heard.append(timer["label"])
        return True

    mgr = TimerManager(timers_file, notify_fn=notify)
    await mgr.start()
    try:
        timer = mgr.add_timer(
            "leki", datetime.datetime.now(WARSAW_TZ) + datetime.timedelta(seconds=0.05), kind="reminder"
        )
        await asyncio.sleep(0.15)
        assert mgr.pending_timers()

        assert mgr.cancel_timers(label="leki", kind="reminder") == [timer["id"]]
        assert mgr.pending_timers() == []
        assert json.loads(timers_file.read_text(encoding="utf-8"))["pending"] == []

        online = True  # the user reconnects: nothing may be announced for the cancelled reminder
        await mgr.flush_pending()
        assert heard == []
    finally:
        await mgr.stop()


@pytest.mark.asyncio
async def test_delete_removes_a_pending_timer_so_it_is_never_spoken(tmp_path: Path) -> None:
    timers_file = tmp_path / "timers.json"
    heard: list[str] = []
    online = False

    async def notify(timer: dict, note: str | None = None):
        if not online:
            return False
        heard.append(timer["label"])
        return True

    mgr = TimerManager(timers_file, notify_fn=notify)
    await mgr.start()
    try:
        timer = mgr.add_timer(
            "herbata", datetime.datetime.now(WARSAW_TZ) + datetime.timedelta(seconds=0.05), kind="timer"
        )
        await asyncio.sleep(0.15)
        assert mgr.pending_timers()

        assert mgr.delete_timer(timer["id"]) is True
        assert mgr.pending_timers() == []
        assert json.loads(timers_file.read_text(encoding="utf-8"))["pending"] == []

        online = True
        await mgr.flush_pending()
        assert heard == []
    finally:
        await mgr.stop()


# --------------------------------------------------------------------------
# F5: a pending timer held back only by quiet hours must be flushed once,
# at the end of the window, without waiting for the device to reconnect.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quiet_hours_schedule_a_flush_for_the_end_of_the_window(tmp_path: Path) -> None:
    """A device connected all night (no separate hello) must still hear the reminder after
    07:00: the manager arms one flush at the window end when delivery was held by quiet hours."""
    timers_file = tmp_path / "timers.json"
    heard: list[str] = []
    online = False
    quiet = "22:00-07:00"
    # 0.6 s before the window ends; a fixed clock keeps the test independent of the host time.
    moment = datetime.datetime(2026, 10, 28, 6, 59, 59, 400000, tzinfo=WARSAW_TZ)

    async def notify(timer: dict, note: str | None = None):
        if not online:
            return False  # only a device speaker is connected, and it is quiet hours
        heard.append(timer["label"])
        return True

    mgr = TimerManager(
        timers_file,
        notify_fn=notify,
        quiet_hours_fn=lambda: quiet,
        now_fn=lambda: moment,
    )
    await mgr.start()
    try:
        mgr._pending.append({"timer": {"id": "t1", "label": "leki", "kind": "reminder"}, "note": "restart"})
        await mgr.flush_pending()
        assert mgr.pending_timers(), "held by quiet hours, so it must stay pending"
        assert mgr._quiet_flush_task is not None, "quiet hours must schedule the end-of-window flush"

        online = True  # nobody reconnects; the scheduled flush is what delivers it
        await asyncio.sleep(0.9)
        assert heard == ["leki"], f"the end-of-window flush did not announce it: {heard}"
        assert mgr.pending_timers() == []
    finally:
        await mgr.stop()


@pytest.mark.asyncio
async def test_quiet_hours_keep_at_most_one_flush_task_and_cancel_it_on_stop(tmp_path: Path) -> None:
    quiet = "22:00-07:00"
    moment = datetime.datetime(2026, 10, 28, 23, 0, tzinfo=WARSAW_TZ)  # eight hours to go

    async def notify(timer: dict, note: str | None = None):
        return False

    mgr = TimerManager(
        tmp_path / "timers.json",
        notify_fn=notify,
        quiet_hours_fn=lambda: quiet,
        now_fn=lambda: moment,
    )
    await mgr.start()
    task = None
    try:
        mgr._pending.append({"timer": {"id": "t1", "label": "leki", "kind": "reminder"}, "note": "restart"})
        await mgr.flush_pending()
        task = mgr._quiet_flush_task
        assert task is not None
        # Another failed flush (e.g. a second hello) must not pile up tasks.
        await mgr.flush_pending()
        assert mgr._quiet_flush_task is task
    finally:
        await mgr.stop()
    await asyncio.sleep(0)
    assert task is not None and task.cancelled(), "stop() must cancel the scheduled flush"


@pytest.mark.asyncio
async def test_no_quiet_hours_flush_is_armed_outside_the_window(tmp_path: Path) -> None:
    async def notify(timer: dict, note: str | None = None):
        return False

    mgr = TimerManager(
        tmp_path / "timers.json",
        notify_fn=notify,
        quiet_hours_fn=lambda: "22:00-07:00",
        now_fn=lambda: datetime.datetime(2026, 10, 28, 12, 0, tzinfo=WARSAW_TZ),
    )
    await mgr.start()
    try:
        mgr._pending.append({"timer": {"id": "t1", "label": "leki", "kind": "reminder"}, "note": "restart"})
        await mgr.flush_pending()
        assert mgr._quiet_flush_task is None
        assert mgr.pending_timers(), "an unheard timer still stays pending"
    finally:
        await mgr.stop()
