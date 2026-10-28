from __future__ import annotations

import asyncio
import datetime
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

from gaja.devices.quiet_hours import is_quiet_hour, parse_quiet_hours

log = logging.getLogger(__name__)

WARSAW_TZ = ZoneInfo("Europe/Warsaw")


def _pending_timer(entry: Any) -> dict[str, Any] | None:
    """The timer object inside a _pending entry (None for a malformed entry)."""
    timer = entry.get("timer") if isinstance(entry, dict) else None
    return timer if isinstance(timer, dict) else None


def _pending_id(entry: Any) -> str | None:
    timer = _pending_timer(entry)
    tid = timer.get("id") if timer is not None else None
    return str(tid) if tid else None


def _quiet_hours_end(window: tuple[datetime.time, datetime.time], now: datetime.datetime) -> datetime.datetime:
    """The next moment the quiet window ends (today, or tomorrow when it wraps midnight)."""
    _start, end = window
    end_dt = now.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)
    if end_dt <= now:
        end_dt += datetime.timedelta(days=1)
    return end_dt


class TimerManager:
    def __init__(
        self,
        file_path: Path | str = "data/timers.json",
        notify_fn: Callable[[dict[str, Any], str | None], Awaitable[bool | None]] | None = None,
        *,
        quiet_hours_fn: Callable[[], str] | None = None,
        now_fn: Callable[[], datetime.datetime] | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.notify_fn = notify_fn
        # Quiet hours ("HH:MM-HH:MM", read at call time because the panel can change it) and
        # the clock. When delivery is held only by quiet hours the manager schedules one
        # flush for the end of the window, so a device connected overnight hears the timer.
        self._quiet_hours_fn = quiet_hours_fn
        self._now_fn = now_fn or (lambda: datetime.datetime.now(WARSAW_TZ))
        self._quiet_flush_task: asyncio.Task[None] | None = None
        self._timers: list[dict[str, Any]] = []
        # Timers that came due but were not handed to any session yet (nobody was connected,
        # every send failed, or only device speakers were connected during quiet hours).
        # Kept across restarts so the next hello announces them.
        self._pending: list[dict[str, Any]] = []
        self._tasks: dict[str, asyncio.Task[None]] = {}
        # Ids currently being delivered, plus a flush-in-progress flag: notify now awaits
        # (delivery truth with a bound), so a flush racing a fire -- or two hellos racing
        # each other -- must not announce the same timer twice.
        self._inflight: set[str] = set()
        self._flushing = False
        self._started = False
        self._load()

    def _load(self) -> None:
        if not self.file_path.exists():
            self._timers = []
            self._pending = []
            return
        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                timers = list(data.get("timers") or [])
                pending = list(data.get("pending") or [])
            elif isinstance(data, list):
                timers = list(data)
                pending = []
            else:
                timers = []
                pending = []
        except Exception as exc:
            log.warning("Could not read timers file %s: %s", self.file_path, exc)
            timers = []
            pending = []
        # A malformed entry (hand-edited file, old format) must not crash start() -- and with
        # it the adapter boot: skip it and say so.
        self._timers = [t for t in timers if isinstance(t, dict)]
        for t in timers:
            if not isinstance(t, dict):
                log.warning("Skipping malformed timer entry in %s: %r", self.file_path, t)
        self._pending = []
        for entry in pending:
            timer = entry.get("timer") if isinstance(entry, dict) else None
            if isinstance(timer, dict) and timer.get("id"):
                self._pending.append(entry)
            else:
                log.warning("Skipping malformed pending entry in %s: %r", self.file_path, entry)

    def _save(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file_path.with_suffix(".tmp")
        payload = {"timers": self._timers, "pending": self._pending}
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.file_path)

    async def start(self) -> None:
        """Start scheduler. Fire past-due timers with a note and schedule upcoming ones."""
        self._started = True
        now = datetime.datetime.now(WARSAW_TZ)
        past_due: list[dict[str, Any]] = []
        upcoming: list[dict[str, Any]] = []

        for timer in list(self._timers):
            try:
                due_dt = datetime.datetime.fromisoformat(timer["due"])
                if due_dt.tzinfo is None:
                    due_dt = due_dt.replace(tzinfo=WARSAW_TZ)
            except Exception:
                log.warning("Skipping timer with an unreadable 'due' in %s: %r", self.file_path, timer)
                continue

            if due_dt <= now:
                past_due.append(timer)
            else:
                upcoming.append(timer)

        # Keep past due until a session actually receives them, not only in memory.
        self._timers = upcoming
        for t in past_due:
            self._pending.append({"timer": t, "note": "restart"})
        self._save()

        await self.flush_pending()

        # Schedule upcoming
        for t in list(self._timers):
            self._schedule(t)

    def _schedule(self, timer: dict[str, Any]) -> None:
        timer_id = timer.get("id") if isinstance(timer, dict) else None
        if not timer_id:
            log.warning("Cannot schedule timer without an id: %r", timer)
            return
        if timer_id in self._tasks:
            self._tasks[timer_id].cancel()

        try:
            due_dt = datetime.datetime.fromisoformat(timer["due"])
            if due_dt.tzinfo is None:
                due_dt = due_dt.replace(tzinfo=WARSAW_TZ)
        except Exception:
            return

        now = datetime.datetime.now(WARSAW_TZ)
        delay_s = max(0.0, (due_dt - now).total_seconds())

        async def _run() -> None:
            try:
                await asyncio.sleep(delay_s)
                await self._fire(timer_id)
            except asyncio.CancelledError:
                pass

        self._tasks[timer_id] = asyncio.create_task(_run())

    async def _fire(self, timer_id: str, note: str | None = None) -> None:
        found = None
        for i, t in enumerate(self._timers):
            if t.get("id") == timer_id:
                found = self._timers.pop(i)
                break
        self._tasks.pop(timer_id, None)
        if found:
            self._pending.append({"timer": found, "note": note})
            self._save()
            if await self._deliver(found, note):
                self._pending = [p for p in self._pending if _pending_id(p) != timer_id]
                self._save()
            else:
                self._arm_quiet_hours_flush()

    def _arm_quiet_hours_flush(self) -> None:
        """Schedule one flush for the end of the quiet window after an unheard delivery.

        During quiet hours a pending timer is deliberately kept off device speakers, so a
        device that stays connected all night would otherwise hear it only on a reconnect.
        No-op outside quiet hours, and at most one task at a time; stop() cancels it.
        """
        if self._quiet_hours_fn is None:
            return
        if self._quiet_flush_task is not None and not self._quiet_flush_task.done():
            return
        window = parse_quiet_hours(str(self._quiet_hours_fn() or ""))
        if window is None:
            return
        now = self._now_fn()
        if not is_quiet_hour(str(self._quiet_hours_fn() or ""), now=now):
            return  # not quiet hours: the failed delivery had another reason (nobody connected)
        delay = max(0.0, (_quiet_hours_end(window, now) - now).total_seconds())

        async def _run() -> None:
            try:
                await asyncio.sleep(delay)
                await self.flush_pending()
            except asyncio.CancelledError:
                pass
            finally:
                if self._quiet_flush_task is asyncio.current_task():
                    self._quiet_flush_task = None

        self._quiet_flush_task = asyncio.create_task(_run())
        log.info("Quiet hours: flushing pending timers in %.0f s", delay)

    async def _deliver(self, timer: dict[str, Any], note: str | None) -> bool:
        """Hand a timer to the notifier; False means nobody received it and it must stay pending."""
        if self.notify_fn is None:
            return True
        tid = str(timer.get("id") or uuid.uuid4().hex)
        if tid in self._inflight:
            return False  # another flush/fire is announcing this timer right now
        self._inflight.add(tid)
        try:
            result = await self.notify_fn(timer, note)
        except Exception as exc:
            log.exception("Error in timer notify_fn: %s", exc)
            return False
        finally:
            self._inflight.discard(tid)
        # Existing/test notifiers return None; only an explicit False means "no session".
        return result is not False

    def pending_timers(self) -> list[dict[str, Any]]:
        """Fired timers still waiting for a session to announce them."""
        return [entry["timer"] for entry in self._pending]

    def has_pending(self) -> bool:
        return bool(self._pending)

    async def flush_pending(self) -> None:
        """Announce due timers that nobody was connected to hear; keep the ones still unheard."""
        if not self._pending or self._flushing:
            return
        self._flushing = True
        try:
            # Iterate a snapshot: notify now awaits, so a _fire racing this flush appends to
            # (and rebinds) the live list while we are suspended. Walking the live list made
            # the flush see stale content (a timer added mid-flush announced twice, one
            # appended mid-flush never seen).
            snapshot = list(self._pending)
            delivered_ids: set[str] = set()
            unheard = False
            for entry in snapshot:
                timer = _pending_timer(entry)
                if timer is None:
                    log.warning("Skipping malformed pending entry in flush: %r", entry)
                    continue
                if await self._deliver(timer, entry.get("note") or "restart"):
                    # An entry without a note came from a live fire nobody heard: announcing it
                    # now is late, so use the restart wording ("... minęło podczas restartu")
                    # instead of speaking as if it had just fired.
                    tid = _pending_id(entry)
                    if tid is not None:
                        delivered_ids.add(tid)
                else:
                    unheard = True
            if delivered_ids:
                # Remove exactly the entries this flush delivered, by id, and leave everything
                # else alone: a _fire racing this flush may have appended an unheard entry.
                self._pending = [e for e in self._pending if _pending_id(e) not in delivered_ids]
                self._save()
            if unheard:
                self._arm_quiet_hours_flush()
        finally:
            self._flushing = False

    def list_timers(self) -> list[dict[str, Any]]:
        """Scheduled timers plus fired-but-unheard ones (marked ``overdue``).

        GET /api/timers uses this: a reminder that fired while nobody was connected is
        still cancellable and must be visible, not silently waiting for the next hello.
        """
        return [{**t, "overdue": False} for t in self._timers] + [
            {**timer, "overdue": True}
            for entry in self._pending
            if (timer := _pending_timer(entry)) is not None
        ]

    def add_timer(
        self,
        label: str,
        due: datetime.datetime,
        kind: str = "timer",
        session_id: str | None = None,
        device_id: str | None = None,
    ) -> dict[str, Any]:
        if due.tzinfo is None:
            due = due.replace(tzinfo=WARSAW_TZ)
        now = datetime.datetime.now(WARSAW_TZ)
        timer_id = f"t_{uuid.uuid4().hex[:6]}"
        item = {
            "id": timer_id,
            "label": label.strip() or ("minutnik" if kind == "timer" else "przypomnienie"),
            "kind": kind,
            "due": due.isoformat(),
            "created": now.isoformat(),
            "session_id": session_id,
            "device_id": device_id,
        }
        self._timers.append(item)
        self._save()
        if self._started:
            self._schedule(item)
        return item

    def delete_timer(self, timer_id: str) -> bool:
        task = self._tasks.pop(timer_id, None)
        if task:
            task.cancel()
        # A fired-but-unheard timer is no longer in _timers: deleting it here is what
        # DELETE /api/timers/{id} must do, otherwise the "deleted" reminder is spoken later.
        before = len(self._timers) + len(self._pending)
        self._timers = [t for t in self._timers if t.get("id") != timer_id]
        self._pending = [e for e in self._pending if _pending_id(e) != timer_id]
        if len(self._timers) + len(self._pending) < before:
            self._save()
            return True
        return False

    def cancel_timers(self, label: str | None = None, kind: str | None = None) -> list[str]:
        cancelled: list[str] = []
        keep: list[dict[str, Any]] = []

        def _matches(t: dict[str, Any]) -> bool:
            match_kind = (kind is None) or (t.get("kind") == kind)
            match_label = (label is None) or (label.lower() in t.get("label", "").lower())
            return match_kind and match_label

        for t in self._timers:
            if _matches(t):
                cancelled.append(t["id"])
                task = self._tasks.pop(t["id"], None)
                if task:
                    task.cancel()
            else:
                keep.append(t)
        # Pending (fired, still unheard) timers are cancellable too: they left _timers when
        # they fired, but flush_pending would speak them at the next hello.
        keep_pending: list[dict[str, Any]] = []
        for entry in self._pending:
            timer = _pending_timer(entry)
            if timer is not None and _matches(timer):
                tid = _pending_id(entry)
                if tid is not None and tid not in cancelled:
                    cancelled.append(tid)
            else:
                keep_pending.append(entry)
        if cancelled:
            self._timers = keep
            self._pending = keep_pending
            self._save()
        return cancelled

    def get_status(self, kind: str | None = None) -> list[dict[str, Any]]:
        now = datetime.datetime.now(WARSAW_TZ)
        res: list[dict[str, Any]] = []
        for t in self._timers:
            if kind and t.get("kind") != kind:
                continue
            try:
                due_dt = datetime.datetime.fromisoformat(t["due"])
                if due_dt.tzinfo is None:
                    due_dt = due_dt.replace(tzinfo=WARSAW_TZ)
                rem_s = max(0, int((due_dt - now).total_seconds()))
            except Exception:
                rem_s = 0
            copy = dict(t)
            copy["remaining_seconds"] = rem_s
            copy["overdue"] = False
            res.append(copy)
        # Fired but not heard by any session: overdue, still waiting for the next hello.
        for entry in self._pending:
            timer = _pending_timer(entry)
            if timer is None or (kind and timer.get("kind") != kind):
                continue
            copy = dict(timer)
            copy["remaining_seconds"] = 0
            copy["overdue"] = True
            res.append(copy)
        return sorted(res, key=lambda x: x["remaining_seconds"])

    async def stop(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
        if self._quiet_flush_task is not None:
            self._quiet_flush_task.cancel()
            self._quiet_flush_task = None


_default_manager: TimerManager | None = None


def get_timer_manager() -> TimerManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = TimerManager()
    return _default_manager


def set_timer_manager(manager: TimerManager) -> None:
    global _default_manager
    _default_manager = manager
