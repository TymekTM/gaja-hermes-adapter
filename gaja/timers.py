from __future__ import annotations

import asyncio
import datetime
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

WARSAW_TZ = ZoneInfo("Europe/Warsaw")


class TimerManager:
    def __init__(
        self,
        file_path: Path | str = "data/timers.json",
        notify_fn: Callable[[dict[str, Any], str | None], Awaitable[None]] | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.notify_fn = notify_fn
        self._timers: list[dict[str, Any]] = []
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._started = False
        self._load()

    def _load(self) -> None:
        if not self.file_path.exists():
            self._timers = []
            return
        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "timers" in data:
                self._timers = list(data["timers"])
            elif isinstance(data, list):
                self._timers = list(data)
            else:
                self._timers = []
        except Exception as exc:
            log.warning("Could not read timers file %s: %s", self.file_path, exc)
            self._timers = []

    def _save(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file_path.with_suffix(".tmp")
        payload = {"timers": self._timers}
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
                continue

            if due_dt <= now:
                past_due.append(timer)
            else:
                upcoming.append(timer)

        # Clear past due from active list and save
        self._timers = upcoming
        self._save()

        # Fire past-due with restart note
        for t in past_due:
            if self.notify_fn:
                try:
                    await self.notify_fn(t, "restart")
                except Exception as exc:
                    log.warning("Error notifying past-due timer: %s", exc)

        # Schedule upcoming
        for t in list(self._timers):
            self._schedule(t)

    def _schedule(self, timer: dict[str, Any]) -> None:
        timer_id = timer["id"]
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
            self._save()
            if self.notify_fn:
                try:
                    await self.notify_fn(found, note)
                except Exception as exc:
                    log.exception("Error in timer notify_fn: %s", exc)

    def list_timers(self) -> list[dict[str, Any]]:
        return list(self._timers)

    def add_timer(self, label: str, due: datetime.datetime, kind: str = "timer") -> dict[str, Any]:
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
        before = len(self._timers)
        self._timers = [t for t in self._timers if t.get("id") != timer_id]
        if len(self._timers) < before:
            self._save()
            return True
        return False

    def cancel_timers(self, label: str | None = None, kind: str | None = None) -> list[str]:
        cancelled: list[str] = []
        keep: list[dict[str, Any]] = []
        for t in self._timers:
            match_kind = (kind is None) or (t.get("kind") == kind)
            match_label = (label is None) or (label.lower() in t.get("label", "").lower())
            if match_kind and match_label:
                cancelled.append(t["id"])
                task = self._tasks.pop(t["id"], None)
                if task:
                    task.cancel()
            else:
                keep.append(t)
        if cancelled:
            self._timers = keep
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
            res.append(copy)
        return sorted(res, key=lambda x: x["remaining_seconds"])

    async def stop(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()


_default_manager: TimerManager | None = None


def get_timer_manager() -> TimerManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = TimerManager()
    return _default_manager


def set_timer_manager(manager: TimerManager) -> None:
    global _default_manager
    _default_manager = manager
