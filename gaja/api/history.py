"""Conversation history API: persistent turns across devices.

Stores data/history.jsonl (append-only, capped/rotated).
Endpoints:
  GET    /api/history?limit=50
  POST   /api/history
  DELETE /api/history
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

DEFAULT_LIMIT = 50
MAX_LIMIT = 500
MAX_TURNS_STORED = 1000

_lock = threading.Lock()


class TurnItem(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = Field(default_factory=time.time)
    role: str
    text: str
    lang: str = "pl"
    tool: str | None = None


class TurnCreate(BaseModel):
    role: str
    text: str
    lang: str = "pl"
    tool: str | None = None


def read_turns(path: Path, limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with _lock:
        try:
            with path.open("r", encoding="utf-8") as fh:
                lines = [line.strip() for line in fh if line.strip()]
        except OSError:
            return []
    turns: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            turns.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return turns


def append_turn(path: Path, item: TurnItem, max_stored: int = MAX_TURNS_STORED) -> dict[str, Any]:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = item.model_dump()
    raw_line = json.dumps(payload, ensure_ascii=False) + chr(10)

    with _lock:
        lines: list[str] = []
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as fh:
                    lines = [l.strip() for l in fh if l.strip()]
            except OSError:
                lines = []

        lines.append(json.dumps(payload, ensure_ascii=False))

        # Rotate/cap if exceeding max_stored
        if len(lines) > max_stored:
            lines = lines[-max_stored:]
            tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix="hist_", suffix=".tmp")
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_fh:
                for l in lines:
                    tmp_fh.write(l + chr(10))
            os.replace(tmp_path, path)
        else:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(raw_line)

    return payload


def clear_history(path: Path) -> None:
    with _lock:
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass


def create_history_router(
    history_path: Path,
    origin_ok: Callable[[str | None], bool],
    max_turns: int = MAX_TURNS_STORED,
) -> APIRouter:
    def check_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        if not origin_ok(origin):
            log.warning("rejected history request from origin %r", origin)
            raise HTTPException(403, "origin not allowed")

    router = APIRouter(prefix="/api/history", dependencies=[Depends(check_origin)])

    @router.get("")
    async def get_history(limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)) -> dict[str, Any]:
        return {"turns": read_turns(history_path, limit=limit)}

    @router.post("")
    async def post_turn(turn: TurnCreate) -> dict[str, Any]:
        if not turn.text.strip():
            raise HTTPException(400, "text must not be empty")
        if turn.role not in ("user", "assistant"):
            raise HTTPException(400, "role must be user or assistant")
        item = TurnItem(role=turn.role, text=turn.text, lang=turn.lang, tool=turn.tool)
        saved = append_turn(history_path, item, max_stored=max_turns)
        return {"ok": True, "turn": saved}

    @router.delete("")
    async def delete_history() -> dict[str, Any]:
        clear_history(history_path)
        return {"ok": True}

    return router
