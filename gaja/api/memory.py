from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from gaja.household import ID_RE
from gaja.memory import HOME, MemoryStore, get_memory_store


class FactCreateRequest(BaseModel):
    text: str
    # "home" (everyone) or a person id from the household.
    scope: str = Field(default=HOME, pattern=ID_RE)


def create_memory_router(
    store: MemoryStore | Callable[[], MemoryStore],
    is_origin_allowed: Callable[[str | None], bool],
) -> APIRouter:
    router = APIRouter(prefix="/api/memory", tags=["memory"])

    def _get_store() -> MemoryStore:
        st = store() if callable(store) else store
        return st if st is not None else get_memory_store()

    def _check_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        if not is_origin_allowed(origin):
            raise HTTPException(status_code=403, detail="Origin not allowed")

    @router.get("")
    async def list_facts(request: Request) -> dict[str, Any]:
        _check_origin(request)
        return {"facts": _get_store().list_facts()}

    @router.post("")
    async def add_fact(req: FactCreateRequest, request: Request) -> dict[str, Any]:
        _check_origin(request)
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        fact = _get_store().add_fact(text, scope=req.scope)
        return {"fact": fact}

    @router.delete("/{fact_id}")
    async def delete_fact(fact_id: str, request: Request) -> dict[str, Any]:
        _check_origin(request)
        # MemoryStore.delete_fact returns False when nothing matched: report that (404) instead of
        # pretending the fact was deleted. Both web clients treat non-2xx as "nothing removed".
        if not _get_store().delete_fact(fact_id):
            raise HTTPException(status_code=404, detail="Fact not found")
        return {"ok": True}

    return router
