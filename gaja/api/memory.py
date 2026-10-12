from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from gaja.memory import MemoryStore, get_memory_store


class FactCreateRequest(BaseModel):
    text: str


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
        fact = _get_store().add_fact(text)
        return {"fact": fact}

    @router.delete("/{fact_id}")
    async def delete_fact(fact_id: str, request: Request) -> dict[str, Any]:
        _check_origin(request)
        _get_store().delete_fact(fact_id)
        return {"ok": True}

    return router
