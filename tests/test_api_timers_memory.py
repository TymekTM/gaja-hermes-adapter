from __future__ import annotations

import datetime
from pathlib import Path
from starlette.testclient import TestClient
import pytest

from gaja.config import Settings
from gaja.memory import MemoryStore
from gaja.server import create_app
from gaja.timers import TimerManager, WARSAW_TZ


def test_api_timers_and_memory_crud_with_origin(tmp_path: Path):
    settings = Settings(
        server={"allowed_origins": ["http://localhost:8790", "http://127.0.0.1:8790"]},
    )
    mem_store = MemoryStore(tmp_path / "memory.json")
    timer_mgr = TimerManager(tmp_path / "timers.json")

    # Pre-populate timer and memory
    now = datetime.datetime.now(WARSAW_TZ)
    t = timer_mgr.add_timer("pizza", now + datetime.timedelta(minutes=10), kind="timer")
    f = mem_store.add_fact("Użytkownik lubi herbatę")

    from gaja.pipeline import Services, SystemPrompt
    services = Services(
        cfg=settings,
        stt=None,
        tts=None,
        llm=None,
        prompt=SystemPrompt(text="sys", tools=[]),
        hermes=None,
        metrics=None,
        new_vad=None,
        timer_manager=timer_mgr,
        memory_store=mem_store,
    )
    app = create_app(settings, services=services)

    with TestClient(app) as client:
        # Disallowed origin check on GET /api/timers
        res_blocked = client.get("/api/timers", headers={"origin": "http://evil.com"})
        assert res_blocked.status_code == 403

        # Allowed origin on GET /api/timers
        res_timers = client.get("/api/timers", headers={"origin": "http://localhost:8790"})
        assert res_timers.status_code == 200
        timers_data = res_timers.json()
        assert "timers" in timers_data
        assert len(timers_data["timers"]) == 1
        assert timers_data["timers"][0]["id"] == t["id"]

        # DELETE /api/timers/{id}
        res_del_t = client.delete(f"/api/timers/{t['id']}", headers={"origin": "http://localhost:8790"})
        assert res_del_t.status_code == 200
        assert res_del_t.json() == {"ok": True}
        assert len(timer_mgr.list_timers()) == 0

        # Disallowed origin check on GET /api/memory
        res_mem_blocked = client.get("/api/memory", headers={"origin": "http://evil.com"})
        assert res_mem_blocked.status_code == 403

        # Allowed origin on GET /api/memory
        res_mem = client.get("/api/memory", headers={"origin": "http://localhost:8790"})
        assert res_mem.status_code == 200
        facts_data = res_mem.json()
        assert "facts" in facts_data
        assert len(facts_data["facts"]) == 1
        assert facts_data["facts"][0]["id"] == f["id"]

        # POST /api/memory
        res_post_mem = client.post(
            "/api/memory",
            json={"text": "Nowy fakt o psie"},
            headers={"origin": "http://localhost:8790"},
        )
        assert res_post_mem.status_code == 200
        created_fact = res_post_mem.json()["fact"]
        assert created_fact["text"] == "Nowy fakt o psie"
        assert len(mem_store.list_facts()) == 2

        # DELETE /api/memory/{id}
        res_del_m = client.delete(f"/api/memory/{created_fact['id']}", headers={"origin": "http://localhost:8790"})
        assert res_del_m.status_code == 200
        assert res_del_m.json() == {"ok": True}
        assert len(mem_store.list_facts()) == 1


def test_delete_missing_memory_fact_returns_404(tmp_path: Path):
    settings = Settings(server={"allowed_origins": ["http://localhost:8790"]})
    mem_store = MemoryStore(tmp_path / "memory.json")

    from gaja.pipeline import Services, SystemPrompt
    services = Services(
        cfg=settings,
        stt=None,
        tts=None,
        llm=None,
        prompt=SystemPrompt(text="sys", tools=[]),
        hermes=None,
        metrics=None,
        new_vad=None,
        memory_store=mem_store,
    )
    app = create_app(settings, services=services)

    with TestClient(app) as client:
        # An unknown id used to answer {"ok": true} as if something had been deleted.
        res = client.delete("/api/memory/m_niema", headers={"origin": "http://localhost:8790"})
        assert res.status_code == 404, res.text
        assert mem_store.list_facts() == []

        # An existing fact still deletes with {"ok": true}.
        fact = mem_store.add_fact("Kotek śpi na kanapie")
        res = client.delete(f"/api/memory/{fact['id']}", headers={"origin": "http://localhost:8790"})
        assert res.status_code == 200, res.text
        assert res.json() == {"ok": True}
        assert mem_store.list_facts() == []
