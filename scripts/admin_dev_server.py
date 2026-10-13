"""Model-free dev server for the admin panel (web/admin.html).

Serves the real admin router, the memory/timers/vocab/devices/diagnostics/history routers and the
web/ directory, with stub /health and /metrics. No STT/TTS/LLM model is loaded. Everything it writes
lives in a temp directory (config copy, credentials, memory, timers, vocab), so the real
config.local.toml is never touched.

    python scripts/admin_dev_server.py [--password dev-password-123] [--port 8795] [--data-dir DIR]

Needs the project's Python environment (fastapi, uvicorn, tomlkit), but not `uv sync` in a worktree:
run it with any interpreter that has those packages, from the repository root.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from gaja import config as config_module  # noqa: E402

LOCAL_SEED = """# Local overrides (dev server copy)
[llm]
temperature = 0.45 # a little warmer than the default

[vad]
min_silence_ms = 650
"""

SUGGESTIONS = [
    {"id": "s_aaaa0001", "term": "Tailscale", "heard": ["teilskejl", "tejlskejl"], "count": 3, "reason": "unknown-token"},
    {"id": "s_aaaa0002", "term": "Hermes", "heard": ["hermez"], "count": 1, "reason": "near-miss"},
]


def build_app(data_dir: Path, password: str, port: int):
    # Point the config module at temp copies before Settings() is ever built.
    base = data_dir / "config.toml"
    local = data_dir / "config.local.toml"
    shutil.copyfile(ROOT / "config.toml", base)
    local.write_text(LOCAL_SEED)
    config_module.CONFIG_PATH = base
    config_module.LOCAL_CONFIG_PATH = local

    from gaja.admin.api import create_admin_router
    from gaja.admin.auth import AdminAuth, write_credentials
    from gaja.api.devices import create_devices_router, record_device_activity
    from gaja.api.diagnostics import create_diagnostics_router
    from gaja.api.history import append_turn, create_history_router, TurnItem
    from gaja.api.memory import create_memory_router
    from gaja.api.timers import create_timers_router
    from gaja.config import Settings
    from gaja.devices.controller import WAKE_LOG
    from gaja.devices.registry import append_device
    from gaja.memory import MemoryStore
    from gaja.metrics import MetricsRecorder, TurnMetrics
    from gaja.stt.vocab import VocabularyManager, create_vocab_router
    from gaja.timers import TimerManager
    import datetime

    creds = data_dir / "admin.toml"
    write_credentials(creds, password)

    cfg = Settings()
    tokens = data_dir / "devices.toml"
    append_device(tokens, "atom-salon", "x" * 32, "Salon")
    append_device(tokens, "atom-kuchnia", "y" * 32, "Kuchnia")
    append_device(tokens, "atom-garaz", "z" * 32)
    cfg.devices.tokens_file = tokens

    metrics = MetricsRecorder(data_dir / "turns.jsonl", 200)
    for i, (stt, router, ttft, tts, e2e) in enumerate(
        [(180, 40, 220, 120, 1500), (210, 38, 190, 110, 1610), (190, 45, 260, 135, 1720), (230, 41, 205, 118, 1580)]
    ):
        metrics.record(
            "dev",
            TurnMetrics(turn=i, kind="audio", route="local", stt_ms=stt, router_ms=router, llm_ttft_ms=ttft,
                        tts_first_ms=tts, e2e_first_audio_ms=e2e),
        )
    metrics.count("wake_accepted", 7)
    metrics.count("wake_rejected:no_phrase", 2)

    memory = MemoryStore(data_dir / "memory.json")
    memory.add_fact("Ulubiona kawa Tymka to flat white.")
    memory.add_fact("Kot ma na imię Pierożek i nie lubi odkurzacza.")

    timers = TimerManager(data_dir / "timers.json")
    now = datetime.datetime.now(config_module_tz())
    timers.add_timer("herbata", now + datetime.timedelta(minutes=4), kind="timer", device_id="atom-kuchnia")
    timers.add_timer("zadzwonić do mamy", now + datetime.timedelta(hours=3), kind="reminder")

    (data_dir / "vocab_suggestions.json").write_text(json.dumps(SUGGESTIONS, ensure_ascii=False))
    vocab = VocabularyManager(
        default_path=ROOT / cfg.vocab.default_path,
        user_path=data_dir / "vocab.toml",
        suggestions_path=data_dir / "vocab_suggestions.json",
    )
    vocab.put_terms([{"term": "Pierożek", "aliases": ["pierozek", "pirożek"], "lang": "pl"}])

    history = data_dir / "history.jsonl"
    append_turn(history, TurnItem(role="user", text="Która jest godzina?"))
    append_turn(history, TurnItem(role="assistant", text="Jest dwudziesta pierwsza dziesięć."))

    t0 = time.time()
    WAKE_LOG.append({"ts": t0 - 90, "device_id": "atom-salon", "word": "hej gaja", "score": 0.93, "accepted": True, "reason": "ok"})
    WAKE_LOG.append({"ts": t0 - 30, "device_id": "atom-kuchnia", "word": "hej gaja", "score": 0.71, "accepted": False, "reason": "no_phrase"})
    record_device_activity("atom-salon", is_wake=True)

    allowed = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
    origin_ok = lambda origin: origin is None or origin.rstrip("/").lower() in allowed  # noqa: E731

    app = FastAPI()

    class Services:
        pass

    services = Services()
    services.cfg = cfg
    services.metrics = metrics
    services.memory_store = memory
    services.timer_manager = timers
    services.vocab = vocab
    app.state.services = services
    app.state.device_listener = SimpleNamespace(connected=["atom-salon"], registry=None)

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "llm": {"ok": True, "model": cfg.llm.model, "slots": 2},
            "hermes": True,
            "hermes_backend": cfg.hermes.backend,
            "engines": ["piper", "paradee", "supertonic"],
            "engines_loaded": ["piper"],
            "engine_by_language": cfg.tts.engine_by_language,
            "stt_engine": "faster-whisper",
            "stt_requested": "faster-whisper",
            "stt_device": "cuda",
            "stt_fallback": None,
            "stt_degraded": False,
            "devices": {"enabled": True, "listening": True, "connected": ["atom-salon"]},
            "router": {"enabled": True, "threshold": cfg.router.threshold, "tools": True},
        }

    @app.get("/metrics")
    async def metrics_route(request: Request) -> dict:
        return request.app.state.services.metrics.summary()

    app.include_router(create_admin_router(AdminAuth(creds, cfg.admin.session_hours), origin_ok))
    app.include_router(create_memory_router(memory, origin_ok))
    app.include_router(create_timers_router(timers, origin_ok))
    app.include_router(create_vocab_router(vocab, origin_ok))
    app.include_router(create_devices_router(origin_ok))
    app.include_router(create_diagnostics_router(origin_ok))
    app.include_router(create_history_router(history, origin_ok))

    @app.get("/admin", include_in_schema=False)
    async def admin_page() -> RedirectResponse:
        return RedirectResponse("/admin.html")

    app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
    return app


def config_module_tz():
    from gaja.timers import WARSAW_TZ

    return WARSAW_TZ


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--password", default="dev-password-123")
    parser.add_argument("--port", type=int, default=8795)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--data-dir", type=Path, default=None, help="default: a fresh temp directory")
    args = parser.parse_args()
    data_dir = args.data_dir or Path(tempfile.mkdtemp(prefix="gaja-admin-dev-"))
    data_dir.mkdir(parents=True, exist_ok=True)
    app = build_app(data_dir, args.password, args.port)
    print(f"admin dev server: http://127.0.0.1:{args.port}/admin  password: {args.password}", flush=True)
    print(f"temp data (config.local.toml, ...): {data_dir}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
