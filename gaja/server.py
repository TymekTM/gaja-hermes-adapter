from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.status import WS_1008_POLICY_VIOLATION

from gaja.audio.vad import load_silero_prob
from gaja.config import Settings, resolve
from gaja.hermes.base import HermesClient, StubHermes
from gaja.hermes.client import HermesClient as AcpHermesClient
from gaja.hermes.gateway import GatewayHermes
from gaja.llm.client import LLMClient, LLMUnavailable
from gaja.llm.prompt import load_system_prompt
from gaja.llm.router import Router
from gaja.metrics import MetricsRecorder
from gaja.pipeline import Services, Session
from gaja.tts import prosody
from gaja.stt.faster_whisper_stt import FasterWhisperSTT
from gaja.stt.select import SwitchingSTT
from gaja.tts.base import TTS, EnginePool, get_tts
from gaja.admin.api import create_admin_router
from gaja.admin.auth import AdminAuth
from gaja.api.devices import create_devices_router
from gaja.api.diagnostics import create_diagnostics_router
from gaja.api.history import create_history_router
from gaja.stt.vocab import create_vocab_router
from gaja.wakeword_samples import create_router as create_wakeword_router
from gaja.api.memory import create_memory_router
from gaja.api.timers import create_timers_router
from gaja.household import HouseholdStore
from gaja.memory import MemoryStore, set_memory_store
from gaja.timers import TimerManager, set_timer_manager

log = logging.getLogger(__name__)

TTS_ENGINES = ("piper", "piper-darkman", "paradee", "supertonic")
HERMES_WATCH_S = 30.0


def origin_allowed(origin: str | None, allowed: list[str]) -> bool:
    """WebSocket Origin check (browsers always send Origin on WS; CLI clients usually send none).

    Only an exact allowlist match is accepted. The Host header is deliberately not used to infer
    "same origin": a DNS-rebinding page would send a Host that matches its own Origin.
    """
    if origin is None:
        return True
    normalized = origin.strip().rstrip("/").lower()
    return normalized in {entry.strip().rstrip("/").lower() for entry in allowed}


def _engine_kwargs(settings: Settings, name: str) -> dict[str, Any]:
    tts = settings.tts
    if name == "piper":
        return {"model_path": resolve(tts.piper.model_path), "threads": tts.piper.threads}
    if name == "piper-darkman":
        return {"model_path": resolve(tts.piper_darkman.model_path), "threads": tts.piper_darkman.threads}
    if name == "paradee":
        return {"threads": tts.paradee.threads}
    if name == "supertonic":
        return {
            "steps": tts.supertonic.steps,
            "voice": tts.supertonic.voice,
            "speed": tts.supertonic.speed,
            "threads": tts.supertonic.threads,
            "output_sample_rate": tts.supertonic.output_sample_rate,
        }
    raise ValueError(f"unknown TTS engine {name!r}")


def _build_stt(settings: Settings) -> SwitchingSTT:
    factory = None
    if settings.stt.engine == "faster-whisper":
        fw = settings.stt.faster_whisper
        factory = partial(
            FasterWhisperSTT, model=fw.model, device=fw.device, compute_type=fw.compute_type, beam_size=fw.beam_size
        )
    return SwitchingSTT(settings.stt.engine, factory)


def _build_hermes(settings: Settings) -> HermesClient:
    cfg = settings.hermes
    if cfg.backend == "stub":
        return StubHermes(cfg.stub_delay_s)
    host = "127.0.0.1" if settings.server.host in ("0.0.0.0", "::") else settings.server.host
    api_url = f"http://{host}:{settings.server.port}"
    acp = AcpHermesClient(
        cfg,
        command=cfg.command,
        cwd=Path(cfg.cwd).expanduser(),
        timeout_s=cfg.timeout_s,
        speaker_mcp=cfg.speaker_mcp,
        api_url=api_url,
    )
    if not cfg.gateway.enabled:
        return acp
    return GatewayHermes(
        cfg.gateway.url,
        Path(cfg.gateway.hermes_home),
        key_var=cfg.gateway.key_var,
        timeout_s=cfg.timeout_s,
        idle_s=cfg.gateway.session_idle_min * 60,
        fallback=acp,
    )


async def build_services(settings: Settings) -> Services:
    prosody.configure(pitch_shift=settings.style.pitch_shift)
    prompt = load_system_prompt(resolve(settings.llm.system_prompt_path))
    stt = _build_stt(settings)
    await stt.load()

    factories: dict[str, Callable[[], TTS]] = {
        name: partial(get_tts, name, **_engine_kwargs(settings, name)) for name in TTS_ENGINES
    }
    default_engines = sorted(set(settings.tts.engine_by_language.values()))
    unknown = [name for name in default_engines if name not in factories]
    if unknown:
        raise ValueError(f"tts.engine_by_language uses unknown engines: {unknown}")
    tts = EnginePool(factories)
    # Only the per-language defaults are loaded at startup; a hello asking for another engine loads it then.
    for name in default_engines:
        await tts.get(name)

    llm = LLMClient(settings.llm, prompt.tools)
    if await llm.health():
        await llm.warm_up(prompt.text)
    else:
        log.warning("llama-server not reachable at %s; replies will be spoken as errors", settings.llm.base_url)

    tools = None
    if settings.tools.enabled:
        from gaja.tools import default_registry

        tools = default_registry
    router = None
    if settings.router.enabled:
        router = Router.from_file(
            llm,
            resolve(settings.router.prompt_path),
            threshold=settings.router.threshold,
            history_turns=settings.router.history_turns,
            timeout_s=settings.router.timeout_s,
            slot=settings.llm.router_slot,
            tools=tools.schemas() if tools is not None else None,
            tool_threshold=settings.router.tool_threshold,
        )
        try:
            await router.p_hermes("Dzień dobry.", [])
        except LLMUnavailable as exc:
            log.warning("router warm-up skipped: %s", exc)

    from gaja.stt.vocab import VocabularyManager

    smart_turn = None
    smart_turn_executor = None
    if settings.vad.smart_turn:
        from concurrent.futures import ThreadPoolExecutor
        from gaja.audio.smart_turn import SmartTurn, resolve_smart_turn_model

        try:
            model_path = resolve_smart_turn_model(settings.vad.smart_turn_model_path)
            smart_turn = SmartTurn(model_path, threads=1)
            smart_turn_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="smart-turn")
        except Exception:
            log.exception("Failed to load Smart Turn model; continuing with smart_turn disabled")

    memory_store = MemoryStore(resolve(Path("data/memory.json")))
    set_memory_store(memory_store)

    timer_manager = TimerManager(resolve(Path("data/timers.json")))
    set_timer_manager(timer_manager)

    household = HouseholdStore(resolve(settings.household.path))

    services = Services(
        cfg=settings,
        stt=stt,
        tts=tts,
        llm=llm,
        prompt=prompt,
        hermes=_build_hermes(settings),
        metrics=MetricsRecorder(resolve(settings.metrics.turns_log), settings.metrics.window),
        new_vad=load_silero_prob,
        router=router,
        tools=tools,
        normalize=_load_normalizer() if settings.text.normalize_tts else None,
        vocab=VocabularyManager(
            default_path=resolve(settings.vocab.default_path),
            user_path=resolve(settings.vocab.user_path),
            suggestions_path=resolve(settings.vocab.suggestions_path),
            max_hotwords=settings.vocab.max_hotwords,
            fuzzy=settings.vocab.fuzzy,
            enabled=settings.vocab.enabled,
        ) if settings.vocab.enabled else None,
        smart_turn=smart_turn,
        smart_turn_executor=smart_turn_executor,
        timer_manager=timer_manager,
        memory_store=memory_store,
        household=household,
    )

    def refresh_identities() -> None:
        # Bindings or the house note changed in the panel: re-resolve who each connection is.
        for session in list(services.sessions):
            session.refresh_identity()

    household.on_change.append(refresh_identities)

    notify_tasks: set[asyncio.Task[None]] = set()

    async def notify_sessions(timer: dict[str, Any], note: str | None = None) -> None:
        sessions = list(services.sessions)
        if not sessions:
            return
        orig_session_id = timer.get("session_id")
        orig_device_id = timer.get("device_id")
        speaker_session = None

        if orig_session_id:
            for s in sessions:
                if s.id == orig_session_id:
                    speaker_session = s
                    break
        if speaker_session is None and orig_device_id:
            for s in sessions:
                if s.device_id == orig_device_id:
                    speaker_session = s
                    break

        if speaker_session is None:
            devices = [s for s in sessions if s.is_device]
            if devices:
                speaker_session = max(devices, key=lambda s: s.last_active)

        if speaker_session is None:
            web_sessions = [s for s in sessions if not s.is_device]
            if web_sessions:
                speaker_session = max(web_sessions, key=lambda s: s.last_active)

        for s in sessions:
            should_speak = (s is speaker_session)
            task = asyncio.create_task(s.on_timer(timer, note, speak=should_speak))
            notify_tasks.add(task)  # the loop keeps only weak references to tasks
            task.add_done_callback(notify_tasks.discard)

    timer_manager.notify_fn = notify_sessions
    memory_store.on_change = lambda: [s.update_system_prompt() for s in services.sessions]
    return services


def _load_normalizer() -> Callable[[str, str], str] | None:
    try:
        from gaja.text.normalize import normalize_for_tts
    except Exception:
        log.exception("gaja.text.normalize unavailable; TTS gets the raw text")
        return None
    return normalize_for_tts


async def _warm_hermes(hermes: HermesClient) -> None:
    try:
        ready = await hermes.health()
    except Exception:
        log.exception("Hermes warm-up failed")
        ready = False
    log.info("Hermes worker %s", "ready" if ready else "unavailable; handoffs will report an error")
    # Keep the ACP worker alive: if its reader died (worker crashed, EOF), respawn it before the next task.
    while True:
        await asyncio.sleep(HERMES_WATCH_S)
        if hermes.ready():
            continue
        log.warning("Hermes worker not ready; respawning")
        try:
            await hermes.health()
        except Exception:
            log.exception("Hermes respawn failed")


class _WebSocketOutbox:
    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._lock = asyncio.Lock()

    async def send_json(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            await self._websocket.send_json(payload)

    async def send_bytes(self, data: bytes) -> None:
        async with self._lock:
            await self._websocket.send_bytes(data)


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = services if services is not None else await build_services(settings)
        if getattr(app.state.services, "timer_manager", None) is not None:
            await app.state.services.timer_manager.start()
        hermes_warmup = (
            asyncio.create_task(_warm_hermes(app.state.services.hermes))
            if getattr(app.state.services, "hermes", None) is not None
            else None
        )
        # Optional device listener (WS /device on its own port, protocol v1.8); off unless [devices] enabled.
        app.state.device_listener = None
        if getattr(app.state.services.cfg, "devices", None) and app.state.services.cfg.devices.enabled:
            from gaja.devices.listener import DeviceListener

            listener = DeviceListener(app.state.services)
            if await listener.start():
                app.state.device_listener = listener
        try:
            yield
        finally:
            if getattr(app.state.services, "timer_manager", None) is not None:
                await app.state.services.timer_manager.stop()
            if app.state.device_listener is not None:
                await app.state.device_listener.stop()
            if hermes_warmup is not None:
                hermes_warmup.cancel()
                await asyncio.gather(hermes_warmup, return_exceptions=True)
            if getattr(app.state.services, "smart_turn_executor", None) is not None:
                app.state.services.smart_turn_executor.shutdown(wait=False)
            stt_close = getattr(app.state.services.stt, "aclose", None)
            if stt_close is not None:
                await stt_close()
            if getattr(app.state.services, "hermes", None) is not None:
                await app.state.services.hermes.aclose()
            if getattr(app.state.services, "llm", None) is not None:
                await app.state.services.llm.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        svc: Services = request.app.state.services
        return {
            "status": "ok",
            "llm": await svc.llm.health(),
            "hermes": svc.hermes.ready(),
            "hermes_backend": svc.cfg.hermes.backend,
            "hermes_gateway": svc.cfg.hermes.gateway.enabled and svc.cfg.hermes.backend != "stub",
            "engines": svc.tts.names(),
            "engines_loaded": svc.tts.loaded(),
            "engine_by_language": svc.cfg.tts.engine_by_language,
            "stt_engine": svc.stt.engine,
            "stt_requested": svc.stt.requested,
            "stt_device": svc.stt.device,
            "stt_fallback": svc.stt.fallback_reason,
            "stt_degraded": getattr(svc.stt, "degraded", False),
            "devices": {
                "enabled": svc.cfg.devices.enabled,
                "listening": request.app.state.device_listener is not None,
                "connected": request.app.state.device_listener.connected if request.app.state.device_listener else [],
            },
            "router": {
                "enabled": svc.router is not None,
                "threshold": svc.cfg.router.threshold if svc.router is not None else None,
                "tools": svc.router is not None and svc.router.tools_enabled,
            },
        }

    @app.get("/metrics")
    async def metrics(request: Request) -> dict[str, Any]:
        svc: Services = request.app.state.services
        return svc.metrics.summary()

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        if not origin_allowed(origin, websocket.app.state.services.cfg.server.allowed_origins):
            log.warning("rejected WebSocket from origin %r", origin)
            # Closing before accept makes the server answer the handshake with HTTP 403.
            await websocket.close(code=WS_1008_POLICY_VIOLATION)
            return
        await websocket.accept()
        session = Session(websocket.app.state.services, _WebSocketOutbox(websocket))
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await session.handle_audio(message["bytes"])
                elif message.get("text") is not None:
                    await session.handle_text(message["text"])
        except WebSocketDisconnect:
            pass
        finally:
            await session.close()

    origin_ok = lambda origin: origin_allowed(origin, settings.server.allowed_origins)
    app.include_router(
        create_wakeword_router(
            resolve(settings.wakeword.recordings_dir),
            origin_ok,
            settings.wakeword.max_total_mb,
        )
    )
    app.include_router(
        create_timers_router(
            lambda: app.state.services.timer_manager,
            origin_ok,
        )
    )
    app.include_router(
        create_memory_router(
            lambda: app.state.services.memory_store,
            origin_ok,
        )
    )
    app.include_router(create_history_router(resolve(Path("data/history.jsonl")), origin_ok))
    app.include_router(create_devices_router(origin_ok))
    app.include_router(create_diagnostics_router(origin_ok))
    app.include_router(create_vocab_router(services.vocab if services is not None else None, origin_ok))

    if settings.admin.enabled:
        auth = AdminAuth(settings.admin.credentials_file, settings.admin.session_hours)
        app.include_router(create_admin_router(auth, origin_ok))
        # Device take recording (protocol v1.10): admin-only, shares the wakeword SampleStore.
        from gaja.api.device_record import create_device_record_router
        from gaja.wakeword_samples import get_store

        app.include_router(
            create_device_record_router(
                auth,
                get_store(resolve(settings.wakeword.recordings_dir), settings.wakeword.max_total_mb),
                origin_ok,
            )
        )

        @app.get("/admin", include_in_schema=False)
        async def admin_page() -> RedirectResponse:
            return RedirectResponse("/admin.html")

    web_dir = resolve(settings.server.web_dir)
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app
