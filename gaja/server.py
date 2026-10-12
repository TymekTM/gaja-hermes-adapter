from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.status import WS_1008_POLICY_VIOLATION

from gaja.audio.vad import load_silero_prob
from gaja.config import Settings, resolve
from gaja.hermes.base import HermesClient, StubHermes
from gaja.hermes.client import HermesClient as AcpHermesClient
from gaja.llm.client import LLMClient, LLMUnavailable
from gaja.llm.prompt import load_system_prompt
from gaja.llm.router import Router
from gaja.metrics import MetricsRecorder
from gaja.pipeline import Services, Session
from gaja.tts import prosody
from gaja.stt.faster_whisper_stt import FasterWhisperSTT
from gaja.stt.select import SwitchingSTT
from gaja.tts.base import TTS, EnginePool, get_tts
from gaja.stt.vocab import create_vocab_router
from gaja.wakeword_samples import create_router as create_wakeword_router

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
    return AcpHermesClient(command=cfg.command, cwd=Path(cfg.cwd).expanduser(), timeout_s=cfg.timeout_s)


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
    return Services(
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
    )


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

    async def send_json(self, payload: dict[str, Any]) -> None:
        await self._websocket.send_json(payload)

    async def send_bytes(self, data: bytes) -> None:
        await self._websocket.send_bytes(data)


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = services if services is not None else await build_services(settings)
        hermes_warmup = asyncio.create_task(_warm_hermes(app.state.services.hermes))
        # Optional device listener (WS /device on its own port, protocol v1.8); off unless [devices] enabled.
        app.state.device_listener = None
        if app.state.services.cfg.devices.enabled:
            from gaja.devices.listener import DeviceListener

            listener = DeviceListener(app.state.services)
            if await listener.start():
                app.state.device_listener = listener
        try:
            yield
        finally:
            if app.state.device_listener is not None:
                await app.state.device_listener.stop()
            hermes_warmup.cancel()
            await asyncio.gather(hermes_warmup, return_exceptions=True)
            stt_close = getattr(app.state.services.stt, "aclose", None)
            if stt_close is not None:
                await stt_close()
            await app.state.services.hermes.aclose()
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

    app.include_router(
        create_wakeword_router(
            resolve(settings.wakeword.recordings_dir),
            lambda origin: origin_allowed(origin, settings.server.allowed_origins),
            settings.wakeword.max_total_mb,
        )
    )
    app.include_router(create_vocab_router(settings, lambda origin: origin_allowed(origin, settings.server.allowed_origins)))

    web_dir = resolve(settings.server.web_dir)
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app
