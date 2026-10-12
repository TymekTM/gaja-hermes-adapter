from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from gaja.audio.pcm import float_to_pcm16, pcm16_to_float
from gaja.audio.vad import Endpointer, ProbFn, VadEvent
from gaja.config import Settings
from gaja.devices.audio import DeviceCaps, ResamplingOutbox
from gaja.devices.controller import DeviceController
from gaja.devices.registry import DeviceInfo
from gaja.hermes.base import HermesClient
from gaja.hermes.progress import TEXT_DELTA, ProgressCoalescer, ProgressItem
from gaja.llm.client import LLMClient, LLMUnavailable
from gaja.llm.history import History
from gaja.llm.parse import ToolCall
from gaja.llm.prompt import SystemPrompt
from gaja.llm.router import Route, Router, promises_handoff
from gaja.metrics import MetricsRecorder, TurnMetrics
from gaja.stt.base import STT
from gaja.text.chunker import SentenceChunker, SpeakerPrefixStripper, StyleTagStripper, speakable
from gaja.text.emotion import auto_style, has_explicit_tag
from gaja.text.style import Style, parse_styled
from gaja.tts.base import TTS, EnginePool

log = logging.getLogger(__name__)

HANDOFF_TOOL = "handoff_to_hermes"
ACKS = {
    "pl": ("Jasne, przekazuję to Hermesowi.", "Dobrze, zlecam to Hermesowi.", "Już to sprawdza Hermes, chwilkę."),
    "en": ("Sure, passing this to Hermes.", "Alright, handing this over to Hermes.", "Hermes is on it, one moment."),
}
LLM_DOWN = {"pl": "Model językowy jest chwilowo niedostępny.", "en": "The language model is temporarily unavailable."}
HERMES_DOWN = {
    "pl": "Hermes jest teraz niedostępny, nie udało się wykonać zadania.",
    "en": "Hermes is unavailable right now, the task could not be run.",
}
CANCELLED = {"pl": "Dobrze, anulowałam zadanie Hermesa.", "en": "Okay, I cancelled the Hermes task."}
NOTHING_TO_CANCEL = {"pl": "Hermes nie wykonuje teraz żadnego zadania.", "en": "Hermes isn't running any task right now."}
STT_NOTICES = {
    "stt_degraded": "Rozpoznawanie mowy działa awaryjnie na CPU (wolniej); GPU będzie sprawdzane ponownie.",
    "stt_recovered": "Rozpoznawanie mowy znów działa na GPU.",
}
LANGUAGES = ("pl", "en")
RESULT_PROMPT = "Wynik zadania od Hermesa:\n{result}\n\nPowiedz to krótko, w 1 do 3 zdaniach."
INTERRUPTED_MARK = "…"
# Values the tool selector copies from the signature instead of naming a city.
PLACEHOLDER_LOCATIONS = {"", "miasto", "city", "location", "nazwa miasta", "tutaj", "here"}
# Echo of Gaja's own voice keeps arriving while the client plays it back; the VAD is strict until then.
ECHO_TAIL_S = 0.4

# Explicit intent to stop a running Hermes task, checked only while one runs (a bare "stop" is barge-in).
CANCEL_INTENT = re.compile(
    r"^\W*(?:(?:no\s+to\s+|dobra,?\s+|ok,?\s+|okej,?\s+|proszę\s+|please\s+)?"
    r"(?:anuluj|przerwij|zatrzymaj|odwołaj|wstrzymaj|cancel|abort|stop)\w*"
    r"(?:\s+(?:to|tego|ten|te|je|go|it|that|this|the|proszę|please|już|teraz|now|zadanie|zadania|task|"
    r"hermes\w*|polecenie|request|job|wszystko|everything|all)){0,4}"
    r"|(?:niech\s+)?hermes\w*\s*,?\s*(?:przestań|stop|anuluj|zatrzymaj\s+się)\w*)\W*$",
    re.IGNORECASE,
)


def is_cancel_intent(text: str) -> bool:
    return bool(CANCEL_INTENT.match(text.strip()))


@dataclass
class Services:
    cfg: Settings
    stt: STT
    tts: EnginePool
    llm: LLMClient
    prompt: SystemPrompt
    hermes: HermesClient
    metrics: MetricsRecorder
    new_vad: Callable[[], ProbFn]
    router: Router | None = None
    tools: Any = None
    normalize: Callable[[str, str], str] | None = None
    timer_manager: Any = None
    memory_store: Any = None
    sessions: set[Session] = field(default_factory=set)

    def get_system_prompt(self) -> str:
        base = self.prompt.text
        if self.memory_store is not None:
            return base + self.memory_store.format_prompt_block()
        return base


class Outbox(Protocol):
    async def send_json(self, payload: dict[str, Any]) -> None: ...

    async def send_bytes(self, data: bytes) -> None: ...


class Hello(BaseModel):
    lang: Literal["pl", "en", "auto"] = "pl"
    tts: str | None = None
    client: Literal["web", "cli", "device"] = "web"
    # Protocol v1.8, client "device" only (the token is checked by the device endpoint, never stored here).
    device_id: str | None = Field(default=None, max_length=64)
    fw: str | None = Field(default=None, max_length=64)
    caps: DeviceCaps | None = None


@dataclass(frozen=True)
class Sentence:
    text: str
    style: Style
    speech: str | None = None


@dataclass(frozen=True)
class _LLMDown:
    pass


_Item = Sentence | ToolCall | _LLMDown


@dataclass
class _Turn:
    t0: float
    language: str
    metrics: TurnMetrics
    out: Outbox
    user_text: str = ""
    spoken: bool = False
    played: list[str] = field(default_factory=list)
    spec: _Speculation | None = None


@dataclass
class _HermesJob:
    id: str
    task: str
    language: str
    runner: asyncio.Task[None] | None = None


class _Gate:
    """Holds a speculative turn's outbound messages until it is committed, or drops them."""

    def __init__(self, out: Outbox) -> None:
        self._out = out
        self._held: list[dict[str, Any] | bytes] | None = []
        self._dropped = False

    async def send_json(self, payload: dict[str, Any]) -> None:
        await self._send(payload)

    async def send_bytes(self, data: bytes) -> None:
        await self._send(data)

    async def _send(self, item: dict[str, Any] | bytes) -> None:
        if self._dropped:
            return
        if self._held is not None:
            self._held.append(item)
        elif isinstance(item, bytes):
            await self._out.send_bytes(item)
        else:
            await self._out.send_json(item)

    async def open(self) -> None:
        held, self._held = self._held or [], None
        for item in held:
            await self._send(item)

    def drop(self) -> None:
        self._dropped = True
        self._held = None


@dataclass
class _Speculation:
    gate: _Gate
    started: float
    t0: float
    prev_state: str
    metrics: TurnMetrics
    committed: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    turn: _Turn | None = None
    # History writes made before commit; flushed on commit, dropped on cancel.
    pending: list[tuple[str, str]] = field(default_factory=list)
    reportable: bool = False


def _rebase(metrics: TurnMetrics, t0: float, now: float) -> None:
    shift = (now - t0) * 1000
    for name in ("first_sentence_ms", "e2e_first_audio_ms"):
        value = getattr(metrics, name)
        if value is not None:
            setattr(metrics, name, max(0.0, round(value - shift, 1)))


def _ms_since(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _to_sentences(
    raw: list[str],
    style: Style,
    *,
    auto_emotion: bool,
    language: str = "pl",
    normalize: Callable[[str, str], str] | None = None,
) -> tuple[list[Sentence], Style]:
    sentences: list[Sentence] = []
    for chunk in raw:
        auto = auto_emotion and not has_explicit_tag(chunk)
        for segment_style, segment in parse_styled(chunk, style):
            text = speakable(segment)
            if not text:
                continue
            if auto:
                segment_style = auto_style(text)
            speech = None
            if normalize is not None:
                try:
                    speech = normalize(text, language)
                except Exception:
                    log.exception("normalize_for_tts failed; speaking the raw text")
            sentences.append(Sentence(text, segment_style, speech if speech and speech != text else None))
            style = replace(segment_style, events=())
    return sentences, style


class Session:
    def __init__(self, services: Services, out: Outbox, device: DeviceInfo | None = None) -> None:
        self.id = uuid.uuid4().hex
        self._svc = services
        self._svc.sessions.add(self)
        self._out: Outbox = out
        # Device mode (WS /device, protocol v1.8): output is resampled to the device rate and the wake /
        # listen-window flow is handled by DeviceController.
        self._device: DeviceController | None = None
        if device is not None:
            self._out = ResamplingOutbox(out, services.cfg.devices.out_chunk_ms)
            self._device = DeviceController(self, services.cfg.devices, self._out, device)
        self._history = History(
            services.get_system_prompt(), services.cfg.llm.history_max_turns, services.cfg.llm.history_max_chars
        )
        self._hello: Hello | None = None
        self._engine_override: str | None = None
        self._endpointer: Endpointer | None = None
        self._state = ""
        self._active: asyncio.Task[None] | None = None
        self._jobs: dict[str, _HermesJob] = {}
        self._background: set[asyncio.Task[None]] = set()
        self._result_lock = asyncio.Lock()
        self._spec: _Speculation | None = None
        self._turns = 0
        self._utterances = 0
        self._acks = 0
        self._speech_end_at = 0.0
        self._playback_until = 0.0
        self._stt_degraded = False

    # ---- client messages -------------------------------------------------------------------

    async def handle_text(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._error("invalid JSON")
            return
        kind = msg.get("type") if isinstance(msg, dict) else None
        if kind == "hello":
            await self._on_hello(msg)
        elif kind == "ping":
            await self._out.send_json({"type": "pong"})
        elif self._hello is None:
            await self._error("send hello first")
        elif kind == "text" and isinstance(msg.get("text"), str):
            await self._start_user_turn(text=msg["text"])
        elif kind == "wake":
            if self._device is None:
                await self._error("wake is only for device clients")
            else:
                await self._device.on_wake(msg)
        elif kind == "interrupt":
            # Stops Gaja's speech/reply only; Hermes tasks keep running unless scope is "all".
            await self._interrupt_speech()
            if msg.get("scope") == "all":
                await self._cancel_jobs(None)
        elif kind == "cancel_task":
            job_id = msg.get("id")
            if job_id is not None and not isinstance(job_id, str):
                await self._error("cancel_task id must be a string")
                return
            if not await self._cancel_jobs(job_id):
                await self._error(f"no running Hermes task {job_id!r}" if job_id else "no running Hermes task")
        elif kind == "end_of_input":
            for event in self._endpointer.flush() if self._endpointer else []:
                await self._on_vad(event)
        else:
            await self._error(f"unknown message type {kind!r}")

    async def handle_audio(self, data: bytes) -> None:
        if self._endpointer is None:
            await self._error("send hello first")
            return
        if len(data) % 2:
            await self._error("PCM16 frame has odd length")
            return
        samples = pcm16_to_float(data)
        if self._device is not None:
            filtered = self._device.filter_audio(samples)
            if filtered is None:
                return
            samples = filtered
        self._endpointer.set_strict(time.perf_counter() < self._playback_until)
        for event in self._endpointer.feed(samples):
            await self._on_vad(event)

    async def close(self) -> None:
        self._svc.sessions.discard(self)
        if self._device is not None:
            await self._device.close()
        await self._cancel_speculation()
        jobs = [j.runner for j in self._jobs.values() if j.runner is not None]
        tasks = [t for t in (self._active, *jobs, *self._background) if t is not None]
        self._active = None
        self._jobs.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._svc.hermes.forget(self.id)

    async def _on_hello(self, msg: dict[str, Any]) -> None:
        try:
            hello = Hello.model_validate(msg)
        except ValidationError as exc:
            await self._error(f"invalid hello: {exc.errors()[0]['msg']}")
            return
        if self._device is not None:
            if hello.client != "device":
                await self._error("invalid hello: the device endpoint needs client=device")
                return
            if self._hello is not None:
                await self._error("hello already sent")
                return
        elif hello.client == "device":
            await self._error("invalid hello: device clients must use the device endpoint")
            return
        if hello.tts is not None and hello.tts not in self._svc.tts:
            await self._error(f"unknown tts engine {hello.tts!r}")
            return
        language = self._svc.cfg.stt.default_language if hello.lang == "auto" else hello.lang
        try:
            engine = await self._svc.tts.get(hello.tts or self._svc.cfg.tts.engine_by_language[language])
        except Exception as exc:
            log.exception("TTS engine failed to load")
            await self._error(f"tts engine unavailable: {exc}")
            return
        first = self._hello is None
        self._hello = hello
        self._engine_override = hello.tts
        if self._endpointer is None:
            prob = await asyncio.to_thread(self._svc.new_vad)
            self._endpointer = Endpointer(prob, self._svc.cfg.vad)
        ready: dict[str, Any] = {
            "type": "ready",
            "session_id": self.id,
            "tts": engine.name,
            "sample_rate": engine.sample_rate,
            "langs": sorted(self._svc.cfg.tts.engine_by_language),
            "engines": self._svc.tts.names(),
            "engine_by_language": dict(self._svc.cfg.tts.engine_by_language),
        }
        if self._device is not None:
            ready["device"] = self._device.configure(hello.caps or DeviceCaps())
            ready["sample_rate"] = self._device.out_rate
        await self._out.send_json(ready)
        await self._set_state("listening")
        await self._check_stt()
        if first:
            # Session-start preload: system prompt (+ tools) into the chat slot, router prefix into its slot.
            self._spawn(self._preload())

    async def _preload(self) -> None:
        # Both slots at once, so a turn arriving right after hello does not queue behind the chat prefix.
        jobs = []
        if self._svc.router is not None:
            jobs.append(self._svc.router.prime(self._history.messages()[1:]))
        prime_chat = getattr(self._svc.llm, "prime_chat", None)
        if prime_chat is not None:
            jobs.append(prime_chat(self._history.messages()))
        for result in await asyncio.gather(*jobs, return_exceptions=True):
            if isinstance(result, LLMUnavailable):
                log.debug("preload skipped: %s", result)
            elif isinstance(result, BaseException):
                log.warning("preload failed: %r", result)

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _on_vad(self, event: VadEvent) -> None:
        if event.kind == "speech_start":
            await self._out.send_json({"type": "vad", "state": "speech_start"})
            if self._device is not None:
                self._device.on_speech_start()
            await self._interrupt_speech(reason="barge_in")
            self._prime_router()
            return
        if event.kind == "speculate":
            # The first utterance after a device wake is checked for the wake phrase: no speculation then.
            if event.audio is not None and not (self._device is not None and self._device.blocks_speculation()):
                await self._start_speculation(event.audio)
            return
        if event.kind == "speculation_cancel":
            await self._cancel_speculation()
            return
        await self._out.send_json({"type": "vad", "state": "speech_end"})
        if self._device is not None and event.audio is not None:
            await self._device.on_speech_end()
        if self._spec is not None:
            if event.audio is None:
                await self._cancel_speculation()
            else:
                await self._commit_speculation()
            return
        if event.audio is not None:
            self._speech_end_at = time.perf_counter()
            await self._start_user_turn(audio=event.audio)

    def _default_language(self) -> str:
        assert self._hello is not None
        return self._svc.cfg.stt.default_language if self._hello.lang == "auto" else self._hello.lang

    async def _engine_for(self, language: str) -> TTS:
        name = self._engine_override or self._svc.cfg.tts.engine_by_language[language]
        return await self._svc.tts.get(name)

    # ---- turn lifecycle --------------------------------------------------------------------

    async def _start_user_turn(self, audio: np.ndarray | None = None, text: str | None = None) -> None:
        await self._interrupt_speech()
        self._turns += 1
        t0 = self._speech_end_at if audio is not None else time.perf_counter()
        metrics = TurnMetrics(turn=self._turns, kind="audio" if audio is not None else "text")
        self._active = asyncio.create_task(self._user_turn(metrics, t0, audio, text, None))

    async def _start_speculation(self, audio: np.ndarray) -> None:
        await self._interrupt_speech()
        self._turns += 1
        now = time.perf_counter()
        spec = _Speculation(
            gate=_Gate(self._out),
            started=now,
            t0=now,
            prev_state=self._state,
            metrics=TurnMetrics(turn=self._turns, kind="audio"),
        )
        spec.task = asyncio.create_task(self._user_turn(spec.metrics, now, audio, None, spec))
        self._spec = spec

    async def _cancel_speculation(self) -> None:
        spec = self._spec
        if spec is None:
            return
        self._spec = None
        spec.gate.drop()
        assert spec.task is not None
        spec.task.cancel()
        await asyncio.gather(spec.task, return_exceptions=True)
        spec.pending.clear()
        self._state = spec.prev_state
        spec.metrics.spec_ms = _ms_since(spec.started)
        self._svc.metrics.record(self.id, spec.metrics)

    async def _commit_speculation(self) -> None:
        spec = self._spec
        if spec is None:
            return
        self._spec = None
        assert spec.task is not None
        now = time.perf_counter()
        spec.metrics.committed_speculative = True
        spec.metrics.spec_ms = _ms_since(spec.started)
        _rebase(spec.metrics, spec.t0, now)
        spec.t0 = now
        if spec.turn is not None:
            spec.turn.t0 = now
        self._history.extend(spec.pending)
        spec.pending.clear()
        done = spec.task.done()
        self._active = None if done else spec.task
        spec.committed.set()
        await spec.gate.open()
        if done:
            await self._set_state("listening")
            if spec.reportable:
                await self._send_metrics(spec.metrics)
            if self._device is not None:
                await self._device.turn_done()
            self._prime_next_turn()  # the turn's own finally skipped it while uncommitted

    def _remember(self, turn: _Turn, role: str, text: str) -> None:
        """History write; held back while the turn is an uncommitted speculation."""
        if not text.strip():
            return
        spec = turn.spec
        if spec is not None and not spec.committed.is_set():
            spec.pending.append((role, text))
            return
        self._history.add(role, text)

    def _messages(self, turn: _Turn) -> list[dict[str, Any]]:
        spec = turn.spec
        if spec is not None and not spec.committed.is_set():
            return self._history.messages_with(spec.pending)
        return self._history.messages()

    @staticmethod
    def _said(turn: _Turn, start: int = 0) -> str:
        """What was actually played from ``start`` on, marked as cut off."""
        played = " ".join(turn.played[start:]).strip()
        return f"{played}{INTERRUPTED_MARK}" if played else ""

    async def _report(self, metrics: TurnMetrics, spec: _Speculation | None) -> None:
        if spec is not None and not spec.committed.is_set():
            spec.reportable = True
            return
        await self._send_metrics(metrics)

    async def _send_metrics(self, metrics: TurnMetrics) -> None:
        await self._out.send_json({"type": "metrics", **metrics.to_dict()})
        self._svc.metrics.record(self.id, metrics)

    def _prime_router(self) -> None:
        if self._svc.router is None:
            return
        # The next turn's router prompt shows the committed history (its `prior`), so prime exactly that.
        self._spawn(self._run_prime_router(self._svc.router, self._history.messages()[1:]))

    @staticmethod
    async def _run_prime_router(router: Router, history: list[dict[str, Any]]) -> None:
        try:
            await router.prime(history)
        except LLMUnavailable as exc:
            log.debug("router prime skipped: %s", exc)

    def _prime_next_turn(self) -> None:
        """After a turn: cache both slots up to the next user message, off the critical path."""
        self._prime_router()
        prime_chat = getattr(self._svc.llm, "prime_chat", None)
        if prime_chat is not None and self._svc.cfg.llm.prime_after_turn:
            self._spawn(self._run_prime_chat(prime_chat, self._history.messages()))

    @staticmethod
    async def _run_prime_chat(prime_chat: Callable[[list[dict[str, Any]]], Any], messages: list[dict[str, Any]]) -> None:
        try:
            await prime_chat(messages)
        except LLMUnavailable as exc:
            log.debug("chat prime skipped: %s", exc)
        except Exception:
            log.exception("chat prime failed")

    async def _interrupt_speech(self, reason: str | None = None) -> bool:
        """Barge-in: stop Gaja's current reply (and any speculation). Hermes tasks keep running."""
        await self._cancel_speculation()
        self._playback_until = 0.0
        task = self._active
        if task is None:
            return False
        self._active = None
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await self._set_state("listening", reason)
        return True

    async def _cancel_jobs(self, job_id: str | None) -> int:
        """Cancel one Hermes task (or all of them); the ACP worker gets session/cancel."""
        jobs = [j for j in self._jobs.values() if job_id is None or j.id == job_id]
        for job in jobs:
            if job.runner is not None:
                job.runner.cancel()
        await asyncio.gather(*(j.runner for j in jobs if j.runner is not None), return_exceptions=True)
        return len(jobs)

    async def _check_stt(self) -> None:
        degraded = bool(getattr(self._svc.stt, "degraded", False))
        if degraded == self._stt_degraded:
            return
        self._stt_degraded = degraded
        code = "stt_degraded" if degraded else "stt_recovered"
        await self._out.send_json(
            {
                "type": "notice",
                "code": code,
                "message": STT_NOTICES[code],
                "engine": getattr(self._svc.stt, "engine", ""),
            }
        )

    async def _user_turn(
        self,
        metrics: TurnMetrics,
        t0: float,
        audio: np.ndarray | None,
        text: str | None,
        spec: _Speculation | None,
    ) -> None:
        assert self._hello is not None
        out: Outbox = spec.gate if spec is not None else self._out
        device_follow_up = True  # a rejected wake ends silently: no follow-up window
        try:
            await self._set_state("thinking", out=out)
            if audio is not None:
                transcript = await self._svc.stt.transcribe(
                    audio, None if self._hello.lang == "auto" else self._hello.lang
                )
                metrics.stt_ms = round(transcript.duration_ms, 1)
                language = transcript.language if transcript.language in LANGUAGES else self._default_language()
                user_text = transcript.text
                await self._check_stt()
            else:
                language = self._default_language()
                user_text = text or ""
            wake_only = False
            heard = user_text
            if self._device is not None and audio is not None:
                resolved = await self._device.resolve(user_text)
                if resolved is None:
                    device_follow_up = False
                    return
                user_text, wake_only = resolved
            if not user_text.strip() and not wake_only:
                return
            transcript_event = {"type": "transcript", "text": user_text or heard, "lang": language, "final": True}
            if wake_only or heard != user_text:
                transcript_event["heard"] = heard
            await out.send_json(transcript_event)

            turn = _Turn(
                t0=spec.t0 if spec is not None else t0,
                language=language,
                metrics=metrics,
                out=out,
                user_text=user_text,
                spec=spec,
            )
            if spec is not None:
                spec.turn = turn
            if wake_only:
                # "Hej Gaja" alone: acknowledge and keep the mic open; nothing goes to the LLM or the history.
                assert self._device is not None
                turn.metrics.route = "wake"
                await self._device.ack(turn)
                await self._report(metrics, spec)
                return
            prior = self._messages(turn)[1:]
            self._remember(turn, "user", user_text)

            if self._jobs and is_cancel_intent(user_text):
                await self._voice_cancel(turn)
            else:
                await self._answer(turn, prior)
            await self._report(metrics, spec)
        finally:
            if self._active is asyncio.current_task():
                self._active = None
                await self._set_state("listening")
                if self._device is not None and device_follow_up:
                    await self._device.turn_done()
            if spec is None or spec.committed.is_set():
                self._prime_next_turn()

    async def _voice_cancel(self, turn: _Turn) -> None:
        turn.metrics.route = "cancel"
        if turn.spec is not None:
            await turn.spec.committed.wait()
        cancelled = await self._cancel_jobs(None)
        reply = (CANCELLED if cancelled else NOTHING_TO_CANCEL)[turn.language]
        await self._say(turn, reply)
        self._remember(turn, "assistant", reply)

    async def _answer(self, turn: _Turn, prior: list[dict[str, Any]]) -> None:
        route = await self._route(turn, prior)
        turn.metrics.route = route.kind
        handoff: ToolCall | None = None
        if route.kind == "hermes":
            handoff = ToolCall(HANDOFF_TOOL, {"task": turn.user_text, "reason": route.reason})
        elif route.kind == "tool" and await self._tool_turn(turn, prior):
            return
        if handoff is None:
            handoff = await self._local_reply(turn)
        if handoff is not None:
            await self._hand_off(turn, handoff)

    async def _local_reply(self, turn: _Turn) -> ToolCall | None:
        collected: list[str] = []
        start = len(turn.played)
        complete = False
        handoff: ToolCall | None = None
        try:
            handoff = await self._speak_llm(turn, self._messages(turn), allow_tools=True, collected=collected)
            complete = True
        finally:
            # A finished reply is stored as generated (keeps the cached prompt prefix valid). Before a
            # handoff, only what was spoken counts; a reply cut by barge-in is stored as said, marked "…".
            if complete and handoff is None:
                said = "".join(collected).strip()
            elif complete:
                said = " ".join(turn.played[start:]).strip()
            else:
                said = self._said(turn, start)
            self._remember(turn, "assistant", said)
        return handoff

    async def _hand_off(self, turn: _Turn, call: ToolCall) -> None:
        spec = turn.spec
        ack = self._next_ack(turn.language) if not turn.spoken else ""
        if spec is not None and not spec.committed.is_set():
            # Synthesize the ack while still speculating (held by the gate); start Hermes only once committed.
            if ack:
                await self._say(turn, ack)
            await spec.committed.wait()
            self._start_handoff(turn, call)
        else:
            self._start_handoff(turn, call)
            if ack:
                await self._say(turn, ack)
        if ack:
            self._remember(turn, "assistant", ack)

    async def _route(self, turn: _Turn, prior: list[dict[str, Any]]) -> Route:
        router = self._svc.router
        if router is None:
            return Route("local")
        started = time.perf_counter()
        try:
            return await router.classify(turn.user_text, prior)
        except LLMUnavailable as exc:
            log.warning("router unavailable, answering locally: %s", exc)
            return Route("local", reason="router-down")
        finally:
            turn.metrics.router_ms = _ms_since(started)

    async def _tool_turn(self, turn: _Turn, prior: list[dict[str, Any]]) -> bool:
        """Variant C: pick the tool with a constrained generation, run it after commit, speak its text."""
        router, tools = self._svc.router, self._svc.tools
        if router is None or tools is None:
            return False
        started = time.perf_counter()
        try:
            call = await router.select_tool(turn.user_text, prior)
        except LLMUnavailable as exc:
            log.warning("tool selection failed, answering locally: %s", exc)
            call = None
        turn.metrics.tool_select_ms = _ms_since(started)
        if call is None:
            turn.metrics.route = "local"
            return False
        arguments = dict(call.arguments)
        if call.name == "get_weather" and str(arguments.get("location") or "").strip().lower() in PLACEHOLDER_LOCATIONS:
            arguments["location"] = self._svc.cfg.tools.default_location
        if turn.spec is not None:
            await turn.spec.committed.wait()
        started = time.perf_counter()
        result = await tools.execute(call.name, arguments, lang=turn.language, timeout_s=self._svc.cfg.tools.timeout_s)
        turn.metrics.tool_ms = _ms_since(started)
        turn.metrics.tool = call.name
        await turn.out.send_json(
            {"type": "tool", "name": call.name, "arguments": arguments, "ok": result.error is None, "text": result.speak}
        )
        start = len(turn.played)
        complete = False
        try:
            await self._speak_text(turn, result.speak)
            complete = True
        finally:
            self._remember(turn, "assistant", result.speak if complete else self._said(turn, start))
        return True

    def _next_ack(self, language: str) -> str:
        options = ACKS[language]
        text = options[self._acks % len(options)]
        self._acks += 1
        return text

    def _chunker(self) -> SentenceChunker:
        cfg = self._svc.cfg.chunker
        return SentenceChunker(
            first_min_words=cfg.first_chunk_min_words,
            min_words=cfg.min_words,
            max_chars=cfg.max_chars,
            first_max_words=cfg.first_chunk_max_words,
        )

    def _sentences(self, turn: _Turn, raw: list[str], style: Style) -> tuple[list[Sentence], Style]:
        return _to_sentences(
            raw,
            style,
            auto_emotion=self._svc.cfg.style.auto_emotion,
            language=turn.language,
            normalize=self._svc.normalize if self._svc.cfg.text.normalize_tts else None,
        )

    def _enqueue(
        self, turn: _Turn, raw: list[str], style: Style, queue: asyncio.Queue[_Item | None], *, guard: bool
    ) -> tuple[Style, bool]:
        sentences, style = self._sentences(turn, raw, style)
        for sentence in sentences:
            if guard and promises_handoff(sentence.text):
                return style, True
            if turn.metrics.first_sentence_ms is None:
                turn.metrics.first_sentence_ms = _ms_since(turn.t0)
            queue.put_nowait(sentence)
        return style, False

    async def _speak_text(self, turn: _Turn, text: str) -> None:
        chunker = self._chunker()
        sentences, _ = self._sentences(turn, chunker.feed(text) + chunker.flush(), Style())
        for sentence in sentences:
            if turn.metrics.first_sentence_ms is None:
                turn.metrics.first_sentence_ms = _ms_since(turn.t0)
            await self._play(turn, sentence)

    async def _speak_llm(
        self, turn: _Turn, messages: list[dict[str, Any]], *, allow_tools: bool, collected: list[str]
    ) -> ToolCall | None:
        queue: asyncio.Queue[_Item | None] = asyncio.Queue()
        producer = asyncio.create_task(
            self._produce(turn, messages, allow_tools=allow_tools, queue=queue, collected=collected)
        )
        try:
            while (item := await queue.get()) is not None:
                if isinstance(item, Sentence):
                    await self._play(turn, item)
                elif isinstance(item, ToolCall):
                    return item
                else:
                    await self._error("language model unavailable", out=turn.out)
                    await self._say(turn, LLM_DOWN[turn.language])
                    return None
            return None
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    async def _produce(
        self,
        turn: _Turn,
        messages: list[dict[str, Any]],
        *,
        allow_tools: bool,
        queue: asyncio.Queue[_Item | None],
        collected: list[str],
    ) -> None:
        chunker = self._chunker()
        label = SpeakerPrefixStripper()
        stripper = StyleTagStripper()
        style = Style()
        started = time.perf_counter()
        try:
            # Tools are always sent so the rendered system prompt (and its cached prefix) is identical
            # on every turn; a tool call on a turn that may not hand off is ignored.
            async for event in self._svc.llm.stream(messages, allow_tools=True):
                if turn.metrics.llm_ttft_ms is None:
                    turn.metrics.llm_ttft_ms = _ms_since(started)
                if isinstance(event, ToolCall):
                    if not allow_tools:
                        log.info("ignoring tool call %s on a no-tools turn", event.name)
                        continue
                    style, _ = self._enqueue(turn, chunker.flush(), style, queue, guard=False)
                    queue.put_nowait(event)
                    return
                text = label.feed(event.text)
                visible = stripper.feed(text)
                if visible:
                    collected.append(visible)
                style, promised = self._enqueue(turn, chunker.feed(text), style, queue, guard=allow_tools)
                if promised:
                    queue.put_nowait(self._safety_net_call(turn))
                    return
            tail = label.flush()
            collected.append(stripper.feed(tail) + stripper.flush())
            style, promised = self._enqueue(turn, chunker.feed(tail) + chunker.flush(), style, queue, guard=allow_tools)
            if promised:
                queue.put_nowait(self._safety_net_call(turn))
        except LLMUnavailable as exc:
            log.warning("LLM unavailable: %s", exc)
            queue.put_nowait(_LLMDown())
        except Exception:
            log.exception("LLM stream failed")
            queue.put_nowait(_LLMDown())
        finally:
            queue.put_nowait(None)

    @staticmethod
    def _safety_net_call(turn: _Turn) -> ToolCall:
        return ToolCall(HANDOFF_TOOL, {"task": turn.user_text, "reason": "safety-net"})

    async def _play(self, turn: _Turn, sentence: Sentence) -> None:
        turn.spoken = True
        engine = await self._engine_for(turn.language)
        self._utterances += 1
        utterance_id = self._utterances
        await turn.out.send_json(
            {"type": "utterance", "id": utterance_id, "text": sentence.text, "style": asdict(sentence.style)}
        )
        started = time.perf_counter()
        audio = await engine.synthesize(sentence.speech or sentence.text, sentence.style, turn.language)
        if turn.metrics.tts_first_ms is None:
            turn.metrics.tts_first_ms = _ms_since(started)
        if audio.size == 0:
            # Tell the client this utterance has no audio, so it does not wait for it.
            await turn.out.send_json({"type": "audio_end", "id": utterance_id, "empty": True})
            return
        await self._set_state("speaking", out=turn.out)
        if turn.metrics.e2e_first_audio_ms is None:
            turn.metrics.e2e_first_audio_ms = _ms_since(turn.t0)
        await turn.out.send_json({"type": "audio_start", "id": utterance_id, "sample_rate": engine.sample_rate})
        await turn.out.send_bytes(float_to_pcm16(audio))
        await turn.out.send_json({"type": "audio_end", "id": utterance_id})
        turn.played.append(sentence.text)
        if turn.spec is None or turn.spec.committed.is_set():
            now = time.perf_counter()
            self._playback_until = max(now, self._playback_until) + audio.size / engine.sample_rate + ECHO_TAIL_S

    async def on_timer(self, timer: dict[str, Any], note: str | None = None) -> None:
        """WS event to client plus spoken Polish announcement for sessions with audio."""
        timer_id = timer.get("id", "")
        label = timer.get("label", "")
        kind = timer.get("kind", "timer")
        await self._out.send_json({"type": "timer", "id": timer_id, "label": label, "kind": kind})
        if self._hello is not None:
            if note == "restart":
                spoken_text = (
                    f"Minutnik {label} minął podczas restartu."
                    if kind == "timer"
                    else f"Przypomnienie: {label} minęło podczas restartu."
                )
            elif kind == "timer":
                spoken_text = f"Minutnik {label} minął!" if label and label != "minutnik" else "Minutnik minął!"
            else:
                spoken_text = f"Przypomnienie: {label}."
            await self._announce(spoken_text)

    async def _announce(self, text: str) -> None:
        async with self._result_lock:
            while True:
                active = self._active
                if active is not None and not active.done():
                    await asyncio.wait({active})
                    continue
                if self._spec is not None or (self._endpointer is not None and self._endpointer.speaking):
                    await asyncio.sleep(0.1)
                    continue
                break
            task = asyncio.create_task(self._announcement_turn(text))
            self._active = task
            await asyncio.wait({task})

    async def _announcement_turn(self, text: str) -> None:
        self._turns += 1
        turn = _Turn(
            t0=time.perf_counter(),
            language="pl",
            metrics=TurnMetrics(turn=self._turns, kind="notification", route="timer"),
            out=self._out,
        )
        try:
            await self._speak_text(turn, text)
        finally:
            await self._send_metrics(turn.metrics)
            if self._active is asyncio.current_task():
                self._active = None
                await self._set_state("listening")
                if self._device is not None:
                    await self._device.turn_done()
            self._prime_next_turn()

    async def _say(self, turn: _Turn, text: str) -> None:
        sentences, _ = self._sentences(turn, [text], Style())
        for sentence in sentences or [Sentence(text, Style())]:
            await self._play(turn, sentence)

    # ---- Hermes ---------------------------------------------------------------------------

    def _start_handoff(self, turn: _Turn, call: ToolCall) -> None:
        task = str(call.arguments.get("task", "")).strip() or turn.user_text
        reason = str(call.arguments.get("reason", "")).strip()
        job = _HermesJob(id=uuid.uuid4().hex[:8], task=task, language=turn.language)
        context = self._history.context()
        # The current user message is already the last history entry; Hermes gets it as the task.
        if context and context[-1]["role"] == "user" and context[-1]["content"] == turn.user_text:
            context = context[:-1]
        self._jobs[job.id] = job
        job.runner = asyncio.create_task(self._run_hermes(job, context, reason))

    async def _run_hermes(self, job: _HermesJob, context: list[dict[str, Any]], reason: str) -> None:
        interval = self._svc.cfg.hermes.progress_interval_s
        coalescer = ProgressCoalescer(interval)
        next_event: asyncio.Future[Any] | None = None
        events: Any = None
        try:
            await self._handoff_event(job, "started", reason)
            events = self._svc.hermes.submit(
                job.task, context=context, session_id=self.id, language=job.language
            ).__aiter__()
            result: str | None = None
            while True:
                if next_event is None:
                    next_event = asyncio.ensure_future(events.__anext__())
                done, _ = await asyncio.wait({next_event}, timeout=interval if coalescer.pending else None)
                if not done:
                    await self._emit_progress(job, coalescer.flush())
                    continue
                try:
                    event = next_event.result()
                except StopAsyncIteration:
                    break
                finally:
                    next_event = None
                if event.kind == "error":
                    await self._emit_progress(job, coalescer.flush())
                    await self._hermes_failed(job, event.text)
                    return
                if event.kind == "result":
                    result = event.text
                elif event.kind == "progress":
                    kind = (event.data or {}).get("type", TEXT_DELTA)
                    await self._emit_progress(job, coalescer.push(kind, event.text))
            await self._emit_progress(job, coalescer.flush())
            if result is None:
                await self._hermes_failed(job, "Hermes returned no result")
                return
            await self._handoff_event(job, "done", result)
            self._jobs.pop(job.id, None)
            await self._speak_result(result, job.language)
        except asyncio.CancelledError:
            if self._jobs.pop(job.id, None) is not None:
                try:
                    await asyncio.shield(self._handoff_event(job, "cancelled", ""))
                except Exception:
                    pass
            raise
        except Exception as exc:
            log.exception("Hermes task failed")
            await self._hermes_failed(job, str(exc))
        finally:
            self._jobs.pop(job.id, None)
            if next_event is not None:
                next_event.cancel()
                await asyncio.gather(next_event, return_exceptions=True)
            if events is not None:
                await events.aclose()

    async def _emit_progress(self, job: _HermesJob, items: list[ProgressItem]) -> None:
        for item in items:
            await self._handoff_event(job, "progress", item.text)

    async def _hermes_failed(self, job: _HermesJob, detail: str) -> None:
        log.warning("Hermes handoff failed: %s", detail)
        await self._handoff_event(job, "error", detail)
        self._jobs.pop(job.id, None)
        await self._speak_result(HERMES_DOWN[job.language], job.language, direct=True)

    async def _handoff_event(self, job: _HermesJob, state: str, text: str) -> None:
        await self._out.send_json({"type": "handoff", "state": state, "id": job.id, "task": job.task, "text": text})

    async def _speak_result(self, result: str, language: str, *, direct: bool = False) -> None:
        """Announce a Hermes result once Gaja is not talking and the user is not speaking."""
        async with self._result_lock:
            while True:
                active = self._active
                if active is not None and not active.done():
                    await asyncio.wait({active})
                    continue
                if self._spec is not None or (self._endpointer is not None and self._endpointer.speaking):
                    await asyncio.sleep(0.1)
                    continue
                break
            task = asyncio.create_task(self._result_turn(result, language, direct=direct))
            self._active = task
            await asyncio.wait({task})

    async def _result_turn(self, result: str, language: str, *, direct: bool = False) -> None:
        self._turns += 1
        turn = _Turn(
            t0=time.perf_counter(),
            language=language,
            metrics=TurnMetrics(turn=self._turns, kind="result", route="hermes"),
            out=self._out,
        )
        try:
            if not direct:
                self._remember(turn, "user", RESULT_PROMPT.format(result=result))
            if direct or len(result) <= self._svc.cfg.hermes.direct_max_chars:
                complete = False
                try:
                    await self._speak_text(turn, result)
                    complete = True
                finally:
                    self._remember(turn, "assistant", result if complete else self._said(turn))
                return
            collected: list[str] = []
            complete = False
            try:
                await self._speak_llm(turn, self._messages(turn), allow_tools=False, collected=collected)
                complete = True
            finally:
                self._remember(turn, "assistant", "".join(collected).strip() if complete else self._said(turn))
        finally:
            await self._send_metrics(turn.metrics)
            if self._active is asyncio.current_task():
                self._active = None
                await self._set_state("listening")
            self._prime_next_turn()

    async def _set_state(self, state: str, reason: str | None = None, *, out: Outbox | None = None) -> None:
        if state == self._state and reason is None:
            return
        self._state = state
        payload: dict[str, Any] = {"type": "state", "state": state}
        if reason is not None:
            payload["reason"] = reason
        await (out or self._out).send_json(payload)

    async def _error(self, message: str, *, out: Outbox | None = None) -> None:
        await (out or self._out).send_json({"type": "error", "message": message})
