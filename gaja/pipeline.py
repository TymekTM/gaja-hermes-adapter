from __future__ import annotations

import asyncio
import datetime
import json
import logging
import resource
import re
import time
import uuid
from zoneinfo import ZoneInfo
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel, Field, ValidationError

from gaja.audio.pcm import float_to_pcm16, pcm16_to_float
from gaja.audio.vad import Endpointer, ProbFn, VadEvent
from gaja.config import Settings
from gaja.devices.aec import FRAME_REF
from gaja.devices.audio import DeviceCaps, ResamplingOutbox
from gaja.devices.controller import DeviceController, LostReply, REPLY_RECOVERY_NOTICE
from gaja.devices.registry import DeviceInfo
from gaja.hermes.base import HermesClient
from gaja.hermes.progress import TEXT_DELTA, ProgressCoalescer, ProgressItem
from gaja.household import UNKNOWN, Identity, Profile, explicit_profile
from gaja.llm.client import LLMClient, LLMUnavailable
from gaja.llm.history import History
from gaja.llm.parse import ToolCall
from gaja.llm.prompt import SystemPrompt
from gaja.llm.router import Route, Router, clock_fast_path, promises_handoff
from gaja.metrics import MetricsRecorder, TurnMetrics
from gaja.stt.base import STT
from gaja.text.chunker import SentenceChunker, SpeakerPrefixStripper, StyleTagStripper, speakable
from gaja.text.emotion import auto_style, has_explicit_tag
from gaja.text.style import Style, parse_styled
from gaja.text.normalize import normalize_for_tts
from gaja.tools.base import PLACEHOLDER_LOCATIONS, current_session_context, polish_declension
from gaja.tools.weather_tool import resolve_tool_location
from gaja.tools.time_tool import PL_MONTHS, PL_WEEKDAYS
from gaja.tts.base import TTS, EnginePool

if TYPE_CHECKING:
    from gaja.household import HouseholdStore
    from gaja.memory import MemoryStore
    from gaja.stt.vocab import VocabularyManager
    from gaja.timers import TimerManager

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
# Echo of Gaja's own voice keeps arriving while the client plays it back; the VAD is strict until then.
ECHO_TAIL_S = 0.4
RESULT_QUIET_S = 1.0

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


# "Gaja, nowy temat": fresh Hermes sessions and a fresh conversation for this person.
NEW_TOPIC = re.compile(
    r"^\W*(?:(?:hej\s+|hey\s+)?gaj\w*\W+)?(?:(?:to\s+)?(?:nowy\s+temat|zmiana\s+tematu|zacznijmy\s+od\s+nowa)"
    r"|(?:new\s+topic|let'?s\s+start\s+over))\W*$",
    re.IGNORECASE,
)
NEW_TOPIC_REPLY = {"pl": "Dobrze, zaczynamy nowy temat.", "en": "Okay, new topic."}


def is_new_topic(text: str) -> bool:
    return bool(NEW_TOPIC.match(text.strip()))


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
    vocab: VocabularyManager | None = None
    smart_turn: Any = None
    smart_turn_executor: Any = None
    timer_manager: TimerManager | None = None
    memory_store: MemoryStore | None = None
    household: HouseholdStore | None = None
    sessions: set[Session] = field(default_factory=set)
    lost_replies: dict[str, LostReply] = field(default_factory=dict)

    def get_system_prompt(self, identity: Identity = UNKNOWN) -> str:
        base = self.prompt.text
        # Date only: it changes once a day, so the llama.cpp prompt cache stays warm between turns.
        now = datetime.datetime.now(ZoneInfo("Europe/Warsaw"))
        today = f"{PL_WEEKDAYS[now.weekday()]}, {now.day} {PL_MONTHS[now.month]} {now.year} roku"
        base = base + f"\n\nDzisiejsza data (Warszawa): {today} ({normalize_for_tts(today, 'pl')})."
        if self.household is not None and self.household.data.note.strip():
            base = base + "\n\nO domu:\n" + self.household.data.note.strip()
        if identity.person is not None:
            base = base + f"\n\nRozmawiasz z: {identity.person.name}."
        if self.memory_store is not None:
            base = base + self.memory_store.format_prompt_block(identity.person.id if identity.person else None)
        return base

    def hermes_instructions(self, identity: Identity) -> str | None:
        """Who Hermes is talking to, for setups with a household (None keeps the old behaviour)."""
        if self.household is None or not (self.household.data.people or self.household.data.note.strip()):
            return None
        who = identity.person.name if identity.person is not None else "nierozpoznana osoba (gość)"
        text = f"[Gaja: rozmówca: {who}.]"
        note = self.household.data.note.strip()
        if note:
            text += f"\n[O domu: {note}]"
        return text


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
    # Web/CLI: a stable random id per phone/browser, bound to a person in the admin panel.
    client_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{8,64}$")
    client_label: str | None = Field(default=None, max_length=80)


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
    reply_text: str = ""
    delivered: bool = False
    lost_saved: bool = False  # the reply is already held for reconnect recovery (record it once)
    spec: _Speculation | None = None
    identity: Identity = UNKNOWN
    voice: bool = False


@dataclass
class _HermesJob:
    id: str
    task: str
    language: str
    runner: asyncio.Task[None] | None = None
    identity: Identity = UNKNOWN
    voice: bool = False
    utterance: str = ""
    profile: str | None = None


class _Gate:
    """Holds a speculative turn's outbound messages until it is committed, or drops them."""

    def __init__(self, out: Outbox) -> None:
        self._out = out
        self._held: list[dict[str, Any] | bytes] | None = []
        self._dropped = False
        self._lock = asyncio.Lock()

    async def send_json(self, payload: dict[str, Any]) -> None:
        await self._send(payload)

    async def send_bytes(self, data: bytes) -> None:
        await self._send(data)

    async def _send(self, item: dict[str, Any] | bytes) -> None:
        async with self._lock:
            if self._dropped:
                return
            if self._held is not None:
                self._held.append(item)
            elif isinstance(item, bytes):
                await self._out.send_bytes(item)
            else:
                await self._out.send_json(item)

    async def open(self) -> None:
        async with self._lock:
            if self._dropped or self._held is None:
                return
            held = self._held
            self._held = None
            for item in held:
                if self._dropped:
                    break
                if isinstance(item, bytes):
                    await self._out.send_bytes(item)
                else:
                    await self._out.send_json(item)

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
    audio_s: float = 0.0


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


def format_timer_announcement(timer: dict[str, Any], note: str | None = None, lang: str = "pl") -> str:
    label = (timer.get("label") or "").strip()
    kind = timer.get("kind", "timer")
    if kind == "reminder":
        if note == "restart":
            spoken = f"Przypomnienie: {label} minęło podczas restartu."
        else:
            spoken = f"Przypomnienie: {label}."
        return normalize_for_tts(spoken, lang)

    m_dur = re.match(r"^(\d+)\s+minut\w*$", label, re.I)
    if m_dur:
        count = int(m_dur.group(1))
        if count == 1:
            dur_phrase = "na jedną minutę"
        else:
            word_noun = polish_declension(count, "minutę", "minuty", "minut")
            dur_phrase = f"na {count} {word_noun}"
        if note == "restart":
            spoken = f"Minął minutnik {dur_phrase} podczas restartu."
        else:
            spoken = f"Minął minutnik {dur_phrase}."
    elif label and label.lower() != "minutnik":
        if note == "restart":
            spoken = f"Minutnik {label} minął podczas restartu."
        else:
            spoken = f"Minął minutnik: {label}."
    else:
        spoken = "Minutnik minął podczas restartu." if note == "restart" else "Minutnik minął!"

    return normalize_for_tts(spoken, lang)


class Session:
    def __init__(self, services: Services, out: Outbox, device: DeviceInfo | None = None) -> None:
        self.id = uuid.uuid4().hex
        self._svc = services
        self._svc.sessions.add(self)
        self._out: Outbox = out
        self.device_id: str | None = device.id if device is not None else None
        self.is_device: bool = device is not None
        self.last_active: float = time.perf_counter()
        # A reconnect makes a newer session the device's owner; an older one must not arm reply recovery.
        self._opened_at: float = self.last_active
        # Device mode (WS /device, protocol v1.8): output is resampled to the device rate and the wake /
        # listen-window flow is handled by DeviceController.
        self._device: DeviceController | None = None
        if device is not None:
            self._out = ResamplingOutbox(out, services.cfg.devices.out_chunk_ms)
            self._device = DeviceController(self, services.cfg.devices, self._out, device)
        self._identity: Identity = UNKNOWN
        self._history = self._new_history(UNKNOWN)
        # One conversation per person: a shared device switches between them as speakers change.
        self._histories: dict[str, History] = {UNKNOWN.key: self._history}
        self._hello: Hello | None = None
        self._engine_override: str | None = None
        self._endpointer: Endpointer | None = None
        self._state = ""
        self._active: asyncio.Task[None] | None = None
        self._current_turn: _Turn | None = None
        self._jobs: dict[str, _HermesJob] = {}
        self._background: set[asyncio.Task[None]] = set()
        self._result_lock = asyncio.Lock()
        self._spec: _Speculation | None = None
        self._turns = 0
        self._utterances = 0
        self._acks = 0
        self._speech_end_at = 0.0
        self._playback_end = 0.0
        self._last_interrupt = 0.0
        self._playback_until = 0.0
        self._gone = False  # the client's socket died under a send: a normal end, not a turn error
        self._stt_degraded = False
        self._last_user_text: str | None = None
        self._last_user_time: float = 0.0

    # ---- client messages -------------------------------------------------------------------

    async def handle_text(self, raw: str) -> None:
        self.last_active = time.perf_counter()
        try:
            # strict=False: ESPHome log lines carry raw ANSI colour bytes, which ArduinoJson leaves unescaped.
            msg = json.loads(raw, strict=False)
        except json.JSONDecodeError:
            if self._device is not None:
                # Never answered: the device logs the error and forwards that line, which would loop.
                log.warning("device sent invalid JSON (%d chars)", len(raw))
            else:
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
        elif kind in ("record_started", "record_mark", "record_end", "record_error"):
            if self._device is None:
                await self._error(f"{kind} is only for device clients")
            else:
                await self._device.on_record_event(msg)
        elif kind == "log":
            if self._device is not None:
                self._device.on_log(msg)  # never answered: an error reply would itself be logged and forwarded
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
        self.last_active = time.perf_counter()
        if self._endpointer is None:
            await self._error("send hello first")
            return
        if self._device is not None and self._device.caps.ref:
            # Protocol v1.11: tagged binary frames (1-byte tag + PCM16 16 kHz payload).
            if not data:
                return
            kind, data = data[0], data[1:]
            if kind == FRAME_REF:
                self._device.feed_ref(data)
                return
        if len(data) % 2:
            await self._error("PCM16 frame has odd length")
            return
        if self._device is not None and self._device.recording_streaming:
            # Frames while a take records belong to the recording, never the pipeline (protocol v1.10).
            self._device.on_record_frame(data)
            return
        samples = pcm16_to_float(data)
        if self._device is not None:
            if self._device.caps.ref:
                # Remove the device's own playback echo before the VAD/wake path sees it.
                samples = self._device.cancel_echo(samples)
            filtered = self._device.filter_audio(samples)
            if filtered is None:
                return
            samples = filtered
        self._endpointer.set_strict(time.perf_counter() < self._playback_until)
        for event in self._endpointer.feed(samples):
            await self._on_vad(event)

    def update_system_prompt(self) -> None:
        self._history.set_system(self._svc.get_system_prompt(self._identity))

    def _new_history(self, identity: Identity) -> History:
        cfg = self._svc.cfg.llm
        return History(self._svc.get_system_prompt(identity), cfg.history_max_turns, cfg.history_max_chars)

    @property
    def identity(self) -> Identity:
        return self._identity

    def _resolve_identity(self) -> Identity:
        household = self._svc.household
        if household is None:
            return UNKNOWN
        client_id = self._hello.client_id if self._hello is not None else None
        return household.identify(client_id=client_id, device_id=self.device_id)

    def refresh_identity(self) -> None:
        self._switch_identity(self._resolve_identity())

    def _switch_identity(self, identity: Identity) -> None:
        if identity.key != self._identity.key:
            history = self._histories.get(identity.key)
            if history is None:
                history = self._histories[identity.key] = self._new_history(identity)
            self._history = history
        self._identity = identity
        self.update_system_prompt()

    def record_lost_reply(self) -> None:
        """If this device session died while delivering a user-turn reply, save the reply for reconnect recovery."""
        if self._device is None:
            return
        turn = self._current_turn
        if turn is None or turn.delivered:
            return
        if turn.spec is not None and not turn.spec.committed.is_set():
            return
        self._hold_lost_reply(turn)

    def _owns_device(self) -> bool:
        """True while no newer session has taken over this device's connection."""
        return not any(
            other is not self and other.device_id == self.device_id and other._opened_at > self._opened_at
            for other in self._svc.sessions
        )

    def _hold_lost_reply(self, turn: _Turn) -> None:
        """Hold ``turn``'s reply for reconnect recovery. At most once per turn, and only while still owning the device.

        The reply is recorded twice on a reconnect: ``supersede()`` records it and the old connection records
        it again when it finishes closing. That late close can land after the reconnected device already spoke
        the recovered reply, so a second record would arm a second recovery window. The late close can also land
        after the reconnected device already answered a newer user turn, so a session that no longer owns the
        device must not arm a reply behind the live session's back.
        """
        if self._device is None or turn.lost_saved or not self._owns_device():
            return
        text = turn.reply_text.strip()
        if not text:
            return
        turn.lost_saved = True
        self._device.save_lost_reply(text, user_text=turn.user_text, language=turn.language)

    async def close(self) -> None:
        self._gone = True
        self.record_lost_reply()
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
        if hello.client_id is not None and self._svc.household is not None:
            self._svc.household.touch_client(hello.client_id, hello.client_label or "", time.time())
        self._switch_identity(self._resolve_identity())
        if self._endpointer is None:
            prob = await asyncio.to_thread(self._svc.new_vad)
            self._endpointer = Endpointer(
                prob,
                self._svc.cfg.vad,
                smart_turn=self._svc.smart_turn,
                executor=self._svc.smart_turn_executor,
            )
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
        if self._device is not None:
            self._device.schedule_recovery()
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
        if self._device is not None:
            await self._device.cancel_recovery()
        await self._interrupt_speech()
        self._turns += 1
        t0 = self._speech_end_at if audio is not None else time.perf_counter()
        metrics = TurnMetrics(turn=self._turns, kind="audio" if audio is not None else "text")
        self._active = asyncio.create_task(self._user_turn(metrics, t0, audio, text, None))

    async def _start_speculation(self, audio: np.ndarray) -> None:
        if self._device is not None:
            await self._device.cancel_recovery()
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
        send_started = time.perf_counter()
        try:
            await spec.gate.open()
        except (ConnectionError, WebSocketDisconnect):
            # The device died while the gate was flushing: none of the held reply can be delivered.
            self._gone = True
            spec.committed.set()  # a turn waiting on the commit must unwind, never hang on it
            if spec.turn is not None:
                self._hold_lost_reply(spec.turn)
            if not spec.task.done():
                # Do not let the turn go on and start Hermes / a tool for a dead device.
                spec.task.cancel()
                await asyncio.gather(spec.task, return_exceptions=True)
            return
        if spec.audio_s > 0.0:
            self._playback_end = max(send_started, self._playback_end) + spec.audio_s
            self._playback_until = self._playback_end + ECHO_TAIL_S
        done = spec.task.done()
        self._active = None if done else spec.task
        spec.committed.set()
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
        self._last_interrupt = time.perf_counter()
        if self._device is not None:
            self._device.clear_lost_reply()
        await self._cancel_speculation()
        # A device keeps playing what it already buffered (up to ~44 s of reply) unless it is
        # told to stop: unlike a browser it does not notice the cancelled downstream by itself.
        # v1.11 playback_stop; harmless no-op on the device when nothing is playing.
        was_playing = time.perf_counter() < self._playback_until or self._active is not None
        self._playback_end = 0.0
        self._playback_until = 0.0
        self._current_turn = None
        task = self._active
        if task is None:
            return False
        self._active = None
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if self._device is not None and was_playing:
            await self._out.send_json({"type": "playback_stop"})
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
        turn: _Turn | None = None
        try:
            await self._set_state("thinking", out=out)
            if audio is not None:
                hotwords = (self._device.wake_hotwords() if self._device else None) or (
                    self._svc.vocab.hotwords() if self._svc.vocab else None
                )
                flt0 = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
                transcript = await self._svc.stt.transcribe(
                    audio, None if self._hello.lang == "auto" else self._hello.lang,
                    hotwords=hotwords,
                )
                flt1 = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
                metrics.stt_ms = round(transcript.duration_ms, 1)
                metrics.stt_wait_ms = round(getattr(transcript, "wait_ms", 0.0), 1)
                metrics.stt_majflt = max(0, flt1 - flt0)
                metrics.stt_audio_s = round(transcript.audio_s, 2)
                if metrics.stt_ms > 1500:
                    log.info(
                        "slow STT: stt_ms=%.1f stt_wait_ms=%.1f stt_majflt=%d stt_audio_s=%.2f",
                        metrics.stt_ms,
                        metrics.stt_wait_ms,
                        metrics.stt_majflt,
                        metrics.stt_audio_s,
                    )
                language = transcript.language if transcript.language in LANGUAGES else self._default_language()
                user_text = transcript.text
                if self._svc.vocab:
                    now_ts = time.time()
                    corrected = self._svc.vocab.correct(user_text)
                    self._svc.vocab.observe(
                        user_text,
                        previous_turn_text=self._last_user_text,
                        timestamp=now_ts,
                        previous_turn_time=self._last_user_time,
                    )
                    self._last_user_text = user_text
                    self._last_user_time = now_ts
                    user_text = corrected
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
            self._current_turn = turn
            self._switch_identity(self._resolve_identity())
            turn.identity = self._identity
            turn.voice = audio is not None
            prior = self._messages(turn)[1:]
            self._remember(turn, "user", user_text)

            if self._jobs and is_cancel_intent(user_text):
                await self._voice_cancel(turn)
            elif is_new_topic(user_text):
                await self._new_topic(turn)
            else:
                await self._answer(turn, prior)
            turn.delivered = True
            await self._report(metrics, spec)
        except (ConnectionError, WebSocketDisconnect):
            # The client disappeared mid-turn (closed socket or send timeout): nobody is left to hear the rest.
            self._gone = True
            self.record_lost_reply()
            log.info("client left during the turn; ending it quietly")
        finally:
            if self._gone:
                self.record_lost_reply()
            if turn is not None and self._current_turn is turn:
                self._current_turn = None
            if self._active is asyncio.current_task():
                self._active = None
                if not self._gone:
                    try:
                        await self._set_state("listening")
                        if self._device is not None and device_follow_up:
                            await self._device.turn_done()
                    except (ConnectionError, WebSocketDisconnect):
                        self._gone = True
            if spec is None or spec.committed.is_set():
                self._prime_next_turn()

    async def _recover_turn(self, lost: LostReply) -> None:
        """Speak the reconnect notice and the unfinished user-turn reply once after the device reconnects."""
        self._turns += 1
        turn = _Turn(
            t0=time.perf_counter(),
            language=lost.language if lost.language in LANGUAGES else "pl",
            metrics=TurnMetrics(turn=self._turns, kind="notification", route="recovery"),
            out=self._out,
            user_text=lost.user_text,
            reply_text=lost.text,
        )
        try:
            if lost.user_text:
                self._remember(turn, "user", lost.user_text)
            await self._say(turn, REPLY_RECOVERY_NOTICE)
            await self._speak_text(turn, lost.text)
            turn.delivered = True
            self._remember(turn, "assistant", lost.text)
        except (ConnectionError, WebSocketDisconnect):
            self._gone = True
            log.info("client left during reply recovery; ending it quietly")
        finally:
            try:
                if not self._gone:
                    await self._send_metrics(turn.metrics)
                if self._active is asyncio.current_task():
                    self._active = None
                    if not self._gone:
                        await self._set_state("listening")
                        if self._device is not None:
                            await self._device.turn_done()
            except (ConnectionError, WebSocketDisconnect):
                self._gone = True
            self._prime_next_turn()

    async def _voice_cancel(self, turn: _Turn) -> None:
        turn.metrics.route = "cancel"
        if turn.spec is not None:
            await turn.spec.committed.wait()
        cancelled = await self._cancel_jobs(None)
        reply = (CANCELLED if cancelled else NOTHING_TO_CANCEL)[turn.language]
        await self._say(turn, reply)
        self._remember(turn, "assistant", reply)

    async def _new_topic(self, turn: _Turn) -> None:
        turn.metrics.route = "new_topic"
        if turn.spec is not None:
            await turn.spec.committed.wait()
        reset = getattr(self._svc.hermes, "reset", None)
        if reset is not None:
            reset(turn.identity.key)
        new_topic = getattr(self._svc.hermes, "new_topic", None)
        if new_topic is not None:
            new_topic(self.id)  # the ACP worker keys its Hermes session by connection, not by person
        self._history = self._histories[turn.identity.key] = self._new_history(turn.identity)
        await self._say(turn, NEW_TOPIC_REPLY[turn.language])

    async def _answer(self, turn: _Turn, prior: list[dict[str, Any]]) -> None:
        self.update_system_prompt()
        route = await self._route(turn, prior)
        turn.metrics.route = route.kind
        handoff: ToolCall | None = None
        if route.kind == "hermes":
            handoff = ToolCall(HANDOFF_TOOL, {"task": turn.user_text, "reason": route.reason})
        elif route.kind == "tool" and await self._tool_turn(turn, prior):
            return
        elif route.reason == "router-down" and self._svc.tools is not None:
            # A cold or crashed router must not send a plain clock question to Hermes.
            fast = clock_fast_path(turn.user_text)
            if fast is not None and await self._execute_tool_call(turn, fast):
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
                await self._say(turn, ack, record_reply=False)
            await spec.committed.wait()
            self._start_handoff(turn, call)
        else:
            self._start_handoff(turn, call)
            if ack:
                await self._say(turn, ack, record_reply=False)
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
        return await self._execute_tool_call(turn, call, prior=prior)

    async def _execute_tool_call(
        self,
        turn: _Turn,
        call: ToolCall,
        *,
        prior: list[dict[str, Any]] | None = None,
    ) -> bool:
        tools = self._svc.tools
        if tools is None:
            return False
        arguments = dict(call.arguments)
        if call.name in ("get_weather", "get_sun"):
            arguments["location"] = resolve_tool_location(
                arguments.get("location"),
                turn.user_text,
                self._svc.cfg.tools.default_location,
                history=prior,
            )
            if call.name == "get_weather":
                if re.search(r"\bjutro\b|\btomorrow\b", turn.user_text, re.I):
                    arguments["when"] = "tomorrow"
                elif "when" not in arguments and re.search(r"\bw\s+weekend\b", turn.user_text, re.I):
                    # The 7-day forecast covers the coming Saturday/Sunday: ask for both days.
                    arguments["when"] = "weekend"
                elif "when" not in arguments and re.search(
                    r"\bdzi[sś]\b|\bdzisiaj\b|\btoday\b", turn.user_text, re.I
                ):
                    arguments["when"] = "today"
        elif call.name == "get_time":
            if (arguments.get("kind") is None or arguments.get("kind") == "all") and not arguments.get("target_date"):
                if re.search(r"godzin|czas|time|clock", turn.user_text, re.I) and not re.search(r"dat[aęy]|dzie[nń]|dni\b|date|day", turn.user_text, re.I):
                    arguments["kind"] = "time"
                elif re.search(r"dat[aęy]|dzie[nń]|dni\b|date|day|weekday", turn.user_text, re.I) and not re.search(r"godzin|czas|time|clock", turn.user_text, re.I):
                    arguments["kind"] = "date"
        if turn.spec is not None:
            await turn.spec.committed.wait()
        started = time.perf_counter()
        tok = current_session_context.set({
            "session_id": self.id,
            "device_id": self.device_id,
            "person": turn.identity.person.id if turn.identity.person is not None else None,
            "utterance": turn.user_text,
            "default_location": self._svc.cfg.tools.default_location,
        })
        try:
            result = await tools.execute(call.name, arguments, lang=turn.language, timeout_s=self._svc.cfg.tools.timeout_s)
        finally:
            current_session_context.reset(tok)
        turn.metrics.tool_ms = _ms_since(started)
        turn.metrics.tool = call.name
        turn.metrics.route = "tool"
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
        if not turn.reply_text:
            turn.reply_text = " ".join(s.text for s in sentences).strip() or text.strip()
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
                    turn.reply_text = "".join(collected).strip()
                style, promised = self._enqueue(turn, chunker.feed(text), style, queue, guard=allow_tools)
                if promised:
                    queue.put_nowait(self._safety_net_call(turn))
                    return
            tail = label.flush()
            collected.append(stripper.feed(tail) + stripper.flush())
            turn.reply_text = "".join(collected).strip()
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

    async def _play(self, turn: _Turn, sentence: Sentence, *, record_reply: bool = True) -> None:
        turn.spoken = True
        if record_reply and not turn.reply_text:
            turn.reply_text = sentence.text
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
            turn.played.append(sentence.text)
            return
        await self._set_state("speaking", out=turn.out)
        if turn.metrics.e2e_first_audio_ms is None:
            turn.metrics.e2e_first_audio_ms = _ms_since(turn.t0)
        await turn.out.send_json({"type": "audio_start", "id": utterance_id, "sample_rate": engine.sample_rate})
        # Playback starts when the first chunk reaches the device, so the end is counted from before the
        # send: under TCP backpressure send_bytes can take seconds, and measuring after it would add that
        # time to the playback window (a ghost "still speaking" period).
        send_started = time.perf_counter()
        await turn.out.send_bytes(float_to_pcm16(audio))
        await turn.out.send_json({"type": "audio_end", "id": utterance_id})
        turn.played.append(sentence.text)
        duration = audio.size / engine.sample_rate
        if turn.spec is None or turn.spec.committed.is_set():
            self._playback_end = max(send_started, self._playback_end) + duration
            self._playback_until = self._playback_end + ECHO_TAIL_S
        else:
            turn.spec.audio_s += duration

    async def on_timer(self, timer: dict[str, Any], note: str | None = None, *, speak: bool = True) -> None:
        """WS event to client plus spoken Polish announcement for sessions with audio."""
        timer_id = timer.get("id", "")
        label = timer.get("label", "")
        kind = timer.get("kind", "timer")
        try:
            await self._out.send_json({"type": "timer", "id": timer_id, "label": label, "kind": kind})
        except (ConnectionError, WebSocketDisconnect):
            return  # the client is gone; the session is being torn down
        if speak and self._hello is not None:
            spoken_text = format_timer_announcement(timer, note, lang=self._hello.lang)
            await self._announce(spoken_text)

    async def announce(self, text: str) -> None:
        """Speak ``text`` to this client unprompted (a speaker test from the admin API)."""
        await self._announce(text)

    async def _announce(self, text: str) -> None:
        async with self._result_lock:
            await self._wait_exclusive_turn()
            task = asyncio.create_task(self._announcement_turn(text))
            self._active = task
            try:
                await asyncio.wait({task})
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                if self._active is task:
                    self._active = None

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
        except (ConnectionError, WebSocketDisconnect):
            self._gone = True
            log.info("client left during an announcement; ending it quietly")
        finally:
            try:
                if not self._gone:
                    await self._send_metrics(turn.metrics)
                if self._active is asyncio.current_task():
                    self._active = None
                    if not self._gone:
                        await self._set_state("listening")
                        if self._device is not None:
                            await self._device.turn_done()
            except (ConnectionError, WebSocketDisconnect):
                self._gone = True
            self._prime_next_turn()

    async def play_sound(self, pcm: bytes, sample_rate: int, *, name: str = "") -> None:
        """Play raw mono PCM16 audio at ``sample_rate`` to this client unprompted."""
        await self._play_sound(pcm, sample_rate, name=name)

    async def _play_sound(self, pcm: bytes, sample_rate: int, *, name: str = "") -> None:
        async with self._result_lock:
            await self._wait_exclusive_turn()
            task = asyncio.create_task(self._sound_turn(pcm, sample_rate, name=name))
            self._active = task
            try:
                await asyncio.wait({task})
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                if self._active is task:
                    self._active = None

    async def _sound_turn(self, pcm: bytes, sample_rate: int, *, name: str = "") -> None:
        self._turns += 1
        turn = _Turn(
            t0=time.perf_counter(),
            language="pl",
            metrics=TurnMetrics(turn=self._turns, kind="notification", route="sound"),
            out=self._out,
        )
        try:
            if not pcm:
                return
            turn.spoken = True
            self._utterances += 1
            utterance_id = self._utterances
            await self._set_state("speaking", out=turn.out)
            turn.metrics.e2e_first_audio_ms = _ms_since(turn.t0)
            await turn.out.send_json({"type": "audio_start", "id": utterance_id, "sample_rate": sample_rate})
            send_started = time.perf_counter()
            await turn.out.send_bytes(pcm)
            await turn.out.send_json({"type": "audio_end", "id": utterance_id})
            if name:
                turn.played.append(f"[{name}]")
            duration = (len(pcm) // 2) / sample_rate
            self._playback_end = max(send_started, self._playback_end) + duration
            self._playback_until = self._playback_end + ECHO_TAIL_S
        except (ConnectionError, WebSocketDisconnect):
            self._gone = True
            log.info("client left during sound playback; ending it quietly")
        finally:
            try:
                if not self._gone:
                    await self._send_metrics(turn.metrics)
                if self._active is asyncio.current_task():
                    self._active = None
                    if not self._gone:
                        await self._set_state("listening")
                        if self._device is not None:
                            await self._device.turn_done()
            except (ConnectionError, WebSocketDisconnect):
                self._gone = True
            self._prime_next_turn()

    async def _say(self, turn: _Turn, text: str, *, record_reply: bool = True) -> None:
        sentences, _ = self._sentences(turn, [text], Style())
        if record_reply and not turn.reply_text:
            turn.reply_text = " ".join(s.text for s in sentences).strip() or text.strip()
        for sentence in sentences or [Sentence(text, Style())]:
            if turn.metrics.first_sentence_ms is None:
                turn.metrics.first_sentence_ms = _ms_since(turn.t0)
            await self._play(turn, sentence, record_reply=record_reply)

    # ---- Hermes ---------------------------------------------------------------------------

    def _start_handoff(self, turn: _Turn, call: ToolCall) -> None:
        task = str(call.arguments.get("task", "")).strip() or turn.user_text
        reason = str(call.arguments.get("reason", "")).strip()
        job = _HermesJob(
            id=uuid.uuid4().hex[:8], task=task, language=turn.language,
            identity=turn.identity, voice=turn.voice, utterance=turn.user_text,
        )
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
            profile = await self._choose_profile(job)
            job.profile = profile.label if profile is not None else None
            await self._handoff_event(job, "started", reason)
            events = self._svc.hermes.submit(
                job.task,
                context=context,
                session_id=self.id,
                language=job.language,
                profile=profile.id if profile is not None else "default",
                identity=job.identity,
                instructions=self._svc.hermes_instructions(job.identity),
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

    async def _choose_profile(self, job: _HermesJob) -> Profile | None:
        """The Hermes profile for this task: the person's default, unless they name another one or the
        router is confident the task belongs to another of their profiles. None = no household set up."""
        household = self._svc.household
        if household is None:
            return None
        usable = household.usable_profiles(job.identity, voice=job.voice)
        if len(usable) <= 1:
            return usable[0] if usable else household.data.profile(household.data.general_profile)
        named = explicit_profile(job.utterance or job.task, usable)
        if named is not None:
            return named
        router = self._svc.router
        if router is None:
            return usable[0]
        try:
            index, probability = await router.pick_profile(job.task, [(p.label, p.description) for p in usable])
        except LLMUnavailable as exc:
            log.warning("profile pick skipped: %s", exc)
            return usable[0]
        if probability >= self._svc.cfg.household.profile_threshold:
            return usable[index]
        return usable[0]

    async def _emit_progress(self, job: _HermesJob, items: list[ProgressItem]) -> None:
        for item in items:
            await self._handoff_event(job, "progress", item.text)

    async def _hermes_failed(self, job: _HermesJob, detail: str) -> None:
        log.warning("Hermes handoff failed: %s", detail)
        await self._handoff_event(job, "error", detail)
        await self._speak_result(HERMES_DOWN[job.language], job.language, direct=True)

    async def _handoff_event(self, job: _HermesJob, state: str, text: str) -> None:
        event: dict[str, Any] = {"type": "handoff", "state": state, "id": job.id, "task": job.task, "text": text}
        if job.profile is not None:
            event["profile"] = job.profile
        await self._out.send_json(event)

    async def _wait_exclusive_turn(self) -> None:
        while True:
            active = self._active
            if active is not None and not active.done():
                await asyncio.wait({active})
                continue
            device = self._device
            if device is not None and device.recording_active:
                # A wake-word take is recording through the device mic: the firmware would end it on the
                # first audio_start. Announcements, timers and Hermes results wait for it (v1.10).
                assert device.recording is not None
                try:
                    await asyncio.wait_for(device.recording.ended.wait(), 5)
                except TimeoutError:
                    pass
                continue
            if self._spec is not None or (self._endpointer is not None and self._endpointer.speaking):
                await asyncio.sleep(0.1)
                continue
            quiet_remaining = self._last_interrupt + RESULT_QUIET_S - time.perf_counter()
            if quiet_remaining > 0:
                await asyncio.sleep(min(quiet_remaining, 0.05))
                continue
            break

    async def _speak_result(self, result: str, language: str, *, direct: bool = False) -> None:
        """Announce a Hermes result once Gaja is not talking and the user is not speaking."""
        async with self._result_lock:
            await self._wait_exclusive_turn()
            task = asyncio.create_task(self._result_turn(result, language, direct=direct))
            self._active = task
            try:
                await asyncio.wait({task})
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                if self._active is task:
                    self._active = None

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
        except (ConnectionError, WebSocketDisconnect):
            self._gone = True
            log.info("client left during a result; ending it quietly")
        finally:
            try:
                if not self._gone:
                    await self._send_metrics(turn.metrics)
                if self._active is asyncio.current_task():
                    self._active = None
                    if not self._gone:
                        await self._set_state("listening")
                        if self._device is not None:
                            await self._device.turn_done()
            except (ConnectionError, WebSocketDisconnect):
                self._gone = True
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
