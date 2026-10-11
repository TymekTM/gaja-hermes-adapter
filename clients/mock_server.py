#!/usr/bin/env python3
"""
Mock server for the Gaja voice WebSocket protocol (docs/ARCHITECTURE.md).

Serves web/ at GET / and the protocol at WS /ws on one port. Used to develop and test the web UI and
the CLI client without the real adapter (STT/LLM/TTS) running.

Audio is real Polish speech from bench/out/tts/piper/pl_*.wav (22050 Hz), sent as 20 ms PCM16 frames
(`?frames=whole` sends one frame per utterance, like the real server). With `tts=paradee` in the
hello message the audio is resampled to 24000 Hz.

Scripted scenarios drive every UI state. Pick one with `?demo=<name>` in the page URL (the web client
forwards the query string to /ws) or with env GAJA_MOCK_DEMO:

  (none)      reactive: typed text -> transcript echo, thinking, 2-3 sentences with emotions, metrics.
              Keywords: "przerwij"/"barge" = barge-in in the middle of an utterance,
              "sprawdź"/"pogoda"/"hermes" = Hermes hand-off (v1.7: ids, several at once, progress, done);
              "pogoda w warszawie" = local tool turn (`tool` event + spoken answer), "ile to"/"awaria" = other tools;
              "stt-degraded"/"stt-recovered" = the matching `notice` event.
  live        like (none), plus a simple energy VAD on the mic frames: loud audio -> vad speech_start,
              ~0.7 s of quiet -> speech_end, transcript, reply. Speaking over Gaja is a barge-in.
  user        vad speech_start at +0.8 s and nothing after (the "user is speaking" state)
  thinking    state thinking at +0.8 s and nothing after
  speak       three sentences in one emotion (&emotion=happy|excited|calm|sad|serious|neutral)
  emotions    one sentence per emotion in turn
  hermes      voice request -> hand-off that works for ~8 s (&hermes_s=N) -> result is spoken
  notice      a `notice{stt_degraded}` at +0.8 s and `stt_recovered` at +5 s
  barge       a long utterance paced in real time; 1.2 s in the "user" starts talking: vad speech_start,
              state listening (reason barge_in), stale frames and a stale audio_end are sent afterwards
  stale       frames before audio_start and after audio_end (must be dropped by the client)
  error       an `error` event at +1.2 s
  drop        closes the connection at +1.5 s and refuses new ones for &down=N seconds (default 6)

Other query options: frames=whole, pace=realtime, lang, tts, stale=1 (inject stale frames in any scenario),
engines=none (the `ready` event carries no engine list, like an older server), deltas=1 (send assistant_delta
like a pre-v1.7 server; the current server sends none), hermes_s=N, stt=degraded (start with the degraded-STT
notice).

Protocol v1.7 behaviour: Hermes tasks carry an `id`, run in the background and survive barge-in / `interrupt`;
`cancel_task{id?}` and `interrupt{scope:"all"}` cancel them (`handoff{state:"cancelled", text:""}`, no `done`);
`cancel_task` with nothing running answers `error: no running Hermes task`.

Engines and their sample rates: piper 22050, piper-darkman 22050, paradee 24000, supertonic 44100.
"""

import argparse
import asyncio
import json
import math
import mimetypes
import os
import struct
import time
import uuid
import wave
from http import HTTPStatus
from typing import Dict, List, Optional, Set
from urllib.parse import parse_qs, urlparse

import numpy as np
import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
WAV_DIR = os.path.join(ROOT, "bench", "out", "tts", "piper")

# (text, wav index, default emotion). Lengths vary on purpose: caption sizes follow the text length.
BANK = [
    ("Cześć! Miło Cię słyszeć.", 1, "happy"),
    ("Jasne, już to sprawdzam.", 2, "neutral"),
    ("To dobre pytanie. Pomyślmy o tym spokojnie, krok po kroku.", 3, "calm"),
    ("Niestety, tym razem się nie udało.", 4, "sad"),
    ("Uwaga, to ważna sprawa i musimy zrobić to dokładnie.", 5, "serious"),
    ("Wow, to naprawdę świetna wiadomość!", 6, "excited"),
    ("Dzisiaj w Krakowie będzie pochmurno, ale bez deszczu, a wieczorem zrobi się zupełnie przyjemnie.", 7, "neutral"),
    ("Zaraz wracam z odpowiedzią.", 8, "neutral"),
    ("Dobrze, zleciłam to Hermesowi. Daj mi chwilkę.", 9, "calm"),
    ("W porządku, możemy do tego wrócić, kiedy tylko zechcesz.", 10, "calm"),
    ("Mam wynik. Jutro w Krakowie osiemnaście stopni, słonecznie, bez wiatru.", 11, "happy"),
    ("To naprawdę długa wypowiedź, żeby zobaczyć, jak napisy radzą sobie z dłuższym tekstem, który zajmuje kilka linijek na małym ekranie telefonu.", 12, "neutral"),
]
ENGINE_RATES = {"piper": 22050, "piper-darkman": 22050, "paradee": 24000, "supertonic": 44100}
EMOTION_CYCLE = ["happy", "excited", "calm", "sad", "serious"]
EMOTION_STYLE = {
    "happy": {"rate": 1.05, "pitch_semitones": 1.0, "energy": 1.1},
    "excited": {"rate": 1.15, "pitch_semitones": 2.0, "energy": 1.3},
    "calm": {"rate": 0.92, "pitch_semitones": -1.0, "energy": 0.8},
    "sad": {"rate": 0.88, "pitch_semitones": -2.0, "energy": 0.7},
    "serious": {"rate": 0.95, "pitch_semitones": -1.0, "energy": 0.9},
}

_wav_cache: Dict[tuple, bytes] = {}


def synthetic_speech(sr: int, seconds: float = 2.0) -> bytes:
    """Speech-like harmonics (F0 130 Hz + two formants) with a syllabic envelope; fallback without wavs."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    env = 0.5 + 0.5 * np.sin(2 * math.pi * 3.2 * t) ** 2
    v = 0.4 * np.sin(2 * math.pi * 130 * t) + 0.3 * np.sin(2 * math.pi * 700 * t) + 0.2 * np.sin(2 * math.pi * 1200 * t)
    return (np.clip(v * env, -1, 1) * 18000).astype("<i2").tobytes()


def load_pcm(index: int, sample_rate: int, wav_path: Optional[str] = None) -> bytes:
    """PCM16 mono bytes of pl_pl{index:02d}.wav at `sample_rate` (linear resample if needed)."""
    key = (index, sample_rate, wav_path)
    if key in _wav_cache:
        return _wav_cache[key]
    path = wav_path or os.path.join(WAV_DIR, f"pl_pl{index:02d}.wav")
    pcm: Optional[np.ndarray] = None
    src_rate = sample_rate
    if os.path.isfile(path):
        try:
            with wave.open(path, "rb") as w:
                if w.getnchannels() == 1 and w.getsampwidth() == 2:
                    src_rate = w.getframerate()
                    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        except Exception as e:  # noqa: BLE001
            print(f"[mock-server] cannot read {path}: {e}; using synthetic speech")
    if pcm is None:
        data = synthetic_speech(sample_rate)
    elif src_rate == sample_rate:
        data = pcm.tobytes()
    else:
        n_out = int(len(pcm) * sample_rate / src_rate)
        x = np.interp(np.linspace(0, len(pcm) - 1, n_out), np.arange(len(pcm)), pcm.astype(np.float32))
        data = np.clip(x, -32768, 32767).astype("<i2").tobytes()
    _wav_cache[key] = data
    return data


def slice_frames(pcm: bytes, sample_rate: int, ms: float = 20.0) -> List[bytes]:
    step = int(sample_rate * ms / 1000) * 2
    return [pcm[i : i + step] for i in range(0, len(pcm), step)]


def load_or_generate_speech_chunks(wav_path: Optional[str] = None, sample_rate: int = 22050, chunk_duration_ms: float = 20.0):
    """Kept for backward compatibility with earlier tests/tools."""
    pcm = load_pcm(1, sample_rate, wav_path)
    return sample_rate, slice_frames(pcm, sample_rate, chunk_duration_ms)


class Session:
    def __init__(self, ws, params: Dict[str, str]):
        self.ws = ws
        self.params = params
        self.lang = params.get("lang", "pl")
        self.tts = params.get("tts") or "piper"
        self.sample_rate = ENGINE_RATES.get(self.tts, 22050)
        self.demo = params.get("demo", "")
        self.demo_started = False
        self.turn = 0
        self.utt_id = 0
        self.turn_task: Optional[asyncio.Task] = None
        self.bg: Set[asyncio.Task] = set()
        self.jobs: Dict[str, asyncio.Task] = {}  # running Hermes tasks by handoff id
        self.stt_degraded = False
        self.frames_in = 0
        self.speaking = False
        # simple VAD on mic frames
        self.vad_on = False
        self.quiet_ms = 0.0
        self.last_frame_at = 0.0
        self.vad_watch: Optional[asyncio.Task] = None

    @property
    def whole(self) -> bool:
        return self.params.get("frames") == "whole"

    @property
    def frame_sleep(self) -> float:
        return 0.02 if self.params.get("pace") == "realtime" else 0.002

    async def send(self, obj: dict):
        await self.ws.send(json.dumps(obj, ensure_ascii=False))

    def next_utt(self) -> int:
        self.utt_id += 1
        return self.utt_id


class MockServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8790,
        web_dir: str = "web",
        wav_path: Optional[str] = None,
        barge_in_mode: bool = False,
    ):
        self.host = host
        self.port = port
        self.web_dir = os.path.abspath(web_dir)
        self.wav_path = wav_path
        self.barge_in_mode = barge_in_mode
        self.active_connections: Set[object] = set()
        self.default_demo = os.environ.get("GAJA_MOCK_DEMO", "")
        self.refuse_until = 0.0
        self.frames_in_total = 0
        self.sessions_total = 0

        self.sample_rate = 22050
        _, self.speech_chunks = load_or_generate_speech_chunks(wav_path=wav_path, sample_rate=22050)

        mimetypes.init()
        mimetypes.add_type("application/javascript", ".js")
        mimetypes.add_type("text/css", ".css")
        mimetypes.add_type("text/html", ".html")
        mimetypes.add_type("font/woff2", ".woff2")
        mimetypes.add_type("image/svg+xml", ".svg")

    # ------------------------------------------------------------------ HTTP

    async def process_request(self, connection, request):
        clean_path = request.path.split("?")[0]
        if clean_path == "/ws":
            if time.monotonic() < self.refuse_until:
                body = b"unavailable\n"
                return Response(HTTPStatus.SERVICE_UNAVAILABLE, "Service Unavailable", Headers([("Content-Length", str(len(body)))]), body)
            return None

        if clean_path == "/reset":  # end a `drop` outage early
            self.refuse_until = 0.0
            body = b"ok\n"
            return Response(HTTPStatus.OK, "OK", Headers([("Content-Length", str(len(body)))]), body)

        if clean_path == "/health":
            body = json.dumps(
                {"status": "ok", "mock": True, "engines": list(ENGINE_RATES), "engine_by_language": {"pl": "piper", "en": "paradee"},
                 "stt_engine": "faster-whisper", "stt_device": "cuda", "stt_degraded": False,
                 "router": {"enabled": True, "threshold": 0.5, "tools": True},
                 "connections": len(self.active_connections), "audio_frames_in": self.frames_in_total, "sessions": self.sessions_total}
            ).encode() + b"\n"
            headers = Headers([("Content-Type", "application/json"), ("Content-Length", str(len(body))), ("Access-Control-Allow-Origin", "*")])
            return Response(HTTPStatus.OK, "OK", headers, body)

        rel_path = "index.html" if clean_path == "/" else clean_path.lstrip("/")
        file_path = os.path.normpath(os.path.join(self.web_dir, rel_path))
        if not file_path.startswith(self.web_dir) or not os.path.isfile(file_path):
            body = b"404 Not Found\n"
            return Response(HTTPStatus.NOT_FOUND, "Not Found", Headers([("Content-Type", "text/plain"), ("Content-Length", str(len(body)))]), body)

        content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        with open(file_path, "rb") as f:
            body = f.read()
        headers = Headers([("Content-Type", content_type), ("Content-Length", str(len(body))), ("Cache-Control", "no-store")])
        return Response(HTTPStatus.OK, "OK", headers, body)

    # ------------------------------------------------------------------ WebSocket

    async def handle_ws(self, ws):
        self.active_connections.add(ws)
        self.sessions_total += 1
        query = parse_qs(urlparse(ws.request.path).query) if getattr(ws, "request", None) else {}
        params = {k: v[0] for k, v in query.items()}
        if not params.get("demo") and self.default_demo:
            params["demo"] = self.default_demo
        sess = Session(ws, params)
        print(f"[mock-server] client connected demo={sess.demo or '-'} params={params}")

        try:
            async for message in ws:
                if isinstance(message, str):
                    await self.on_json(sess, message)
                else:
                    await self.on_audio(sess, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            for t in list(sess.bg) + [sess.turn_task, sess.vad_watch]:
                if t and not t.done():
                    t.cancel()
            self.active_connections.discard(ws)
            print("[mock-server] client disconnected")

    def spawn(self, sess: Session, coro) -> asyncio.Task:
        task = asyncio.create_task(self.guard(coro))
        sess.bg.add(task)
        task.add_done_callback(sess.bg.discard)
        return task

    @staticmethod
    async def guard(coro):
        try:
            await coro
        except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed):
            pass

    async def on_json(self, sess: Session, message: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            await sess.send({"type": "error", "message": "Invalid JSON"})
            return
        kind = data.get("type")

        if kind == "hello":
            tts = data.get("tts")
            if tts and tts not in ENGINE_RATES:
                await sess.send({"type": "error", "message": f"unknown tts engine {tts!r}"})
                return
            sess.lang = data.get("lang") or sess.lang
            if tts:
                sess.tts = tts
            elif sess.lang == "en":
                sess.tts = "paradee"
            elif not sess.params.get("tts"):
                sess.tts = "piper"
            sess.sample_rate = ENGINE_RATES.get(sess.tts, 22050)
            print(f"[mock-server] hello lang={sess.lang} tts={sess.tts} client={data.get('client')}")
            ready = {"type": "ready", "session_id": f"mock-{id(sess.ws):x}", "tts": sess.tts, "sample_rate": sess.sample_rate, "langs": ["pl", "en"]}
            if sess.params.get("engines") != "none":
                ready["engines"] = list(ENGINE_RATES)
            await sess.send(ready)
            if sess.params.get("stt") == "degraded":  # like the server: one notice per state change
                await self.set_stt(sess, True)
            if not sess.demo_started:  # a repeated hello (settings change) re-sends `ready` only
                sess.demo_started = True
                self.spawn(sess, self.run_demo(sess))
        elif kind == "ping":
            await sess.send({"type": "pong"})
        elif kind == "text":
            text = (data.get("text") or "").strip()
            if text.lower() in ("stt-degraded", "stt-recovered"):
                await self.set_stt(sess, text.lower() == "stt-degraded")
                return
            await self.start_turn(sess, text, via_voice=False)
        elif kind == "cancel_task":
            job_id = data.get("id")
            if job_id is not None and not isinstance(job_id, str):
                await sess.send({"type": "error", "message": "cancel_task id must be a string"})
            elif not await self.cancel_jobs(sess, job_id):
                await sess.send({"type": "error", "message": f"no running Hermes task {job_id!r}" if job_id else "no running Hermes task"})
        elif kind == "end_of_input":
            if sess.vad_on:
                await self.end_speech(sess)
            else:
                await self.start_turn(sess, "Wiadomość z pliku dźwiękowego", via_voice=True)
        elif kind == "interrupt":
            if data.get("scope") == "all":
                await self.cancel_jobs(sess, None)
            if sess.turn_task and not sess.turn_task.done():
                sess.turn_task.cancel()
                sess.speaking = False
                await sess.send({"type": "state", "state": "listening"})

    async def set_stt(self, sess: Session, degraded: bool):
        if degraded == sess.stt_degraded:
            return
        sess.stt_degraded = degraded
        code = "stt_degraded" if degraded else "stt_recovered"
        message = (
            "Rozpoznawanie mowy działa awaryjnie na CPU (wolniej); GPU będzie sprawdzane ponownie."
            if degraded
            else "Rozpoznawanie mowy znów działa na GPU."
        )
        await sess.send({"type": "notice", "code": code, "message": message, "engine": "whistle" if degraded else "faster-whisper"})

    async def cancel_jobs(self, sess: Session, job_id: Optional[str]) -> int:
        jobs = [(i, t) for i, t in sess.jobs.items() if job_id is None or i == job_id]
        for _, task in jobs:
            task.cancel()
        await asyncio.gather(*(t for _, t in jobs), return_exceptions=True)
        return len(jobs)

    # ------------------------------------------------------------------ mic frames + simple VAD

    async def on_audio(self, sess: Session, frame: bytes):
        sess.frames_in += 1
        self.frames_in_total += 1
        if sess.demo != "live" and sess.params.get("vad") != "1":
            return
        samples = np.frombuffer(frame[: len(frame) // 2 * 2], dtype="<i2").astype(np.float32)
        if samples.size == 0:
            return
        rms = float(np.sqrt(np.mean(samples**2)))
        frame_ms = samples.size / 16.0
        sess.last_frame_at = time.monotonic()
        if rms > 600:
            sess.quiet_ms = 0.0
            if not sess.vad_on:
                sess.vad_on = True
                if sess.turn_task and not sess.turn_task.done():  # barge-in
                    sess.turn_task.cancel()
                    sess.speaking = False
                    await sess.send({"type": "vad", "state": "speech_start"})
                    await sess.send({"type": "state", "state": "listening", "reason": "barge_in"})
                else:
                    await sess.send({"type": "vad", "state": "speech_start"})
                if not sess.vad_watch or sess.vad_watch.done():
                    sess.vad_watch = asyncio.create_task(self.vad_idle_watch(sess))
        elif sess.vad_on:
            sess.quiet_ms += frame_ms
            if sess.quiet_ms >= 700:
                await self.end_speech(sess)

    async def vad_idle_watch(self, sess: Session):
        """The mic stopped sending (PTT release / mic off): treat that as the end of speech."""
        try:
            while sess.vad_on:
                await asyncio.sleep(0.2)
                if sess.vad_on and time.monotonic() - sess.last_frame_at > 0.9:
                    await self.end_speech(sess)
        except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed):
            pass

    async def end_speech(self, sess: Session):
        if not sess.vad_on:
            return
        sess.vad_on = False
        sess.quiet_ms = 0.0
        await sess.send({"type": "vad", "state": "speech_end"})
        await self.start_turn(sess, "Słyszę Cię bardzo wyraźnie", via_voice=True, announce_vad=False)

    # ------------------------------------------------------------------ turns

    async def start_turn(self, sess: Session, text: str, via_voice: bool, announce_vad: bool = True):
        if sess.turn_task and not sess.turn_task.done():
            sess.turn_task.cancel()
            await asyncio.sleep(0)
        sess.turn += 1
        sess.turn_task = asyncio.create_task(self.guard(self.run_turn(sess, text, via_voice, sess.turn)))

    async def stream_text(self, sess: Session, text: str):
        """assistant_delta (pre-v1.7 servers only, `?deltas=1`): the raw LLM text, a few words at a time."""
        if sess.params.get("deltas") != "1":
            return
        words = text.split(" ")
        for i in range(0, len(words), 2):
            chunk = " ".join(words[i : i + 2]) + " "
            await sess.send({"type": "assistant_delta", "text": chunk})
            await asyncio.sleep(0.03)

    async def speak(self, sess: Session, text: str, wav_index: int, emotion: str, stream: bool = True):
        """One sentence: assistant_delta (streamed text), utterance, audio_start, PCM frames, audio_end."""
        if stream:
            await self.stream_text(sess, text)
        uid = sess.next_utt()
        style = {"emotion": emotion, "rate": 1.0, "pitch_semitones": 0.0, "energy": 1.0, "events": []}
        style.update(EMOTION_STYLE.get(emotion, {}))
        pcm = load_pcm(wav_index, sess.sample_rate, self.wav_path)
        frames = [pcm] if sess.whole else slice_frames(pcm, sess.sample_rate)
        inject = sess.params.get("stale") == "1" or sess.demo == "stale"

        await sess.send({"type": "utterance", "id": uid, "text": text, "style": style})
        if inject:  # frames with no open audio_start: the client has to drop them
            for fr in frames[:3]:
                await sess.ws.send(fr)
        await sess.send({"type": "state", "state": "speaking"})
        sess.speaking = True
        await sess.send({"type": "audio_start", "id": uid, "sample_rate": sess.sample_rate})
        for fr in frames:
            await sess.ws.send(fr)
            await asyncio.sleep(sess.frame_sleep)
        await sess.send({"type": "audio_end", "id": uid})
        if inject:  # ... and after audio_end
            for fr in frames[:3]:
                await sess.ws.send(fr)
        return uid

    async def metrics(self, sess: Session, via_voice: bool):
        n = sess.turn
        jitter = (n * 37) % 90
        await sess.send(
            {
                "type": "metrics",
                "turn": n,
                "stt_ms": (190 + jitter) if via_voice else 0.0,
                "llm_ttft_ms": 120 + jitter // 2,
                "first_sentence_ms": 260 + jitter,
                "tts_first_ms": 140 + jitter // 3,
                "e2e_first_audio_ms": 640 + jitter * 2,
            }
        )

    async def run_turn(self, sess: Session, text: str, via_voice: bool, turn: int):
        lower = text.lower()
        if via_voice or text:
            await sess.send({"type": "transcript", "text": text, "lang": sess.lang, "final": True})
        await sess.send({"type": "state", "state": "thinking"})
        await asyncio.sleep(0.25)

        tool = self.pick_tool(lower)
        if tool is not None:
            await self.tool_turn(sess, tool, via_voice)
            return

        if any(k in lower for k in ("sprawdź", "sprawdz", "pogoda", "hermes")):
            await self.hermes_flow(sess, ack=True, task=text if "hermes" in lower else None)
            return

        barge = self.barge_in_mode or any(k in lower for k in ("barge", "przerwij", "interrupt"))

        if barge:
            uid = sess.next_utt()
            text_long = BANK[6][0]
            await self.stream_text(sess, text_long)
            await sess.send({"type": "utterance", "id": uid, "text": text_long, "style": {"emotion": "neutral"}})
            await sess.send({"type": "state", "state": "speaking"})
            await sess.send({"type": "audio_start", "id": uid, "sample_rate": sess.sample_rate})
            frames = slice_frames(load_pcm(7, sess.sample_rate, self.wav_path), sess.sample_rate)
            for fr in frames[:30]:
                await sess.ws.send(fr)
                await asyncio.sleep(0.004)
            # the user starts talking mid-utterance: vad first, then the server's cancel, then in-flight frames
            await asyncio.sleep(0.5)
            await sess.send({"type": "vad", "state": "speech_start"})
            await asyncio.sleep(0.05)
            await sess.send({"type": "state", "state": "listening", "reason": "barge_in"})
            for fr in frames[30:40]:
                await sess.ws.send(fr)
            await sess.send({"type": "audio_end", "id": uid})
            sess.speaking = False
            return

        picks = [BANK[(turn * 2) % len(BANK)], BANK[(turn * 2 + 1) % len(BANK)]]
        picks = [p for p in picks if p[1] != 12 or turn % 3 == 0]
        for i, (sentence, wav_idx, emo) in enumerate(picks):
            await self.speak(sess, sentence, wav_idx, emo)
        sess.speaking = False
        await sess.send({"type": "state", "state": "listening"})
        await self.metrics(sess, via_voice)

    @staticmethod
    def pick_tool(lower: str) -> Optional[dict]:
        if "pogod" in lower and "warszaw" in lower:
            return {"name": "get_weather", "arguments": {"location": "Warszawa"}, "ok": True, "text": "W Warszawie jest dziś dwanaście stopni, pochmurno.", "wav": 7}
        if "ile to" in lower:
            return {"name": "calculate", "arguments": {"expression": "12 * 7"}, "ok": True, "text": "Dwanaście razy siedem to osiemdziesiąt cztery.", "wav": 2}
        if "awaria" in lower:
            return {"name": "exchange_rate", "arguments": {"currency": "EUR", "to": "PLN"}, "ok": False, "text": "Nie udało się sprawdzić kursu, serwis nie odpowiada.", "wav": 4}
        return None

    async def tool_turn(self, sess: Session, tool: dict, via_voice: bool):
        """Router -> TOOL: the `tool` event, then the tool's text spoken directly (no second LLM pass)."""
        await sess.send({"type": "tool", "name": tool["name"], "arguments": tool["arguments"], "ok": tool["ok"], "text": tool["text"]})
        await self.speak(sess, tool["text"], tool["wav"], "neutral", stream=False)
        sess.speaking = False
        await sess.send({"type": "state", "state": "listening"})
        await self.metrics(sess, via_voice)

    async def hermes_flow(self, sess: Session, ack: bool, seconds: Optional[float] = None, task: Optional[str] = None):
        seconds = seconds if seconds is not None else float(sess.params.get("hermes_s", 8))
        task = task or "Prognoza pogody dla Krakowa na jutro"
        job_id = uuid.uuid4().hex[:8]
        job = asyncio.create_task(self.guard(self.hermes_job(sess, job_id, task, seconds)))
        sess.jobs[job_id] = job
        sess.bg.add(job)
        job.add_done_callback(sess.bg.discard)
        if ack:
            await self.speak(sess, BANK[8][0], 3, "calm", stream=False)  # short clip: the text is what the UI shows
        sess.speaking = False
        await sess.send({"type": "state", "state": "listening"})
        await self.metrics(sess, True)

    async def hermes_job(self, sess: Session, job_id: str, task: str, seconds: float):
        """A background Hermes task (v1.7): started, progress, done - or `cancelled` with no `done`."""

        async def event(state: str, text: str):
            await sess.send({"type": "handoff", "state": state, "id": job_id, "task": task, "text": text})

        try:
            await event("started", "router")
            t0 = time.monotonic()
            marks = [(seconds * 0.35, "Hermes uruchamia narzędzie: web_search"), (seconds * 0.7, "Porównuję wyniki.")]
            for at, note in marks:
                await asyncio.sleep(max(0.0, at - (time.monotonic() - t0)))
                await event("progress", note)
            await asyncio.sleep(max(0.0, seconds - (time.monotonic() - t0)))
            sess.jobs.pop(job_id, None)
            await event("done", "Jutro w Krakowie 18°C, słonecznie, wiatr słaby.")
            await sess.send({"type": "state", "state": "thinking"})
            await asyncio.sleep(0.3)
            await self.speak(sess, BANK[10][0], 5, "happy", stream=False)
            sess.speaking = False
            await sess.send({"type": "state", "state": "listening"})
            # the announcement of a finished task is a `result` turn; its latency is not a reply latency
            await sess.send({"type": "metrics", "turn": sess.turn + 1, "kind": "result", "route": "hermes", "e2e_first_audio_ms": 9999.0, "first_sentence_ms": 9999.0})
        except asyncio.CancelledError:
            if sess.jobs.pop(job_id, None) is not None:  # cancelled while still running
                try:
                    await asyncio.shield(event("cancelled", ""))
                except Exception:  # noqa: BLE001 - the socket may be gone
                    pass
            raise
        finally:
            sess.jobs.pop(job_id, None)

    # ------------------------------------------------------------------ scripted scenarios

    async def run_demo(self, sess: Session):
        demo = sess.demo
        if demo in ("", "live"):
            return
        await asyncio.sleep(0.8)

        if demo == "user":
            await sess.send({"type": "vad", "state": "speech_start"})
        elif demo == "thinking":
            await sess.send({"type": "transcript", "text": "Opowiedz mi coś ciekawego o kosmosie", "lang": "pl", "final": True})
            await sess.send({"type": "state", "state": "thinking"})
        elif demo == "speak":
            emotion = sess.params.get("emotion", "neutral")
            await sess.send({"type": "transcript", "text": "Jak się dzisiaj czujesz?", "lang": "pl", "final": True})
            await sess.send({"type": "state", "state": "thinking"})
            await asyncio.sleep(0.3)
            for sentence, idx, _ in (BANK[0], BANK[2], BANK[6]):
                await self.speak(sess, sentence, idx, emotion)
            await sess.send({"type": "state", "state": "listening"})
            await self.metrics(sess, True)
        elif demo == "emotions":
            await sess.send({"type": "state", "state": "thinking"})
            for i, emotion in enumerate(EMOTION_CYCLE):
                sentence, idx, _ = BANK[i]
                await self.speak(sess, sentence, idx, emotion)
                await asyncio.sleep(0.4)
            await sess.send({"type": "state", "state": "listening"})
            await self.metrics(sess, False)
        elif demo == "hermes":
            await sess.send({"type": "transcript", "text": "Sprawdź mi proszę, jaka jutro będzie pogoda w Krakowie.", "lang": "pl", "final": True})
            await sess.send({"type": "state", "state": "thinking"})
            await asyncio.sleep(0.3)
            await self.hermes_flow(sess, ack=True)
        elif demo == "notice":
            await self.set_stt(sess, True)
            await asyncio.sleep(4.2)
            await self.set_stt(sess, False)
        elif demo == "barge":
            await self.start_turn(sess, "przerwij mnie", via_voice=False)
            await asyncio.sleep(0.2)
        elif demo == "stale":
            await sess.send({"type": "state", "state": "thinking"})
            await self.speak(sess, BANK[1][0], BANK[1][1], "neutral")
            await sess.send({"type": "state", "state": "listening"})
        elif demo == "error":
            await asyncio.sleep(0.4)
            await sess.send({"type": "error", "message": "Silnik głosu nie odpowiada. Spróbuję jeszcze raz za chwilę."})
        elif demo == "drop":
            await asyncio.sleep(0.7)
            down = float(sess.params.get("down", 6))
            self.refuse_until = time.monotonic() + down
            await sess.ws.close(1011, "mock outage")

async def run_server(host: str, port: int, web_dir: str, wav_path: Optional[str] = None, barge_in: bool = False):
    mock = MockServer(host=host, port=port, web_dir=web_dir, wav_path=wav_path, barge_in_mode=barge_in)
    print(f"[mock-server] http://{host}:{port}/  ws://{host}:{port}/ws  (try /?demo=speak&emotion=happy)")
    async with websockets.serve(mock.handle_ws, host=host, port=port, process_request=mock.process_request, max_size=None):
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description="Mock server for the Gaja voice WebSocket protocol")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--web-dir", default=os.path.join(ROOT, "web"))
    parser.add_argument("--wav", default=None, help="speech .wav (mono PCM16) used for every sentence")
    parser.add_argument("--barge-in", action="store_true", help="always barge in mid-utterance on typed text")
    args = parser.parse_args()
    try:
        asyncio.run(run_server(args.host, args.port, args.web_dir, wav_path=args.wav, barge_in=args.barge_in))
    except KeyboardInterrupt:
        print("\n[mock-server] shutting down")


if __name__ == "__main__":
    main()
