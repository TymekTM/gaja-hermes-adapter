"""Wake-word sample collection ("Hej Gaja"): upload / progress / undo / export endpoints.

Backs the static recorder page `web/record.html`. Layout on disk (all path components are `[a-z0-9_-]` only):

    <recordings_dir>/<session>/<label>/<prompt_id>/<take:03d>.wav
    <recordings_dir>/<session>/meta.jsonl      one JSON line per upload (and per undo, with "deleted": true)

Audio is stored exactly as uploaded after validation: RIFF/WAVE, PCM16, mono, 16 kHz (the browser encodes it).
Takes recorded through an IoT device's microphone (protocol v1.10, `gaja/devices/recording.py`) land in the
same tree through the same `SampleStore`, with extra meta (`source: "device"`, marks, ...).
Every endpoint refuses a request whose `Origin` header is present and not in `server.allowed_origins`
(the same allowlist as `/ws`); clients without an Origin header (curl, scripts) are allowed, as for the WebSocket.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
import time
import wave
import zipfile
from collections.abc import Callable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

log = logging.getLogger(__name__)

LABELS = ("wake", "hard_negative", "ambient")
SAMPLE_RATE = 16000
WAV_HEADER_BYTES = 44
BYTES_PER_SECOND = SAMPLE_RATE * 2
MAX_BYTES = 4 * 1024 * 1024  # 130 s of 16 kHz PCM16; an ambient chunk is <= 60 s (~1.9 MB)
MIN_DURATION_S = 0.2
MAX_TAKE = 9999
MAX_FIELD_CHARS = 200
_NAME = re.compile(r"^[a-z0-9_-]{1,64}$")
_WAV_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave", "application/octet-stream"}


def _name(value: str, what: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise HTTPException(400, f"invalid {what}: only [a-z0-9_-], 1-64 characters")
    return value


def _label(value: str) -> str:
    if value not in LABELS:
        raise HTTPException(400, f"invalid label: one of {', '.join(LABELS)}")
    return value


def _take(value: int) -> int:
    if not 0 <= value <= MAX_TAKE:
        raise HTTPException(400, f"invalid take: 0..{MAX_TAKE}")
    return value


def _clean(text: str | None) -> str:
    """Free text from the client (mic label, user agent) for the metadata file."""
    return "".join(ch for ch in (text or "") if ch.isprintable())[:MAX_FIELD_CHARS]


def _inspect_wav(data: bytes) -> tuple[float, float, float]:
    """Validate the upload and return (duration_s, peak, rms) with peak/rms as fractions of full scale."""
    if len(data) < WAV_HEADER_BYTES or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise HTTPException(415, "not a RIFF/WAVE file")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
            comp, frames = wav.getcomptype(), wav.getnframes()
            pcm = wav.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise HTTPException(415, f"unreadable WAV: {exc}") from exc
    if comp != "NONE" or width != 2 or channels != 1 or rate != SAMPLE_RATE:
        raise HTTPException(415, f"WAV must be PCM16 mono {SAMPLE_RATE} Hz (got {channels} ch, {8 * width} bit, {rate} Hz)")
    samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    if samples.size < MIN_DURATION_S * SAMPLE_RATE:
        raise HTTPException(422, f"recording shorter than {MIN_DURATION_S} s")
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples * samples)))
    return samples.size / SAMPLE_RATE, peak, rms


class _ZipSink:
    """Write-only, non-seekable file object: ZipFile writes here and the generator drains it."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.pos = 0

    def write(self, data: bytes) -> int:
        self.buf += data
        self.pos += len(data)
        return len(data)

    def tell(self) -> int:
        return self.pos

    def flush(self) -> None:
        pass

    def drain(self) -> bytes:
        out = bytes(self.buf)
        self.buf.clear()
        return out


class SampleStore:
    """The recordings tree on disk, shared by the browser upload API and device takes.

    One instance per directory (see ``get_store``): a single lock serialises writes and a single byte
    cache tracks the ``max_total_mb`` budget across both sources.
    """

    def __init__(self, root: Path, max_total_mb: int = 8192) -> None:
        self.root = Path(root)
        self.max_total_mb = max_total_mb
        self.lock = threading.Lock()
        self._cached_bytes: int | None = None

    # ---- tree ------------------------------------------------------------------------------

    def session_dir(self, session: str) -> Path:
        return self.root / _name(session, "session")

    def wav_files(self, base: Path) -> Iterator[Path]:
        return iter(sorted(base.glob("*/*/*.wav"))) if base.is_dir() else iter(())

    def scan(self, session: str) -> dict[str, dict[str, dict[str, Any]]]:
        labels: dict[str, dict[str, dict[str, Any]]] = {}
        for path in self.wav_files(self.session_dir(session)):
            label, prompt_id = path.parent.parent.name, path.parent.name
            if label not in LABELS or not path.stem.isdigit():
                continue
            entry = labels.setdefault(label, {}).setdefault(prompt_id, {"count": 0, "duration_s": 0.0, "next_take": 0})
            entry["count"] += 1
            entry["duration_s"] += max(0, path.stat().st_size - WAV_HEADER_BYTES) / BYTES_PER_SECOND
            entry["next_take"] = max(entry["next_take"], int(path.stem) + 1)
        for prompts in labels.values():
            for entry in prompts.values():
                entry["duration_s"] = round(entry["duration_s"], 2)
        return labels

    def next_take(self, session: str, label: str, prompt_id: str) -> int:
        """The next free take number for a prompt, allocated like the browser upload does."""
        with self.lock:
            return self.scan(session).get(label, {}).get(prompt_id, {"next_take": 0})["next_take"]

    def total_bytes(self) -> int:
        if self._cached_bytes is None:
            self._cached_bytes = sum(p.stat().st_size for p in self.root.rglob("*.wav")) if self.root.is_dir() else 0
        return self._cached_bytes

    def append_meta(self, session: str, record: dict[str, Any]) -> None:
        with (self.session_dir(session) / "meta.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- writes ----------------------------------------------------------------------------

    def write(
        self,
        session: str,
        label: str,
        prompt_id: str,
        take: int,
        data: bytes,
        *,
        extra: dict[str, Any] | None = None,
        mic: str | None = None,
        user_agent: str | None = None,
        bump: bool = False,
    ) -> dict[str, Any]:
        """Validate a PCM16 mono 16 kHz WAV and store it as a take; returns the upload-style response.

        ``bump`` (device takes): when the allocated take number was taken in the meantime, use the next
        free one instead of failing.
        """
        _name(session, "session"), _label(label), _name(prompt_id, "prompt_id"), _take(take)
        duration_s, peak, rms = _inspect_wav(data)
        with self.lock:
            if self.total_bytes() + len(data) > self.max_total_mb * 1024 * 1024:
                raise HTTPException(507, f"recordings directory is over {self.max_total_mb} MB")
            target = self.session_dir(session) / label / prompt_id / f"{take:03d}.wav"
            if target.exists():
                if not bump:
                    raise HTTPException(409, "take already exists")
                while target.exists():
                    take += 1
                    target = self.session_dir(session) / label / prompt_id / f"{take:03d}.wav"
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, target)
            self._cached_bytes = self.total_bytes() + len(data)
            self.append_meta(
                session,
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "session": session,
                    "label": label,
                    "prompt_id": prompt_id,
                    "take": take,
                    "file": str(target.relative_to(self.session_dir(session))),
                    "bytes": len(data),
                    "duration_s": round(duration_s, 3),
                    "peak": round(peak, 4),
                    "rms": round(rms, 5),
                    "user_agent": _clean(user_agent),
                    "mic": _clean(mic),
                    **(extra or {}),
                },
            )
            labels = self.scan(session)
        prompt = labels[label][prompt_id]
        return {
            "ok": True,
            "session": session,
            "label": label,
            "prompt_id": prompt_id,
            "take": take,
            "duration_s": round(duration_s, 3),
            "peak": round(peak, 4),
            "rms": round(rms, 5),
            "count": prompt["count"],
            "next_take": prompt["next_take"],
            "total": sum(p["count"] for prompts in labels.values() for p in prompts.values()),
        }

    def delete(self, session: str, label: str, prompt_id: str, take: int) -> dict[str, Any]:
        _name(session, "session"), _label(label), _name(prompt_id, "prompt_id"), _take(take)
        target = self.session_dir(session) / label / prompt_id / f"{take:03d}.wav"
        with self.lock:
            if not target.is_file():
                raise HTTPException(404, "no such take")
            size = target.stat().st_size
            target.unlink()
            self._cached_bytes = max(0, self.total_bytes() - size)
            self.append_meta(
                session,
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "session": session,
                    "label": label,
                    "prompt_id": prompt_id,
                    "take": take,
                    "deleted": True,
                },
            )
            labels = self.scan(session)
        prompt = labels.get(label, {}).get(prompt_id, {"count": 0, "next_take": 0})
        return {"ok": True, "count": prompt["count"], "next_take": prompt["next_take"]}

    # ---- summaries -------------------------------------------------------------------------

    def sessions_summary(self) -> list[dict[str, Any]]:
        out = []
        if self.root.is_dir():
            for d in sorted(p for p in self.root.iterdir() if p.is_dir() and _NAME.fullmatch(p.name)):
                labels = self.scan(d.name)
                counts = {label: sum(p["count"] for p in prompts.values()) for label, prompts in labels.items()}
                seconds = sum(p["duration_s"] for prompts in labels.values() for p in prompts.values())
                out.append(
                    {
                        "session": d.name,
                        "counts": counts,
                        "total": sum(counts.values()),
                        "duration_s": round(seconds, 1),
                        "modified": int(d.stat().st_mtime),
                    }
                )
        out.sort(key=lambda s: s["modified"], reverse=True)
        return out

    def device_stats(self, session: str) -> dict[str, Any]:
        """On-device recall over real device recordings: device-detected wake takes / device wake takes.

        Replays ``meta.jsonl`` (uploads, device takes and undos, in order) and counts the takes that are
        still on disk with ``source: "device"``.
        """
        alive: set[tuple[str, str, int]] = set()
        path = self.session_dir(session) / "meta.jsonl"
        detected = 0
        marks_total = 0
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            take = rec.get("take", -1)
            if isinstance(take, bool) or not isinstance(take, int):
                continue  # malformed line: never let one bad meta entry break the progress endpoint
            key = (str(rec.get("label", "")), str(rec.get("prompt_id", "")), take)
            if rec.get("deleted"):
                alive.discard(key)
            elif rec.get("source") == "device":
                alive.add(key)
                if key[0] == "wake":
                    marks = rec.get("marks") or []
                    marks_total += len(marks) if isinstance(marks, list) else 0
                    if rec.get("device_detected"):
                        detected += 1
        takes = sum(1 for label, _, _ in alive if label == "wake")
        return {
            "wake": {
                "takes": takes,
                "detected": detected,
                "marks": marks_total,
                **({"recall": round(detected / takes, 3)} if takes else {}),
            }
        }


@lru_cache(maxsize=8)
def get_store(recordings_dir: Path, max_total_mb: int = 8192) -> SampleStore:
    """The process-wide store for a directory: the upload router and the device-take API share one lock."""
    return SampleStore(recordings_dir, max_total_mb)


def create_router(recordings_dir: Path, origin_ok: Callable[[str | None], bool], max_total_mb: int = 8192) -> APIRouter:
    store = get_store(Path(recordings_dir), max_total_mb)
    root = store.root

    def check_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        if not origin_ok(origin):
            log.warning("rejected wake-word request from origin %r", origin)
            raise HTTPException(403, "origin not allowed")

    router = APIRouter(prefix="/api/wakeword", dependencies=[Depends(check_origin)])

    # ---- endpoints ---------------------------------------------------------------------------

    @router.post("/samples")
    async def upload(
        request: Request,
        session: str,
        label: str,
        prompt_id: str,
        take: int,
        mic: str | None = None,
    ) -> dict[str, Any]:
        _name(session, "session"), _label(label), _name(prompt_id, "prompt_id"), _take(take)
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype not in _WAV_TYPES:
            raise HTTPException(415, "Content-Type must be audio/wav (raw WAV body)")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BYTES:
            raise HTTPException(413, f"body larger than {MAX_BYTES} bytes")
        data = bytearray()
        async for part in request.stream():
            data += part
            if len(data) > MAX_BYTES:
                raise HTTPException(413, f"body larger than {MAX_BYTES} bytes")
        return store.write(
            session, label, prompt_id, take, bytes(data),
            mic=mic, user_agent=request.headers.get("user-agent"),
        )

    @router.delete("/samples/{session}/{label}/{prompt_id}/{take}")
    async def delete(session: str, label: str, prompt_id: str, take: int) -> dict[str, Any]:
        return store.delete(session, label, prompt_id, take)

    @router.get("/progress")
    async def progress(session: str) -> dict[str, Any]:
        labels = store.scan(session)
        return {
            "session": session,
            "total": sum(p["count"] for prompts in labels.values() for p in prompts.values()),
            "labels": labels,
            "device": store.device_stats(session),
        }

    @router.get("/sessions")
    async def sessions() -> dict[str, Any]:
        return {"sessions": store.sessions_summary()}

    @router.get("/export.zip")
    async def export(session: str | None = None) -> StreamingResponse:
        base = store.session_dir(session) if session else root
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"gaja-wakeword-{session or 'all'}-{stamp}.zip"

        def generate() -> Iterator[bytes]:
            sink = _ZipSink()
            with zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED) as zf:  # type: ignore[arg-type]
                files = sorted(p for p in base.rglob("*") if p.is_file() and p.suffix in (".wav", ".jsonl")) if base.is_dir() else []
                for path in files:
                    info = zipfile.ZipInfo.from_file(path, str(Path("wakeword") / path.relative_to(root)))
                    info.compress_type = zipfile.ZIP_STORED
                    with path.open("rb") as src, zf.open(info, "w", force_zip64=True) as dst:
                        while block := src.read(1024 * 1024):
                            dst.write(block)
                            if sink.buf:
                                yield sink.drain()
                    yield sink.drain()
            yield sink.drain()

        return StreamingResponse(
            generate(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{name}"', "Cache-Control": "no-store"},
        )

    return router
