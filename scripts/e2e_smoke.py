from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]
FRAME_SAMPLES = 512


def load_pcm16(path: Path) -> bytes:
    samples, rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != 16000:
        raise SystemExit(f"{path} must be 16 kHz, got {rate}")
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def paced_send(ws, pcm: bytes, realtime: bool) -> None:
    frame_bytes = FRAME_SAMPLES * 2
    frame_s = FRAME_SAMPLES / 16000
    next_at = time.perf_counter()
    for start in range(0, len(pcm), frame_bytes):
        ws.send(pcm[start : start + frame_bytes])
        if realtime:
            next_at += frame_s
            delay = next_at - time.perf_counter()
            if delay > 0:
                time.sleep(delay)


def run(url: str, wav: Path, lang: str, realtime: bool, tail_s: float, timeout_s: float) -> dict:
    pcm = load_pcm16(wav)
    silence = bytes(int(tail_s * 16000) * 2)
    with connect(url, max_size=None, open_timeout=10) as ws:
        ws.send(json.dumps({"type": "hello", "lang": lang, "client": "e2e_smoke"}))
        while json.loads(ws.recv())["type"] != "state":
            pass
        sent_at = time.perf_counter()
        paced_send(ws, pcm + silence, realtime)
        deadline = time.perf_counter() + timeout_s
        ws.socket.settimeout(timeout_s)
        while time.perf_counter() < deadline:
            message = ws.recv()
            if isinstance(message, bytes):
                continue
            event = json.loads(message)
            kind = event.get("type")
            if kind in {"vad", "transcript", "utterance", "handoff", "error"}:
                print(f"{time.perf_counter() - sent_at:7.2f}s {kind}: {json.dumps(event, ensure_ascii=False)}")
            if kind == "metrics":
                event["wall_s_since_send"] = round(time.perf_counter() - sent_at, 3)
                return event
    raise SystemExit(f"no metrics event within {timeout_s} s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a WAV through the Gaja WebSocket and print turn metrics.")
    parser.add_argument("--url", default="ws://127.0.0.1:8790/ws")
    parser.add_argument("--wav", type=Path, default=ROOT / "bench/out/input_pl_16k.wav")
    parser.add_argument("--lang", default="pl")
    parser.add_argument("--no-realtime", action="store_true", help="send audio as fast as possible")
    parser.add_argument("--tail-s", type=float, default=1.0, help="trailing silence that triggers end of speech")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    metrics = run(args.url, args.wav, args.lang, not args.no_realtime, args.tail_s, args.timeout_s)
    print("METRICS " + json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
