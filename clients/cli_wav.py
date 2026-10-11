#!/usr/bin/env python3
"""
CLI test client for Gaja Voice WebSocket protocol.

Streams a WAV file (resampled to 16 kHz mono PCM16) or typed text over WebSocket.
Measures timing, captures assistant speech audio and transcript/utterance/handoff/metric events.

Usage:
    python clients/cli_wav.py input.wav [--url ws://127.0.0.1:8790/ws] [--lang pl] [--realtime] [--out reply.wav] [--text "..."]

Protocol reference: docs/ARCHITECTURE.md
"""

import argparse
import asyncio
import io
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np
import soundfile as sf
import websockets


def log_event(name: str, detail: str = ""):
    ts = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
    print(f"[{ts}] [{name.upper():<12}] {detail}")


def prepare_audio_pcm16_chunks(wav_path: str, chunk_duration_ms: float = 40.0) -> List[bytes]:
    """Reads a wav file, converts to 16 kHz mono signed 16-bit PCM, and slices into 40ms chunks."""
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    data, orig_sr = sf.read(wav_path, dtype="float32")

    # Downmix to mono if multi-channel
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    target_sr = 16000
    if orig_sr != target_sr:
        # Resample using linear interpolation
        orig_indices = np.linspace(0, len(data), num=len(data), endpoint=False)
        target_num_samples = int(len(data) * target_sr / orig_sr)
        target_indices = np.linspace(0, len(data), num=target_num_samples, endpoint=False)
        data = np.interp(target_indices, orig_indices, data)

    # Clip to [-1.0, 1.0] and convert to PCM16
    data = np.clip(data, -1.0, 1.0)
    pcm16 = (data * 32767.0).astype(np.int16)

    # 40ms at 16kHz = 640 samples = 1280 bytes
    samples_per_chunk = int(target_sr * (chunk_duration_ms / 1000.0))
    raw_bytes = pcm16.tobytes()
    bytes_per_chunk = samples_per_chunk * 2

    chunks = []
    for i in range(0, len(raw_bytes), bytes_per_chunk):
        chunks.append(raw_bytes[i : i + bytes_per_chunk])
    return chunks


async def run_client(args: argparse.Namespace) -> int:
    url = args.url
    lang = args.lang
    text_input = args.text
    input_wav = args.input_wav
    realtime = args.realtime
    out_wav = args.out
    timeout = args.timeout

    log_event("init", f"Connecting to {url} (lang={lang})")

    # Storage for reply audio and metrics
    reply_audio_buffers = []
    reply_sample_rate = 24000
    turn_metrics = None
    transcript_text = None
    assistant_full_text = []

    try:
        async with websockets.connect(url, max_size=10 * 1024 * 1024) as ws:
            log_event("connected", f"Connected to {url}")

            # 1. Send hello
            hello_msg = {
                "type": "hello",
                "lang": lang,
                "tts": None,
                "client": "cli",
            }
            await ws.send(json.dumps(hello_msg))
            log_event("send", f"hello -> lang={lang}, client=cli")

            # 2. Wait for ready
            first_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            if isinstance(first_msg, str):
                ready_data = json.loads(first_msg)
                if ready_data.get("type") == "ready":
                    reply_sample_rate = ready_data.get("sample_rate", 24000)
                    log_event("ready", f"session_id={ready_data.get('session_id')}, sample_rate={reply_sample_rate}")
                else:
                    log_event("warn", f"Expected ready, got: {first_msg}")
            else:
                log_event("error", "Expected text JSON ready message")

            # Sender task
            async def send_input_stream():
                if text_input:
                    log_event("send", f"text: '{text_input}'")
                    await ws.send(json.dumps({"type": "text", "text": text_input}))
                elif input_wav:
                    chunks = prepare_audio_pcm16_chunks(input_wav, chunk_duration_ms=40.0)
                    total_dur = len(chunks) * 0.04
                    log_event("streaming", f"{len(chunks)} frames (~{total_dur:.2f}s audio)")

                    t0 = time.time()
                    for idx, chunk in enumerate(chunks):
                        await ws.send(chunk)
                        if realtime:
                            expected_time = (idx + 1) * 0.04
                            elapsed = time.time() - t0
                            sleep_time = expected_time - elapsed
                            if sleep_time > 0:
                                await asyncio.sleep(sleep_time)
                        else:
                            # Yield control to prevent starvation
                            if idx % 10 == 0:
                                await asyncio.sleep(0.001)

                    log_event("send", "end_of_input")
                    await ws.send(json.dumps({"type": "end_of_input"}))
                else:
                    log_event("error", "Neither input_wav nor --text provided")
                    await ws.send(json.dumps({"type": "end_of_input"}))

            # Receiver loop
            turn_finished = asyncio.Event()

            async def receive_events():
                nonlocal reply_sample_rate, turn_metrics, transcript_text, assistant_full_text
                audio_receiving = False

                async for message in ws:
                    if isinstance(message, bytes):
                        # Audio chunk
                        reply_audio_buffers.append(message)
                    elif isinstance(message, str):
                        try:
                            evt = json.loads(message)
                        except json.JSONDecodeError:
                            log_event("error", f"Invalid JSON received: {message}")
                            continue

                        evt_type = evt.get("type")

                        if evt_type == "vad":
                            log_event("vad", f"state={evt.get('state')}")
                        elif evt_type == "transcript":
                            transcript_text = evt.get("text")
                            log_event("transcript", f"[{evt.get('lang')}] '{transcript_text}' (final={evt.get('final')})")
                        elif evt_type == "assistant_delta":  # pre-v1.7 servers only; the current one sends none
                            assistant_full_text.append(evt.get("text", ""))
                        elif evt_type == "utterance":
                            style = evt.get("style", {})
                            log_event("utterance", f"id={evt.get('id')}: '{evt.get('text')}' (style={style})")
                        elif evt_type == "audio_start":
                            reply_sample_rate = evt.get("sample_rate", reply_sample_rate)
                            audio_receiving = True
                            log_event("audio_start", f"id={evt.get('id')}, sample_rate={reply_sample_rate}")
                        elif evt_type == "audio_end":
                            audio_receiving = False
                            log_event("audio_end", f"id={evt.get('id')}")
                        elif evt_type == "handoff":
                            log_event("handoff", f"state={evt.get('state')}, id={evt.get('id')}, task='{evt.get('task')}', text='{evt.get('text')}'")
                        elif evt_type == "tool":
                            log_event("tool", f"{evt.get('name')}({evt.get('arguments')}) ok={evt.get('ok')}: '{evt.get('text')}'")
                        elif evt_type == "notice":
                            log_event("notice", f"{evt.get('code')}: {evt.get('message')}")
                        elif evt_type == "state":
                            log_event("state", f"state={evt.get('state')}")
                        elif evt_type == "metrics":
                            turn_metrics = evt
                            log_event("metrics", f"stt={evt.get('stt_ms')}ms, llm_ttft={evt.get('llm_ttft_ms')}ms, sentence={evt.get('first_sentence_ms')}ms, tts={evt.get('tts_first_ms')}ms, e2e={evt.get('e2e_first_audio_ms')}ms")
                            # Turn complete
                            turn_finished.set()
                        elif evt_type == "error":
                            log_event("error", f"Server error: {evt.get('message')}")
                            turn_finished.set()

            sender_task = asyncio.create_task(send_input_stream())
            receiver_task = asyncio.create_task(receive_events())

            # Wait for turn completion or timeout
            try:
                await asyncio.wait_for(turn_finished.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                log_event("timeout", f"Timed out after {timeout} seconds waiting for turn completion")
                return 1
            finally:
                sender_task.cancel()
                receiver_task.cancel()

        # Save audio if requested
        if out_wav and reply_audio_buffers:
            raw_audio = b"".join(reply_audio_buffers)
            int16_samples = np.frombuffer(raw_audio, dtype=np.int16)
            float_samples = int16_samples.astype(np.float32) / 32768.0
            sf.write(out_wav, float_samples, reply_sample_rate)
            log_event("saved_audio", f"{out_wav} ({len(float_samples)} samples, {reply_sample_rate} Hz, {len(float_samples)/reply_sample_rate:.2f}s)")

        log_event("done", "Turn completed successfully")
        return 0

    except Exception as e:
        log_event("fatal", str(e))
        return 2


def main():
    parser = argparse.ArgumentParser(description="CLI client for Gaja Voice WebSocket adapter")
    parser.add_argument("input_wav", nargs="?", default=None, help="Path to input .wav audio file")
    parser.add_argument("--url", default="ws://127.0.0.1:8790/ws", help="WebSocket endpoint URL")
    parser.add_argument("--lang", default="pl", choices=["pl", "en", "auto"], help="Language code")
    parser.add_argument("--realtime", action="store_true", help="Pace streaming at 40ms real-time")
    parser.add_argument("--out", default=None, help="Path to save reply audio .wav")
    parser.add_argument("--text", default=None, help="Direct typed text input instead of audio")
    parser.add_argument("--timeout", type=float, default=30.0, help="Turn timeout in seconds")

    args = parser.parse_args()

    if not args.input_wav and not args.text:
        parser.error("Must provide either input.wav or --text '...'")

    code = asyncio.run(run_client(args))
    sys.exit(code)


if __name__ == "__main__":
    main()
