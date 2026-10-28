#!/usr/bin/env python3
"""Realtime E2E na PYTANIACH: TTS generuje pytanie -> to samo audio w pętli WS.

Pytania faktograficzne/arytmetyczne (słabe punkty budget=0) idą pełną pętlą:
audio-in -> (reasoning) -> audio+text-out. Mierzone: TTFA, round-trip, treść
odpowiedzi. Uruchamiać po jednym serwerze (serwer startuje zadanym wariantem).

  BENCH_LLM_PORT=8794 REASON_ARGS="--reasoning-budget 128" \
      uv run python bench/lfm3_realtime_questions.py --tag b128
"""
import argparse
import asyncio
import base64
import json
import os
import sys
import time
import wave
from pathlib import Path

import requests
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfm3_audio_stt_smoke import gguf_path, S  # noqa: E402
from lfm3_realtime_e2e import RTSession, stream_clip  # noqa: E402

RESULTS = S.ROOT / "bench/results"
PORT = int(os.environ.get("BENCH_LLM_PORT", "8794"))
RATE = 16000

QUESTIONS = [
    "Jaka jest stolica Australii?",
    "Ile to dwadzieścia pięć procent z dwustu czterdziestu?",
    "Ile planet jest w Układzie Słonecznym?",
]

PERSONA = ("Jesteś Gają, asystentką głosową w polskim domu. Słuchasz pytania użytkownika "
           "i udzielasz krótkiej, rzeczowej odpowiedzi po polsku: jedno, góra dwa zdania. "
           "Liczby zapisuj słownie. Jeśli czegoś nie wiesz — powiedz to szczerze.")


def start_server(reason_args):
    import subprocess
    model = gguf_path("models--LiquidAI--LFM3-Audio-2B-GGUF", "LFM3-Audio-2B-Q4_K_M.gguf")
    mmproj = gguf_path("models--LiquidAI--LFM3-Audio-2B-GGUF", "mmproj-LFM3-Audio-2B-Q8_0.gguf")
    cmd = [str(S.BIN), "-m", str(model), "--mmproj", str(mmproj),
           "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99", "-c", "8192",
           "-np", "1", "--jinja", "-fa", "on"] + reason_args
    print("server:", " ".join(cmd[2:]), flush=True)
    log = open(S.ROOT / "bench/logs/rt_questions_server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(240):
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1).status_code == 200:
                return proc
        except requests.RequestException:
            pass
        if proc.poll() is not None:
            raise SystemExit("serwer padł")
        time.sleep(0.5)
    raise SystemExit("serwer nie wstał")


def tts_question(text: str) -> Path:
    r = requests.post(f"http://127.0.0.1:{PORT}/tts", json={"input": text, "language": "pl"}, timeout=300)
    r.raise_for_status()
    p = RESULTS / f"rt_question_{abs(hash(text)) % 10000}.wav"
    p.write_bytes(r.content)
    return p


def wav_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 24000, "TTS zwraca 24k; realtime chce 16k — resamplujemy"
        import numpy as np
        pcm24 = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        # prosty decymator 24k->16k (x2/3): średnia par + co 2 z 3
        n = (len(pcm24) // 3) * 2
        res = np.empty(n, dtype=np.int16)
        res[0::2] = ((pcm24[0::3].astype(np.int32) + pcm24[1::3]) // 2).astype(np.int16)[:n // 2]
        res[1::2] = ((pcm24[1::3].astype(np.int32) + pcm24[2::3]) // 2).astype(np.int16)[:n // 2]
        return res.tobytes()


async def ask(ws_url, clip: Path):
    async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
        sess = RTSession(ws)
        await sess.send({"type": "session.update", "session": {
            "type": "realtime", "instructions": PERSONA,
            "input_audio_format": "pcm16", "output_modalities": ["text", "audio"]}})
        pcm = wav_pcm16(clip)
        await stream_clip(sess, pcm)
        t0 = time.perf_counter()
        await sess.send({"type": "input_audio_buffer.commit"})
        await sess.send({"type": "response.create"})
        marks = {"ttfa": None, "ttft": None, "done": None, "text": "", "audio_b64": ""}
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=180)
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                break
            now = time.perf_counter()
            ev = json.loads(raw)
            t = ev.get("type", "")
            if t == "response.output_text.delta":
                marks["ttft"] = marks["ttft"] or round(now - t0, 2)
                marks["text"] += ev.get("delta", "")
            elif t == "response.output_audio.delta":
                marks["ttfa"] = marks["ttfa"] or round(now - t0, 2)
                marks["audio_b64"] += ev.get("delta", "")
            elif t == "response.done":
                marks["done"] = round(now - t0, 2)
                break
            elif t == "error":
                marks.setdefault("errors", []).append(ev.get("error", ev))
        wav_out = None
        if marks["audio_b64"]:
            pcm_out = base64.b64decode(marks["audio_b64"])
            wav_out = RESULTS / f"rt_answer_{clip.stem[:24]}.wav"
            with wave.open(str(wav_out), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
                w.writeframes(pcm_out)
        del marks["audio_b64"]
        marks["answer_wav"] = str(wav_out) if wav_out else None
        return marks


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    reason = [a for a in os.environ.get("REASON_ARGS", "").split() if a]
    RESULTS.mkdir(exist_ok=True)
    proc = start_server(reason)
    try:
        await asyncio.sleep(2)
        out = []
        for q in QUESTIONS:
            clip = tts_question(q)
            marks = await ask(f"ws://127.0.0.1:{PORT}/v1/realtime", clip)
            marks["question"] = q
            out.append(marks)
            print(f"[{args.tag}] {q}\n  ttft={marks.get('ttft')} ttfa={marks.get('ttfa')} "
                  f"done={marks.get('done')}\n  odp: {marks.get('text','')[:120]!r}", flush=True)
        (RESULTS / f"rt_questions_{args.tag}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", RESULTS / f"rt_questions_{args.tag}.json", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
