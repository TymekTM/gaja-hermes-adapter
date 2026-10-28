#!/usr/bin/env python3
"""E2E /v1/realtime (upstream PR #31269): LFM3-Audio-2B jako pętla głosowa single-model.

Dwa tryby testu (WebSocket, OpenAI-Realtime-style):
1. TRANSCRIPTION — 44 klipy FLEURS PL streamingiem (chunki 200 ms) -> WER
   (jiwer+normalize, metodologia repo) + latencja commit->transkrypcja.
2. REALTIME conversation — klip audio jako "wypowiedź użytkownika", odpowiedź
   modelu w audio+text: time-to-first-transcript-delta, time-to-first-audio,
   full round-trip; audio-out zapisywane do WAV (ocena jakości przez człowieka).

  BENCH_LLM_PORT=8794 uv run python bench/lfm3_realtime_e2e.py
"""
import asyncio
import base64
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import requests
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_hotwords_cost import calc_wer, normalize_text, read_tsv  # noqa: E402
from lfm3_audio_stt_smoke import gguf_path, S  # noqa: E402

MAIN = Path("/home/ttm-server/.t3/projects/gaja-hermes-adapter/bench/data/stt")
RESULTS = S.ROOT / "bench/results"
PORT = int(os.environ.get("BENCH_LLM_PORT", "8794"))
WS_URL = f"ws://127.0.0.1:{PORT}/v1/realtime"
RATE = 16000
PERSONA = ("Jesteś Gają, asystentką głosową w polskim domu. Odpowiadasz bardzo krótko "
           "po polsku, maksymalnie jedno-dwa zdania, zwykłym tekstem bez emoji.")


def start_server():
    model = gguf_path("models--LiquidAI--LFM3-Audio-2B-GGUF", "LFM3-Audio-2B-Q4_K_M.gguf")
    mmproj = gguf_path("models--LiquidAI--LFM3-Audio-2B-GGUF", "mmproj-LFM3-Audio-2B-Q8_0.gguf")
    import subprocess
    cmd = [str(S.BIN), "-m", str(model), "--mmproj", str(mmproj),
           "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99", "-c", "8192",
           "-np", "1", "--jinja", "-fa", "on"]
    log = open(S.ROOT / "bench/logs/realtime_server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    t0 = time.perf_counter()
    for _ in range(240):
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1).status_code == 200:
                return proc, time.perf_counter() - t0
        except requests.RequestException:
            pass
        if proc.poll() is not None:
            raise SystemExit("serwer padł: bench/logs/realtime_server.log")
        time.sleep(0.5)
    raise SystemExit("serwer nie wstał")


def wav_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2, \
            f"clip {path} nie jest 16k mono pcm16"
        return w.readframes(w.getnframes())


class RTSession:
    def __init__(self, ws):
        self.ws = ws
        self.events = []

    async def send(self, obj):
        await self.ws.send(json.dumps(obj))

    async def recv_until(self, stop_types, timeout=120):
        """Zbieraj eventy aż przyjdzie któryś ze stop_types; zwraca (stop_event, wszystkie)."""
        out = []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                break
            ev = json.loads(raw)
            out.append((time.perf_counter(), ev))
            if ev.get("type") in stop_types or ev.get("type") == "error":
                return ev, out
        return None, out


async def stream_clip(sess: RTSession, pcm: bytes, chunk_ms=200):
    """Append audio w chunkach (symulacja streamingu); zwraca czas ostatniego chunka."""
    chunk = int(RATE * 2 * chunk_ms / 1000)
    n = max(1, len(pcm) // chunk)
    t_last = None
    for i in range(n):
        part = pcm[i * chunk:(i + 1) * chunk] if i < n - 1 else pcm[i * chunk:]
        await sess.send({"type": "input_audio_buffer.append",
                         "audio": base64.b64encode(part).decode()})
        t_last = time.perf_counter()
        await asyncio.sleep(chunk_ms / 1000 * 0.9)  # realtime-ish pacing
    return t_last


async def transcription_mode():
    """WER na 44 FLEURS + latencja commit->completed."""
    rows, lats = [], []
    async with websockets.connect(WS_URL, max_size=64 * 1024 * 1024) as ws:
        sess = RTSession(ws)
        await sess.send({"type": "session.update", "session": {
            "type": "transcription",
            "input_audio_format": "pcm16",
            "input_audio_transcription": {"language": "pl"},
        }})
        refs = read_tsv(MAIN / "pl/refs.tsv")
        agg = {"S": 0, "D": 0, "I": 0, "H": 0, "N": 0}
        for row in refs:
            pcm = wav_pcm16(MAIN / "pl" / row["file"])
            t_last = await stream_clip(sess, pcm)
            t_commit = time.perf_counter()
            await sess.send({"type": "input_audio_buffer.commit"})
            stop, evs = await sess.recv_until(
                {"conversation.item.input_audio_transcription.completed"}, timeout=180)
            dt = time.perf_counter() - t_commit
            text = (stop or {}).get("transcript", "")
            c = calc_wer(normalize_text(row["ref"]), normalize_text(text))
            for k in agg:
                agg[k] += c[k]
            rows.append({"id": row["utt_id"], "ref": row["ref"], "hyp": text,
                         "commit_to_done_s": round(dt, 2)})
            lats.append(dt)
            print(f"  {row['utt_id']}: {dt:5.1f}s {text[:70]!r}", flush=True)
        wer = (agg["S"] + agg["D"] + agg["I"]) / agg["N"] if agg["N"] else 1.0
    return {"wer": round(wer, 4),
            "commit_to_done_p50_ms": round(float(np.percentile(lats, 50)) * 1000),
            "commit_to_done_p90_ms": round(float(np.percentile(lats, 90)) * 1000),
            "rows": rows}


async def realtime_conversation(clips: list[Path]):
    out = []
    for clip in clips:
        async with websockets.connect(WS_URL, max_size=64 * 1024 * 1024) as ws:
            sess = RTSession(ws)
            await sess.send({"type": "session.update", "session": {
                "type": "realtime",
                "instructions": PERSONA,
                "input_audio_format": "pcm16",
                "output_modalities": ["text", "audio"],
            }})
            pcm = wav_pcm16(clip)
            t_last = await stream_clip(sess, pcm)
            t_commit = time.perf_counter()
            await sess.send({"type": "input_audio_buffer.commit"})
            # odpowiedź nie startuje automatycznie: wymagany jawny response.create
            await sess.send({"type": "response.create"})
            marks = {"first_transcript_delta": None, "first_text_delta": None,
                     "first_audio_delta": None, "response_done": None, "transcript": "",
                     "text": "", "audio_b64": ""}
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=180)
                except (asyncio.TimeoutError, websockets.ConnectionClosed):
                    break
                now = time.perf_counter()
                ev = json.loads(raw)
                t = ev.get("type", "")
                if t == "conversation.item.input_audio_transcription.delta":
                    marks["first_transcript_delta"] = marks["first_transcript_delta"] or (now - t_commit)
                    marks["transcript"] += ev.get("delta", "")
                elif t == "response.output_text.delta":
                    marks["first_text_delta"] = marks["first_text_delta"] or (now - t_commit)
                    marks["text"] += ev.get("delta", "")
                elif t == "response.output_audio.delta" or "audio" in t and "delta" in t:
                    marks["first_audio_delta"] = marks["first_audio_delta"] or (now - t_commit)
                    marks["audio_b64"] += ev.get("delta", "")
                elif t == "response.done":
                    marks["response_done"] = now - t_commit
                    break
                elif t == "error":
                    marks.setdefault("errors", []).append(ev)
            if marks["audio_b64"]:
                pcm_out = base64.b64decode(marks["audio_b64"])
                outwav = RESULTS / f"lfm3_realtime_out_{clip.stem[:20]}.wav"
                with wave.open(str(outwav), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
                    w.writeframes(pcm_out)
                marks["audio_wav"] = str(outwav)
                marks["audio_len_s"] = round(len(pcm_out) / 2 / 24000, 2)
            del marks["audio_b64"]
            marks["clip"] = str(clip)
            out.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in marks.items()})
            print(f"  {clip.name}: ttfd={marks.get('first_transcript_delta')} "
                  f"ttfa={marks.get('first_audio_delta')} done={marks.get('response_done')} "
                  f"transcript={marks.get('transcript','')[:50]!r} text={marks.get('text','')[:60]!r}", flush=True)
    return out


async def main():
    proc, load_s = start_server()
    try:
        time.sleep(2)
        print("== TRYB 1: transcription (44 FLEURS PL, streaming 200 ms) ==", flush=True)
        tr = await transcription_mode()
        print(f"   WER: {tr['wer']:.2%} | commit->done p50={tr['commit_to_done_p50_ms']} ms "
              f"p90={tr['commit_to_done_p90_ms']} ms", flush=True)
        print("== TRYB 2: realtime conversation (3 klipy, audio+text out) ==", flush=True)
        clips = [MAIN / "pl/wer" / f for f in
                 ("pl_000_fleurs1829.wav", "pl_006_fleurs*.wav", "pl_004_fleurs*.wav")]
        clips = [next(MAIN.glob(f"pl/wer/{c.name.split('_fleurs')[0]}_fleurs*.wav")) if "*" in str(c) else c for c in clips]
        convo = await realtime_conversation([c for c in clips if c.exists()])
        result = {"server": "llama-server master (e486a4b, PR #31269)", "load_time_s": round(load_s, 1),
                  "transcription_mode": {k: v for k, v in tr.items() if k != "rows"},
                  "realtime_conversation": convo, "transcription_rows": tr["rows"]}
        (RESULTS / "lfm3_realtime_e2e.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                                        encoding="utf-8")
        print("wrote bench/results/lfm3_realtime_e2e.json", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
