"""End-to-end scenarios against a running adapter (real STT, LLM, TTS and Hermes), over WS /ws.

Run:  flock /tmp/gaja-bench.lock PYTHONPATH=. .venv/bin/python scripts/e2e_suite.py --scenarios all
Scenarios: pl (plain Polish turns), en (English turns), text (typed turn), barge (barge-in
mid-reply), handoff (real Hermes, read-only). Prints per-turn metrics, p50/p90 and load average.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]
FRAME_SAMPLES = 512
PL_CLIPS_ALL = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "bench/data/stt/pl/latency").glob("pl_lat_*.wav"))]
EN_CLIPS_ALL = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "bench/data/stt/en/latency").glob("en_lat_*.wav"))]
BARGE_LONG = "bench/data/stt/pl/latency/pl_lat_002_4.79s.wav"
BARGE_CUT = "bench/data/stt/pl/latency/pl_lat_003_2.12s.wav"
HANDOFF_WAV = "bench/out/e2e/handoff_pl_16k.wav"
TIMING_FIELDS = ("stt_ms", "router_ms", "llm_ttft_ms", "first_sentence_ms", "tts_first_ms", "e2e_first_audio_ms")
ROUTING_HERMES = [
    "Hermes, sprawdź która godzina.",
    "Sprawdź w Hermesie, tylko odczytaj, pogodę w Krakowie na jutro.",
    "Zapytaj Hermesa o aktualny kurs euro, tylko do odczytu.",
    "Przekaż Hermesowi do odczytu: jakie mam dziś wydarzenia w kalendarzu.",
]
# "kim jest Hermes w mitologii" is the false-positive trap: it names Hermes but is a local question.
ROUTING_LOCAL = [
    "Ile to dwa plus dwa?",
    "Opowiedz mi coś miłego.",
    "Jak masz na imię?",
    "Kim jest Hermes w mitologii?",
]
TYPED_TURNS = [("pl", "Opowiedz mi coś miłego, krótko."), ("en", "Tell me something nice, briefly.")]
BARGE_THINK_TEXT = "Opowiedz mi długą historię o Krakowie, z wieloma szczegółami."


def load_pcm16(path: Path) -> bytes:
    samples, rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != 16000:
        raise SystemExit(f"{path} must be 16 kHz, got {rate}")
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class Session:
    def __init__(self, url: str) -> None:
        self.url = url
        self.log: list[dict[str, Any]] = []
        self.audio_times: list[float] = []
        self.ws: Any = None
        self._reader: asyncio.Task[None] | None = None

    async def open(self, lang: str, tts: str | None = None) -> dict[str, Any]:
        self.ws = await connect(self.url, max_size=None, open_timeout=15)
        self._reader = asyncio.create_task(self._read())
        hello: dict[str, Any] = {"type": "hello", "lang": lang, "client": "cli"}
        if tts:
            hello["tts"] = tts
        await self.ws.send(json.dumps(hello))
        _, ready = await self.wait_for(lambda e: e["type"] in {"ready", "error"})
        if ready["type"] == "error":
            raise SystemExit(f"hello failed: {ready}")
        return ready

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
        if self.ws:
            await self.ws.close()

    async def _read(self) -> None:
        async for message in self.ws:
            now = time.perf_counter()
            if isinstance(message, bytes):
                self.audio_times.append(now)
                continue
            event = json.loads(message)
            event["_t"] = now
            self.log.append(event)

    async def wait_for(
        self, pred: Callable[[dict[str, Any]], bool], after: int = 0, timeout: float = 150.0
    ) -> tuple[int, dict[str, Any]]:
        deadline = time.perf_counter() + timeout
        index = after
        while True:
            while index < len(self.log):
                if pred(self.log[index]):
                    return index, self.log[index]
                index += 1
            if time.perf_counter() > deadline:
                raise TimeoutError(f"no matching event within {timeout}s; last={self.log[-3:]}")
            await asyncio.sleep(0.01)

    async def send_paced(self, pcm: bytes) -> float:
        frame_bytes = FRAME_SAMPLES * 2
        frame_s = FRAME_SAMPLES / 16000
        next_at = time.perf_counter()
        for start in range(0, len(pcm), frame_bytes):
            await self.ws.send(pcm[start : start + frame_bytes])
            next_at += frame_s
            delay = next_at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
        return time.perf_counter()


@dataclass
class Turn:
    scenario: str
    label: str
    transcript: str = ""
    reply: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    wall_from_speech_end_ms: float | None = None


async def audio_turn(session: Session, scenario: str, label: str, wav: str, tail_s: float = 1.0) -> Turn:
    start = len(session.log)
    pcm = load_pcm16(ROOT / wav)
    speech_end = await session.send_paced(pcm)
    await session.send_paced(bytes(int(tail_s * 16000) * 2))
    _, metrics_event = await session.wait_for(lambda e: e["type"] == "metrics" and e.get("kind") != "result", after=start)
    turn = Turn(scenario, label, metrics=metrics_event)
    turn.transcript = next((e["text"] for e in session.log[start:] if e["type"] == "transcript"), "")
    turn.reply = " ".join(e["text"] for e in session.log[start:] if e["type"] == "utterance")
    first_audio = next((t for t in session.audio_times if t > speech_end), None)
    if first_audio is not None:
        turn.wall_from_speech_end_ms = round((first_audio - speech_end) * 1000, 1)
    return turn


async def text_turn(session: Session, text: str) -> Turn:
    start = len(session.log)
    sent = time.perf_counter()
    await session.ws.send(json.dumps({"type": "text", "text": text}))
    _, metrics_event = await session.wait_for(lambda e: e["type"] == "metrics" and e.get("kind") != "result", after=start)
    turn = Turn("text", text, metrics=metrics_event)
    turn.reply = " ".join(e["text"] for e in session.log[start:] if e["type"] == "utterance")
    first_audio = next((t for t in session.audio_times if t > sent), None)
    if first_audio is not None:
        turn.wall_from_speech_end_ms = round((first_audio - sent) * 1000, 1)
    return turn


async def scenario_plain(url: str, clips: list[str], lang: str, name: str) -> list[Turn]:
    session = Session(url)
    await session.open(lang)
    turns = []
    try:
        for index, clip in enumerate(clips, 1):
            turns.append(await audio_turn(session, name, f"{name}{index}", clip))
    finally:
        await session.close()
    return turns


async def scenario_text(url: str) -> list[Turn]:
    turns = []
    for lang, text in TYPED_TURNS:
        session = Session(url)
        await session.open(lang)
        try:
            turns.append(await text_turn(session, text))
        finally:
            await session.close()
    return turns


async def scenario_barge(url: str) -> dict[str, Any]:
    session = Session(url)
    await session.open("pl")
    try:
        start = len(session.log)
        await session.send_paced(load_pcm16(ROOT / BARGE_LONG))
        await session.send_paced(bytes(int(1.0 * 16000) * 2))
        _, first_audio = await session.wait_for(lambda e: e["type"] == "audio_start", after=start)
        reply_id = first_audio["id"]
        cut_index = len(session.log)
        cut_at = time.perf_counter()
        frames_before_cut = len(session.audio_times)
        await session.send_paced(load_pcm16(ROOT / BARGE_CUT))
        speech_index, speech = await session.wait_for(
            lambda e: e["type"] == "vad" and e["state"] == "speech_start", after=cut_index
        )
        listening_index, listening = await session.wait_for(
            lambda e: e["type"] == "state" and e["state"] == "listening", after=speech_index + 1
        )
        reply_finished_before_cut = any(
            e["type"] == "audio_end" and e["id"] == reply_id for e in session.log[start:cut_index]
        )
        frames_after_speech_start = sum(1 for t in session.audio_times[frames_before_cut:] if t > speech["_t"])
        await session.send_paced(bytes(int(1.0 * 16000) * 2))
        _, metrics_event = await session.wait_for(lambda e: e["type"] == "metrics", after=listening_index, timeout=150)
        return {
            "reply_id": reply_id,
            "reply_still_speaking_at_cut": not reply_finished_before_cut,
            "speech_start_after_cut_ms": round((speech["_t"] - cut_at) * 1000, 1),
            "listening_after_speech_start_ms": round((listening["_t"] - speech["_t"]) * 1000, 1),
            "audio_frames_after_speech_start": frames_after_speech_start,
            "next_turn_metrics": metrics_event,
        }
    finally:
        await session.close()


async def scenario_handoff(url: str) -> dict[str, Any]:
    session = Session(url)
    await session.open("pl")
    try:
        start = len(session.log)
        turn = await audio_turn(session, "handoff", "handoff", HANDOFF_WAV)
        try:
            index, _ = await session.wait_for(
                lambda e: e["type"] == "handoff" and e["state"] in {"done", "error"}, after=start, timeout=180
            )
        except TimeoutError:
            return {"turn": turn, "handoff_events": [], "spoken_after_done": [], "handoff_fired": False}
        handoff_events = [
            {"state": e["state"], "text": e["text"][:200], "t_rel_s": round(e["_t"] - session.log[start]["_t"], 2)}
            for e in session.log[start : index + 1]
            if e["type"] == "handoff"
        ]
        spoken_after = [e["text"] for e in session.log[index:] if e["type"] == "utterance"]
        await session.wait_for(lambda e: e["type"] == "state" and e["state"] == "listening", after=index, timeout=120)
        return {"turn": turn, "handoff_events": handoff_events, "spoken_after_done": spoken_after, "handoff_fired": True}
    finally:
        await session.close()


async def scenario_barge_thinking(url: str) -> dict[str, Any]:
    session = Session(url)
    await session.open("pl")
    try:
        start = len(session.log)
        await session.ws.send(json.dumps({"type": "text", "text": BARGE_THINK_TEXT}))
        await session.wait_for(lambda e: e["type"] == "state" and e["state"] == "thinking", after=start)
        cut_index = len(session.log)
        state_at_cut = next(
            (e["state"] for e in reversed(session.log[:cut_index]) if e["type"] == "state"), None
        )
        cut_at = time.perf_counter()
        await session.send_paced(load_pcm16(ROOT / BARGE_CUT))
        speech_index, speech = await session.wait_for(
            lambda e: e["type"] == "vad" and e["state"] == "speech_start", after=cut_index
        )
        listening_index, listening = await session.wait_for(
            lambda e: e["type"] == "state" and e["state"] == "listening", after=speech_index + 1
        )
        await session.send_paced(bytes(int(1.0 * 16000) * 2))
        await session.wait_for(lambda e: e["type"] == "metrics", after=listening_index, timeout=150)
        return {
            "state_at_cut": state_at_cut,
            "speech_start_after_cut_ms": round((speech["_t"] - cut_at) * 1000, 1),
            "listening_after_speech_start_ms": round((listening["_t"] - speech["_t"]) * 1000, 1),
            "listening_reason": listening.get("reason"),
            "utterances_before_cut": sum(1 for e in session.log[start:cut_index] if e["type"] == "utterance"),
        }
    finally:
        await session.close()


async def scenario_routing(url: str) -> list[dict[str, Any]]:
    session = Session(url)
    await session.open("pl")
    rows: list[dict[str, Any]] = []
    try:
        plan = [(True, text) for text in ROUTING_HERMES] + [(False, text) for text in ROUTING_LOCAL]
        for expect_handoff, text in plan:
            start = len(session.log)
            await session.ws.send(json.dumps({"type": "text", "text": text}))
            _, metrics_event = await session.wait_for(lambda e: e["type"] == "metrics", after=start, timeout=180)
            started = next((e for e in session.log[start:] if e["type"] == "handoff" and e["state"] == "started"), None)
            done_ms = None
            if started is not None:
                done_index, done = await session.wait_for(
                    lambda e: e["type"] == "handoff" and e["state"] in {"done", "error"}, after=start, timeout=180
                )
                done_ms = round((done["_t"] - started["_t"]) * 1000, 1)
                await session.wait_for(lambda e: e["type"] == "state" and e["state"] == "listening", after=done_index, timeout=120)
            rows.append({
                "text": text,
                "expect_handoff": expect_handoff,
                "handoff": started is not None,
                "by_router": started is not None and started.get("text") == "router",
                "router_ms": metrics_event.get("router_ms"),
                "ack_first_audio_ms": metrics_event.get("e2e_first_audio_ms"),
                "hermes_done_ms": done_ms,
                "reply": " ".join(e["text"] for e in session.log[start:] if e["type"] == "utterance")[:160],
            })
    finally:
        await session.close()
    return rows


def routing_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for r in rows if r["expect_handoff"] and r["handoff"])
    fn = sum(1 for r in rows if r["expect_handoff"] and not r["handoff"])
    fp = sum(1 for r in rows if not r["expect_handoff"] and r["handoff"])
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "max": None}
    return {
        "p50": round(float(np.percentile(values, 50)), 1),
        "p90": round(float(np.percentile(values, 90)), 1),
        "max": round(float(max(values)), 1),
    }


def summarise(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"p50={np.percentile(values, 50):.0f} p90={np.percentile(values, 90):.0f} (n={len(values)})"


async def run(args: argparse.Namespace) -> None:
    all_scenarios = {"pl", "en", "text", "barge", "barge_think", "handoff", "routing"}
    wanted = all_scenarios if args.scenarios == "all" else set(args.scenarios.split(","))
    load_start = Path("/proc/loadavg").read_text().split()[:3]
    report: dict[str, Any] = {"load_start": load_start, "turns": []}
    turns: list[Turn] = []
    if "pl" in wanted:
        turns += await scenario_plain(args.url, PL_CLIPS_ALL[: args.pl_clips], "pl", "pl")
    if "en" in wanted:
        turns += await scenario_plain(args.url, EN_CLIPS_ALL[: args.en_clips], "en", "en")
    if "text" in wanted:
        turns += await scenario_text(args.url)
    if "barge" in wanted:
        report["barge"] = await scenario_barge(args.url)
    if "barge_think" in wanted:
        report["barge_think"] = await scenario_barge_thinking(args.url)
    if "handoff" in wanted:
        handoff = await scenario_handoff(args.url)
        report["handoff"] = {**handoff, "turn": None}
        turns.append(handoff["turn"])
    if "routing" in wanted:
        report["routing"] = await scenario_routing(args.url)
        report["routing_summary"] = routing_summary(report["routing"])
    report["load_end"] = Path("/proc/loadavg").read_text().split()[:3]

    print(f"{'turn':10} {'transcript':44} " + " ".join(f"{f:>18}" for f in TIMING_FIELDS) + f" {'wall_speech_end':>16}")
    for turn in turns:
        row = [turn.metrics.get(f) for f in TIMING_FIELDS]
        cells = " ".join(f"{v if v is not None else '-':>18}" for v in row)
        print(f"{turn.label:10} {turn.transcript[:44]:44} {cells} {str(turn.wall_from_speech_end_ms):>16}")
        report["turns"].append({"scenario": turn.scenario, "label": turn.label, "transcript": turn.transcript,
                                "reply": turn.reply, "metrics": turn.metrics,
                                "wall_from_speech_end_ms": turn.wall_from_speech_end_ms})
    # Audio turns (pl/en clips) are the headline: typed turns skip STT/VAD and the handoff turn
    # speaks a fixed acknowledgement, so they are summarised separately.
    groups = {
        "audio": [t for t in turns if t.scenario in ("pl", "en")],
        "text": [t for t in turns if t.scenario == "text"],
        "ack": [t for t in turns if t.scenario not in ("pl", "en", "text")],
    }
    report["summary"] = {}
    for kind, timed in groups.items():
        if not timed:
            continue
        print(f"\n{kind} turns: per-field summary over {len(timed)} turns, ms:")
        stats: dict[str, Any] = {"n": len(timed)}
        for field_name in TIMING_FIELDS + ("tool_select_ms", "tool_ms"):
            values = [t.metrics[field_name] for t in timed if t.metrics.get(field_name) is not None]
            if values or field_name in TIMING_FIELDS:
                print(f"  {field_name:20} {summarise(values)}")
                stats[field_name] = percentiles(values)
        walls = [t.wall_from_speech_end_ms for t in timed if t.wall_from_speech_end_ms is not None]
        print(f"  {'wall_from_speech_end':20} {summarise(walls)}")
        stats["wall_from_speech_end_ms"] = percentiles(walls)
        stats["routes"] = sorted({str(t.metrics.get("route")) for t in timed})
        report["summary"][kind] = stats
    print(f"\nload avg start={' '.join(load_start)} end={' '.join(report['load_end'])}")
    if "barge" in report:
        print("barge:", json.dumps({k: v for k, v in report["barge"].items() if k != "next_turn_metrics"}, ensure_ascii=False))
    if "barge_think" in report:
        print("barge_think:", json.dumps(report["barge_think"], ensure_ascii=False))
    if "handoff" in report:
        print("handoff fired:", report["handoff"]["handoff_fired"])
        print("handoff:", json.dumps(report["handoff"]["handoff_events"], ensure_ascii=False, indent=1))
        print("spoken after done:", report["handoff"]["spoken_after_done"])
    if "routing" in report:
        print("\nrouting (text turns; expect_handoff vs observed):")
        for row in report["routing"]:
            print(f"  expect={row['expect_handoff']!s:5} handoff={row['handoff']!s:5} by_router={row['by_router']!s:5} "
                  f"router_ms={row['router_ms']} ack_ms={row['ack_first_audio_ms']} hermes_done_ms={row['hermes_done_ms']} "
                  f"| {row['text'][:48]}")
        print("routing summary:", json.dumps(report["routing_summary"]))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://127.0.0.1:8790/ws")
    parser.add_argument("--scenarios", default="all", help="all or comma list of pl,en,text,barge,barge_think,handoff,routing")
    parser.add_argument("--pl-clips", type=int, default=10, help="PL audio turns from bench/data/stt/pl/latency (max 10)")
    parser.add_argument("--en-clips", type=int, default=4, help="EN audio turns from bench/data/stt/en/latency (max 10)")
    parser.add_argument("--json", default=str(ROOT / "bench/out/e2e/results.json"))
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
