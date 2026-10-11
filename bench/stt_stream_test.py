"""Whistle live-streaming test (paced chunks) and silero-vad endpointing evaluation."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stt_bench import DATA, RESULTS, normalize, read_tsv, word_errors  # noqa: E402

SR = 16000


def corpus_wer(errs):
    if not errs:
        return None
    total = {k: sum(e[k] for e in errs) for k in ("S", "D", "I", "N")}
    return (total["S"] + total["D"] + total["I"]) / total["N"]


def stream_clip(model, samples, chunk_s, lang):
    n = int(round(chunk_s * SR))
    total_s = len(samples) / SR
    marks = {}

    def paced():
        start = marks["start"] = time.perf_counter()
        i = 0
        while i < len(samples):
            chunk = samples[i:i + n]
            i += len(chunk)
            delay = start + i / SR - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            marks.setdefault("feed", []).append(time.perf_counter() - start)
            yield chunk

    events = []
    for step in model.stream(paced(), language=lang):
        events.append({"recv_s": time.perf_counter() - marks["start"], "text": step["text"],
                       "pending": step.get("pending", ""), "words": step.get("words", []),
                       "pass_ms": step.get("pass_ms"), "received_s": step.get("received")})
    final_text = " ".join(e["text"] for e in events if e["text"]).strip()
    tail = events[-1]
    word_lat = [e["recv_s"] - w["end"] for e in events[:-1] for w in e["words"]]
    last_feed = marks["feed"][-1]
    return {
        "audio_s": round(total_s, 3), "chunk_s": chunk_s, "final_text": final_text, "last_pending": tail["pending"],
        "final_latency_s": round(tail["recv_s"] - last_feed, 3),
        "word_commit_lag_s": {"median": statistics.median(word_lat) if word_lat else None,
                              "max": max(word_lat) if word_lat else None, "n": len(word_lat)},
        "pass_ms": {"median": statistics.median([e["pass_ms"] for e in events if e["pass_ms"] is not None]),
                    "max": max(e["pass_ms"] for e in events if e["pass_ms"] is not None)},
        "partials": [{"t_s": round(e["recv_s"], 2), "received_s": e["received_s"], "text": e["text"],
                      "pending": e["pending"], "pass_ms": e["pass_ms"]} for e in events],
        "last_feed_s": round(last_feed, 3),
    }


def run_stream(args):
    from needle import Whistle

    model = Whistle()
    model.transcribe(np.zeros(SR, np.float32), language="pl")
    rows = read_tsv(DATA / "pl" / "refs.tsv")
    clips = [r for r in rows if 4.5 <= float(r["duration_s"]) <= 9.5][:3]
    jobs = [(str(DATA / "pl" / r["file"]), r["ref"]) for r in clips]
    jobs.append((str(Path(__file__).resolve().parent / "out" / "input_pl_16k.wav"), None))
    results = []
    for path, ref in jobs:
        samples = sf.read(path, dtype="float32")[0]
        for chunk_s in args.chunks:
            model.transcribe(samples[: SR * 2], language="pl")
            out = stream_clip(model, samples, chunk_s, "pl")
            entry = {"clip": Path(path).name, **out}
            if ref is not None:
                entry["ref"] = normalize(ref, "pl")
                entry["err"] = word_errors(entry["ref"], normalize(out["final_text"], "pl"))
            results.append(entry)
            print(json.dumps({k: entry[k] for k in ("clip", "chunk_s", "final_latency_s", "word_commit_lag_s", "pass_ms", "final_text")},
                             ensure_ascii=False), flush=True)
    summary = {}
    for chunk_s in args.chunks:
        sub = [r for r in results if r["chunk_s"] == chunk_s]
        summary[str(chunk_s)] = {
            "final_latency_s_median": statistics.median([r["final_latency_s"] for r in sub]),
            "pass_ms_median": statistics.median([r["pass_ms"]["median"] for r in sub]),
            "pass_ms_max": max(r["pass_ms"]["max"] for r in sub),
            "word_commit_lag_median": statistics.median([r["word_commit_lag_s"]["median"] for r in sub if r["word_commit_lag_s"]["median"] is not None]),
            "corpus_wer": corpus_wer([r["err"] for r in sub if "err" in r]),
        }
    (RESULTS / "stt_stream_whistle.json").write_text(json.dumps({"summary": summary, "clips": results},
                                                                ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def vad_ends(vad, audio, min_silence_ms, speech_pad_ms):
    import torch
    from silero_vad import VADIterator

    it = VADIterator(vad, threshold=0.5, sampling_rate=SR, min_silence_duration_ms=min_silence_ms,
                     speech_pad_ms=speech_pad_ms)
    ends = []
    for i in range(0, len(audio) - 511, 512):
        ev = it(torch.from_numpy(audio[i:i + 512]), return_seconds=True)
        if ev and "end" in ev:
            ends.append(ev["end"])
    return ends


def run_vad(args):
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    torch.set_num_threads(1)
    vad = load_silero_vad(onnx=True)
    frame = torch.zeros(512)
    for _ in range(50):
        vad(frame, SR)
    t = time.perf_counter()
    reps = 2000
    for _ in range(reps):
        vad(frame, SR)
    per_frame_us = (time.perf_counter() - t) / reps * 1e6

    rows = read_tsv(DATA / "pl" / "refs.tsv")
    clips = []
    for row in rows:
        audio = sf.read(DATA / "pl" / row["file"], dtype="float32")[0]
        segs = get_speech_timestamps(torch.from_numpy(audio), vad, sampling_rate=SR, min_silence_duration_ms=100,
                                     return_seconds=True)
        clips.append((audio, segs))
    silence = np.zeros(int(2.0 * SR), np.float32)
    configs = [{"min_silence_ms": ms, "threshold": 0.5, "speech_pad_ms": 30} for ms in (200, 300, 400, 500, 700)]
    results = []
    for cfg in configs:
        mid_cuts, delays, missed = 0, [], 0
        for audio, segs in clips:
            last_end = segs[-1]["end"]
            ends = vad_ends(vad, np.concatenate([audio, silence]), cfg["min_silence_ms"], cfg["speech_pad_ms"])
            if any(seg["start"] > e + 0.05 for e in ends for seg in segs):
                mid_cuts += 1
            after = [e for e in ends if e >= last_end - 0.05]
            if after:
                delays.append(after[0] - last_end)
            else:
                missed += 1
        false_on_silence = bool(vad_ends(vad, np.concatenate([silence, silence]), cfg["min_silence_ms"], cfg["speech_pad_ms"]))
        results.append({**cfg, "clips": len(clips), "clips_with_mid_utterance_cut": mid_cuts, "clips_no_end_detected": missed,
                        "end_delay_s_median": statistics.median(delays) if delays else None,
                        "end_delay_s_p90": float(np.percentile(delays, 90)) if delays else None,
                        "silence_only_2s_triggers_end": false_on_silence})
        print(json.dumps(results[-1]), flush=True)
    out = {"per_frame_us_median_run": per_frame_us, "frame_samples": 512, "frame_ms": 32, "torch_threads": 1,
           "configs": results}
    (RESULTS / "stt_vad_silero.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"per_frame_us": per_frame_us}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["stream", "vad"], required=True)
    p.add_argument("--chunks", type=float, nargs="+", default=[0.1, 0.2, 0.5, 1.0])
    args = p.parse_args()
    run_stream(args) if args.mode == "stream" else run_vad(args)


if __name__ == "__main__":
    main()
