#!/usr/bin/env python3
"""Style demo and objective proxies for the Gaja style presets, through the production engines.

Each engine is the class the adapter loads (gaja/tts/*.py), so the audio measured and written here is
the audio the server plays: Piper, Supertonic and Paradee apply gaja.tts.prosody.apply_post (pitch,
energy, pauses). See docs/PROSODY.md.

Outputs:
  bench/out/tts/demo_styles_<engine>.wav        one sentence per style, 8 styles, <= 30 s
  bench/out/tts/demo_styles_<engine>.json       start time (s) of each style in that wav
  bench/out/tts/demo_conversation_<engine>.wav  mixed-tag conversation, parsed with parse_styled, <= 30 s
  bench/out/tts/prosody_wer/<engine>/<style>_<id>.wav  the bench sentences per style
  bench/results/prosody_<engine>.json           per-style proxies and Whistle WER

Proxies are signal measurements, not a listening test: F0 median and semitone SD (pyin on voiced
frames), rate (neutral trimmed speech duration / style duration), loudness (RMS dB vs neutral).
WER gate: mean Whistle round-trip WER on the bench sentences within +0.05 of neutral.

Timed run (use the lock, record loadavg):
  flock /tmp/gaja-bench.lock .venv/bin/python bench/prosody_demo.py --engine piper_gosia
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH))

from gaja.text.style import STYLE_PRESETS, Style, parse_styled  # noqa: E402
from gaja.tts.paradee_tts import ParadeeTTS  # noqa: E402
from gaja.tts.piper_tts import PiperTTS  # noqa: E402
from gaja.tts.supertonic_tts import SupertonicTTS  # noqa: E402
from tts_bench import load_sentences, stt_round_trip  # noqa: E402

OUT = BENCH / "out" / "tts"
RESULTS = BENCH / "results"
MAX_DEMO_S = 30.0
STYLE_GAP_S = 0.8
SEGMENT_GAP_S = 0.35
STYLES = ["neutral", "happy", "excited", "calm", "sad", "serious", "fast", "slow"]
ENGINE_LANG = {"piper_gosia": "pl", "piper_darkman": "pl", "supertonic": "pl", "paradee": "en"}
DEMO = {
    "pl": ["Dobrze, zapisałam to w kalendarzu.", "Mam dla ciebie trzy wolne terminy w przyszłym tygodniu, który wybierasz?"],
    "en": ["Alright, I have added it to your calendar.", "I have three free slots next week, which one do you prefer?"],
}
CONVERSATION = {
    "pl": "[happy] Super, udało się! [calm] Teraz spokojnie sprawdzę resztę. [sad] Niestety, jedna rzecz nie działa.",
    "en": "[happy] Great, it worked! [calm] Now let me check the rest calmly. [sad] Unfortunately, one thing does not work.",
}
PIPER_GOSIA = ROOT / "models" / "pl_PL-gosia-medium.onnx"
PIPER_DARKMAN = ROOT / "models" / "pl_PL-darkman-medium.onnx"


class Adapter:
    def __init__(self, engine, lang: str):
        self.engine, self.lang = engine, lang
        self.loop = asyncio.new_event_loop()
        self.sr = engine.sample_rate

    def synth(self, text, style):
        return np.asarray(self.loop.run_until_complete(self.engine.synthesize(text, style, self.lang)), dtype=np.float32)


def build_engine(a):
    if a.engine == "piper_gosia":
        return PiperTTS("piper", PIPER_GOSIA, threads=a.piper_threads)
    if a.engine == "piper_darkman":
        return PiperTTS("piper-darkman", PIPER_DARKMAN, threads=a.piper_threads)
    if a.engine == "supertonic":
        return SupertonicTTS(steps=a.supertonic_steps, voice=a.supertonic_voice, threads=a.supertonic_threads,
                             output_sample_rate=a.supertonic_sr or None)
    return ParadeeTTS(threads=a.paradee_threads)


def join(parts, sr, gap_s):
    gap = np.zeros(int(gap_s * sr), dtype=np.float32)
    pieces = []
    for i, p in enumerate(parts):
        if i:
            pieces.append(gap)
        pieces.append(p)
    return np.concatenate(pieces)


def speech_s(x, sr):
    _, idx = librosa.effects.trim(x, top_db=35)
    return (idx[1] - idx[0]) / sr


def f0_stats(x, sr):
    f0, voiced, _ = librosa.pyin(x, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=256)
    v = f0[voiced & np.isfinite(f0)]
    if len(v) < 5:
        return float("nan"), float("nan")
    return float(np.median(v)), float(np.std(12 * np.log2(v)))


def rms_db(x):
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=list(ENGINE_LANG))
    ap.add_argument("--piper-threads", type=int, default=2)
    ap.add_argument("--paradee-threads", type=int, default=1)
    ap.add_argument("--supertonic-voice", default="M4")
    ap.add_argument("--supertonic-steps", type=int, default=6)
    ap.add_argument("--supertonic-threads", type=int, default=2)
    ap.add_argument("--supertonic-sr", type=int, default=22050)
    ap.add_argument("--skip-wer", action="store_true")
    a = ap.parse_args()
    engine_name, lang = a.engine, ENGINE_LANG[a.engine]

    loadavg_start = [round(x, 2) for x in os.getloadavg()]
    eng = Adapter(build_engine(a), lang)
    sr = eng.sr
    eng.synth(DEMO[lang][0], Style())

    per_style, demo_audio = {}, {}
    for name in STYLES:
        style = STYLE_PRESETS[name]
        parts, synth_s = [], []
        for text in DEMO[lang]:
            t0 = time.perf_counter()
            parts.append(eng.synth(text, style))
            synth_s.append(time.perf_counter() - t0)
        demo_audio[name] = parts
        f0s = [f0_stats(p, sr) for p in parts]
        per_style[name] = {
            "emotion": style.emotion,
            "rate_preset": style.rate,
            "pitch_st_preset": style.pitch_semitones,
            "energy_preset": style.energy,
            "f0_median_hz": round(float(np.nanmean([f for f, _ in f0s])), 1),
            "f0_sd_st": round(float(np.nanmean([s for _, s in f0s])), 2),
            "speech_s": round(float(np.mean([speech_s(p, sr) for p in parts])), 3),
            "rms_db": round(float(np.mean([rms_db(p) for p in parts])), 2),
            "synth_s_median": round(statistics.median(synth_s), 3),
        }
    neutral = per_style["neutral"]
    for row in per_style.values():
        row["rate_vs_neutral"] = round(neutral["speech_s"] / row["speech_s"], 3)
        row["pitch_vs_neutral_st"] = round(12 * np.log2(row["f0_median_hz"] / neutral["f0_median_hz"]), 2)
        row["loudness_vs_neutral_db"] = round(row["rms_db"] - neutral["rms_db"], 2)

    OUT.mkdir(parents=True, exist_ok=True)
    timeline, stitched, t = [], [], 0.0
    for name in STYLES:
        clip = demo_audio[name][0]
        timeline.append({"style": name, "start_s": round(t, 2), "duration_s": round(len(clip) / sr, 2)})
        stitched.append(clip)
        stitched.append(np.zeros(int(STYLE_GAP_S * sr), dtype=np.float32))
        t += len(clip) / sr + STYLE_GAP_S
    styles_wav = np.concatenate(stitched[:-1])
    if len(styles_wav) / sr > MAX_DEMO_S:
        raise SystemExit(f"[{engine_name}] style demo is {len(styles_wav) / sr:.1f} s, over {MAX_DEMO_S} s")
    sf.write(OUT / f"demo_styles_{engine_name}.wav", styles_wav, sr)
    (OUT / f"demo_styles_{engine_name}.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")

    conv_segments, conv_parts = [], []
    for seg_style, seg_text in parse_styled(CONVERSATION[lang], Style()):
        conv_segments.append({"emotion": seg_style.emotion, "rate": seg_style.rate, "text": seg_text})
        conv_parts.append(eng.synth(seg_text, seg_style))
    conv_wav = join(conv_parts, sr, SEGMENT_GAP_S)
    if len(conv_wav) / sr > MAX_DEMO_S:
        raise SystemExit(f"[{engine_name}] conversation demo is {len(conv_wav) / sr:.1f} s, over {MAX_DEMO_S} s")
    sf.write(OUT / f"demo_conversation_{engine_name}.wav", conv_wav, sr)

    if not a.skip_wer:
        wer_dir = OUT / "prosody_wer" / engine_name
        wer_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for name in STYLES:
            for item in load_sentences(lang):
                path = wer_dir / f"{name}_{item['id']}.wav"
                sf.write(path, eng.synth(item["text"], STYLE_PRESETS[name]), sr)
                rows.append({"style": name, "id": item["id"], "text": item["text"], "wav_path": str(path)})
        stt_round_trip(rows, lang)
        for name in STYLES:
            wers = [r["wer"] for r in rows if r["style"] == name and "wer" in r]
            per_style[name]["wer_mean"] = round(statistics.mean(wers), 3)
            per_style[name]["wer_median"] = round(statistics.median(wers), 3)
        for row in per_style.values():
            row["wer_mean_delta_vs_neutral"] = round(row["wer_mean"] - neutral["wer_mean"], 3)
            row["wer_gate_ok"] = row["wer_mean_delta_vs_neutral"] <= 0.05

    loadavg_end = [round(x, 2) for x in os.getloadavg()]
    doc = {
        "engine": engine_name,
        "lang": lang,
        "sample_rate": sr,
        "args": vars(a),
        "loadavg_start": loadavg_start,
        "loadavg_end": loadavg_end,
        "contaminated": loadavg_start[0] > 2.0,
        "demo_sentences": DEMO[lang],
        "demo_styles_s": round(len(styles_wav) / sr, 2),
        "conversation": {"text": CONVERSATION[lang], "segments": conv_segments,
                         "duration_s": round(len(conv_wav) / sr, 2)},
        "metric_definitions": {
            "f0_median_hz": "pyin median over voiced frames, mean over demo sentences",
            "f0_sd_st": "SD of voiced F0 in semitones, mean over demo sentences",
            "rate_vs_neutral": "neutral trimmed speech duration / style trimmed speech duration (>1 = faster)",
            "loudness_vs_neutral_db": "RMS dB of style audio minus RMS dB of neutral",
            "synth_s_median": "wall synth time per demo sentence (timing; load-sensitive)",
            "wer_mean": "Whistle round-trip on the bench sentences in the set language, post-processed audio",
            "wer_gate_ok": "mean WER within +0.05 of neutral",
        },
        "styles": per_style,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"prosody_{engine_name}.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{engine_name}] style     rate  pitch_st  f0_hz  f0_sd  loud_db  wer_mean  wer_d  gate", flush=True)
    for name, r in per_style.items():
        wer = (f"{r['wer_mean']:8.3f} {r['wer_mean_delta_vs_neutral']:+6.3f} {r['wer_gate_ok']}"
               if "wer_mean" in r else "-")
        print(f"[{engine_name}] {name:9s} {r['rate_vs_neutral']:5.2f} {r['pitch_vs_neutral_st']:+8.2f} "
              f"{r['f0_median_hz']:6.1f} {r['f0_sd_st']:6.2f} {r['loudness_vs_neutral_db']:+7.2f}  {wer}", flush=True)
    print(f"[{engine_name}] demo {len(styles_wav) / sr:.1f} s, conversation {len(conv_wav) / sr:.1f} s, "
          f"loadavg {loadavg_start} -> {loadavg_end}", flush=True)
    print(f"[{engine_name}] wrote {OUT / f'demo_styles_{engine_name}.wav'}", flush=True)
    print(f"[{engine_name}] wrote {OUT / f'demo_conversation_{engine_name}.wav'}", flush=True)
    print(f"[{engine_name}] wrote {RESULTS / f'prosody_{engine_name}.json'}", flush=True)


if __name__ == "__main__":
    main()
