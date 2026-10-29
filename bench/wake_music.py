#!/usr/bin/env python3
"""Reproducible wake-word-vs-music benchmark for the on-device "Hej Gaja" model.

Mixes real wake takes with music at several SNRs, plays the mix through the
device chain (speech scaled to the measured Atom mic level, then software gain 4 with
clipping and int16 quantisation), and scores it with the deployed streaming tflite model
using ESPHome's detection rule (sliding-window average > probability cutoff, refractory).

Reports:
  * recall per SNR at cutoffs 0.961 / 0.9 / 0.8 / 0.7 / 0.6 / 0.5
  * false accepts per hour on music-only, ambient and hard-negative sets at the same cutoffs
  * device-captured sets scored as recorded (post-gain uplink audio): wake attempts over music
    ("incident" takes) and non-wake speech over music (false-accept set)
  * the same grid again with a WebRTC noise-suppression prefilter (webrtc-noise-gain),
    applied before the software gain the way a port to the device would sit

Output: bench/results/wake_music.json (gitignored) + markdown tables on stdout.

One-off runtime deps (never project deps):
    uv venv bench/.venv --python 3.11
    uv pip install --python bench/.venv/bin/python soundfile scipy tflite-runtime webrtc-noise-gain
    # microwakeword has packages without __init__.py, so install it editable from a clone
    git clone https://github.com/OHF-Voice/micro-wake-word <micro-wake-word checkout>
    uv pip install --python bench/.venv/bin/python -e <micro-wake-word checkout>

Run (recordings/logs/music are gitignored; point the flags at the checkout that has them):
    bench/.venv/bin/python bench/wake_music.py --music <music files...> \
        [--recordings <root>/data/wakeword/recordings] [--incident <globs>] [--room-speech <globs>]
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
STEP_MS = 10  # the micro frontend step the device uses
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CUTOFFS = [0.961, 0.9, 0.8, 0.7, 0.6, 0.5]
DEFAULT_SNR = [10, 5, 0, -5, -10]
DEFAULT_LEVEL_DBFS = -36.0  # measured loud-speech RMS on the Atom mic before software gain
QUIET_LEVEL_DBFS = -42.0  # median of the same takes: the quiet-speech sensitivity point
MUSIC_LOUD_DBFS = -30.0  # music-only false-accept level: clearly audible across the room
CHORD_ROOTS = [48, 45, 41, 43]  # C3 Am F2 G2-ish progression, MIDI notes


def _rng(key: str, seed: int) -> np.random.Generator:
    """Deterministic per-task generator: identical mixes on every run and for every filter."""
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return np.random.default_rng([seed, h])


# --- audio helpers (numpy only) ---------------------------------------------------------------------


def calc_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0


def calc_dbfs(x: np.ndarray) -> float:
    rms = calc_rms(x)
    return 20 * math.log10(rms) if rms > 1e-12 else -120.0


def active_rms(x: np.ndarray, sr: int = SR, frame_s: float = 0.02, rel_db: float = -20.0) -> float:
    """RMS over speech-active frames (within rel_db of the loudest frame), so leading/trailing
    room silence does not drag the level reference down."""
    n = max(1, int(sr * frame_s))
    if x.size < n:
        return calc_rms(x)
    frames = x[: x.size // n * n].reshape(-1, n).astype(np.float64)
    db = 10 * np.log10((frames ** 2).mean(axis=1) + 1e-12)
    active = frames[db >= db.max() + rel_db]
    return float(np.sqrt((active ** 2).mean()))


def scale_active_dbfs(x: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Scale so the active-speech RMS equals target_dbfs (the level real speech arrives at)."""
    rms = active_rms(x)
    if rms <= 1e-12:
        return x.astype(np.float32, copy=True)
    return np.clip(x * (10 ** (target_dbfs / 20.0) / rms), -1.0, 1.0).astype(np.float32)


def read_mono(path: Path, sr: int = SR) -> np.ndarray:
    """Read any audio file as float32 mono at sr (soundfile first, ffmpeg fallback)."""
    try:
        data, rate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        import io
        import subprocess as sp

        out = sp.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1",
                      "-ar", str(sr), "-f", "wav", "-"], check=True, capture_output=True).stdout
        data, rate = sf.read(io.BytesIO(out), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if rate != sr:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(rate), sr)
        data = resample_poly(data, sr // g, rate // g)
    return data.astype(np.float32)


def device_gain_chain(x: np.ndarray, gain: float = 4.0) -> np.ndarray:
    """The device's software gain_factor on int16 PCM: multiply with saturation, then the int16
    grid. This is the last stage before the wake model, so the bench reproduces it exactly."""
    y = np.clip(x * gain, -1.0, 1.0)
    return np.round(y * 32767.0).astype(np.float32) / 32767.0


def quantise_cutoff(c: float) -> float:
    """ESPHome stores int(cutoff*255)/255; detection compares against that, not the raw float."""
    return int(c * 255) / 255.0


def cutoff_key(c: float) -> str:
    return f"{c:g}"


# --- music ------------------------------------------------------------------------------------------


def _fade_edges(x: np.ndarray, ms: float = 20.0) -> np.ndarray:
    f = min(len(x) // 2, int(SR * ms / 1000))
    if f:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        x[:f] *= ramp
        x[-f:] *= ramp[::-1]
    return x


def music_segment(src: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random n-sample window of src, tiled with faded seams when the source is shorter."""
    if len(src) >= n:
        start = int(rng.integers(0, len(src) - n + 1))
        return src[start:start + n].copy()
    pieces: list[np.ndarray] = []
    total = 0
    start = int(rng.integers(0, len(src)))
    while total < n:
        piece = src[start:].copy() if total == 0 else src.copy()
        _fade_edges(piece)
        pieces.append(piece)
        total += len(piece)
        start = 0
    return np.concatenate(pieces)[:n].astype(np.float32)


def synth_music_block(seconds: float, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Music-like bed: sine-chord progression with bass, kick/snare/hat noise hits and a soft
    noise floor. Clearly synthetic (labelled as such in the results), but it exercises the same
    harmonic-plus-percussive structure real music has."""
    n = int(seconds * sr)
    out = np.zeros(n, np.float64)
    beat = 60.0 / 108.0  # 108 BPM
    # chords: one pad per 2 bars with a slow swell at the edges
    pad_len = int(8 * beat * sr)
    for k in range(int(math.ceil(n / pad_len)) if pad_len else 1):
        i0 = k * pad_len
        if i0 >= n:
            break
        ln = min(pad_len, n - i0)
        seg = np.arange(ln) / sr
        swell = np.ones(ln)
        fade = min(ln // 4, int(0.4 * sr))
        if fade:
            swell[:fade] = np.linspace(0, 1, fade)
            swell[-fade:] = np.linspace(1, 0, fade)
        root = CHORD_ROOTS[k % len(CHORD_ROOTS)]
        members = [root, root + (4 if k % 2 else 3), root + 7, root + 12]
        for m in members:
            f0 = 440.0 * 2 ** ((m - 69) / 12)
            for h, a in ((1, 1.0), (2, 0.35), (3, 0.15)):
                det = 1.0 + 0.003 * ((h % 2) - 0.5)
                out[i0:i0 + ln] += a * 0.05 * np.sin(2 * np.pi * f0 * h * det * seg + h) * swell
    # bass: plucked root an octave down on every beat, following the pad's 2-bar changes
    beat_n = int(beat * sr)
    for k in range(0, int(math.ceil(n / beat_n))):
        s0 = k * beat_n
        if s0 >= n:
            break
        ln = min(int(0.4 * sr), n - s0)
        tt = np.arange(ln) / sr
        root = CHORD_ROOTS[(k // 8) % len(CHORD_ROOTS)]  # pads change every 8 beats
        f0 = 440.0 * 2 ** ((root - 12 - 69) / 12)
        out[s0:s0 + ln] += 0.16 * np.sin(2 * np.pi * f0 * tt) * np.exp(-tt * 6)
    # drums: kick on beats, snare on 2/4, hats on eighths
    half = beat_n // 2
    for k in range(0, int(math.ceil(n / half))):
        s0 = k * half
        if s0 >= n:
            break
        if k % 2 == 0:
            ln = min(int(0.14 * sr), n - s0)
            tt = np.arange(ln) / sr
            out[s0:s0 + ln] += 0.5 * np.sin(2 * np.pi * 55 * tt) * np.exp(-tt * 18)
        if k % 8 in (2, 6):
            ln = min(int(0.11 * sr), n - s0)
            hp = np.diff(rng.standard_normal(ln), prepend=0.0)  # crude highpass snare snap
            out[s0:s0 + ln] += 0.10 * hp * np.exp(-np.arange(ln) / sr * 22)
        ln = min(int(0.03 * sr), n - s0)
        out[s0:s0 + ln] += 0.04 * rng.standard_normal(ln) * np.exp(-np.arange(ln) / sr * 90)
    out += 0.004 * rng.standard_normal(n)  # room noise floor
    peak = np.abs(out).max()
    if peak > 0.7:
        out *= 0.7 / peak
    return out.astype(np.float32)


# --- model scoring (replicates wakeword/gaja_ww/evaluate.py's Detector) ------------------------------


class Detector:
    """Streaming quantised model scoring: 1.5 s state flush before, sliding-window average,
    1.0 s tail after a clip, exactly like the evaluation harness the cutoff was chosen with."""

    def __init__(self, tflite: Path, window: int = 5):
        from microwakeword.inference import Model

        self.model = Model(str(tflite))
        self.window = window
        self.step_s = 0.01 * self.model.stride  # frame step after the stride (30 ms for stride 3)

    def probs(self, x: np.ndarray, flush: bool = True) -> np.ndarray:
        # 1.5 s of leading silence resets the streaming state between clips (the training harness
        # does the same); flush=False scores a continuous stream, e.g. music priming the front end
        if flush:
            x = np.concatenate([np.zeros(int(1.5 * SR), np.float32), x.astype(np.float32)])
        p = np.asarray(self.model.predict_clip(x, step_ms=STEP_MS), dtype=np.float32)
        skip = int(1.5 / self.step_s) if flush else 0
        return p[skip:] if len(p) > skip else p

    def smooth(self, p: np.ndarray) -> np.ndarray:
        w = self.window
        if len(p) < w:
            return np.zeros(1, np.float32) if len(p) == 0 else np.array([p.mean()], np.float32)
        c = np.cumsum(np.concatenate([[0.0], p]))
        return ((c[w:] - c[:-w]) / w).astype(np.float32)

    def clip_score(self, x: np.ndarray) -> float:
        tail = np.zeros(int(1.0 * SR), np.float32)  # let the model finish after the phrase
        return float(self.smooth(self.probs(np.concatenate([x, tail]))).max())


def count_events(avg: np.ndarray, cutoffs: list[float], refractory_s: float, step_s: float) -> dict[str, int]:
    """Greedy detection count per cutoff with a refractory period (ESPHome-style rule)."""
    gap = max(1, int(refractory_s / step_s))
    out: dict[str, int] = {}
    for c in cutoffs:
        q = quantise_cutoff(c)
        last, n = -10 ** 9, 0
        for i in np.flatnonzero(avg > q):
            if i - last >= gap:
                n += 1
                last = int(i)
        out[cutoff_key(c)] = n
    return out


# --- prefilter ---------------------------------------------------------------------------------------


class WebRtcNS:
    """WebRTC noise suppression (webrtc-noise-gain: APM NS + high-pass) on 10 ms int16 frames —
    the same form a wake-word port would run on the ESP32-S3. Applied before software gain."""

    def __init__(self, level: int = 2):
        from webrtc_noise_gain import AudioProcessor

        self.proc = AudioProcessor(noise_suppression=level)
        self.frame = 160  # 10 ms at 16 kHz

    def __call__(self, x: np.ndarray) -> np.ndarray:
        n16 = np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
        pad = (-len(n16)) % self.frame
        if pad:
            n16 = np.concatenate([n16, np.zeros(pad, np.int16)])
        out = np.empty(len(n16), np.int16)
        for i in range(0, len(n16), self.frame):
            chunk = self.proc.Process10ms(n16[i:i + self.frame].tobytes())
            out[i:i + self.frame] = np.frombuffer(chunk.audio, dtype=np.int16)
        return out[: len(out) - pad].astype(np.float32) / 32767.0


def make_filter(name: str, ns_level: int):
    return WebRtcNS(ns_level) if name.startswith("webrtc") else None


# --- data discovery ----------------------------------------------------------------------------------


def _session_mic(session_dir: Path) -> str:
    """'atom' when the session was recorded through the device mic, 'interface' otherwise."""
    meta = session_dir / "meta.jsonl"
    if meta.exists():
        for line in meta.read_text(encoding="utf-8", errors="ignore").splitlines()[:50]:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            mic = str(row.get("mic", ""))
            if mic:
                return "atom" if "usb uac" in mic.lower() else "interface"
    return "unknown"


def find_recordings(root: Path) -> dict[str, list[dict]]:
    sets: dict[str, list[dict]] = {"wake": [], "hard_negative": [], "ambient": []}
    if not root.exists():
        return sets
    for session in sorted(p for p in root.iterdir() if p.is_dir()):
        mic = _session_mic(session)
        for label in sets:
            for wav in sorted((session / label).rglob("*.wav")):
                prompt = wav.parent.name if wav.parent != session / label else ""
                sets[label].append({"path": wav, "session": session.name, "mic": mic, "prompt": prompt})
    return sets


# --- measurements ------------------------------------------------------------------------------------


def recall_table(det: Detector, clips: list[dict], snrs: list[float | None], levels: list[float],
                 filters: list[str], ns_level: int, music_pool: list[tuple[str, np.ndarray]],
                 cutoffs: list[float], gain: float, seed: int) -> dict:
    """clip_score for every (filter, level, SNR, take); speech scaled to the level, music mixed
    at SNR against the active-speech RMS, then the device chain. Same seed -> same mix for every
    filter, so filter deltas are not noise."""
    snr_key = lambda s: "clean" if s is None else f"{s:+g}"
    out: dict = {}
    for fname in filters:
        ns = make_filter(fname, ns_level)
        for level in levels:
            scores: dict[str, list[float]] = {snr_key(s): [] for s in snrs}
            by_mic: dict[str, dict[str, list[float]]] = {}
            for it in clips:
                speech = scale_active_dbfs(read_mono(it["path"]), level)
                for s in snrs:
                    rng = _rng(f"{it['path'].name}:{level}:{snr_key(s)}", seed)
                    if s is None:
                        mix = speech
                    else:
                        seg = music_segment(music_pool[int(rng.integers(0, len(music_pool)))][1],
                                            len(speech), rng)
                        seg *= calc_rms(speech) * 10 ** (-s / 20.0) / max(calc_rms(seg), 1e-12)
                        mix = speech + seg
                    if ns is not None:
                        mix = ns(mix)
                    sc = det.clip_score(device_gain_chain(mix, gain))
                    key = snr_key(s)
                    scores[key].append(sc)
                    by_mic.setdefault(it["mic"], {}).setdefault(key, []).append(sc)
            out[f"{fname}@{level:g}"] = {
                "n": len(clips),
                "scores": {k: [float(v) for v in vals] for k, vals in scores.items()},
                "by_mic": {m: {k: [float(v) for v in vals] for k, vals in d.items()} for m, d in by_mic.items()},
            }
            print(f"  scored {fname} @ {level:g} dBFS: {len(clips)} takes x {len(snrs)} conditions", flush=True)
    return out


def score_captures(det: Detector, files: list[Path], filters: list[str], ns_level: int,
                   cutoffs: list[float], label: str, preroll_pool: list[tuple[str, np.ndarray]] | None = None,
                   preroll_dbfs: float = -21.0, preroll_s: float = 30.0, seed: int = 0) -> dict:
    """Score device-captured clips (post-gain uplink audio) as recorded: no re-gain, no
    re-levelling — the capture already is what the model saw.

    Two protocols per capture:
      * fresh: 1.5 s silence flush then the clip (the training harness's clip scoring)
      * primed: ``preroll_s`` of music from the pool at the device level precedes the clip and the
        score is read only over the clip's frames — the live condition, where the front end has
        already been fed music before the phrase arrives
    """
    out: dict = {}
    for fname in filters:
        ns = make_filter(fname, ns_level)
        fresh: dict[str, float] = {}
        primed: dict[str, float] = {}
        hours = 0.0
        for f in files:
            x = read_mono(f)
            hours += len(x) / SR / 3600
            if ns is not None:
                x = ns(x)
            fresh[f.name] = det.clip_score(x)
            if preroll_pool:
                rng = _rng(f"preroll:{f.name}", seed)
                pre = music_segment(preroll_pool[int(rng.integers(0, len(preroll_pool)))][1],
                                    int(preroll_s * SR), rng)
                pre *= 10 ** (preroll_dbfs / 20.0) / max(calc_rms(pre), 1e-12)
                tail = np.zeros(int(1.0 * SR), np.float32)
                stream = np.concatenate([pre, x, tail])
                avg = det.smooth(det.probs(stream, flush=False))
                i0 = int(round(len(pre) / SR / det.step_s))
                i1 = i0 + int(round(len(x) / SR / det.step_s)) + 5  # a few frames of slack
                primed[f.name] = float(avg[max(0, i0 - 2):min(len(avg), i1)].max())
        arr = np.asarray(list(primed.values())) if primed else np.asarray(list(fresh.values()))
        out[fname] = {"n": len(fresh), "hours": round(hours, 4),
                      "scores": {k: round(float(v), 4) for k, v in fresh.items()},
                      "primed_scores": {k: round(float(v), 4) for k, v in primed.items()},
                      "rate": {cutoff_key(c): float((arr > quantise_cutoff(c)).mean()) for c in cutoffs},
                      "fa_per_hour_equiv": {cutoff_key(c): (float((arr > quantise_cutoff(c)).mean()) * len(fresh) / hours
                                                           if hours else None) for c in cutoffs}}
        peak = max(primed.values()) if primed else max(fresh.values())
        print(f"  {label} {fname}: {len(fresh)} captures, best score {peak:.2f}", flush=True)
    return out


def fa_music(det: Detector, pool: list[tuple[str, np.ndarray]], minutes: float, real_minutes: float,
             levels: list[float], filters: list[str], ns_level: int, cutoffs: list[float],
             refractory: float, gain: float, block_s: float, seed: int) -> dict:
    """False accepts on assembled music-only streams: the real pool first, then a synthetic bed.
    Each level reuses identical block content, so filter and level deltas stay comparable."""
    n_blocks = int(math.ceil(minutes * 60 / block_s))
    real_blocks = int(round(real_minutes * 60 / block_s))
    out: dict = {}
    for fname in filters:
        ns = make_filter(fname, ns_level)
        out[fname] = {}
        for level in levels:
            tally = {kind: {"events": {cutoff_key(c): 0 for c in cutoffs}, "hours": 0.0}
                     for kind in ("real", "synthetic")}
            for k in range(n_blocks):
                kind = "real" if k < real_blocks else "synthetic"
                rng = _rng(f"block:{k}", seed)
                if kind == "real":
                    x = music_segment(pool[int(rng.integers(0, len(pool)))][1], int(block_s * SR), rng)
                else:
                    x = synth_music_block(block_s, rng)
                x *= 10 ** (level / 20.0) / max(calc_rms(x), 1e-12)
                if ns is not None:
                    x = ns(x)
                ev = count_events(det.smooth(det.probs(device_gain_chain(x, gain))), cutoffs,
                                  refractory, det.step_s)
                h = len(x) / SR / 3600
                tally[kind]["hours"] += h
                for c, v in ev.items():
                    tally[kind]["events"][c] += v
            all_h = sum(t["hours"] for t in tally.values())
            all_e = {c: sum(t["events"][c] for t in tally.values()) for c in [cutoff_key(c) for c in cutoffs]}
            for kind, t in tally.items():
                t["fa_per_hour"] = {c: (v / t["hours"] if t["hours"] else None) for c, v in t["events"].items()}
            out[fname][f"{level:g}"] = {**tally,
                                        "all": {"events": all_e, "hours": round(all_h, 3),
                                                "fa_per_hour": {c: (v / all_h if all_h else None)
                                                                for c, v in all_e.items()}}}
            print(f"  music-only {fname} @ {level:g} dBFS done", flush=True)
    return out


def fa_ambient(det: Detector, files: list[Path], filters: list[str], ns_level: int,
               cutoffs: list[float], refractory: float, gain: float) -> dict:
    """False accepts per hour on long room recordings, scored file by file like
    the training harness does (each file gets a fresh streaming state)."""
    out: dict = {}
    for fname in filters:
        ns = make_filter(fname, ns_level)
        ev = {cutoff_key(c): 0 for c in cutoffs}
        hours = 0.0
        for f in files:
            x = read_mono(f)
            hours += len(x) / SR / 3600
            if ns is not None:
                x = ns(x)
            evs = count_events(det.smooth(det.probs(device_gain_chain(x, gain))), cutoffs,
                               refractory, det.step_s)
            for c, v in evs.items():
                ev[c] += v
        out[fname] = {"events": ev, "hours": round(hours, 3),
                      "fa_per_hour": {c: (v / hours if hours else None) for c, v in ev.items()}}
        print(f"  ambient {fname}: {hours:.2f} h, events {ev}", flush=True)
    return out


def fa_clips(det: Detector, clips: list[dict], filters: list[str], ns_level: int,
             cutoffs: list[float], gain: float) -> dict:
    """Near-miss speech takes: fraction triggering at each cutoff, plus an hours-equivalent rate
    (one event per triggering take over the total take audio) for comparison with the FA/h rows."""
    out: dict = {}
    for fname in filters:
        ns = make_filter(fname, ns_level)
        scores: list[float] = []
        hours = 0.0
        for it in clips:
            x = read_mono(it["path"])
            hours += len(x) / SR / 3600
            if ns is not None:
                x = ns(x)
            scores.append(det.clip_score(device_gain_chain(x, gain)))
        arr = np.asarray(scores)
        by_mic: dict[str, list[float]] = {}
        for it, sc in zip(clips, scores):
            by_mic.setdefault(it["mic"], []).append(sc)
        out[fname] = {
            "n": len(clips), "hours": round(hours, 4),
            "rate": {cutoff_key(c): float((arr > quantise_cutoff(c)).mean()) for c in cutoffs},
            "fa_per_hour_equiv": {cutoff_key(c): (float((arr > quantise_cutoff(c)).mean()) * len(clips) / hours
                                                  if hours else None) for c in cutoffs},
            "by_mic_rate": {m: {cutoff_key(c): float((np.asarray(v) > quantise_cutoff(c)).mean())
                                for c in cutoffs} for m, v in by_mic.items()},
        }
        print(f"  hard-negative {fname}: {len(clips)} takes", flush=True)
    return out


# --- reporting ---------------------------------------------------------------------------------------


def _cell(v, kind: str) -> str:
    if v is None:
        return "-"
    return f"{v:.2f}" if kind == "fa" else f"{v:.3f}"


def recall_row(scores, cutoffs) -> str:
    arr = np.asarray(scores)
    return " | ".join(_cell(float((arr > quantise_cutoff(c)).mean()), "rate") for c in cutoffs)


def print_recall(recall: dict, cutoffs: list[float], title: str) -> None:
    print(f"\n## {title}\n")
    print("| SNR dB | speech level | filter | n | " + " | ".join(f"@{c:g}" for c in cutoffs) + " |")
    print("|---|---|---|---|" + "---|" * len(cutoffs))
    for fkey in sorted(recall):
        fname, level = fkey.split("@")
        res = recall[fkey]
        for skey, scores in res["scores"].items():
            print(f"| {skey} | {level} dBFS | {fname} | {len(scores)} | {recall_row(scores, cutoffs)} |")


def print_recall_by_mic(recall: dict, cutoffs: list[float]) -> None:
    print("\n## Recall by microphone (atom = the device mic)\n")
    print("| mic | SNR dB | speech level | filter | n | " + " | ".join(f"@{c:g}" for c in cutoffs) + " |")
    print("|---|---|---|---|---|" + "---|" * len(cutoffs))
    for fkey in sorted(recall):
        fname, level = fkey.split("@")
        for mic, per in sorted(recall[fkey]["by_mic"].items()):
            for skey, scores in per.items():
                print(f"| {mic} | {skey} | {level} dBFS | {fname} | {len(scores)} | "
                      f"{recall_row(scores, cutoffs)} |")


def print_fa(fa_music_res: dict, fa_ambient_res: dict, fa_clips_res: dict, room_speech_res: dict,
             cutoffs: list[float]) -> None:
    print("\n## False accepts per hour (device quantised cutoffs)\n")
    print("| set | filter | hours / n | " + " | ".join(f"@{c:g}" for c in cutoffs) + " |")
    print("|---|---|---|" + "---|" * len(cutoffs))
    for level in sorted({lv for per in fa_music_res.values() for lv in per}):
        for fname in sorted(fa_music_res):
            t = fa_music_res[fname].get(level)
            if not t:
                continue
            cells = " | ".join(_cell(t["all"]["fa_per_hour"][cutoff_key(c)], "fa") for c in cutoffs)
            print(f"| music only {level} dBFS | {fname} | {t['all']['hours']:.2f} h | {cells} |")
    for fname in sorted(fa_ambient_res):
        t = fa_ambient_res[fname]
        cells = " | ".join(_cell(t["fa_per_hour"][cutoff_key(c)], "fa") for c in cutoffs)
        print(f"| ambient room | {fname} | {t['hours']:.2f} h | {cells} |")
    for fname in sorted(room_speech_res):
        t = room_speech_res[fname]
        cells = " | ".join(_cell(t["rate"][cutoff_key(c)], "rate") for c in cutoffs)
        print(f"| room speech over music, share triggering | {fname} | n={t['n']} | {cells} |")
        cells = " | ".join(_cell(t["fa_per_hour_equiv"][cutoff_key(c)], "fa") for c in cutoffs)
        print(f"| room speech over music, FA/h if looped | {fname} | {t['hours']:.2f} h | {cells} |")
    for fname in sorted(fa_clips_res):
        t = fa_clips_res[fname]
        cells = " | ".join(_cell(t["rate"][cutoff_key(c)], "rate") for c in cutoffs)
        print(f"| hard negatives, share triggering | {fname} | n={t['n']} | {cells} |")
        cells = " | ".join(_cell(t["fa_per_hour_equiv"][cutoff_key(c)], "fa") for c in cutoffs)
        print(f"| hard negatives, FA/h if looped | {fname} | {t['hours']:.2f} h | {cells} |")


def print_incident(incident_res: dict, cutoffs: list[float]) -> None:
    print("\n## Incident takes: wake attempts over music, captured by the device (as-is)\n")
    print("fresh = silence flush (clip protocol); primed = 30 s of music before the phrase (live-like)\n")
    print("| capture | filter | fresh | primed | " + " | ".join(f"@{c:g}" for c in cutoffs) + " |")
    print("|---|---|---|---|" + "---|" * len(cutoffs))
    names = sorted({n for per in incident_res.values() for n in per["primed_scores"]})
    for name in names:
        for fname in sorted(incident_res):
            sc = incident_res[fname]["primed_scores"].get(name)
            if sc is None:
                continue
            fr = incident_res[fname]["scores"].get(name, 0.0)
            hits = " | ".join("yes" if sc > quantise_cutoff(c) else "no" for c in cutoffs)
            print(f"| {name[:24]} | {fname} | {fr:.3f} | {sc:.3f} | {hits} |")


def print_music_split(fa_music_res: dict, cutoffs: list[float]) -> None:
    print("\n## Music-only split: real pool vs synthetic bed\n")
    print("| part | level dBFS | filter | hours | " + " | ".join(f"@{c:g}" for c in cutoffs) + " |")
    print("|---|---|---|---|" + "---|" * len(cutoffs))
    for level in sorted({lv for per in fa_music_res.values() for lv in per}):
        for fname in sorted(fa_music_res):
            t = fa_music_res[fname].get(level)
            if not t:
                continue
            for kind in ("real", "synthetic"):
                cells = " | ".join(_cell(t[kind]["fa_per_hour"][cutoff_key(c)], "fa") for c in cutoffs)
                print(f"| {kind} | {level} dBFS | {fname} | {t[kind]['hours']:.2f} h | {cells} |")


# --- main --------------------------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Wake-word vs music benchmark")
    ap.add_argument("--recordings", type=Path, default=Path("data/wakeword/recordings"),
                    help="recordings root with <session>/{wake,hard_negative,ambient}/ (gitignored)")
    ap.add_argument("--model", type=Path, default=Path("firmware/esphome/models/hej_gaja.tflite"))
    ap.add_argument("--music", type=Path, nargs="+", required=True,
                    help="music files for the mixing pool (local data, not in the repo)")
    ap.add_argument("--room-music", type=str, nargs="+", default=[],
                    help="globs of non-speech room-music clips for the mixing pool (optional)")
    ap.add_argument("--incident", type=str, nargs="+", default=[],
                    help="globs of device-captured wake attempts over music (scored as-is, post-gain)")
    ap.add_argument("--room-speech", type=str, nargs="+", default=[],
                    help="globs of device-captured non-wake speech over music (false-accept set)")
    ap.add_argument("--minutes", type=float, default=30.0, help="total music-only eval time")
    ap.add_argument("--real-minutes", type=float, default=10.0,
                    help="share of --minutes filled by the real music pool (rest is synthetic)")
    ap.add_argument("--speech-dbfs", type=float, default=DEFAULT_LEVEL_DBFS)
    ap.add_argument("--quiet-dbfs", type=float, default=QUIET_LEVEL_DBFS)
    ap.add_argument("--music-loud-dbfs", type=float, default=MUSIC_LOUD_DBFS)
    ap.add_argument("--snr", type=float, nargs="+", default=DEFAULT_SNR)
    ap.add_argument("--cutoffs", type=float, nargs="+", default=DEFAULT_CUTOFFS)
    ap.add_argument("--filters", nargs="+", default=["none", "webrtc"],
                    help="prefilters to compare: none, webrtc (webrtc-noise-gain)")
    ap.add_argument("--ns-level", type=int, default=2)
    ap.add_argument("--gain", type=float, default=4.0, help="device software gain_factor")
    ap.add_argument("--window", type=int, default=5, help="ESPHome sliding_window_size")
    ap.add_argument("--refractory", type=float, default=2.0)
    ap.add_argument("--block-s", type=float, default=30.0, help="music stream block length in s")
    ap.add_argument("--preroll-dbfs", type=float, default=-21.0,
                    help="device-level RMS of the music pre-roll for the primed capture protocol "
                         "(≈ -33 dBFS in the room before software gain)")
    ap.add_argument("--preroll-s", type=float, default=30.0, help="music pre-roll length in s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("bench/results/wake_music.json"))
    args = ap.parse_args()

    for f in args.music:
        if not f.exists():
            raise SystemExit(f"missing music file: {f} (point --music at the checkout that has it)")
    room_files: list[Path] = []
    for pat in args.room_music:
        found = sorted(Path(p) for p in glob.glob(pat))  # glob module: absolute patterns work too
        if not found:
            print(f"warning: --room-music pattern matched nothing: {pat}", flush=True)
        room_files += found

    def resolve_globs(patterns: list[str]) -> list[Path]:
        files: list[Path] = []
        for pat in patterns:
            found = sorted(Path(p) for p in glob.glob(pat))
            if not found:
                print(f"warning: pattern matched nothing: {pat}", flush=True)
            files += found
        return files

    incident_files = resolve_globs(args.incident)
    room_speech_files = resolve_globs(args.room_speech)
    if not args.model.exists():
        raise SystemExit(f"missing model: {args.model}")

    music_pool = [(p.name, read_mono(p)) for p in list(args.music) + room_files]
    print(f"music pool: {[n for n, _ in music_pool]}", flush=True)
    try:
        os.nice(19 - os.nice(0))  # be nice to production co-tenants
    except (OSError, AttributeError):
        pass

    rec = find_recordings(args.recordings)
    if not rec["wake"]:
        raise SystemExit(f"no wake recordings under {args.recordings} (gitignored; pass --recordings)")
    det = Detector(args.model, args.window)
    speech_levels = [args.speech_dbfs, args.quiet_dbfs]
    music_levels = [args.speech_dbfs, args.music_loud_dbfs]
    snrs: list[float | None] = [None] + list(args.snr)

    print(f"\n=== recall: {len(rec['wake'])} wake takes, levels {speech_levels}, SNR {snrs}", flush=True)
    recall = recall_table(det, rec["wake"], snrs, speech_levels, args.filters, args.ns_level,
                          music_pool, args.cutoffs, args.gain, args.seed)

    print(f"\n=== false accepts: music-only {args.minutes:g} min "
          f"({args.real_minutes:g} real + rest synthetic)", flush=True)
    fa_m = fa_music(det, music_pool, args.minutes, args.real_minutes, music_levels, args.filters,
                    args.ns_level, args.cutoffs, args.refractory, args.gain, args.block_s, args.seed)

    print(f"\n=== false accepts: ambient ({len(rec['ambient'])} files)", flush=True)
    fa_a = fa_ambient(det, [it["path"] for it in rec["ambient"]], args.filters, args.ns_level,
                      args.cutoffs, args.refractory, args.gain)

    print(f"\n=== false accepts: hard negatives ({len(rec['hard_negative'])} takes)", flush=True)
    fa_h = fa_clips(det, rec["hard_negative"], args.filters, args.ns_level, args.cutoffs, args.gain)

    print(f"\n=== device captures: incident wake attempts over music ({len(incident_files)})", flush=True)
    incident = score_captures(det, incident_files, args.filters, args.ns_level, args.cutoffs,
                              "incident", preroll_pool=music_pool, preroll_dbfs=args.preroll_dbfs,
                              preroll_s=args.preroll_s, seed=args.seed) if incident_files else {}

    print(f"\n=== device captures: non-wake speech over music ({len(room_speech_files)})", flush=True)
    room_speech = score_captures(det, room_speech_files, args.filters, args.ns_level, args.cutoffs,
                                 "room speech", preroll_pool=music_pool, preroll_dbfs=args.preroll_dbfs,
                                 preroll_s=args.preroll_s, seed=args.seed) if room_speech_files else {}

    print_recall(recall, args.cutoffs, "Recall (wake takes + music at SNR)")
    print_recall_by_mic(recall, args.cutoffs)
    print_fa(fa_m, fa_a, fa_h, room_speech, args.cutoffs)
    print_music_split(fa_m, args.cutoffs)
    if incident:
        print_incident(incident, args.cutoffs)

    counts = {
        "wake": {m: sum(1 for it in rec["wake"] if it["mic"] == m)
                 for m in sorted({it["mic"] for it in rec["wake"]})},
        "hard_negative": {m: sum(1 for it in rec["hard_negative"] if it["mic"] == m)
                          for m in sorted({it["mic"] for it in rec["hard_negative"]})},
        "ambient_files": len(rec["ambient"]),
        "incident_captures": len(incident_files),
        "room_speech_captures": len(room_speech_files),
    }
    result = {
        "meta": {
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "git": _git_sha(),
            "model": str(args.model), "window": args.window, "gain": args.gain,
            "refractory_s": args.refractory, "step_s": det.step_s, "seed": args.seed,
            "snrs_db": args.snr, "speech_levels_dbfs": speech_levels, "music_levels_dbfs": music_levels,
            "cutoffs": args.cutoffs,
            "device_cutoffs": {cutoff_key(c): quantise_cutoff(c) for c in args.cutoffs},
            "ns_level": args.ns_level, "filters": args.filters,
            "music_minutes": args.minutes, "music_real_minutes": args.real_minutes,
            "music_sources": sorted(n for n, _ in music_pool),
            "synthetic_music": True, "counts": counts,
        },
        "recall": recall,
        "false_accepts": {"music_only": fa_m, "ambient": fa_a, "hard_negative": fa_h,
                          "room_speech_over_music": room_speech},
        "incident_captures": incident,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
