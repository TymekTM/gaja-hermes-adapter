#!/usr/bin/env python3
"""TTS benchmark for the Gaja voice adapter.

Metrics (definitions also written into each results JSON):
  RTF    = audio_seconds / synth_seconds, warm. >1 means faster than real time.
  TTFA   = seconds from text in to first playable audio chunk. Items are single
           utterances; an engine that yields audio per sentence (Piper) reports the
           time to its first chunk, the others report the full synth time.
  WER    = word error rate of a Whistle round-trip with forced language. Lower is better.
  PESQ   = torchaudio SQUIM objective speech quality score (1.0 to 4.5 scale). Proxy for MOS.
  SubjMOS= torchaudio SQUIM subjective MOS score against speaker reference.
Warm = discarded warm-up calls before timing. Cold model load is reported separately.
llama-tts (Qwen3-TTS) is a CLI that reloads the model on every call, so its synth_seconds
come from the llama-tts timing log (prompt + generation + vocoder) and exclude the reload.
"""
import argparse
import functools
import glob
import json
import os
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BENCH = ROOT / "bench"
OUT = BENCH / "out" / "tts"
RESULTS = BENCH / "results"
HF_HUB = Path.home() / ".cache/huggingface/hub"
LLAMA_BIN = ROOT / "third_party/llama.cpp/build/bin"
PIPER_DEFAULT = ROOT / "models/pl_PL-darkman-medium.onnx"

METRIC_DEFS = {
    "rtf": "audio_seconds / synth_seconds (warm). >1 = faster than real time.",
    "ttfa_s": "wall seconds from text in to first audio chunk (= synth_seconds for non-streaming engines).",
    "wer": "Whistle round-trip, language forced; text lowercased, numbers spelled out (num2words), punctuation stripped; jiwer WER.",
    "pesq": "torchaudio SQUIM objective PESQ score (proxy for subjective speech quality / MOS).",
    "subj_mos": "torchaudio SQUIM subjective MOS prediction against clean speaker reference.",
    "warm": "2 discarded warm-up calls before timing; cold model load reported separately.",
}


PROBE_PL = BENCH / "tts_probe_sentences_pl.txt"
QUIET_LOAD = 2.0


def load_sentences(lang, probe=False):
    path = PROBE_PL if probe else BENCH / f"sentences_{lang}.txt"
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    items = []
    for i, text in enumerate(lines, 1):
        n = len(text.split())
        cat = "short" if n <= 6 else "medium" if n <= 16 else "long"
        items.append({"id": f"{lang}{i:02d}", "text": text, "words": n, "cat": cat})
    return items


class GpuMon:
    def __init__(self):
        self.base = self._used()
        self.peak = self.base
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    @staticmethod
    def _used():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            return int(out.strip().splitlines()[0])
        except Exception:
            return 0

    def _run(self):
        while not self._stop.wait(0.5):
            self.peak = max(self.peak, self._used())

    def stop(self):
        self._stop.set()
        self._t.join()
        return {"baseline_used_mb": self.base, "peak_total_mb": self.peak, "peak_delta_mb": self.peak - self.base}


def rss_mb():
    self_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    child_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    return round(max(self_mb, child_mb))


class Paradee:
    streaming = "no"

    def __init__(self, a):
        from paradee import SAMPLE_RATE, Paradee as P
        self.sr = SAMPLE_RATE
        self.m = P(threads=a.threads)
        self.pl_espeak = a.paradee_pl_espeak
        self.device = f"cpu ({a.threads} thread{'s' if a.threads > 1 else ''})"
        if self.pl_espeak:
            from piper.phonemize_espeak import EspeakPhonemizer
            self.espeak = EspeakPhonemizer()
            self.device += ", pl via espeak-ng IPA"

    def phonemes_pl(self, text):
        sentences = self.espeak.phonemize("pl", text)
        return " ".join("".join(s) for s in sentences)

    def synth(self, text, lang):
        t0 = time.perf_counter()
        if lang == "pl" and self.pl_espeak:
            wav = self.m.generate_from_phonemes(self.phonemes_pl(text))
        else:
            wav = self.m(text)
        wav = np.asarray(wav, dtype=np.float32).squeeze()
        t = time.perf_counter() - t0
        return {"wav": wav, "sr": self.sr, "synth_s": t, "ttfa_s": t}


class ChatterboxTurbo:
    streaming = "no (generate() returns full utterance)"

    def __init__(self, a):
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        self.torch = torch
        self.half = a.cb_dtype == "fp16"
        self.device = a.cb_device
        t0 = time.perf_counter()
        tts = ChatterboxTurboTTS.from_pretrained(device="cpu" if self.half else a.cb_device)
        if self.half:
            for m in (tts.t3, tts.s3gen, tts.ve):
                m.half().to("cuda")
            if tts.conds is not None:
                tts.conds = tts.conds.to("cuda")
            tts.device = "cuda"
        if a.cb_cfm_steps != 2:
            orig_flow = tts.s3gen.inference

            def flow(*args, **kwargs):
                kwargs["n_cfm_timesteps"] = a.cb_cfm_steps
                return orig_flow(*args, **kwargs)
            tts.s3gen.inference = flow
        if a.cb_max_gen_len != 1000:
            tts.t3.inference_turbo = functools.partial(tts.t3.inference_turbo, max_gen_len=a.cb_max_gen_len)
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.load_s = time.perf_counter() - t0
        self.tts = tts
        self.sr = tts.sr
        self.device = f"cuda ({a.cb_dtype})" if self.device == "cuda" else "cpu (fp32)"

    def synth(self, text, lang):
        torch = self.torch
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
            wav = self.tts.generate(text)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t = time.perf_counter() - t0
        wav = np.asarray(wav.detach().float().cpu().numpy(), dtype=np.float32).squeeze()
        return {"wav": wav, "sr": self.sr, "synth_s": t, "ttfa_s": t}


class Piper:
    streaming = "sentence-level (production PiperTTS; items are single sentences, TTFA = their synth time)"

    def __init__(self, a):
        import asyncio
        from gaja.tts.piper_tts import PiperTTS
        model = Path(a.piper_model)
        t0 = time.perf_counter()
        self.engine = PiperTTS("piper", model, threads=a.piper_threads)
        self.load_s = time.perf_counter() - t0
        self.loop = asyncio.new_event_loop()
        self.sr = self.engine.sample_rate
        self.device = f"cpu (onnxruntime, threads={a.piper_threads}, production PiperTTS)"

    def synth(self, text, lang):
        from gaja.text.style import Style
        t0 = time.perf_counter()
        wav = self.loop.run_until_complete(self.engine.synthesize(text, Style(), lang))
        t = time.perf_counter() - t0
        return {"wav": np.asarray(wav, dtype=np.float32), "sr": self.sr, "synth_s": t, "ttfa_s": t}


class QwenCLI:
    streaming = "no (llama-tts writes WAV after the last frame; PCM is buffered internally)"

    def __init__(self, a):
        snap = sorted(glob.glob(str(HF_HUB / "models--mradermacher--Qwen3-TTS-Polish-GGUF/snapshots/*")))[-1]
        self.bin = LLAMA_BIN / "llama-tts"
        self.model = a.qwen_model or f"{snap}/Qwen3-TTS-Polish.Q4_K_S.gguf"
        self.mmproj = a.qwen_mmproj or f"{snap}/Qwen3-TTS-Polish.mmproj-Q8_0.gguf"
        self.a = a
        self.env = {**os.environ, "LD_LIBRARY_PATH": f"{LLAMA_BIN}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
        self.tmp = Path(tempfile.mkdtemp(prefix="qwen_tts_"))
        self.device = f"gpu ngl={a.qwen_ngl}, mmproj={'cpu' if a.qwen_mmproj_cpu else 'gpu'}"
        self.load_s = None
        self.load_estimates = []

    def tts_lang(self, lang):
        return self.a.qwen_lang_pl if lang == "pl" else "en"

    def synth(self, text, lang):
        out = self.tmp / "out.wav"
        cmd = [
            str(self.bin), "-m", self.model, "-mm", self.mmproj,
            "-ngl", str(self.a.qwen_ngl), "-c", str(self.a.qwen_ctx), "-n", str(self.a.qwen_n),
            "--tts-lang", self.tts_lang(lang),
        ]
        if self.a.qwen_mmproj_cpu:
            cmd.append("--no-mmproj-offload")
        if self.a.qwen_batch:
            cmd += ["-b", str(self.a.qwen_batch), "-ub", str(self.a.qwen_batch)]
        if self.a.qwen_speaker:
            cmd += ["--tts-speaker-file", self.a.qwen_speaker]
        cmd += ["-p", text, "--output", str(out)]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, env=self.env)
        wall = time.perf_counter() - t0
        log = p.stdout + p.stderr
        if p.returncode != 0:
            raise RuntimeError(f"llama-tts exit {p.returncode}: {log[-1500:]}")
        m = re.search(r"timings: prompt eval ([\d.]+)s \+ generation ([\d.]+)s \+ vocoder ([\d.]+)s = total ([\d.]+)s", log)
        if not m:
            raise RuntimeError(f"no timings in llama-tts log: {log[-1500:]}")
        synth = float(m.group(4))
        self.load_estimates.append(wall - synth)
        wav, sr = sf.read(out, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return {"wav": wav, "sr": sr, "synth_s": synth, "ttfa_s": synth, "wall_s": wall, "load_est_s": wall - synth}


class MMS:
    streaming = "no (full utterance synthesis)"

    def __init__(self, a):
        import torch
        from transformers import VitsModel, AutoTokenizer
        self.torch = torch
        self.device = a.mms_device
        t0 = time.perf_counter()
        model = VitsModel.from_pretrained("facebook/mms-tts-pol")
        if self.device == "cuda":
            model = model.cuda()
        self.model = model
        self.tok = AutoTokenizer.from_pretrained("facebook/mms-tts-pol")
        self.sr = model.config.sampling_rate
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.load_s = time.perf_counter() - t0
        self.device = f"cuda (fp32)" if self.device == "cuda" else "cpu (fp32)"

    def synth(self, text, lang):
        torch = self.torch
        inputs = self.tok(text, return_tensors="pt")
        if self.device.startswith("cuda"):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(**inputs).waveform
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t = time.perf_counter() - t0
        wav = np.asarray(out.squeeze().float().cpu().numpy(), dtype=np.float32)
        return {"wav": wav, "sr": self.sr, "synth_s": t, "ttfa_s": t}


class SupertonicEngine:
    streaming = "no (full utterance synthesis)"

    def __init__(self, a):
        import asyncio
        from gaja.tts.supertonic_tts import SupertonicTTS
        t0 = time.perf_counter()
        self.engine = SupertonicTTS(
            steps=a.supertonic_steps, voice=a.supertonic_style, speed=a.supertonic_speed,
            threads=a.supertonic_threads, output_sample_rate=a.supertonic_sr or None,
        )
        self.load_s = time.perf_counter() - t0
        self.loop = asyncio.new_event_loop()
        self.sr = self.engine.sample_rate
        self.device = (f"cpu (onnxruntime, threads={a.supertonic_threads}, steps={a.supertonic_steps}, "
                       f"voice={a.supertonic_style}, speed={a.supertonic_speed}, sr={self.sr})")

    def synth(self, text, lang):
        from gaja.text.style import Style
        t0 = time.perf_counter()
        wav = self.loop.run_until_complete(self.engine.synthesize(text, Style(), lang))
        t = time.perf_counter() - t0
        return {"wav": np.asarray(wav, dtype=np.float32), "sr": self.sr, "synth_s": t, "ttfa_s": t}


def build_engine(a):
    if a.engine == "paradee":
        return Paradee(a)
    if a.engine == "chatterbox-turbo":
        return ChatterboxTurbo(a)
    if a.engine == "piper":
        return Piper(a)
    if a.engine == "qwen-polish":
        return QwenCLI(a)
    if a.engine == "mms":
        return MMS(a)
    if a.engine == "supertonic":
        return SupertonicEngine(a)
    raise SystemExit(f"unknown engine {a.engine}")


def normalize(text, lang):
    import num2words
    t = unicodedata.normalize("NFC", text).lower()
    t = re.sub(r"(\d{1,2}):(\d{2})", lambda m: f"{num2words.num2words(int(m[1]), lang=lang)} {num2words.num2words(int(m[2]), lang=lang)}", t)
    t = re.sub(r"\d+", lambda m: num2words.num2words(int(m[0]), lang=lang), t)
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def stt_round_trip(rows, lang):
    import jiwer
    import torchaudio
    import torch
    from needle import Whistle
    stt = Whistle()
    for r in rows:
        if "error" in r:
            continue
        wav, sr = sf.read(r["wav_path"], dtype="float32")
        x = torch.from_numpy(wav).unsqueeze(0)
        x16 = torchaudio.functional.resample(x, sr, 16000).squeeze(0).numpy()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        sf.write(tmp, x16, 16000)
        try:
            res = stt.transcribe(tmp, language=lang)
        finally:
            os.unlink(tmp)
        hyp = res.get("text", "") if isinstance(res, dict) else str(res)
        r["stt_raw"] = hyp
        ref_n, hyp_n = normalize(r["text"], lang), normalize(hyp, lang)
        r["ref_norm"], r["hyp_norm"] = ref_n, hyp_n
        r["wer"] = 1.0 if not hyp_n else round(jiwer.wer(ref_n, hyp_n), 4)


def mos_round_trip(rows):
    import torch
    import torchaudio
    try:
        obj_model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()
        subj_model = torchaudio.pipelines.SQUIM_SUBJECTIVE.get_model()
        ref_path = BENCH / "out" / "tts" / "speaker_ref_piper.wav"
        ref_wav, rsr = None, 16000
        if ref_path.exists():
            ref_wav, rsr = torchaudio.load(str(ref_path))
            if rsr != 16000:
                ref_wav = torchaudio.functional.resample(ref_wav, rsr, 16000)

        for r in rows:
            if "error" in r or "wav_path" not in r:
                continue
            w, sr = torchaudio.load(r["wav_path"])
            if sr != 16000:
                w = torchaudio.functional.resample(w, sr, 16000)
            stoi, pesq, si_sdr = obj_model(w)
            r["pesq"] = round(pesq.item(), 3)
            r["stoi"] = round(stoi.item(), 3)
            r["si_sdr"] = round(si_sdr.item(), 3)
            if ref_wav is not None:
                try:
                    score = subj_model(w, ref_wav)
                    r["subj_mos"] = round(score.item(), 3)
                except Exception:
                    pass
    except Exception as e:
        print(f"MOS estimation skipped/failed: {e}", flush=True)


def stats(xs):
    if not xs:
        return None
    return {"median": round(statistics.median(xs), 3), "min": round(min(xs), 3), "max": round(max(xs), 3)}


def summarize(rows):
    failed = sum("error" in r for r in rows)
    rows = [r for r in rows if "error" not in r]
    cats = {}
    for c in ("short", "medium", "long"):
        sub = [r for r in rows if r["cat"] == c]
        cats[c] = {"n": len(sub), "rtf_median": round(statistics.median([r["rtf"] for r in sub]), 3) if sub else None,
                   "wer_median": round(statistics.median([r["wer"] for r in sub]), 3) if sub and all("wer" in r for r in sub) else None}
    wers = [r["wer"] for r in rows if "wer" in r]
    pesqs = [r["pesq"] for r in rows if "pesq" in r]
    subjs = [r["subj_mos"] for r in rows if "subj_mos" in r]
    return {
        "n": len(rows),
        "failed": failed,
        "rtf": stats([r["rtf"] for r in rows]),
        "rtf_wall": stats([r["rtf_wall"] for r in rows]),
        "ttfa_s": stats([r["ttfa_s"] for r in rows]),
        "synth_s": stats([r["synth_s"] for r in rows]),
        "audio_s": stats([r["audio_s"] for r in rows]),
        "wer": stats(wers),
        "wer_mean": round(statistics.mean(wers), 3) if wers else None,
        "pesq": stats(pesqs),
        "pesq_mean": round(statistics.mean(pesqs), 3) if pesqs else None,
        "subj_mos": stats(subjs),
        "subj_mos_mean": round(statistics.mean(subjs), 3) if subjs else None,
        "by_category": cats,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, choices=["paradee", "chatterbox-turbo", "qwen-polish", "piper", "mms", "supertonic"])
    ap.add_argument("--langs", default=None, help="comma list of pl,en (default per engine)")
    ap.add_argument("--tag", default="", help="suffix for results/out dirs, e.g. fp16 or coresident")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only first N sentences per language (probes)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--paradee-pl-espeak", action="store_true", help="feed Polish through espeak-ng 'pl' IPA")
    ap.add_argument("--cb-dtype", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--cb-device", default="cuda")
    ap.add_argument("--cb-cfm-steps", type=int, default=2)
    ap.add_argument("--cb-max-gen-len", type=int, default=1000)
    ap.add_argument("--qwen-model", default=None)
    ap.add_argument("--qwen-mmproj", default=None)
    ap.add_argument("--qwen-lang-pl", default="ru", help="--tts-lang token used for the Polish set")
    ap.add_argument("--qwen-speaker", default=None)
    ap.add_argument("--qwen-ngl", type=int, default=99)
    ap.add_argument("--qwen-ctx", type=int, default=2048)
    ap.add_argument("--qwen-n", type=int, default=768)
    ap.add_argument("--qwen-mmproj-cpu", action="store_true")
    ap.add_argument("--qwen-batch", type=int, default=0, help="-b/-ub value (0 = llama-tts default)")
    ap.add_argument("--piper-model", default=str(PIPER_DEFAULT))
    ap.add_argument("--piper-threads", type=int, default=2, help="ORT threads, as in config.toml [tts.piper]")
    ap.add_argument("--mms-device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--supertonic-steps", type=int, default=6)
    ap.add_argument("--supertonic-style", default="M4")
    ap.add_argument("--supertonic-speed", type=float, default=1.05)
    ap.add_argument("--supertonic-threads", type=int, default=2)
    ap.add_argument("--supertonic-sr", type=int, default=22050, help="output rate; 0 = native 44100")
    ap.add_argument("--set", choices=["main", "probe"], default="main",
                    help="main: 12 PL + 12 EN bench sentences; probe: 6-word and 11-13-word PL sentences (TTFA)")
    ap.add_argument("--skip-wer", action="store_true")
    ap.add_argument("--skip-mos", action="store_true", help="skip SQUIM objective/subjective scores")
    a = ap.parse_args()

    default_langs = {
        "paradee": "pl,en", "chatterbox-turbo": "en", "qwen-polish": "pl,en",
        "piper": "pl", "mms": "pl", "supertonic": "pl"
    }
    langs = (a.langs or default_langs[a.engine]).split(",")
    name = a.engine + (f"_{a.tag}" if a.tag else "")
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    sets = {lang: load_sentences(lang, probe=a.set == "probe") for lang in langs}
    if a.limit:
        sets = {k: v[: a.limit] for k, v in sets.items()}

    mon = GpuMon()
    loadavg_start = os.getloadavg()
    t_load0 = time.perf_counter()
    eng = build_engine(a)
    load_s = getattr(eng, "load_s", None)
    if load_s is None:
        load_s = time.perf_counter() - t_load0
    print(f"[{name}] loaded in {load_s:.2f}s on {eng.device}", flush=True)

    warm_text = sets[langs[0]][0]["text"]
    for i in range(a.warmup):
        r = eng.synth(warm_text, langs[0])
        print(f"[{name}] warm-up {i + 1}: synth {r['synth_s']:.2f}s", flush=True)

    results = {}
    for lang in langs:
        rows = []
        for item in sets[lang]:
            try:
                r = eng.synth(item["text"], lang)
            except Exception as e:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                rows.append({**item, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                print(f"[{name}] {item['id']} FAILED {type(e).__name__}", flush=True)
                continue
            audio_s = len(r["wav"]) / r["sr"]
            wav_path = out_dir / f"{lang}_{item['id']}.wav"
            sf.write(wav_path, r["wav"], r["sr"])
            row = {
                **item,
                "wav_path": str(wav_path.relative_to(ROOT)),
                "audio_s": round(audio_s, 3),
                "synth_s": round(r["synth_s"], 3),
                "ttfa_s": round(r["ttfa_s"], 3),
                "rtf": round(audio_s / r["synth_s"], 3),
                "rtf_wall": round(audio_s / r.get("wall_s", r["synth_s"]), 3),
            }
            if "load_est_s" in r:
                row["load_est_s"] = round(r["load_est_s"], 3)
            rows.append(row)
            print(f"[{name}] {item['id']} audio {audio_s:.2f}s synth {r['synth_s']:.2f}s RTF {row['rtf']:.2f}", flush=True)
        results[lang] = rows

    gpu = mon.stop()
    device_name, streaming = eng.device, eng.streaming
    if a.engine == "qwen-polish":
        load_s = statistics.median(eng.load_estimates)
    del eng
    import gc
    gc.collect()

    loadavg_end = os.getloadavg()
    if not a.skip_wer:
        for lang in langs:
            stt_round_trip(results[lang], lang)
            wers = [r["wer"] for r in results[lang] if "wer" in r]
            if wers:
                print(f"[{name}] {lang} WER median {statistics.median(wers):.3f}", flush=True)

    # Automatic MOS / speech quality predictor (SQUIM)
    for lang in langs:
        if not a.skip_mos:
            mos_round_trip(results[lang])
        pesqs = [r["pesq"] for r in results[lang] if "pesq" in r]
        subjs = [r["subj_mos"] for r in results[lang] if "subj_mos" in r]
        if pesqs:
            print(f"[{name}] {lang} PESQ median {statistics.median(pesqs):.3f}, Subj MOS median {statistics.median(subjs):.3f}", flush=True)

    # Save showcase wav (first 3 sentences concatenated with 0.5s pause)
    for lang in langs:
        valid_rows = [r for r in results[lang] if "error" not in r]
        if len(valid_rows) >= 3:
            w1, sr1 = sf.read(str(ROOT / valid_rows[0]["wav_path"]), dtype="float32")
            w2, sr2 = sf.read(str(ROOT / valid_rows[1]["wav_path"]), dtype="float32")
            w3, sr3 = sf.read(str(ROOT / valid_rows[2]["wav_path"]), dtype="float32")
            gap = np.zeros(int(0.5 * sr1), dtype=np.float32)
            showcase_audio = np.concatenate([w1, gap, w2, gap, w3])
            showcase_path = OUT / f"showcase_{name}.wav"
            sf.write(showcase_path, showcase_audio, sr1)
            print(f"[{name}] wrote showcase audio {showcase_path}", flush=True)

    doc = {
        "engine": a.engine,
        "set": a.set,
        "loadavg_start": [round(x, 2) for x in loadavg_start],
        "loadavg_end": [round(x, 2) for x in loadavg_end],
        "contaminated": loadavg_start[0] > QUIET_LOAD,
        "contaminated_rule": f"1-min loadavg > {QUIET_LOAD} at start of the timed section (end value includes the run itself)",
        "tag": a.tag,
        "device": device_name,
        "args": vars(a),
        "metric_definitions": METRIC_DEFS,
        "streaming": streaming,
        "load_s": round(load_s, 2),
        "load_s_note": "per-call process load estimate (llama-tts reloads each call)" if a.engine == "qwen-polish" else "in-process cold load",
        "gpu": gpu,
        "peak_rss_mb": rss_mb(),
        "stt": "Whistle (cactus-needle 3.1.2), language forced per set",
        "langs": {lang: {"summary": summarize(rows), "items": rows} for lang, rows in results.items()},
    }
    (RESULTS / f"tts_{name}.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{name}] wrote {RESULTS / f'tts_{name}.json'}", flush=True)


if __name__ == "__main__":
    main()
