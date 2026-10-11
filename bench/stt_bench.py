"""STT benchmark on FLEURS pl/en: WER, latency, RTF, RSS for Whistle and faster-whisper."""
import argparse
import json
import os
import platform
import re
import resource
import shutil
import statistics
import subprocess
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
from num2words import num2words

BENCH = Path(__file__).resolve().parent
DATA = BENCH / "data" / "stt"
RESULTS = BENCH / "results"
NATIVE_CLI = BENCH / "tools" / "needle"
WHISTLE_CACT = Path.home() / ".cache/cactus-needle/whistle/2.0.0/whistle.cact"
PIPER_MODEL = Path.home() / "voice-agent/pl_PL-darkman-medium.onnx"
SENTENCES_PL = BENCH / "sentences_pl.txt"
SR = 16000


def normalize(text, lang):
    text = text.lower()
    text = re.sub(r"\d+", lambda m: f" {num2words(int(m.group()), lang=lang)} ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def read_tsv(path):
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in handle]


def load_audio(path):
    data, rate = sf.read(path, dtype="float32", always_2d=False)
    assert rate == SR, path
    return data


def word_errors(ref, hyp):
    ref_words, hyp_words = ref.split(), hyp.split()
    if not hyp_words:
        return {"S": 0, "D": len(ref_words), "I": 0, "H": 0, "N": len(ref_words)}
    if not ref_words:
        return {"S": 0, "D": 0, "I": len(hyp_words), "H": 0, "N": 0}
    out = jiwer.process_words(ref, hyp)
    return {"S": out.substitutions, "D": out.deletions, "I": out.insertions, "H": out.hits, "N": len(ref_words)}


def pct(values, q):
    return float(np.percentile(values, q)) if values else None


def cpu_model():
    with open("/proc/cpuinfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


class WhistlePy:
    def __init__(self, args):
        from needle import Whistle

        started = time.perf_counter()
        self.model = Whistle()
        self.load_s = time.perf_counter() - started
        self.config = {"api": "needle.Whistle().transcribe", "threads": "library default", "beam": "library default",
                       "audio_depth": "library default (full)", "keywords": None}

    def run(self, path, audio, lang):
        result = self.model.transcribe(audio, language=lang)
        return {"text": result["text"], "language": result["language"] or None, "ttft_s": result["ttft_ms"] / 1000}


class WhistleCli:
    def __init__(self, args):
        self.extra = []
        if args.depth:
            self.extra += ["--audio-depth", str(args.depth)]
        if args.threads:
            self.extra += ["--threads", str(args.threads)]
        self.config = {"api": "native needle CLI (bench/tools/needle)", "threads": args.threads or "CLI default (<=4)",
                       "beam": "CLI default", "audio_depth": args.depth or "full", "keywords": None}
        self.load_s = None

    def cmd(self, path, lang):
        cmd = [str(NATIVE_CLI), "--model", str(WHISTLE_CACT), "--audio", path, *self.extra]
        if lang:
            cmd += ["--audio-language", lang]
        return cmd

    def run(self, path, audio, lang):
        done = subprocess.run(self.cmd(path, lang), capture_output=True, text=True, check=True)
        out = json.loads(done.stdout.strip().splitlines()[-1])
        return {"text": out["text"], "language": out["language"] or None, "ttft_s": out["ttft_ms"] / 1000}

    def calibrate(self):
        silence = str(DATA / "synthetic" / "silence_2s.wav")
        walls = []
        for _ in range(3):
            started = time.perf_counter()
            subprocess.run(self.cmd(silence, "pl"), capture_output=True, text=True, check=True)
            walls.append(time.perf_counter() - started)
        self.load_s = statistics.median(walls)


class FasterWhisper:
    def __init__(self, args):
        from faster_whisper import WhisperModel

        started = time.perf_counter()
        self.model = WhisperModel(args.model, device=args.device, compute_type=args.compute, cpu_threads=args.threads or 0,
                                  local_files_only=True)
        self.load_s = time.perf_counter() - started
        self.beam = args.beam
        self.decode = {"vad_filter": args.vad_filter, "condition_on_previous_text": not args.no_cond_prev,
                       "without_timestamps": args.without_timestamps, "no_speech_threshold": args.no_speech_threshold,
                       "log_prob_threshold": args.log_prob_threshold}
        self.config = {"api": "faster_whisper.WhisperModel.transcribe", "model": args.model, "device": args.device,
                       "compute_type": args.compute, "beam_size": args.beam, "cpu_threads": args.threads or "default",
                       **self.decode}

    def run(self, path, audio, lang):
        started = time.perf_counter()
        segments, info = self.model.transcribe(audio, language=lang, beam_size=self.beam, **self.decode)
        pieces, first_s = [], None
        for segment in segments:
            if first_s is None:
                first_s = time.perf_counter() - started
            pieces.append(segment.text.strip())
        return {"text": " ".join(pieces), "language": info.language, "ttft_s": first_s}


class GpuSampler:
    """Polls nvidia-smi: total GPU memory and this process's share; aborts the process above the cap."""

    def __init__(self, cap_mb):
        self.cap_mb = cap_mb
        self.baseline_mb = None
        self.peak_total_mb = 0.0
        self.peak_own_mb = 0.0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    @staticmethod
    def _query(args):
        out = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=10).stdout
        return out.strip()

    def snapshot(self):
        total = float(self._query(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).splitlines()[0])
        own = 0.0
        for line in self._query(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).splitlines():
            pid, mem = [p.strip() for p in line.split(",")]
            if int(pid) == os.getpid():
                own += float(mem)
        return total, own

    def start(self):
        self.baseline_mb, _ = self.snapshot()
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            total, own = self.snapshot()
            self.samples += 1
            self.peak_total_mb = max(self.peak_total_mb, total)
            self.peak_own_mb = max(self.peak_own_mb, own)
            if total > self.cap_mb:
                print(f"ABORT: GPU total {total:.0f} MiB > cap {self.cap_mb} MiB", file=sys.stderr, flush=True)
                os._exit(3)
            time.sleep(0.2)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        return {"baseline_total_mb": self.baseline_mb, "peak_total_mb": self.peak_total_mb,
                "peak_own_process_mb": self.peak_own_mb, "cap_mb": self.cap_mb, "samples": self.samples}


class PiperLoad:
    def __init__(self):
        from piper import PiperVoice

        self.voice = PiperVoice.load(PIPER_MODEL, config_path=Path(str(PIPER_MODEL) + ".json"), use_cuda=False)
        self.sentences = [line.strip() for line in open(SENTENCES_PL, encoding="utf-8") if line.strip()]


def run_piper_loop():
    load = PiperLoad()
    print("READY", flush=True)
    while True:
        for sentence in load.sentences:
            for _ in load.voice.synthesize(sentence):
                pass


def start_contention():
    proc = subprocess.Popen([sys.executable, __file__, "--piper-loop", "--tag", "piper_worker"], stdout=subprocess.PIPE, text=True)
    line = proc.stdout.readline().strip()
    if line != "READY":
        proc.kill()
        raise RuntimeError(f"piper load did not start: {line!r}")
    return proc


def build_items(lang, which):
    items = []
    if which in ("wer", "nat3to5"):
        for row in read_tsv(DATA / lang / "refs.tsv"):
            dur = float(row["duration_s"])
            if which == "nat3to5" and not (3.0 <= dur < 5.0):
                continue
            items.append({"set": "wer", "id": row["utt_id"], "path": str(DATA / lang / row["file"]), "duration_s": dur,
                          "ref": normalize(row["ref"], lang), "raw_ref": row["ref"]})
    if which in ("latency", "lat4"):
        for row in read_tsv(DATA / lang / "latency.tsv"):
            target = float(row["cut_target_s"])
            dur = float(row["duration_s"])
            if not 2.0 <= dur <= 5.0 or (which == "lat4" and target != 4.0):
                continue
            items.append({"set": "latency", "id": Path(row["clip"]).stem, "path": str(DATA / lang / row["clip"]),
                          "duration_s": float(row["duration_s"]), "target_s": target, "ref": None})
    if which == "hallu":
        for name in ("silence_2s", "noise_2s"):
            items.append({"set": "hallu", "id": name, "path": str(DATA / "synthetic" / f"{name}.wav"),
                          "duration_s": 2.0, "ref": None})
    return items


def ensure_synthetic():
    out = DATA / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "silence_2s.wav").exists():
        sf.write(out / "silence_2s.wav", np.zeros(2 * SR, np.float32), SR, subtype="PCM_16")
    if not (out / "noise_2s.wav").exists():
        rng = np.random.default_rng(0)
        sf.write(out / "noise_2s.wav", (rng.standard_normal(2 * SR) * 0.01).astype(np.float32), SR, subtype="PCM_16")


def summarize_set(records, lang, lang_mode):
    wer_recs = [r for r in records if r["set"] == "wer"]
    lat_recs = [r for r in records if r["set"] == "latency"]
    hallu_recs = [r for r in records if r["set"] == "hallu"]
    summary = {}
    if wer_recs:
        total = {k: sum(r["err"][k] for r in wer_recs) for k in ("S", "D", "I", "H", "N")}
        wall = [r["wall_s"] for r in wer_recs]
        summary["wer"] = {
            "n": len(wer_recs),
            "corpus_wer": (total["S"] + total["D"] + total["I"]) / total["N"],
            "median_utt_wer": statistics.median([(r["err"]["S"] + r["err"]["D"] + r["err"]["I"]) / r["err"]["N"]
                                                 for r in wer_recs if r["err"]["N"]]),
            "latency_median_s": statistics.median(wall),
            "latency_p90_s": pct(wall, 90),
            "rtf_corpus": sum(r["duration_s"] for r in wer_recs) / sum(wall),
            "ttft_median_s": statistics.median([r["ttft_s"] for r in wer_recs if r["ttft_s"] is not None]) if any(
                r["ttft_s"] is not None for r in wer_recs) else None,
        }
        if lang_mode == "auto":
            summary["wer"]["lang_detect_errors"] = sum(1 for r in wer_recs if r["language"] != lang)
            summary["wer"]["lang_detect_errors_list"] = [r["id"] for r in wer_recs if r["language"] != lang]
        nat = [r for r in wer_recs if 3.0 <= r["duration_s"] < 5.0]
        if nat:
            n_tot = {k: sum(r["err"][k] for r in nat) for k in ("S", "D", "I", "H", "N")}
            summary["wer_natural_3to5s"] = {
                "n": len(nat), "corpus_wer": (n_tot["S"] + n_tot["D"] + n_tot["I"]) / n_tot["N"],
                "latency_median_s": statistics.median([r["wall_s"] for r in nat]),
                "latency_p90_s": pct([r["wall_s"] for r in nat], 90)}
    if lat_recs:
        wall = [r["wall_s"] for r in lat_recs]
        summary["latency_2to5s"] = {
            "n": len(lat_recs),
            "latency_median_s": statistics.median(wall),
            "latency_p90_s": pct(wall, 90),
            "latency_mean_s": statistics.fmean(wall),
            "rtf_corpus": sum(r["duration_s"] for r in lat_recs) / sum(wall),
            "rtf_median": statistics.median([r["rtf"] for r in lat_recs]),
            "by_target": {},
        }
        for target in sorted({r["target_s"] for r in lat_recs}):
            sub = [r["wall_s"] for r in lat_recs if r["target_s"] == target]
            summary["latency_2to5s"]["by_target"][f"{target:g}s"] = {
                "n": len(sub), "latency_median_s": statistics.median(sub), "latency_p90_s": pct(sub, 90)}
    if lat_recs:
        short = [r["wall_s"] for r in lat_recs if 1.0 <= r["duration_s"] < 3.0]
        if short:
            summary["latency_short_1to3s"] = {"n": len(short), "latency_p50_s": statistics.median(short),
                                              "latency_p90_s": pct(short, 90)}
    if wer_recs:
        long_ = [r["wall_s"] for r in wer_recs if 5.0 <= r["duration_s"] <= 10.0]
        if long_:
            summary["wer_long_5to10s"] = {"n": len(long_), "latency_p50_s": statistics.median(long_),
                                          "latency_p90_s": pct(long_, 90)}
    if hallu_recs:
        summary["hallucination"] = {r["id"]: r["hyp"] for r in hallu_recs}
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--piper-loop", action="store_true")
    p.add_argument("--engine", choices=["whistle", "whistle-cli", "faster-whisper"], default="whistle")
    p.add_argument("--lang", choices=["pl", "en"], default="pl")
    p.add_argument("--lang-mode", choices=["forced", "auto"], default="forced")
    p.add_argument("--sets", default="wer,latency,hallu", help="wer, latency, lat4, nat3to5, hallu")
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--model", default="small")
    p.add_argument("--device", default="cpu")
    p.add_argument("--compute", default="int8")
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--contention", action="store_true", help="run a Piper synthesis loop in parallel")
    p.add_argument("--vad-filter", action="store_true", help="faster-whisper internal Silero VAD (off by default)")
    p.add_argument("--no-cond-prev", action="store_true", help="condition_on_previous_text=False")
    p.add_argument("--without-timestamps", action="store_true")
    p.add_argument("--no-speech-threshold", type=float, default=0.6)
    p.add_argument("--log-prob-threshold", type=lambda s: None if s.lower() == "none" else float(s), default=-1.0,
                   help="'none' disables the avg-logprob override of no_speech_threshold")
    p.add_argument("--vram-cap-mb", type=float, default=3900, help="cuda only: abort the process above this total GPU use")
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    if args.piper_loop:
        return run_piper_loop()

    ensure_synthetic()
    RESULTS.mkdir(parents=True, exist_ok=True)
    lang = None if args.lang_mode == "auto" else args.lang
    gpu = None
    if args.device == "cuda":
        gpu = GpuSampler(args.vram_cap_mb)
        gpu.start()
    if args.engine == "whistle":
        engine = WhistlePy(args)
    elif args.engine == "whistle-cli":
        engine = WhistleCli(args)
        engine.calibrate()
    else:
        engine = FasterWhisper(args)
    rss_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    gpu_after_load = gpu.snapshot()[0] if gpu is not None else None

    items = []
    for which in args.sets.split(","):
        items += build_items(args.lang, which.strip())
    if not items:
        raise SystemExit("no items selected")

    warm = [i for i in items if i["set"] == "wer"][:args.warmup] or items[:args.warmup]
    for item in warm:
        engine.run(item["path"], load_audio(item["path"]), lang)

    proc = start_contention() if args.contention else None
    load_before = os.getloadavg()[0]
    records = []
    try:
        for item in items:
            audio = load_audio(item["path"])
            started = time.perf_counter()
            out = engine.run(item["path"], audio, lang)
            wall = time.perf_counter() - started
            rec = {"set": item["set"], "id": item["id"], "duration_s": round(item["duration_s"], 3), "wall_s": wall,
                   "rtf": item["duration_s"] / wall, "ref": item["ref"], "hyp": normalize(out["text"], args.lang),
                   "raw_hyp": out["text"], "language": out["language"], "ttft_s": out["ttft_s"]}
            if item["set"] == "latency":
                rec["target_s"] = item["target_s"]
            if item["ref"] is not None:
                rec["err"] = word_errors(item["ref"], rec["hyp"])
            records.append(rec)
    finally:
        if proc is not None:
            proc.kill()
            proc.wait()
    gpu_stats = gpu.stop() if gpu is not None else None
    if gpu_stats is not None:
        gpu_stats["total_after_load_mb"] = gpu_after_load

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    summary = summarize_set(records, args.lang, args.lang_mode)
    result = {
        "tag": args.tag, "engine": args.engine, "lang": args.lang, "lang_mode": args.lang_mode,
        "config": engine.config, "sets": args.sets, "warmup_calls": args.warmup,
        "contention_piper_loop": args.contention, "loadavg_1m_before_run": load_before,
        "load_s": engine.load_s, "rss_after_load_mb": rss_after_load, "peak_rss_mb": peak_rss,
        "host": {"cpu": cpu_model(), "logical_cpus": os.cpu_count(), "python": platform.python_version(),
                 "gpu_in_use_by_this_run": args.device == "cuda"},
        "versions": {"jiwer": version("jiwer"), "numpy": np.__version__, "cactus-needle": version("cactus-needle"),
                     "faster-whisper": version("faster-whisper") if args.engine == "faster-whisper" else None},
        "gpu": gpu_stats,
        "summary": summary, "items": records,
        "definitions": {
            "wer": "jiwer word error rate on lowercased, punctuation-stripped, digit-spelled text (num2words); corpus = sum errors / sum ref words",
            "latency": "wall seconds around the transcribe call, audio already in memory, model warm; excludes VAD endpoint wait",
            "rtf_corpus": "sum(audio seconds) / sum(wall seconds)",
            "ttft": "Whistle: ttft_ms from the engine; faster-whisper: time to first segment (not first token)",
            "peak_rss_mb": "ru_maxrss of the engine process (includes numpy and Python imports)",
        },
    }
    if args.engine == "whistle-cli":
        result["config"]["cli_calibration_overhead_s"] = engine.load_s
        result["summary_note"] = "CLI latency includes process start and model load; subtract cli_calibration_overhead_s for engine-only time"
    out_path = RESULTS / f"stt_{args.tag}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"tag": args.tag, "summary": summary, "peak_rss_mb": round(peak_rss), "load_s": engine.load_s},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
