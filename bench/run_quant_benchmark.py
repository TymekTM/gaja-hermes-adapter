"""Benchmark runner for STT quantization comparison (faster-whisper and whisper.cpp)."""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
from num2words import num2words

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
DATA = BENCH / "data" / "stt"
RESULTS = BENCH / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
WHISPER_CPP_BIN = Path("/home/ttm-server/.cache/gaja-whispercpp/build/bin/whisper-cli")
WHISPER_CPP_MODELS = Path("/home/ttm-server/.cache/gaja-whispercpp/models")

TECH_TERMS = [
    "git", "deploy", "docker", "kubernetes", "pull request", "merge",
    "commit", "backend", "frontend", "api", "endpoint", "gpu",
    "grafana", "home assistant", "tailscale", "python", "typescript",
    "build", "release"
]

def normalize_text(text: str, lang: str = "pl") -> str:
    text = text.lower()
    text = re.sub(r"\d+", lambda m: f" {num2words(int(m.group()), lang=lang)} ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

def read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in handle]

def load_audio(path: str) -> np.ndarray:
    data, rate = sf.read(path, dtype="float32", always_2d=False)
    assert rate == 16000, f"Expected 16kHz audio, got {rate} in {path}"
    return data

def calc_wer(ref: str, hyp: str) -> dict[str, int]:
    r_words, h_words = ref.split(), hyp.split()
    if not h_words:
        return {"S": 0, "D": len(r_words), "I": 0, "H": 0, "N": len(r_words)}
    if not r_words:
        return {"S": 0, "D": 0, "I": len(h_words), "H": 0, "N": 0}
    out = jiwer.process_words(ref, hyp)
    return {"S": out.substitutions, "D": out.deletions, "I": out.insertions, "H": out.hits, "N": len(r_words)}

def check_term_accuracy(raw_ref: str, hyp: str) -> tuple[int, int]:
    """Check how many tech terms from raw_ref appear in hyp."""
    ref_lower = raw_ref.lower()
    hyp_lower = hyp.lower()
    total_in_ref = 0
    matched = 0
    for term in TECH_TERMS:
        # Check if term appears in ref (allowing Polish suffixes like 'kubernetesa', 'dockera', 'grafanie')
        pattern = r"\b" + re.escape(term)
        if re.search(pattern, ref_lower):
            total_in_ref += 1
            # Check if recognized in hyp
            # For Polish inflections, stem match or exact term
            stem = term
            if term == "grafana":
                stem = "grafan"
            if re.search(r"\b" + re.escape(stem), hyp_lower) or term in hyp_lower:
                matched += 1
    return matched, total_in_ref

class GpuSampler:
    """Polls nvidia-smi: total GPU memory and this process share."""
    def __init__(self, cap_mb: float = 3900.0):
        self.cap_mb = cap_mb
        self.baseline_mb = None
        self.peak_total_mb = 0.0
        self.peak_own_mb = 0.0
        self._stop = False

    @staticmethod
    def _query(args):
        out = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=10).stdout
        return out.strip()

    def snapshot(self):
        try:
            total = float(self._query(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).splitlines()[0])
            own = 0.0
            for line in self._query(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and int(parts[0]) == os.getpid():
                    own += float(parts[1])
            return total, own
        except Exception:
            return 0.0, 0.0

    def start(self):
        import threading
        self.baseline_mb, _ = self.snapshot()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        import time
        while not self._stop:
            total, own = self.snapshot()
            self.peak_total_mb = max(self.peak_total_mb, total)
            self.peak_own_mb = max(self.peak_own_mb, own)
            time.sleep(0.1)

    def stop(self):
        self._stop = True
        return {
            "baseline_total_mb": self.baseline_mb,
            "peak_total_mb": self.peak_total_mb,
            "peak_own_process_mb": self.peak_own_mb,
        }

class FasterWhisperRunner:
    def __init__(self, model: str, compute: str, beam: int, device: str = "cuda"):
        from faster_whisper import WhisperModel
        self.model_name = model
        self.compute = compute
        self.beam = beam
        started = time.perf_counter()
        self.model = WhisperModel(model, device=device, compute_type=compute, cpu_threads=4, local_files_only=True)
        self.load_s = time.perf_counter() - started

    def transcribe(self, path: str, audio: np.ndarray, lang: str = "pl") -> tuple[str, float]:
        started = time.perf_counter()
        segments, _ = self.model.transcribe(
            audio,
            language=lang,
            beam_size=self.beam,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        wall = time.perf_counter() - started
        return text, wall

class WhisperCppRunner:
    def __init__(self, model_file: str, beam: int):
        self.model_path = str(WHISPER_CPP_MODELS / model_file)
        assert os.path.exists(self.model_path), f"Missing {self.model_path}"
        self.beam = beam
        # Calibrate load time using a short 2s audio clip
        silence_path = str(DATA / "synthetic" / "silence_2s.wav")
        started = time.perf_counter()
        cmd = [
            str(WHISPER_CPP_BIN),
            "-m", self.model_path,
            "-f", silence_path,
            "-l", "pl",
            "-bs", str(beam),
            "-nt", "-np",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse load time from stderr if present
        m = re.search(r"load time\s*=\s*([\d\.]+)\s*ms", res.stderr)
        if m:
            self.load_s = float(m.group(1)) / 1000.0
        else:
            self.load_s = time.perf_counter() - started

    def transcribe(self, path: str, audio: np.ndarray, lang: str = "pl") -> tuple[str, float]:
        tmp_json = f"/tmp/wcpp_{os.getpid()}_{time.time_ns()}"
        cmd = [
            str(WHISPER_CPP_BIN),
            "-m", self.model_path,
            "-f", path,
            "-l", lang,
            "-bs", str(self.beam),
            "-nt", "-oj", "-of", tmp_json,
        ]
        started = time.perf_counter()
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        wall = time.perf_counter() - started
        json_path = Path(f"{tmp_json}.json")
        text = ""
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                text = " ".join(t.get("text", "").strip() for t in data.get("transcription", []))
            finally:
                json_path.unlink()
        else:
            text = res.stdout.strip()
        # Parse pure compute time (encode + decode + batchd) from stderr if available
        # Else subtract load time from wall
        m_encode = re.search(r"encode time\s*=\s*([\d\.]+)\s*ms", res.stderr)
        m_decode = re.search(r"decode time\s*=\s*([\d\.]+)\s*ms", res.stderr)
        if m_encode and m_decode:
            inf_s = (float(m_encode.group(1)) + float(m_decode.group(1))) / 1000.0
            return text, inf_s
        return text, max(0.01, wall - self.load_s)

def run_evaluation(runner, pl_refs, mixed_refs, pl_latency_clips, mixed_latency_clips):
    gpu = GpuSampler()
    gpu.start()

    # Warmup
    warmup_audio = load_audio(str(DATA / "pl" / pl_refs[0]["file"]))
    runner.transcribe(str(DATA / "pl" / pl_refs[0]["file"]), warmup_audio, "pl")

    # 1. PL FLEURS set
    pl_err_s = pl_err_d = pl_err_i = pl_err_n = 0
    pl_records = []
    for row in pl_refs:
        path = str(DATA / "pl" / row["file"])
        audio = load_audio(path)
        hyp_raw, wall = runner.transcribe(path, audio, "pl")
        hyp_norm = normalize_text(hyp_raw, "pl")
        ref_norm = normalize_text(row["ref"], "pl")
        err = calc_wer(ref_norm, hyp_norm)
        pl_err_s += err["S"]
        pl_err_d += err["D"]
        pl_err_i += err["I"]
        pl_err_n += err["N"]
        pl_records.append({"id": row["utt_id"], "wall_s": wall, "ref": ref_norm, "hyp": hyp_norm})
    pl_wer = (pl_err_s + pl_err_d + pl_err_i) / max(1, pl_err_n)

    # 2. Mixed PL/EN set
    mix_err_s = mix_err_d = mix_err_i = mix_err_n = 0
    tech_matched_tot = tech_ref_tot = 0
    mix_records = []
    for row in mixed_refs:
        path = str(DATA / "mixed" / row["file"])
        audio = load_audio(path)
        hyp_raw, wall = runner.transcribe(path, audio, "pl")
        hyp_norm = normalize_text(hyp_raw, "pl")
        ref_norm = normalize_text(row["ref"], "pl")
        err = calc_wer(ref_norm, hyp_norm)
        mix_err_s += err["S"]
        mix_err_d += err["D"]
        mix_err_i += err["I"]
        mix_err_n += err["N"]
        matched, ref_count = check_term_accuracy(row["raw_ref"], hyp_raw)
        tech_matched_tot += matched
        tech_ref_tot += ref_count
        mix_records.append({
            "id": row["utt_id"], "wall_s": wall, "raw_ref": row["raw_ref"],
            "hyp_raw": hyp_raw, "term_matched": matched, "term_ref": ref_count
        })
    mixed_wer = (mix_err_s + mix_err_d + mix_err_i) / max(1, mix_err_n)
    term_acc = tech_matched_tot / max(1, tech_ref_tot)

    # 3. Latency measurement (2-5s utterances across both sets)
    lat_walls = []
    # Sample 20 clips from PL and 20 clips from Mixed for consistency
    selected_lat = (pl_latency_clips[:20] + mixed_latency_clips[:20])
    for clip_row, base_dir in selected_lat:
        path = str(base_dir / clip_row["clip"])
        audio = load_audio(path)
        _, wall = runner.transcribe(path, audio, "pl")
        lat_walls.append(wall)

    lat_walls.sort()
    lat_p50 = float(np.percentile(lat_walls, 50))
    lat_p90 = float(np.percentile(lat_walls, 90))

    gpu_stats = gpu.stop()

    return {
        "pl_wer": pl_wer,
        "mixed_wer": mixed_wer,
        "term_accuracy": term_acc,
        "term_stats": {"matched": tech_matched_tot, "total": tech_ref_tot},
        "latency_p50_s": lat_p50,
        "latency_p90_s": lat_p90,
        "load_s": runner.load_s,
        "gpu": gpu_stats,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, help="e.g. fw_small_int8_b1, wcpp_medium_q5_0_b1, etc.")
    args = parser.parse_args()

    pl_refs = read_tsv(DATA / "pl" / "refs.tsv")
    mixed_refs = read_tsv(DATA / "mixed" / "refs.tsv")
    pl_lat = [(row, DATA / "pl") for row in read_tsv(DATA / "pl" / "latency.tsv") if 2.0 <= float(row["duration_s"]) <= 5.0]
    mixed_lat = [(row, DATA / "mixed") for row in read_tsv(DATA / "mixed" / "latency.tsv") if 2.0 <= float(row["duration_s"]) <= 5.0]

    v = args.variant
    print(f"=== Running variant: {v} ===", flush=True)

    # Dispatch variant
    if v.startswith("fw_"):
        # e.g. fw_small_int8_b1, fw_medium_int8_b1, fw_large-v3-turbo_int8_b1, fw_small_int8f32_b1, fw_small_f32_b1, etc.
        parts = v.split("_")
        model = parts[1] # small, medium, large-v3-turbo
        compute = parts[2] # int8, int8_float32 (int8f32), float32 (f32)
        if compute == "int8f32":
            compute = "int8_float32"
        elif compute == "f32":
            compute = "float32"
        beam = int(parts[3].replace("b", ""))
        runner = FasterWhisperRunner(model=model, compute=compute, beam=beam)
    elif v.startswith("wcpp_"):
        # e.g. wcpp_small_q8_0_b1, wcpp_medium_q4_0_b1, wcpp_large-v3-turbo_q5_0_b1, etc.
        parts = v.split("_")
        if parts[1] == "large":
            # wcpp_large_v3_turbo_q4_0_b1
            model_key = "large-v3-turbo"
            qtype = parts[4] + "_" + parts[5] if len(parts) == 7 else parts[4]
            beam = int(parts[-1].replace("b", ""))
        else:
            model_key = parts[1] # small, medium, large-v3-turbo
            qtype = parts[2] + "_" + parts[3]
            beam = int(parts[4].replace("b", ""))
        model_file = f"ggml-{model_key}-{qtype}.bin"
        runner = WhisperCppRunner(model_file=model_file, beam=beam)
    else:
        raise ValueError(f"Unknown variant {v}")

    results = run_evaluation(runner, pl_refs, mixed_refs, pl_lat, mixed_lat)
    out_file = RESULTS / f"stt_quant_{v}.json"
    out_data = {
        "variant": v,
        "results": results,
    }
    out_file.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({v: results}, indent=2), flush=True)

if __name__ == "__main__":
    main()
