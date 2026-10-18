#!/usr/bin/env python3
"""Evaluates the production Parakeet STT engine (gaja/stt/parakeet_stt.py) on every Gaja STT set.

- PL FLEURS WER set and the mixed PL/EN technical set with the vocab hotwords, scored raw (comparable with
  bench/eval_hotwords_cost.py) and after VocabularyManager.correct as in the pipeline. The mixed WAVs come
  from bench/stt_prepare_mixed.py.
- Short Polish commands (synthesized here with Piper, both voices): exact-match count.
- Wake verification on the private recordings in data/wakeword (first 2.5 s, verify hotwords, the
  production matcher): accepted wakes and accepted hard negatives.
- Latency on the 2-5 s PL latency clips.

Usage: uv run python bench/eval_parakeet.py [--hotwords-score 3.0] [--prefix-below-s 2.6]
Results: bench/results/parakeet_eval.json (appended per configuration).
"""

import argparse
import asyncio
import collections
import fcntl
import io
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from bench.eval_hotwords_cost import calc_wer, check_term_accuracy, normalize_text, read_tsv  # noqa: E402
from gaja.devices.wake import match_wake_phrase  # noqa: E402
from gaja.stt.parakeet_stt import ParakeetSTT  # noqa: E402
from gaja.stt.vocab import VocabularyManager  # noqa: E402

DATA = REPO_ROOT / "bench" / "data" / "stt"
SHORT_DIR = DATA / "short"
WAKE_DIR = REPO_ROOT / "data" / "wakeword" / "recordings"
RESULTS = REPO_ROOT / "bench" / "results" / "parakeet_eval.json"
VERIFY_HOTWORDS = "Hej Gaja, Gajo"
SHORT_COMMANDS = [
    "Stop.", "Dzięki.", "Tak.", "Nie.", "Która godzina?", "Jaka jest pogoda?", "Głośniej.", "Ciszej.",
    "Włącz muzykę.", "Zgaś światło.", "Nastaw minutnik na pięć minut.", "Co nowego?", "Powtórz.", "Anuluj.",
    "Dobranoc.", "Ile jest stopni?", "Opowiedz żart.", "Jaki dziś dzień?", "Kurs euro.", "Pomóż mi.",
    "Dziękuję bardzo.", "Wyłącz się.", "Kto to jest Kopernik?", "Przypomnij mi jutro o ósmej.", "Jak się masz?",
]


def short_commands() -> list[tuple[Path, str]]:
    """Each command in both Piper voices, 0.3 s of silence around it (cached as WAVs)."""
    out: list[tuple[Path, str]] = []
    missing = [(i, s, v) for i, s in enumerate(SHORT_COMMANDS) for v in ("gosia", "darkman")
               if not (SHORT_DIR / f"{i:02d}_{v}.wav").exists()]
    if missing:
        import soxr
        from piper import PiperVoice

        SHORT_DIR.mkdir(parents=True, exist_ok=True)
        voices = {v: PiperVoice.load(str(REPO_ROOT / "models" / f"pl_PL-{v}-medium.onnx")) for v in ("gosia", "darkman")}
        for i, text, v in missing:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                voices[v].synthesize_wav(text, w)
            buf.seek(0)
            audio, rate = sf.read(buf, dtype="float32")
            pad = np.zeros(4800, dtype=np.float32)
            sf.write(SHORT_DIR / f"{i:02d}_{v}.wav", np.concatenate([pad, soxr.resample(audio, rate, 16000), pad]), 16000)
    for i, text in enumerate(SHORT_COMMANDS):
        out += [(SHORT_DIR / f"{i:02d}_{v}.wav", text) for v in ("gosia", "darkman")]
    return out


async def evaluate(stt: ParakeetSTT, vm: VocabularyManager) -> dict:
    await stt.warm_up()
    hotwords = vm.hotwords()
    res: dict = {}
    for name in ("pl", "mixed"):
        tally = {key: collections.Counter() for key in ("raw", "corrected")}
        for row in read_tsv(DATA / name / "refs.tsv"):
            audio = sf.read(str(DATA / name / row["file"]), dtype="float32")[0]
            raw = (await stt.transcribe(audio, "pl", hotwords=hotwords)).text
            for key, hyp in (("raw", raw), ("corrected", vm.correct(raw))):
                err = calc_wer(normalize_text(row["ref"]), normalize_text(hyp))
                tally[key]["errors"] += err["S"] + err["D"] + err["I"]
                tally[key]["words"] += err["N"]
                if name == "mixed":
                    m, t = check_term_accuracy(row["raw_ref"], hyp)
                    tally[key]["matched"] += m
                    tally[key]["terms"] += t
        for key, c in tally.items():
            res[f"{name}_wer_{key}"] = round(c["errors"] / c["words"], 4)
            if name == "mixed":
                res[f"term_accuracy_{key}"] = {"matched": c["matched"], "total": c["terms"]}

    exact = 0
    clips = short_commands()
    for path, text in clips:
        hyp = (await stt.transcribe(sf.read(path, dtype="float32")[0], "pl", hotwords=hotwords)).text
        exact += hyp.lower().strip(".?! ") == text.lower().strip(".?! ")
    res["short_commands_exact"] = {"matched": exact, "total": len(clips)}

    counts: collections.Counter = collections.Counter()
    verify_ms = []
    for kind in ("wake", "hard_negative"):
        for path in sorted(WAKE_DIR.glob(f"*/{kind}/**/*.wav")):
            audio = sf.read(path, dtype="float32")[0][: 16000 * 25 // 10]
            t0 = time.perf_counter()
            text = (await stt.transcribe(audio, "pl", hotwords=VERIFY_HOTWORDS)).text
            verify_ms.append((time.perf_counter() - t0) * 1000)
            counts[kind, match_wake_phrase(text).matched] += 1
    for kind in ("wake", "hard_negative"):
        res[f"{kind}_accepted"] = {"matched": counts[kind, True], "total": counts[kind, True] + counts[kind, False]}
    res["verify_p50_ms"] = round(float(np.percentile(verify_ms, 50))) if verify_ms else None

    latencies = []
    for row in read_tsv(DATA / "pl" / "latency.tsv"):
        if 2.0 <= float(row["duration_s"]) <= 5.0:
            audio = sf.read(str(DATA / "pl" / row["clip"]), dtype="float32")[0]
            t0 = time.perf_counter()
            await stt.transcribe(audio, "pl", hotwords=hotwords)
            latencies.append((time.perf_counter() - t0) * 1000)
    res["latency_p50_ms"] = round(float(np.percentile(latencies, 50)))
    res["latency_p90_ms"] = round(float(np.percentile(latencies, 90)))
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotwords-score", type=float, default=3.0)
    parser.add_argument("--prefix-below-s", type=float, default=2.6)
    args = parser.parse_args()
    with open("/tmp/gaja-bench.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        vm = VocabularyManager(
            default_path=REPO_ROOT / "prompts/vocab_default.toml",
            user_path=REPO_ROOT / "data/vocab.toml",
            suggestions_path=REPO_ROOT / "bench" / "results" / "parakeet_vocab_suggestions.json",
        )
        vm._schedule_save_suggestions = lambda *a, **k: None  # near-miss suggestions stay out of data/
        stt = ParakeetSTT(hotwords_score=args.hotwords_score, prefix_below_s=args.prefix_below_s)
        res = {"hotwords_score": args.hotwords_score, "prefix_below_s": args.prefix_below_s}
        res.update(asyncio.run(evaluate(stt, vm)))
    print(json.dumps(res, ensure_ascii=False))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    runs = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    RESULTS.write_text(json.dumps(runs + [res], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
