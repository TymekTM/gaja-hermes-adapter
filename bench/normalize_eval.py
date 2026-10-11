#!/usr/bin/env python3
"""Objective evaluation of TTS text normalizer (gaja/text/normalize.py).

Synthesizes ~30 PL sentences containing numbers, times, units, currencies,
abbreviations, and Roman numerals using Piper gosia on CPU:
  Condition 1: WITHOUT normalization (raw text with digits and symbols)
  Condition 2: WITH normalization (normalize_for_tts)

Transcribes synthesized audio using faster-whisper small on CPU int8.
Computes:
  - Numbers/entities correct recognition rate (with vs without)
  - Word Error Rate (WER) before and after
  - Audio duration and latency
  - Load average at start and finish

Saves results to bench/results/normalize_eval.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from faster_whisper import WhisperModel

from gaja.text.normalize import normalize_for_tts
from gaja.text.style import Style
from gaja.tts.piper_tts import PiperTTS

MODEL_VOICE_PATH = ROOT / "models/pl_PL-gosia-medium.onnx"
RESULTS_PATH = ROOT / "bench/results/normalize_eval.json"

TEST_SENTENCES: list[dict[str, Any]] = [
    {
        "id": "s_01",
        "raw": "Temperatura wynosi 12°C, a w nocy spadnie do -5°C.",
        "expected_numbers": ["12", "-5"],
    },
    {
        "id": "s_02",
        "raw": "Spotkanie zaplanowano na godzinę 06:54 rano.",
        "expected_numbers": ["6", "54"],
    },
    {
        "id": "s_03",
        "raw": "Kolejny pociąg odjeżdża o 17:59 ze stacji.",
        "expected_numbers": ["17", "59"],
    },
    {
        "id": "s_04",
        "raw": "Wydarzenie odbędzie się 9 października 2026 r.",
        "expected_numbers": ["9", "2026"],
    },
    {
        "id": "s_05",
        "raw": "Mikołaj Kopernik żył w XVI wieku.",
        "expected_numbers": ["16"],
    },
    {
        "id": "s_06",
        "raw": "Aktualny kurs euro wynosi 4,38 zł za jedno euro.",
        "expected_numbers": ["4", "38"],
    },
    {
        "id": "s_07",
        "raw": "Za bilet zapłaciłem 100 EUR oraz 50 USD.",
        "expected_numbers": ["100", "50"],
    },
    {
        "id": "s_08",
        "raw": "Samochód poruszał się z prędkością 100 km/h.",
        "expected_numbers": ["100"],
    },
    {
        "id": "s_09",
        "raw": "Do celu pozostało jeszcze około 50 km drogi.",
        "expected_numbers": ["50"],
    },
    {
        "id": "s_10",
        "raw": "Paczka waży dokładnie 2 kg.",
        "expected_numbers": ["2"],
    },
    {
        "id": "s_11",
        "raw": "Szansa na opady deszczu wynosi dzisiaj 15%.",
        "expected_numbers": ["15"],
    },
    {
        "id": "s_12",
        "raw": "Podwyżka wyniosła 2.5% w tym roku.",
        "expected_numbers": ["2", "5"],
    },
    {
        "id": "s_13",
        "raw": "Zostało 11.2 z 15.5 GB pamięci RAM.",
        "expected_numbers": ["11", "2", "15", "5"],
    },
    {
        "id": "s_14",
        "raw": "Karta graficzna posiada 4096 MiB pamięci.",
        "expected_numbers": ["4096"],
    },
    {
        "id": "s_15",
        "raw": "W sklepie kupiłem np. 3 kg jabłek.",
        "expected_numbers": ["3"],
    },
    {
        "id": "s_16",
        "raw": "Zajęcia rozpoczynają się o 08:30 w sali numer 14.",
        "expected_numbers": ["8", "30", "14"],
    },
    {
        "id": "s_17",
        "raw": "Urodził się w 1990 roku w Krakowie.",
        "expected_numbers": ["1990"],
    },
    {
        "id": "s_18",
        "raw": "Najbliższe święto wypada 1 listopada za 23 dni.",
        "expected_numbers": ["1", "23"],
    },
    {
        "id": "s_19",
        "raw": "Wiatr osiąga prędkość do 25 m/s w porywach.",
        "expected_numbers": ["25"],
    },
    {
        "id": "s_20",
        "raw": "Wynik tego działania to 240 pomnożone przez 15.",
        "expected_numbers": ["240", "15"],
    },
    {
        "id": "s_21",
        "raw": "Długość trasy wynosi 1500 m, czyli 1.5 km.",
        "expected_numbers": ["1500", "1", "5"],
    },
    {
        "id": "s_22",
        "raw": "Na koncie zostało 1250 zł.",
        "expected_numbers": ["1250"],
    },
    {
        "id": "s_23",
        "raw": "Temperatura w pokoju wynosi 21°C.",
        "expected_numbers": ["21"],
    },
    {
        "id": "s_24",
        "raw": "W nocy temperatura spadnie do -12°C.",
        "expected_numbers": ["-12"],
    },
    {
        "id": "s_25",
        "raw": "W XXI wieku nastąpił rozwój sztucznej inteligencji.",
        "expected_numbers": ["21"],
    },
    {
        "id": "s_26",
        "raw": "Wzrost zawodnika to 185 cm, a waga 82 kg.",
        "expected_numbers": ["185", "82"],
    },
    {
        "id": "s_27",
        "raw": "Cena paliwa spadła o 5% w tym tygodniu.",
        "expected_numbers": ["5"],
    },
    {
        "id": "s_28",
        "raw": "W bibliotece znajduje się ok. 1000 książek.",
        "expected_numbers": ["1000"],
    },
    {
        "id": "s_29",
        "raw": "Wizyta u lekarza o godz. 14:15 potrwa 45 minut.",
        "expected_numbers": ["14", "15", "45"],
    },
    {
        "id": "s_30",
        "raw": "Święty Jan napisał ewangelię w I wieku.",
        "expected_numbers": ["1"],
    },
]


def _get_loadavg() -> list[float]:
    try:
        return list(os.getloadavg())
    except Exception:
        return [0.0, 0.0, 0.0]


def _canonicalize_for_eval(text: str) -> str:
    """Canonicalize text by converting words and numbers to normalized speech representation."""
    norm = normalize_for_tts(text, "pl").lower()
    norm = re.sub(r"[^\w\s]", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def _check_numbers_present(transcript: str, expected_numbers: list[str]) -> bool:
    """Check if all expected numbers/dates/entities are present in the transcription."""
    canon_trans = _canonicalize_for_eval(transcript)
    for exp in expected_numbers:
        canon_exp = _canonicalize_for_eval(exp)
        if canon_exp not in canon_trans:
            # Also check if digit appears directly
            if exp not in transcript:
                return False
    return True


def _compute_wer(ref: str, hyp: str) -> float:
    r_words = ref.split()
    h_words = hyp.split()
    if not r_words:
        return 0.0 if not h_words else 1.0

    d = np.zeros((len(r_words) + 1, len(h_words) + 1), dtype=int)
    for i in range(len(r_words) + 1):
        d[i][0] = i
    for j in range(len(h_words) + 1):
        d[0][j] = j

    for i in range(1, len(r_words) + 1):
        for j in range(1, len(h_words) + 1):
            if r_words[i - 1] == h_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return float(d[len(r_words)][len(h_words)]) / float(len(r_words))


async def main() -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    loadavg_start = _get_loadavg()
    t_start = time.perf_counter()

    print("=" * 80)
    print("GAJA TTS NORMALIZER OBJECTIVE EVALUATION (Piper gosia + Whisper small CPU int8)")
    print(f"Start loadavg: {loadavg_start}")
    print(f"Test cases: {len(TEST_SENTENCES)}")
    print("=" * 80)

    # 1. Initialize Piper TTS
    print("Loading Piper TTS (gosia, CPU, threads=2)...")
    tts = PiperTTS("piper", MODEL_VOICE_PATH, language="pl", threads=2)
    await tts.warm_up()

    # 2. Initialize Whisper on CPU int8
    print("Loading faster-whisper small (CPU, int8)...")
    whisper = WhisperModel("small", device="cpu", compute_type="int8")

    results_without = []
    results_with = []

    for idx, item in enumerate(TEST_SENTENCES, 1):
        sid = item["id"]
        raw_text = item["raw"]
        expected = item["expected_numbers"]
        norm_text = normalize_for_tts(raw_text, "pl")

        print(f"\n[{idx}/{len(TEST_SENTENCES)}] Evaluating {sid}...")
        print(f"  Raw:  \"{raw_text}\"")
        print(f"  Norm: \"{norm_text}\"")

        # --- CONDITION 1: WITHOUT NORMALIZATION ---
        t0 = time.perf_counter()
        audio_unnorm = await tts.synthesize(raw_text, Style(), "pl")
        tts_unnorm_ms = (time.perf_counter() - t0) * 1000

        # Transcribe
        segments, _ = whisper.transcribe(audio_unnorm, language="pl", beam_size=1)
        transcript_unnorm = " ".join(seg.text for seg in segments).strip()
        ok_unnorm = _check_numbers_present(transcript_unnorm, expected)

        # WER against normalized reference
        ref_canon = _canonicalize_for_eval(raw_text)
        hyp_unnorm_canon = _canonicalize_for_eval(transcript_unnorm)
        wer_unnorm = _compute_wer(ref_canon, hyp_unnorm_canon)

        print(f"  [Unnormalized TTS] -> ASR: \"{transcript_unnorm}\" (Numbers OK: {ok_unnorm}, WER: {wer_unnorm:.2f})")

        results_without.append({
            "id": sid,
            "text": raw_text,
            "transcript": transcript_unnorm,
            "numbers_ok": ok_unnorm,
            "wer": round(wer_unnorm, 3),
            "tts_latency_ms": round(tts_unnorm_ms, 1),
        })

        # --- CONDITION 2: WITH NORMALIZATION ---
        t0 = time.perf_counter()
        audio_norm = await tts.synthesize(norm_text, Style(), "pl")
        tts_norm_ms = (time.perf_counter() - t0) * 1000

        # Transcribe
        segments, _ = whisper.transcribe(audio_norm, language="pl", beam_size=1)
        transcript_norm = " ".join(seg.text for seg in segments).strip()
        ok_norm = _check_numbers_present(transcript_norm, expected)

        hyp_norm_canon = _canonicalize_for_eval(transcript_norm)
        wer_norm = _compute_wer(ref_canon, hyp_norm_canon)

        print(f"  [Normalized TTS]   -> ASR: \"{transcript_norm}\" (Numbers OK: {ok_norm}, WER: {wer_norm:.2f})")

        results_with.append({
            "id": sid,
            "text": norm_text,
            "transcript": transcript_norm,
            "numbers_ok": ok_norm,
            "wer": round(wer_norm, 3),
            "tts_latency_ms": round(tts_norm_ms, 1),
        })

    # Summary metrics
    n = len(TEST_SENTENCES)
    acc_unnorm = sum(1 for r in results_without if r["numbers_ok"]) / n
    acc_norm = sum(1 for r in results_with if r["numbers_ok"]) / n
    mean_wer_unnorm = sum(r["wer"] for r in results_without) / n
    mean_wer_norm = sum(r["wer"] for r in results_with) / n

    loadavg_end = _get_loadavg()
    total_eval_duration_s = time.perf_counter() - t_start

    summary = {
        "n_samples": n,
        "loadavg_start": loadavg_start,
        "loadavg_end": loadavg_end,
        "duration_total_s": round(total_eval_duration_s, 1),
        "without_normalization": {
            "numbers_correct_rate": round(acc_unnorm, 4),
            "mean_wer": round(mean_wer_unnorm, 4),
            "results": results_without,
        },
        "with_normalization": {
            "numbers_correct_rate": round(acc_norm, 4),
            "mean_wer": round(mean_wer_norm, 4),
            "results": results_with,
        },
        "improvement": {
            "numbers_correct_delta_pp": round((acc_norm - acc_unnorm) * 100, 2),
            "wer_reduction_pp": round((mean_wer_unnorm - mean_wer_norm) * 100, 2),
        },
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("OBJECTIVE EVALUATION SUMMARY")
    print(f"Numbers Correct Rate WITHOUT normalization: {acc_unnorm * 100:.1f}%")
    print(f"Numbers Correct Rate WITH normalization:    {acc_norm * 100:.1f}% (+{(acc_norm - acc_unnorm) * 100:.1f} pp)")
    print(f"Mean WER WITHOUT normalization:            {mean_wer_unnorm * 100:.1f}%")
    print(f"Mean WER WITH normalization:               {mean_wer_norm * 100:.1f}% (-{(mean_wer_unnorm - mean_wer_norm) * 100:.1f} pp)")
    print(f"End loadavg: {loadavg_end}")
    print(f"Results saved to: {RESULTS_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
