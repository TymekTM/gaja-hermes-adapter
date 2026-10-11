"""Compare audio-turn latency between e2e result files (bench/out/e2e/*.json from scripts/e2e_suite.py).

    .venv/bin/python bench/e2e_compare.py bench/out/e2e/r6_baseline.json bench/out/e2e/r6_after.json

Recomputes p50/p90 from the per-turn records (older files have no "summary"), audio turns only
(scenarios pl/en). Turns without a wall time (e.g. a clip whose speech was merged into the previous
turn) are listed but not counted in the wall column.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

FIELDS = ("stt_ms", "router_ms", "llm_ttft_ms", "tool_select_ms", "tool_ms", "first_sentence_ms",
          "tts_first_ms", "e2e_first_audio_ms")


def stats(values: list[float]) -> str:
    if not values:
        return "-"
    return f"{np.percentile(values, 50):.0f} / {np.percentile(values, 90):.0f} (n={len(values)})"


def main() -> None:
    reports = [(Path(p).name, json.loads(Path(p).read_text())) for p in sys.argv[1:]]
    print(f"{'audio turns, p50 / p90 ms':26}" + "".join(f"{name:>28}" for name, _ in reports))
    for field in FIELDS + ("wall_from_speech_end_ms",):
        row = []
        for _, report in reports:
            turns = [t for t in report["turns"] if t["scenario"] in ("pl", "en")]
            if field == "wall_from_speech_end_ms":
                values = [t[field] for t in turns if t.get(field) is not None]
            else:
                values = [t["metrics"][field] for t in turns if t["metrics"].get(field) is not None]
            row.append(stats(values))
        print(f"{field:26}" + "".join(f"{cell:>28}" for cell in row))
    print(f"{'routes':26}" + "".join(
        f"{','.join(sorted({str(t['metrics'].get('route')) for t in r['turns'] if t['scenario'] in ('pl', 'en')})):>28}"
        for _, r in reports))
    print(f"{'loadavg start -> end':26}" + "".join(
        f"{' '.join(r['load_start'][:1]) + ' -> ' + ' '.join(r['load_end'][:1]):>28}" for _, r in reports))


if __name__ == "__main__":
    main()
