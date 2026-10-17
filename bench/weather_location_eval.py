#!/usr/bin/env python3
"""Offline evaluation of weather location resolution over tests/data/weather_locations.tsv."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gaja.tools.weather_tool import resolve_tool_location


def load_corpus(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 3:
                continue
            if row[0] == "utterance" and row[2] == "expected":
                continue
            rows.append((row[0], row[1], row[2]))
    return rows


def run_eval(corpus_path: Path | None = None) -> int:
    path = corpus_path or (REPO_ROOT / "tests/data/weather_locations.tsv")
    if not path.exists():
        print(f"Corpus file not found: {path}", file=sys.stderr)
        return 1

    rows = load_corpus(path)
    total = len(rows)
    correct = 0
    failures: list[tuple[str, str, str, str]] = []

    for utterance, llm_location, expected in rows:
        got = resolve_tool_location(llm_location or None, utterance, "Katowice")
        if got == expected:
            correct += 1
        else:
            failures.append((utterance, llm_location, expected, got))

    pct = (correct / total * 100.0) if total else 0.0
    print(f"Weather Location Accuracy: {correct}/{total} ({pct:.2f}%)")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for utt, llm, exp, got in failures:
            llm_repr = f"'{llm}'" if llm else "None"
            print(f"  FAIL: utterance='{utt}', llm={llm_repr} -> expected='{exp}', got='{got}'")
        return 1

    print("\nAll cases passed!")
    return 0


if __name__ == "__main__":
    sys.exit(run_eval())
