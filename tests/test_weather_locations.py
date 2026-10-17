from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gaja.tools.weather_tool import resolve_tool_location

CORPUS_PATH = Path(__file__).resolve().parent / "data" / "weather_locations.tsv"


def _load_corpus() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 3:
                continue
            if row[0] == "utterance" and row[2] == "expected":
                continue
            rows.append((row[0], row[1], row[2]))
    return rows


_CORPUS_ROWS = _load_corpus()


@pytest.mark.parametrize(
    ("utterance", "llm_location", "expected"),
    _CORPUS_ROWS,
)
def test_weather_location_corpus(utterance: str, llm_location: str, expected: str):
    got = resolve_tool_location(llm_location or None, utterance, "Katowice")
    assert got == expected
