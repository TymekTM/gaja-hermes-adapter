#!/usr/bin/env python3
"""Summarize device wake detections and server verdicts from adapter journal logs.

Reads journal text from stdin (e.g. ``journalctl --user -u gaja-adapter -o short-iso``),
or runs ``journalctl`` itself when ``--since`` / ``--until`` is given.

    uv run python scripts/wake_stats.py --since 2026-10-29
    journalctl --user -u gaja-adapter -o short-iso --since today | uv run python scripts/wake_stats.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

_DATE_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})(?:[T ]|$)")
_WAKE_RE = re.compile(
    r"\bdevice\s+(\S+)\s+wake\s+word=.*?\s+score=([0-9]+(?:\.[0-9]+)?)\s+busy=(True|False)\s+verify=(True|False)\b"
)
_VERDICT_RE = re.compile(
    r"\bdevice\s+(\S+)\s+wake_verdict\s+accepted=(True|False)\s+reason=(\S+)\s+score=([0-9]+(?:\.[0-9]+)?)\s+transcript=(.*)$"
)
_REASON_ORDER = ("ok", "unverified", "no_phrase", "no_speech", "stt_error", "timeout")


@dataclass
class WakeBucket:
    day: str
    device_id: str
    detections: int = 0
    verdicts: int = 0
    accepted: int = 0
    rejected: int = 0
    no_verdict: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def accept_rate(self) -> float | None:
        if self.verdicts == 0:
            return None
        return self.accepted / self.verdicts

    def to_dict(self) -> dict[str, Any]:
        rate = self.accept_rate
        return {
            "day": self.day,
            "device_id": self.device_id,
            "detections": self.detections,
            "verdicts": self.verdicts,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "accept_rate": round(rate, 4) if rate is not None else None,
            "no_verdict": self.no_verdict,
            "reasons": dict(sorted(self.reasons.items())),
        }


def _extract_day(line: str) -> str:
    match = _DATE_RE.match(line)
    if match is not None:
        return match.group(1)
    return "unknown"


def parse_journal(lines: Iterable[str], *, include_sim: bool = False) -> dict[str, Any]:
    buckets: dict[tuple[str, str], WakeBucket] = {}
    pending: dict[str, str] = {}

    def bucket_for(day: str, device_id: str) -> WakeBucket:
        key = (day, device_id)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = WakeBucket(day=day, device_id=device_id)
            buckets[key] = bucket
        return bucket

    for line in lines:
        verdict = _VERDICT_RE.search(line)
        if verdict is not None:
            device_id = verdict.group(1)
            if not include_sim and device_id.startswith("sim-"):
                continue
            accepted = verdict.group(2) == "True"
            reason = verdict.group(3)
            pending_day = pending.pop(device_id, None)
            day = pending_day if pending_day is not None else _extract_day(line)
            bucket = bucket_for(day, device_id)
            bucket.verdicts += 1
            if accepted:
                bucket.accepted += 1
            else:
                bucket.rejected += 1
            bucket.reasons[reason] = bucket.reasons.get(reason, 0) + 1
            continue

        wake = _WAKE_RE.search(line)
        if wake is not None:
            device_id = wake.group(1)
            if not include_sim and device_id.startswith("sim-"):
                continue
            day = _extract_day(line)
            bucket_for(day, device_id).detections += 1
            previous_day = pending.get(device_id)
            if previous_day is not None:
                bucket_for(previous_day, device_id).no_verdict += 1
            pending[device_id] = day

    for device_id, pending_day in pending.items():
        bucket_for(pending_day, device_id).no_verdict += 1

    ordered = [buckets[key] for key in sorted(buckets)]
    totals = WakeBucket(day="TOTAL", device_id="")
    for row in ordered:
        totals.detections += row.detections
        totals.verdicts += row.verdicts
        totals.accepted += row.accepted
        totals.rejected += row.rejected
        totals.no_verdict += row.no_verdict
        for reason, count in row.reasons.items():
            totals.reasons[reason] = totals.reasons.get(reason, 0) + count

    totals_dict = totals.to_dict()
    totals_dict.pop("day")
    totals_dict.pop("device_id")
    return {
        "rows": [row.to_dict() for row in ordered],
        "totals": totals_dict,
    }


def _ordered_reasons(reasons: dict[str, int]) -> list[str]:
    known = [r for r in _REASON_ORDER if r in reasons]
    extra = sorted(r for r in reasons if r not in _REASON_ORDER)
    return known + extra


def format_table(report: dict[str, Any]) -> str:
    rows: list[dict[str, Any]] = report["rows"]
    totals: dict[str, Any] = report["totals"]
    if not rows:
        return "No wake events found."

    reason_cols = _ordered_reasons(totals["reasons"])
    headers = ["day", "device", "detections", "verdicts", "accepted", "accept_rate", "no_verdict", *reason_cols]

    def format_rate(rate: float | None) -> str:
        return "-" if rate is None else f"{rate * 100:.1f}%"

    table_rows: list[list[str]] = []
    for row in rows:
        table_rows.append(
            [
                row["day"],
                row["device_id"],
                str(row["detections"]),
                str(row["verdicts"]),
                str(row["accepted"]),
                format_rate(row["accept_rate"]),
                str(row["no_verdict"]),
                *(str(row["reasons"].get(r, 0)) for r in reason_cols),
            ]
        )

    if len(rows) > 1:
        table_rows.append(
            [
                "TOTAL",
                "",
                str(totals["detections"]),
                str(totals["verdicts"]),
                str(totals["accepted"]),
                format_rate(totals["accept_rate"]),
                str(totals["no_verdict"]),
                *(str(totals["reasons"].get(r, 0)) for r in reason_cols),
            ]
        )

    widths = [max(len(headers[i]), *(len(r[i]) for r in table_rows)) for i in range(len(headers))]

    def render_line(cells: list[str]) -> str:
        parts = [cell.rjust(widths[i]) if i >= 2 else cell.ljust(widths[i]) for i, cell in enumerate(cells)]
        return "  ".join(parts).rstrip()

    lines = [render_line(headers), render_line(["-" * w for w in widths])]
    for r in table_rows:
        lines.append(render_line(r))
    return "\n".join(lines)


def read_journalctl(*, since: str | None = None, until: str | None = None) -> str:
    cmd = ["journalctl", "--user", "-u", "gaja-adapter", "-o", "short-iso", "--no-pager"]
    if since is not None:
        cmd.extend(["--since", since])
    if until is not None:
        cmd.extend(["--until", until])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or f"exit code {proc.returncode}"
        raise RuntimeError(f"journalctl failed: {detail}")
    return proc.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default=None, help="pass --since to journalctl --user -u gaja-adapter")
    parser.add_argument("--until", default=None, help="pass --until to journalctl --user -u gaja-adapter")
    parser.add_argument("--include-sim", action="store_true", help="include simulator devices (id starting with sim-)")
    parser.add_argument("--json", action="store_true", help="output JSON instead of a plain-text table")
    args = parser.parse_args(argv)

    if args.since is not None or args.until is not None:
        try:
            text = read_journalctl(since=args.since, until=args.until)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        lines: Iterable[str] = text.splitlines()
    else:
        lines = sys.stdin

    report = parse_journal(lines, include_sim=args.include_sim)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_table(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
