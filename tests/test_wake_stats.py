"""Tests for scripts/wake_stats.py."""

from __future__ import annotations

import io
import json
import subprocess

import pytest

from scripts import wake_stats

SYNTHETIC_JOURNAL = """\
2026-11-11T09:00:01+01:00 host python[1]: 2026-11-11 09:00:01,100 INFO gaja.devices.controller: device atom-salon wake word=Hej Gaja score=0.97 busy=False verify=True
2026-11-11T09:00:02+01:00 host python[1]: 2026-11-11 09:00:02,200 INFO gaja.devices.controller: device atom-salon wake_verdict accepted=True reason=ok score=0.97 transcript='Hej Gaja, jaka pogoda?'
2026-11-11T09:05:00+01:00 host python[1]: 2026-11-11 09:05:00,000 INFO gaja.devices.controller: device atom-salon wake word=hej_gaja score=0.95 busy=False verify=True
2026-11-11T09:05:01+01:00 host python[1]: 2026-11-11 09:05:01,000 INFO gaja.devices.controller: device atom-salon wake_verdict accepted=False reason=no_phrase score=0.95 transcript='Hej Maja'
2026-11-11T09:10:00+01:00 host python[1]: 2026-11-11 09:10:00,000 INFO gaja.devices.controller: device atom-salon wake word=hej_gaja score=0.96 busy=False verify=True
2026-11-11T09:10:02+01:00 host python[1]: 2026-11-11 09:10:02,000 INFO gaja.devices.controller: device atom-salon wake_verdict accepted=False reason=no_speech score=0.96 transcript=''
2026-11-11T09:15:00+01:00 host python[1]: 2026-11-11 09:15:00,000 INFO gaja.devices.controller: device sim-test wake word=hej_gaja score=0.99 busy=False verify=True
2026-11-11T09:15:01+01:00 host python[1]: 2026-11-11 09:15:01,000 INFO gaja.devices.controller: device sim-test wake_verdict accepted=True reason=ok score=0.99 transcript='Hej Gaja'
2026-11-12T10:00:00+01:00 host python[1]: 2026-11-12 10:00:00,000 INFO gaja.devices.controller: device atom-salon wake word=hej_gaja score=0.97 busy=False verify=True
2026-11-12T10:00:01+01:00 host python[1]: 2026-11-12 10:00:01,000 INFO gaja.devices.controller: device atom-salon wake_verdict accepted=False reason=stt_error score=0.97 transcript=''
2026-11-12T10:05:00+01:00 host python[1]: 2026-11-12 10:05:00,000 INFO gaja.devices.controller: device atom-salon wake word=hej_gaja score=0.97 busy=False verify=True
2026-11-12T10:05:08+01:00 host python[1]: 2026-11-12 10:05:08,000 INFO gaja.devices.controller: device atom-salon wake_verdict accepted=False reason=timeout score=0.97 transcript=''
2026-11-12T10:10:00+01:00 host python[1]: 2026-11-12 10:10:00,000 INFO gaja.devices.controller: device atom-salon wake word=hej_gaja score=0.97 busy=True verify=False
2026-11-12T10:10:00+01:00 host python[1]: 2026-11-12 10:10:00,010 INFO gaja.devices.controller: device atom-salon wake_verdict accepted=True reason=unverified score=0.97 transcript=''
2026-11-12T10:20:00+01:00 host python[1]: 2026-11-12 10:20:00,000 INFO gaja.devices.controller: device atom-salon wake word=hej_gaja score=0.97 busy=False verify=True
"""


def test_parse_journal_excludes_sim_by_default_and_aggregates_per_day_and_device() -> None:
    report = wake_stats.parse_journal(SYNTHETIC_JOURNAL.splitlines())
    assert report["rows"] == [
        {
            "day": "2026-11-11",
            "device_id": "atom-salon",
            "detections": 3,
            "verdicts": 3,
            "accepted": 1,
            "rejected": 2,
            "accept_rate": 0.3333,
            "no_verdict": 0,
            "reasons": {"no_phrase": 1, "no_speech": 1, "ok": 1},
        },
        {
            "day": "2026-11-12",
            "device_id": "atom-salon",
            "detections": 4,
            "verdicts": 3,
            "accepted": 1,
            "rejected": 2,
            "accept_rate": 0.3333,
            "no_verdict": 1,
            "reasons": {"stt_error": 1, "timeout": 1, "unverified": 1},
        },
    ]
    assert report["totals"] == {
        "detections": 7,
        "verdicts": 6,
        "accepted": 2,
        "rejected": 4,
        "accept_rate": 0.3333,
        "no_verdict": 1,
        "reasons": {
            "no_phrase": 1,
            "no_speech": 1,
            "ok": 1,
            "stt_error": 1,
            "timeout": 1,
            "unverified": 1,
        },
    }


def test_parse_journal_includes_sim_devices_when_requested() -> None:
    report = wake_stats.parse_journal(SYNTHETIC_JOURNAL.splitlines(), include_sim=True)
    devices = [(r["day"], r["device_id"]) for r in report["rows"]]
    assert devices == [
        ("2026-11-11", "atom-salon"),
        ("2026-11-11", "sim-test"),
        ("2026-11-12", "atom-salon"),
    ]
    assert report["totals"]["detections"] == 8
    assert report["totals"]["verdicts"] == 7
    assert report["totals"]["accepted"] == 3


def test_format_table_and_cli_json_and_journalctl(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(SYNTHETIC_JOURNAL))
    assert wake_stats.main([]) == 0
    table = capsys.readouterr().out
    assert "atom-salon" in table
    assert "sim-test" not in table
    assert "2026-11-11" in table and "2026-11-12" in table
    assert "33.3%" in table
    assert "no_phrase" in table and "stt_error" in table

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], capture_output: bool, text: bool, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=SYNTHETIC_JOURNAL, stderr="")

    monkeypatch.setattr(wake_stats.subprocess, "run", fake_run)
    assert wake_stats.main(["--since", "2026-11-11", "--until", "2026-11-13", "--include-sim", "--json"]) == 0
    assert calls == [
        [
            "journalctl",
            "--user",
            "-u",
            "gaja-adapter",
            "-o",
            "short-iso",
            "--no-pager",
            "--since",
            "2026-11-11",
            "--until",
            "2026-11-13",
        ]
    ]
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["rows"]) == 3
    assert payload["totals"]["no_verdict"] == 1
