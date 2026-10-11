#!/usr/bin/env bash
# One documented command for the end-to-end run against the live adapter (ws://127.0.0.1:8790/ws):
#   bench/e2e.sh                 all scenarios: PL audio x10, EN audio x4, typed PL+EN, barge-in x2, Hermes handoff, routing x8
#   bench/e2e.sh --pl-clips 3 --scenarios pl,text   any subset; extra args go to scripts/e2e_suite.py
# Runs under /tmp/gaja-bench.lock so timed runs never overlap. Writes bench/out/e2e/results.json and run.log.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bench/out/e2e
ARGS=("$@")
flock /tmp/gaja-bench.lock bash -c '
  echo "loadavg before: $(cut -d" " -f1-3 /proc/loadavg)"
  PYTHONPATH=. .venv/bin/python scripts/e2e_suite.py --scenarios all "$@"
  echo "loadavg after:  $(cut -d" " -f1-3 /proc/loadavg)"
' e2e "${ARGS[@]}" 2>&1 | tee bench/out/e2e/run.log
