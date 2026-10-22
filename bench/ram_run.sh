#!/usr/bin/env bash
# One isolated RAM measurement at a time: check free RAM, take the bench lock, run the probe,
# save output under bench/results/ram/. Usage: ram_run.sh <name> <probe args...>
set -u
NAME="$1"; shift
OUT="bench/results/ram/${NAME}.txt"
avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
{
  echo "=== $NAME $(date -Is) MemAvailable ${avail} MB env: $(env | grep -E '^(MALLOC|OMP|OMP_NUM)' | tr '\n' ' ') ==="
  if [ "$avail" -lt 2500 ]; then
    echo "SKIP: MemAvailable ${avail} MB < 2500 MB"
    exit 0
  fi
  exec flock /tmp/gaja-bench.lock uv run python -I bench/ram_probe.py "$@"
} > "$OUT" 2>&1
cat "$OUT"
