#!/usr/bin/env bash
# Sequential CPU STT matrix. Run detached: nohup bash bench/run_stt_matrix.sh > bench/logs/stt_matrix.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
B=bench/stt_bench.py

run() {
  echo "=== $(date +%T) START $*"
  $PY $B "$@" > "bench/logs/stt_$(echo "$*" | sed -E 's/.*--tag ([^ ]+).*/\1/').log" 2>&1
  echo "=== $(date +%T) EXIT $? $*"
}

run --engine whistle --lang pl --sets wer,latency,hallu --tag whistle_py_pl_forced
run --engine whistle --lang pl --lang-mode auto --sets wer --tag whistle_py_pl_auto
run --engine whistle --lang en --sets wer,latency,hallu --tag whistle_py_en_forced
run --engine whistle --lang en --lang-mode auto --sets wer --tag whistle_py_en_auto

run --engine whistle-cli --lang pl --sets wer --tag whistle_cli_pl_default
run --engine whistle-cli --lang pl --sets wer --depth 2 --tag whistle_cli_pl_depth2
run --engine whistle-cli --lang pl --sets wer --depth 4 --tag whistle_cli_pl_depth4
run --engine whistle-cli --lang pl --sets wer --threads 1 --tag whistle_cli_pl_threads1
run --engine whistle-cli --lang pl --sets wer --threads 2 --tag whistle_cli_pl_threads2
run --engine whistle-cli --lang pl --sets wer --threads 4 --tag whistle_cli_pl_threads4

run --engine faster-whisper --model small --lang pl --sets wer,latency,hallu --tag fw_small_cpu_pl
run --engine faster-whisper --model small --lang en --sets wer,latency,hallu --tag fw_small_cpu_en
run --engine faster-whisper --model base --lang pl --sets wer,latency,hallu --tag fw_base_cpu_pl
run --engine faster-whisper --model base --lang en --sets wer,latency,hallu --tag fw_base_cpu_en

run --engine whistle --lang pl --sets lat4 --tag whistle_py_pl_lat4_baseline
run --engine whistle --lang pl --sets lat4 --contention --tag whistle_py_pl_lat4_contention

echo "=== $(date +%T) MATRIX DONE"
