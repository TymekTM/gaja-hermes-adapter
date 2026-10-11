#!/usr/bin/env bash
# GPU faster-whisper matrix on the same FLEURS data as run_stt_matrix.sh. Each timed run holds /tmp/gaja-bench.lock.
# Detached: nohup bash bench/run_stt_gpu.sh > bench/logs/stt_gpu_matrix.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
B=bench/stt_bench.py
SETS=wer,latency,hallu

run() {
  local tag=$1; shift
  echo "=== $(date +%T) START $tag load=$(cut -d' ' -f1-3 /proc/loadavg)"
  flock /tmp/gaja-bench.lock $PY $B "$@" --sets $SETS --device cuda --compute int8 --warmup 3 --tag "$tag" \
    > "bench/logs/stt_${tag}.log" 2>&1
  echo "=== $(date +%T) EXIT $? $tag load_after=$(cut -d' ' -f1-3 /proc/loadavg)"
}

for lang in pl en; do
  run gpu_fw_small_int8_b5_$lang --engine faster-whisper --model small --lang $lang --beam 5
  run gpu_fw_small_int8_b1_$lang --engine faster-whisper --model small --lang $lang --beam 1
  run gpu_fw_base_int8_b5_$lang --engine faster-whisper --model base --lang $lang --beam 5
  run gpu_fw_base_int8_b1_$lang --engine faster-whisper --model base --lang $lang --beam 1
  run gpu_fw_small_int8_b5_nocondprev_$lang --engine faster-whisper --model small --lang $lang --beam 5 --no-cond-prev
  run gpu_fw_small_int8_b5_notimestamps_$lang --engine faster-whisper --model small --lang $lang --beam 5 --without-timestamps
  run gpu_fw_small_int8_b5_nst03_$lang --engine faster-whisper --model small --lang $lang --beam 5 --no-speech-threshold 0.3
done

echo "=== $(date +%T) PHASE1 DONE"
