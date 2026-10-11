#!/usr/bin/env bash
# Phase 2 (after run_stt_gpu.sh): Whistle CPU re-measure, compute-type check, larger models under the 3900 MiB GPU cap,
# then co-residency runs. Every timed run holds /tmp/gaja-bench.lock.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
B=bench/stt_bench.py
TURBO=$(ls -d ~/.cache/huggingface/hub/models--deepdml--faster-whisper-large-v3-turbo-ct2/snapshots/*/ 2>/dev/null | head -1)
TURBO=${TURBO%/}
SETS=wer,latency,hallu

until grep -q "PHASE1 DONE" bench/logs/stt_gpu_matrix.log; do sleep 20; done

run() {
  local tag=$1; shift
  echo "=== $(date +%T) START $tag load=$(cut -d' ' -f1-3 /proc/loadavg)"
  flock /tmp/gaja-bench.lock $PY $B "$@" --warmup 3 --tag "$tag" > "bench/logs/stt_${tag}.log" 2>&1
  local code=$?
  echo "=== $(date +%T) EXIT $code $tag load_after=$(cut -d' ' -f1-3 /proc/loadavg)"
  return $code
}

run whistle_py_pl_forced_rerun --engine whistle --lang pl --sets $SETS

run gpu_fw_small_int8f32_b5_pl --engine faster-whisper --model small --lang pl --beam 5 --sets $SETS --device cuda --compute int8_float32
run gpu_fw_small_int8f32_b5_en --engine faster-whisper --model small --lang en --beam 5 --sets $SETS --device cuda --compute int8_float32

if run gpu_fw_medium_int8_b5_pl --engine faster-whisper --model medium --lang pl --beam 5 --sets $SETS --device cuda --compute int8; then
  run gpu_fw_medium_int8_b5_en --engine faster-whisper --model medium --lang en --beam 5 --sets $SETS --device cuda --compute int8
fi

if [ -n "$TURBO" ]; then
  if run gpu_fw_turbo_int8_b5_pl --engine faster-whisper --model "$TURBO" --lang pl --beam 5 --sets $SETS --device cuda --compute int8; then
    run gpu_fw_turbo_int8_b5_en --engine faster-whisper --model "$TURBO" --lang en --beam 5 --sets $SETS --device cuda --compute int8
  fi
fi

echo "=== $(date +%T) PHASE2 BENCH DONE"
