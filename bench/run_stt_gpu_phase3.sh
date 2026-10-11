#!/usr/bin/env bash
# Phase 3 (after run_stt_gpu_phase2.sh): hallucination fix via log_prob_threshold=None, then co-residency with the LFM
# llama-server on 127.0.0.1:8791. Every timed run holds /tmp/gaja-bench.lock.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SETS=wer,latency,hallu

until grep -q "PHASE2 BENCH DONE" bench/logs/stt_gpu_phase2.log; do sleep 20; done

run() {
  local tag=$1; shift
  echo "=== $(date +%T) START $tag load=$(cut -d' ' -f1-3 /proc/loadavg)"
  flock /tmp/gaja-bench.lock $PY "$@" --tag "$tag" > "bench/logs/${tag}.log" 2>&1
  local code=$?
  echo "=== $(date +%T) EXIT $code $tag load_after=$(cut -d' ' -f1-3 /proc/loadavg)"
}

for lang in pl en; do
  run gpu_fw_small_int8_b5_nologprob_$lang bench/stt_bench.py --engine faster-whisper --model small --lang $lang --beam 5 \
    --log-prob-threshold none --sets $SETS --device cuda --compute int8 --warmup 3
done

run llm_only bench/stt_coresident.py --mode llm --seconds 60
run stt_small_b5 bench/stt_coresident.py --mode stt --model small --beam 5 --seconds 60
run both_small_b5 bench/stt_coresident.py --mode both --model small --beam 5 --seconds 60
run stt_base_b5 bench/stt_coresident.py --mode stt --model base --beam 5 --seconds 60
run both_base_b5 bench/stt_coresident.py --mode both --model base --beam 5 --seconds 60

echo "=== $(date +%T) PHASE3 DONE"
