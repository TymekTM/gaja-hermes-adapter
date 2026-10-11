#!/usr/bin/env bash
# Quiet-window re-measurement for docs/TTS_RESULTS.md, plus proxies, demos and the report.
#
#   bash bench/tts_final.sh [all|piper|supertonic|en-mms|demos|proxy|report]
#
# Every timed run takes /tmp/gaja-bench.lock. Before each run, the 1-min loadavg must drop below
# QUIET_MAX (1.5) or the run waits up to QUIET_WAIT_S (180) seconds, still holding the lock. If the
# box never gets quiet, the run goes ahead and its JSON is marked "contaminated": true (the report
# shows that). The loadavg is also written into every result JSON. Stages can run one at a time to
# stay under a 10-minute foreground limit; "all" runs everything in order.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOCK=/tmp/gaja-bench.lock
STAGE=${1:-all}
SUPER_VOICE=${SUPER_VOICE:-M4}
QUIET_MAX=1.5
QUIET_WAIT_S=180
GOSIA=bench/models/piper/pl_PL-gosia-medium.onnx
DARKMAN=models/pl_PL-darkman-medium.onnx
MC=bench/models/piper/pl_PL-mc_speech-medium.onnx
BASS=bench/models/piper/pl_PL-bass-high.onnx

for f in "$DARKMAN" "$GOSIA" "$MC" "$BASS"; do
  [ -f "$f" ] || { echo "missing model $f" >&2; exit 1; }
done

quiet_gate() {
  local waited=0 l
  while :; do
    l=$(cut -d' ' -f1 /proc/loadavg)
    if awk -v l="$l" -v m="$QUIET_MAX" 'BEGIN{exit !(l<m)}'; then break; fi
    if [ "$waited" -ge "$QUIET_WAIT_S" ]; then
      echo "quiet gate timed out at 1-min load $l; this run will be marked contaminated" >&2
      break
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "gate: waited ${waited}s, loadavg $(cat /proc/loadavg)"
}
export -f quiet_gate
export QUIET_MAX QUIET_WAIT_S

timed() {
  flock "$LOCK" bash -c 'quiet_gate; exec "$@"' _ "$PY" "$@"
}

want() { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }

echo "stage=$STAGE loadavg before: $(cat /proc/loadavg)"

if want piper; then
  for spec in "final_darkman:$DARKMAN" "final_gosia:$GOSIA" "final_mc_speech:$MC" "final_bass-high:$BASS"; do
    tag=${spec%%:*}; model=${spec#*:}
    timed bench/tts_bench.py --engine piper --piper-model "$model" --piper-threads 2 --tag "$tag" --langs pl --skip-mos
    timed bench/tts_bench.py --engine piper --piper-model "$model" --piper-threads 2 --tag "${tag}_probe" --langs pl --set probe --skip-wer --skip-mos
  done
fi

if want supertonic; then
  SUPER=(--engine supertonic --supertonic-steps 6 --supertonic-threads 2 --supertonic-style "$SUPER_VOICE" --skip-mos)
  timed bench/tts_bench.py "${SUPER[@]}" --supertonic-sr 22050 --tag final --langs pl
  timed bench/tts_bench.py "${SUPER[@]}" --supertonic-sr 22050 --tag final_probe --langs pl --set probe --skip-wer
  timed bench/tts_bench.py "${SUPER[@]}" --supertonic-sr 0 --tag native_M4 --langs pl
  timed bench/tts_bench.py "${SUPER[@]}" --supertonic-sr 0 --tag native_M4_probe --langs pl --set probe --skip-wer
fi

if want en-mms; then
  timed bench/tts_bench.py --engine paradee --threads 1 --tag final_en --langs en --skip-mos
  timed bench/tts_bench.py --engine paradee --threads 1 --tag final_en_probe --langs en --set probe --skip-wer --skip-mos
  timed bench/tts_bench.py --engine mms --mms-device cpu --tag final_cpu --langs pl --skip-mos
  timed bench/tts_bench.py --engine mms --mms-device cpu --tag final_cpu_probe --langs pl --set probe --skip-wer --skip-mos
fi

if want demos; then
  timed bench/prosody_demo.py --engine piper_gosia --piper-threads 2
  timed bench/prosody_demo.py --engine piper_darkman --piper-threads 2
  timed bench/prosody_demo.py --engine supertonic --supertonic-voice "$SUPER_VOICE" --supertonic-steps 6 --supertonic-threads 2 --supertonic-sr 22050
  timed bench/prosody_demo.py --engine paradee --paradee-threads 1
fi

if want proxy; then
  flock "$LOCK" "$PY" bench/tts_quality_proxy.py
fi

if want report; then
  "$PY" bench/tts_report.py
fi

echo "stage=$STAGE loadavg after: $(cat /proc/loadavg)"
