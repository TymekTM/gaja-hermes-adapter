#!/bin/bash
set -ex

exec 200>/tmp/gaja-bench.lock
echo "[SCOUT] Waiting for /tmp/gaja-bench.lock..."
flock 200
echo "[SCOUT] Bench lock acquired at $(date)!"

# 1. Piper Gosia (medium)
echo "[SCOUT] Benchmarking Piper Gosia..."
.venv/bin/python bench/tts_bench.py \
    --engine piper \
    --piper-model bench/models/piper/pl_PL-gosia-medium.onnx \
    --tag gosia

# 2. Piper Bass (high)
echo "[SCOUT] Benchmarking Piper Bass..."
.venv/bin/python bench/tts_bench.py \
    --engine piper \
    --piper-model bench/models/piper/pl_PL-bass-high.onnx \
    --tag bass

# 3. Meta MMS-TTS Polish (GPU CUDA)
echo "[SCOUT] Benchmarking Meta MMS-TTS..."
.venv/bin/python bench/tts_bench.py \
    --engine mms \
    --mms-device cuda \
    --tag cuda

# 4. Supertonic 3 (CPU, 4 steps)
echo "[SCOUT] Benchmarking Supertonic 3..."
.venv/bin/python bench/tts_bench.py \
    --engine supertonic \
    --supertonic-steps 4 \
    --tag steps4

echo "[SCOUT] All benchmarks finished successfully at $(date)!"
