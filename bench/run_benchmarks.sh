#!/usr/bin/env bash
# ==============================================================================
# Gaja STT Benchmarking Suite
#
# Runs the full STT quantization evaluation matrix, co-residency tests,
# hotwords cost analysis, and false-positive checks.
#
# Usage:
#   ./bench/run_benchmarks.sh [all|matrix|coresident|hotwords|fp]
#
# Constraints:
#   - Holds /tmp/gaja-bench.lock during execution
#   - Stops/restarts gaja-adapter and gaja-llm services cleanly
#   - Automatically restores services and verifies health on exit
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=".venv/bin/python"
LOCK_FD=200

MODE="${1:-all}"

exec 200>/tmp/gaja-bench.lock
echo "Acquiring /tmp/gaja-bench.lock..."
flock 200

restore_services() {
    echo "Restoring services on exit..."
    echo "$(date -Iseconds) restarting: restoring services after benchmark" >> /tmp/gaja-adapter-restart.log
    systemctl --user start gaja-llm.service || true
    for i in {1..30}; do
        if curl -s http://127.0.0.1:8791/health >/dev/null 2>&1; then break; fi
        sleep 1
    done
    systemctl --user start gaja-adapter.service || true
    for i in {1..30}; do
        if curl -s http://127.0.0.1:8790/health >/dev/null 2>&1; then break; fi
        sleep 1
    done
    echo "Services restored and verified."
}
trap restore_services EXIT

stop_services() {
    echo "Stopping user services for clean GPU measurement..."
    echo "$(date -Iseconds) restarting: stopping services for clean GPU benchmark" >> /tmp/gaja-adapter-restart.log
    systemctl --user stop gaja-adapter.service gaja-llm.service || true
    sleep 2
}

run_matrix() {
    echo "=== Running Quantization Benchmark Matrix ==="
    stop_services

    # 1. faster-whisper variants
    echo "--- faster-whisper variants ---"
    $PYTHON bench/run_quant_benchmark.py --variant fw_small_int8_b1
    $PYTHON bench/run_quant_benchmark.py --variant fw_small_int8f32_b1
    $PYTHON bench/run_quant_benchmark.py --variant fw_small_f32_b1
    $PYTHON bench/run_quant_benchmark.py --variant fw_small_int8_b2
    $PYTHON bench/run_quant_benchmark.py --variant fw_medium_int8_b1
    $PYTHON bench/run_quant_benchmark.py --variant fw_medium_int8f32_b1
    $PYTHON bench/run_quant_benchmark.py --variant fw_large-v3-turbo_int8_b1
    $PYTHON bench/run_quant_benchmark.py --variant fw_large-v3-turbo_int8f32_b1
    $PYTHON bench/run_quant_benchmark.py --variant fw_large-v3-turbo_int8_b2

    # 2. whisper.cpp variants
    echo "--- whisper.cpp variants ---"
    $PYTHON bench/run_quant_benchmark.py --variant wcpp_small_q8_0_b1
    $PYTHON bench/run_quant_benchmark.py --variant wcpp_medium_q4_0_b1
    $PYTHON bench/run_quant_benchmark.py --variant wcpp_medium_q5_0_b1
    $PYTHON bench/run_quant_benchmark.py --variant wcpp_medium_q5_1_b1
    $PYTHON bench/run_quant_benchmark.py --variant wcpp_medium_q8_0_b1
    $PYTHON bench/run_quant_benchmark.py --variant wcpp_large_v3_turbo_q4_0_b1
    $PYTHON bench/run_quant_benchmark.py --variant wcpp_large_v3_turbo_q5_0_b1
    $PYTHON bench/run_quant_benchmark.py --variant wcpp_large_v3_turbo_q5_1_b1
    $PYTHON bench/run_quant_benchmark.py --variant wcpp_large_v3_turbo_q8_0_b1
}

run_coresident() {
    echo "=== Running Co-resident Benchmarks ==="
    echo "Starting gaja-llm.service for co-residency evaluation..."
    systemctl --user start gaja-llm.service
    for i in {1..30}; do
        if curl -s http://127.0.0.1:8791/health >/dev/null 2>&1; then break; fi
        sleep 1
    done

    echo "Running LLM-only baseline (30s)..."
    $PYTHON bench/stt_coresident.py --mode llm --seconds 30 --tag llm_only

    echo "Running co-resident STT + LLM (small int8 beam 1)..."
    $PYTHON bench/stt_coresident.py --mode both --model small --beam 1 --compute int8 --seconds 30 --tag both_small_b1
}

run_hotwords() {
    echo "=== Running Hotwords Cost Evaluation ==="
    $PYTHON bench/eval_hotwords_cost.py
}

run_fp() {
    echo "=== Running Vocabulary False-Positive Check ==="
    $PYTHON bench/vocab_fp_eval.py
}

case "$MODE" in
    matrix)
        run_matrix
        ;;
    coresident)
        run_coresident
        ;;
    hotwords)
        run_hotwords
        ;;
    fp)
        run_fp
        ;;
    all)
        run_matrix
        run_coresident
        run_hotwords
        run_fp
        ;;
    *)
        echo "Unknown mode: $MODE (choose: all, matrix, coresident, hotwords, fp)"
        exit 1
        ;;
esac

echo "=== Benchmark Suite Finished Successfully ==="
