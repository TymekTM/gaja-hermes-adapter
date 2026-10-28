# LLM Noisy Evaluation Baseline (LFM3-1.7B)

This evaluation measures the resilience of Gaja's local LLM (LiquidAI LFM3-1.7B on llama-server at `127.0.0.1:8791`) and router pipeline against far-field Polish speech transcripts exhibiting typical ASR errors: phonetic misspellings, split/merged words, dropped prepositions, and English intrusions.

## Evaluation Setup

- **Dataset**: `bench/data/llm_pl_noisy.jsonl` (214 cases: 107 clean, 107 ASR-noisy).
- **Execution Command**:
  ```bash
  .venv/bin/python bench/llm_noisy_eval.py
  ```
- **Configuration**:
  - Non-streaming `/v1/chat/completions` with `temperature: 0`, `max_tokens: 200`, `cache_prompt: true`, `chat_template_kwargs: {"enable_thinking": false}`.
  - Full system prompt from `prompts/system_pl.md` with dynamic date injected via `Services.get_system_prompt`.
  - Full 15-tool runtime registry schemas (`build_runtime_registry(settings).schemas()`) with dummy Home Assistant probe credentials.
  - Concurrency 1 with 0.5s pause between consecutive calls.
- **Result Artifact**: `bench/results/llm_noisy_2026-10-28.json` (gitignored).

## Baseline Metrics Table

All values are extracted directly from `bench/results/llm_noisy_2026-10-28.json`:

| Metric | Overall (N=214) | Clean (N=107) | ASR Noisy (N=107) | Gap (ASR vs Clean) |
|---|---|---|---|---|
| **Route Accuracy** | **87.4%** | 90.7% | 84.1% | -6.6 pp |
| **Tool Accuracy** (tool cases, N=174) | **82.2%** | 86.2% | 78.2% | -8.0 pp |
| **False Tool Call Rate** (non-tool, N=40) | **15.0%** | 10.0% | 20.0% | +10.0 pp |
| **Polish-Only Rate** | **100.0%** | 100.0% | 100.0% | 0.0 pp |
| **Think-Leak Rate** (reasoning that would be spoken) | **0.5%** | 0.9% | 0.0% | -0.9 pp |
| **Max-Token Hit Rate** (200 tokens) | **0.9%** | 1.9% | 0.0% | -1.9 pp |
| **Mean Latency** | 1149.4 ms | 1162.8 ms | 1136.1 ms | -26.7 ms |
| **Median Latency** | 926.1 ms | 891.5 ms | 934.1 ms | +42.6 ms |
| **P95 Latency** | 3186.5 ms | 3294.1 ms | 3223.9 ms | -70.2 ms |
| **Mean Completion Tokens** | 33.8 | 33.7 | 33.9 | +0.2 tokens |

## Key Findings

1. **ASR Noise Degrades Tool & Route Accuracy**:
   - ASR noise drops route accuracy from 90.7% to 84.1% (-6.6 pp) and tool accuracy from 86.2% to 78.2% (-8.0 pp).
   - False tool calls double from 10.0% to 20.0% under ASR noise (e.g. conversational chit-chat falsely triggering tools like `exchange_rate`, `server_status`, or `calculate`).
2. **Think-Tag Leakage**:
   - Every completion starts with a stray `</think>`, which the client's ThinkFilter (`gaja/llm/parse.py`) drops, so it is not counted. One case (time_02) leaked reasoning beyond it. Rate recomputed from the same JSON with the fixed check.
3. **Language Consistency**:
   - Polish-only rate remained 100% across all evaluated turns when evaluated with `temperature: 0` (though ASR English intrusions caused tool misrouting rather than full English replies).
