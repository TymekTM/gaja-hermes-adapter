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

## Re-run 2026-10-29 (fixed server, claimed-action metric)

During the first attempt the production llama-server was OOM-killed: its host-RAM prompt cache
(`--cache-ram`, 8 GiB by default) outgrew the unit's 3 GiB `MemoryHigh`. With that cache off
(`--cache-ram 0`; Gaja pins chat and router to their own slots, so it bought nothing), the eval was re-run with the current tree (original
`prompts/system_pl.md`, bench now retries transient slot failures) and a new per-case metric:

**Claimed-action-without-call** (`claimed_action_rate`): the raw reply announces a handoff
("Jasne, przekazuję to Hermesowi…"), writes a tool call as plain text ("Wywołaj narzędzie
handoff_to_hermes"), or claims a finished action ("Zrobiłem to teraz") while the router and the
model executed nothing for the turn — the speaker hears about work that never ran
("mówi, a nie robi"). Detectors live in `gaja/text/claims.py`; the eval measures the raw model,
so this metric counts the failures the pipeline guard must cover.

All values from `bench/results/llm_noisy_2026-10-29.json`:

| Metric | Overall (N=214) | Clean (N=107) | ASR Noisy (N=107) |
|---|---|---|---|
| **Route Accuracy** | **88.8%** | 94.4% | 83.2% |
| **Tool Accuracy** (tool cases, N=174) | **83.3%** | 86.2% | 80.5% |
| **False Tool Call Rate** (non-tool, N=40) | **15.0%** | 5.0% | 25.0% |
| **Claimed-Action-Without-Call Rate** | **0.0%** | 0.0% | 0.0% |
| **Polish-Only Rate** | **99.5%** | 100.0% | 99.1% |
| **Think-Leak Rate** | **0.0%** | 0.0% | 0.0% |
| **Max-Token Hit Rate** (200 tokens) | **0.5%** | 0.9% | 0.0% |
| **Mean Latency** | 790.4 ms | 813.2 ms | 767.6 ms |
| **Median Latency** | 692.8 ms | 691.5 ms | 706.6 ms |
| **P95 Latency** | 1659.3 ms | 1699.3 ms | 1648.2 ms |
| **Mean Completion Tokens** | 32.1 | 32.2 | 31.9 |

In this run the raw failures that motivated this metric did not reproduce (0/214; why the
2026-10-28 run produced them is not known). The runtime guard
covers them when they do happen: on a local reply, `Session._enqueue` checks every streamed
sentence before it reaches TTS, and an announced handoff, a written-out tool call or a claimed
action with no call behind it turns into a real `handoff_to_hermes` carrying the user's
utterance (the same safety-net path as the handoff-promise guard), so the action is performed
instead of spoken. Nothing is dropped silently, there is no extra model call and no added
latency on a normal turn, and sentences already played cannot be recalled — the guard acts at
the first offending sentence, before it is synthesized. Covered by `tests/test_action_claims.py`
(the pipeline tests fail with the guard disabled).

### Candidate prompt measured and reverted

A tightened handoff paragraph (announcement only together with the call, tool name banned from
reply text) was measured on the same day/server (`bench/results/llm_noisy_2026-10-29_candidate_prompt.json`):

| Metric | Overall (N=214) | Clean (N=107) | ASR Noisy (N=107) |
|---|---|---|---|
| **Route Accuracy** | **85.0%** | 89.7% | 80.4% |
| **Tool Accuracy** (tool cases, N=174) | **78.2%** | 81.6% | 74.7% |
| **False Tool Call Rate** (non-tool, N=40) | **10.0%** | 5.0% | 15.0% |
| **Claimed-Action-Without-Call Rate** | **0.5%** (1/214) | 0.0% | 0.9% |

It regressed route (-3.8 pp) and tool (-5.1 pp) accuracy — 12 tool-expecting cases flipped
wrong against 4 fixed — and did not reduce the claimed-action rate (the one case: `sun_10`
answered "o kturej w schut słonca" with "Jasne, przekazuję to Hermesowi, chwilkę." and no
call), so it was reverted and the original prompt kept.
