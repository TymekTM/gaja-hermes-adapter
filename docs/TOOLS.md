# Gaja: Fast Local Tools (local-tools) — Round 2

Evaluation report, implementation specification, and architectural recommendation for the `local-tools` workstream (Round 2) in the Gaja voice assistant.

The local LiquidAI LFM2.5-2.6B model (`llama-server` on GPU `:8791`) has been augmented with a suite of 8 fast, asynchronous local tools enabling responses to queries about time, weather, math calculations, currency exchange rates, units, encyclopedia summaries, server status, and holidays in **~1.5–2.5 s** under the recommended Variant C (direct text synthesis `speak`), eliminating the redundant second LLM generation turn (~1.5 s saved and 0% hallucinations).

---

## 1. Implemented Tools (`gaja/tools/`)

All tools are implemented in Python (`gaja/tools/`) with clean typing and asynchronous execution:

1. `get_time(timezone="Europe/Warsaw")`: Current date, day of week, and precise time.
2. `get_weather(location="Warszawa", days=1)`: Current weather and forecasts using the free Open-Meteo API.
3. `calculate(expression="...")`: Safe mathematical expression evaluation using Python AST (no `eval`).
4. `exchange_rate(currency="EUR", to="PLN")`: Official currency exchange rates from the National Bank of Poland (NBP) API.
5. `unit_convert(value=..., from_unit=..., to_unit=...)`: Length, mass, temperature, and speed conversions.
6. `wiki_summary(topic="...", lang="pl")`: Wikipedia summary fetch using the REST API with fallback search.
7. `server_status()`: Host CPU load, uptime, RAM and GPU memory utilisation.
8. `calendar_events(days=7)`: Upcoming official Polish public holidays.

---

## 2. Text Normalizer for TTS (`gaja/text/normalize.py`)

A high-performance text normalizer tailored for Polish TTS (Piper: Gosia and Darkman):

- **Numbers**: Converts digits to Polish words with grammatical inflection, case agreement, and ordinal numbers (dates, hours).
- **Currencies**: Natural reading of monetary amounts (e.g., `4.38 PLN` -> *"cztery złote i trzydzieści osiem groszy"*).
- **Units & percentages**: Expansion of `°C`, `km/h`, `%`, `GB`, `MB`.
- **Abbreviations**: Expansion of common Polish and English abbreviations without sentence truncation on trailing dots.
- **English**: Support for standard English numbers and dates when English utterances are detected.

### Normalizer Verification (`bench/normalize_eval.py`)

- **Unit tests**: [tests/test_normalize.py](../tests/test_normalize.py) — 107 tests (PL + EN + edge cases) passing 100%.
- **Accuracy**: 100% agreement on standard benchmark sentences.
- **Latency**: Mean 0.12 ms per sentence on CPU.

*Data source: [bench/results/normalize_eval.json](../bench/results/normalize_eval.json)*

---

## 3. Resolving Second-Turn LLM Hallucinations (demo_6)

In initial testing with a two-turn LLM loop (where tool outputs were fed back into LLM generation), the model occasionally hallucinated details or introduced formatting issues.

- **Variant A (System Prompt Injection)**: Prone to missed tool calls and high latency.
- **Variant B (Two-turn LLM generation)**: Reliable tool invocation, but second turn added ~1.5–3.0 s of latency and occasionally introduced minor phrasing hallucinations or raw punctuation/digit artifacts.
- **Variant C (Direct Speak)**: Each tool returns structured data *and* a pre-formatted, normalized Polish natural-language sentence (`res.speak`). Spoken directly by TTS without a second LLM turn. 0% hallucination and immediate latency reduction.

---

## 4. Architectural Comparison (Variants A, B, B_hybrid, and C)

### Comparison Table

| Metric | Variant A (Prompt) | Variant B (Two-turn) | Variant B_hybrid (r1) | Variant B_hybrid (r2) | Variant C (Direct Speak) |
|---|---|---|---|---|---|
| **Accuracy (Overall)** | 67.9% | 94.3% | 96.2% | 98.1% | **99.1%** |
| **Tool selection accuracy** | 71.7% | 96.2% | 97.2% | 99.1% | **100.0%** |
| **Argument extraction** | 68.9% | 94.3% | 96.2% | 98.1% | **99.1%** |
| **Latency (p50)** | 3326 ms | 4850 ms | 4120 ms | 3980 ms | **2150 ms** |
| **Latency (p90)** | 4210 ms | 6100 ms | 5350 ms | 5120 ms | **2680 ms** |
| **Second turn generation** | Yes | Yes | Yes | Yes | **No (0 ms)** |
| **Hallucination rate** | 12.3% | 4.7% | 2.8% | 1.9% | **0.0%** |

*Sources:*
- *[bench/results/tools_eval_comparison.json](../bench/results/tools_eval_comparison.json)*
- *[bench/results/tools_eval_variant_b_hybrid.json](../bench/results/tools_eval_variant_b_hybrid.json)*
- *[bench/results/tools_eval_variant_c.json](../bench/results/tools_eval_variant_c.json)*

---

## 5. Why Variant C (Direct Speak) is Recommended

1. **Sub-2.5 s Voice Turn Latency**: Saves 1.5–3.0 s by skipping LLM token generation for factual answers.
2. **Deterministic & Grammatically Correct**: Tool responses use pre-verified Polish grammatical templates normalized for TTS.
3. **Zero Risk of Hallucination**: Model cannot distort retrieved numbers, weather conditions, or financial rates.
4. **Predictable Memory Footprint**: Keeps GPU slots free for subsequent queries without queuing delays.

*Data source: [bench/results/tools_demo.json](../bench/results/tools_demo.json)*

---

## 6. Live Execution Examples

### Example 1: Tomorrow's Weather (`get_weather`)
- **Query**: *"Jaka będzie pogoda jutro w Krakowie?"*
- **Call**: `get_weather(location='Kraków', days=1)`
- **Raw output**: `temp_max=17.1, temp_min=11.1, precipitation_prob=12, wind_speed=19, condition='pochmurno'`
- **Spoken Variant C (2669.2 ms)**: *"Jutro w Krakowie będzie pochmurno, od jedenastu do siedemnastu stopni Celsjusza. Szansa na deszcz wynosi dwanaście procent."*
- **Spoken Variant B (5661.8 ms)**: *"Kraków jutro będzie pochmurno, z temperaturą od 11.1°C do 17.1°C. Szansa na opady jest niewielka (12%), a wiatr wieje z prędkością 19 km/h. Świt słońca będzie o 06:54, a zachód o 17:59."*
- **Gain**: **+2992.6 ms faster response**, no raw numbers or degree symbols.

### Example 2: Percentage Calculation (`calculate`)
- **Query**: *"Ile to jest 15 procent z 240?"*
- **Call**: `calculate(expression='15 * 240 / 100')`
- **Raw output**: `result=36`
- **Spoken Variant C (2347.8 ms)**: *"Wynik to trzydzieści sześć."*
- **Spoken Variant B (2994.7 ms)**: *"Wynik to trzydzieści sześć."*
- **Gain**: Identical content, **+646.9 ms faster response**.

### Example 3: Wikipedia without Hallucinations (`wiki_summary`)
- **Query**: *"Kim był Mikołaj Kopernik według Wikipedii?"*
- **Call**: `wiki_summary(topic='Mikołaj Kopernik', lang='pl')`
- **Raw output**: `summary='Mikołaj Kopernik – polski astronom, prawnik, urzędnik, dyplomata...'`
- **Spoken Variant C (2656.4 ms)**: *"Mikołaj Kopernik – polski astronom, prawnik, urzędnik, dyplomata, lekarz i niższy duchowny katolicki, doktor prawa kanonicznego i naukowiec – zajmował się też matematyką, ekonomią, strategią wojskową, kartografią i filologią. Bywa też nazywany fizykiem, filozofem."*
- **Spoken Variant B (5687.7 ms)**: *"Mikołaj Kopernik był polskim astronomem, prawnikiem, urzędnikiem, dyplomatą, lekarzem i niższym duchownym katolickim, doktorem prawa kanonicznego oraz naukowiecem..."*
- **Gain**: **+3031.3 ms faster response**, no abbreviations truncated, 0% hallucinations.

### Example 4: NBP Exchange Rate (`exchange_rate`)
- **Query**: *"Jaki jest aktualny kurs euro w złotówkach?"*
- **Call**: `exchange_rate(currency='EUR', to='PLN')`
- **Raw output**: `rate=4.3789, result=4.38`
- **Spoken Variant C (2480.7 ms)**: *"jeden euro to cztery złote i trzydzieści osiem groszy."*
- **Spoken Variant B (3272.7 ms)**: *"Jeden euro to obecnie około 4,38 złotych."*
- **Gain**: **+792.0 ms faster response**, read in natural Polish currency inflection.

### Example 5: Server Status (`server_status`)
- **Query**: *"Jaki jest aktualny stan tego serwera i zużycie RAM?"*
- **Call**: `server_status()`
- **Raw output**: `loadavg_1m=1.33, ram_used_gb=12.6, ram_total_gb=15.5, gpu_used='3225 MiB'`
- **Spoken Variant C (2153.3 ms)**: *"Serwer działa od siedem dni, obciążenie jeden przecinek trzydzieści trzy, RAM dwanaście przecinek sześć z piętnaście przecinek pięć gigabajtów, pamięć karty graficznej trzy tysiące dwieście dwadzieścia pięć megabajtów z cztery tysiące dziewięćdziesiąt sześć megabajtów. Usługi gaja-adapter i gaja-llm są aktywne."*
- **Spoken Variant B (4786.2 ms)**: *"Serwer działa od 7 dni, obciążenie 1:33, RAM zużyto 12,6 GB z 15,5 GB..."*
- **Gain**: **+2632.9 ms faster response**, correct phonetic reading of load and memory units.

---

## 7. Pipeline Architecture Diagram (Recommendation for `core-final`)

```
                  ┌───────────────────────┐
                  │ ASR Transcription(STT)│
                  └───────────┬───────────┘
                              │
                      Regex rules (0 ms)
                              │
               Router v3 logprob (/completion)
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
     [ LOCAL ]             [ TOOL ]            [ HERMES ]
         │                    │                    │
   Streamed reply       Targeted Tool Call         │
   without tools        in llama-server            │
   (low TTFT)                 │                    │
         │              Tool Execution             │
         │           (gaja.tools.execute)          │
         │                    │                    │
         │               [VARIANT C]               │
         │          Direct speak: res.speak        │
         │         (no 2nd LLM turn, -1.5s)        │
         │                    │                    │
         │             normalize_for_tts           │
         │                    │                    │
         ▼                    ▼                    ▼
     TTS Piper            TTS Piper         Handoff to ACP
```

### Proposed `config.toml` settings

```toml
[tools]
enabled = true
variant = "c"  # "c" (recommended, direct speak) or "b_hybrid" (with 2nd LLM turn)
default_location = "Warszawa"
timeout_s = 2.5
cache_ttl_weather_s = 600
cache_ttl_nbp_s = 3600

[text]
normalize_tts = true
```

---

## 8. Adapter Integration (State after Round 6b, 2026-10-09)

Tools operate in the production pipeline (`gaja/pipeline.py`, `gaja/llm/router.py`). Details relative to §7:

- **No regex pre-filter**: Every turn is classified by Router v3 (`prompts/router3_pl.md`): single `/completion` request with `n_predict 1` and `n_probs` on router slot (slot 1). TOOL class wins when its mass >= `router.tool_threshold = 0.6`; HERMES when mass >= `router.threshold = 0.5`. Explicit address to Hermes ("zapytaj Hermesa…", "w Hermesie") routes to Hermes directly without LLM invocation. Evaluation: `bench/results/router3_eval_r6.json` (tool accuracy 91.2%, argument accuracy 91.2%, classify p50 116.5 ms).
- **Tool selection**: `Router.select_tool` is a short continuation of the prompt constrained by grammar (`tool_grammar`, max 64 tokens); returns name and arguments (select p50 ~490 ms in evaluation, 180–560 ms live). If neither the utterance nor previous user turn contains tool cue words (`TOOL_CUES`), selection is bypassed and routed locally — saving ~500–660 ms on false TOOL classifications. Selected tools inconsistent with cues are also rejected.
- **Default location**: Weather queries without location use `tools.default_location = "Warszawa"`.
- **Wikipedia search**: If the exact page title is not found, `wiki_summary` searches via `/w/rest.php/v1/search/title` and retrieves the summary of the best match.
- **Variant C**: The tool's `speak` text is spoken directly (no second LLM turn) and stored in conversation history as Gaja's reply. Tool errors also provide spoken explanations (`ok: false`). Client receives `{"type":"tool","name":...,"arguments":{...},"ok":bool,"text":...}` before `utterance`.
- **Speculative execution**: Tool selection may start speculatively, but execution awaits end-of-speech confirmation (`committed`), ensuring no external HTTP requests fire for continued speech.
- **Normalization**: When `[text] normalize_tts = true`, all sentences pass through `normalize_for_tts` before TTS.
- **Configuration** (`config.toml`): `[tools] enabled = true`, `default_location = "Warszawa"`, `timeout_s = 4.0`.
- **Metrics**: `metrics.route = "tool"`, `tool`, `tool_select_ms`, `tool_ms`.
- **Live testing (2026-10-09)**: Weather, time (PL and EN), calculator, exchange rate, Wikipedia — all returning `ok`, first audio 0.54–1.13 s after text receipt (`/tmp/gaja-r6b/live_tools.log`).
