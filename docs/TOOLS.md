# Gaja: Fast local tools (local-tools) — Round 2

Evaluation report, implementation specification and architecture recommendation for the `local-tools` workstream (Round 2) for the Gaja voice assistant.

The local LiquidAI LFM2.5-2.6B model (`llama-server` on GPU `:8791`) gained a set of fast, asynchronous local tools that answer queries about time, weather, calculations, currency rates, units of measure, encyclopedia lookups, server status and holidays in **~1.5–2.5 s** with the recommended Variant C (direct synthesis of the `speak` text), eliminating the unnecessary second LLM generation turn (~1.5 s saved and 0% hallucination).

---

## 1. Implemented tools (`gaja/tools/`)

All tools are fully asynchronous, run operations in $\le 2.5\text{ s}$ (hard timeout), require no API keys or secrets, and return a `ToolResult` structure with the fields:
- `data`: a concise dictionary of parameters and technical results,
- `speak`: normalized text for direct speech synthesis via Piper TTS / Paradee (numbers, times, dates and units written out in words in the nominative/locative, temperatures rounded to whole numbers, rates to 2 decimal places, no `°`, `%`, `km/h` characters, no colons and no raw digits),
- `error`: an error message or `None` (the tool never raises an exception into the assistant's pipeline).

The default location comes from configuration (`Settings().tools.default_location`, configured under `[tools] default_location = "Katowice"` in `config.toml` and overridable via `GAJA_TOOLS__DEFAULT_LOCATION`) through `gaja.tools.base.get_default_location()`, shared by both `get_weather` and `get_sun`.

| Tool | Signature | Data source / Algorithm | `speak` format after normalization (Round 2) | Execution time (typ.) |
|---|---|---|---|---|
| `get_time` | `get_time(timezone="Europe/Warsaw")` | Local system clock, `zoneinfo.ZoneInfo` | *"Jest piątek, dziewiątego października dwa tysiące dwudziestego szóstego roku, godzina dziesiąta pięćdziesiąt sześć."* | < 1 ms |
| `get_weather` | `get_weather(location="Warszawa", when="now"\|"today"\|"tomorrow")` | Open-Meteo Geocoding + Forecast API (WMO code $\to$ PL/EN words, sunrise/sunset, precipitation, wind), 10 min cache. Non-inflecting template: `"{lokalizacja}, {pora}: od X do Y stopni..."` | *"Kraków, jutro: od jedenaście do siedemnaście stopni, pochmurno, szansa na opady dwanaście procent, wschód słońca o szóstej pięćdziesiąt cztery, zachód o siedemnastej pięćdziesiąt dziewięć."* | ~250–350 ms (cache: < 1 ms) |
| `calculate` | `calculate(expression: str)` | Safe Python AST parser and evaluator (`+ - * / ** % sqrt`, percentages, powers, protection against AST DoS, no eval/exec) | *"Wynik to trzydzieści sześć."* | < 2 ms |
| `convert_units` | `convert_units(value, from_unit, to_unit)` | Unit conversions: length, mass, temperature, volume, speed, time; supports colloquial Polish and English unit names | *"sto kilometrów to około sześćdziesiąt dwa przecinek czternaście mil."* | < 1 ms |
| `exchange_rate` | `exchange_rate(currency, amount=1, to="PLN")` | Table A of the National Bank of Poland (NBP API), cross rates, amounts in złoty and grosz, 1 h cache | *"jeden euro to cztery złote i trzydzieści osiem groszy."* | ~80 ms (cache: < 1 ms) |
| `wiki_summary` | `wiki_summary(topic: str, lang="pl"\|"en")` | Wikipedia REST Summary API, smart sentence splitting that is robust to abbreviations (`łac.`, `niem.`, `ur.`, `zm.`, `ok.`, `m.in.`, `tzw.`), removal of linguistic insertions and IPA | *"Mikołaj Kopernik – polski astronom, prawnik, urzędnik, dyplomata, lekarz i niższy duchowny katolicki..."* | ~180–230 ms |
| `server_status` | `server_status()` | `/proc/uptime`, `/proc/loadavg`, `/proc/meminfo`, `nvidia-smi` (GPU memory), `systemctl --user is-active` (Gaja services) | *"Serwer działa od siedem dni, obciążenie jeden przecinek trzydzieści trzy, RAM dwanaście przecinek sześć z piętnaście przecinek pięć gigabajtów..."* | ~35 ms |
| `next_holiday` | `next_holiday(holiday=None)` | Local calculations: 13 statutory public holidays in Poland, Meeus/Jones/Butcher algorithm for Easter, days until the next holiday or until a holiday named by the user | *"Najbliższe święto to Wszystkich Świętych, pierwszego listopada, za dwadzieścia trzy dni."* | < 1 ms |
| `get_sun` | `get_sun(location="Katowice", kind="all"|"sunrise"|"sunset"|"day_length")` | Pure NOAA astronomical formula, offline, Katowice by default | *"Katowice: wschód słońca o szóstej trzydzieści dwa, zachód o osiemnastej zero pięć, dzień trwa jedenaście godzin i trzydzieści trzy minuty."* | < 1 ms |
| `manage_timer` | `manage_timer(action="set"|"cancel"|"status", duration_min=None, time=None, label=None, kind="timer"|"reminder")` | Management of local timers and reminders in `data/timers.json`, WS notifications and speech | *"Nastawiono minutnik na dziesięć minut."* | < 1 ms |
| `remember` | `remember(fact: str)` | Persistent, lightweight memory of user facts in `data/memory.json` | *"Zapamiętałam: moja siostra ma na imię Ola."* | < 1 ms |
| `forget` | `forget(fact: str)` | Removal of facts from `data/memory.json` | *"Usunęłam z pamięci: imię siostry."* | < 1 ms |

---

## 2. Text normalizer for TTS (`gaja/text/normalize.py`)

Piper TTS reads raw text through espeak, which garbles numbers, times, fractions and units (e.g. "10.8" $\to$ "dziesięć kropka osiem", "06:54" $\to$ "zero sześć dwukropek...", "12°C" $\to$ "dwanaście stopień c").
A fast and safe phonetic normalization module `normalize_for_tts(text: str, lang: "pl"|"en") -> str` has been added:
- **Execution time**: ~0.55 ms per sentence (< 1 ms budget).
- **Rule scope (Round 3 / 3b)**:
  - Inflection of cardinal numerals across grammatical cases (nominative, genitive, locative, instrumental) for numbers 0–999 999, depending on syntactic context (genitive-governing prepositions: `od`, `do`, `z`, `ze`, `około`, `ok.`, `bez`, `u`, `dla` $\to$ genitive; locative-governing prepositions: `o`, `przy`, `po` $\to$ locative for hours).
  - Government and agreement of the numeral with the noun (rules 1 / 2–4 except 12–14 / 5+; inflection table for key units and concepts: `dzień`, `stopień`, `procent`, `minuta`, `godzina`, `sekunda`, `kilometr`, `metr`, `złoty`, `grosz`, `euro`, `dolar`, `gigabajt`, `megabajt`, `rok`, `mila`, `kilogram`, `litr`).
  - Gender distinction for 1 and 2 ("jedno euro", "dwie godziny"), and the invariant "jeden" in multi-digit numbers ending in 1 ("dwadzieścia jeden minut", "sto jeden minut", "dwadzieścia jeden euro" — never "dwadzieścia jedna").
  - Preservation of the original capitalization of non-numeric words at the start of a sentence ("Od 11 do 17 stopni." $\to$ "Od jedenastu do siedemnastu stopni.").
  - A consistent rule for decimal fractions in the genitive of memory units ("12,6 z 15,5 GB" $\to$ "dwanaście przecinek sześć z piętnastu przecinek pięciu gigabajtów").
  - Cardinal and decimal numbers ("10,8" / "10.8" $\to$ "dziesięć przecinek osiem"), negative numbers.
  - Times in the nominative and locative ("06:54" $\to$ "szósta pięćdziesiąt cztery", "o 17:59" $\to$ "o siedemnastej pięćdziesiąt dziewięć").
  - Dates and ordinal numerals ("9 października 2026 r." $\to$ "dziewiątego października dwa tysiące dwudziestego szóstego roku").
  - Currencies ("4,38 zł" $\to$ "cztery złote i trzydzieści osiem groszy", "10 EUR" $\to$ "dziesięć euro").
  - Abbreviations and Roman-numeral centuries ("w XVI wieku" $\to$ "w szesnastym wieku", "np." $\to$ "na przykład", "m.in." $\to$ "między innymi", "ok." $\to$ "około").
- **Unit tests**: [tests/test_normalize.py](../tests/test_normalize.py) — 107 tests (PL + EN + edge cases), all passing (100%).

### Objective verification of the normalizer (`bench/normalize_eval.py`)
30 representative test sentences (numbers, dates, times, currencies, units) were synthesized with the Piper voice `pl_PL-gosia-medium` on CPU, and then transcribed with the Faster-Whisper `small` model on **CPU int8**, measuring number correctness and word error rate (WER).

*Data source: bench/results/normalize_eval.json*
- **Numbers Correct Rate**:
  - Without normalization: **76.7%** (23/30)
  - With normalization: **86.7%** (26/30) — **an increase of +10.0 pp**
- **Average Word Error Rate (WER)**:
  - Without normalization: **19.4%**
  - With normalization: **10.1%** — **error reduction of 9.31 pp (-47.9% relative)**
- **Measurement duration**: **122.2 s** (30 sentences, 2 conditions on CPU)
- **Measurement loadavg**: start: `[0.64, 0.39, 0.49]`, end: `[4.45, 1.94, 1.05]`.

---

## 3. Fixing hallucination in the 2nd LLM turn (demo_6)

In Round 1, the query about Mikołaj Kopernik (`demo_6`) resulted in the Wikipedia text being cut off at *"Mikołaj Kopernik, łac. Nicolaus Copernicus, niem."*, after which LFM generated a distorted fact *"polsko-polskim astronomem..."*.
- **Cause**: a primitive splitter divided sentences at the period after the language abbreviation `"niem."`.
- **Fix in `gaja/tools/wiki_tool.py`**:
  1. Added a smart regex `_split_sentences_smart` that accounts for a base of Polish and Latin abbreviations: `łac.`, `niem.`, `ur.`, `zm.`, `ok.`, `m.in.`, `tzw.`, `np.`, `r.`, `w.`, `św.`, `gr.`, `ang.`.
  2. Added automatic removal of etymological and language asides in parentheses and after commas (e.g. `(łac. Nicolaus Copernicus, niem. Niklas Koppernigk)`), as well as phonetic IPA markers.
- **Effect**: In Round 2, `wiki_summary` returns clean, complete sentences: *"Mikołaj Kopernik – polski astronom, prawnik, urzędnik, dyplomata, lekarz i niższy duchowny katolicki..."*. In both variants (B_hybrid and C) the hallucination rate dropped to 0%.

---

## 4. Comprehensive comparison of architectures (Variant A, B, B_hybrid r1, B_hybrid r2, Variant C)

The evaluation was run on the main test set [bench/data/tools_eval.jsonl](../bench/data/tools_eval.jsonl) (**106 queries**) and on a separate held-out set [bench/data/tools_heldout.jsonl](../bench/data/tools_heldout.jsonl) (**22 queries**) that was not seen during tuning.

### Comparison table (all variants and rounds)

| Metric | Variant A (Native) | Variant B (Pure Router) | Variant B_hybrid R1 | Variant B_hybrid R2 | Variant C (Direct Speak) | Variant C (Held-out 22) |
|---|---|---|---|---|---|---|
| **Number of samples ($N$)** | 106 | 106 | 106 | 106 | **106** | 22 |
| **Action accuracy (Action Acc)** | 78.30% | 89.62% | 90.57% | **96.23%** | **96.23%** | **90.91%** |
| **Tool selection accuracy (Tool Acc)** | 76.42% | 86.79% | 87.74% | **96.23%** | **96.23%** | **90.91%** |
| **Argument accuracy (Arg Acc)** | 76.42% | 86.79% | 87.74% | **96.23%** | **96.23%** | **90.91%** |
| **Tool selection recall** | 95.59% | 91.18% | 92.65% | **95.59%** | **95.59%** | **100.00%** |
| **False Tool Rate on chit-chat** | 62.50% | 12.50% | 12.50% | **0.00%** | **0.00%** | 25.00% (1/4) |
| **Hermes boundary effectiveness** | 62.50% | 100.00% | 100.00% | **93.75%** | **93.75%** | **75.00%** (3/4) |
| **Trap effectiveness (Traps)** | 25.00% | 25.00% | 25.00% | **100.00%** | **100.00%** | **100.00%** (4/4) |
| **Faithfulness rate** | N/A | N/A | N/A | 93.40% | **100% (by def.)** / 88.68% token-match | 100% (by def.) / 81.82% |
| **Router latency (p50)** | — | 947.5 ms | 954.6 ms | 958.0 ms | **951.7 ms** | 971.3 ms |
| **E2E latency (median p50)** | 4168.0 ms | 3657.6 ms | 3556.2 ms | 3137.4 ms | **1690.5 ms (-1.45 s!)** | **1711.8 ms** |
| **E2E latency (90th percentile p90)** | 6523.5 ms | 5776.3 ms | 5340.7 ms | 5014.9 ms | **2771.2 ms (-2.24 s!)** | **3188.3 ms** |
| **E2E latency (mean)** | 4747.8 ms | 3563.9 ms | 3205.5 ms | 3096.3 ms | **1872.3 ms (-1.22 s!)** | **2492.3 ms** |
| **Measurement loadavg (start $\to$ end)** | — | — | — | 1.15 $\to$ 4.09 | 3.05 $\to$ 3.29 | 5.86 $\to$ 5.06 |

*Data sources:*
- *bench/results/tools_eval_comparison.json*
- *bench/results/tools_eval_variant_b_hybrid.json*
- *bench/results/tools_eval_variant_c.json*

---

## 5. Why do we recommend Variant C (Direct Speak)?

1. **Drastic reduction of E2E latency (-1.45 s at p50, -2.24 s at p90)**:
   - In Variant B_hybrid, after the tool runs, the assistant must start a second full LLM generation (prompt evaluation of the second stage + generation of 30–50 response tokens), which takes an extra 1.5–3.0 s.
   - In Variant C, once the tool result is received, the `speak` field goes **immediately to TTS**. The p50 time is just **1.69 s** (of which ~950 ms is the router and ~300 ms is the weather/network API call).
2. **Complete elimination of hallucination risk (100% Faithfulness)**:
   - In the second turn, the LLM tends to format numbers as raw digits (`"10.8°C"`, `"06:54"`), add unnecessary embellishments or distort facts (as with Kopernik in R1).
   - In Variant C the `speak` text is generated from deterministic, tested tool templates and passes through `normalize_for_tts`, which guarantees full factual consistency and no unexpected words.
3. **Optimal phonetic quality for speech synthesis**:
   - The direct `speak` text contains ready, inflected number words, without percent signs, degree signs or abbreviations that are problematic for Piper.

---

## 6. Examples from real live runs (Live Demo)

*Data source: bench/results/tools_demo.json*

### Example 1: Tomorrow's weather (`get_weather`)
- **Question**: *"Jaka jest jutro pogoda w Krakowie?"* ("What is the weather in Kraków tomorrow?")
- **Call**: `get_weather(location='Kraków', when='tomorrow')`
- **Raw result**: `temp_min=11.1, temp_max=17.1, conditions='pochmurno', precip=12%, sunrise='06:54', sunset='17:59'`
- **Variant C response (2669.2 ms)**: *"Kraków, jutro: od jedenaście do siedemnaście stopni, pochmurno, szansa na opady dwanaście procent, wschód słońca o szóstej pięćdziesiąt cztery, zachód o siedemnastej pięćdziesiąt dziewięć."*
- **Variant B response (5661.8 ms)**: *"Kraków jutro będzie pochmurno, z temperaturą od 11.1°C do 17.1°C. Szansa na opady jest niewielka (12%), a wiatr wieje z prędkością 19 km/h. Świt słońca będzie o 06:54, a zachód o 17:59."*
- **Gain**: **+2992.6 ms faster response time**, no raw digits and no degree signs or colons.

### Example 2: Percentage calculation (`calculate`)
- **Question**: *"Ile to jest 15 procent z 240?"* ("What is 15 percent of 240?")
- **Call**: `calculate(expression='15 * 240 / 100')`
- **Raw result**: `result=36`
- **Variant C response (2347.8 ms)**: *"Wynik to trzydzieści sześć."*
- **Variant B response (2994.7 ms)**: *"Wynik to trzydzieści sześć."*
- **Gain**: Identical content, **+646.9 ms faster response time**.

### Example 3: Wikipedia without hallucination (`wiki_summary`)
- **Question**: *"Kim był Mikołaj Kopernik według Wikipedii?"* ("Who was Mikołaj Kopernik according to Wikipedia?")
- **Call**: `wiki_summary(topic='Mikołaj Kopernik', lang='pl')`
- **Raw result**: `summary='Mikołaj Kopernik – polski astronom, prawnik, urzędnik, dyplomata...'`
- **Variant C response (2656.4 ms)**: *"Mikołaj Kopernik – polski astronom, prawnik, urzędnik, dyplomata, lekarz i niższy duchowny katolicki, doktor prawa kanonicznego i naukowiec – zajmował się też matematyką, ekonomią, strategią wojskową, kartografią i filologią. Bywa też nazywany fizykiem, filozofem."*
- **Variant B response (5687.7 ms)**: *"Mikołaj Kopernik był polskim astronomem, prawnikiem, urzędnikiem, dyplomatą, lekarzem i niższym duchownym katolickim, doktorem prawa kanonicznego oraz naukowiecem..."*
- **Gain**: **+3031.3 ms faster response time**, no truncation error at the `niem.` abbreviation, and 0% hallucination.

### Example 4: NBP exchange rate (`exchange_rate`)
- **Question**: *"Jaki jest aktualny kurs euro w złotówkach?"* ("What is the current euro rate in złoty?")
- **Call**: `exchange_rate(currency='EUR', to='PLN')`
- **Raw result**: `rate=4.3789, result=4.38`
- **Variant C response (2480.7 ms)**: *"jeden euro to cztery złote i trzydzieści osiem groszy."*
- **Variant B response (3272.7 ms)**: *"Jeden euro to obecnie około 4,38 złotych."*
- **Gain**: **+792.0 ms faster response time**, the amount is read in natural Polish in groszy instead of as a decimal fraction.

### Example 5: Server status (`server_status`)
- **Question**: *"Jaki jest aktualny stan tego serwera i zużycie RAM?"* ("What is the current status of this server and its RAM usage?")
- **Call**: `server_status()`
- **Raw result**: `loadavg_1m=1.33, ram_used_gb=12.6, ram_total_gb=15.5, gpu_used='3225 MiB'`
- **Variant C response (2153.3 ms)**: *"Serwer działa od siedem dni, obciążenie jeden przecinek trzydzieści trzy, RAM dwanaście przecinek sześć z piętnaście przecinek pięć gigabajtów, pamięć karty graficznej trzy tysiące dwieście dwadzieścia pięć megabajtów z cztery tysiące dziewięćdziesiąt sześć megabajtów. Usługi gaja-adapter i gaja-llm są aktywne."*
- **Variant B response (4786.2 ms)**: *"Serwer działa od 7 dni, obciążenie 1:33, RAM zużyto 12,6 GB z 15,5 GB..."*
- **Gain**: **+2632.9 ms faster response time**, correct phonetic rendering of the load (not as the time "1:33") and of the RAM/GPU units.

---

## 7. Target pipeline diagram (Recommendation for `core-final`)

```
                  ┌────────────────────────┐
                  │ Transkrypcja ASR (STT)│
                  └───────────┬────────────┘
                              │
                    Reguły regex (0 ms)
                              │
               Router v3 logprob (/completion)
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
     [ LOCAL ]             [ TOOL ]            [ HERMES ]
         │                    │                    │
  Strumieniowanie       Targeted Tool Call         │
  odpowiedzi bez        w llama-server             │
  narzędzi (niski TTFT)       │                    │
         │             Wykonanie narzędzia         │
         │             (gaja.tools.execute)        │
         │                    │                    │
         │               [WARIANT C]               │
         │          Direct speak: res.speak        │
         │         (brak 2. tury LLM, -1.5s)       │
         │                    │                    │
         │             normalize_for_tts           │
         │                    │                    │
         ▼                    ▼                    ▼
     TTS Piper            TTS Piper         Handoff do ACP
```

### Proposed configuration entry for `config.toml`

```toml
[tools]
enabled = true
variant = "c"  # "c" (rekomendowany, direct speak) lub "b_hybrid" (z 2. turą LLM)
default_location = "Katowice"
timeout_s = 2.5
cache_ttl_weather_s = 600
cache_ttl_nbp_s = 3600

[text]
normalize_tts = true
```

---

## 8. Integration in the adapter (state after round 6b, 2026-10-09)

The tools run in the production pipeline (`gaja/pipeline.py`, `gaja/llm/router.py`). Differences from the diagram in §7:

- **No regex pre-filter.** Every turn is classified by Router v3 (`prompts/router3_pl.md`): a single
  `/completion` call with `n_predict 1` and `n_probs` on the router slot (slot 1). The TOOL class wins when its mass is ≥
  `router.tool_threshold = 0.6`; HERMES wins when its mass is ≥ `router.threshold = 0.5`. Exception: an explicit address to Hermes
  („zapytaj Hermesa…”, „w Hermesie”) goes to Hermes without the model. Evaluation: `bench/results/router3_eval_r6.json`
  (tool accuracy 91.2 %, argument accuracy 91.2 %, classify p50 116.5 ms).
- **Tool selection**: `Router.select_tool` is a short continuation of the same prompt, constrained by grammar
  (`tool_grammar`, max 64 tokens); it returns the name and arguments (select p50 ~490 ms in evaluation, 180–560 ms live).
  If the utterance (or, for a short follow-up, the previous user utterance) contains no tool cue word
  (`TOOL_CUES`), selection is skipped and the turn goes to a local answer, which saves ~500–660 ms
  when the TOOL class was misclassified. A selected tool that does not match the cues is also rejected.
- **Location resolution (`get_weather` and `get_sun`)**:
  1. **Utterance extraction & bias override (`resolve_tool_location`, `extract_location_from_utterance`)**: when the user names a place in the utterance (e.g. *„Jaka pogoda w Krakowie?”*, *„a w Gdańsku jutro?”*, *„pogoda we Wrocławiu”*, *„jak będzie w Zakopanem w weekend”*, *„pogoda w Nowym Targu”*, *„a w Berlinie?”*), the adapter extracts it from `w/we/na/dla <Place>` (ignoring non-place phrases like *„w weekend”*, *„na jutro”*, *„w salonie”*) and overrides missing, placeholder (*„Miasto”*, *„tutaj”*, *„here”*), or default-biased (*„Katowice”*) LLM arguments.
  2. **Polish locative normalization & geocoding fallback (`normalize_location`, `location_candidates`)**: Open-Meteo Geocoding fails on Polish locative forms (*„Gdańsku”*, *„Wrocławiu”*, *„Zakopanem”*, *„Nowym Targu”*, *„Berlinie”*) or returns false prefix matches (*„Krakowie”* $\to$ *„Krakowiec”*). `get_weather` queries Open-Meteo with ordered candidates: the name as given first (rejecting false prefix hits when a known irregular nominative differs), followed by normalized/lemmatized nominative candidates (irregular city table + deterministic Polish locative suffix rules for single- and two-word names).
  3. **Follow-up city retention (`extract_location_from_history`)**: a short follow-up without a city after a city weather question (e.g. *„a jutro?”*) keeps the city from the previous weather turn in the session history.
  4. **Configured default (`get_default_location`)**: when no place is mentioned and there is no prior weather city in history, both `get_weather` and `get_sun` use `Settings().tools.default_location` (`[tools] default_location = "Katowice"` in `config.toml`).
- **Wikipedia**: when a page with the exact title does not exist, `wiki_summary` searches for the title via
  `/w/rest.php/v1/search/title` and fetches the summary of the page found.
- **Variant C**: the tool's `speak` text is spoken directly (without a second LLM turn) and saved in history as Gaja's reply.
  A tool error is also given its own spoken sentence (`ok: false`). The client receives the event
  `{"type":"tool","name":...,"arguments":{...},"ok":bool,"text":...}` before `utterance`.
- **Speculation**: tool selection may start speculatively, but the execution itself waits for confirmation that the
  utterance has ended (`committed`), so no HTTP request goes out for an utterance the user goes on to continue.
- **Normalization**: with `[text] normalize_tts = true`, every sentence (not only from tools) goes through
  `normalize_for_tts` before TTS; the on-screen text is unchanged.
- **Configuration** (`config.toml`): `[tools] enabled = true`, `default_location = "Katowice"`, `timeout_s = 4.0`
  (2.5 s was not enough for Wikipedia under load). The `variant` and `cache_ttl_*` fields from §7 are not used.
- **Metrics**: `metrics.route = "tool"`, `tool`, `tool_select_ms`, `tool_ms`.
- **Live (2026-10-09)**: weather (default and „jutro w Krakowie”), time (PL and EN), calculator, euro rate,
  Wikipedia („kim był Stanisław Lem”, „kim jest Hermes w mitologii”) — all `ok`, first audio 0.54–1.13 s after sending the text
  (`/tmp/gaja-r6b/live_tools.log`).


---

## 9. Extension with timers, fact memory and offline tools (Task I2, 2026-10-12)

As part of task I2, the tool set was extended with:
- **Default location**: Katowice in `config.toml`, `weather_tool`, `sun_tool` and tests.
- **Timers and reminders**: module `gaja/timers.py`, tool `manage_timer`, WS event `{"type":"timer", ...}` (protocol v1.9) and HTTP API `GET /api/timers`, `DELETE /api/timers/{id}`.
- **Fact memory**: module `gaja/memory.py`, tools `remember` and `forget`, injection of facts at the end of the system prompt (600-character limit), dynamic update of the session history (`update_system_prompt`), HTTP API `GET/POST/DELETE /api/memory`.
- **Offline clock/calendar tools**:
  - `get_sun`: sunrise, sunset and day length (pure NOAA astronomy).
  - `get_time`: date arithmetic (`target_date`, days until, day of the week).
  - `next_holiday`: holiday lookup by name (`holiday="Boże Narodzenie"`).
  - `convert_units`: kitchen measures (cups, tablespoons, teaspoons) and data sizes (byte, kB, MB, GB, TB).

### Evaluation results (bench/results/)

1. **Router v3 (`bench/results/router3_eval.json`, N=353 questions)**:
   - **Action accuracy**: 91.7 % (133/145).
   - **Tool accuracy**: 83.3 % (80/96).
   - **Argument accuracy**: 82.3 % (79/96).
   - **Hermes recall (strict)**: 93.8 % (91/97), precision: 98.9 % (91/92).
   - **False handoff on local queries**: 0.8 % (1/129).
   - **Router latency**:
     - Classification: p50 = 118.8 ms, p90 = 128.8 ms.
     - Tool selection: p50 = 366.6 ms, p90 = 627.5 ms.

2. **E2E tool evaluation (`bench/results/tools_eval_comparison.json`)**:
   - **Eval set (N=119)**:
     - Action accuracy: 90.76 %
     - Tool accuracy: 85.71 %
     - Argument accuracy: 85.71 %
     - Tool selection recall: 81.25 %
     - Faithfulness rate: 86.55 %
     - Trap accuracy: 100.0 %
     - Router latency: p50 = 984.2 ms, p90 = 1065.5 ms
     - E2E latency: p50 = 2154.5 ms, p90 = 3184.4 ms, mean = 2096.2 ms
   - **Held-out set (N=26)**:
     - Action accuracy: 88.46 %
     - Tool accuracy: 84.62 %
     - Argument accuracy: 84.62 %
     - Tool selection recall: 87.50 %
     - Faithfulness rate: 84.62 %
     - Trap accuracy: 100.0 %
     - Router latency: p50 = 1021.0 ms, p90 = 1153.7 ms
     - E2E latency: p50 = 2306.3 ms, p90 = 3571.9 ms, mean = 2872.4 ms


---

## 10. Quality fixes for tool turns (Task I11, 2026-10-12)

As part of task I11, quality fixes identified during live testing were introduced:

1. **Handling countdowns to holidays and protection against date hallucination**:
   - Extended `TOOL_CUES` for `next_holiday` with holiday inflections (`bo[zż]\w*\s+narodzeni\w*`, Wigilia, Gwiazdka, Wielkanoc, Nowy Rok, Trzech Króli, Boże Ciało).
   - Updated the router prompt (`prompts/router3_pl.md`) with examples of counting days to holidays (`next_holiday(holiday="Boże Narodzenie")`).
   - Today's date (Europe/Warsaw timezone) is injected into the system prompt in both text and spoken form. Only the date is injected, without the time: it changes once a day, so the llama.cpp prompt cache stays warm between turns.

2. **Clock fast path (regex fast-path) and elimination of false handoffs to Hermes**:
   - Added a deterministic regex fast path `clock_fast_path` in `gaja/llm/router.py` for the most common time and date queries (*„która godzina”*, *„która jest godzina”*, *„jaki dziś dzień”*, *„jaka jest data”*). It returns a `get_time` call immediately with zero router classification time (0 ms), protecting against LLM cold start.
   - When the router is unavailable (`router-down`, e.g. cold start after a restart), simple time and date questions still reach `get_time` via the same fast path, instead of the Hermes safety net. Other questions keep the existing path, including handoff to Hermes.

3. **Conciseness of `get_time` answers (answer only what was asked)**:
   - Added the parameter `kind: "time" | "date" | "all"` to `get_time`.
   - A time query returns only the time (*„Jest dziewiąta trzydzieści pięć.”* ("It's nine thirty-five.")).
   - A day/date query returns only the day and date (*„Dziś jest poniedziałek, dwunastego października.”* ("Today is Monday, October twelfth.")).
   - The full form (`kind="all"`) is kept as the default for general queries.

4. **Normalizing memory facts to the 3rd person with natural confirmation**:
   - Remembered facts are normalized to the 3rd person about the user (*„Użytkownik lubi herbatę z cytryną”*, *„Siostra użytkownika ma na imię Ola”*), which removes ambiguity for the assistant once injected into the prompt.
   - The assistant's spoken confirmation keeps the natural 2nd person (*„Zapamiętałam, że lubisz herbatę z cytryną.”*).

5. **Correct Polish inflection for timers and natural end-of-timer messages**:
   - Timer labels use the correct numeral declension: *„1 minuta”*, *„2 minuty”*, *„5 minut”*, *„22 minuty”*.
   - End-of-timer messages are phrased naturally and phonetically: *„Minął minutnik na dwie minuty.”*, *„Minął minutnik na jedną minutę.”*, *„Minął minutnik na dwadzieścia dwie minuty.”*, *„Przypomnienie: <treść>.”*.

### Regression test

`bench/tools_live_smoke.py` sends 25 questions over WebSocket to the running adapter and removes the timers and facts it created. Results are written to `bench/results/tools_live_smoke.json`.

Result on the live adapter after deployment (2026-10-12): 25/25 questions reached the expected tool, with no errors. The median `latency_e2e_ms` is 992.2 ms. Time and date questions go through the fast path and take 96.6–146.5 ms (before the change, when they went through the LLM router: 660.8–826.6 ms). The countdown to Christmas now goes to `next_holiday` instead of a regular LLM answer, which previously made up the date.

---

## 11. Voice-device speaker MCP tools for Hermes (`gaja/mcp_speaker.py`)

Hermes can proactively speak announcements or play short notification sounds on connected household voice devices (such as `atom-pokoj`) through the stdio MCP server `gaja.mcp_speaker`.

### HTTP endpoints on the Gaja adapter (`127.0.0.1:8790`)
- `GET /api/devices` — lists registered voice devices with `id`, `name`, `online`, `last_seen`, and `last_wake`.
- `GET /api/devices/sounds` — lists allow-listed sound names from `assets/sounds/*.wav` (`alert`, `chime`, `success`).
- `POST /api/devices/{device_id}/say` — body `{"text": "..."}` (1–500 chars); synthesizes and plays the text as an exclusive announcement turn on the connected device (`404` if offline).
- `POST /api/devices/{device_id}/play` — body `{"sound": "<name>"}`; plays an allow-listed mono PCM16 WAV file from `assets/sounds/` (`chime`, `alert`, `success`) as an exclusive announcement-style turn (`400` for unknown sound names, `404` if offline).

### Exposed MCP tools (`gaja_speaker`)
Runnable as `python -m gaja.mcp_speaker` (uses `GAJA_API_URL`, defaulting to `http://127.0.0.1:8790`):
- `list_speakers()` — returns the device list from `GET /api/devices` (including `online` status).
- `list_sounds()` — returns the allow-listed sound names (`["alert", "chime", "success"]`).
- `speaker_say(device_id, text)` — speaks `text` (1–500 characters) via TTS on `device_id`.
- `speaker_play_sound(device_id, sound)` — plays an allow-listed sound (`chime`, `alert`, or `success`) on `device_id`.

### Automatic registration in ACP sessions
For Hermes sessions started by the Gaja adapter (`gaja/hermes/client.py`), `gaja_speaker` is automatically passed in the ACP `session/new` `mcpServers` list when enabled in `config.toml`:

```toml
[hermes]
speaker_mcp = true  # default: true (override via GAJA_HERMES__SPEAKER_MCP=false)
```

### Enabling `gaja_speaker` for the Hermes Gateway (Telegram / Discord / CLI)
ACP `session/new` only registers the MCP server for sessions spawned by the voice adapter. To make the same speaker tools (`mcp_gaja_speaker_*`) available to the persistent `hermes-gateway` service (e.g. from Telegram or Discord) or standalone CLI sessions, add `gaja_speaker` under `mcp_servers` in `~/.hermes/config.yaml` (and in any profile config such as `~/.hermes/profiles/<profile>/config.yaml` if profiles override `mcp_servers`):

```yaml
mcp_servers:
  gaja_speaker:
    command: "/path/to/gaja-hermes-adapter/.venv/bin/python"
    args:
      - "-m"
      - "gaja.mcp_speaker"
    env:
      PYTHONPATH: "/path/to/gaja-hermes-adapter"
      GAJA_API_URL: "http://127.0.0.1:8790"
```

After updating `~/.hermes/config.yaml`, reload or restart `hermes-gateway.service` during a maintenance window so the gateway discovers the new MCP server.
