# Gaja adapter — core implementation

Realtime Polish voice loop: a browser or CLI streams mic audio over `WS /ws`; the adapter endpoints speech
(Silero VAD), transcribes it (faster-whisper on GPU by default, Whistle as the selectable and automatic fallback), decides whether the request goes to
Hermes, otherwise asks the local LLM (LFM2.5-2.6B via llama-server), chunks the streamed reply into sentences,
synthesizes each sentence with the selected voice engine and streams PCM back.
Before the local reply, Router v3 (one logprob call on the same llama-server) sends the turn to one of three
routes: a local LLM reply, a local read-only tool (time, weather, calculator, units, NBP rates, Wikipedia, server
status, holidays), or Hermes. Hermes is also reached through the LLM's `handoff_to_hermes` tool call and a
safety net. Hermes tasks run in the background on a persistent `hermes acp` worker; Gaja keeps talking and
announces each result when it is done.
Contracts: `docs/ARCHITECTURE.md`. Setup and deploy steps: `docs/SETUP.md`.

## Layout

| Path | Content |
|---|---|
| `gaja/__main__.py` | entry point: `python -m gaja` (uvicorn on `server.host:server.port`) |
| `gaja/config.py`, `config.toml` | pydantic-settings; TOML, then env `GAJA_<SECTION>__<KEY>` |
| `gaja/server.py` | FastAPI app: `GET /health`, `GET /metrics`, `WS /ws`, static `web/`; builds services, warms engines and Hermes |
| `gaja/pipeline.py` | per-connection `Session`: state machine, LLM -> TTS overlap, barge-in, handoff, Hermes-failure notice |
| `gaja/audio/vad.py`, `gaja/audio/pcm.py` | Silero endpointer (16 kHz, 512-sample windows), PCM16 helpers |
| `gaja/looplag.py` | event-loop lag monitor (WARNING + `/api/diagnostics` `loop_lag`), `freeze_startup_heap` |
| `gaja/audio/keep.py` | keeps the audio of device turns that decoded to nothing (`[metrics] empty_turn_audio_keep`, off at 0) as WAVs under `logs/empty_turns` |
| `gaja/stt/` | `base.py` protocol, `select.py` (`SwitchingSTT`: engine choice, load and inference fallback to Whistle), `faster_whisper_stt.py` (CTranslate2 on CUDA, default), `whistle.py` (CPU, single-thread executor, fallback) |
| `gaja/llm/` | `client.py` streaming llama-server client with `warm_up` and one-shot `complete`, `router.py` logprob router and safety-net regexes, `parse.py`, `history.py`, `prompt.py` |
| `gaja/text/chunker.py` | sentence chunker (Polish abbreviations, early first chunk at a comma), style-tag and "Gaja:" label strippers |
| `gaja/text/normalize.py` | numbers, units, dates and abbreviations read out for TTS (`[text] normalize_tts`) |
| `gaja/tools/` | `ToolRegistry` and the eight read-only local tools (docs/TOOLS.md) |
| `gaja/text/emotion.py` | rule-based emotion for untagged sentences (`auto_emotion`) |
| `gaja/hermes/progress.py` | `ProgressCoalescer`: batches Hermes `text_delta`, passes `tool_call` and `permission_cancelled` through |
| `prompts/router3_pl.md` | Router v3 system text (LOCAL / TOOL / HERMES); `prompts/router_pl.md` is the old 2-class v2 |
| `deploy/lfm_nothink.jinja` | chat template for llama-server (`--chat-template-file`), copy of `bench/templates/lfm_nothink.jinja` |
| `gaja/tts/base.py` | TTS protocol, engine registry, `EnginePool` (build and warm-up per engine name) |
| `gaja/tts/piper_tts.py` | `piper` (gosia, default pl) and `piper-darkman` engines; ORT session capped to `threads` |
| `gaja/tts/paradee_tts.py` | `paradee` engine (en) |
| `gaja/tts/supertonic_tts.py` | `supertonic` engine (optional, pl and en, loaded on first selection) |
| `gaja/hermes/base.py` | `HermesClient` protocol (`@runtime_checkable`) and `StubHermes` (tests only) |
| `gaja/hermes/client.py` | `AcpHermesClient`: JSON-RPC over stdio to `hermes acp`, session map, cancellation |
| `gaja/devices/` | optional IoT device listener (`WS /device`, protocol v1.8): `listener.py`, `registry.py` (tokens), `controller.py` (wake flow), `wake.py` (phrase matcher), `audio.py` (rate negotiation, resampling); docs/DEVICE.md |
| `clients/device_sim.py` | device simulator; `firmware/` holds the (untested) ESPHome config and the firmware plan |
| `gaja/metrics.py` | per-turn metrics, `logs/turns.jsonl`, p50/p90 summary |
| `deploy/` | `gaja-llm.service`, `gaja-adapter.service`, `README.md` |
| `scripts/e2e_suite.py` | all end-to-end scenarios over `WS /ws` (PL, EN, typed, barge-in, handoff) with metrics and load average |
| `scripts/e2e_smoke.py` | streams one WAV at real-time pace and prints the turn's metrics |
| `scripts/hermes_smoke.py` | one read-only task through the real Hermes worker, with timings |
| `tests/` | pytest: chunker, style, VAD, LLM parsing, WS protocol, engine pool, Hermes client, live stack check |

## Run

Prerequisites: `.venv` synced with `uv sync`; `models/pl_PL-gosia-medium.onnx(.json)` and
`models/pl_PL-darkman-medium.onnx(.json)` present; llama-server built at `third_party/llama.cpp/build/bin/llama-server`.

In production both processes run as systemd user units (see `deploy/README.md`):

```bash
systemctl --user enable --now gaja-llm gaja-adapter
curl -s http://127.0.0.1:8790/health
```

Manual run for development:

```bash
third_party/llama.cpp/build/bin/llama-server -m "$(cat bench/logs/lfm_gguf_path.txt)" \
  --host 127.0.0.1 --port 8791 -ngl 99 -c 8192 -np 2 --no-kv-unified --jinja --chat-template-file deploy/lfm_nothink.jinja \
  --reasoning-budget 0 -fa on
.venv/bin/python -m gaja          # http://127.0.0.1:8790
```

The adapter starts without the LLM and reports `llm: false` until it is up (the unit waits up to 60 s for it).
Startup builds the STT, builds and warms the default TTS engine of each language (`piper`, `paradee`; the others
load on first selection), builds the router, warms the LLM and primes both llama slots, and warms the Hermes
worker in the background. `HF_HUB_OFFLINE=1` is set in `deploy/gaja-adapter.service`, so Hugging Face loads use
local files only.

## Endpoints

- `GET /health`: `{status, llm, hermes, hermes_backend, engines, engines_loaded, engine_by_language, stt_engine, stt_requested, stt_device, stt_fallback, router}`.
  `stt_engine` is the engine serving transcription now (`faster-whisper` or `whistle`). `stt_requested` is the configured engine.
  `stt_device` is `cuda` or `cpu`. `stt_fallback` is `null` while the requested engine serves; otherwise it is `"<stage>: <error>"`
  (`load` or `inference`), and the adapter has switched to Whistle.
  `stt_degraded` is true while the preferred STT is on cooldown after a failure. `router` is `{enabled, threshold, tools}`. `hermes` is true when the ACP worker is initialized and its process is alive.
- `GET /metrics`: rolling p50/p90 over the last `metrics.window` turns.
- `GET /api/diagnostics`: `loop_lag` is `{running, warn_ms, worst_recent_ms (last 10 min), worst_ever_ms, stalls, last_stall{at, lag_ms, gc_ms}}`.
  The adapter logs `event loop blocked for N ms` (WARNING, at most once per 10 s) when a 100 ms tick wakes over 250 ms late.
- `WS /ws`: see Protocol below.

## Event loop

The device audio path runs on the asyncio loop. Measured on this box (8 cores, 1 thread per session, resident memory):

| Call (per 20 ms device frame unless noted) | Cost | Decision |
|---|---|---|
| `pcm16_to_float` / `float_to_pcm16` | 2-5 us | inline |
| `Endpointer.feed` (one Silero window per 32 ms, ONNX) | ~0.13 ms | inline: an executor hop costs more than the call, and per-frame ordering would need a queue |
| `EchoCanceller.push_ref` + `process` (caps.ref devices) | ~0.1 ms | inline |
| `filter_audio`, `WakeMatcher.match`, `record_device_activity`, `WAKE_LOG` | < 0.02 ms | inline |
| `np.concatenate` of a 30 s utterance (speculate / smart-turn submit) | ~1.4 ms, once per utterance | inline |
| `beep_pcm16`, resampling a TTS reply (> 100 ms in `to_thread`, shorter inline) | 0.1-2.5 ms | inline short, offloaded long |
| Silero session build + first inference | 90-140 ms cold | warmed at startup (`warm_vad`), per-session build stays in `to_thread` at hello |
| Smart Turn inference, STT (verify and turn), TTS, metrics write | tens of ms and up | already in executors |
| Full garbage collection | 26 ms with bare imports resident | `gc.freeze()` after startup (`freeze_startup_heap`) |

`gaja/looplag.py` reports what the table cannot: a loop that was blocked by page-in or a collection.

## Voices

| Engine | Language | Default | Selection |
|---|---|---|---|
| `piper` | pl | yes (`tts.engine_by_language.pl`) | default |
| `piper-darkman` | pl | no | `hello.tts = "piper-darkman"` or a household voice |
| `paradee` | en | yes (`tts.engine_by_language.en`) | default |
| `supertonic` | pl, en | no | `hello.tts = "supertonic"`; loaded on first selection, about 1 GB RAM |

`ready.engines` and `/health.engines` list every selectable name. `engines_loaded` shows which are built.
Engines are built and warmed once per process (`EnginePool`): the defaults at startup, the others on first use. A
failing optional engine returns an `error` to the client that selected it; the default engines stop startup on
failure.

Household voices work the same way: `Person.tts` and `Household.general_voice` (edited in the admin panel)
select the engine per speaker in `Session._play` — the speaker's voice wins over `hello.tts` and
`engine_by_language`, unknown voices get `general_voice`, and a voice that cannot speak the turn's language
falls back to the global engine for that turn. `GET /api/admin/tts/voices` lists the selectable voices per
language from the declarative catalog (`gaja/tts/catalog.py`) without building anything. RAM: every distinct
voice named by a person or the household is one more engine kept for the process lifetime — the first turn
that uses it pays the build and warm-up (a Piper voice is one ONNX model; `supertonic` is about 1 GB), after
which it behaves like a preloaded default.

## Configuration

`config.toml` holds the defaults (commented). Environment overrides use `__` for nesting, e.g.
`GAJA_LLM__BASE_URL=http://127.0.0.1:8791`, `GAJA_SERVER__PORT=8790`. `GAJA_CONFIG` selects another TOML file.

Notable keys:
- `devices.*`: IoT device listener, `enabled = false` by default (docs/DEVICE.md); not enabled in `config.toml`.
- `llm.tool_strategy = "native"`: llama-server parses tool calls (`--jinja`). `"pythonic"` parses `<|tool_call_start|>[...]<|tool_call_end|>` from content instead.
- `router.*`: `enabled = true`, `prompt_path = prompts/router3_pl.md`, `threshold = 0.5` (HERMES mass), `tool_threshold = 0.6`
  (TOOL mass), `history_turns = 2` (chat messages, so 2 = the previous exchange), `timeout_s = 2.0`. The router runs on
  llama slot 1 and adds `router_ms` to each turn. A router outage falls back to the local reply.
- `tools.*`: `enabled = true`, `default_location = "Katowice"` (weather without a city), `timeout_s = 4.0`.
- `text.normalize_tts = true`: every sentence goes through `normalize_for_tts` before TTS; the on-screen text is unchanged.
- `llm.prime_after_turn = true`: after each turn the chat slot (0) is pre-filled with the conversation, so the next
  reply prefills only the new user message. The router slot (1) is always primed with system prompt + recent history.
- `style.auto_emotion = true`: untagged sentences get a rule-based emotion (`excited`, `happy`, `sad`, `calm`, `serious`, else `neutral`).
  An explicit `[tag]` always wins.
- `stt.engine = "faster-whisper"` (production) or `"whistle"`. `stt.faster_whisper.*`: `model = "small"`, `device = "cuda"`,
  `compute_type = "int8"`, `beam_size = 1` (beam 1 measured as the better latency trade-off; PL WER 0.187 vs 0.169 at beam 5).
  `hello.lang` forces the transcription language, so Whisper is not asked to detect it.
- Fallback: if faster-whisper fails to load (for example CUDA OOM at model build or warm-up), or raises during a decode,
  the adapter logs the error with traceback, serves from Whistle, and reports it in `/health`. The failed utterance is retried on Whistle.
  Whistle is built lazily, so it is only resident when it is used. The adapter does not crash and does not touch Hermes.
- `chunker.first_chunk_min_words = 3`: the first sentence may be cut at a comma once it has this many words.
- `vad.*`: threshold 0.5, 300 ms pre-roll, 500 ms min silence, 250 ms min speech, 60 s max utterance.
  `spec_silence_ms = 250`: speculative endpointing (below). `barge_in_ms = 250` of speech fires a barge-in; while Gaja's
  own audio plays (plus 0.4 s) it takes `barge_in_strict_ms = 400` at `strict_threshold = 0.7` (echo guard).
- `tts.engine_by_language`: `pl = "piper"`, `en = "paradee"`.
- `tts.piper.threads = 2`, `tts.supertonic.*`: `steps = 6`, `voice = "M4"`, `speed = 1.05`, `threads = 2`, `output_sample_rate = 22050`.
  `threads` caps the ONNX Runtime intra-op pool of each engine.
- `hermes.backend = "acp"`: real Hermes via `hermes.command` (`hermes acp`), working directory `hermes.cwd` (`~` expanded), per-turn `hermes.timeout_s = 300`.
  `"stub"` is only for tests.
- `hermes.direct_max_chars = 160`: Hermes results up to this length are spoken as-is; longer ones are summarized by the LLM.
- `hermes.progress_interval_s = 0.5`: Hermes `text_delta` progress is sent at most this often; `tool_call` and `permission_cancelled` are sent at once.

## Protocol notes

The WS protocol follows `docs/ARCHITECTURE.md` (v1.9). Behaviour the contract does not spell out:
- `hello` must come first; any other message before it returns `error: send hello first`. A repeated `hello`
  changes language/voice and keeps the history.
- `ready` carries `tts` (the active engine), `sample_rate`, `langs`, `engines` and `engine_by_language`.
- An unknown `hello.tts` returns `error: unknown tts engine '<name>'`. An engine that fails to load returns `error: tts engine unavailable: <detail>`.
- WS handshakes from a browser `Origin` outside `server.allowed_origins` get HTTP 403 (loopback origins always pass;
  no `Origin` passes). Static files come from `web/` only.
- Barge-in: a `speech_start` during `thinking` or `speaking` cancels the reply and TTS and sends
  `state{listening, reason:"barge_in"}`, also when the state was already `listening`. It does **not** cancel Hermes.
- Speculative endpointing: after `vad.spec_silence_ms` of silence, STT, router and the reply start, but every event,
  history write, tool call, Hermes start and voice cancel is held until `min_silence_ms` confirms the end of speech. If the
  user keeps talking, the speculation is dropped with no visible effect. `metrics.spec_ms` and
  `committed_speculative` record it.
- Routing: the router decides LOCAL / TOOL / HERMES. Explicit addressing ("zapytaj Hermesa…", "Hermes, …",
  "w Hermesie", "przez Hermesa", "ask Hermes") goes to Hermes without the model; "Hermes" as the god or the brand
  never does. TOOL runs a constrained `select_tool` call (skipped when no tool cue occurs), executes the tool, sends
  `tool{name, arguments, ok, text}` and speaks the tool's sentence (on failure its error sentence, `ok:false`). No tool
  picked gives a local reply.
- Safety net: a local reply that says it will hand off (PL and EN regexes in `gaja/llm/router.py`) without a tool call
  is replaced by a handoff of the same turn, so no unexecuted promise is spoken.
- A handoff emits `handoff{state:"started", id, text: <reason>}` and a fixed ack (rotated per language in `ACKS`), then the
  turn ends and the task runs in the background (`progress` events). When it finishes, `handoff{state:"done", id}` is
  sent and Gaja announces the result (directly when short, otherwise summarized by the LLM) as soon as she is not
  speaking and the user is not talking; that turn's `metrics` has `kind:"result"`.
- Timers and reminders (protocol v1.9): when a timer or reminder expires, the server broadcasts a WS event `{"type":"timer","id":"...","label":"...","kind":"timer|reminder"}` to all connected sessions (web, CLI, device) and synthesizes a spoken announcement for sessions with audio. A timer that expired while no session could hear it stays pending (persisted in `data/timers.json`) and is announced at the next hello; it also shows up as overdue in `GET /api/timers` and in the status tool (`"overdue": true`, `remaining_seconds` 0), and `DELETE /api/timers/{id}` cancels it so it is not spoken later. A pending timer held back only by device quiet hours is flushed once when the window ends. REST API: `GET /api/timers`, `DELETE /api/timers/{id}`, `GET /api/memory`, `POST /api/memory`, `DELETE /api/memory/{id}` (with Origin check matching `server.allowed_origins`).
- Cancelling Hermes: `cancel_task{id?}`, `interrupt{scope:"all"}`, or a phrase that names the task —
  "anuluj zadanie" / "przerwij to zadanie" / "zatrzymaj timer" / "stop the task" / "Hermes, stop" — while a
  task runs (fillers like "ok", "okej", "dobra", "no to" and "proszę" may come first). Bare interrupt words
  ("stop", "przestań", "przestań już", "przerwij", "zatrzymaj", with or without fillers or "proszę") are one
  class: barge-in only. They stop Gaja's playback, never cancel a task, and are never routed to the LLM or
  Hermes as a new request — when nothing is playing the turn ends silently. The adapter sends
  `session/cancel`, emits `handoff{state:"cancelled", id, text:""}`
  and (for the spoken form) says "Dobrze, anulowałam zadanie Hermesa."
- `metrics.router_ms` is the router round trip; it is `null` when the router did not run.
- Hermes unavailable, failed or silent: the adapter emits `handoff{state:"error", text: <detail>}` and speaks
  `Hermes jest teraz niedostępny, nie udało się wykonać zadania.` The session stays in the voice loop.
- STT failure: the utterance is retried on Whistle and `notice{code:"stt_degraded"}` is sent; `stt_recovered` follows
  when the GPU engine passes a re-probe.
- LLM outage: the adapter speaks `Model językowy jest chwilowo niedostępny.` and sends `error`. It does not crash.

## Hermes

The `hermes acp` worker is one process for the adapter's lifetime. It is started and initialized by the first
`health()` call (at startup) and respawned on the next request if it exits or its reader dies (16 MiB line limit). Each Gaja WebSocket session maps to one
Hermes session, created on its first handoff. The mapping is dropped when the WebSocket closes (`HermesClient.forget`),
so the session map is bounded by the number of open connections. Hermes 0.19.0 has no `session/close`, so the
worker keeps forgotten sessions in memory until the adapter restarts. Permission requests from Hermes are auto-cancelled,
so a handoff can only run tasks that need no approval. Progress from Hermes is coalesced: `text_delta` at most every
`hermes.progress_interval_s`, while `tool_call` and `permission_cancelled` are sent at once. See `docs/HERMES.md`
for the worker's protocol and earlier measurements.

## Measured latency

Measured with `scripts/e2e_suite.py` against the running stack (systemd units, real STT, LLM and TTS, real Hermes).
Load average is `/proc/loadavg` 1-minute value at the start and end of the run.

### Round 6/6b (2026-10-09), current

14 audio turns per run (10 PL + 4 EN latency clips, `--scenarios pl,en`), each run under `/tmp/gaja-bench.lock`
after a restart and a warm-up, started at load < 1.5. Values are p50 / p90 in ms; `wall` is client-side, from the
end of the clip's speech to the first audio frame. Source: `bench/out/e2e/<run>.json`.

| run | n | stt | router | llm_ttft | first_sentence | tts_first | e2e_first_audio | wall | load start → end |
|---|---|---|---|---|---|---|---|---|---|
| r6_baseline | 14 | 405 / 461 | 345 / 444 | 227 / 1594 | 1598 / 3093 | 165 / 274 | 1530 / 2918 | 1995 / 3474 | 0.25 → 1.32 |
| r6_after | 14 | 400 / 453 | 319 / 362 | 344 / 453 | 1343 / 1567 | 140 / 254 | 1448 / 1739 | 2031 / 2305 | 0.47 → 0.94 |
| r6b_spec0_b | 14 | 396 / 452 | 125 / 139 | 367 / 505 | 1247 / 1407 | 158 / 267 | 1360 / 1589 | 1920 / 2107 | 1.49 → 1.27 |
| r6b_spec250_b | 14 | 415 / 460 | 122 / 142 | 312 / 460 | 912 / 1149 | 164 / 304 | 1077 / 1282 | 1615 / 1845 | 1.38 → 1.11 |
| r6b_final | 14 | 392 / 444 | 124 / 135 | 191 / 206 | 747 / 816 | 130 / 282 | 890 / 970 | 1428 / 1495 | 1.19 → 1.07 |
| r6b_final_b | 14 | 399 / 442 | 117 / 140 | 187 / 204 | 739 / 913 | 157 / 264 | 945 / 1045 | 1476 / 1557 | 1.43 → 1.10 |
| r6b_final_c | 14 | 389 / 444 | 122 / 145 | 166 / 204 | 712 / 775 | 152 / 261 | 905 / 976 | 1415 / 1547 | 1.31 → 1.07 |

- `r6_baseline`: code before round 6. `r6_after`: round 6 fixes, no speculation. `r6b_spec0_b` / `r6b_spec250_b`:
  history-primed router, speculation off / 250 ms. `r6b_final*`: plus the chat prime after each turn
  (`final_c` also has the explicit "w Hermesie" rule and the "Gaja:" strip).
- **router_ms 319 → ~120 ms.** The prime cached only the router's system prefix, so ~90 history tokens were
  evaluated on every classify. The router slot is now primed with system prompt + recent history after each turn;
  the classify call evaluates ~30 tokens. Probe (`/tmp/gaja-r6/probe_router_hist.py`): median 256 → 112 ms.
- **llm_ttft ~350 → ~190 ms.** The chat slot re-prefilled the last exchange (103–129 tokens) per turn. The chat
  prime after each turn leaves 13–14 tokens for the request (llama-server journal, `prompt eval` per request).
- **first_sentence vs llm_ttft.** `first_sentence_ms` = time to the first chunk handed to TTS. Without speculation it
  is STT (~390) + router (~125) + TTFT + ~350 ms of generation to the 5-word first chunk (~11 tokens at ~34 tok/s,
  median `gen_to_first_chunk` 351/353 ms in the spec0 runs). With speculation it is measured from the commit, which
  hides the STT and router time behind the 250 ms of confirmation silence.
- **Speculation: kept at 250 ms.** Sweep (same day, same procedure; `r6b_spec{0,200,250,300}`, `_b` repeats):
  wall p50 0 → 1870 / 1920, 200 → 1722, 250 → 1584 / 1615, 300 → 1684. In the EN sessions exactly one speculation
  per session was discarded (`committed_speculative: false` in `logs/turns.jsonl`) with nothing leaking to the client.
- **Typed turns** sent the instant `listening` arrives can show router_ms 300–900 ms: the classify queues behind the
  post-turn prime on the same slot. With a 3 s gap it is 113–218 ms (`/tmp/gaja-r6b/gap_probe.log`). A voice turn
  cannot arrive that early (the reply is still playing).
- VRAM (`nvidia-smi`, 2026-10-09 19:07): 3225 / 4096 MiB idle, peak 3353 MiB sampled every 200 ms over 4 PL audio turns
  (llama-server 1856, adapter incl. faster-whisper 378, a Hermes-owned process 986).

### Live verification (2026-10-09, adapter on :8790, scripts in `/tmp/gaja-r6b/`)

| Check | Result | Evidence |
|---|---|---|
| Audio PL / EN turns | observed | e2e runs above, 14/14 turns answered |
| Typed PL / EN turns | observed | `live_tools.log`, `r6b_live_checks.json` |
| Tools: weather (default + Kraków tomorrow), time, calc, currency, wiki | observed, all `ok` | `live_tools.log` |
| Barge-in while speaking | observed: `listening{barge_in}` 1.8 ms after `speech_start`, 0 audio frames after | `live_D.log` |
| Barge-in while thinking | observed: cut in `thinking`, `listening{barge_in}` 1.3 ms after `speech_start` | `r6b_live_checks.json` |
| Hermes read-only task + barge-in during it | observed: the local question (calc) answered while the task ran; the task finished (`done`, same id) and was announced | `live_AC.log` |
| Voice "Anuluj zadanie." during a Hermes task | observed: `handoff{cancelled}` same id, "Dobrze, anulowałam zadanie Hermesa."; Hermes log `Cancelled session …`, `Turn ended: reason=interrupted_during_api_call`; no late events in 25 s; Hermes processes unchanged | `live_B.log`, adapter journal |
| `cancel_task` with no task / non-string id | observed: `error` for both | `live_AC.log` |
| Routing 4 Hermes + 4 local typed | observed 4/4, 0 false handoffs | `bench/out/e2e/r6b_final_c.json` |
| Cross-origin WS | observed: `Origin: http://evil.example` → 403; tailnet origin and no Origin → 101 | curl |
| Static path traversal | observed: `/../`, `%2e%2e`, `..%2f`, `%252e`, `/static/../../` → 404, no file content | curl `--path-as-is` |


Round 4 (`bench/e2e.sh --pl-clips 10 --en-clips 4`, same command and clips before and after; load average is the
1-minute value from `/proc/loadavg`):

| Field | before p50 | before p90 | after p50 | after p90 | n (before / after) |
|---|---|---|---|---|---|
| `stt_ms` | 414 | 448 | 400 | 442 | 15 / 15 |
| `router_ms` | 206 | 225 | 202 | 271 | 17 / 17 |
| `llm_ttft_ms` | 261 | 1801 | 274 | 1366 | 16 / 16 |
| `first_sentence_ms` | 1455 | 1521 | 1630 | 2675 | 3 / 9 |
| `tts_first_ms` | 160 | 342 | 152 | 262 | 17 / 17 |
| `e2e_first_audio_ms` | 1894 | 3144 | 1583 | 2300 | 17 / 17 |
| wall time from speech end | 2298 | 3708 | 1966 | 2827 | 17 / 17 |

Load average (1-minute) start and end: before 0.47 and 1.05, after 0.60 and 0.45. GPU memory (`nvidia-smi`): llama-server
config before 3127 MiB total, after 3217 MiB idle and 3225 MiB during the run. `first_sentence_ms` is not comparable
across the two runs: after the change it is measured on 9 turns instead of 3, because the 5-word cap now cuts most
replies. Over all 17 turns `e2e_first_audio_ms` fell by 311 ms at p50 and 844 ms at p90.

The 1.2 s target is not met (1583 ms p50). Turns PL 8 and 9 still have `llm_ttft_ms` of 1.8 s with ~510-token prefills
(`prompt eval` 940 ms); the cause is not confirmed. The llama journal shows `selected slot by LRU` on unpinned requests
during the run, which points at another client on :8791, not the adapter's pinned slots.

Round 4 changes and their measured effect:
- llama-server `-c 8192 -np 2 --no-kv-unified` (4096 per slot) replaces `-c 2048 -np 2 --kv-unified`. With the old
  setting, restoring an evicted slot failed (`failed to find 1708 available cells in kv cache`) and the chat prompt was
  re-prefilled (~767 tokens, ~1.4 s). With the new setting the restore failures are gone; chat TTFT p50 ~175 ms in the
  probe, with 910 cached tokens.
- Chat slot pinning (`id_slot`) on `/v1/chat/completions` works; pinned chat with no chat prime: TTFT p50 162 ms,
  911 cached tokens.
- Priming the chat prefix on `speech_start` was removed. The prime overwrote the slot's cached conversation, so the
  next turn re-prefilled the history: TTFT p50 521 to 1623 ms in the probes. The router prime stays (router `cache_n` 455).
- Speculative parallel chat (router and chat concurrently) was not built: concurrent chat TTFT p50 was 328 ms, against
  about 317 ms for router then chat in sequence, so there is no gain.
- `chunker.first_chunk_max_words = 5` cuts the first sentence after 5 words when no punctuation has come yet. Replayed
  first-chunk p50 on the same probe streams fell from 544 to 481 ms (base) and from 569 to 473 ms (new config); p90
  fell by 100 to 170 ms. The cut lands mid-phrase; prosody was not auditioned by ear.
- STT was not changed: `without_timestamps` costs WER (0.169 to 0.178 cWER at beam 5), so it was not made default.

Round 3 run (`bench/e2e.sh --pl-clips 10 --en-clips 4`, STT faster-whisper, router prefix primed on `speech_start`).
Load average 1-minute 6.62 at the start of the run and 0.71 at the end. The start value was not quiet (another agent's
benchmark was running), so these numbers are indicative only. 17 turns: PL 10, EN 4, plus 2 typed. Times in ms.

| Field | p50 | p90 | n |
|---|---|---|---|
| `stt_ms` | 396 | 442 | 15 |
| `router_ms` | 261 | 453 | 17 |
| `llm_ttft_ms` | 706 | 2961 | 16 |
| `first_sentence_ms` | 1743 | 4216 | 16 |
| `tts_first_ms` | 215 | 356 | 17 |
| `e2e_first_audio_ms` | 1997 | 4380 | 17 |
| wall time from speech end | 2261 | 4953 | 17 |

The p50 end-of-speech to first audio is 2.0 s, above the 1.2 s target. The slowest turns (PL 7 to 10) have
`llm_ttft_ms` of 2.7 to 3.2 s, which points at the shared machine, not the router. Routing 8/8 (precision 1.0,
recall 1.0), barge-in verified (see below).

Router prefix cache (probe against llama-server, same prompt family, 6 utterances): the system prefix is about 480
tokens. Without priming a call re-evaluated the whole prompt (`cache_n` 0 to 95, ~680 to 900 ms). With the system
prefix primed, `cache_n` was 455 and the call took ~105 ms. In the live run `router_ms` p50 fell from 885 to 261 ms,
so part of the router time is still outside the cache.

Previous run, plain turns (`pl`, `en`, n = 7; pl 5 clips, en 2 clips), at 21:19. Load average 10.8 at start, 7.2 at end; the
machine was shared with other agents' benchmarks (1-minute load 10 to 14 during the session), so these numbers are
pessimistic. Times in ms.

| Field | p50 | p90 |
|---|---|---|
| `stt_ms` | 1155 | 2525 |
| `router_ms` | 932 | 968 |
| `llm_ttft_ms` | 220 | 999 |
| `first_sentence_ms` (n = 4) | 3313 | 4501 |
| `tts_first_ms` | 373 | 553 |
| `e2e_first_audio_ms` | 3502 | 4377 |
| wall time from speech end | 3644 | 4960 |

Router A/B for the llama-server slot count (`/tmp/gaja-check/ttft.py`, 30 samples, six prompts, five rounds):

| Setup | Metric | p50 | p90 |
|---|---|---|---|
| `-np 1` | chat only, first event | 1051 | 3548 |
| `-np 1` | chat after router | 129 | 2640 |
| `-np 2 --kv-unified` | chat only, first event | 1090 | 3517 |
| `-np 2 --kv-unified` | chat after router | 128 | 1070 |
| both | router round trip | 978 / 1022 | 1017 / 1041 |

The router round trip stayed near 1 s at p50 under both slot settings. It is the largest latency item and is not
explained yet.

Routing (live, six text turns through `WS /ws`): 3 read-only Hermes prompts all handed off (precision 1.0,
recall 1.0 with the 3 local prompts not handing off). Ack first audio came 1.2 to 1.3 s after the turn was sent.
Hermes completion took 18 s, 26 s and 112 s, so the Hermes side dominates handoff latency.

Barge-in (live, after the fix in this session):

| Scenario | Cut during | `speech_start` after cut | `listening` after `speech_start` | Audio after `speech_start` | `listening` reason |
|---|---|---|---|---|---|
| `barge` | speaking | 1155 ms | 0.2 ms | 0 frames | `barge_in` |
| `barge_think` | thinking (Hermes handoff) | 1155 ms | 0.0 ms | n/a | `barge_in` |

Before the fix, `barge_think` reached `listening` only after the Hermes result was spoken (49.9 s) and without a reason.
The 1.15 s from cut to `speech_start` is the VAD detection delay on the cut clip; it was not investigated further.

## Known gaps

- The 1.2 s end-of-speech to first-audio target is not fully met: wall p50 1415–1476 ms (`r6b_final*`). About 520 ms
  of it (wall minus `e2e_first_audio_ms`, p50 890–945 ms) is the 500 ms end-of-speech confirmation plus VAD lag.
- The chat LLM sometimes hands garbled STT text to Hermes itself (`handoff_to_hermes` tool call on a local turn);
  Hermes answers "Nie łapię…". Harmless but slow.
- LLM summaries of long Hermes results can lose meaning (Hermes' "suszy" — drier — became "susza", drought, in a
  live weather summary). Results up to 160 characters are spoken directly.
- Paradee (en) synthesis is slower than real time on this CPU; English `tts_first_ms` p50 is ~250 ms against ~140 ms for PL.
- Whistle (fallback STT) `auto` language detection is not separately validated; faster-whisper is the default and
  `hello.lang` forces its language.
- A history past ~4000 tokens per slot returns "Context size has been exceeded"; history is trimmed well before
  (`history_max_chars = 6000`), and the adapter treats the error as `LLMUnavailable`.
- Router v3 was tuned on Polish; English routing is covered by the eval's EN rows and live tests only.
- `temperature = 0.3`: simple arithmetic in a local reply can still be wrong; arithmetic often (not always) routes
  to the `calculate` tool.
- A one-segment STT decode cannot be interrupted mid-segment (REVIEW L2).
- No authentication on `/ws`; the adapter binds to `127.0.0.1` and checks `Origin`.
- `assistant_delta` is not sent; Hermes `progress` is coalesced text fragments.
