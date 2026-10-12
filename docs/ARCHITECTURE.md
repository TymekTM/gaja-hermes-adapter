# Gaja realtime voice adapter — architecture & contracts

This file is the contract between parallel workstreams. Do not change an interface here
without noting it in the "Change log" at the bottom; other agents code against it.

## Pipeline

```
mic (browser / CLI) --PCM16 16k--> WS /ws --> VAD (silero, CPU) --utterance--> STT (Whistle, CPU)
   --> LLM (LFM2.5-2.6B, llama-server on GPU, streamed)
   --> SentenceChunker + StyleTagParser --> TTS (per sentence, CPU) --PCM16--> WS --> speaker
                              \--> handoff_to_hermes(task) --> HermesClient (background) --> result
                                                                  --> LLM summarises --> TTS
```

Latency budget (end of user speech -> first audio byte), target <= 1.2 s:
VAD endpoint silence ~500 ms (not counted) | STT <= 300 ms | LLM TTFT <= 250 ms |
first sentence <= 400 ms | TTS first sentence <= 300 ms.

## Package layout (Python 3.11, uv venv `.venv`)

| Path | Owner workstream | Content |
|---|---|---|
| `gaja/config.py`, `config.toml` | core | pydantic-settings config loaded from `config.toml`, env overrides `GAJA_*` |
| `gaja/audio/vad.py` | core | silero-vad wrapper, endpointing |
| `gaja/stt/` | core | `base.py` protocol, `whistle.py` |
| `gaja/llm/` | core | `client.py` streaming llama-server client, tool-call parsing |
| `gaja/text/chunker.py` | core | sentence chunking of the token stream |
| `gaja/text/style.py` | prosody | style-tag parser (`[happy]` etc.) -> `Style` |
| `gaja/tts/base.py` | core | TTS protocol + registry |
| `gaja/tts/piper_tts.py`, `gaja/tts/paradee_tts.py` | core | engines |
| `gaja/tts/prosody.py` | prosody | `Style` -> engine params, pitch shift / post-processing |
| `gaja/tts/<new>.py` | tts-scout | any new better-quality engine, same protocol |
| `gaja/hermes/` | hermes | `base.py` protocol + `StubHermes` (core writes these two), `client.py` real client (hermes workstream) |
| `gaja/pipeline.py` | core | per-connection `Session` state machine, barge-in |
| `gaja/server.py` | core | FastAPI app: `GET /` serves `web/`, `WS /ws`, `GET /health`, `GET /metrics` |
| `web/` | web | static browser client (no build step: plain HTML/JS/CSS) |
| `clients/cli_wav.py` | web | CLI client: streams a wav file as a fake mic, saves reply audio, prints timings |
| `deploy/` | core | systemd user units, run scripts |
| `tests/` | everyone | pytest, per module |

## Core contracts (Python)

```python
# gaja/text/style.py
@dataclass(frozen=True)
class Style:
    emotion: str = "neutral"      # one of STYLE_TAGS emotions
    rate: float = 1.0             # speed multiplier, >1 faster
    pitch_semitones: float = 0.0
    energy: float = 1.0           # loudness / expressiveness multiplier
    events: tuple[str, ...] = ()  # paralinguistic events e.g. ("laugh",) – engines that can't, ignore

STYLE_TAGS = ["neutral", "happy", "excited", "calm", "sad", "serious", "fast", "slow", "laugh", "sigh"]
def parse_styled(text: str, default: Style) -> list[tuple[Style, str]]  # strips tags; unknown tags dropped, never spoken

# gaja/stt/base.py
@dataclass
class Transcript: text: str; language: str; duration_ms: float; audio_s: float
class STT(Protocol):
    async def transcribe(self, pcm: np.ndarray, language: str | None = None) -> Transcript  # float32 mono 16 kHz

# gaja/tts/base.py
class TTS(Protocol):
    name: str
    sample_rate: int
    languages: set[str]           # e.g. {"pl"} / {"en"}
    async def synthesize(self, text: str, style: Style, language: str) -> np.ndarray  # float32 mono
def get_tts(name: str, **cfg) -> TTS   # registry; engines register via @register_tts("piper")

# gaja/hermes/base.py
@dataclass
class HermesEvent:
    kind: Literal["accepted", "progress", "result", "error"]
    text: str
    data: dict | None = None
class HermesClient(Protocol):
    async def submit(self, task: str, *, context: list[dict], session_id: str) -> AsyncIterator[HermesEvent]
    async def health(self) -> bool
```

Blocking engines (onnxruntime, Whistle) run in a thread pool (`asyncio.to_thread`) so the event loop
never stalls. The LLM is a separate process (llama-server) on `127.0.0.1:8791`.

## LLM output format (produced by LFM, consumed by chunker/style parser)

Plain spoken text, optionally with style tags at the start of a sentence: `[happy] Super, udało się! [calm] Teraz sprawdzę resztę.`
Hand-off is the native LFM tool call `handoff_to_hermes(task: str, reason: str)` (exact strategy decided by
`docs/LLM_RESULTS.md`; the LLM client exposes it as a `ToolCall` event regardless of strategy).

## WebSocket protocol `WS /ws`

Client -> server
- binary frame: PCM16 LE mono 16 kHz, any size (20–100 ms recommended)
- `{"type":"hello","lang":"pl"|"en"|"auto","tts":"<engine name or null>","client":"web|cli"}` (first message)
- `{"type":"text","text":"..."}` typed input, skips STT
- `{"type":"interrupt"}` stop current speech (barge-in from client side); Hermes tasks keep running.
  `{"type":"interrupt","scope":"all"}` also cancels every running Hermes task (v1.7).
- `{"type":"cancel_task","id":"<handoff id>"}` cancel one Hermes task; without `id`, all of them. With no matching
  task the server answers `error` (`no running Hermes task`); a non-string `id` is an `error` too (v1.7).
- `{"type":"end_of_input"}` CLI client: no more audio, flush VAD and treat as end of speech
- `{"type":"ping"}`

Server -> client (JSON text frames unless noted)
- `{"type":"ready","session_id":...,"tts":...,"sample_rate":<tts sr>,"langs":[...]}`
- `{"type":"vad","state":"speech_start"|"speech_end"}`
- `{"type":"transcript","text":...,"lang":...,"final":true}`
- `{"type":"assistant_delta","text":...}` raw LLM text as it streams (tags stripped). **Not sent by the current
  server**; clients build the reply from `utterance` events.
- `{"type":"utterance","id":n,"text":...,"style":{...}}` sentence about to be spoken
- `{"type":"audio_start","id":n,"sample_rate":sr}` then **binary frames** PCM16 LE mono at `sr`, then `{"type":"audio_end","id":n}`
- `{"type":"handoff","state":"started"|"progress"|"done"|"error"|"cancelled","id":"<8 hex>","task":...,"text":...}`.
  `id` is stable for one task; several tasks may run at once. `text`: on `started` the reason (`explicit`, `router`,
  `safety-net` or the model's reason), on `progress`/`done`/`error` Hermes' text, on `cancelled` empty. When a task
  is done Gaja announces the result as its own turn (`utterance`s and `metrics` with `kind:"result"`).
- `{"type":"tool","name":...,"arguments":{...},"ok":bool,"text":...}` a local tool ran (v1.7); `text` is what Gaja
  speaks.
- `{"type":"notice","code":"stt_degraded"|"stt_recovered","message":...,"engine":...}` informational, sent on a
  state change only (v1.7).
- `{"type":"metrics","turn":n,"kind":"audio"|"text"|"result","route":"local"|"tool"|"hermes"|"cancel"|null,
  "tool":...,"stt_ms":...,"router_ms":...,"tool_select_ms":...,"tool_ms":...,"llm_ttft_ms":...,"first_sentence_ms":...,
  "tts_first_ms":...,"e2e_first_audio_ms":...,"spec_ms":...,"committed_speculative":bool}` (unknown fields: ignore)
- `{"type":"state","state":"listening"|"thinking"|"speaking","reason"?:"barge_in"}`
- `{"type":"error","message":...}`, `{"type":"pong"}`

Barge-in: if VAD detects user speech while the server is `speaking` or `thinking`, the server cancels the
pending TTS/LLM, sends `{"type":"state","state":"listening","reason":"barge_in"}` and the client must drop queued
audio immediately. Barge-in does **not** cancel Hermes tasks (v1.7); `cancel_task`, `interrupt{scope:"all"}` or the
spoken "anuluj zadanie" do.

The WS handshake is refused with HTTP 403 when the browser's `Origin` is not in `server.allowed_origins`
(loopback origins are always allowed; a missing `Origin`, as from CLI clients, is allowed).

### Device clients, protocol v1.8 (`WS /device`)

IoT voice devices with on-device wake word use a **separate, optional listener** (`[devices] enabled`, default
off, port 8792, only `/device`, per-device token from `~/.config/gaja/devices.toml`). Same event loop and services;
the `Session` machinery is shared. Full description: `docs/DEVICE.md`. Delta to the protocol above:

Client -> server
- `{"type":"hello","client":"device","device_id":...,"token":...,"fw":...,"lang":...,
  "caps":{"aec":bool,"out_rates":[16000,...],"preroll_ms":0..5000}}` (first message, authenticated; `client:"device"`
  is refused on `/ws` and `web|cli` on `/device`)
- `{"type":"wake","word":"hej_gaja","score":0..1,"preroll_ms":n}` followed by pre-roll PCM16 16 kHz (binary) and live frames

Server -> client
- `ready` additionally carries `device:{id,aec,out_rate,verify_wake,verify_ms,follow_up_ms,ack}`; `sample_rate`
  is the negotiated output rate (16000 preferred, else lowest offered >= 16 kHz, else highest); all TTS audio is
  resampled to it (`audio_start.sample_rate` equals it)
- `{"type":"wake_accepted"}`, `{"type":"wake_rejected","reason":"no_phrase|no_speech|timeout|low_score|stt_error|superseded"}`
- `{"type":"follow_up","ms":6000}` keep the mic open without a wake phrase; `{"type":"listen_end"}` stop streaming
- `transcript.heard` (optional): the text before the wake phrase was stripped; `metrics.route` may be `"wake"` (bare phrase acknowledged)
- Verification (`verify_wake`): the first `verify_ms` is transcribed and must start with hej/hey/ej + gaja/gajo/gaju/gaia/gaya.
- Barge-in: with `caps.aec` the device keeps streaming (VAD barge-in); without it the device mutes the mic during playback and a
  wake (score >= `barge_in_min_score`) cancels the reply. Hermes tasks keep running either way.

## Ports & processes

- adapter (FastAPI/uvicorn): `127.0.0.1:8790` (env `GAJA_HOST`/`GAJA_PORT`)
- llama-server (LFM2.5): `127.0.0.1:8791` (benchmarks use 8792+)
- Browser mic requires a secure context: use `http://localhost:8790` locally, or remote via
  `tailscale serve --bg --https=8443 http://127.0.0.1:8790` (ttm-server is on the tailnet as 100.107.121.67).
- Never touch: Hermes processes / `~/hermes-agent`, `~/voice-agent` (old service `gaja-voice.service`, now disabled).

## Shared-machine rules for agents

- Wrap every TIMED benchmark run in `flock /tmp/gaja-bench.lock <cmd>` so timings never overlap.
- Change dependencies only via `flock /tmp/gaja-uv.lock uv add <pkg>`; never hand-edit `pyproject.toml`/`uv.lock`.
- Only edit files owned by your workstream (table above); if you need a change elsewhere, write it under
  "Requests" below instead.
- No git commits.

## Requests (cross-workstream)

Status notes marked **r6b** were added by core in round 6b (2026-10-09); the request text above them is kept as
history.

- [lead -> core] Hermes: spawn/warm the ACP worker at adapter startup (`HermesClient.health()` starts it; cold first turn is ~15 s, warm ~5 s — see docs/HERMES.md). Keep one long-lived HermesClient per adapter process, map Gaja session -> Hermes session.
  **r6b: done** (v1.3 warm-up; one client per process, one ACP session per WS session).
- [lead -> all] The machine is heavily loaded while many agents work in parallel (load avg ~20). Any timing you take now is load-contaminated: record `cat /proc/loadavg` next to it and say so. The lead runs ONE final quiet timing pass of all benchmarks at the end; make your benchmark runnable as a single documented command.
  **r6b: done** — `bench/e2e.sh`; every r6b number in docs/ADAPTER.md has its loadavg next to it.
- [lead -> core] Don't keep a CPU-only llama-server (-ngl 0) running longer than needed — it eats all 4 cores and starves the other agents' benchmarks.
  **r6b: obsolete** — only the GPU llama-server on 8791 runs.
- [core -> lead] `pyproject.toml`: add `[tool.pytest.ini_options] testpaths = ["tests"]`. Without it, `pytest` collects `third_party/llama.cpp` test modules and fails collection. Core runs `pytest tests` meanwhile. **r6b: done** (`testpaths = ["tests"]` is in pyproject.toml).
- [core -> lead] `.gitignore`: add `models/` (Piper/Paradee weights, ~100 MB+) and `logs/` (runtime `turns.jsonl`, server logs). **r6b: done** (both are in `.gitignore`).
- [core -> hermes] `tests/test_hermes_client.py` fails 3 tests at last run: `test_protocol_conformance` (`isinstance` against `HermesClient` raises TypeError because the Protocol is not `@runtime_checkable`) and `test_cancellation` (expects a cancel call that the client does not make). Core does not own this file. **r6b: done** — the file passes in the full suite.
- [core -> prosody] `tests/test_style.py` was overwritten by core with chunker-to-style integration tests. Core only relaxed the rate assertion to `> 1.0` after prosody changed `fast`. If prosody owns this file, merge its tests back in. **r6b: closed** — the file passes; no prosody tests are missing.
- [llm-results -> core-final] **r6b: superseded.** Routing is Router v3 (v1.7 below: three classes, threshold 0.5,
  tool threshold 0.6, no regex pre-filter except explicit "ask Hermes" phrasing; eval in
  bench/results/router3_eval_r6.json). Emotion follows item 2: rule-based `gaja/text/emotion.py`, no tags in the
  prompt. The original spec:
  Router v2 + Emotion tags v2 integration spec (based on Router v2 benchmark in docs/LLM_RESULTS.md):
  1. **Routing Architecture (Candidate H_hybrid)**:
     - **Operating Point**: Threshold $\\tau^* = 0.97$ (balanced: Recall 95.0%, Precision 95.0%, False Handoff Rate 5.0% on holdout).
     - **Turn Execution Flow**:
       a) **Step 1: Cheap Regex Rules Pre-filter (0 ms)**: Check user utterance against `CHEAP_RULES` (time, date, weather, email, calendar, reminders, smarthome, server). If matched, trigger `handoff_to_hermes` immediately.
       b) **Step 2: Logprob Classification via `/completion` (median 53.9 ms)**: If rules do not match, send single-token completion request to `http://127.0.0.1:8791/completion` with prompt from `prompts/router_pl.md` (including last 1-2 turns of history). Extract top logprobs, compute normalized $P(\\text{hermes})$. If $P(\\text{hermes}) \\ge 0.97$, trigger `handoff_to_hermes`.
       c) **Step 3: Local Streamed Answer**: If $P(\\text{hermes}) < 0.97$, launch local streaming reply via `/v1/chat/completions` (using `prompts/system_pl.md` v4).
       d) **Step 4: Adapter Safety Net (Fallback Regex)**: If during local generation the model speaks a handoff announcement (matching `PROMISE_PL` or `PROMISE_EN`), synthesize handoff.
     - **Streaming Interaction (Measured)**: Because llama-server on GPU has `total_slots: 1`, requests cannot execute concurrently without queuing. Router v2 MUST run sequentially before local generation starts. Since router latency is only **53.9 ms**, this adds negligible delay to local TTFT while saving 2.5-3.0s of wasted local generation on Hermes queries.
     - **Request JSON Shape**:
       ```json
       {
         "prompt": "<|im_start|>system\\nJesteś klasyfikatorem zapytań dla asystentki Gaja...<|im_end|>\\n{history_turns}<|im_start|>user\\nZapytanie: {utterance}<|im_end|>\\n<|im_start|>assistant\\nKategoria:",
         "n_predict": 1,
         "n_probs": 10,
         "temperature": 0.0,
         "cache_prompt": true
       }
       ```
     - **Fallback Regexes**:
       - PL: `re.compile(r"(przekaż|przekażę|przekazuję|przekazać|przekaże|Hermes)", re.I)`
       - EN: `re.compile(r"(\\bhand(ing)? (it |this )?(off|over)|\\bHermes\\b|\\bpass(ing)? (it |this )?to Hermes)", re.I)`
  2. **Emotion Tags v2 Recommendation**:
     - Do NOT use in-prompt priming or tag instructions in the system prompt (causes style degradation, language mixing, and +40ms TTFT penalty).
     - Recommended: Use rule-based / lexicon post-processor (`gaja/text/style.py`) on completed sentences before TTS (0.01 ms CPU overhead, 0 GPU contention), or leave tags neutral by default.

- [tts-quality -> core-final] **r6b: done** (items 1–5: `[tts.supertonic]`, piper/paradee `apply_post`, voices in
  `models/`, TTS cancelled on barge-in). TTS configuration requests (see docs/TTS_RESULTS.md for the numbers behind them):
  1. **Supertonic config model**: add `[tts.supertonic]` with fields `steps: int = 6`, `voice: str = "M4"`, `speed: float = 1.05`, `threads: int = 2`, `output_sample_rate: int | None = 22050` (`None` = native 44.1 kHz). Add the matching field on `TTSConfig` (`supertonic`). Engine class: `gaja/tts/supertonic_tts.py`, `SupertonicTTS`, registered as `"supertonic"`. Female alternative: `voice = "F1"`.
  2. **`_engine_kwargs` branch in `gaja/server.py`** for `supertonic`: pass `steps`, `voice`, `speed`, `threads`, `output_sample_rate` from the config model. Do not pass `threads` above 2: 4 ORT threads gave no speedup under load on this 4C/8T box.
  3. **Piper style path in `gaja/tts/piper_tts.py`**: in `PiperTTS._render`, build the config as `SynthesisConfig(**engine_params("piper", style))` and finish with `apply_post(audio, sr, style)` instead of `apply_energy`. This is what makes length_scale/noise_scale/noise_w_scale and the pitch/pause post-processing take effect. Do not also pass Piper's `volume` argument: energy is applied once, in `apply_post`.
  4. **Polish voice paths**: default PL voice darkman stays at `models/pl_PL-darkman-medium.onnx`; gosia (quality PL candidate) at `bench/models/piper/pl_PL-gosia-medium.onnx` until the models directory is consolidated. Point config at the same file layout the benchmark uses.
  5. **Barge-in / threads**: the Supertonic engine runs on a single-worker executor; cancel pending synthesis on barge-in as in request 4 of the STT block below.

- [tts-quality -> core-final] **r6b: done** (v1.5 Paradee `apply_post`). Round 2 (numbers and method in docs/TTS_RESULTS.md and docs/PROSODY.md):
  1. **Paradee prosody**: `gaja/tts/paradee_tts.py` `_render` finishes with `apply_energy` only, so `[sad]`, `[calm]`, `[excited]`, `[sigh]` and similar tags change nothing in EN except energy and the rate. Request: finish `_render` with `apply_post(audio, sr, style)` as Piper and Supertonic already do, keeping `speed = rate`. Cost is about 11–15 ms per 3–4 s sentence (docs/PROSODY.md). Until this lands, the EN style demo is energy-only.

- [stt-results -> core-final] **r6b: superseded** by faster-whisper `small` int8 on CUDA (v1.5; stt_ms p50 ~390 ms in
  r6b), with Whistle as the automatic fallback. Items 2 (forced language from `hello.lang`) and 4 (TTS cancelled on
  barge-in) hold. STT runtime and configuration requests (see docs/STT_RESULTS.md):
  1. **Retain Whistle Python binding**: Keep `WhistleSTT` (`cactus-needle`) as the default engine for Polish and English. Do NOT switch to faster-whisper on CPU (small model is ~16x over latency budget at 4.85s median; base model is 7.4x higher memory at 745 MB) or Whistle CLI (subprocess/reload overhead adds 10-60s).
  2. **Language enforcement**: When client supplies `hello.lang = "pl"` or `"en"`, continue passing forced language to `Whistle.transcribe()`. For `hello.lang = "auto"`, pass `language=None` only when explicitly requested by client, but note that auto-detection incurs a severe 2.4x-7.5x latency penalty on CPU (23.78s median on EN). If `hello.lang` is unspecified or invalid, fall back to forced `config.toml [stt] default_language = "pl"`.
  3. **No audio depth pruning**: Ensure Whistle runs with default full audio depth (`--audio-depth 2` degrades Polish WER from 36.1% to 57.8%, depth 4 to 42.1%).
  4. **Barge-in & Contention isolation**: Ensure that when VAD signals `speech_start` (barge-in), active TTS synthesis tasks are aborted immediately. Contention benchmarks revealed that concurrent unconstrained TTS synthesis and Whistle STT on this 4C/8T CPU cause severe thread thrashing and memory bus saturation, degrading STT latency by >40x. [lead: the >40x number is unverified — no saved data; the request to cancel TTS on barge-in and cap TTS threads still stands.]

- [web-final -> core-final] **r6b answers:** 1 kept (`ready.engines` on every `ready`; `engine_by_language` is on
  `/health` only). 2 kept. 3 `assistant_delta` is still **not** sent; keep building the bubble from `utterance`.
  4 `reason:"barge_in"` is sent. 5 the server is strict while its own audio plays (`vad.barge_in_strict_ms` 400,
  `strict_threshold` 0.7, ECHO_TAIL_S 0.4) — no energy comparison. 6 kept. 7(c) fixed: STT is faster-whisper on
  CUDA, stt_ms p50 ~390 ms; a CUDA failure falls back to Whistle and sends `notice{stt_degraded}`.
  Web UI contract notes (nothing blocks the UI; these make it better). The UI is coded against the current server and degrades gracefully without them:
  1. **Voice list**: the UI reads `ready.engines` (today: a list of engine names, e.g. `["piper","piper-darkman","paradee","supertonic"]`) and falls back to `GET /health` -> `engines`. It also accepts objects `{"id","label","note","langs":["pl"]}` per entry, so richer metadata can be added without a UI change. Please keep `ready.engines` in every re-sent `ready`. Optional nice-to-have: `ready.engine_by_language` (same shape as `/health`), so the UI can say which voice "Automatycznie" picks right now.
  2. **Repeated `hello`** (settings change: language / voice) is used instead of reconnecting so the LLM history survives. It works today; please keep it (re-send `ready`, no history reset). An unknown `hello.tts` returns `error` and the session stays closed - the UI recovers by dropping the saved voice and saying `hello` again.
  3. **`assistant_delta`** drives the streaming bubble; each `utterance.text` is matched against the streamed text, so please keep the delta text and the sentence text identical up to whitespace/punctuation (tags stripped). Servers without deltas still work (sentences alone fill the bubble).
  4. **`state: listening` with `reason: "barge_in"`** is optional, the UI flushes on any `listening` that arrives while audio is still queued or playing.
  5. **Echo / self barge-in** is handled client-side (mic gate while Gaja is audible, see docs/WEB_CLIENT.md). If the server could additionally ignore VAD `speech_start` while it is `speaking` and the mic energy is within a few dB of its own output, that would help non-browser clients too - not needed for the web UI.
  6. `audio_start.sample_rate` is trusted per utterance (22.05 / 24 / 44.1 kHz are all resampled in the playback worklet); please keep sending it for every utterance, the `ready.sample_rate` is only the default.
  7. **Observed on the live adapter (8790, 2026-10-08)**: (a) typed turns, `ready.engines` and a repeated `hello` (same `session_id`, `tts` switched to `supertonic`) all work and play correctly in the browser; (b) the server sends no `assistant_delta` (the UI then shows each sentence when its `utterance` arrives, which is fine); (c) **a speech turn completes, but STT is very slow right now**: `metrics.stt_ms` was 23 s (CLI) and 47 s (browser, fake mic playing the 12 s FLEURS clip) while the machine was at load ~11-16 and the pytest run printed `CUDA failed with error out of memory` for faster-whisper - so the STT is probably starved on GPU memory/CPU. The browser handled it correctly (VAD -> thinking -> transcript -> 3 sentences played, 0 underruns), but on a phone that wait would feel broken. Not a client issue.
- [web-final -> core-final] **r6b answers:** 1 fixed — the router is v3 (threshold 0.5) and the safety net only fires
  on a promise phrase. Rechecked live 2026-10-09 (see docs/ADAPTER.md). 2 unchanged (an LLM outage mid-reply still
  sends `error`). 3 fixed differently: barge-in no longer cancels Hermes at all; an explicit cancel sends
  `handoff{state:"cancelled",id}` — see the new request below. 4 unchanged: `progress` is still coalesced text
  every 0.5 s plus one line per tool step; the full answer is in `done.text`. 5 fixed (ACP reader no longer dies;
  a dead worker is respawned on next use).
  Round 2, verified against the live adapter on 8790 (2026-10-09; details in docs/WEB_CLIENT.md §6). Everything in the UI passes: typed turn, mic turn, voice switch, Hermes card, voice barge-in, EN switch, reconnect. Observations that are yours:
  1. **Router/safety-net false positives**: the plain question `Cześć Gaja, opowiedz mi krótko, co potrafisz.` was handed to Hermes twice (`handoff.started`, `text: "safety-net"` once and `"router"` once; the reply text lists "przeszukiwać internet, czytać maile, patrzeć w kalendarz" and trips `PROMISE_PL`, or the router hits >= 0.97). Expected: a normal local answer.
  2. **Mid-stream LLM failure**: one run of the same prompt sent `error: "language model unavailable"` after 4 sentences were already spoken (llama-server restarting/overloaded at that moment); the UI handles it (toast + note), just FYI.
  3. **Silent Hermes cancel**: on barge-in the server cancels the running hand-off but sends no `handoff` event for it (only `state: listening, reason: "barge_in"`). The UI now closes any working Hermes card as "Zadanie przerwane" on that event. If you would rather send `handoff{state:"error"|"cancelled"}`, the UI accepts `error` today. (The `ACP reader died on a cancelled request` wedge you fixed at 08:39 was most likely triggered by exactly this path: my gallery run typed a new turn while a safety-net hand-off was running.)
  4. **Hermes streams its answer as `handoff.progress` token fragments** (`"\n\nPi"`, `"ątek, dziewiątego…"`); the UI now glues fragments into one line. A cleaner contract would be one `progress` per tool step and the answer only in `done.text`; not required.
  5. `session/new timed out after 60.0s` surfaced once as `handoff{error}` ("Błąd Hermes: ACP request session/new timed out after 60.0s") while the wedge above was present; the UI showed it correctly.

- [local-tools -> core-final] **r6b: integrated** (docs/TOOLS.md "Integration"). Differences from the spec: no regex
  pre-filter — Router v3 decides TOOL, then a grammar-constrained `select_tool` call picks the tool (skipped when no
  tool's cue words occur); Variant C direct speak; `normalize_for_tts` runs on every sentence when
  `[text] normalize_tts = true`; `[tools]` has `enabled`, `default_location = "Katowice"`, `timeout_s = 4.0` (2.5 s
  timed out on Wikipedia under load). Local Tools & Text Normalizer (Round 2) integration spec (detailed evaluation and measurements in docs/TOOLS.md):
  1. **Registry & Execution Package**: `gaja.tools` provides `ToolRegistry` with 8 read-only tools (`get_time`, `get_weather`, `calculate`, `convert_units`, `exchange_rate`, `wiki_summary`, `server_status`, `next_holiday`). Call via `await registry.execute(name, args, lang=session_lang)`. Every tool is hard-capped at 2.5s and returns safe `ToolResult` with `data` + Piper-friendly, pre-normalized `speak` text (never raises into pipeline).
  2. **Recommended Pipeline Routing: Variant C (Direct Speak, No Second Pass LLM)**:
     - Routing uses cheap regex pre-filter (0 ms) + 1-token `/completion` Router v3 (`prompts/router_pl.md` without `<|startoftext|>` BOS duplication) classifying into `LOCAL`, `TOOL`, `HERMES`.
     - On `LOCAL`: stream reply via `client.chat_stream` without tools (0% false tool calls, fast TTFT).
     - On `HERMES`: speak handoff ack, trigger `HermesClient`.
     - On `TOOL`: targeted tool selection call with `ToolRegistry.schemas(lang)` -> execute tool -> **speak `tool_res.speak` directly to TTS** (Variant C).
     - **Why Variant C over B_hybrid?**: Variant C saves 1.5–3.0 seconds of latency (median E2E 2347 ms vs 5661 ms), eliminates hallucination completely (Faithfulness 100.0% vs 91.5% in LLM second pass), and guarantees numbers/units/dates are pre-normalized. Second pass LLM generation is optional (`tools.llm_synthesis = false` by default).
  3. **TTS Normalization Hook (`gaja/text/normalize.py`)**:
     - Hook location in `gaja/pipeline.py`: In the sentence playback / synthesis loop, on each sentence chunk before calling `tts.synthesize`, after style/emotion tag parsing (`parse_styled`):
       ```python
       from gaja.text.normalize import normalize_for_tts
       # After extracting style and clean sentence text:
       clean_text = normalize_for_tts(clean_text, lang=session.language)
       audio = await tts.synthesize(clean_text, style, session.language)
       ```
     - Overhead: < 0.6 ms per sentence on CPU. Handles cardinals, decimals, dates ("9 października"), times ("o 17:59" locative, "06:54"), currencies (zł, EUR, USD), units (km/h, kg, GB), Roman numerals ("XVI wieku"), and abbreviations (np., tzn., m.in.).
  4. **Config Model**: Add `[tools]` to `config.toml`: `enabled = true`, `default_location = "Katowice"`, `timeout_s = 2.5`, `max_history_turns = 1`, `direct_speak = true`, `ack_threshold_s = 1.0`.

- [core -> web-final] (r6b, 2026-10-09) Hermes task contract changed in v1.7; the UI needs these changes:
  1. **Stop cancelling the task card on barge-in.** `web/app.js:610` calls `tasks.cancelRunning()` on
     `state{reason:"barge_in"}`, but barge-in no longer cancels Hermes: the task finishes and Gaja announces it.
     Today the card shows "Zadanie przerwane" while the task is still running.
  2. **Key task cards by `handoff.id`** (8 hex chars, stable for one task; several tasks can run at once), not
     by `task` text. Every `handoff` event carries `id`.
  3. **Handle `handoff{state:"cancelled",id,text:""}`**: the task was stopped by `cancel_task`,
     `interrupt{scope:"all"}` or the spoken "anuluj zadanie". No `done` follows.
  4. Optional: a cancel button on a running card that sends `{"type":"cancel_task","id":"<id>"}`. Without `id` it
     cancels all tasks; with no running task the server answers `error` `no running Hermes task`.
  5. **Show `notice` events** (`{"type":"notice","code":"stt_degraded"|"stt_recovered","message":...,"engine":...}`),
     e.g. as a toast; `message` is Polish and ready to display. Sent only on a change.
  6. Optional: render `tool` events (`{"type":"tool","name","arguments","ok","text"}`) as a small chip; the spoken
     answer arrives as normal `utterance`s, so ignoring them is safe.
  7. `metrics` has new fields (`kind`, `route`, `tool`, `tool_select_ms`, `tool_ms`, `spec_ms`,
     `committed_speculative`); `kind:"result"` marks the announcement of a finished Hermes task.
  **web-final r3: done** (2026-10-09). Items 1–7 are in `web/` (cards keyed by `id`, `cancelled`, Przerwij button →
  `cancel_task{id}`, `notice` toast + status pill, `tool` chips, `kind:"result"` metrics ignored, `/health`
  `stt_degraded`/`router.tools` in diagnostics); details in docs/WEB_CLIENT.md §7. Verified on the live adapter (8790):
  tool turn, Hermes task surviving a typed interruption and then completing, cancel via the button (`handoff{cancelled}`,
  no `done`). Server-side observations (nothing fixed on our side): (a) the first live run of "jaka jest pogoda w
  Warszawie" in a fresh session was routed LOCAL and the model said it has no internet access; 5 later tries (UI and raw
  WS) all went to the TOOL route, so it looks like a cold/loaded router, not reproducible; (b) `handoff.started.text` is
  sometimes a routing label (`router`, `safety-net`) and sometimes the model's free-text reason, so the UI shows only
  non-label text; (c) the spoken result of a task carries no `handoff` id, so a client cannot attach the announcement
  utterances to the card.

## Change log

- v1 — initial contract.
- v1.1 — web workstream (web/, clients/) reassigned to a new agent (UI redesign); Gemini web round 2 was stopped mid-way.
- v1.2 (core) — config env naming: `GAJA_<SECTION>__<KEY>` (e.g. `GAJA_SERVER__PORT`, `GAJA_LLM__BASE_URL`) instead of the flat `GAJA_HOST`/`GAJA_PORT` written above. Host/port live in `config.toml [server]`.
- v1.2 (core) — `gaja/metrics.py` added: `TurnMetrics`, `MetricsRecorder` (appends `logs/turns.jsonl`, serves p50/p90 at `GET /metrics`). The `metrics` WS event shape is unchanged.
- v1.2 (core) — `hello.lang` is strict: `pl` | `en` | `auto`; an invalid hello returns an `error` event and the session is not opened.
- v1.2 (core) — barge-in also works during `thinking`, not only `speaking`: VAD `speech_start` cancels the LLM stream and any pending TTS, then `state: listening`.
- v1.2 (core) — `hello.tts` override applies to every language (it bypasses per-language routing in `tts.engine_by_language`).
- v1.2 (core) — new TTS engines: file `gaja/tts/<name>_tts.py` decorated `@register_tts("<name>")`, loaded lazily by `get_tts(name, **cfg)`. Each engine needs a config model field on `TTSConfig` (`piper`, `paradee` today).
- v1.2 (core) — `handoff` event `text` semantics: on `started` it is the model's `reason`; on `progress`/`done`/`error` it is Hermes' text.
- v1.2 (core) — LLM tool path: llama-server runs with `--jinja`, which parses tool calls into `delta.tool_calls`. The default `llm.tool_strategy` is therefore `native` (verified with LFM2.5-2.6B). `pythonic` remains as a fallback.
- v1.2 (core) — reasoning: llama-server with `--reasoning-budget 0` strips `<think>` before it reaches `content`, so the client's think filter starts outside a block. A stray `<think>...</think>` in content is still removed.
- v1.3 (lead) — round 3 ownership (final integration). Lead measured quietly-ish (load ~7): Piper `pl_PL-gosia-medium` RTF ~21, darkman ~11, `bass-high` ~3.8 (the scout numbers for gosia 6.2 / bass 0.86 were load-contaminated). Gosia becomes the default Polish voice; model files move to `models/`.
  | Workstream | Model | Owns |
  |---|---|---|
  | core-final | Haiku 5.5 | `gaja/` except the files below, `config.toml`, `gaja/config.py`, `pyproject.toml` (via uv lock), `.gitignore`, `deploy/`, `scripts/`, `docs/ADAPTER.md`, `docs/SETUP.md` |
  | tts-quality | Haiku 5.5 | `gaja/tts/supertonic_tts.py`, `gaja/tts/mms_tts.py`, `gaja/tts/prosody.py`, `gaja/text/style.py`, `bench/tts_*`, `bench/prosody_demo.py`, `docs/PROSODY.md`, `docs/TTS_RESULTS.md`, tests for these. Config needs -> "Requests" for core-final. |
  | llm-results | Gemini 3.8 Flash | `bench/routing_eval.py`, `bench/llm_*.py`, `bench/data/`, `prompts/`, `docs/LLM_RESULTS.md` (llama-server for experiments on 8792+ only) |
  | stt-results | Gemini 3.8 Flash | `bench/stt_*`, `bench/run_stt_matrix.sh`, `docs/STT_RESULTS.md` |
  | web-final | Sonnet 5.5 | `web/`, `clients/`, `tests/test_web_e2e.py`, `tests/test_cli_client.py`, `docs/WEB_CLIENT.md`, `docs/ui/` |
  | local-tools (v1.5, lead) | Gemini 3.8 Flash | `gaja/tools/` (new package), `prompts/tools_*.md`, `bench/tools_*`, `bench/data/tools_*`, `tests/test_tools*.py`, `docs/TOOLS.md`, and (round 2) `gaja/text/normalize.py` + `tests/test_normalize*.py` + `bench/normalize_*`. Integration into `gaja/pipeline.py` / `gaja/llm/` / `config.toml` only after core-final r3 finishes, coordinated by the lead. |
  The live adapter on 8790 and llama-server on 8791 belong to core-final (it may restart them). Others use `clients/mock_server.py` or their own ports.
- v1.3 (core-final) — `ready` carries `engines` (every selectable TTS name); `/health` carries `engines`, `engines_loaded`, `engine_by_language`, `hermes_backend`.
- v1.3 (core-final) — `hello.tts` accepts any engine in `ready.engines`: `piper`, `piper-darkman`, `paradee`, `supertonic`. Supertonic loads on first selection and is not a default.
- v1.3 (core-final) — `hermes.backend = "acp"` is the default: a persistent `hermes acp` worker, started at adapter startup. `stub` is for tests only.
- v1.3 (core-final) — a Hermes failure emits `handoff{state:"error"}` and speaks `Hermes jest teraz niedostępny, nie udało się wykonać zadania.`; the voice loop continues.
- v1.3 (core-final) — `tts.piper` is the gosia voice (`models/pl_PL-gosia-medium.onnx`); `tts.piper_darkman` is the alternative.
- v1.4 (core) — routing: before the local reply, a one-token logprob router on the same llama-server decides `hermes` vs `local`. It sends `POST /completion` with `n_predict:1, n_probs:10, temperature:0`; P(hermes) is the normalized mass of top tokens starting with `h` vs `l`. Handoff when P >= `router.threshold` (default 0.97). A router outage falls back to the local reply.
- v1.4 (core) — on handoff the adapter speaks one of three fixed acks (`ACKS` in `gaja/pipeline.py`) before calling Hermes. The rotation is per language.
- v1.4 (core) — safety net: when the router did not hand off, a local reply that promises a handoff (regex, PL and EN) is replaced by the handoff path, so the user never hears an unexecuted "przekażę Hermesowi".
- v1.4 (core) — `metrics` gains `router_ms` (router round trip, zero when skipped). `TIMING_FIELDS` is `stt_ms, router_ms, llm_ttft_ms, first_sentence_ms, tts_first_ms, e2e_first_audio_ms`.
- v1.4 (core) — `state` events carry `reason` when a barge-in cancels a reply (`"barge_in"`), including when the state was already `listening`.
- v1.4 (core) — `style.auto_emotion = true` (default): untagged sentences get a rule-based emotion (`gaja/text/emotion.py`). An explicit `[tag]` always wins.
- v1.4 (core) — Hermes progress is coalesced (`hermes.progress_interval_s = 0.5`): `text_delta` is batched; `tool_call` and `permission_cancelled` are sent immediately.
- v1.4 (core) — the Hermes session for a WS connection is forgotten when the connection closes (`HermesClient.forget`).
- v1.4 (core) — STT engine is selectable: `stt.engine = "whistle"` (default) or `"faster-whisper"` (`stt.faster_whisper.*`, scaffold). `hello.lang` forces the transcription language.
- v1.4 (core) — supersedes the v1.3 "supertonic loads on first selection" note: every TTS engine, supertonic included, is built at startup. A failing optional engine is logged and does not stop startup; the default engines still do. Piper/darkman ORT sessions are capped by `threads` (default 2). `chunker.first_chunk_min_words = 3`.
- v1.4 (core) — the llama-server unit uses `--chat-template-file deploy/lfm_nothink.jinja` (copy of `bench/templates/lfm_nothink.jinja`) and `-np 2 --kv-unified` (router and chat prompts keep separate KV slots). Native tool calls still parse; the handoff tool is unchanged.
- v1.4 (core) — the adapter unit sets `HF_HUB_OFFLINE=1`; models load from local files only.
- v1.4 (core) — barge-in also cancels a running Hermes handoff: the Hermes task is cancelled, `session/cancel` is sent, and its result is not spoken. The `state: listening` with `reason: "barge_in"` is sent once, after cancellation completes.
- v1.5 (core) — STT default is faster-whisper on CUDA (`small`, `int8`, `beam_size = 1`, language forced from `hello.lang`). `stt.engine = "whistle"` stays selectable. `gaja/stt/select.py` (`SwitchingSTT`) falls back to Whistle automatically when faster-whisper fails at load or in a decode (for example CUDA OOM); the failed utterance is retried on Whistle. The adapter never crashes on this path.
- v1.5 (core) — `/health` adds `stt_requested`, `stt_device` (`cuda`|`cpu`) and `stt_fallback` (`null`, or `"<load|inference>: <error>"`). `stt_engine` is the engine serving now.
- v1.5 (core) — Hermes ACP client: a request cancelled from outside (barge-in) no longer kills the stdout reader. Pending entries are removed on every exit, and a late reply for a done future is ignored. Before this, all later handoffs timed out in `session/new` until restart.
- v1.5 (core) — llama-server error chunks in the streaming chat path raise `LLMUnavailable` with the server message, not a `KeyError`. The pipeline speaks its normal LLM-error path.
- v1.5 (core) — `bench/e2e.sh` is the single documented e2e command (`scripts/e2e_suite.py` under `/tmp/gaja-bench.lock`, load average recorded before and after). Its default turns are PL audio x10, EN audio x4, typed PL and EN chit-chat, barge-in x2, one Hermes handoff, and routing x8 (4 hand-off, 4 local including the "Kim jest Hermes w mitologii?" trap).
- v1.5 (core) — the router prompt does not start with a literal `<|startoftext|>`; llama-server adds BOS itself, and the literal produced a double BOS. Router probabilities shift (e.g. "Ile to dwa plus dwa?" p_hermes 0.14 → 0.001), so routing is re-checked by the e2e routing scenario.
- v1.5 (core) — the router system prefix is primed on `speech_start` (`Router.prime`, `n_predict 0`, `cache_prompt`). The router call then reuses the KV prefix: in a probe, `cache_n` 455 of ~480 tokens and wall time ~105 ms, against ~850 ms when the prefix is cold. Without priming, only an exact repeat of the prompt reused the cache.
- v1.5 (core) — Paradee renders through `apply_post` (the same prosody post-processing as piper), `speed` still from `style.rate`.
- v1.6 (core-final, round 4 latency) — llama-server runs `-c 8192 -np 2 --no-kv-unified` (per slot 4096), replacing `-c 2048 -np 2 --kv-unified`. The shared unified cache could not restore an evicted slot (`failed to find 1708 available cells`) and re-prefilled ~770 tokens. VRAM: 3127 MiB before, 3217 MiB idle and 3225 MiB under load after.
- v1.6 (core-final) — `LLMConfig.router_slot = 1` and `chat_slot = 0`. The router and chat requests pin `id_slot` (`/completion` and `/v1/chat/completions`). `-1` leaves the choice to the server.
- v1.6 (core-final) — `Router.prime` sends the router system prefix on `speech_start` to `router_slot`. The chat prefix is no longer primed: a prime overwrote the slot's cached conversation and raised chat TTFT p50 from ~175 ms to 520 ms or more.
- v1.6 (core-final) — `[chunker] first_chunk_max_words = 5` (0 = off). The first sentence is cut after this many words when no punctuation has come yet. Applies only to the first chunk of a reply.
- v1.6 (core-final) — `tests/test_faster_whisper_stt.py` skips when CUDA memory is held by the live stack (`out of memory` on warm-up). The full suite runs green on the live machine.
- v1.6 (core-final) — `metrics` and `TIMING_FIELDS` are unchanged. Measured round-4 e2e (17 turns): `e2e_first_audio_ms` p50 1894 → 1583 ms, p90 3144 → 2300 ms.
- v1.7 (core, rounds 6/6b, 2026-10-09) — **Router v3**: one 1-token logprob call classifies LOCAL / TOOL / HERMES
  (`router.threshold = 0.5` for HERMES, `router.tool_threshold = 0.6` for TOOL, `history_turns = 2`). Explicit
  addressing ("zapytaj Hermesa…", "Hermes, …", "w Hermesie", "przez Hermesa", "ask Hermes") hands off without the
  model; mythology/brand mentions never do. Eval (bench/results/router3_eval_r6.json): Hermes recall 92.9 %,
  precision 95.8 %, false handoff 2.3 %, holdout 100/100; classify p50 116.5 ms. Supersedes the v1.4 threshold 0.97.
- v1.7 (core) — **TOOL route**: a grammar-constrained `select_tool` call on the router slot picks one of the
  `gaja.tools` tools; the tool's `speak` text is spoken directly (Variant C) and a `tool` event is sent. The select
  call is skipped when none of the tools' cue words occur in the utterance or recent history; no tool picked gives
  a local reply. A failed tool speaks its own error sentence (`tool.ok = false`).
- v1.7 (core) — **Priming**: the router slot is primed with system prompt + recent history (not just the system
  prompt) after every turn and on `speech_start`; the classify call then evaluates ~30 tokens (router_ms p50
  312–319 → 117–129 ms). With `llm.prime_after_turn = true` (default) the chat slot is also primed with the whole
  conversation after each turn, so the next request prefills only the new user message (llm_ttft p50 ~350 →
  ~190 ms). Supersedes the v1.6 "chat prefix is no longer primed" note: the v1.6 prime replaced the slot with the
  bare system prompt; this one extends it. Both primes are off the critical path.
- v1.7 (core) — **Speculative endpointing on**: `vad.spec_silence_ms = 250`. STT, router and the reply start after
  250 ms of silence; output, history writes, tool execution, Hermes start and voice cancel wait for the
  `min_silence_ms` (500) confirmation. Speech that resumes discards the speculation with no side effects. Audio wall
  p50 1870/1920 → 1584/1615 ms (bench/out/e2e/r6b_spec0{,_b}.json vs r6b_spec250{,_b}.json).
- v1.7 (core) — **Hermes tasks**: barge-in no longer cancels Hermes (supersedes v1.4). Tasks run in the background,
  several at once, each with an `id`; results are announced when Gaja is idle and the user is not speaking. Cancel
  via `cancel_task{id?}`, `interrupt{scope:"all"}` or a spoken cancel ("anuluj zadanie", "przerwij to", "Hermes,
  stop", …, checked only while a task runs); the client gets `handoff{state:"cancelled",id}` and the ACP worker
  `session/cancel`. Hermes has no session/close (acp_adapter 0.19.0), so sessions are only forgotten.
- v1.7 (core) — the ACP stdout reader has a 16 MiB line limit (large tool outputs killed it at 64 KiB); a dead
  reader respawns the worker on next use.
- v1.7 (core) — STT: on a faster-whisper failure the utterance is retried on Whistle and the preferred engine gets
  a cooldown (60 s doubling to 600 s) with a background re-probe; `notice{stt_degraded|stt_recovered}` is sent and
  `/health` has `stt_degraded`. `/health.router` adds `tools`.
- v1.7 (core) — VAD: barge-in needs `vad.barge_in_ms = 250` of speech, `barge_in_strict_ms = 400` at
  `strict_threshold = 0.7` while Gaja's own audio is playing (echo). `max_utterance_s = 60` force-ends a very long
  utterance.
- v1.7 (core) — WS `Origin` allowlist (`server.allowed_origins`), handshake refused with 403. Static files are served
  from `web/` only (path traversal returns 404).
- v1.7 (core) — TTS engines load lazily (first use) except the defaults per language; `/health.engines_loaded`
  lists the loaded ones. The adapter unit waits up to 60 s for llama-server health (`ExecStartPre`).
- v1.7 (core) — `[text] normalize_tts = true`: `gaja/text/normalize.py` runs on every sentence before TTS. A leading
  "Gaja:" speaker label from the model is stripped from the reply stream.
- v1.7 (core) — `metrics` gains `kind`, `route`, `tool`, `tool_select_ms`, `tool_ms`, `spec_ms`,
  `committed_speculative`. On a committed speculation `first_sentence_ms` and `e2e_first_audio_ms` are measured
  from the commit (the confirmed end of speech); `stt_ms`, `router_ms` and `llm_ttft_ms` stay raw durations.
- v1.8 (web/server) — Wake-word sample recorder: `web/record.html` + `gaja/wakeword_samples.py` (`/api/wakeword/{samples,progress,sessions,export.zip}`,
  Origin-allowlisted like `/ws`), config `[wakeword] recordings_dir = "data/wakeword/recordings"`, `max_total_mb`; `data/` git-ignored. No change to
  the WebSocket protocol. See docs/WEB_CLIENT.md §9.
- v1.8 (device) — **Device protocol**: client kind `device` on the optional `WS /device` listener (`[devices]`, off by default, port 8792,
  per-device tokens outside the repo, `hmac.compare_digest`, network allow-list, optional TLS). Hello `caps` {aec, out_rates, preroll_ms}
  and `fw`; the server negotiates the output rate and resamples TTS. New messages: `wake` (c->s), `wake_accepted`, `wake_rejected`,
  `listen_end`, `follow_up` (s->c), `transcript.heard`, `ready.device`, `metrics.route:"wake"`. Server-side wake verification
  (`gaja/devices/wake.py`), bare-phrase ack, follow-up window, wake barge-in. `/health.devices`, `/metrics.counters`. Web/cli unchanged.
  Simulator `clients/device_sim.py`, docs/DEVICE.md, firmware plan and untested ESPHome config in `firmware/`.
- v1.9 (tools/timers, 2026-10-12) — **Local timers, reminders, and offline utilities**: default location changed to Katowice across config, tools, docs, and tests. Offline pure astronomical sunrise/sunset and day length (`get_sun`), date arithmetic in `get_time` (target_date, days until date, day of week), holiday query by name in `next_holiday`, cooking units (szklanki, łyżki, łyżeczki, cup, tbsp, tsp) and data sizes (bajt, kB, MB, GB, TB) in `convert_units`. Persistent local timers and reminders in `data/timers.json` (`manage_timer`), past-due fired once on startup with a note, WS broadcast event `{"type":"timer","id":"...","label":"...","kind":"timer|reminder"}` to all connected sessions (web, CLI, device) plus spoken Polish announcement for sessions with audio. REST API `GET /api/timers` and `DELETE /api/timers/{id}`. Persistent light fact memory in `data/memory.json` (`remember`, `forget`), injected compactly at the end of system prompt (capped at 600 chars), with REST API `GET /api/memory`, `POST /api/memory`, `DELETE /api/memory/{id}`. Rewritten concise natural persona in `prompts/system_pl.md`.
