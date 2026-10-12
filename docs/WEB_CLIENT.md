# Gaja Web Client & CLI Tools

Documentation for the browser interface and CLI client for the Gaja voice adapter.

---

## 1. Web interface (`web/`)

Single-page application (`index.html`, `app.js`, `style.css`), served by FastAPI under `/` (static files from `web/`). Works on desktop browsers and mobile devices (tested on Chrome, Firefox, Safari on iOS, and Android).

### What is on the screen

- **Orb** (central sphere): canvas with WebGL fragment shader (`orb.js`), visualising the assistant state:
  - Blue, calm (`idle` / `ready`): waiting for speech.
  - Golden-yellow, pulsating (`listening`): user is speaking (VAD detected speech).
  - Magenta, spinning (`thinking`): processing (STT / routing / LLM / tools).
  - Blue, wavy (`speaking`): assistant speech playback (amplitude-driven).
  - Crimson pulse (`barge_in`): assistant speech interrupted by user speech.
  - Green pulse (`wake`): wake word detected (on devices supporting wake word).
  - Satellite orbiting the orb: visualises Hermes background task status (orbiting while working, flash on completion).
- **Conversation history**: list of messages (user query + assistant response), with latency chips and tool cards.
- **Latency bar**: real-time breakdown of turn timings (STT, LLM TTFT, TTS TTFA, total latency).
- **Status pill**: connection state (`connected`, `reconnecting`, `offline`, `degraded STT`).
- **Settings drawer** (gear icon):
  - Audio input device selector (`AudioController.openMicStream`).
  - Voice selection (Gosia / Darkman for Piper PL, Paradee for EN).
  - Playback speed slider (0.8x – 1.4x).
  - Pitch shift toggle.
  - Language toggle (PL / EN).
  - Dark / Light theme toggle.
  - Link to wake word recorder (`/record.html`).

### Desktop shortcuts

- `Space` (hold): Push-to-talk.
- `Escape`: Cancel current turn / interrupt playback.
- `Enter` (in text input): Send text turn without microphone.

### Audio pipeline — how it works

1. **Microphone capture**: `navigator.mediaDevices.getUserMedia({ audio: ... })` captured via AudioWorklet (`web/capture-processor.js`), downsampled to 16 kHz 16-bit mono PCM.
2. **WebSocket streaming**: Audio chunks streamed over `WS /ws` as binary frames.
3. **Playback**: Assistant speech arrives as binary WAV chunks, decoded through Web Audio API (`AudioContext`), played seamlessly via queued `AudioBufferSourceNode` buffers.
4. **Barge-in**: When the server detects user speech during assistant playback, it sends `state{listening, reason:"barge_in"}`. The client immediately stops queued audio buffers and clears the playback pipeline.

### Technical details

- **No external framework**: Vanilla ES2022 JavaScript, Web Components / clean DOM manipulation, zero build step.
- **PWA support**: `manifest.json` and service worker (`sw.js`) allow installing Gaja as a Progressive Web App on mobile and desktop.
- **AudioWorklet**: Audio capture runs on a dedicated audio rendering thread for low jitter and glitch-free streaming.

---

## 2. Running and access

### Locally

Open `http://localhost:8790` (or `http://127.0.0.1:8790`) in your browser.

### Mobile access via Tailscale (HTTPS)

Browsers require HTTPS (or `localhost`) to grant microphone access. To access Gaja securely from a phone or other device:

1. Configure Tailscale Serve on the host:
   ```bash
   tailscale serve --bg --https=8443 http://127.0.0.1:8790
   ```
2. Add your Tailnet domain to `config.local.toml`:
   ```toml
   [server]
   allowed_origins = [
     "http://localhost:8790", "https://localhost:8790",
     "http://127.0.0.1:8790", "https://127.0.0.1:8790",
     "https://<host-name>.<tailnet-name>.ts.net:8443",
   ]
   ```
3. Open `https://<host-name>.<tailnet-name>.ts.net:8443` on your mobile browser.

### Troubleshooting

| Issue | Cause & Solution |
|---|---|
| "Microphone requires secure connection" | Accessing over plain HTTP on a non-localhost IP. Use HTTPS (e.g. Tailscale Serve) or `localhost`. |
| Microphone permission denied | Check browser permission settings (Safari Settings -> Microphone; Android padlock icon -> Permissions). |
| "No signal detected from microphone" | Audio level below threshold for 5 s. Check input device selector in settings and system input volume. |

---

## 3. CLI Client (`clients/cli_wav.py`)

A standalone CLI tool to send pre-recorded WAV audio files or interactive microphone audio to the adapter over WebSocket.

### Syntax:

```bash
uv run python clients/cli_wav.py [options] [audio.wav]
```

### Options:

- `--url URL`: WebSocket URL (default: `ws://127.0.0.1:8790/ws`).
- `--out PATH`: Save received TTS audio to WAV file.
- `--text TEXT`: Send text query directly instead of audio.
- `--mic`: Record from microphone until Enter is pressed.
- `--session ID`: Explicit session ID.
- `--metrics`: Print detailed latency breakdown.

### Examples:

```bash
# Send a WAV file and play reply
uv run python clients/cli_wav.py bench/data/stt/01_pogoda_krakow.wav

# Send a text query
uv run python clients/cli_wav.py --text "Która jest godzina?"

# Save reply audio
uv run python clients/cli_wav.py --text "Opowiedz żart" --out reply.wav
```

---

## 4. Development test server (`clients/mock_server.py`)

A lightweight FastAPI mock server simulating adapter WebSocket events without loading heavy AI models (useful for UI development and automated E2E tests).

```bash
uv run python clients/mock_server.py --port 8799
```

Supports simulating barge-in, long Hermes jobs, tool execution, degraded STT notifications, and custom latencies.

---

## 5. Automated tests and screenshots

```bash
# E2E UI tests (Chromium with fake microphone playing Polish audio from bench/data/stt/)
flock /tmp/gaja-e2e.lock uv run pytest -q tests/test_web_e2e.py

# Capture gallery screenshots of all UI states into docs/ui/ (mock on 8799; --base points to another server)
uv run python clients/mock_server.py --port 8799 &
uv run python clients/capture_ui_states.py --base http://127.0.0.1:8799
```

---

## 6. Real adapter verification (2026-10-09)

Verified against the live adapter on `:8790`:

- Text queries, voice turns, voice switching, tool execution, Hermes handoff, barge-in, language switching, and reconnection handling.
- Saved UI screenshots in `docs/ui/real-*.png`.

### Adapter troubleshooting (visible in UI)

| Symptom | Cause |
|---|---|
| Toast "language model unavailable" during response | llama-server stopped or restarting; adapter responds "Model językowy jest chwilowo niedostępny". Try again shortly. |
| Card "Hermes: error · ACP request … timed out" | Hermes (ACP) unresponsive; verify health on `/health` -> `hermes`. |
| First turn after restart takes longer | STT / TTS models warming up. |

---

## 7. Protocol v1.7 in UI (web-final, round 3, 2026-10-09)

Adapted to core protocol changes (round 6b, see "Requests" in `docs/ARCHITECTURE.md`):

| Event / Field | UI Action |
|---|---|
| `state{listening, reason:"barge_in"}` | Clears playback and marks bubble as "interrupted". Does **not** touch Hermes cards. |
| `handoff{started/progress/done/error, id}` | Cards tracked by `id` (supports multiple concurrent). Fallback key = task text without `id`. |
| `handoff{state:"cancelled", id, text:""}` | Card shows "Task cancelled", conversation records "Task cancelled: …", without `done`. |
| `cancel_task{id}` (**Cancel** button) | Sent from card; button displays "Cancelling…" until `cancelled`. Error `error: no running Hermes task …` unlocks button with informative toast. |
| `notice{code, message, engine}` | Toast + badge "Speech: degraded mode" (`stt_degraded`) / toast "restored to normal" (`stt_recovered`). Labels from `i18n.js` (PL/EN). |
| `tool{name, arguments, ok, text}` | Badge in conversation; spoken text arrives as standard `utterance`. |
| `metrics{kind:"result"}` | Excluded from latency bar calculations. |
| `assistant_delta` | Deprecated on server; legacy client support retained (mock: `?deltas=1`). |
| `GET /health` | `engines` (voices list), `stt_degraded` (badge), `stt_engine`/`stt_device`/`router.tools` (diagnostics in settings). |

### Real adapter verification (8790), round 3

`clients/real_adapter_check.py --only i,j,k` (read-only queries; Hermes receives only date/time requests):

| # | Check | Result |
|---|---|---|
| i | "jaka jest pogoda w Warszawie" -> event `tool` (`get_weather`, `location: Warszawa`), badge "🌤 pogoda · Warszawa" between query and response, spoken answer | OK (3 of 4 runs in UI; first routed locally — see routing note in "Requests" in `docs/ARCHITECTURE.md`) |
| j | "Zapytaj Hermesa jaki jest dziś dzień tygodnia" + typed turn ("dowcip o kocie") while running | OK: card remained `working` (1.5 s after typing), no `cancel_task`/`interrupt`/`cancelled`, events `started, progressx4, done`, single `id`, result "Dziś jest piątek, dziewiątego października." |
| k | Second task ("…dzień miesiąca") cancelled with button | OK: sent `cancel_task{id}`, server answered `handoff{cancelled}` with matching `id`, card "Task cancelled", no `done` or error over 9 s |

Screenshots: `docs/ui/real-tool-chip.png`, `real-tool-chip-start.png`, `real-settings-diag.png`, `real-hermes-running.png`, `real-hermes-running-typed.png`, `real-hermes-done-v17.png`, `real-hermes-cancel-button.png`, `real-hermes-cancelled.png`.

---

## 8. Microphone selector (lead, 2026-10-10)

Settings -> "Input device": the first option, "Default system", stores `mic: '\` and opens the microphone with no `deviceId` (OS default). The list is populated from `enumerateDevices()` (Chrome's `default` / `communications` pseudo-devices are filtered out; labels appear after permission is granted). Choice is saved in `localStorage['gaja.settings'].mic`. `AudioController.openMicStream` attempts the chosen device, then the default with AEC/NS/AGC, then fallback `{audio:true}`; if the saved device is disconnected, it reverts to default with a toast notice. Switching devices while the mic is active reopens the stream cleanly. The device list refreshes on `devicechange`. Orb: shader output fades across the outer 14% of the canvas so the satellite halo does not clip at the border; satellite orbit is 1.32 * R0.

### 8.1 "No signal" watchdog (lead, 2026-10-10)

While the microphone is open (`armed` or `live`), `app.js` samples raw input RMS every 250 ms (`audio.getMicRms()`, un-gated by push-to-talk). If RMS remains <= `0.0003` (digital silence — a real room is typically ~1e-3 and up) for 5 seconds, a `nosignal` notice alerts the user that no audio is reaching the app, identifies the active device, and provides a "Select microphone" button (opens settings and focuses `#set-mic`). It never overwrites another notice, triggers once per mic-open/device-switch, and dismisses automatically as soon as audio signal is detected.

---

## 9. Wake-word sample recorder (2026-10-10)

`/record.html` (linked from the bottom of settings: "Record wake-word samples") records real "Hej Gaja" training and evaluation samples for the custom wake word (microWakeWord pipeline: 16 kHz mono 16-bit WAV). Files: `web/record.html`, `web/record.js`, prompt list in `web/record-script.js` (one `SECTIONS` array: id, text, instructions, target count; IDs must remain stable once recorded as they define directory structure), styles `.rec-*` in `style.css`. Screenshots: `docs/ui/record.png` (desktop dark), `docs/ui/record-phone-light.png`.

- **Script**: Sections *Positives* (`wake`), *Hard negatives* (`hard_negative`, x3 each), and *Ambient background* (`ambient`: 6 takes up to 10 min, auto-uploaded in 60 s chunks; ambient prompt counts as complete after 5 min). Per-prompt and overall progress are tracked on the server (`GET /api/wakeword/progress`), resuming sessions on page reload; "Session" allows creating new sessions or resuming existing ones.
- **Recording**: Raw microphone: `AudioController.openMicStream(deviceId, { raw: true })` disables AEC, noise suppression, and AGC (`AEC/NS/AGC off`). Same device selector and settings as main app (`gaja.settings.mic`). Audio passes through `capture-processor.js` (16 kHz PCM16) and encodes to WAV in the browser. 100 ms is trimmed from both ends of each take to eliminate key clicks. Takes auto-stop after 12 s.
- **Controls**: Space = start/stop, Enter = Save, Backspace or R = Retry. After stopping, the take is played back (toggle "Playback after recording") with waveform visualization; warnings for silence (peak < -40 dBFS), clipping (peak >= 0.99), or duration < 0.8 s. "Undo last save" deletes the latest saved take on the server.
- **Server**: `gaja/wakeword_samples.py` (FastAPI router included in `create_app`):
  - `POST /api/wakeword/samples?session&label&prompt_id&take[&mic]` (raw `audio/wav` body, PCM16 mono 16 kHz, <= 4 MB; HTTP 409 if take exists).
  - `GET /api/wakeword/progress?session=`
  - `GET /api/wakeword/sessions`
  - `DELETE /api/wakeword/samples/<session>/<label>/<prompt_id>/<take>`
  - `GET /api/wakeword/export.zip[?session=]`
  Files stored under `data/wakeword/recordings/<session>/<label>/<prompt_id>/<take:03d>.wav` plus `<session>/meta.jsonl` (timestamp, label, prompt_id, take, duration_s, peak, rms, user agent, mic label; undo appends a `{"deleted": true}` record). Path components validated against `[a-z0-9_-]`. Endpoints validate `Origin` against `server.allowed_origins`. Configured via `[wakeword]` (`recordings_dir`, `max_total_mb`). `data/` is git-ignored.
- **Exporting**: Download zip from `/api/wakeword/export.zip` (or "Download archive" button in UI). Directory structure matches training labels (`wake`, `hard_negative`, `ambient`).
- **Tests**: `tests/test_wakeword_samples.py` (TestClient). Manual testing with fake mic: Chromium `--use-fake-ui-for-media-stream --use-fake-device-for-media-stream [--use-file-for-fake-audio-capture=x.wav]`.
