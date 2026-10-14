# Gaja Web Client & CLI Tools

Documentation for the browser interface and the CLI client of the Gaja voice adapter.

---

## 1. Web interface (`web/`)

A static application with no build step (HTML + ES Modules + CSS + two AudioWorklets), served by the adapter at `GET /`.
The main device is a phone (iPhone / Android via Tailscale), but it looks just as good on a computer: a wide screen
gets a two-column layout (orb on the left, conversation on the right).

Screenshots of every state (phone 390×844 @2x, desktop 1440×900) are in `docs/ui/`, and are reproduced by
`clients/ui_screenshots.py` (see §5).

### What is on the screen

| Element | What it does |
|---|---|
| **Orb** (WebGL, with a fallback Canvas 2D) | The only "living" element. It changes color and motion depending on state (listening / hearing you / thinking / speaking / no connection / error), pulses with the real audio level (microphone or speaker), and the emotion of the sentence Gaja is speaking subtly tints the whole orb. When Hermes is working, a small satellite detaches from the orb. Tapping the orb while it is speaking interrupts Gaja. |
| **Status bar** | One short line: „Gotowa”, „Słucham…”, „Słyszę Cię”, „Myślę…”, „Hermes pracuje w tle”, „Mówię · radośnie”… |
| **Large microphone button** | One control for everything: a tap turns the microphone on/off (*hands-free* mode), holding = talk (*push-to-talk*). When Gaja is speaking or thinking, it turns into **Stop** (interrupts). The ring around the button pulses with the microphone level. |
| **Conversation** | Bubbles: you on the right, Gaja on the left. Gaja's reply is built from sentences (`utterance`; since v1.7 the server does not send `assistant_delta`, but the UI still tolerates it from older servers), and each sentence has a state: *queued → speaking → spoken*. The sentence that is currently playing is highlighted in the color of its emotion (a dot next to the sentence shows the emotion), so the text keeps up with the voice, not the network. An interrupted utterance gets an „przerwane” ("interrupted") marker. When you speak, an equalizer bubble appears. When Gaja uses a local tool (`tool` event), a small **chip** appears between the question and the answer, e.g. „🌤 pogoda · Warszawa” (time, weather, calculation, unit conversion, currency rate, Wikipedia, server, next holiday); a failed tool has a warning color and the note „nie udało się” ("failed"). |
| **Hermes card** | One card per task (key = `handoff.id`; several tasks at once means several cards), pinned above the buttons: „Hermes pracuje · 12 s” ("Hermes is working · 12 s") + the last progress line + an activity bar. When tapped it expands into a timeline (dispatched / progress / result / error / cancelled). A running card has a **Przerwij** ("Cancel") button (on a narrow screen just the ■ icon with a label for screen readers), which sends `{"type":"cancel_task","id":"…"}`; the card changes to „Zadanie przerwane” ("Task cancelled") only when the server replies `handoff{state:"cancelled"}` (the same after a spoken "anuluj zadanie" ("cancel task") or `interrupt{scope:"all"}`; after `cancelled` no `done` arrives). **Interrupting Gaja (by voice, text, or the Stop button) does not cancel tasks** – Hermes keeps working, and Gaja announces the result as a separate turn. The card stays until you close it (it disappears by itself after 90 s; „przerwane” after 12 s). An error expands automatically. The trace is also in the conversation. The streamed Hermes response (text fragments) is joined into a single line. |
| **Server notifications** | The `notice` event (`stt_degraded` / `stt_recovered`): a discreet toast + a persistent pill „Mowa: tryb awaryjny” ("Speech: fallback mode") in the top bar, for as long as speech recognition runs on the backup engine (the pill has `role="status"`; the info toast is also read politely, not as an alert). `GET /health` (`stt_degraded`) also reports this state, so the pill appears even when the session started already in fallback mode. |
| **Text field** | A keyboard button (phone) or always visible (desktop). Text is sent as `{"type":"text"}` and **is spoken aloud, even if the microphone was never turned on**. On an empty screen, three suggestions also invite you to start. |
| **Settings** (gear icon) | Conversation language (pl / en / auto – together with the interface language), **voice / TTS engine** (list from the server), microphone mode (hands-free / hold), voice interruption, appearance (system / light / dark), show metrics, diagnostics. |
| **Metrics** (optional) | A thin bar: time to sound (e2e), STT, LLM TTFT, 1st sentence, TTS – last value + p50. Style parameters (rate, pitch, energy) also appear next to sentences. Hidden by default. |

The interface is in Polish; once `en` is selected (or `auto` in an English browser), all of it is in English
(`web/i18n.js` – a dictionary; missing English keys fall back to Polish).
The dark and light themes follow the system (`prefers-color-scheme`); you can switch to a fixed one in settings.
Animations respect `prefers-reduced-motion`.

### Shortcuts (desktop)

`Space` – hold and talk · `/` – text field · `Esc` – interrupt Gaja / close the panel.

### Audio – how it works (and why)

- **Unlocking audio.** `AudioContext` is created and resumed *synchronously* on the first user gesture
  (tapping anything, sending text), together with a silent sample – this is the only way iOS Safari does it. It does not depend on
  the microphone, so replies to text are audible without turning on the microphone. After „interrupted” (a call, Siri)
  the next gesture resumes the audio. On iOS 17+ we set `navigator.audioSession.type`: `playback` (plays despite the muted
  switch) and `play-and-record` when the microphone is on.
- **Sample rate.** Each utterance carries `audio_start.sample_rate` (Piper 22.05 kHz, Paradee 24 kHz,
  Supertonic 44.1 kHz…). The playback worklet (`playback-processor.js`) resamples it linearly and continuously, separately for each
  utterance, to the context's sample rate (ring buffer, 80 ms pre-buffer, no gaps between sentences).
  `ready.sample_rate` is only a default value.
- **Capture.** `getUserMedia` with `echoCancellation`, `noiseSuppression`, `autoGainControl`; the worklet
  (`capture-processor.js`) mixes down to mono, filters (low-pass at 7 kHz) and converts to **16 kHz PCM16**
  (iOS ignores the requested context sample rate and runs at 44.1/48 kHz), sending frames every 40 ms.
- **Interruption (barge-in).** The server sends `state: listening` or VAD `speech_start` → the client **immediately** clears
  the player buffer and discards late frames (`droppedStale`). Tapping the Stop button / orb / Esc / typing
  text does the same and sends `{"type":"interrupt"}` (it also works while Gaja is still „thinking”).
- **Echo and self-interruption (hands-free mode).** With a speaker, the microphone hears Gaja. The browser's AEC removes most of it,
  but the residue (≈ −20…−30 dB) can be loud enough for the server VAD to treat it as the user and cut off Gaja's own utterance.
  Therefore, **as long as Gaja is audible** (playback, pending sentences, +250 ms), the client sends instead of
  the microphone *silence* (zero frames – the VAD stream stays continuous), unless the microphone is clearly louder
  than what we are playing ourselves: ≥ 3 consecutive 40 ms frames above `max(0.03, 0.22 × RMS of output)`. Then the gate
  opens (with a ~320 ms "pre-run", so the first syllable is not cut off) and stays open for 700 ms after the last loud frame –
  the server sees real speech and does a normal barge-in. It works flawlessly with headphones.
  The **„Przerywanie głosem”** ("Voice interruption") switch disables this completely (*half-duplex* mode: while Gaja speaks, the microphone is
  always "deaf", and you can only interrupt by tapping) – choose it if Gaja still hears herself.
  In *hold* mode there is no gate (pressing the button already interrupted playback).
- **Screen.** In hands-free mode (microphone on) the page requests a `wakeLock` so the phone does not turn off.

### Technical notes

- Changing the language or voice in settings sends `hello` again in the same session (the LLM history stays); if the
  server does not answer `ready` within 2.5 s – a full reconnect.
- The voice list comes from `ready.engines` (or `GET /health` → `engines`); entries may be strings or objects
  `{id,label,note,langs}`. Without a list, the UI shows the basic engines (`piper`, `paradee`) and says so.
  A remembered voice that the server no longer knows is silently replaced with "Automatic".
- Settings are kept in `localStorage['gaja.settings']`. URL parameters: `?metrics=1`, `?lang=en`, `?theme=light`.
- Connection: exponential backoff (500 ms × 1.7ⁿ, max 10 s, ±15 % jitter), a „Teraz” ("Now") button, a retry when the
  tab / network comes back (`visibilitychange`, `online`). On disconnect, running Hermes tasks get the error
  „Połączenie zostało przerwane” ("The connection was interrupted") (a new session does not know the old `id`s).
- Diagnostics in settings also show `stt: <engine>/<device>` and `tools on/off` from `GET /health`
  (`stt_engine`, `stt_device`, `stt_degraded`, `router.tools`).
- Metrics with `kind: "result"` (announcement of the Hermes result) do not go into the latency bar – it is not an answer to a question.
- The server error `no running Hermes task …` (Przerwij was clicked just as the task had finished) becomes an
  info toast „To zadanie już się zakończyło” ("That task has already finished"), without a red error in the conversation.
- PWA: `manifest.webmanifest` + icons (`web/icons/`, `icon.svg`), no service worker (the app needs the server anyway).

---

## 2. Running and access

### Locally
```
http://localhost:8790
```
(or `http://127.0.0.1:8790`). `localhost` is a secure context, so the microphone works.

### From a phone via Tailscale (HTTPS)

The microphone (`getUserMedia`) only works in a secure context, so HTTPS is needed from a phone.
The adapter listens only on `127.0.0.1:8790`; HTTPS is provided by `tailscale serve` (exposed by the lead / server owner –
the client does not start anything itself).
1. On the server (once):
   ```bash
   tailscale serve --bg --https=8443 http://127.0.0.1:8790
   ```
   (`tailscale serve status` shows the address; to turn it off: `tailscale serve --https=8443 off`).
2. On the phone (with Tailscale on, in the same tailnet) open:
   ```
   https://<nazwa-hosta>.ts.net:8443
   ```
   e.g. `https://<host>.<twoj-tailnet>.ts.net:8443`. You will see the full name in `tailscale status` / `tailscale serve status`.
3. Tap the microphone and allow access.
4. **Add to the home screen**: iPhone – Safari → Share → „Do ekranu początkowego” ("Add to Home Screen"); Android – Chrome menu →
   „Zainstaluj aplikację” ("Install app"). The app then opens full screen, without the browser bar.

Without HTTPS (e.g. `http://100.x.y.z:8790`) the page works, but there is no microphone: the interface immediately shows this in
Polish together with the `tailscale serve` command, and typing to Gaja still works.

### Troubleshooting

| Symptom | What to do |
|---|---|
| „Mikrofon wymaga bezpiecznego połączenia” ("The microphone requires a secure connection") | Open the page via `https://…ts.net:8443` (see above) or `localhost`. |
| „Nie mam dostępu do mikrofonu” ("I don't have access to the microphone") | iPhone: Ustawienia → Safari → Mikrofon → „Pytaj/Zezwól” (Settings → Safari → Microphone → Ask/Allow; for the home-screen app: Ustawienia → Gaja → Mikrofon). Android/Chrome: padlock next to the address → Permissions → Microphone. Desktop: padlock → microphone. Once you allow it, the panel disappears by itself. |
| „Nie widzę mikrofonu” ("I can't see a microphone") | Connect a microphone; in the meantime you can type. |
| „Mikrofon jest zajęty” ("The microphone is busy") | Another app (a call, a recording) is using it – close it and try again. |
| No sound on iPhone | Check the mute switch and the volume; tap the screen (iOS allows sound only after a gesture). A phone call interrupts the sound – the next tap resumes it. |
| Gaja cuts herself off, „słyszy samą siebie” ("hears herself") | Use headphones, or turn off „Przerywanie głosem” ("Voice interruption") in settings (half-duplex mode; you interrupt by tapping). Lower the volume. |
| „Brak połączenia · ponawiam” ("No connection · retrying") | The adapter is not running, or the network / Tailscale is off. The page retries by itself with growing intervals; „Teraz” ("Now") retries immediately. |
| Odd voice / wrong speed | Check the selected voice in settings; „Automatycznie” ("Automatic") picks one matching the language. Details in *Diagnostyka* ("Diagnostics") (sample rates, underruns, AEC/NS/AGC). |
| The page does not refresh after an update | Reload manually (the adapter serves files without cache, but it is worth closing and reopening an installed PWA). |

---

## 3. CLI client (`clients/cli_wav.py`)

A command-line tool that simulates the microphone with a `.wav` audio file or text.

### Syntax:
```bash
.venv/bin/python clients/cli_wav.py [input.wav] [opcje]
```

### Options:
- `input.wav` – path to the input audio file (any sample rate/channels; automatically converted to mono 16 kHz PCM16).
- `--url <ws_url>` – WebSocket address (default `ws://127.0.0.1:8790/ws`).
- `--lang <pl|en|auto>` – language code (default `pl`).
- `--realtime` – send packets at real-time pace (every 40 ms); without this flag packets are sent immediately.
- `--out <reply.wav>` – path where the received audio of the assistant is saved.
- `--text "<treść>"` – send direct text instead of an audio file.
- `--timeout <seconds>` – time limit for waiting for the end of the turn (default 30.0s).

### Examples:
1. Streaming an audio file in real time and saving the reply:
   ```bash
   .venv/bin/python clients/cli_wav.py mowa.wav --realtime --out odpowiedz.wav
   ```

2. Text test with a metrics check:
   ```bash
   .venv/bin/python clients/cli_wav.py --text "Która godzina?"
   ```

---

## 4. Test / development server (`clients/mock_server.py`)

For testing the web interface and the CLI client without the real adapter (STT/LLM/TTS):
```bash
.venv/bin/python clients/mock_server.py --port 8799
```
The server simulates:
- static files `GET /…`, `GET /health` (with the list of engines),
- the full WebSocket cycle `WS /ws` in the v1.7 shape: hello (also a repeated one – a settings change), ready (with `engines`), vad,
  transcript, utterance, binary audio (Piper 22.05 kHz / Paradee 24 kHz / Supertonic 44.1 kHz), metrics, state,
  interrupt (`scope:"all"` cancels tasks), `tool` (e.g. „pogoda w Warszawie”, „ile to”, „awaria” ("failure")), `notice`
  (`stt-degraded` / `stt-recovered` written as text), Hermes tasks in the background with `id`, progress, `done` or
  `cancelled`, `cancel_task{id?}` (no task: `error`), `metrics{kind:"result"}`. Barge-in does not touch tasks.

Scenarios are selected with a URL parameter (the client passes it on to `/ws`): `?demo=live|user|thinking|speak|emotions|hermes|barge|stale|error|drop`,
and `&emotion=…`, `&pace=realtime`, `&tts=paradee`, `&engines=none` (an older server with no engine list), `&deltas=1` (sends `assistant_delta` like
a server from before v1.7), `&stt=degraded` (the session starts with a fallback-mode notice), `&hermes_s=N` (task duration;
default 8 s), `?demo=notice`. Full list – the file header.

---

## 5. Tests and screenshots

```bash
# testy (Chromium z fałszywym mikrofonem odtwarzającym polskie nagranie z bench/data/stt/)
TMPDIR=/tmp .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_web_e2e.py tests/test_cli_client.py

# te testy, które nie potrzebują scenariuszy mocka, na prawdziwym adapterze:
GAJA_E2E_BASE=http://127.0.0.1:8790/ TMPDIR=/tmp .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_web_e2e.py

# prawdziwy adapter na 8790: wpisany tekst, tura z fałszywym mikrofonem, zmiana głosu, Hermes, przerwanie, EN, ponowne
# połączenie (przez własny proxy TCP – adaptera nie dotyka), v1.7: tura z narzędziem (i), zadanie Hermesa przeżywa
# wpisaną turę (j), anulowanie przyciskiem (k), galeria zrzutów docs/ui/real-*.png
TMPDIR=/tmp .venv/bin/python clients/real_adapter_check.py [--only a,b,c,d,e,f,g,h,i,j,k,s]

# zrzuty wszystkich stanów do docs/ui/ (mock na 8799; --base wskazuje inny serwer)
.venv/bin/python clients/mock_server.py --port 8799 &
TMPDIR=/tmp .venv/bin/python clients/ui_screenshots.py [--only speaking-happy,history] [--sizes phone]
```

---

## 6. Verification on the real adapter (2026-10-09)

`clients/real_adapter_check.py` drives Chromium (headless, Playwright) through the real adapter at `:8790`; the microphone is a
WAV file (`--use-file-for-fake-audio-capture`, a Polish recording from `bench/`). Results:

| # | Check | Result |
|---|---|---|
| a | Typed text → audio + transcript + sentence highlighting, without turning on the microphone | OK (0 underruns, 0 microphone frames) |
| b | Microphone turn (WAV „Zapytaj Hermesa…” ("Ask Hermes…")) → Polish transcript, the reply is heard | OK (STT 465 ms, e2e to sound 1.7 s) |
| c | Voice change in settings (`hello.tts` → `piper-darkman`, `supertonic`), same session | OK (`ready` < 1 s, reply in the next voice) |
| d | Hermes card („sprawdź w Hermesie, jaki jest dziś dzień tygodnia” ("check with Hermes what day of the week it is today")): dispatched → progress → result, result spoken | OK (≈ 15–30 s, depending on load) |
| e | Tap interruption during a reply → buffer empty < 200 ms, zero further samples | OK |
| f | Switch to English: interface in English, reply in English with the `paradee` voice (24 kHz) | OK |
| g | Adapter restart (simulated by a TCP proxy that we kill and bring back up) → „Brak połączenia” ("No connection"); `ready` returns within a fraction of a second, the conversation keeps working | OK |
| h | Voice interruption (the fake-microphone loop speaks during a long reply) → `state: listening, reason: barge_in`, playback cut, „przerwane” bubble | OK |

Bugs found and fixed in `web/`: (1) Hermes response fragments on the card's timeline were split into dozens of lines;
the „Hermes pracuje” ("Hermes is working") card hung forever after an interruption (the server cancels the task without a
`handoff` event) – now „Zadanie przerwane” ("Task cancelled"); (3) the routing label `safety-net` was shown as the task description;
(4) the result repeated the last progress line.

Screenshots of the real run (`docs/ui/`): `real-{phone,desktop}-{dark,light}-{conversation-speaking,conversation,
hermes-working,hermes-done}.png` and `real-typed-done`, `real-speaking`, `real-listening`, `real-user-speaking`,
`real-mic-reply`, `real-settings*`, `real-hermes-*`, `real-barge-in`, `real-voice-barge-in`, `real-english`,
`real-offline`.

### Troubleshooting the adapter (visible in the UI)

| Symptom | Cause |
|---|---|
| Toast „language model unavailable” in the middle of a reply | llama-server crashed / is restarting; the adapter says „Model językowy jest chwilowo niedostępny” ("The language model is temporarily unavailable"). Try again in a moment. |
| Card „Hermes: błąd · ACP request … timed out” | Hermes (ACP) is not responding; this is on the adapter / Hermes side (`/health` → `hermes`). |
| The first turn after an adapter restart takes longer | The STT/TTS models are warming up. |

---

## 7. Protocol v1.7 in the UI (web-final, round 3, 2026-10-09)

Adaptation to core changes (round 6b, see „Requests” in `docs/ARCHITECTURE.md`):

| Event / field | What the UI does |
|---|---|
| `state{listening, reason:"barge_in"}` | Clears playback and marks the bubble as „przerwane”. **Does not** touch Hermes cards (previously: „Zadanie przerwane” – a bug). |
| `handoff{started/progress/done/error, id}` | Card keyed by `id` (several at once). Without `id` (old server) the key = task text. |
| `handoff{state:"cancelled", id, text:""}` | Card „Zadanie przerwane”, entry „Przerwano zadanie: …” ("Task interrupted: …") in the conversation, no `done`. |
| `cancel_task{id}` (**Przerwij** button) | Sent from the card; the button shows „Przerywam…” ("Cancelling…") until `cancelled`. `error: no running Hermes task …` unblocks it and shows an info toast. |
| `notice{code, message, engine}` | Toast + pill „Mowa: tryb awaryjny” (`stt_degraded`) / toast „znów działa normalnie” ("working normally again") (`stt_recovered`). Texts from `i18n.js` (PL/EN); unknown `code` → `message` from the server. |
| `tool{name, arguments, ok, text}` | Chip in the conversation; the spoken text arrives anyway as a regular `utterance`. |
| `metrics{kind:"result"}` | Skipped in the latency bar. |
| `assistant_delta` | The server does not send it; handling is kept for older servers (mock: `?deltas=1`). |
| `GET /health` | `engines` (voice list), `stt_degraded` (pill), `stt_engine`/`stt_device`/`router.tools` (diagnostics in settings). |

### Verification on the real adapter (8790), round 3

`clients/real_adapter_check.py --only i,j,k` (read-only questions only; Hermes gets only „jaki jest dziś
dzień tygodnia / miesiąca” ("what is today's day of the week / month")):

| # | Check | Result |
|---|---|---|
| i | „jaka jest pogoda w Warszawie” ("what's the weather in Warsaw") → `tool` event (`get_weather`, `location: Warszawa`), chip „🌤 pogoda · Warszawa” between the question and the answer, spoken answer | OK (3 of 4 runs in the UI; the first went local – see the routing note in „Requests” in `docs/ARCHITECTURE.md`) |
| j | „Zapytaj Hermesa jaki jest dziś dzień tygodnia” ("Ask Hermes what day of the week it is today") + a typed turn („dowcip o kocie”, "a joke about a cat") while work is in progress | OK: the card stayed `working` (1.5 s after typing), no `cancel_task`/`interrupt`/`cancelled`, events `started, progress×4, done`, one `id`, result „Dziś jest piątek, dziewiątego października.” ("Today is Friday, the ninth of October.") |
| k | Second task („…dzień miesiąca”, "…day of the month") cancelled with the button | OK: `cancel_task{id}` was sent, the server replied `handoff{cancelled}` with the same `id`, card „Zadanie przerwane”, for 9 s no `done` and no error in the conversation |

Screenshots: `docs/ui/real-tool-chip.png`, `real-tool-chip-start.png`, `real-settings-diag.png`, `real-hermes-running.png`,
`real-hermes-running-typed.png`, `real-hermes-done-v17.png`, `real-hermes-cancel-button.png`, `real-hermes-cancelled.png`.

## 8. Microphone selector (lead, 2026-10-10)
Settings → "Urządzenie wejściowe" ("Input device"): the first option, "Domyślny systemowy" ("System default"), stores `mic: ''` and opens the
mic with no `deviceId` (OS default). The list comes from `enumerateDevices()` (Chrome's `default` /
`communications` pseudo-devices are hidden; names appear once permission was granted). The choice is kept
in `localStorage['gaja.settings'].mic`. `AudioController.openMicStream` tries the chosen device, then the
default with AEC/NS/AGC, then a bare `{audio:true}`; if the remembered device is gone it resets to the
default and shows a toast. Changing the device while the mic is open reopens it (the PTT gate is kept).
The list refreshes on `devicechange`. Orb: the shader fades its output over the outer 14% of the canvas,
so the Hermes satellite's halo can't be cut by the canvas edge; the satellite orbit is now 1.32·R0.

### 8.1 "No signal" watchdog (lead, 2026-10-10)

While the mic is open (`armed` or `live`), `app.js` samples the raw input RMS every 250 ms (`audio.getMicRms()`, not gated by
push-to-talk). If it stays at or below `0.0003` (digital silence — a real room is ~1e-3 and up) for 5 s, a `nosignal` notice tells
the user that no sound is reaching the app, names the device in use, and offers "Wybierz mikrofon" ("Choose microphone") (opens settings, focuses
`#set-mic`). It never covers another notice, fires once per mic-open/device-switch, and clears itself as soon as signal appears.

## 9. Wake-word sample recorder (2026-10-10)

`/record.html` (linked from the bottom of the settings sheet: "Nagraj próbki słowa kluczowego" ("Record wake-word samples")) records real "Hej Gaja"
training/eval samples for the custom wake word (microWakeWord pipeline: 16 kHz mono 16-bit WAV). Files: `web/record.html`,
`web/record.js`, the prompt list in `web/record-script.js` (one `SECTIONS` array: id, what to say, how, target count – edit
it freely, but never rename an `id` once recordings exist, it is the folder name), styles `.rec-*` at the end of `style.css`.
Screenshots: `docs/ui/record.png` (desktop, dark), `docs/ui/record-phone-light.png`.

- **Script.** Sections *Pozytywy* ("Positives") (`wake`), *Trudne negatywy* ("Hard negatives") (`hard_negative`, ×3 each) and *Tło* ("Background") (`ambient`: 6 takes of
  up to 10 min, auto-uploaded in 60 s chunks; an ambient prompt counts as done after 5 min). Per-prompt and overall progress
  come from the server (`GET /api/wakeword/progress`), so reloading resumes the session; "Sesja" ("Session") lets you start a new one
  or continue an old one.
- **Recording.** Raw microphone: `AudioController.openMicStream(deviceId, { raw: true })` turns echo cancellation, noise
  suppression and AGC off (the page shows what the browser reports, `AEC/NS/AGC off`). Same device list and the same
  setting as the main app (`gaja.settings.mic`). Audio goes through `capture-processor.js` (16 kHz PCM16) and is encoded
  to WAV in the browser. 100 ms are cut from both ends of a take (key clicks), so the speaker should wait ~0.5 s before and
  after speaking. A take auto-stops after 12 s.
- **Continuous recording.** The ambient prompt `amb_continuous` ("Nagrywanie ciągłe", `continuous: true` in `record-script.js`) has no
  600 s cap: it records until Stop and uploads every 60 s chunk as its own take under `ambient/amb_continuous/`, meant for leaving a USB
  mic running for hours or days (~115 MB per hour; the default `max_total_mb` is 32768). While it runs the page holds a screen wake lock
  (taken again on `visibilitychange`), keeps recording in a background tab, and shows a panel with elapsed time, chunks and MB uploaded,
  the last upload time and the pending count. A failed upload is retried forever with backoff (2 s ... 60 s; the "online" event retries at
  once); beyond 30 pending chunks the oldest is dropped and counted ("utracone"). If the mic track ends or no audio frame arrives for 10 s
  the page drops the dead graph, uploads what it has and re-opens the same device (`openMicStream(..., { strict: true })`, never a
  fallback to another mic) every 5 s, continuing the same take numbering. HTTP 507 stops the recording and keeps the unsent chunks for
  "Ponów wysyłanie". Space does not stop a continuous run (only the button), to avoid a stray key press ending it.
- **Keys.** Space = start/stop, Enter = Zapisz ("Save"), Backspace or R = Powtórz ("Repeat"). After stopping, the take is played back (toggle
  "Odtwarzaj po nagraniu" ("Play after recording")), with a waveform; warnings for silence (peak below -40 dBFS), clipping (peak >= 0.99) and takes
  shorter than 0.8 s. "Cofnij ostatni zapis" ("Undo last save") deletes the last saved take on the server.
- **Server.** `gaja/wakeword_samples.py` (router included in `create_app`):
  `POST /api/wakeword/samples?session&label&prompt_id&take[&mic]` (raw `audio/wav` body, PCM16 mono 16 kHz, <= 4 MB;
  409 if the take exists), `GET /api/wakeword/progress?session=`, `GET /api/wakeword/sessions`,
  `DELETE /api/wakeword/samples/<session>/<label>/<prompt_id>/<take>`, `GET /api/wakeword/export.zip[?session=]`.
  Stored as `data/wakeword/recordings/<session>/<label>/<prompt_id>/<take:03d>.wav` plus `<session>/meta.jsonl` (ts, label,
  prompt_id, take, duration_s, peak, rms, user agent, mic label; an undo appends a `"deleted": true` line). Path components
  are `[a-z0-9_-]` only. Every endpoint refuses a present-but-unlisted `Origin` (same `server.allowed_origins` as `/ws`).
  Config: `[wakeword] recordings_dir`, `max_total_mb` (returns 507 beyond that). `data/` is git-ignored.
- **To Drive/Colab.** Open `/api/wakeword/export.zip` (or the "Pobierz archiwum" ("Download archive") button) and upload the zip; folder names are the
  labels (`wake`, `hard_negative`, `ambient`).
- Tests: `tests/test_wakeword_samples.py` (TestClient). Run by hand with a fake mic: Chromium
  `--use-fake-ui-for-media-stream --use-fake-device-for-media-stream [--use-file-for-fake-audio-capture=x.wav]` against any server that mounts
  the router and `web/`.
