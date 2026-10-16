# Gaja IoT voice device (protocol v1.9)

Server side for a small voice device with on-device wake word (target: M5Stack Atom VoiceS3R / EchoS3R,
ESP32-S3, one MEMS mic, ES8311 codec). The device detects "Hej Gaja" itself (microWakeWord) and streams audio
only after that. The hardware is not bought yet: everything here is verified against the simulator
(`clients/device_sim.py`) and stubbed STT/LLM/TTS, **not against a real device**.
Contract summary: `docs/ARCHITECTURE.md` (v1.9). Firmware plan: `firmware/README.md`.

## Enabling it

The device listener is a separate, optional port. It is **off** (`[devices] enabled = false`) and is not enabled
in the repo `config.toml`. It runs in the adapter process (same event loop, same STT/LLM/TTS), serves only
`WS /device` (no static UI, no `/health`, no docs) and is started by the app lifespan.

1. Create a token (printed once, stored with mode 600 in `~/.config/gaja/devices.toml`, outside the repo):
   `uv run python scripts/device_token.py atom-salon --name Salon`
2. Enable it, for example in the adapter's environment: `GAJA_DEVICES__ENABLED=true` (or `[devices] enabled = true`
   in a private config selected with `GAJA_CONFIG`). Restart the adapter (the lead does that on this host).
3. Put the token, `device_id` and `ws://<server-ip>:8792/device` into the firmware (or `clients/device_sim.py --token`).

`/health` shows `devices: {enabled, listening, connected: [ids]}`; `/metrics` gains `counters`
(`wake_accepted`, `wake_rejected`, `wake_rejected:<reason>`).

## Protocol

Everything of the web protocol applies (see ARCHITECTURE.md): PCM16 LE mono 16 kHz up as binary frames,
`transcript`, `utterance`, `audio_start`, binary PCM16 mono down, `audio_end`, `state`, `metrics`, `error`, `ping`.
The device endpoint adds:

Device -> server
- `hello` (first message, within 5 s, max 4096 bytes):
  `{"type":"hello","client":"device","device_id":"atom-salon","token":"...","fw":"0.1.0","lang":"pl",
  "caps":{"aec":false,"out_rates":[16000,24000],"preroll_ms":1500}}`.
  `caps.aec`: the device cancels its own speaker echo, so it may keep streaming while playing.
  `caps.out_rates`: 1 to 8 rates in 8000..48000 Hz the speaker supports. `caps.preroll_ms`: 0..5000, the ring buffer.
- `{"type":"wake","word":"hej_gaja","score":0.97,"preroll_ms":1500}` the on-device detector fired. The next binary
  frames are the pre-roll (audio before the wake, up to `preroll_ms`, sent as fast as possible) and then live audio.
- binary frames: 20 ms (640 B) recommended, any size accepted.

Server -> device
- `ready` as on `/ws` plus `sample_rate` = the **negotiated output rate** and `device:{id,aec,out_rate,verify_wake,
  verify_ms,follow_up_ms,ack}`. Every `audio_start.sample_rate` equals that rate; the server resamples TTS
  (scipy polyphase) and sends chunks of `out_chunk_ms` (100 ms).
- `wake_accepted`: the phrase was confirmed (or verification is off); the turn proceeds.
- `wake_rejected{reason}`: `no_phrase` (transcript did not start with the phrase), `no_speech`, `timeout` (no
  verdict in `wake_timeout_s`), `low_score` (a weak wake while Gaja was busy), `stt_error`, `superseded` (a newer
  wake replaced it). The device returns to idle at once and drops the mic stream.
- `transcript` gains `heard` (the full text including the phrase) when the phrase was stripped from `text`.
- `follow_up{ms}`: after a reply the mic should stay open for `ms` (default 6000) with no wake phrase needed.
- `listen_end`: stop streaming and go back to wake-word-only listening (silence after an accepted wake, or an
  expired follow-up window). Sent for devices without AEC after each utterance end, because they close the mic.
- `timer`: `{"type":"timer","id":"...","label":"...","kind":"timer|reminder"}` a scheduled timer or reminder fired on the server, accompanied by spoken Polish TTS audio frames for devices with audio playback.

Output rate: the server picks 16000 if offered, else the lowest offered rate >= 16 kHz, else the highest.

### Flow

1. `wake`, then pre-roll + live frames. The server runs VAD and, with `verify_wake`, transcribes the first
   `verify_ms` (2500) of the stream as soon as speech ended or the buffer is full.
2. Verification: the text must start (after at most 2 filler words) with a greeting `hej/hey/ej` and a name
   `gaja/gajo/gaju/gaia/gaya` (Polish diacritics normalised, a merged "hejgaja" accepted; a fuzzy name must keep the
   shape of the name — "ga" + a j/i/y glide + vowels, e.g. "gayo"/"gaio" — and stay within one edit of
   `gaja/gajo/gaju`, so real words like `gala`, `gaj` or `gajda` are rejected). `hej maja` is rejected. Otherwise
   `wake_rejected{no_phrase}` and the audio of this wake is dropped without an LLM call.
3. On accept the phrase is stripped. If a command remains in the same utterance, the normal turn runs. If it was
   the bare phrase, the server acknowledges (`ack = "spoken"`: TTS "Slucham?"; `"earcon"`: a short beep as audio;
   `"none"`) with `metrics.route = "wake"` (no LLM, no history entry) and keeps listening; the next utterance needs
   no phrase.
4. After the reply: `follow_up{ms}`. If nobody speaks, `listen_end`.

**Lone "Gaja" policy** (`accept_bare_name = true`): a name as the very first word is accepted ("Gaja, jaka jest
pogoda"), because the device already heard the full phrase and Whisper often drops the "hej". A name later in
the sentence ("opowiedz mi o Gaji") is never a wake. Set `accept_bare_name = false` to demand the greeting.

### Barge-in

- With `caps.aec = true` the device keeps streaming while the reply plays; the existing VAD barge-in applies.
- Without AEC the device stops streaming during playback (so the server never hears its own voice). An
  on-device wake during playback cancels the reply (same path as barge-in; Hermes tasks keep running). The
  pre-roll is then skipped, because it contains Gaja's own voice.
- A wake while Gaja is busy needs `score >= barge_in_min_score` (0.9); weaker ones get `wake_rejected{low_score}`.

## Security model

- Per-device random token (`secrets.token_urlsafe`, 256 bit), stored only in `~/.config/gaja/devices.toml`
  (mode 600 enforced on creation, a warning is logged otherwise, hot-reloaded on mtime change). Compared with
  `hmac.compare_digest`; an unknown id is compared against a decoy so timing does not reveal ids. Tokens are
  never logged. Failure answer is the same for unknown id and wrong token (`error: unauthorized`), after a 1 s delay.
- Only `/device` is served on that port; peers must be inside `allowed_networks` (loopback, RFC1918, tailnet CGNAT,
  link-local, ULA) by default. A hello that is not the first frame, is larger than 4 KiB or late (5 s) closes the
  socket. Frames are capped (`ws_max_size` 1 MiB). `max_connections = 8`. A reconnect of the same device replaces
  the old (possibly half-open) connection.
- Plain `ws://` on the LAN sends the token and the audio **in cleartext**. For anything beyond a trusted LAN set
  `tls_certfile` and `tls_keyfile` (wss; not exercised by the tests) or put the port behind a VPN/tailnet. The
  default bind is `0.0.0.0:8792`, so firewall it from the internet.
- A device-kind hello on `/ws`, or a web hello on `/device`, is refused.

## Configuration (`[devices]`)

`enabled=false`, `listen_host="0.0.0.0"`, `port=8792`, `tokens_file`, `allowed_networks`, `max_connections=8`,
`hello_timeout_s=5`, `tls_certfile/tls_keyfile`, `verify_wake=true`, `verify_ms=2500`, `accept_bare_name=true`,
`extra_wake_names`, `extra_wake_greetings` (add spellings Whisper really produces), `wake_timeout_s=8`,
`listen_timeout_s=8`, `barge_in_min_score=0.9`, `ack="spoken"`, `ack_text={pl="Slucham?",en="Yes?"}`,
`follow_up_ms=6000`, `listen_grace_s=1.0`, `preferred_out_rate=16000`, `out_chunk_ms=100`.
Defaults and comments: `gaja/config.py`, `config.toml`.

## Simulator

```bash
uv run python clients/device_sim.py cmd.wav --token $TOKEN --device-id atom-salon \
    --url ws://127.0.0.1:8792/device --out reply.wav [--wake-at 1.2] [--no-aec] [--barge-in-at 1.5] \
    [--follow-up-wav next.wav] [--out-rates 16000,24000]
```

It sends hello, `wake` at `--wake-at` seconds of the WAV, the pre-roll as a burst, the rest in real-time 20 ms
frames, then silence while the mic is open; models playback (a barge-in drops what was not played), writes the
audio and prints events plus timings: wake -> verdict, wake -> first audio, end of speech sent -> first audio.
Exit codes: 0 ok, 1 timeout, 2 connection or auth error, 3 wake rejected.
Note that `wake -> first audio` includes the time the WAV itself takes to be spoken; the second number is the
latency the user perceives.

## Firmware behaviour notes (ESPHome device side, v1.10)

- Reconnect: exponential backoff 2 s .. 60 s measured from the start of each attempt (>= 30 s after an
  `unauthorized` error); a missing `ready` (6 s) counts as a failed attempt and backs off like any other. The main
  loop never waits for the WebSocket task: a failed connect ends by itself, a live link is stopped on a helper task.
- If the `wake` message cannot be sent (link busy), the turn is aborted and the wake word re-armed: the device never
  streams audio to a server that did not see the wake.
- Server text messages are queued (32 deep). Under pressure only log-only types (`transcript`, `utterance`,
  `metrics`, `state`, `vad`, `handoff`, `tool`, `notice`, `pong`) are discarded (above half full) or evicted; control
  messages (`ready`, `listen_end`, `audio_start/end`, `follow_up`, `error`, `wake_*`, `record_*`, `timer`, `ping`)
  never are.
- Watchdogs: a reply whose speaker stops making progress (8 s) or does not stop after `finish()` (3 s) is ended with
  the amplifier off and the playback buffer dropped; if the mic does not release the shared I2S bus within 5 s
  playback is attempted anyway (amplifier and adaptive volume still applied).

## Verification status

- Automated (`tests/test_device_*.py`): matcher, registry, auth, negotiation and resampling, wake flows, ack modes,
  barge-in, follow-up, rejection paths, embedded listener on a real socket, simulator end-to-end. STT, LLM and TTS
  are stubs (amplitude VAD).
- Real device (M5Stack Atom VoiceS3R, ESPHome firmware in `firmware/esphome/atom-voices3r-gaja.yaml`, not yet flashed):
  - The mic and the speaker share one I2S bus, so the mic is physically off while the speaker plays. The no-AEC
    on-device-wake barge-in ("a wake during playback cancels the reply") therefore **cannot happen on this hardware**:
    the user must wait for the reply to finish; `barge_in_min_score` is irrelevant for it. The device also never hears
    its own voice, so the pre-roll is never its own playback (`skip_samples` stays 0 in practice).
  - It advertises `caps.out_rates: [48000]` (`gaja_device.out_rate`; the component also asserts that
    rate on the speaker before every playback — ESPHome's i2s_audio speaker stream info otherwise
    defaults to 16 kHz and the YAML `sample_rate` does not set it), so the server's polyphase
    resampling path (16/22.05/24 kHz TTS -> 48 kHz, 100 ms chunks) is what this device exercises.
  - `wake.score` is a configured constant (0.97): the ESPHome wake trigger exposes no detection probability; the score
    only matters for the barge-in admission above.
- Not verified: real hardware end-to-end (the ESPHome firmware compiles but nobody has flashed it), ESPHome yaml on the
  device, real Whisper spellings of the phrase (collect them from the live logs and add to `extra_wake_*`), Silero VAD
  on device audio, TLS, echo behaviour of the real speaker/mic, Wi-Fi jitter.
- Known risk: the device hearing its own "Gaja" during a long reply (mitigated by the score threshold and no-AEC
  muting; moot on the shared-bus Atom, which mutes its mic during playback by construction).
