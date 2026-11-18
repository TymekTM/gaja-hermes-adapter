# Gaja desktop client (`gaja-desktop`)

A Windows PC joins Gaja as one more **device** (protocol v1.14, docs/DESKTOP.md): Gaja hears an audio
source of your choice and speaks into an output device of your choice — and the agent (Hermes) can run
PC tools on the machine through the adapter (`shell`, `open`, `screenshot`, `type`, `keys`, `mouse`,
`clipboard`). Runs on Windows 10/11; Linux works too (PipeWire/Pulse) and is what the tests use.

```
gaja-desktop run            # connect, wake word + hotkey + tray icon
gaja-desktop run --no-tray  # headless (console only)
gaja-desktop devices        # list capture, loopback and playback devices
gaja-desktop settings       # open the settings window
```

## Install

### With uv (recommended for setup)

1. Install [uv](https://docs.astral.sh/uv/), then from a checkout of this repo:

   ```sh
   cd desktop
   uv sync
   uv run gaja-desktop devices   # sanity check: your audio devices
   uv run gaja-desktop run
   ```

   On Windows `uv sync` fetches a matching Python automatically (3.11+; `.python-version` pins 3.13
   for development).

### Built exe

Download `gaja-desktop.exe` from a `desktop-*` release artifact (built by
`.github/workflows/desktop.yml`) and run it. The exe bundles the wake-word model; configuration still
lives in `%APPDATA%\Gaja\desktop.toml`.

## Server side (once, on ttm-server)

1. Enable the device listener (`[devices] enabled = true`, docs/DEVICE.md) — the lead owns the
   adapter process; agents never restart it.
2. Create a device token for the PC:

   ```sh
   uv run python scripts/device_token.py pc-<name>
   ```

   It prints the token once and stores it in `~/.config/gaja/devices.toml`.
3. The client connects to `ws://<tailscale-ip>:8792/device` (Tailscale = encrypted; plain LAN
   `ws://` sends the token and the audio in cleartext). Firewall: only the device port 8792 needs to
   be reachable, and preferably only over the tailnet.

## Configuration

`%APPDATA%\Gaja\desktop.toml` (Linux: `~/.config/gaja/desktop.toml`), edited by the settings window
(`gaja-desktop settings`) or by hand:

```toml
[device]
url = "ws://100.x.y.z:8792/device"
device_id = "pc-tymek"
token = "…"          # keep this file private!
lang = "pl"

[audio]
input = "loopback:Speakers"   # or a mic name substring, or "" for the default mic
outputs = ["CABLE Input", "Speakers"]
out_rates = [48000, 24000, 16000]
half_duplex = true            # no mic upload while Gaja speaks (input hears the output)

[wake]
enabled = true
preroll_ms = 1500
model = ""                    # optional: path to a custom model .json

[ptt]
enabled = true
hotkey = "ctrl+alt+g"

[tools]
enabled = ["shell", "open", "screenshot", "type", "keys", "mouse", "clipboard"]
```

`input` is matched by substring against the capture devices; `loopback:<name>` captures what the
**output** device `<name>` plays (WASAPI loopback — no virtual driver needed). `outputs` may list
several devices; every device gets the reply resampled to its own rate.

## VB-CABLE setup recipes

Install [VB-CABLE](https://vb-audio.com/Cable/) once (it adds one virtual device pair: playback
`CABLE Input`, recording `CABLE Output`).

### (a) Gaja as a virtual microphone — apps hear Gaja

Gaja's voice must reach other apps (Discord, OBS, a game pick `CABLE Output` as their microphone):

- config: `outputs = ["CABLE Input", "Speakers"]` (`Speakers` optional, as your local monitor);
- in the app, choose **CABLE Output** as the microphone.

Now Hermes talking on the PC ("say …" / `speaker_say`) arrives in any app that listens on the cable.

### (b) Gaja hears an app — "Gaja as a virtual speaker"

Two ways:

1. **Loopback** (no extra driver): set `input = "loopback:<output name>"` for the output the app
   plays through. Gaja then hears *everything* that output plays — including other apps.
2. **A second cable** (cleaner): a second VB-CABLE instance or Voicemeeter gives you e.g.
   `CABLE A Input`/`CABLE A Output`. Set `input = "CABLE A Output"` and make the one app you want
   Gaja to hear play into `CABLE A Input`. Other apps keep their normal output.

## PC tools — read this

**The agent gets full shell access on this PC, with no confirmation per call** (the user's decision,
2026-11-18; docs/DESKTOP.md "Security notes"). Hermes can run commands, type, move the mouse, read
and set the clipboard and take screenshots. Only connect to a server you trust, preferably over
Tailscale or `wss://`. The tray menu has a **"PC actions"** switch: turning it off reconnects without
`caps.pc`, so the server sees a plain voice device and cannot call tools. Individual tools can be
disabled in `[tools] enabled`; a tool whose backend is missing (e.g. no clipboard utility on Linux)
is simply not advertised. The device token (voice access) and the server's PC API token are separate
secrets; neither is ever logged by the client.

## Push-to-talk

The global hotkey (default `ctrl+alt+g`, configurable) sends a `push_to_talk` wake: the server never
verifies it, the press is the intent. Useful when the wake word is off (`[wake] enabled = false`) or
noisy.

## Session flow

Same as the ESP32 firmware: idle (wake word listening) → `wake` + pre-roll burst → 20 ms PCM16
16 kHz frames → reply plays at the negotiated rate (48000 offered first) → `follow_up` window reopens
the mic without a phrase → `listen_end` returns to idle. Half duplex by default: while a reply plays,
the mic is not uploaded (the captured source usually hears the playback). Dropouts reconnect with
backoff (0.5 s × 1.7ⁿ capped at 10 s with jitter).
