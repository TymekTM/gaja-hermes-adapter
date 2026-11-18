# Gaja desktop client (Windows) — contract, device protocol v1.14

A Windows PC joins Gaja as one more **device** on the existing `WS /device` listener (docs/DEVICE.md). It does
two things the Atom cannot:

1. **Virtual audio.** Gaja hears an audio source chosen on the PC and speaks into an audio output chosen on the
   PC. With VB-CABLE installed, Gaja's voice goes to `CABLE Input`, so any app (Discord, OBS, a game) can pick
   `CABLE Output` as its microphone ("Gaja as a virtual microphone"). Gaja listens either to a capture device
   (a real mic, or a second virtual cable that apps play into: "Gaja as a virtual speaker") or to a **loopback**
   of any output device (Gaja hears what that output plays, no driver needed).
2. **PC actions for the agent.** The client advertises tools (shell, open, screenshot, keyboard, mouse,
   clipboard). Hermes calls them through the `gaja_pc` MCP server on ttm-server, which relays each call over
   the adapter to the connected PC. Every call is executed without asking (the user's decision, 2026-11-18).

```
 Windows PC: gaja-desktop                                ttm-server
 ┌─────────────────────────────────┐   WS /device   ┌───────────────────────────────────────────┐
 │ capture (mic | cable | loopback)├──PCM16 16k────▶│ device controller ─▶ STT ─▶ LLM ─▶ TTS     │
 │ wake word (hej_gaja.tflite)     │                │                                           │
 │ push-to-talk hotkey             │◀──PCM16 out────│ pc relay (pc_call / pc_result)            │
 │ playback ─▶ CABLE Input/speakers│                │   ▲  POST /api/pc/{id}/call (bearer token) │
 │ tool executor (auto-approved)   │◀──pc_call──────│   │                                       │
 │                                 │──pc_result────▶│ gaja.mcp_pc (stdio MCP) ◀── Hermes        │
 └─────────────────────────────────┘                └───────────────────────────────────────────┘
```

## Protocol v1.14 (delta to docs/DEVICE.md v1.13)

All of v1.13 applies unchanged. A server that predates v1.14 ignores the new caps; the client then works as a
plain voice device and must not send `pc_result`.

### Hello

`caps` gains two optional fields (unknown caps are already ignored by older servers):

- `caps.ptt: bool` (default false) — the device has a push-to-talk control. A `wake` with
  `"word":"push_to_talk"` from such a device is **never verified** (the press is the intent): it is accepted
  like an unverified wake (`_decide(wake, True, "push_to_talk")`). From a device without `caps.ptt` that word
  is treated like any other word (verified as usual).
- `caps.pc: object | null` (default null) — the device executes PC tools:
  `{"os":"windows"|"linux"|"macos","host":"<hostname, ≤64 chars>","tools":["shell","open","screenshot","type","keys","mouse","clipboard"]}`.
  `tools` lists only tools the client has enabled; names outside the set above are dropped by the server.

`ready.device` gains `"pc": true` when the server accepted `caps.pc` (`[pc] enabled` and the device id is not in
`[pc] deny_devices`), else `false`. A client must not expect `pc_call` unless it saw `pc: true`.

Desktop client hello example:
```json
{"type":"hello","client":"device","device_id":"pc-tymek","token":"…","fw":"desktop-0.1.0","lang":"pl",
 "caps":{"aec":false,"out_rates":[48000,24000,16000],"preroll_ms":1500,"ptt":true,
         "pc":{"os":"windows","host":"TYMEK-PC","tools":["shell","open","screenshot","type","keys","mouse","clipboard"]}}}
```

### Server -> device: `pc_call`

`{"type":"pc_call","id":"<12 lowercase hex>","tool":"<name>","args":{...},"timeout_s":<1..300>}`

Sent at any time, including mid-turn and while audio streams. Several calls may be in flight; the client runs
them concurrently (a slow `shell` must not block the audio loop or other calls) and answers each exactly once.
The client enforces `timeout_s` itself (kills the process for `shell`) and answers `ok:false` with
`"error":"timeout"`.

### Device -> server: `pc_result`

`{"type":"pc_result","id":"<same id>","ok":true|false,"text":"…","data":{…},"error":"…","image":{"mime":"image/jpeg"|"image/png","b64":"…","width":W,"height":H}}`

- `text`: human/agent-readable output, ≤ 65536 characters (the client truncates the middle with
  `\n…[truncated N chars]…\n`). `data`: small structured extras (e.g. `exit_code`). `error`: short code or
  message when `ok` is false. `image`: only for `screenshot`.
- The whole frame must stay under the listener's 1 MiB `ws_max_size`: the client keeps `image.b64` ≤ 900 KiB
  (downscale, then JPEG quality steps) — a frame over the limit would kill the connection.
- A result for an unknown, finished or timed-out id is ignored by the server (logged at DEBUG).
- A pending call whose device disconnects fails at once with `error:"disconnected"`; a server-side timeout
  (`timeout_s` + 5 s grace) fails with `error:"timeout"`.

### Tools (arguments and results)

| tool | args | result |
|---|---|---|
| `shell` | `command: str` (required), `shell: "powershell"\|"cmd"\|"sh"` (default `powershell` on Windows, `sh` elsewhere), `cwd: str?` | `text` = combined stdout+stderr (UTF-8, errors replaced), `data.exit_code`; `ok` = process ran (a non-zero exit is still `ok:true`) |
| `open` | `target: str` — URL, file/folder path, or program name/path; `args: list[str]?` for programs | `text` = what was opened |
| `screenshot` | `monitor: int` (0 = all monitors combined, 1.. = one monitor; default 0), `max_width: int` (default 1600) | `image` + `text` with the mouse-space rect and the image→mouse mapping (see below); `data.monitors` = per-monitor `{index,left,top,width,height}` |
| `type` | `text: str` (≤ 10000 chars) | `text` = number of characters typed |
| `keys` | `combo: str` like `"ctrl+shift+esc"`, `"win+r"`, `"enter"`; or `sequence: list[str]` of combos pressed in order | `text` = what was pressed |
| `mouse` | `action: "move"\|"click"\|"double_click"\|"right_click"\|"scroll"`, `x: int?`, `y: int?` (absolute virtual-screen pixels, the screenshot monitor-0 rect before downscaling; **negative is valid** — the origin is the primary monitor's top-left, a monitor left of/above it has negative coordinates), `amount: int?` (scroll clicks, + up) | `text` = final cursor position |
| `clipboard` | `action: "get"\|"set"`, `text: str?` (for set) | `text` = clipboard content (get) |

Unknown tool, a tool not in the device's advertised list, or invalid args: `ok:false`, `error:"bad_request: …"`.

**Image → mouse mapping.** The screenshot is downscaled to `max_width`, so image pixels must be
scaled back to mouse coordinates: for monitor *m* with rect `{left, top, width, height}` (mouse
space, before downscaling — `data.monitors[m]`, also spelled out in `text`),
`mouse_x = left + img_x * width / image_width`, `mouse_y = top + img_y * height / image_height`
(e.g. `mouse = (-1920 + img_x*3840/1600, 0 + img_y*1080/450)` for two 1920×1080 monitors side by
side, the second one primary). The client is DPI-aware, so one image pixel is one mouse pixel at
native resolution.

### Push-to-talk

`{"type":"wake","word":"push_to_talk","score":1.0,"preroll_ms":0}` then live frames, exactly like a wake. The
server endpoints the utterance with its VAD as usual. Barge-in rules are those of a device wake (score 1.0 ≥
`barge_in_min_score`, so PTT always interrupts a reply).

## Server side

- **Config** `[pc]` (`gaja/config.py`, `PcConfig`): `enabled = true`; `api_token_file =
  "~/.config/gaja/pc_api.token"`; `default_timeout_s = 60`; `max_timeout_s = 300`; `deny_devices = []`.
  `[pc]` only has an effect while `[devices] enabled` (the PC connects to the device listener).
- **Relay** (`gaja/devices/pc.py`): per connected PC device a registry entry `{device_id, host, os, tools}` and
  a map of pending call futures; `call(device_id, tool, args, timeout_s) -> PcResult`. The device controller
  routes `pc_result` messages to it and fails pending calls on disconnect/supersede.
- **HTTP API** on the main app (`gaja/api/pc.py`, mounted at `/api/pc`):
  - `GET /api/pc` → `{"devices":[{"id","host","os","tools","connected_s"}]}`
  - `POST /api/pc/{device_id}/call` body `{"tool","args","timeout_s"?}` → `{"ok","text","data","error","image"?}`;
    404 unknown/disconnected device, 400 tool not advertised, 504 timeout (body still JSON).
  - **Auth**: the main app is reachable over `tailscale serve`, so the origin check is not enough for a
    remote-code-execution endpoint. Every `/api/pc` request needs `Authorization: Bearer <token>` where the token
    is the content of `api_token_file` (created on first start with 32 random bytes hex, mode 600, parent dir
    700). Compare with `hmac.compare_digest`. Any request carrying an `Origin` header is refused (403): browsers
    never get to call it. Missing/invalid token → 401.
- **MCP** `gaja.mcp_pc` (stdio, `python -m gaja.mcp_pc`, `FastMCP("gaja_pc")`), modelled on `gaja.mcp_speaker`:
  reads the token file, talks to `GAJA_API_URL` (default `http://127.0.0.1:8790`). Tools: `pc_list`,
  `pc_shell`, `pc_open`, `pc_screenshot` (returns an MCP image plus text), `pc_type`, `pc_keys`, `pc_mouse`,
  `pc_clipboard`. Every tool takes an optional `device_id`; when omitted and exactly one PC is connected it is
  used, otherwise the tool returns an error listing the connected PCs. Per-call HTTP budget = `timeout_s` + 10 s,
  capped below one Hermes turn like the speaker MCP.
- Registered for Hermes in `~/.hermes/config.yaml` under `mcp_servers.gaja_pc` (done by the lead, not by agents).
- The PC is also an ordinary speaker device: `speaker_say` / `speaker_play` from the speaker MCP can target it,
  so Hermes can talk into the virtual microphone.

## Desktop client (`desktop/`)

A separate Python project (own `desktop/pyproject.toml` and `desktop/uv.lock`, package `gaja_desktop`) so the
Windows install never pulls the server stack (torch, onnx STT, …). Python ≥ 3.11, runs on Windows 10/11
(target) and Linux (development and tests; PipeWire/Pulse through the same libraries).

- **Audio** via `soundcard` (WASAPI on Windows, Pulse on Linux; supports loopback capture with
  `include_loopback=True`). Input = one capture source: a microphone/cable by name substring, or
  `loopback:<output name substring>`. Output = one or more playback devices by name substring (e.g.
  `CABLE Input` plus the speakers as a monitor). Captured audio → mono → 16 kHz PCM16, 20 ms frames.
  Playback at the negotiated `ready.sample_rate` (offer 48000 first), resampled to each device's rate.
- **Wake word**: `pymicro-wakeword` with the repo's `firmware/esphome/models/hej_gaja.{tflite,json}` (copied into
  the package at build time; the client also accepts a custom path). Pre-roll ring buffer `preroll_ms`.
  Half duplex by default (no wake detection and no mic upload while Gaja's audio plays to a device that the
  capture source can hear; configurable).
- **Push-to-talk**: a global hotkey (default `ctrl+alt+g`, `pynput`) sends the PTT wake.
- **Session flow** = the firmware's: idle → `wake` (+pre-roll) → stream until `wake_rejected`/`listen_end`/the
  reply ends; `follow_up{ms}` re-opens the mic without a wake for `ms`; reconnect with backoff
  (0.5 s × 1.7ⁿ, max 10 s, jitter).
- **Tools** (`gaja_desktop/tools/`): one module per tool, Windows implementation plus a Linux fallback where
  cheap (`sh`, `xdg-open`, `mss`, `pynput`); a tool whose backend is missing is simply not advertised.
  Each call runs in a worker thread/subprocess; output capping and timeouts per the protocol.
- **UI**: tray icon (`pystray`) with state (connected / listening / speaking / error), a mute-listening toggle,
  "Settings…" (a small `tkinter` window: server URL, device id, token, input source and output devices picked
  from live lists, wake word on/off, hotkey, enabled tools) and Quit. Headless `gaja-desktop run --no-tray` too.
- **Config**: `%APPDATA%\Gaja\desktop.toml` (Linux: `~/.config/gaja/desktop.toml`), token stored there with a
  warning to keep the file private. CLI: `gaja-desktop run`, `gaja-desktop devices` (list capture, loopback and
  playback devices), `gaja-desktop settings`.
- **Packaging**: PyInstaller spec producing `gaja-desktop.exe` (windowed) + a GitHub Actions workflow
  (`.github/workflows/desktop.yml`, `windows-latest`, manual trigger and tags `desktop-*`) that builds it and
  uploads the artifact. PyInstaller cannot cross-build from Linux.

## Security notes

- The PC trusts whatever server it connects to: use the Tailscale address (encrypted) rather than plain LAN `ws://`
  where possible, or `wss://` with the listener's TLS option.
- Hermes gets full shell access on the PC with no confirmation. The tray menu has a "PC actions" switch that
  stops advertising tools (reconnects with `caps.pc` absent) for moments when that is unwanted.
- The device token and the PC API token are separate secrets; never logged.
