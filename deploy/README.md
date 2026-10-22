# Gaja systemd user units

Two unit templates, both `WantedBy=default.target`, both run as the current user (`systemctl --user`):

- `gaja-llm.service`: llama-server (LFM2.5-2.6B Q4_K_M) on `127.0.0.1:8791`, all layers on GPU (`-ngl 99`).
  The GGUF path is read from `bench/logs/lfm_gguf_path.txt`.
- `gaja-adapter.service`: `.venv/bin/python -m gaja` on `127.0.0.1:8790`. Starts after `gaja-llm.service`.

Both unit templates contain a `@REPO@` placeholder for the repository root directory.

LLM unit flags were swept end-to-end on 2026-10-22 (`docs/LLM_RESULTS.md` section 11): the deployed
combination is the local optimum — batch defaults beat `-ub 128..2048`, `-fa on` beats off, f16 KV
beats `q8_0`, `-np 2` costs nothing vs `np 1`, and pinning RAM (`--load-mode mlock` in this llama.cpp
build) is harmful on this swap-heavy host. Two gotchas if you ever retune: with `--no-kv-unified`
the `-c` value is divided per slot (`-c 4096 -np 2` = 2048-token slots, too small for 24-turn
history), and one llama-server instance already fills the VRAM left over next to the adapter's
faster-whisper, so never run a second one alongside production.

## Installation

Render and copy the units to `~/.config/systemd/user/` using the install script:

```bash
./scripts/install_units.sh
```

Then reload systemd and enable the services:

```bash
systemctl --user daemon-reload
systemctl --user enable --now gaja-llm gaja-adapter
```

## Operation

```bash
systemctl --user status gaja-llm gaja-adapter
systemctl --user restart gaja-adapter
journalctl --user -u gaja-adapter -f
curl -s http://127.0.0.1:8790/health
```

Boot behaviour: user units start at boot when lingering is on (`loginctl show-user $USER -p Linger` reports `Linger=yes`).
