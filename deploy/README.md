# Gaja systemd user units

Two unit templates, both `WantedBy=default.target`, both run as the current user (`systemctl --user`):

- `gaja-llm.service`: llama-server (LFM2.5-2.6B Q4_K_M) on `127.0.0.1:8791`, all layers on GPU (`-ngl 99`).
  The GGUF path is read from `bench/logs/lfm_gguf_path.txt`.
- `gaja-adapter.service`: `.venv/bin/python -m gaja` on `127.0.0.1:8790`. Starts after `gaja-llm.service`.

Both unit templates contain a `@REPO@` placeholder for the repository root directory.

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
