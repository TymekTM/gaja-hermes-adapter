# Gaja systemd user units

Two units, both `WantedBy=default.target`, both run as the current user (`systemctl --user`):

- `gaja-llm.service`: llama-server (LFM2.5-2.6B Q4_K_M) on `127.0.0.1:8791`, all layers on the GPU (`-ngl 99`).
  The GGUF path is read from `bench/logs/lfm_gguf_path.txt`. The GPU is shared with other workloads; if the model fails to load, lower `-ngl` in the unit.
- `gaja-adapter.service`: `.venv/bin/python -m gaja` on `127.0.0.1:8790`. Starts after `gaja-llm.service`
  (`Wants=`, so it still starts if the LLM is down and reports `llm: false` on `/health`).

Both units use `Restart=on-failure`.

Install or update (repeat after editing a unit):

```bash
systemctl --user link  "$PWD/deploy/gaja-llm.service"
systemctl --user link  "$PWD/deploy/gaja-adapter.service"
systemctl --user daemon-reload
systemctl --user enable --now gaja-llm gaja-adapter
```

Operate:

```bash
systemctl --user status gaja-llm gaja-adapter
systemctl --user restart gaja-adapter
journalctl --user -u gaja-adapter -f
curl -s http://127.0.0.1:8790/health
```

Boot behaviour: both units are enabled. User units start at boot only when lingering is on
(`loginctl show-user $USER -p Linger` should report `Linger=yes`). Linger is currently enabled and was not changed
as part of this deployment.
