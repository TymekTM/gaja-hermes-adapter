#!/usr/bin/env python3
"""Quality gate (bench/llm_quality.py) against a custom llama-server config (llm-speed workstream).

Starts the server exactly like the production unit (or like a sweep config from
bench/llm_speed_sweep.py CONFIGS), then runs the standard 30-prompt quality pass with the
standard seed/temperature, writing bench/results/llm_quality_<tag>.json as usual.

Only point --port at 8791 while the production gaja-llm unit is down; default 8797.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_server as S  # noqa: E402
import llm_speed_sweep as W  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", default="prod_baseline", help="key of llm_speed_sweep.CONFIGS")
    ap.add_argument("--port", type=int, default=8797)
    args = ap.parse_args()
    if args.config not in W.CONFIGS:
        raise SystemExit(f"unknown config {args.config}; known: {', '.join(W.CONFIGS)}")
    S.PORT = args.port
    S.URL = f"http://127.0.0.1:{args.port}"
    # this worktree has no llama.cpp build; use the same binary the production unit runs
    S.BIN = W.BIN
    S.GGUF = W.GGUF
    # production-matching base (llm_speed_sweep.COMMON carries the nothink template + budget 0)
    S.BASE_ARGS = list(W.COMMON) + list(W.CONFIGS[args.config])
    import llm_quality as Q  # import after patching so its URL/BASE_ARGS pick up the overrides

    sys.argv = ["llm_quality.py", "--tag", args.tag, "--variant", "budget0"]
    Q.main()


if __name__ == "__main__":
    main()
