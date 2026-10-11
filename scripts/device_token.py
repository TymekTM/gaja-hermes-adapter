#!/usr/bin/env python3
"""Create a token for an IoT voice device and add it to the device token file (mode 600).

    uv run python scripts/device_token.py atom-salon --name Salon

The token is printed once; put it in the device firmware / ESPHome secrets. Default file: [devices] tokens_file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaja.config import Settings  # noqa: E402
from gaja.devices.registry import append_device, generate_token  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("device_id")
    parser.add_argument("--name", default="")
    parser.add_argument("--file", type=Path, default=None, help="token file (default: [devices] tokens_file)")
    args = parser.parse_args()
    path = (args.file or Settings().devices.tokens_file).expanduser()
    token = generate_token()
    try:
        append_device(path, args.device_id, token, args.name)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"added {args.device_id!r} to {path}\ntoken: {token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
