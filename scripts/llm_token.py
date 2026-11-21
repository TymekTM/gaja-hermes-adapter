#!/usr/bin/env python3
"""Create (or rotate) the bearer token of the shared LLM API (/llm/, docs/LLM_API.md).

    uv run python scripts/llm_token.py            # create it if missing
    uv run python scripts/llm_token.py --rotate   # replace it; old clients get 401 at once

The token goes to [llm_proxy] token_file (mode 600) and is not printed unless --show is given;
services on this machine read the file, a tailnet client gets a copy over a secure channel.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaja.config import Settings  # noqa: E402


def write_token(path: Path) -> str:
    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(token + "\n")
    os.replace(tmp, path)
    return token


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=None, help="token file (default: [llm_proxy] token_file)")
    parser.add_argument("--rotate", action="store_true", help="replace an existing token")
    parser.add_argument("--show", action="store_true", help="print the token")
    args = parser.parse_args()
    path = (args.file or Settings().llm_proxy.token_file).expanduser()
    if path.exists() and not args.rotate:
        print(f"{path} already exists (use --rotate to replace it)")
        return 0
    token = write_token(path)
    print(f"wrote {path} (mode 600)")
    if args.show:
        print(f"token: {token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
