#!/usr/bin/env python3
"""Set the admin panel password (asked twice, never echoed). Logs out every open admin session.

    uv run python scripts/admin_password.py

Default file: [admin] credentials_file (mode 600). Works on a running adapter without a restart.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaja.admin.auth import write_credentials  # noqa: E402
from gaja.config import Settings  # noqa: E402

MIN_LENGTH = 10


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=None, help="credentials file (default: [admin] credentials_file)")
    args = parser.parse_args()
    path = (args.file or Settings().admin.credentials_file).expanduser()
    password = getpass.getpass("New admin password: ")
    if len(password) < MIN_LENGTH:
        print(f"error: use at least {MIN_LENGTH} characters", file=sys.stderr)
        return 1
    if getpass.getpass("Repeat: ") != password:
        print("error: passwords differ", file=sys.stderr)
        return 1
    write_credentials(path, password)
    print(f"saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
