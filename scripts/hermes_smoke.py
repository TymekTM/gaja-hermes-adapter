"""Send one harmless read-only task to the real Hermes ACP worker and print its events with timings."""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from gaja.config import Settings
from gaja.hermes.client import HermesClient

DEFAULT_TASK = "Jaki jest dzisiaj dzień tygodnia? Odpowiedz krótko."


async def main(task: str) -> None:
    cfg = Settings().hermes
    client = HermesClient(command=cfg.command, cwd=Path(cfg.cwd).expanduser(), timeout_s=cfg.timeout_s)
    started = time.perf_counter()
    print(f"{0.0:7.2f}s health={await client.health()}")
    async for event in client.submit(task, context=[], session_id="hermes-smoke"):
        print(f"{time.perf_counter() - started:7.2f}s {event.kind}: {event.text[:300]!r}")
    await client.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    args = parser.parse_args()
    asyncio.run(main(args.task))
