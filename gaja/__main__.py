from __future__ import annotations

import logging

import uvicorn

from gaja.config import Settings
from gaja.server import create_app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    uvicorn.run(create_app(settings), host=settings.server.host, port=settings.server.port, log_level="info")


if __name__ == "__main__":
    main()
