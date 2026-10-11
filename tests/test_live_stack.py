from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

ADAPTER_HEALTH = "http://127.0.0.1:8790/health"


def _health() -> dict:
    try:
        with urllib.request.urlopen(ADAPTER_HEALTH, timeout=3) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError):
        pytest.skip("adapter not running on 127.0.0.1:8790")


@pytest.mark.live
def test_running_adapter_reports_full_stack() -> None:
    health = _health()
    assert health["status"] == "ok"
    assert health["llm"] is True
    assert health["hermes_backend"] == "acp"
    assert health["hermes"] is True
    assert {"piper", "piper-darkman", "paradee", "supertonic"} <= set(health["engines"])
    assert health["engine_by_language"]["pl"] == "piper"
