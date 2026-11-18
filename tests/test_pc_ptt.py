"""Protocol v1.14 push-to-talk: a wake with word "push_to_talk" from a caps.ptt device is never verified.

The press itself is the intent (docs/DESKTOP.md, "Push-to-talk"): it is accepted like an unverified
wake with the verdict reason ``push_to_talk``. From a device WITHOUT ``caps.ptt`` the word is
treated like any other word and verified as usual (rejected when the transcript misses the phrase).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gaja.devices import listener as listener_module
from gaja.devices.controller import get_wake_log
from gaja.devices.listener import create_device_app
from gaja.devices.registry import DeviceRegistry
from tests.conftest import FakeSTT
from tests.test_device_protocol import FAST, HELLO, TOKEN
from tests.test_ws_protocol import is_type, read, read_until, silence_frames, speech_frames

CAPS_PTT: dict[str, Any] = {"aec": False, "out_rates": [16000], "preroll_ms": 0, "ptt": True}
# A transcript that never contains the wake phrase, so verification MUST reject it.
NO_PHRASE = "wykorzystanie procesora dwadzieścia procent"


class FixedSTT(FakeSTT):
    def __init__(self, text: str = NO_PHRASE) -> None:
        super().__init__(text)
        self.calls = 0

    async def transcribe(self, pcm, language=None, *, hotwords=None, prompt=None):  # type: ignore[override]
        self.calls += 1
        return await super().transcribe(pcm, language, hotwords=hotwords, prompt=prompt)


@pytest.fixture(autouse=True)
def _no_auth_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(listener_module, "AUTH_FAIL_DELAY_S", 0.0)


@pytest.fixture
def pc_ptt_client(make_services, tmp_path):
    def factory(**kwargs: Any) -> tuple[TestClient, Any]:
        path = tmp_path / "devices.toml"
        path.write_text(f'[devices.atom-1]\ntoken = "{TOKEN}"\n', encoding="utf-8")
        path.chmod(0o600)
        services = make_services(cfg={"devices": {**FAST, "verify_wake": True}}, **kwargs)
        return TestClient(create_device_app(services, DeviceRegistry(path)), client=("127.0.0.1", 50000)), services

    return factory


def _send_ptt_wake(ws: Any) -> None:
    ws.send_text(json.dumps({"type": "wake", "word": "push_to_talk", "score": 1.0, "preroll_ms": 0}))


def test_ptt_wake_from_caps_ptt_device_is_accepted_without_verification(pc_ptt_client) -> None:
    client, _ = pc_ptt_client(stt=FixedSTT())
    with client, client.websocket_connect("/device") as ws:
        ws.send_text(json.dumps({**HELLO, "caps": CAPS_PTT}))
        ready = read(ws)
        assert ready["type"] == "ready"
        read_until(ws, is_type("state"))
        _send_ptt_wake(ws)
        for frame in speech_frames(0.4) + silence_frames(0.6):
            ws.send_bytes(frame)
        accepted = read_until(ws, is_type("wake_accepted"))[-1]
        assert accepted["type"] == "wake_accepted"
        verdict = get_wake_log()[-1]
        assert verdict["accepted"] is True
        assert verdict["reason"] == "push_to_talk"
        assert verdict["word"] == "push_to_talk"


def test_ptt_word_without_caps_ptt_is_still_verified_and_rejected(pc_ptt_client) -> None:
    client, _ = pc_ptt_client(stt=FixedSTT())
    with client, client.websocket_connect("/device") as ws:
        ws.send_text(json.dumps({**HELLO, "caps": {**CAPS_PTT, "ptt": False}}))
        ready = read(ws)
        assert ready["type"] == "ready"
        read_until(ws, is_type("state"))
        _send_ptt_wake(ws)
        for frame in speech_frames(0.4) + silence_frames(0.6):
            ws.send_bytes(frame)
        rejected = read_until(ws, is_type("wake_rejected"))[-1]
        assert rejected["reason"] == "no_phrase"


def test_ptt_wake_skips_stt_entirely(pc_ptt_client) -> None:
    """The press is the intent: not a single verify transcription happens for a PTT wake."""
    stt = FixedSTT()
    client, _ = pc_ptt_client(stt=stt)
    with client, client.websocket_connect("/device") as ws:
        ws.send_text(json.dumps({**HELLO, "caps": CAPS_PTT}))
        read(ws)
        read_until(ws, is_type("state"))
        before = stt.calls
        _send_ptt_wake(ws)
        for frame in silence_frames(0.2):
            ws.send_bytes(frame)
        read_until(ws, is_type("wake_accepted"))
        assert stt.calls == before
