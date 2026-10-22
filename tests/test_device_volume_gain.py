"""Outbox gain (server-side speaker volume) and Session wiring."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from gaja.devices.audio import ResamplingOutbox
from gaja.devices.registry import DeviceInfo
from gaja.devices.volume import VolumeStore
from gaja.pipeline import Session
from tests.test_device_protocol import FAST, HELLO, TOKEN


class RecordingOutbox:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any] | bytes] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(bytes(data))


def _pcm(level: int, *, count: int = 16) -> bytes:
    return (np.full(count, level, dtype="<i2")).tobytes()


def _bytes_sent(outbox: RecordingOutbox) -> list[bytes]:
    return [m for m in outbox.sent if isinstance(m, bytes)]


# ---- ResamplingOutbox gain -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_outbox_applies_gain_per_send_so_midstream_changes_take_effect() -> None:
    inner = RecordingOutbox()
    outbox = ResamplingOutbox(inner)
    outbox.configure(16000)

    outbox.gain = 1.0
    await outbox.send_bytes(_pcm(32000))
    assert _bytes_sent(inner)[0] == _pcm(32000)  # unity: bit-exact fast path

    def peak(chunk: bytes) -> int:
        return int(np.frombuffer(chunk, dtype="<i2").max())

    outbox.gain = 0.5
    await outbox.send_bytes(_pcm(32000))
    assert peak(_bytes_sent(inner)[1]) == pytest.approx(32000 * 0.5 / 32768 * 32767, abs=1)

    # A change made between sends (e.g. "ciszej" while music streams in ~4 s slices)
    # applies from the next slice on -- send_bytes reads the gain each call.
    outbox.gain = 0.0
    await outbox.send_bytes(_pcm(32000))
    assert _bytes_sent(inner)[2] == _pcm(0)


@pytest.mark.asyncio
async def test_outbox_gain_keeps_int16_bounds_and_handles_odd_bytes() -> None:
    inner = RecordingOutbox()
    outbox = ResamplingOutbox(inner)
    outbox.configure(16000)
    outbox.gain = 1.0
    await outbox.send_bytes(_pcm(32767) + b"\xff")  # odd trailing byte must not crash the fast path
    await outbox.send_bytes(_pcm(-32768))
    outbox.gain = 0.25
    await outbox.send_bytes(_pcm(-32768))
    sent = _bytes_sent(inner)
    for chunk in sent:
        if len(chunk) % 2 == 0:
            samples = np.frombuffer(chunk, dtype="<i2")
            assert samples.min() >= -32768 and samples.max() <= 32767


# ---- Session wiring --------------------------------------------------------------------------------


def test_session_picks_up_stored_gain_and_can_update_it(make_services) -> None:
    store = VolumeStore(None)
    store.set_level("atom-pokoj", 50)
    services = make_services(cfg={"devices": {**FAST}})
    services.volume_store = store

    session = Session(services, RecordingOutbox(), device=DeviceInfo("atom-pokoj"))
    outbox = session._out
    assert isinstance(outbox, ResamplingOutbox)
    assert outbox.gain == pytest.approx(store.gain("atom-pokoj"))

    session.set_volume_gain(0.5)
    assert outbox.gain == 0.5
    assert session._out.gain == 0.5


@pytest.mark.asyncio
async def test_store_on_change_updates_live_device_sessions(make_services) -> None:
    store = VolumeStore(None)
    services = make_services(cfg={"devices": {**FAST}})
    services.volume_store = store
    session = Session(services, RecordingOutbox(), device=DeviceInfo("atom-pokoj"))

    # The wiring the server installs: every live session of that device gets the new gain.
    store.on_change = lambda device_id, level, gain: [
        s.set_volume_gain(gain) for s in list(services.sessions) if s.device_id == device_id
    ]
    store.set_level("atom-pokoj", 25)
    assert session._out.gain == pytest.approx(store.gain("atom-pokoj"))

    # Another device's change must not touch this session.
    store.set_level("atom-salon", 10)
    assert session._out.gain == pytest.approx(store.gain("atom-pokoj"))
    await session.close()


@pytest.mark.asyncio
async def test_volume_reply_is_spoken_at_the_new_gain(make_services) -> None:
    """End-to-end inside the session: a gain change lands on the audio sent after it."""
    store = VolumeStore(None)
    services = make_services(cfg={"devices": {**FAST}})
    services.volume_store = store
    raw_out = RecordingOutbox()
    session = Session(services, raw_out, device=DeviceInfo("atom-pokoj"))
    await session.handle_text(json.dumps({**HELLO, "device_id": "atom-pokoj", "token": TOKEN}))
    raw_out.sent.clear()

    session.set_volume_gain(0.5)
    from gaja.pipeline import _Turn, TurnMetrics

    session._turns += 1
    turn = _Turn(
        t0=0.0,
        language="pl",
        metrics=TurnMetrics(turn=session._turns, kind="notification", route="sound"),
        out=session._out,
    )
    await session._sound_turn(_pcm(32000, count=1600), 16000)
    chunks = [m for m in raw_out.sent if isinstance(m, bytes)]
    assert chunks
    for chunk in chunks:
        samples = np.frombuffer(chunk, dtype="<i2")
        assert samples.max() <= 16000 + 1  # half gain on a 32000 peak, int16 rounding slack
    await session.close()
