"""Timer delivery truth (fix round 2): a timer counts as delivered only when a session
actually received it, and pending timers flushed on hello stay silent during quiet hours."""

from __future__ import annotations

import asyncio
import datetime
from typing import Any

import pytest

from gaja.devices.registry import DeviceInfo
from gaja.pipeline import Session
from gaja.server import deliver_timer


class RecordingOutbox:
    """A healthy outbox that records what the session sent."""

    def __init__(self) -> None:
        self.sent: list[Any] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append({"bytes": len(data)})


class DeadOutbox:
    """Raises like a socket whose peer has closed (same wording as the starlette wrapper)."""

    async def send_json(self, payload: dict[str, Any]) -> None:
        raise RuntimeError('Cannot call "send" once a close message has been sent.')

    async def send_bytes(self, data: bytes) -> None:
        raise RuntimeError('Cannot call "send" once a close message has been sent.')


TIMER = {"id": "t1", "label": "zupa", "kind": "timer"}


@pytest.mark.asyncio
async def test_on_timer_reports_whether_the_client_got_it(make_services) -> None:
    """on_timer must report success: True after the WS event was sent, False when the socket
    is dead. Returning None (the old behaviour) makes every send look delivered."""
    services = make_services()
    out = RecordingOutbox()
    live = Session(services, out)
    dead = Session(services, DeadOutbox())

    assert await live.on_timer(TIMER) is True
    assert {"type": "timer", "id": "t1", "label": "zupa", "kind": "timer"} in out.sent
    assert await dead.on_timer(TIMER) is False
    await live.close()
    await dead.close()


@pytest.mark.asyncio
async def test_delivery_needs_a_session_that_actually_received(make_services) -> None:
    """A session object existing is not delivery: its send fails, so the timer must stay
    pending (deliver_timer returns False and nothing was heard)."""
    services = make_services()
    dead = Session(services, DeadOutbox())
    assert await deliver_timer([dead], "", TIMER, None) is False
    await dead.close()


@pytest.mark.asyncio
async def test_delivery_true_when_a_session_received(make_services) -> None:
    services = make_services()
    out = RecordingOutbox()
    live = Session(services, out)
    assert await deliver_timer([live], "", TIMER, None) is True
    assert any(isinstance(m, dict) and m.get("type") == "timer" for m in out.sent)
    await live.close()


@pytest.mark.asyncio
async def test_delivery_mixed_dead_and_live_sessions(make_services) -> None:
    """One live receiver is enough, even next to a dead socket."""
    services = make_services()
    dead = Session(services, DeadOutbox())
    live = Session(services, RecordingOutbox())
    assert await deliver_timer([dead, live], "", TIMER, None) is True
    await dead.close()
    await live.close()


@pytest.mark.asyncio
async def test_delivery_is_bounded_when_a_send_hangs() -> None:
    """A hung send must not stall flush_pending forever: after the short bound the timer
    counts as unheard (False), while the straggler send keeps running in the background."""

    class HangingSession:
        def __init__(self) -> None:
            self.id = "hang1"
            self.device_id = None
            self.is_device = False
            self.last_active = 0.0
            self.released = asyncio.Event()

        async def on_timer(self, timer: dict[str, Any], note: str | None = None, *, speak: bool = True) -> bool:
            await self.released.wait()
            return True

    hang = HangingSession()
    background: set[asyncio.Task[bool]] = set()
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    delivered = await deliver_timer([hang], "", TIMER, None, timeout_s=0.05, background=background)
    took = loop.time() - t0
    hang.released.set()
    await asyncio.gather(*background, return_exceptions=True)
    assert delivered is False
    assert took < 5.0, f"deliver_timer blocked {took:.1f}s on a hung send"


# --------------------------------------------------------------------------
# F2: a pending timer flushed on hello must not speak on a device speaker
# during quiet hours (22:00-07:00). A web/text session may still receive it;
# live-fired timers keep the previous behaviour.
# --------------------------------------------------------------------------

QUIET = "22:00-07:00"
LATE = "restart"  # note carried by pending timers flushed at hello / boot


@pytest.mark.asyncio
async def test_late_timer_skips_device_during_quiet_hours(make_services) -> None:
    services = make_services()
    dev_out = RecordingOutbox()
    dev = Session(services, dev_out, device=DeviceInfo("atom-pokoj"))
    web_out = RecordingOutbox()
    web = Session(services, web_out)

    delivered = await deliver_timer([dev, web], QUIET, TIMER, LATE, quiet_now=datetime.time(23, 0))
    assert delivered is True, "the web session should still receive the late timer"
    assert dev_out.sent == [], "the device speaker must stay silent during quiet hours"
    assert any(isinstance(m, dict) and m.get("type") == "timer" for m in web_out.sent)
    await dev.close()
    await web.close()


@pytest.mark.asyncio
async def test_late_timer_with_only_a_device_connected_stays_unheard(make_services) -> None:
    """Only a device speaker is connected at 23:00: nobody may hear it now, so delivery
    fails and the timer stays pending until quiet hours end."""
    services = make_services()
    dev_out = RecordingOutbox()
    dev = Session(services, dev_out, device=DeviceInfo("atom-pokoj"))

    delivered = await deliver_timer([dev], QUIET, TIMER, LATE, quiet_now=datetime.time(23, 0))
    assert delivered is False
    assert dev_out.sent == []
    await dev.close()


@pytest.mark.asyncio
async def test_late_timer_reaches_the_device_after_quiet_hours(make_services) -> None:
    services = make_services()
    dev_out = RecordingOutbox()
    dev = Session(services, dev_out, device=DeviceInfo("atom-pokoj"))

    delivered = await deliver_timer([dev], QUIET, TIMER, LATE, quiet_now=datetime.time(7, 30))
    assert delivered is True
    assert any(isinstance(m, dict) and m.get("type") == "timer" for m in dev_out.sent)
    await dev.close()


@pytest.mark.asyncio
async def test_live_timer_keeps_the_device_during_quiet_hours(make_services) -> None:
    """Live-fired timers (note is None) keep today's behaviour: the device announces them
    even inside the quiet window."""
    services = make_services()
    dev_out = RecordingOutbox()
    dev = Session(services, dev_out, device=DeviceInfo("atom-pokoj"))

    delivered = await deliver_timer([dev], QUIET, TIMER, None, quiet_now=datetime.time(23, 0))
    assert delivered is True
    assert any(isinstance(m, dict) and m.get("type") == "timer" for m in dev_out.sent)
    await dev.close()
