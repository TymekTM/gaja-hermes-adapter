"""Hermes gateway client: per-profile URLs and keys, (person, profile) sessions, SSE mapping, fallback."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from gaja.hermes.base import HermesEvent
from gaja.hermes.gateway import GatewayHermes, iter_sse, read_env_key
from gaja.household import UNKNOWN, Identity, Person

TYMEK = Identity(Person(id="tymek", name="Tymek", default_profile="default", profiles=["serzant"]), "client")


def sse(*events: tuple[str, dict[str, Any]]) -> bytes:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


class FakeGateway:
    def __init__(self, stream: bytes | None = None, *, stream_status: int = 200) -> None:
        self.requests: list[httpx.Request] = []
        self.sessions = 0
        self.stream = stream if stream is not None else sse(
            ("run.started", {"run_id": "r1"}),
            ("assistant.delta", {"delta": "Masz "}),
            ("tool.started", {"tool_name": "todo"}),
            ("assistant.delta", {"delta": "trzy zadania."}),
            ("assistant.completed", {"content": "Masz trzy zadania."}),
            ("run.completed", {}),
            ("done", {}),
        )
        self.stream_status = stream_status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/api/sessions"):
            self.sessions += 1
            body = json.loads(request.content)
            return httpx.Response(200, json={"id": body["id"]})
        if path.endswith("/chat/stream"):
            if self.stream_status != 200:
                return httpx.Response(self.stream_status, text="gone")
            return httpx.Response(200, content=self.stream, headers={"content-type": "text/event-stream"})
        if "/approval" in path:
            return httpx.Response(200, json={"ok": True})
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)


class FakeAcp:
    def __init__(self) -> None:
        self.tasks: list[str] = []

    async def submit(self, task: str, **kwargs: Any) -> AsyncIterator[HermesEvent]:
        self.tasks.append(task)
        yield HermesEvent("accepted", "")
        yield HermesEvent("result", "z ACP")

    async def health(self) -> bool:
        return True

    def ready(self) -> bool:
        return True

    async def forget(self, session_id: str) -> None:
        return None

    async def aclose(self) -> None:
        return None


@pytest.fixture
def home(tmp_path: Path) -> Path:
    (tmp_path / ".env").write_text("OTHER=1\nAPI_SERVER_KEY=key-default\n")
    (tmp_path / "profiles" / "serzant").mkdir(parents=True)
    (tmp_path / "profiles" / "serzant" / ".env").write_text("export API_SERVER_KEY='key-serzant'\n")
    return tmp_path


def client(home: Path, gateway: Any, **kwargs: Any) -> GatewayHermes:
    return GatewayHermes("http://gw:8642/", home, transport=httpx.MockTransport(gateway), **kwargs)


async def collect(hermes: GatewayHermes, task: str = "co mam dziś?", **kwargs: Any) -> list[HermesEvent]:
    return [e async for e in hermes.submit(task, context=[], session_id="ws1", language="pl", **kwargs)]


def test_read_env_key(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text('# c\nexport A="x y"\nB=\nA=final\n')
    assert read_env_key(env, "A") == "final"
    assert read_env_key(env, "B") is None
    assert read_env_key(tmp_path / "missing", "A") is None


def test_iter_sse() -> None:
    assert iter_sse(["event: x", 'data: {"a": 1}']) == ("x", {"a": 1})
    assert iter_sse([": keepalive"]) is None
    assert iter_sse(["data: plain"]) == ("message", {"text": "plain"})


@pytest.mark.asyncio
async def test_stream_maps_events_and_uses_profile_key(home: Path) -> None:
    gw = FakeGateway()
    hermes = client(home, gw)
    events = await collect(hermes, profile="serzant", identity=TYMEK, instructions="[rozmówca: Tymek]")
    kinds = [e.kind for e in events]
    assert kinds[0] == "accepted" and kinds[-1] == "result"
    assert events[-1].text == "Masz trzy zadania."
    assert [e.data["type"] for e in events if e.kind == "progress"] == ["text_delta", "tool_call", "text_delta"]

    create, stream = gw.requests
    assert create.url.path == "/p/serzant/api/sessions"
    assert stream.url.path.startswith("/p/serzant/api/sessions/gaja-tymek-")
    for request in (create, stream):
        assert request.headers["authorization"] == "Bearer key-serzant"
        assert request.headers["x-hermes-session-key"] == "gaja:tymek"
    body = json.loads(stream.content)
    assert body["system_message"] == "[rozmówca: Tymek]"
    assert body["author"] == {"id": "gaja:tymek", "name": "Tymek"}
    assert "co mam dziś?" in body["message"]
    await hermes.aclose()


@pytest.mark.asyncio
async def test_default_profile_is_unprefixed_and_sessions_are_reused(home: Path) -> None:
    now = [0.0]
    gw = FakeGateway()
    hermes = client(home, gw, idle_s=60.0, clock=lambda: now[0])
    await collect(hermes, identity=TYMEK)
    now[0] = 30.0
    await collect(hermes, identity=TYMEK)
    assert gw.sessions == 1
    assert gw.requests[0].url.path == "/api/sessions"
    assert gw.requests[0].headers["authorization"] == "Bearer key-default"

    await collect(hermes, identity=UNKNOWN)  # another person: own session
    assert gw.sessions == 2
    now[0] = 200.0  # idle timeout
    await collect(hermes, identity=TYMEK)
    assert gw.sessions == 3
    hermes.reset("tymek")
    await collect(hermes, identity=TYMEK)
    assert gw.sessions == 4
    await hermes.aclose()


@pytest.mark.asyncio
async def test_approval_requests_are_denied(home: Path) -> None:
    gw = FakeGateway(sse(
        ("approval.request", {"run_id": "r9", "request_id": "q1", "command": "rm"}),
        ("assistant.completed", {"content": "Nie mogę tego zrobić."}),
        ("done", {}),
    ))
    hermes = client(home, gw)
    events = await collect(hermes)
    assert any(e.data and e.data.get("type") == "permission_cancelled" for e in events)
    approval = next(r for r in gw.requests if "/approval" in r.url.path)
    assert approval.url.path == "/v1/runs/r9/approval"
    assert json.loads(approval.content) == {"choice": "deny", "request_id": "q1"}
    await hermes.aclose()


@pytest.mark.asyncio
async def test_run_failure_is_an_error(home: Path) -> None:
    hermes = client(home, FakeGateway(sse(("run.failed", {"error": "boom"}))))
    events = await collect(hermes)
    assert events[-1].kind == "error" and "boom" in events[-1].text
    await hermes.aclose()


@pytest.mark.asyncio
async def test_missing_session_is_forgotten(home: Path) -> None:
    gw = FakeGateway(stream_status=404)
    hermes = client(home, gw)
    assert (await collect(hermes))[-1].kind == "error"
    await collect(hermes)
    assert gw.sessions == 2
    await hermes.aclose()


@pytest.mark.asyncio
async def test_fallback_to_acp_for_default_only(home: Path) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    acp = FakeAcp()
    hermes = client(home, down, fallback=acp)
    events = await collect(hermes)
    assert events[-1].text == "z ACP" and acp.tasks == ["co mam dziś?"]
    events = await collect(hermes, profile="serzant")
    assert events[-1].kind == "error" and acp.tasks == ["co mam dziś?"]
    assert hermes.ready()  # ACP still answers default-profile requests
    await hermes.aclose()


@pytest.mark.asyncio
async def test_missing_key_never_sends_a_request(home: Path) -> None:
    gw = FakeGateway()
    hermes = client(home, gw)
    assert hermes.has_key("serzant") and not hermes.has_key("dom")
    events = await collect(hermes, profile="dom")
    assert events[-1].kind == "error" and "API_SERVER_KEY" in events[-1].text
    assert "key-" not in events[-1].text
    assert gw.requests == []
    await hermes.aclose()


@pytest.mark.asyncio
async def test_health(home: Path) -> None:
    hermes = client(home, FakeGateway())
    assert await hermes.health() and hermes.ready()
    await hermes.aclose()
