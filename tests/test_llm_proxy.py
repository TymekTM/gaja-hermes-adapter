import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gaja.api.llm_proxy import create_llm_proxy_router
from gaja.config import LLMConfig, LLMProxyConfig
from gaja.llm.activity import ACTIVITY, LLMActivity
from gaja.llm.client import LLMClient

TOKEN = "s3cret-token"
LOCAL = ("127.0.0.1", 40000)
TAILNET = ("100.90.1.2", 40000)


class Upstream:
    """A fake llama-server: records every request and answers by path."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "lfm3"}]})
        return httpx.Response(200, content=request.content, headers={"content-type": "application/json"})

    def body(self, i: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[i].content)


def _cfg(tmp_path: Path, **kw: Any) -> LLMProxyConfig:
    token = tmp_path / "llm.token"
    token.write_text(TOKEN + "\n")
    kw.setdefault("prefill_chunk_tokens", 0)
    return LLMProxyConfig(enabled=True, token_file=token, **kw)


def _app(cfg: LLMProxyConfig, handler: Any, activity: LLMActivity | None = None, origins=("http://localhost:8790",)):
    app = FastAPI()
    router, _ = create_llm_proxy_router(
        cfg,
        "http://llama",
        lambda origin: origin is None or origin in origins,
        activity=activity or LLMActivity(),
        transport=httpx.MockTransport(handler),
    )
    app.include_router(router)
    return app


def test_loopback_needs_no_token_but_forwarded_and_remote_clients_do(tmp_path: Path) -> None:
    upstream = Upstream()
    app = _app(_cfg(tmp_path), upstream)
    local = TestClient(app, client=LOCAL)
    assert local.get("/llm/v1/models").status_code == 200
    relayed = local.get("/llm/v1/models", headers={"X-Forwarded-For": "100.90.1.2"})
    assert relayed.status_code == 401
    assert relayed.headers["www-authenticate"] == "Bearer"
    assert local.get("/llm/v1/models", headers={"Tailscale-User-Login": "someone"}).status_code == 401
    remote = TestClient(app, client=TAILNET)
    assert remote.get("/llm/v1/models").status_code == 401
    assert remote.get("/llm/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert remote.get("/llm/v1/models", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert len(upstream.requests) == 2


def test_a_web_page_cannot_use_the_loopback_exemption(tmp_path: Path) -> None:
    app = _app(_cfg(tmp_path), Upstream())
    local = TestClient(app, client=LOCAL)
    assert local.post("/llm/tokenize", json={"content": "x"}, headers={"Origin": "https://evil.example"}).status_code == 403
    allowed = local.post(
        "/llm/tokenize",
        json={"content": "x"},
        headers={"Origin": "https://evil.example", "Authorization": f"Bearer {TOKEN}"},
    )
    assert allowed.status_code == 200


def test_missing_token_file_leaves_only_loopback(tmp_path: Path) -> None:
    cfg = LLMProxyConfig(enabled=True, token_file=tmp_path / "absent.token")
    app = _app(cfg, Upstream())
    assert TestClient(app, client=LOCAL).get("/llm/health").status_code == 200
    assert TestClient(app, client=TAILNET).get("/llm/health", headers={"Authorization": "Bearer "}).status_code == 401


def test_generation_is_pinned_to_the_external_slot_and_capped(tmp_path: Path) -> None:
    upstream = Upstream()
    client = TestClient(_app(_cfg(tmp_path, slot=2, max_tokens=512), upstream), client=LOCAL)
    request = {"messages": [{"role": "user", "content": "Zażółć"}], "id_slot": 0, "temperature": 0.1, "x": [1]}
    assert client.post("/llm/v1/chat/completions", json=request).status_code == 200
    assert upstream.body() == {**request, "id_slot": 2, "cache_prompt": True, "max_tokens": 512}
    client.post("/llm/v1/chat/completions", json={**request, "max_tokens": 100, "max_completion_tokens": 9999})
    assert (upstream.body()["max_tokens"], upstream.body()["max_completion_tokens"]) == (100, 512)
    client.post("/llm/v1/completions", json={"prompt": "a", "max_tokens": 99999})
    assert upstream.body()["max_tokens"] == 512
    client.post("/llm/completion", json={"prompt": "a", "n_predict": -1, "id_slot": 1})
    assert (upstream.body()["n_predict"], upstream.body()["id_slot"]) == (512, 2)
    client.post("/llm/completion", json={"prompt": "a", "n_predict": 0})
    assert upstream.body()["n_predict"] == 0
    assert [r.url.path for r in upstream.requests] == [
        "/v1/chat/completions", "/v1/chat/completions", "/v1/completions", "/completion", "/completion",
    ]


def test_non_generation_requests_pass_through_byte_for_byte(tmp_path: Path) -> None:
    upstream = Upstream()
    client = TestClient(_app(_cfg(tmp_path), upstream), client=LOCAL)
    raw = '{"content":  "Zażółć", "with_pieces": true}'.encode()
    response = client.post(
        "/llm/tokenize", content=raw, headers={"Content-Type": "application/json", "Cookie": "gaja_admin=1"}
    )
    assert response.content == raw
    sent = upstream.requests[-1]
    assert sent.content == raw
    assert "cookie" not in sent.headers and "authorization" not in sent.headers
    for path in ("detokenize", "apply-template"):
        assert client.post(f"/llm/{path}", json={}).status_code == 200


def test_only_the_listed_endpoints_are_exposed(tmp_path: Path) -> None:
    upstream = Upstream()
    client = TestClient(_app(_cfg(tmp_path), upstream), client=LOCAL)
    assert client.post("/llm/slots/0?action=erase", json={}).status_code == 404
    assert client.post("/llm/lora-adapters", json=[]).status_code == 404
    assert client.get("/llm/v1/chat/completions").status_code == 404
    assert client.post("/llm/v1/chat/completions", content=b"[1]").status_code == 400
    assert client.post("/llm/v1/chat/completions", content=b"{nope").status_code == 400
    assert upstream.requests == []


def test_unreachable_llama_server_is_a_502(tmp_path: Path) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = TestClient(_app(_cfg(tmp_path), down), client=LOCAL)
    assert client.post("/llm/v1/chat/completions", json={"messages": []}).status_code == 502
    assert client.post("/llm/v1/chat/completions", json={"messages": []}).status_code == 502  # slot released


async def _asgi(app: FastAPI, path: str, body: dict[str, Any], sent: list[dict], on_send=None) -> None:
    payload = json.dumps(body).encode()
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
        "headers": [(b"content-type", b"application/json")], "client": LOCAL, "server": ("127.0.0.1", 8790),
    }
    delivered = False

    async def receive() -> dict:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)
        if on_send is not None:
            await on_send(message)

    await app(scope, receive, send)


def test_sse_chunks_are_relayed_as_they_arrive(tmp_path: Path) -> None:
    async def main() -> None:
        first_relayed = asyncio.Event()

        async def events():
            yield b'data: {"choices":[{"delta":{"content":"Cze"}}]}\n\n'
            await asyncio.wait_for(first_relayed.wait(), 5)  # buffering would deadlock here
            yield b"data: [DONE]\n\n"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=events())

        async def on_send(message: dict) -> None:
            if message["type"] == "http.response.body" and b"Cze" in message.get("body", b""):
                first_relayed.set()

        sent: list[dict] = []
        app = _app(_cfg(tmp_path), handler)
        await _asgi(app, "/llm/v1/chat/completions", {"messages": [], "stream": True}, sent, on_send)
        start = next(m for m in sent if m["type"] == "http.response.start")
        headers = dict(start["headers"])
        assert headers[b"content-type"] == b"text/event-stream"
        assert headers[b"x-accel-buffering"] == b"no"
        chunks = [m["body"] for m in sent if m["type"] == "http.response.body" and m["body"]]
        assert chunks == [b'data: {"choices":[{"delta":{"content":"Cze"}}]}\n\n', b"data: [DONE]\n\n"]

    asyncio.run(main())


def test_external_generations_run_one_at_a_time_and_a_full_queue_is_refused(tmp_path: Path) -> None:
    async def main() -> None:
        release_first = asyncio.Event()
        active = 0
        peak = 0

        async def slow(request: httpx.Request) -> httpx.Response:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if json.loads(request.content)["messages"] == ["first"]:
                await release_first.wait()
            active -= 1
            return httpx.Response(200, json={"ok": True})

        app = _app(_cfg(tmp_path, max_waiting=1, yield_to_gaja_s=0), slow)
        first: list[dict] = []
        second: list[dict] = []
        third: list[dict] = []
        t1 = asyncio.create_task(_asgi(app, "/llm/v1/chat/completions", {"messages": ["first"]}, first))
        await asyncio.sleep(0.05)
        t2 = asyncio.create_task(_asgi(app, "/llm/v1/chat/completions", {"messages": ["second"]}, second))
        await asyncio.sleep(0.05)
        await _asgi(app, "/llm/v1/chat/completions", {"messages": ["third"]}, third)
        refused = next(m for m in third if m["type"] == "http.response.start")
        assert refused["status"] == 503
        assert (b"retry-after", b"10") in refused["headers"]
        assert not second  # still queued behind the first
        release_first.set()
        await asyncio.wait_for(asyncio.gather(t1, t2), 5)
        assert [next(m for m in r if m["type"] == "http.response.start")["status"] for r in (first, second)] == [200, 200]
        assert peak == 1

    asyncio.run(main())


def test_external_generation_waits_for_gaja_to_go_idle(tmp_path: Path) -> None:
    async def main() -> None:
        now = [100.0]
        activity = LLMActivity(clock=lambda: now[0])
        upstream = Upstream()
        app = _app(_cfg(tmp_path, yield_to_gaja_s=4.0, queue_timeout_s=1000), upstream, activity)
        activity.touch()  # Gaja just woke
        sent: list[dict] = []
        task = asyncio.create_task(_asgi(app, "/llm/completion", {"prompt": "a"}, sent))
        await asyncio.sleep(0.2)
        assert upstream.requests == []
        async with activity.request():  # a Gaja LLM call keeps it busy however long it takes
            now[0] += 60
            await asyncio.sleep(0.2)
            assert upstream.requests == []
        now[0] += 3.9
        await asyncio.sleep(0.2)
        assert upstream.requests == []
        now[0] += 0.2
        await asyncio.wait_for(task, 5)
        assert len(upstream.requests) == 1

    asyncio.run(main())


def test_gaja_busy_past_the_queue_timeout_is_a_503_with_retry_after(tmp_path: Path) -> None:
    async def main() -> None:
        activity = LLMActivity()
        upstream = Upstream()
        app = _app(_cfg(tmp_path, yield_to_gaja_s=4.0, queue_timeout_s=0.2), upstream, activity)
        sent: list[dict] = []
        async with activity.request():
            await _asgi(app, "/llm/v1/chat/completions", {"messages": []}, sent)
        start = next(m for m in sent if m["type"] == "http.response.start")
        assert start["status"] == 503
        assert (b"retry-after", b"4") in start["headers"]
        assert upstream.requests == []
        sent.clear()
        await _asgi(app, "/llm/v1/chat/completions", {"messages": []}, sent)  # the slot was released
        assert next(m for m in sent if m["type"] == "http.response.start")["status"] == 503  # still within hold

    asyncio.run(main())


def test_client_hanging_up_mid_stream_closes_upstream_and_frees_the_slot(tmp_path: Path) -> None:
    async def main() -> None:
        closed = asyncio.Event()

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"data: 1\n\n"
                await asyncio.Event().wait()
                yield b""

            async def aclose(self) -> None:
                closed.set()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream())

        async def hang_up(message: dict) -> None:
            if message["type"] == "http.response.body" and message.get("body"):
                raise OSError("client gone")

        app = _app(_cfg(tmp_path, yield_to_gaja_s=0), handler)
        await _asgi(app, "/llm/v1/chat/completions", {"messages": [], "stream": True}, [], hang_up)
        assert closed.is_set()
        again: list[dict] = []
        task = asyncio.create_task(_asgi(app, "/llm/v1/chat/completions", {"messages": [], "stream": True}, again))
        for _ in range(50):
            if any(m["type"] == "http.response.start" for m in again):
                break
            await asyncio.sleep(0.02)
        assert any(m["type"] == "http.response.start" and m["status"] == 200 for m in again)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(main())


def test_activity_counts_requests_and_a_hold_after_them() -> None:
    now = [0.0]
    activity = LLMActivity(clock=lambda: now[0])
    assert not activity.busy(4.0)
    activity.touch()
    now[0] = 3.9
    assert activity.busy(4.0)
    now[0] = 4.1
    assert not activity.busy(4.0)

    async def main() -> None:
        async with activity.request():
            now[0] = 100.0
            assert activity.busy(4.0)
        assert activity.busy(4.0)
        real = LLMActivity()
        real.touch()
        assert await real.wait_idle(4.0, timeout_s=0.1) is False
        assert await real.wait_idle(0.05, timeout_s=1.0) is True

    asyncio.run(main())


def test_gaja_llm_client_requests_mark_activity() -> None:
    seen: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(ACTIVITY.busy(0.0))
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={"content": "x", "prompt": "p⁣gaja-prime⁣"})

    async def main() -> None:
        client = LLMClient(LLMConfig(), tools=[])
        await client.aclose()
        client._http = httpx.AsyncClient(base_url="http://llama", transport=httpx.MockTransport(handler))
        await client.complete("p", timeout_s=1)
        await client.render_chat_prefix([])
        async for _ in client.stream([{"role": "user", "content": "x"}], allow_tools=False):
            pass
        await client.aclose()

    asyncio.run(main())
    assert seen == [True, True, True]
    assert not ACTIVITY.busy(0.0)


class ChunkUpstream(Upstream):
    """Renders every chat prompt to ``n`` tokens and answers the chunked prefill."""

    def __init__(self, n: int, fail_template: bool = False) -> None:
        super().__init__()
        self.n = n
        self.fail_template = fail_template

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/apply-template":
            if self.fail_template:
                return httpx.Response(500, json={"error": "boom"})
            return httpx.Response(200, json={"prompt": "<rendered>"})
        if path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(self.n))})
        if path == "/completion":
            return httpx.Response(200, json={"content": ""})
        return httpx.Response(200, json={"choices": []})


def test_a_long_prompt_is_prefilled_in_chunks_that_stop_before_its_last_token(tmp_path: Path) -> None:
    upstream = ChunkUpstream(300)
    app = _app(_cfg(tmp_path, prefill_chunk_tokens=128), upstream)
    body = {"messages": [{"role": "user", "content": "long"}]}
    assert TestClient(app, client=LOCAL).post("/llm/v1/chat/completions", json=body).status_code == 200
    paths = [r.url.path for r in upstream.requests]
    assert paths == ["/apply-template", "/tokenize", "/completion", "/completion", "/completion", "/v1/chat/completions"]
    assert upstream.body(1) == {"content": "<rendered>", "add_special": True}
    chunks = [upstream.body(i) for i in (2, 3, 4)]
    assert [len(c["prompt"]) for c in chunks] == [128, 256, 299]
    assert all(c["n_predict"] == 0 and c["id_slot"] == 2 and c["cache_prompt"] for c in chunks)
    assert upstream.body()["id_slot"] == 2


def test_short_prompts_and_unrenderable_requests_skip_the_chunked_prefill(tmp_path: Path) -> None:
    short = ChunkUpstream(100)
    client = TestClient(_app(_cfg(tmp_path, prefill_chunk_tokens=128), short), client=LOCAL)
    assert client.post("/llm/v1/chat/completions", json={"messages": []}).status_code == 200
    assert [r.url.path for r in short.requests] == ["/apply-template", "/tokenize", "/v1/chat/completions"]

    broken = ChunkUpstream(1000, fail_template=True)
    client = TestClient(_app(_cfg(tmp_path, prefill_chunk_tokens=128), broken), client=LOCAL)
    assert client.post("/llm/v1/chat/completions", json={"messages": []}).status_code == 200
    assert [r.url.path for r in broken.requests] == ["/apply-template", "/v1/chat/completions"]

    several = ChunkUpstream(1000)
    client = TestClient(_app(_cfg(tmp_path, prefill_chunk_tokens=128), several), client=LOCAL)
    assert client.post("/llm/completion", json={"prompt": ["a", "b"]}).status_code == 200
    assert [r.url.path for r in several.requests] == ["/completion"]


def test_completion_token_prompts_are_chunked_without_tokenizing(tmp_path: Path) -> None:
    upstream = ChunkUpstream(0)
    app = _app(_cfg(tmp_path, prefill_chunk_tokens=128), upstream)
    assert TestClient(app, client=LOCAL).post("/llm/completion", json={"prompt": list(range(200))}).status_code == 200
    assert [len(upstream.body(i)["prompt"]) for i in range(len(upstream.requests))] == [128, 199, 200]


def test_the_chunked_prefill_pauses_while_gaja_is_busy(tmp_path: Path) -> None:
    import time

    activity = LLMActivity()
    upstream = ChunkUpstream(300)
    chunk_times: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/completion":
            chunk_times.append(time.monotonic())
            if len(chunk_times) == 1:
                activity.touch()  # Gaja wakes up right after the first chunk
        return upstream(request)

    app = _app(_cfg(tmp_path, prefill_chunk_tokens=128, yield_to_gaja_s=0.3), handler, activity=activity)
    response = TestClient(app, client=LOCAL).post("/llm/v1/chat/completions", json={"messages": []})
    assert response.status_code == 200
    assert len(chunk_times) == 3
    assert chunk_times[1] - chunk_times[0] >= 0.3
    assert chunk_times[2] - chunk_times[1] < 0.3


def test_n_predict_is_capped_on_every_endpoint_and_cache_prompt_is_forced(tmp_path: Path) -> None:
    upstream = Upstream()
    client = TestClient(_app(_cfg(tmp_path, max_tokens=512), upstream), client=LOCAL)
    for path in ("v1/chat/completions", "v1/completions"):
        client.post(f"/llm/{path}", json={"prompt": "a", "messages": [], "n_predict": -1, "cache_prompt": False})
        assert (upstream.body()["n_predict"], upstream.body()["cache_prompt"]) == (512, True)


def test_more_than_one_completion_per_request_is_refused(tmp_path: Path) -> None:
    upstream = Upstream()
    client = TestClient(_app(_cfg(tmp_path), upstream), client=LOCAL)
    assert client.post("/llm/v1/chat/completions", json={"messages": [], "n": 2}).status_code == 400
    assert client.post("/llm/completion", json={"prompt": "a", "n_cmpl": 3}).status_code == 400
    assert upstream.requests == []
    assert client.post("/llm/v1/chat/completions", json={"messages": [], "n": 1}).status_code == 200
    assert client.post("/llm/v1/chat/completions", json={"messages": [], "n": None}).status_code == 200


def test_bodies_that_cannot_be_re_encoded_are_a_400_and_keep_the_slot(tmp_path: Path) -> None:
    upstream = Upstream()
    client = TestClient(_app(_cfg(tmp_path, max_waiting=1), upstream), client=LOCAL)
    surrogate = b'{"messages": [{"role": "user", "content": "\\ud800"}]}'
    deep = b'{"a":' * 100000 + b"1" + b"}" * 100000
    huge = b'{"prompt": 1' + b"0" * 5000 + b"}"
    for raw in (surrogate, deep, huge, b"\xff\xfe"):
        response = client.post("/llm/v1/chat/completions", content=raw, headers={"Content-Type": "application/json"})
        assert response.status_code == 400, raw[:40]
    assert upstream.requests == []
    assert client.get("/llm/status").json()["active"] == 0
    assert client.post("/llm/v1/chat/completions", json={"messages": []}).status_code == 200


def test_the_prefill_pauses_share_one_queue_timeout(tmp_path: Path) -> None:
    activity = LLMActivity()
    upstream = ChunkUpstream(1000)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/completion":
            activity.touch()  # Gaja wakes after every chunk
        return upstream(request)

    cfg = _cfg(tmp_path, prefill_chunk_tokens=128, yield_to_gaja_s=0.2, queue_timeout_s=0.5)
    response = TestClient(_app(cfg, handler, activity=activity), client=LOCAL).post(
        "/llm/v1/chat/completions", json={"messages": []}
    )
    assert response.status_code == 503
    assert sum(r.url.path == "/completion" for r in upstream.requests) <= 3


def test_a_failing_upstream_close_still_frees_the_slot(tmp_path: Path) -> None:
    async def main() -> None:
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"data: 1\n\n"

            async def aclose(self) -> None:
                raise RuntimeError("close failed")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream())

        app = _app(_cfg(tmp_path, yield_to_gaja_s=0), handler)
        for _ in range(2):  # the second one would hang on the semaphore if the first leaked it
            await asyncio.wait_for(
                asyncio.gather(
                    _asgi(app, "/llm/v1/chat/completions", {"messages": [], "stream": True}, []),
                    return_exceptions=True,
                ),
                5,
            )

    asyncio.run(main())


def test_an_unreadable_token_file_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gaja.api.llm_proxy import TokenFile

    path = tmp_path / "t"
    path.write_text(TOKEN)
    tokens = TokenFile(path)
    real = Path.read_text

    def denied(self: Path, *a: Any, **kw: Any) -> str:
        raise PermissionError(13, "denied")

    monkeypatch.setattr(Path, "read_text", denied)
    assert tokens.get() == ""
    monkeypatch.setattr(Path, "read_text", real)
    assert tokens.get() == TOKEN
