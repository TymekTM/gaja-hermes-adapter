"""Gaja's llama-server shared with other services: a pass-through reverse proxy under /llm/.

Generation requests (OpenAI chat/completions and llama.cpp /completion) are forced into one dedicated
llama-server slot, run one at a time, and start only while Gaja itself is idle, so they never evict
Gaja's router and chat prompt caches nor make a voice turn wait behind a queue. A long prompt is first
evaluated into that slot in small chunks that pause while Gaja is busy (llama-server batches all slots
into one step, so a single big prefill would hold Gaja's calls for seconds); the request itself then
finds its prompt cached. Everything else in a request and a response passes through untouched; SSE
streams are relayed chunk by chunk.

Auth: ``Authorization: Bearer <token>`` (the token file in [llm_proxy]), or a direct loopback client.
Requests relayed by ``tailscale serve`` also arrive from loopback but carry forwarding headers, so they
need the token. See docs/LLM_API.md.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect
from starlette.types import Receive, Scope, Send

from gaja.config import LLMProxyConfig
from gaja.llm.activity import ACTIVITY, LLMActivity

log = logging.getLogger(__name__)

PREFIX = "/llm"
# Path under /llm -> the generation-length field it is capped through.
GENERATE = {
    "v1/chat/completions": "max_tokens",
    "v1/completions": "max_tokens",
    "completion": "n_predict",
}
PASS_POST = {"tokenize", "detokenize", "apply-template"}
PASS_GET = {"health", "v1/models"}

LOOPBACK = {"127.0.0.1", "::1"}
# Set by tailscale serve (and any other reverse proxy) on a request it relays to loopback.
FORWARDED = ("x-forwarded-for", "forwarded", "x-real-ip", "tailscale-user-login")
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "trailers",
    "transfer-encoding", "upgrade", "host", "content-length",
}
# Never relayed upstream: the proxy's own credentials and the adapter's admin cookie.
REQUEST_DROP = HOP_BY_HOP | {"authorization", "cookie"}


def _error(status: int, message: str, *, retry_after: float | None = None) -> JSONResponse:
    headers = {"Retry-After": str(max(1, round(retry_after)))} if retry_after is not None else None
    kind = {401: "authentication_error", 403: "permission_error", 404: "not_found_error"}.get(status, "api_error")
    return JSONResponse({"error": {"message": message, "type": kind, "code": status}}, status, headers=headers)


class TokenFile:
    """The bearer token, re-read when the file changes (rotating it needs no restart)."""

    def __init__(self, path: Path) -> None:
        self._path = path.expanduser()
        self._stamp: tuple[int, int] | None = None
        self._token = ""

    def get(self) -> str:
        try:
            st = self._path.stat()
        except OSError:
            self._stamp, self._token = None, ""
            return ""
        stamp = (st.st_mtime_ns, st.st_size)
        if stamp != self._stamp:
            try:
                self._token = self._path.read_text(encoding="utf-8").strip()
            except OSError:
                self._token, stamp = "", None  # retried next time (fixing permissions keeps mtime/size)
            self._stamp = stamp
        return self._token

    def matches(self, header: str | None) -> bool:
        token = self.get()
        if not token or not header or not header.startswith("Bearer "):
            return False
        return hmac.compare_digest(header[7:].strip().encode(), token.encode())


def direct_loopback(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host in LOOPBACK and not any(name in request.headers for name in FORWARDED)


def capped(body: dict[str, Any], field: str, cap: int) -> None:
    """Clamps the generation length; absent, negative (unlimited) or oversized values become ``cap``.

    llama-server reads ``n_predict`` first on every endpoint, so it is clamped wherever it is present."""
    fields = [field, "max_completion_tokens", "n_predict"] if field == "max_tokens" else [field]
    for name in fields:
        value = body.get(name)
        if name != field and value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > cap:
            body[name] = cap


class _Gate:
    """One external generation at a time (there is one external slot); a bounded FIFO of waiters behind it."""

    def __init__(self, cfg: LLMProxyConfig) -> None:
        self._cfg = cfg
        self._sem: asyncio.Semaphore | None = None
        self.waiting = 0
        self.active = 0

    def sem(self) -> asyncio.Semaphore:
        if self._sem is None:  # created inside the running loop
            self._sem = asyncio.Semaphore(1)
        return self._sem


class _Relay(StreamingResponse):
    """A streamed upstream response whose cleanup runs however the client side ends."""

    def __init__(self, upstream: httpx.Response, headers: dict[str, str], cleanup: Callable[[], Any]) -> None:
        self._upstream = upstream
        self._cleanup = cleanup
        super().__init__(self._chunks(), status_code=upstream.status_code, headers=headers)

    async def _chunks(self) -> AsyncIterator[bytes]:
        if self._upstream.is_stream_consumed:  # an in-memory response (tests' mock transport)
            yield self._upstream.content
            return
        async for chunk in self._upstream.aiter_raw():  # raw: any content-encoding passes through as is
            yield chunk

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        except ClientDisconnect:
            pass  # the client hung up mid-stream; closing upstream stops the generation
        finally:
            await self._cleanup()


def _response_headers(upstream: httpx.Response) -> dict[str, str]:
    headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP}
    if upstream.headers.get("content-type", "").startswith("text/event-stream"):
        headers["X-Accel-Buffering"] = "no"
        headers["Cache-Control"] = "no-cache"
    return headers


def create_llm_proxy_router(
    cfg: LLMProxyConfig,
    upstream_url: str,
    origin_ok: Callable[[str | None], bool],
    activity: LLMActivity = ACTIVITY,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[APIRouter, Callable[[], Any]]:
    """The /llm router and an async close() for its upstream connection pool."""
    router = APIRouter()
    tokens = TokenFile(cfg.token_file)
    gate = _Gate(cfg)
    http = httpx.AsyncClient(
        base_url=upstream_url.rstrip("/"),
        timeout=httpx.Timeout(cfg.timeout_s, connect=5.0),
        limits=httpx.Limits(max_connections=16, max_keepalive_connections=4),
        transport=transport,
    )

    def denied(request: Request) -> Response | None:
        if tokens.matches(request.headers.get("authorization")):
            return None
        if not direct_loopback(request):
            response = _error(401, "missing or invalid bearer token")
            response.headers["WWW-Authenticate"] = "Bearer"
            return response
        if not origin_ok(request.headers.get("origin")):  # a web page may not use the loopback exemption
            return _error(403, "origin not allowed")
        return None

    def upstream_headers(request: Request) -> dict[str, str]:
        return {k: v for k, v in request.headers.items() if k.lower() not in REQUEST_DROP}

    async def send(request: Request, path: str, content: bytes, cleanup: Callable[[], Any]) -> Response:
        upstream_req = http.build_request(
            request.method, f"/{path}", params=request.query_params, headers=upstream_headers(request),
            content=content,
        )
        try:
            upstream = await http.send(upstream_req, stream=True)
        except httpx.HTTPError as exc:
            await cleanup()
            log.warning("llm proxy: llama-server unreachable for /%s: %s", path, exc)
            return _error(502, "llama-server unreachable")
        except BaseException:  # cancelled while llama-server works on a non-streamed request
            await cleanup()
            raise
        closed = False

        async def close() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            try:
                await upstream.aclose()  # mid-stream this drops the connection: llama-server stops generating
            finally:
                await cleanup()

        return _Relay(upstream, _response_headers(upstream), close)

    async def nothing() -> None:
        return None

    @router.get(PREFIX + "/status")
    async def status(request: Request) -> Response:
        if (refusal := denied(request)) is not None:
            return refusal
        return JSONResponse(
            {
                "slot": cfg.slot,
                "max_concurrent": 1,
                "active": gate.active,
                "waiting": gate.waiting,
                "max_waiting": cfg.max_waiting,
                "gaja_busy": activity.busy(cfg.yield_to_gaja_s),
                "max_tokens": cfg.max_tokens,
            }
        )

    @router.api_route(PREFIX + "/{path:path}", methods=["GET", "POST"])
    async def relay(path: str, request: Request) -> Response:
        if (refusal := denied(request)) is not None:
            return refusal
        method = request.method
        if (method == "GET" and path in PASS_GET) or (method == "POST" and path in PASS_POST):
            return await send(request, path, await request.body(), nothing)
        field = GENERATE.get(path) if method == "POST" else None
        if field is None:
            return _error(404, f"{method} /llm/{path} is not exposed")
        try:
            body = json.loads(await request.body() or b"null")
        except (ValueError, RecursionError):  # JSONDecodeError, UnicodeDecodeError, huge ints
            return _error(400, "the request body must be a JSON object")
        if not isinstance(body, dict):
            return _error(400, "the request body must be a JSON object")
        # n > 1 would run the extra completions in other free slots, i.e. Gaja's.
        if any(body.get(name) not in (None, 1) for name in ("n", "n_cmpl")):
            return _error(400, "only one completion per request (n = 1)")
        body["id_slot"] = cfg.slot
        body["cache_prompt"] = True  # the request must reuse what the chunked prefill evaluated
        capped(body, field, cfg.max_tokens)
        try:
            content = json.dumps(body, ensure_ascii=False).encode("utf-8")
        except (UnicodeEncodeError, ValueError, RecursionError):  # e.g. a lone surrogate escape
            return _error(400, "the request body is not valid UTF-8 JSON")
        return await generate(request, path, body, content)

    async def prompt_tokens(path: str, body: dict[str, Any]) -> list[int] | None:
        """The request's prompt as llama-server will tokenize it, or None when it cannot tell."""
        if path == "v1/chat/completions":
            response = await http.post("/apply-template", json=body)
            response.raise_for_status()
            prompt = response.json()["prompt"]
        else:
            prompt = body.get("prompt")
            if isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt):
                return prompt
        if not isinstance(prompt, str):
            return None  # several prompts, multimodal parts...: the request prefills as is
        response = await http.post("/tokenize", json={"content": prompt, "add_special": True})
        response.raise_for_status()
        return response.json()["tokens"]

    async def prefill(request: Request, path: str, body: dict[str, Any]) -> Response | None:
        """Evaluates a long prompt into the external slot chunk by chunk, yielding to Gaja in between.

        Stops at the prompt's last token, so the request itself extends the cached prefix (LFM3's
        recurrent layers resume a cache only by extending it). Any failure just leaves the request
        to prefill as is. Returns a response only when the request has to end here."""
        chunk = cfg.prefill_chunk_tokens
        if chunk <= 0:
            return None
        try:
            ids = await prompt_tokens(path, body)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            log.debug("llm proxy: no chunked prefill for /%s: %s", path, exc)
            return None
        if ids is None or len(ids) <= chunk + 1:
            return None
        deadline = time.monotonic() + cfg.queue_timeout_s  # all pauses of one request together
        for end in [*range(chunk, len(ids) - 1, chunk), len(ids) - 1]:
            if not await activity.wait_idle(cfg.yield_to_gaja_s, max(0.0, deadline - time.monotonic())):
                return _error(503, "Gaja is busy; retry shortly", retry_after=cfg.yield_to_gaja_s)
            if await request.is_disconnected():
                return Response(status_code=499)
            part = {"prompt": ids[:end], "n_predict": 0, "cache_prompt": True, "id_slot": cfg.slot}
            try:
                response = await http.post("/completion", json=part)
            except httpx.HTTPError as exc:
                log.debug("llm proxy: chunked prefill stopped: %s", exc)
                return None
            if response.status_code != 200:
                return None
        return None

    async def generate(request: Request, path: str, body: dict[str, Any], content: bytes) -> Response:
        if gate.waiting >= cfg.max_waiting:
            return _error(503, "external LLM queue is full", retry_after=10)
        started = time.monotonic()
        deadline = started + cfg.queue_timeout_s
        sem = gate.sem()
        gate.waiting += 1
        try:
            async with asyncio.timeout(cfg.queue_timeout_s):
                await sem.acquire()
        except TimeoutError:
            return _error(503, "timed out waiting for the external LLM slot", retry_after=10)
        finally:
            gate.waiting -= 1
        released = False

        async def release() -> None:
            nonlocal released
            if not released:
                released = True
                gate.active -= 1
                sem.release()
                log.info(
                    "llm proxy: /%s from %s done after %.1f s (waited %.1f s)",
                    path, request.client.host if request.client else "?", time.monotonic() - started, waited,
                )

        gate.active += 1
        waited = 0.0
        try:
            # Gaja first: a voice turn is several LLM calls in a row, so wait for a quiet stretch.
            if not await activity.wait_idle(cfg.yield_to_gaja_s, max(0.0, deadline - time.monotonic())):
                await release()
                return _error(503, "Gaja is busy; retry shortly", retry_after=cfg.yield_to_gaja_s)
            waited = time.monotonic() - started
            if await request.is_disconnected():
                await release()
                return Response(status_code=499)
            if (ended := await prefill(request, path, body)) is not None:
                await release()
                return ended
        except BaseException:
            await release()
            raise
        return await send(request, path, content, release)

    return router, http.aclose
