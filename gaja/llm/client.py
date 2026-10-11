from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

import httpx

from gaja.config import LLMConfig
from gaja.llm.parse import NativeCallAccumulator, PythonicCallSplitter, ThinkFilter, ToolCall

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextDelta:
    text: str


LLMEvent = TextDelta | ToolCall


class LLMUnavailable(RuntimeError):
    pass


# Placeholder user message for rendering a history prefix; never sent as a real turn.
PRIME_MARK = "\u2063gaja-prime\u2063"


class LLMClient:
    def __init__(self, cfg: LLMConfig, tools: list[dict[str, Any]]) -> None:
        self._cfg = cfg
        self._tools = tools
        self._chat_prefix: tuple[list[dict[str, Any]], str] | None = None
        self._http = httpx.AsyncClient(base_url=cfg.base_url, timeout=cfg.timeout_s)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def health(self) -> bool:
        try:
            response = await self._http.get("/health", timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def warm_up(self, system_text: str) -> None:
        messages = [{"role": "system", "content": system_text}, {"role": "user", "content": "Dzień dobry."}]
        async with aclosing(self.stream(messages, allow_tools=True)) as events:
            async for _ in events:
                break

    async def complete(self, prompt: str, *, timeout_s: float, slot: int | None = None) -> dict[str, Any]:
        return await self._completion(
            {"prompt": prompt, "n_predict": 1, "n_probs": 10, "temperature": 0, "cache_prompt": True},
            timeout_s=timeout_s,
            slot=slot,
        )

    async def generate(
        self, prompt: str, *, n_predict: int, grammar: str, timeout_s: float, slot: int | None = None
    ) -> str:
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0,
            "grammar": grammar,
            "cache_prompt": True,
            "stop": ["\n"],
        }
        return str((await self._completion(payload, timeout_s=timeout_s, slot=slot)).get("content", ""))

    async def render_chat_prefix(self, messages: list[dict[str, Any]]) -> str:
        """The rendering of ``messages`` (with tools) up to and including the next user header.

        Rendered with a placeholder user turn appended, because llama-server renders a trailing
        assistant message as a prefill (``<think></think>…`` without ``<|im_end|>``); with the user
        turn it renders exactly as in the next chat request, so the cut is a true token prefix.
        """
        payload: dict[str, Any] = {"messages": [*messages, {"role": "user", "content": PRIME_MARK}]}
        if self._tools:
            payload["tools"] = self._tools
        try:
            response = await self._http.post("/apply-template", json=payload, timeout=10.0)
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"llama-server apply-template failed: {exc}") from exc
        if response.status_code != 200:
            raise LLMUnavailable(f"llama-server answered HTTP {response.status_code}")
        prompt = str(response.json().get("prompt", ""))
        cut = prompt.rfind(PRIME_MARK)
        if cut < 0:
            raise LLMUnavailable("llama-server apply-template dropped the placeholder turn")
        return prompt[:cut]

    async def prime_chat(self, messages: list[dict[str, Any]]) -> None:
        """Evaluate ``messages`` (system + tools + history) into the chat slot's cache.

        Used at session start and after every turn, so the next request prefills only the new user
        message instead of re-tokenized history (REVIEW M5).
        """
        if self._chat_prefix is None or self._chat_prefix[0] != messages:
            self._chat_prefix = (messages, await self.render_chat_prefix(messages))
        await self._completion(
            {"prompt": self._chat_prefix[1], "n_predict": 0, "cache_prompt": True},
            timeout_s=self._cfg.timeout_s,
            slot=self._cfg.chat_slot,
        )

    async def prime(self, prompt: str, *, timeout_s: float, slot: int | None = None) -> None:
        await self._completion(
            {"prompt": prompt, "n_predict": 0, "cache_prompt": True}, timeout_s=timeout_s, slot=slot
        )

    async def _completion(
        self, payload: dict[str, Any], *, timeout_s: float, slot: int | None = None
    ) -> dict[str, Any]:
        if slot is not None and slot >= 0:
            payload["id_slot"] = slot
        try:
            response = await self._http.post("/completion", json=payload, timeout=timeout_s)
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"llama-server completion failed: {exc}") from exc
        if response.status_code != 200:
            raise LLMUnavailable(f"llama-server answered HTTP {response.status_code}")
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"llama-server returned invalid JSON: {exc}") from exc

    async def stream(self, messages: list[dict[str, Any]], *, allow_tools: bool) -> AsyncIterator[LLMEvent]:
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "stream": True,
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
            "cache_prompt": True,
        }
        if self._cfg.chat_slot >= 0:
            payload["id_slot"] = self._cfg.chat_slot
        # Tools are sent on every request, even when the caller will ignore a call (allow_tools=False):
        # the chat template renders them into the system message, so leaving them out would change the
        # prompt prefix and throw away llama-server's prompt cache.
        if self._tools:
            payload["tools"] = self._tools
        think = ThinkFilter()
        pythonic = PythonicCallSplitter() if self._cfg.tool_strategy == "pythonic" else None
        native = NativeCallAccumulator() if self._cfg.tool_strategy == "native" else None
        try:
            async with self._http.stream("POST", "/v1/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise LLMUnavailable(f"llama-server answered HTTP {response.status_code}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if "choices" not in chunk:
                        raise LLMUnavailable(f"llama-server error: {chunk.get('error', chunk)}")
                    delta = chunk["choices"][0].get("delta") or {}
                    if native is not None and delta.get("tool_calls"):
                        native.feed(delta["tool_calls"])
                    content = delta.get("content")
                    if not content:
                        continue
                    text = think.feed(content)
                    if pythonic is None:
                        if text:
                            yield TextDelta(text)
                        continue
                    spoken, calls = pythonic.feed(text)
                    if spoken:
                        yield TextDelta(spoken)
                    for call in calls:
                        yield call
                tail = think.flush()
                if pythonic is not None:
                    spoken, calls = pythonic.feed(tail)
                    rest, rest_calls = pythonic.finish()
                    if spoken + rest:
                        yield TextDelta(spoken + rest)
                    for call in (*calls, *rest_calls):
                        yield call
                elif tail:
                    yield TextDelta(tail)
                if native is not None:
                    for call in native.finish():
                        yield call
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise LLMUnavailable(f"llama-server request failed: {exc}") from exc
