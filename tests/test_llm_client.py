from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from gaja.config import LLMConfig
from gaja.llm.client import LLMClient


def client_with(handler: Any, cfg: LLMConfig) -> LLMClient:
    llm = LLMClient(cfg, tools=[])
    llm._http = httpx.AsyncClient(base_url=cfg.base_url, transport=httpx.MockTransport(handler))
    return llm


@pytest.mark.asyncio
async def test_complete_without_slot_leaves_server_choice() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"completion_probabilities": []})

    llm = client_with(handler, LLMConfig())
    await llm.complete("p", timeout_s=5.0)
    assert "id_slot" not in sent[0]


@pytest.mark.asyncio
async def test_complete_with_slot_pins_it() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"completion_probabilities": []})

    llm = client_with(handler, LLMConfig())
    await llm.complete("p", timeout_s=5.0, slot=1)
    assert sent[0]["id_slot"] == 1


@pytest.mark.asyncio
async def test_render_chat_prefix_cuts_at_the_next_user_turn() -> None:
    from gaja.llm.client import PRIME_MARK

    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        rendered = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in body["messages"])
        return httpx.Response(200, json={"prompt": rendered + "<|im_start|>assistant\n"})

    llm = client_with(handler, LLMConfig())
    history = [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    prefix = await llm.render_chat_prefix(history)
    assert sent[0]["messages"][-1] == {"role": "user", "content": PRIME_MARK}
    assert prefix.endswith("<|im_start|>assistant\na<|im_end|>\n<|im_start|>user\n")
    assert PRIME_MARK not in prefix
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_chat_stream_turns_thinking_off_and_drops_a_stray_close() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        chunks = [{"choices": [{"delta": {"content": c}}]} for c in ("</think>", "Dzień dobry.")]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body)

    llm = client_with(handler, LLMConfig())
    text = "".join([e.text async for e in llm.stream([{"role": "user", "content": "hej"}], allow_tools=False)])
    # The LFM3 template opens <think> by default; the reasoning would then be spoken as the reply.
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert text == "Dzień dobry."
