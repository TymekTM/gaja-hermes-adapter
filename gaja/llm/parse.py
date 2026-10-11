from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

TOOL_START = "<|tool_call_start|>"
TOOL_END = "<|tool_call_end|>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


def _partial_suffix(data: str, token: str) -> int:
    for size in range(min(len(token) - 1, len(data)), 0, -1):
        if data.endswith(token[:size]):
            return size
    return 0


def parse_pythonic_calls(source: str) -> list[ToolCall]:
    tree = ast.parse(source.strip(), mode="eval")
    if not isinstance(tree.body, ast.List):
        raise ValueError("expected a list of calls")
    calls: list[ToolCall] = []
    for node in tree.body.elts:
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            raise ValueError("expected a call expression")
        arguments = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
        calls.append(ToolCall(node.func.id, arguments))
    return calls


class ThinkFilter:
    """Removes <think>...</think> reasoning from a streamed reply.

    llama-server with --reasoning-budget 0 already strips reasoning, so by default the reply starts outside the block.
    """

    def __init__(self, *, starts_inside: bool = False, max_reasoning_chars: int = 400) -> None:
        self._inside = starts_inside
        self._max_reasoning_chars = max_reasoning_chars
        self._held = ""
        self._pending = ""

    def feed(self, delta: str) -> str:
        data = self._pending + delta
        self._pending = ""
        out: list[str] = []
        while data:
            if self._inside:
                end = data.find(THINK_CLOSE)
                if end >= 0:
                    self._held = ""
                    self._inside = False
                    data = data[end + len(THINK_CLOSE) :]
                    continue
                keep = _partial_suffix(data, THINK_CLOSE)
                self._held += data[: len(data) - keep]
                data = data[len(data) - keep :]
                if len(self._held) <= self._max_reasoning_chars:
                    self._pending = data
                    break
                out.append(self._held)
                self._held = ""
                self._inside = False
                continue
            opened = data.find(THINK_OPEN)
            closed = data.find(THINK_CLOSE)
            cuts = [i for i in (opened, closed) if i >= 0]
            if cuts:
                cut = min(cuts)
                out.append(data[:cut])
                if cut == opened:
                    self._inside = True
                    data = data[cut + len(THINK_OPEN) :]
                else:
                    data = data[cut + len(THINK_CLOSE) :]
                continue
            keep = max(_partial_suffix(data, THINK_OPEN), _partial_suffix(data, THINK_CLOSE))
            out.append(data[: len(data) - keep])
            self._pending = data[len(data) - keep :]
            break
        return "".join(out)

    def flush(self) -> str:
        """A reply that ends while still inside the block never closed it, so the held text is the reply."""
        text = self._held + self._pending
        self._held = ""
        self._pending = ""
        self._inside = False
        return text


class PythonicCallSplitter:
    """Separates streamed content into spoken text and <|tool_call_start|>[f(...)]<|tool_call_end|> calls."""

    def __init__(self) -> None:
        self._in_call = False
        self._call = ""
        self._pending = ""

    def feed(self, delta: str) -> tuple[str, list[ToolCall]]:
        data = self._pending + delta
        self._pending = ""
        text: list[str] = []
        calls: list[ToolCall] = []
        while data:
            if self._in_call:
                end = data.find(TOOL_END)
                if end < 0:
                    keep = _partial_suffix(data, TOOL_END)
                    self._call += data[: len(data) - keep]
                    self._pending = data[len(data) - keep :]
                    break
                self._call += data[:end]
                data = data[end + len(TOOL_END) :]
                self._in_call = False
                calls.extend(self._parse_call())
                continue
            start = data.find(TOOL_START)
            if start >= 0:
                text.append(data[:start])
                data = data[start + len(TOOL_START) :]
                self._in_call = True
                continue
            keep = _partial_suffix(data, TOOL_START)
            text.append(data[: len(data) - keep])
            self._pending = data[len(data) - keep :]
            break
        return "".join(text), calls

    def finish(self) -> tuple[str, list[ToolCall]]:
        if self._in_call:
            self._in_call = False
            self._pending = ""
            return "", self._parse_call()
        pending, self._pending = self._pending, ""
        return pending, []

    def _parse_call(self) -> list[ToolCall]:
        source, self._call = self._call, ""
        try:
            return parse_pythonic_calls(source)
        except (SyntaxError, ValueError) as exc:
            log.warning("dropping unparseable tool call %r: %s", source, exc)
            return []


class NativeCallAccumulator:
    """Assembles llama-server's streamed delta.tool_calls, which arrive in fragments per index."""

    def __init__(self) -> None:
        self._slots: dict[int, dict[str, str]] = {}

    def feed(self, tool_calls: list[dict[str, Any]]) -> None:
        for fragment in tool_calls:
            slot = self._slots.setdefault(int(fragment.get("index", 0)), {"name": "", "arguments": ""})
            function = fragment.get("function") or {}
            slot["name"] += function.get("name") or ""
            slot["arguments"] += function.get("arguments") or ""

    def finish(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for slot in self._slots.values():
            try:
                arguments = json.loads(slot["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                log.warning("dropping tool call %s with bad arguments: %s", slot["name"], exc)
                continue
            if slot["name"] and isinstance(arguments, dict):
                calls.append(ToolCall(slot["name"], arguments))
        self._slots.clear()
        return calls
