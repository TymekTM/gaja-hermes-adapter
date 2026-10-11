from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TEXT_BLOCK = re.compile(r"^```text\n(.*?)\n```", re.DOTALL | re.MULTILINE)
_JSON_BLOCK = re.compile(r"^```json\n(.*?)\n```", re.DOTALL | re.MULTILINE)


@dataclass(frozen=True)
class SystemPrompt:
    text: str
    tools: list[dict[str, Any]]


def load_text_block(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    text = _TEXT_BLOCK.search(source)
    if text is None:
        raise ValueError(f"{path}: missing ```text block")
    return text.group(1).strip()


def load_system_prompt(path: Path) -> SystemPrompt:
    tools = [json.loads(match) for match in _JSON_BLOCK.findall(path.read_text(encoding="utf-8"))]
    return SystemPrompt(text=load_text_block(path), tools=tools)
