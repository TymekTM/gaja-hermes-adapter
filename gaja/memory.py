from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)


HOME = "home"


class MemoryStore:
    """Facts Gaja remembers. Each fact has a scope: "home" (shared by everyone) or a person id."""

    def __init__(self, file_path: Path | str = "data/memory.json", on_change: Callable[[], None] | None = None) -> None:
        self.file_path = Path(file_path)
        self.on_change = on_change
        self._facts: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.file_path.exists():
            self._facts = []
            return
        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "facts" in data:
                self._facts = list(data["facts"])
            elif isinstance(data, list):
                self._facts = list(data)
            else:
                self._facts = []
        except Exception as exc:
            log.warning("Could not read memory file %s: %s", self.file_path, exc)
            self._facts = []

    def _save(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file_path.with_suffix(".tmp")
        payload = {"facts": self._facts}
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.file_path)
        if self.on_change:
            try:
                self.on_change()
            except Exception as exc:
                log.warning("Memory on_change callback failed: %s", exc)

    def list_facts(self, scopes: set[str] | None = None) -> list[dict[str, Any]]:
        facts = [{**f, "scope": f.get("scope", HOME)} for f in self._facts]
        return facts if scopes is None else [f for f in facts if f["scope"] in scopes]

    def add_fact(self, text: str, scope: str = HOME) -> dict[str, Any]:
        text_clean = text.strip()
        tz = ZoneInfo("Europe/Warsaw")
        fact_id = f"m_{uuid.uuid4().hex[:8]}"
        fact = {
            "id": fact_id,
            "text": text_clean,
            "created": datetime.datetime.now(tz).isoformat(),
            "scope": scope,
        }
        self._facts.append(fact)
        self._save()
        return fact

    def delete_fact(self, fact_id: str) -> bool:
        before = len(self._facts)
        self._facts = [f for f in self._facts if f.get("id") != fact_id]
        if len(self._facts) < before:
            self._save()
            return True
        return False

    def forget_by_query(self, query: str, scopes: set[str] | None = None) -> list[dict[str, Any]]:
        words = [w.lower() for w in query.strip().split() if len(w) >= 3]
        if not words:
            words = [query.strip().lower()]

        stems = set(words)
        for w in words:
            if len(w) >= 4:
                for ending in ("ie", "em", "ów", "ow", "om", "y", "a", "e", "ę", "ą", "u", "i", "o"):
                    if w.endswith(ending) and len(w) - len(ending) >= 4:
                        stems.add(w[:-len(ending)])
                        break

        deleted: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []
        for fact in self._facts:
            if scopes is not None and fact.get("scope", HOME) not in scopes:
                remaining.append(fact)
                continue
            f_text = fact.get("text", "").lower()
            fact_words = [fw.strip(" .,!?;:") for fw in f_text.split()]
            matched = False
            for s in stems:
                if s in f_text:
                    matched = True
                    break
                for fw in fact_words:
                    if len(fw) >= 4 and (fw.startswith(s) or s.startswith(fw[:4])):
                        matched = True
                        break
                if matched:
                    break

            if matched:
                deleted.append(fact)
            else:
                remaining.append(fact)

        if deleted:
            self._facts = remaining
            self._save()
        return deleted

    def format_prompt_block(self, person: str | None = None, max_chars: int = 600) -> str:
        """Compact facts block for the system prompt: the person's facts, then the household's.

        ``person`` None (unknown speaker) gets only the household facts. Capped at max_chars."""
        blocks = []
        if person is not None:
            blocks.append(("\n\nFakty o rozmówcy:\n", self.list_facts({person})))
        blocks.append(("\n\nFakty o domu:\n", self.list_facts({HOME})))
        out = ""
        budget = max_chars
        for header, facts in blocks:
            lines = []
            used = len(header)
            for f in facts:
                line = f"- {f.get('text', '')}\n"
                if used + len(line) > budget:
                    break
                lines.append(line)
                used += len(line)
            if lines:
                out += header + "".join(lines)
                budget -= used
        return out


_default_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    global _default_store
    if _default_store is None:
        _default_store = MemoryStore()
    return _default_store


def set_memory_store(store: MemoryStore) -> None:
    global _default_store
    _default_store = store
