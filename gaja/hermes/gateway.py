"""Hermes over the gateway's HTTP API: one shared gateway process serves every Hermes profile.

Requests for profile P go to `<url>/p/P/...` (the default profile is unprefixed) with that profile's
own API_SERVER_KEY, read at runtime from its `.env` (`~/.hermes/.env`, `~/.hermes/profiles/P/.env`);
the key is never logged or stored elsewhere. One Hermes session per (person, profile): a new one starts
after `idle_s` without a request, or on reset ("nowy temat"). A turn streams over
`POST /api/sessions/{id}/chat/stream` (SSE); closing the stream interrupts the run on the Hermes side.

If the gateway is unreachable, default-profile requests fall back to the ACP worker.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from gaja.hermes.base import HermesClient, HermesEvent
from gaja.hermes.client import clean_spoken_hint, detect_language, format_voice_prompt, with_context
from gaja.household import Identity

log = logging.getLogger(__name__)

DEFAULT_PROFILE = "default"


class GatewayUnavailable(RuntimeError):
    pass


def read_env_key(path: Path, name: str) -> str | None:
    """Value of `name` in a dotenv file (KEY=VALUE, optional `export` and quotes), else None."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    value = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, rest = line.partition("=")
        if sep and key.strip() == name:
            rest = rest.strip()
            if len(rest) >= 2 and rest[0] == rest[-1] and rest[0] in "\"'":
                rest = rest[1:-1]
            value = rest or None
    return value


def profile_env_path(hermes_home: Path, profile: str) -> Path:
    home = hermes_home.expanduser()
    return home / ".env" if profile == DEFAULT_PROFILE else home / "profiles" / profile / ".env"


def list_profiles(hermes_home: Path) -> list[str]:
    """Hermes profile names on disk (directory names only; nothing inside them is read)."""
    root = hermes_home.expanduser() / "profiles"
    try:
        names = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError:
        names = []
    return [DEFAULT_PROFILE, *names]


def iter_sse(lines: list[str]) -> tuple[str, dict[str, Any]] | None:
    """One SSE frame (its lines, without the blank terminator) -> (event, data), or None for comments."""
    event, data = "message", []
    for line in lines:
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if not data:
        return None
    try:
        payload = json.loads("\n".join(data))
    except json.JSONDecodeError:
        payload = {"text": "\n".join(data)}
    return event, payload if isinstance(payload, dict) else {"value": payload}


class GatewayHermes:
    def __init__(
        self,
        url: str,
        hermes_home: Path,
        *,
        key_var: str = "API_SERVER_KEY",
        timeout_s: float = 300.0,
        idle_s: float = 1800.0,
        fallback: HermesClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        self.url = url.rstrip("/")
        self.hermes_home = hermes_home.expanduser()
        self.key_var = key_var
        self.timeout_s = timeout_s
        self.idle_s = idle_s
        self.fallback = fallback
        self._clock = clock
        self._http = httpx.AsyncClient(
            transport=transport, timeout=httpx.Timeout(timeout_s, connect=5.0)
        )
        # (person key, profile) -> (hermes session id, last use)
        self._sessions: dict[tuple[str, str], tuple[str, float]] = {}
        self._healthy = False

    # ---- plumbing ------------------------------------------------------------------------

    def _base(self, profile: str) -> str:
        return self.url if profile == DEFAULT_PROFILE else f"{self.url}/p/{profile}"

    def env_path(self, profile: str) -> Path:
        return profile_env_path(self.hermes_home, profile)

    def has_key(self, profile: str) -> bool:
        return read_env_key(self.env_path(profile), self.key_var) is not None

    def _headers(self, profile: str, identity: Identity | None) -> dict[str, str]:
        key = read_env_key(self.env_path(profile), self.key_var)
        if key is None:
            raise GatewayUnavailable(f"profil {profile!r} nie ma klucza {self.key_var} w swoim .env")
        headers = {"Authorization": f"Bearer {key}"}
        if identity is not None:
            # Stable per-person memory scope on the Hermes side (independent of the transcript).
            headers["X-Hermes-Session-Key"] = f"gaja:{identity.key}"
        return headers

    async def _new_session(self, profile: str, identity: Identity | None, headers: dict[str, str]) -> str:
        who = identity.key if identity is not None else "guest"
        sid = f"gaja-{who}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        name = (identity.name if identity is not None else None) or "gość"
        res = await self._http.post(
            f"{self._base(profile)}/api/sessions", json={"id": sid, "title": f"Gaja: {name}"}, headers=headers
        )
        if res.status_code >= 400:
            raise RuntimeError(f"Hermes nie utworzył sesji ({res.status_code}): {res.text[:200]}")
        return str(res.json().get("id") or res.json().get("session_id") or sid)

    async def _session(self, profile: str, identity: Identity | None, headers: dict[str, str]) -> str:
        key = (identity.key if identity is not None else "guest", profile)
        now = self._clock()
        known = self._sessions.get(key)
        if known is not None and now - known[1] < self.idle_s:
            self._sessions[key] = (known[0], now)
            return known[0]
        sid = await self._new_session(profile, identity, headers)
        self._sessions[key] = (sid, now)
        return sid

    def reset(self, person_key: str) -> None:
        """Start fresh Hermes sessions for this person ("nowy temat")."""
        for key in [k for k in self._sessions if k[0] == person_key]:
            del self._sessions[key]

    # ---- HermesClient --------------------------------------------------------------------

    async def submit(
        self,
        task: str,
        *,
        context: list[dict[str, Any]],
        session_id: str,
        language: str | None = None,
        profile: str = DEFAULT_PROFILE,
        identity: Identity | None = None,
        instructions: str | None = None,
    ) -> AsyncIterator[HermesEvent]:
        lang = language if language in ("pl", "en") else detect_language(task, context)
        try:
            headers = self._headers(profile, identity)
            sid = await self._session(profile, identity, headers)
        except (httpx.ConnectError, httpx.ConnectTimeout, GatewayUnavailable) as exc:
            self._healthy = False
            if profile == DEFAULT_PROFILE and self.fallback is not None:
                log.warning("Hermes gateway unavailable (%s); using the ACP worker", exc)
                async for event in self.fallback.submit(
                    task, context=context, session_id=session_id, language=language
                ):
                    yield event
                return
            yield HermesEvent("error", f"Błąd Hermes: {exc}", {"error": str(exc), "profile": profile})
            return
        except Exception as exc:
            yield HermesEvent("error", f"Błąd Hermes: {exc}", {"error": str(exc), "profile": profile})
            return
        self._healthy = True
        yield HermesEvent("accepted", f"Hermes ({profile})", {"hermes_session_id": sid, "profile": profile})

        body: dict[str, Any] = {"message": format_voice_prompt(with_context(task, context, lang), lang)}
        if instructions:
            body["system_message"] = instructions
        if identity is not None and identity.person is not None:
            body["author"] = {"id": f"gaja:{identity.key}", "name": identity.person.name}
        try:
            async for event in self._stream(profile, sid, body, headers):
                yield event
        except httpx.HTTPError as exc:
            yield HermesEvent("error", f"Błąd Hermes: {exc}", {"error": str(exc), "profile": profile})

    async def _stream(
        self, profile: str, sid: str, body: dict[str, Any], headers: dict[str, str]
    ) -> AsyncIterator[HermesEvent]:
        url = f"{self._base(profile)}/api/sessions/{sid}/chat/stream"
        deltas: list[str] = []
        final: str | None = None
        async with self._http.stream("POST", url, json=body, headers=headers) as res:
            if res.status_code >= 400:
                detail = (await res.aread()).decode("utf-8", "replace")[:200]
                if res.status_code == 404:
                    # Session gone on the Hermes side: the next request starts a new one.
                    self._sessions = {k: v for k, v in self._sessions.items() if v[0] != sid}
                yield HermesEvent("error", f"Błąd Hermes ({res.status_code}): {detail}", {"status": res.status_code})
                return
            frame: list[str] = []
            async for line in res.aiter_lines():
                if line:
                    frame.append(line)
                    continue
                parsed = iter_sse(frame)
                frame = []
                if parsed is None:
                    continue
                name, data = parsed
                if name == "assistant.delta":
                    delta = str(data.get("delta") or "")
                    if delta:
                        deltas.append(delta)
                        yield HermesEvent("progress", delta, {"type": "text_delta", "delta": delta})
                elif name == "tool.started":
                    tool = data.get("tool_name") or "narzędzie"
                    yield HermesEvent("progress", f"Hermes uruchamia narzędzie: {tool}", {"type": "tool_call", "raw": data})
                elif name in ("tool.completed", "tool.failed"):
                    yield HermesEvent("progress", "Trwa przetwarzanie narzędzia...", {"type": "tool_call_update", "raw": data})
                elif name == "approval.request":
                    await self._deny(profile, data, headers)
                    yield HermesEvent(
                        "progress",
                        "Wymagana autoryzacja narzędzia — anulowano w trybie głosowym",
                        {"type": "permission_cancelled", "request": data},
                    )
                elif name == "assistant.completed":
                    final = str(data.get("content") or "")
                elif name in ("run.failed", "run.cancelled", "error"):
                    reason = data.get("error") or data.get("message") or name
                    yield HermesEvent("error", f"Błąd Hermes: {reason}", {"error": str(reason)})
                    return
        raw = (final if final is not None else "".join(deltas)).strip()
        hint = clean_spoken_hint(raw)
        yield HermesEvent(
            "result",
            hint or raw,
            {"raw_text": raw, "spoken_hint": hint, "hermes_session_id": sid, "profile": profile},
        )

    async def _deny(self, profile: str, data: dict[str, Any], headers: dict[str, str]) -> None:
        run_id = data.get("run_id")
        if not run_id:
            return
        body: dict[str, Any] = {"choice": "deny"}
        if data.get("request_id"):
            body["request_id"] = data["request_id"]
        try:
            await self._http.post(f"{self._base(profile)}/v1/runs/{run_id}/approval", json=body, headers=headers)
        except httpx.HTTPError:
            log.warning("could not deny a Hermes approval request for run %s", run_id)

    async def health(self) -> bool:
        try:
            res = await self._http.get(f"{self.url}/health", timeout=5.0)
            self._healthy = res.status_code == 200
        except httpx.HTTPError:
            self._healthy = False
        if self.fallback is not None:
            await self.fallback.health()
        return self._healthy or (self.fallback is not None and self.fallback.ready())

    def ready(self) -> bool:
        return self._healthy or (self.fallback is not None and self.fallback.ready())

    async def forget(self, session_id: str) -> None:
        # Hermes sessions belong to people, not to one connection; nothing to drop here.
        if self.fallback is not None:
            await self.fallback.forget(session_id)

    async def aclose(self) -> None:
        await self._http.aclose()
        if self.fallback is not None:
            await self.fallback.aclose()
