from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from gaja.hermes.base import HermesEvent

logger = logging.getLogger("gaja.hermes.client")

_POLISH_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
_POLISH_KEYWORDS = {
    "jest", "się", "czy", "dla", "nie", "tak", "oraz", "może", "dzień", "dzisiaj",
    "tydzień", "jaki", "ile", "proszę", "powiedz", "zrób", "kiedy", "gdzie",
}


def detect_language(text: str, context: list[dict[str, Any]] | None = None) -> str:
    """Detect language ('pl' or 'en') from task text and optional recent context."""
    corpus = text or ""
    if context:
        for msg in context[-4:]:
            content = msg.get("content")
            if isinstance(content, str):
                corpus += " " + content

    if any(ch in corpus for ch in _POLISH_CHARS):
        return "pl"

    words = set(re.findall(r"\w+", corpus.lower()))
    if words & _POLISH_KEYWORDS:
        return "pl"

    return "en"


CONTEXT_TURNS = 4
CONTEXT_CHARS = 1500
STREAM_LIMIT = 16 * 1024 * 1024
CANCEL_DRAIN_S = 5.0
CANCEL_GIVE_UP_S = 30.0
DEFAULT_ADAPTER_API_URL = "http://127.0.0.1:8790"


def speaker_mcp_server_spec(api_url: str = DEFAULT_ADAPTER_API_URL) -> dict[str, Any]:
    """Build the ACP McpServerStdio descriptor for the Gaja speaker MCP server.

    Matches the Agent Client Protocol ``McpServerStdio`` schema expected by Hermes ACP
    (``name``, ``command``, ``args``, and ``env`` as a list of ``{"name", "value"}`` entries).
    ``PYTHONPATH`` points at the repository root so ``python -m gaja.mcp_speaker`` works even
    when the ACP session's working directory is ``~`` (``[tool.uv] package = false``).
    """
    repo_root = str(Path(__file__).resolve().parents[2])
    return {
        "name": "gaja_speaker",
        "command": sys.executable,
        "args": ["-m", "gaja.mcp_speaker"],
        "env": [
            {"name": "PYTHONPATH", "value": repo_root},
            {"name": "GAJA_API_URL", "value": api_url},
        ],
    }


def with_context(task: str, context: list[dict[str, Any]] | None, language: str) -> str:
    """Prepend the last few conversation turns so Hermes can resolve references like "to" or "tam"."""
    lines: list[str] = []
    budget = CONTEXT_CHARS
    for msg in reversed((context or [])[-CONTEXT_TURNS:]):
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        who = ("Użytkownik" if language == "pl" else "User") if msg.get("role") == "user" else "Gaja"
        line = f"{who}: {content.strip()}"
        if len(line) > budget:
            break
        budget -= len(line)
        lines.append(line)
    if not lines:
        return task
    lines.reverse()
    if language == "pl":
        return "Kontekst rozmowy:\n" + "\n".join(lines) + f"\n---\nZadanie: {task}"
    return "Conversation context:\n" + "\n".join(lines) + f"\n---\nTask: {task}"


def format_voice_prompt(task: str, language: str) -> str:
    """Format voice-specific preamble instructing Hermes to be concise and read-aloud friendly."""
    if language == "pl":
        header = (
            "[Kontekst: Rozmowa głosowa (Gaja). Język użytkownika: polski. "
            "Odpowiedz bardzo zwięźle, bezpośrednio i naturalnie, odpowiednio do przeczytania "
            "na głos przez syntezator mowy. Nie używaj formatowania markdown, nagłówków, "
            "list punktowanych, tabel ani bloków kodu. Swoją pamięć (narzędzie memory) możesz "
            "w tej rozmowie swobodnie czytać, dopisywać, poprawiać i usuwać, gdy rozmówca o to prosi "
            "albo gdy uznasz to za przydatne.]"
        )
    else:
        header = (
            "[Context: Realtime voice conversation (Gaja). User language: English. "
            "Reply very concisely, directly, and naturally, suitable to be read aloud via TTS. "
            "Do not use markdown formatting, headers, bullet lists, tables, or code blocks. "
            "You may freely read, add, correct and remove entries in your memory (memory tool) during "
            "this conversation when the user asks or when you find it useful.]"
        )
    return f"{header}\n\n{task}"


def pick_allow_option(options: list[dict[str, Any]]) -> str | None:
    """The ACP option id that allows a tool call once (else any allow), or None if only rejections exist."""
    allow = [o for o in options if str(o.get("kind", "")).startswith("allow")]
    once = [o for o in allow if o.get("kind") == "allow_once"]
    chosen = (once or allow or [None])[0]
    return str(chosen.get("optionId") or chosen.get("option_id")) if chosen else None


def clean_spoken_hint(text: str) -> str | None:
    """If text contains markdown or non-spoken formatting, return a cleaned plain-text version."""
    if not text:
        return None

    has_markdown = bool(re.search(r"(\*\*|\*|#|`{1,3}|\[.*?\]\(.*?\)|^\s*[-*+]\s+|^\s*\d+\.\s+)", text, re.MULTILINE))
    if not has_markdown and len(text) <= 300:
        return None

    cleaned = text
    # Remove code blocks
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
    # Remove inline code
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    # Remove images and links
    cleaned = re.sub(r"!\[.*?\]\(.*?\)", "", cleaned)
    cleaned = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", cleaned)
    # Remove headers
    cleaned = re.sub(r"^\s*#+\s*", "", cleaned, flags=re.MULTILINE)
    # Remove bold/italic markers
    cleaned = re.sub(r"(\*\*|__)(.*?)\1", r"\2", cleaned)
    cleaned = re.sub(r"(\*|_)(.*?)\1", r"\2", cleaned)
    # Remove bullet points and numbered lists
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)
    # Collapse multiple whitespaces/newlines
    cleaned = re.sub(r"\n\s*\n", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()

    return cleaned if cleaned != text else None


_STDERR_LEVEL = re.compile(r"\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]")


def stderr_level(line: str) -> int:
    """Hermes logs its own level as ``[INFO]`` etc.; anything else on stderr (tracebacks) is a warning."""
    match = _STDERR_LEVEL.search(line[:80])
    if match is None:
        return logging.WARNING
    return max(logging.INFO, logging.getLevelName(match.group(1)))


@dataclass
class HermesClientConfig:
    backend: str = "hermes"
    command: list[str] = field(default_factory=lambda: ["hermes", "acp"])
    cwd: Path | str = field(default_factory=Path.cwd)
    timeout_s: float = 300.0
    approval_timeout_s: float = 5.0
    auto_restart: bool = True
    speaker_mcp: bool = True
    api_url: str = DEFAULT_ADAPTER_API_URL

    @classmethod
    def from_obj(cls, obj: Any) -> HermesClientConfig:
        if isinstance(obj, HermesClientConfig):
            return obj
        cfg = cls()
        if obj is None:
            return cfg

        backend = getattr(obj, "backend", None) or (obj.get("backend") if isinstance(obj, dict) else None)
        if backend:
            cfg.backend = str(backend)

        cmd = getattr(obj, "command", None) or (obj.get("command") if isinstance(obj, dict) else None)
        if cmd:
            if isinstance(cmd, str):
                cfg.command = cmd.split()
            elif isinstance(cmd, list):
                cfg.command = [str(c) for c in cmd]

        cwd = getattr(obj, "cwd", None) or (obj.get("cwd") if isinstance(obj, dict) else None)
        if cwd:
            cfg.cwd = Path(cwd)

        timeout = getattr(obj, "timeout_s", None) if not isinstance(obj, dict) else obj.get("timeout_s")
        if timeout is not None:
            cfg.timeout_s = float(timeout)

        speaker_mcp = getattr(obj, "speaker_mcp", None) if not isinstance(obj, dict) else obj.get("speaker_mcp")
        if speaker_mcp is not None:
            cfg.speaker_mcp = bool(speaker_mcp)

        api_url = getattr(obj, "api_url", None) if not isinstance(obj, dict) else obj.get("api_url")
        if api_url:
            cfg.api_url = str(api_url)

        return cfg


@runtime_checkable
class AcpTransport(Protocol):
    """Transport protocol allowing fake test doubles without running live Hermes subprocesses."""
    async def start(self) -> None: ...
    async def close(self) -> None: ...
    def is_alive(self) -> bool: ...
    async def request(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]: ...
    def register_update_handler(self, session_id: str, handler: Any) -> None: ...
    def unregister_update_handler(self, session_id: str) -> None: ...
    async def send_cancel(self, session_id: str) -> None: ...


class SubprocessAcpTransport:
    """Production ACP transport over stdio subprocess."""

    def __init__(self, command: list[str], cwd: Path | str, approval_timeout_s: float = 5.0) -> None:
        self.command = command
        self.cwd = str(cwd)
        self.approval_timeout_s = approval_timeout_s
        self.proc: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._pending_requests: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._update_handlers: dict[str, Any] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._started = False

    def is_alive(self) -> bool:
        reader_ok = self._reader_task is None or not self._reader_task.done()
        return self.proc is not None and self.proc.returncode is None and reader_ok

    async def start(self) -> None:
        if self.is_alive():
            return
        if self.proc is not None:
            await self.close()

        cmd = list(self.command)
        if not shutil.which(cmd[0]):
            resolved = shutil.which("hermes")
            if resolved:
                cmd[0] = resolved
            else:
                raise FileNotFoundError(f"Command not found: {cmd[0]}")

        logger.info("Spawning Hermes ACP transport: %s (cwd=%s)", cmd, self.cwd)
        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=STREAM_LIMIT,
        )
        self._started = True
        self._reader_task = asyncio.create_task(self._read_loop(), name="gaja-hermes-acp-reader")
        self._stderr_task = asyncio.create_task(self._stderr_loop(), name="gaja-hermes-acp-stderr")

    async def close(self) -> None:
        for task in (self._reader_task, self._stderr_task):
            if task is not None and not task.done() and task is not asyncio.current_task():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._reader_task = None
        self._stderr_task = None

        if self.proc and self.proc.returncode is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                self.proc.terminate()
                await asyncio.wait_for(self.proc.wait(), timeout=3.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None
        self._fail_all_pending("Transport closed")

    def _fail_all_pending(self, reason: str) -> None:
        for fut in list(self._pending_requests.values()):
            if not fut.done():
                fut.set_exception(RuntimeError(reason))
        self._pending_requests.clear()

    async def _send(self, msg: dict[str, Any]) -> None:
        if not self.is_alive() or not self.proc or not self.proc.stdin:
            raise RuntimeError("ACP subprocess is not running")
        line = json.dumps(msg) + "\n"
        async with self._write_lock:
            self.proc.stdin.write(line.encode("utf-8"))
            await self.proc.stdin.drain()

    async def request(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        if not self.is_alive():
            await self.start()

        self._next_id += 1
        rid = self._next_id
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_requests[rid] = fut

        msg = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        try:
            await self._send(msg)
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"ACP request {method} timed out after {timeout}s")
        finally:
            self._pending_requests.pop(rid, None)

    def register_update_handler(self, session_id: str, handler: Any) -> None:
        self._update_handlers[session_id] = handler

    def unregister_update_handler(self, session_id: str) -> None:
        self._update_handlers.pop(session_id, None)

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def send_cancel(self, session_id: str) -> None:
        # ACP defines session/cancel as a notification; sending it as a request would wait for a
        # response that never comes.
        try:
            await self.notify("session/cancel", {"sessionId": session_id})
        except Exception as e:
            logger.debug("Failed sending cancel for session %s: %s", session_id, e)

    async def _handle_request_permission(self, frame: dict[str, Any]) -> None:
        """Approve every Hermes permission request (voice mode has nobody to ask; the owner wants
        tools to just run). Picks a one-off "allow" option so nothing is persisted in Hermes."""
        rid = frame.get("id")
        params = frame.get("params", {})
        session_id = params.get("sessionId")
        option = pick_allow_option(params.get("options") or [])
        if option is None:
            logger.warning("Permission request on session %s has no allow option; cancelling", session_id)
            outcome: dict[str, Any] = {"outcome": "cancelled"}
        else:
            logger.info("Permission request on session %s: auto-approved (%s)", session_id, option)
            outcome = {"outcome": "selected", "optionId": option}
        try:
            await self._send({"jsonrpc": "2.0", "id": rid, "result": {"outcome": outcome}})
        except Exception as e:
            logger.error("Failed to answer request_permission: %s", e)

        if session_id and session_id in self._update_handlers:
            self._update_handlers[session_id]({
                "sessionUpdate": "permission_approved" if option is not None else "permission_cancelled",
                "request": params,
                "text": "Zatwierdzono użycie narzędzia" if option is not None else "Hermes nie dał opcji zgody na narzędzie",
            })

    async def _stderr_loop(self) -> None:
        if not self.proc or not self.proc.stderr:
            return
        window_start, logged, dropped = time.monotonic(), 0, 0
        while True:
            try:
                line = await self.proc.stderr.readline()
            except ValueError:
                continue
            except asyncio.CancelledError:
                return
            if not line:
                return
            now = time.monotonic()
            if now - window_start > 10.0:
                if dropped:
                    logger.warning("hermes: %d stderr lines suppressed", dropped)
                window_start, logged, dropped = now, 0, 0
            if logged < 20:
                logged += 1
                text = line.decode("utf-8", "replace").rstrip()[:500]
                logger.log(stderr_level(text), "hermes: %s", text)
            else:
                dropped += 1

    async def _read_loop(self) -> None:
        if not self.proc or not self.proc.stdout:
            return

        while True:
            try:
                try:
                    line = await self.proc.stdout.readline()
                except ValueError as e:
                    # Line longer than the stream limit: the reader drops it and stays usable.
                    logger.error("ACP frame over %d bytes dropped: %s", STREAM_LIMIT, e)
                    continue
                if not line:
                    break
                line_str = line.decode("utf-8", "replace").strip()
                if not line_str:
                    continue
                try:
                    frame = json.loads(line_str)
                except Exception:
                    continue

                if "id" in frame and frame["id"] in self._pending_requests:
                    fut = self._pending_requests.pop(frame["id"])
                    if fut.done():
                        continue
                    if "error" in frame:
                        fut.set_exception(RuntimeError(frame["error"]))
                    else:
                        fut.set_result(frame.get("result") or {})

                elif "method" in frame:
                    method = frame["method"]
                    params = frame.get("params", {})

                    if method == "session/update":
                        session_id = params.get("sessionId")
                        update = params.get("update", {})
                        if session_id and session_id in self._update_handlers:
                            handler = self._update_handlers[session_id]
                            try:
                                handler(update)
                            except Exception:
                                logger.exception("ACP update handler failed")

                    elif method == "session/request_permission":
                        await self._handle_request_permission(frame)

            except asyncio.CancelledError:
                self._fail_all_pending("ACP reader stopped")
                raise
            except Exception as e:
                logger.error("Error in ACP read loop: %s", e)
                break

        # The reader is gone: nothing can answer pending requests any more. Kill the worker so the
        # next request respawns a fresh one (is_alive() is False from here on).
        logger.error("ACP reader exited; Hermes worker will be respawned on next use")
        self._fail_all_pending("ACP stdout EOF / process exited")
        if self.proc is not None and self.proc.returncode is None:
            try:
                self.proc.kill()
            except ProcessLookupError:
                pass


class HermesClient:
    """Real HermesClient adapter communicating with Hermes via ACP over stdio."""

    def __init__(
        self,
        cfg: Any = None,
        *,
        command: list[str] | str | None = None,
        cwd: Path | str | None = None,
        timeout_s: float = 300.0,
        speaker_mcp: bool | None = None,
        api_url: str | None = None,
        transport: AcpTransport | None = None,
    ) -> None:
        self.config = HermesClientConfig.from_obj(cfg)
        if command:
            self.config.command = command.split() if isinstance(command, str) else [str(c) for c in command]
        if cwd:
            self.config.cwd = Path(cwd)
        if timeout_s is not None:
            self.config.timeout_s = timeout_s
        if speaker_mcp is not None:
            self.config.speaker_mcp = bool(speaker_mcp)
        if api_url is not None:
            self.config.api_url = str(api_url)

        self._transport = transport or SubprocessAcpTransport(
            command=self.config.command,
            cwd=self.config.cwd,
            approval_timeout_s=self.config.approval_timeout_s,
        )
        self._session_map: dict[str, str] = {}
        self._running: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._cancelled: set[str] = set()
        self._watchdogs: set[asyncio.Task[None]] = set()
        self._init_done = False
        self._init_lock = asyncio.Lock()

    async def health(self) -> bool:
        """Start the ACP worker if needed and report whether it is initialized."""
        try:
            if not self.ready():
                cmd0 = self.config.command[0]
                if not shutil.which(cmd0) and not shutil.which("hermes"):
                    return False
                await self._ensure_initialized()
            return self.ready()
        except Exception as e:
            logger.warning("Hermes health check failed: %s", e)
            return False

    def ready(self) -> bool:
        return self._init_done and self._transport.is_alive()

    async def _ensure_initialized(self) -> None:
        async with self._init_lock:
            if self._init_done and self._transport.is_alive():
                return
            if not self._transport.is_alive():
                # A respawned worker knows none of the old ACP sessions.
                self._init_done = False
                self._session_map.clear()
                self._running.clear()
                self._cancelled.clear()
            await self._transport.start()
            await self._transport.request(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {
                        "fs": {"readTextFile": False, "writeTextFile": False},
                        "terminal": False,
                    },
                    "clientInfo": {"name": "gaja-voice-adapter", "version": "0.1.0"},
                },
                timeout=30.0,
            )
            self._init_done = True

    def _mcp_servers(self) -> list[dict[str, Any]]:
        if not self.config.speaker_mcp:
            return []
        return [speaker_mcp_server_spec(self.config.api_url)]

    async def _new_hermes_session(self) -> str:
        res = await self._transport.request(
            "session/new",
            {
                "cwd": str(self.config.cwd),
                "mcpServers": self._mcp_servers(),
            },
            timeout=60.0,
        )
        hermes_sid = res.get("sessionId")
        if not hermes_sid:
            raise RuntimeError(f"session/new returned no sessionId: {res}")
        return hermes_sid

    async def _get_or_create_hermes_session(self, gaja_session_id: str) -> str:
        """Resolve or create an ACP session ID for the given Gaja voice session.

        One ACP session runs one prompt at a time. If the mapped session is still finishing a
        cancelled prompt, wait briefly for it; if it is busy with a live task (several Hermes tasks
        may run at once), the new task gets its own ACP session.
        """
        hermes_sid = self._session_map.get(gaja_session_id)
        if hermes_sid is None:
            hermes_sid = await self._new_hermes_session()
            self._session_map[gaja_session_id] = hermes_sid
            return hermes_sid
        running = self._running.get(hermes_sid)
        if running is not None and hermes_sid in self._cancelled:
            await asyncio.wait({running}, timeout=CANCEL_DRAIN_S)
        if hermes_sid not in self._running:
            return hermes_sid
        fresh = await self._new_hermes_session()
        if hermes_sid in self._cancelled:
            self._session_map[gaja_session_id] = fresh
        return fresh

    def _track(self, hermes_sid: str, prompt_task: asyncio.Task[dict[str, Any]]) -> None:
        self._running[hermes_sid] = prompt_task

        def done(_: asyncio.Task[dict[str, Any]]) -> None:
            if self._running.get(hermes_sid) is prompt_task:
                self._running.pop(hermes_sid, None)
                self._cancelled.discard(hermes_sid)

        prompt_task.add_done_callback(done)

    async def _cancel_prompt(self, hermes_sid: str, prompt_task: asyncio.Task[dict[str, Any]] | None) -> None:
        """Ask Hermes to stop (ACP notification) and let the prompt finish with stopReason=cancelled."""
        self._cancelled.add(hermes_sid)
        try:
            await self._transport.send_cancel(hermes_sid)
        except Exception:
            logger.debug("send_cancel failed for %s", hermes_sid, exc_info=True)
        if prompt_task is not None and not prompt_task.done():
            watchdog = asyncio.create_task(self._give_up(prompt_task))
            self._watchdogs.add(watchdog)
            watchdog.add_done_callback(self._watchdogs.discard)

    @staticmethod
    async def _give_up(prompt_task: asyncio.Task[dict[str, Any]]) -> None:
        try:
            result = await asyncio.wait_for(asyncio.shield(prompt_task), timeout=CANCEL_GIVE_UP_S)
            logger.info("Hermes prompt stopped after cancel: %s", result.get("stopReason"))
        except asyncio.TimeoutError:
            logger.warning("Hermes prompt still running %.0fs after cancel; dropping it", CANCEL_GIVE_UP_S)
            prompt_task.cancel()
        except BaseException:
            pass

    async def submit(
        self,
        task: str,
        *,
        context: list[dict[str, Any]],
        session_id: str,
        language: str | None = None,
        profile: str = "default",
        identity: Any = None,
        instructions: str | None = None,
    ) -> AsyncIterator[HermesEvent]:
        """Submit task to Hermes per HermesClient protocol.

        This ACP worker runs the default Hermes profile only; other profiles need the gateway.

        Yields:
            HermesEvent: accepted, progress (deltas/tool calls/steps), result, error.
        Closing or cancelling the iterator sends ACP session/cancel; so does a timeout.
        """
        queue: asyncio.Queue[HermesEvent | None] = asyncio.Queue()
        hermes_sid: str | None = None
        prompt_task: asyncio.Task[dict[str, Any]] | None = None

        if profile != "default":
            yield HermesEvent(
                kind="error",
                text=f"Błąd Hermes: profil {profile!r} wymaga gatewaya Hermesa",
                data={"error": "profile needs gateway", "profile": profile},
            )
            return
        try:
            # 1. Initialize if needed
            await self._ensure_initialized()

            # 2. Map or create session
            hermes_sid = await self._get_or_create_hermes_session(session_id)
            yield HermesEvent(
                kind="accepted",
                text=f"Task accepted by Hermes (session: {hermes_sid[:8]}...)",
                data={"session_id": session_id, "hermes_session_id": hermes_sid},
            )

            # 3. Detect language & format prompt (with recent conversation for references)
            lang = language if language in ("pl", "en") else detect_language(task, context)
            decorated_prompt = format_voice_prompt(with_context(task, context, lang), lang)
            if instructions:
                decorated_prompt = f"{instructions}\n\n{decorated_prompt}"

            # 4. Wire update listener
            collected_chunks: list[str] = []

            def on_update(update: dict[str, Any]) -> None:
                u_type = update.get("sessionUpdate")

                if u_type == "agent_message_chunk":
                    content = update.get("content") or {}
                    chunk_text = content.get("text", "")
                    if chunk_text:
                        collected_chunks.append(chunk_text)
                        queue.put_nowait(HermesEvent(
                            kind="progress",
                            text=chunk_text,
                            data={"type": "text_delta", "delta": chunk_text},
                        ))

                elif u_type == "tool_call":
                    title = update.get("title") or update.get("name") or "narzędzie"
                    queue.put_nowait(HermesEvent(
                        kind="progress",
                        text=f"Hermes uruchamia narzędzie: {title}",
                        data={"type": "tool_call", "raw": update},
                    ))

                elif u_type == "tool_call_update":
                    queue.put_nowait(HermesEvent(
                        kind="progress",
                        text="Trwa przetwarzanie narzędzia...",
                        data={"type": "tool_call_update", "raw": update},
                    ))

                elif u_type == "step":
                    queue.put_nowait(HermesEvent(
                        kind="progress",
                        text="Hermes analizuje zadanie...",
                        data={"type": "step", "raw": update},
                    ))

                elif u_type in ("permission_approved", "permission_cancelled"):
                    queue.put_nowait(HermesEvent(
                        kind="progress",
                        text=update.get("text", ""),
                        data={"type": u_type, "request": update.get("request")},
                    ))

            self._transport.register_update_handler(hermes_sid, on_update)

            # 5. Launch prompt request as concurrent task
            prompt_task = asyncio.create_task(
                self._transport.request(
                    "session/prompt",
                    {
                        "sessionId": hermes_sid,
                        "prompt": [{"type": "text", "text": decorated_prompt}],
                    },
                    timeout=self.config.timeout_s,
                )
            )
            self._track(hermes_sid, prompt_task)

            # 6. Stream updates from queue while prompt is running
            while not prompt_task.done():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.1)
                    if event is not None:
                        yield event
                except asyncio.TimeoutError:
                    continue

            # Drain any remaining updates
            while not queue.empty():
                event = queue.get_nowait()
                if event is not None:
                    yield event

            try:
                prompt_res = await prompt_task
            except TimeoutError:
                # The request timed out on our side; Hermes may still be working. Stop it.
                await self._cancel_prompt(hermes_sid, None)
                raise

            # 7. Produce final result
            raw_text = "".join(collected_chunks).strip()
            if not raw_text:
                raw_text = str(prompt_res.get("text") or prompt_res.get("content") or "").strip()

            spoken_hint = clean_spoken_hint(raw_text)
            final_text = spoken_hint or raw_text
            stop_reason = prompt_res.get("stopReason", "end_turn")
            if stop_reason == "cancelled":
                raise RuntimeError("Hermes task was cancelled")

            yield HermesEvent(
                kind="result",
                text=final_text,
                data={
                    "raw_text": raw_text,
                    "spoken_hint": spoken_hint,
                    "session_id": session_id,
                    "hermes_session_id": hermes_sid,
                    "stop_reason": stop_reason,
                },
            )

        except (asyncio.CancelledError, GeneratorExit):
            logger.info("Hermes submit cancelled for session %s", session_id)
            if hermes_sid and prompt_task is not None and not prompt_task.done():
                await self._cancel_prompt(hermes_sid, prompt_task)
            raise
        except Exception as exc:
            logger.error("Hermes submit error for session %s: %s", session_id, exc)
            yield HermesEvent(
                kind="error",
                text=f"Błąd Hermes: {exc}",
                data={"error": str(exc), "session_id": session_id},
            )
        finally:
            if hermes_sid:
                self._transport.unregister_update_handler(hermes_sid)

    async def forget(self, session_id: str) -> None:
        hermes_sid = self._session_map.pop(session_id, None)
        if hermes_sid is not None and hermes_sid in self._running and hermes_sid not in self._cancelled:
            await self._cancel_prompt(hermes_sid, self._running.get(hermes_sid))

    async def aclose(self) -> None:
        """Close transport and terminate worker process."""
        for watchdog in list(self._watchdogs):
            watchdog.cancel()
        await self._transport.close()
        self._init_done = False
