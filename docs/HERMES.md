# Hermes Integration Architecture & Protocol Analysis

This document describes the integration of **Hermes** (the personal agent running on this Linux production host at `~/hermes-agent`) into the **Gaja** realtime voice adapter (`gaja-hermes-adapter`).

---

## 1. Context & Production Constraints

Hermes is installed at `~/hermes-agent` (git repository, Python runtime with mise/pipx) and maintains state/config in `~/.hermes`.

### Shared-Machine Rules
1. **Never modify `~/hermes-agent` or `~/.hermes`**: No git commits, no file editing, no `hermes update`.
2. **Never kill or restart Hermes services**: `hermes-gateway.service` and `hermes-serve.service` run in production.
3. **Never leak secrets**: Passwords, API keys, or tokens in `~/.hermes/auth.json` or `.env` are referenced by path only and never output.

### Live System State
- **`hermes-gateway.service`**: running `gateway run`. Listening strictly on Unix domain socket `~/.hermes/gateway.sock`. Active platforms: Discord, Telegram (two profiles).
- **`hermes-serve.service`**: running `hermes dashboard --no-open --port 9119 --skip-build`.

---

## 2. Integration Candidates Analysis

We investigated 5 integration paths in `~/hermes-agent` to determine the best method for submitting tasks, maintaining session continuity, streaming progress, and handling cancellations.

### Candidate 1: Gateway HTTP/WS API (`gateway/platforms/api_server.py`)
- **Code references**:
  - `gateway/platforms/api_server.py:1-40`: Serves OpenAI-compatible routes (`/v1/chat/completions`, `/v1/responses`, `/v1/runs`, `/api/sessions`) via aiohttp.
  - `gateway/run.py:3321`: Platform mapping for `Platform.API_SERVER`.
  - `gateway/control_socket.py:1-55`: Gateway Unix socket (`gateway.sock`) protocol.
- **Investigation findings**:
  - The control socket `gateway.sock` only implements control verbs: `identify` and `status` (`control_socket.py:110-130`). It cannot execute tasks.
  - The `api_server` platform is **disabled** in the live `hermes-gateway` service configuration. The service has no listening HTTP/TCP ports (verified via `ss -tlpn`).
  - Activating `api_server` would require editing `~/.hermes/config.yaml` and restarting `hermes-gateway.service`, both of which are strictly forbidden.
- **Verdict**: **Infeasible** in the live production setup without service reconfiguration.

### Candidate 2: Dashboard API on port 9119
- **Code references**:
  - `hermes_cli/web_routers/chat_ws.py:299-360`: `/api/console` (executes console commands via `HermesConsoleEngine`).
  - `hermes_cli/web_routers/chat_ws.py:701-725`: `/api/ws` (WebSocket connection delegating to `tui_gateway.ws.handle_ws`).
  - `hermes_cli/web_routers/chat_ws.py:161-175`: `_close_unless_sidecar_allowed` authentication gate.
  - `hermes_cli/web_server_chat.py`: Authentication verification via cookies/session tokens.
- **Investigation findings**:
  - The dashboard server is running and answers public `/api/status` (HTTP 200).
  - Session endpoints (`/api/sessions`) and WebSocket endpoints (`/api/ws`) reject unauthenticated requests with HTTP 401 / WS close code 4401 (`chat_ws.py:168`).
  - Authentication requires accessing credentials in `~/.hermes/auth.json` or dashboard cookies.
  - Network-bound to a specific host IP, creating an unnecessary network dependency for two processes co-located on the same machine.
- **Verdict**: **Fragile & overly complex**. Exposes auth credentials and couples the voice adapter to the desktop dashboard server.

### Candidate 3: Webhook / Bot Relay Mechanism (`~/.hermes/bot_relay`)
- **Code references**:
  - `tools/bot_relay.py:1-50`: Relay directory layout (`roster.json`, `outbox/`, `claimed/`, `replies/`).
  - `tui_gateway/methods_bot_relay.py`: Relay dispatching methods.
- **Investigation findings**:
  - Designed exclusively for Desktop cross-connection DM routing between agents across different gateways.
  - Requires an active Desktop app instance to drain outboxes and route messages.
- **Verdict**: **Not suitable** for direct external process task submission.

### Candidate 4: CLI One-shot Mode (`hermes -z` / `hermes chat -q`)
- **Code references**:
  - `hermes_cli/main.py:3079-3105`: Top-level `-z/--oneshot` handler.
  - `hermes_cli/oneshot.py:240-330`: `run_oneshot()` implementation.
- **Investigation findings**:
  - **Buffering**: `oneshot.py:284` redirects `stdout` and `stderr` to `os.devnull` during agent execution. The user receives zero output until the command finishes.
  - **No Progress Events**: Cannot emit `accepted`, tool-call updates, or streaming deltas.
  - **Cold Start**: Each invocation incurs full Python interpreter and dependency initialization overhead (~1.5–2.0s before LLM call).
  - **Measured Latency**: Tested with `hermes -z "Ile to jest 17*23? Odpowiedz krótko."`: took **18.37s** with zero intermediate feedback.
  - **Session Continuity**: Session ID is not written to stdout; maintaining lineage requires writing `--usage-file` and passing `--resume <session_id>`.
- **Verdict**: Useful as an emergency fallback, but unacceptable for voice interaction where prompt-to-first-feedback latency is critical.

### Candidate 5: ACP Adapter (`hermes acp` / Agent Client Protocol over stdio) — **CHOSEN PATH**
- **Code references**:
  - `acp_adapter/entry.py:1-70, 200-230`: Stdio JSON-RPC 2.0 entry point (`acp.run_agent(agent, use_unstable_protocol=True)`).
  - `acp_adapter/server.py:233`: `HermesACPAgent` implementing standard ACP Agent methods.
  - `acp_adapter/server.py:528`: `initialize()` lifecycle method.
  - `acp_adapter/server.py:616`: `new_session(cwd, ...)` creating and registering a persistent session.
  - `acp_adapter/server.py:633`: `resume_session(cwd, session_id, ...)` for seamless session recovery.
  - `acp_adapter/server.py:643`: `cancel(session_id)` cooperative cancellation via `state.cancel_event.set()`.
  - `acp_adapter/server.py:818`: `prompt(prompt, session_id)` turn execution loop.
  - `acp_adapter/server.py:896-940`: Callback wiring for streaming updates:
    - Text deltas: `agent_message_chunk`
    - Tool calls: `tool_call`, `tool_call_update`
    - Reasoning/thought deltas: `step`
  - `acp_adapter/permissions.py:128-160`: `make_approval_callback` bridging tool approval requests (`session/request_permission`).
- **Why ACP is the best fit**:
  1. **Rich Streaming Events**: Emits granular `session_update` notifications (thinking, tool progress, token deltas) in real time.
  2. **True Session Continuity**: Sessions persist in Hermes's `SessionManager` (`acp_adapter/session.py`) and SQLite `SessionDB`. Successive turns on the same `sessionId` retain full conversation history.
  3. **Low Warm Latency**: Subsequent turns on an existing session bypass initialization and session setup overhead.
  4. **Clean Cancellation**: Native `session/cancel` RPC allows aborting long-running turns immediately if the user interrupts or barges in.
  5. **Secure & Independent**: Uses standard input/output over a subprocess pipe. No listening TCP ports, no Tailscale dependency, and zero secret exposure.

---

## 3. Live Benchmark Results (Measured End-to-End)

All benchmarks performed directly on this production machine with live Hermes:

### Comparison Matrix
| Metric | CLI One-Shot (`hermes -z`) | HermesClient (Turn 1 Cold Session) | HermesClient (Turn 2 Warm Session) |
|---|---|---|---|
| **Health Check** | N/A | **4.72 s** (spawns & checks ACP) | < 0.001 s (worker alive) |
| **Latency to `accepted`** | N/A (never emitted) | **7.76 s** (init + `session/new`) | **0.000 s** (instant reuse) |
| **Latency to First Progress** | N/A (buffered) | **15.16 s** (model thinking/streaming) | **5.54 s** |
| **Latency to Final Result** | **18.37 s** | **15.58 s** | **5.67 s** |
| **Event Stream Delivered** | 0 events | 11 events (deltas + progress) | 15 events (deltas + progress) |
| **Spoken Quality** | Raw text | Natural read-aloud format | Natural read-aloud format |

### Real Prompts Evaluated:
1. **Math Query (Turn 1)**:
   - Input: `"Ile to jest 17*23? Odpowiedz krótko."`
   - Accepted: `7.758 s`
   - First progress (text delta): `15.161 s`
   - Result: `15.583 s`
   - Output: `"Trzysta dziewięćdziesiąt jeden."` (Exact, natural Polish words, ready for speech)
2. **Current Date / Day of Week (Turn 2, Same Session)**:
   - Input: `"Jaki jest dzisiaj dzień tygodnia?"`
   - Accepted: `0.000 s` (immediate session reuse via `sessionId`)
   - First progress (text delta): `5.536 s`
   - Result: `5.671 s`
   - Output: `"Czwartek, ósmego października."` (Exact current date, sub-6s total response)
3. **Session Continuity Verification**:
   - Turn 1: `"Zapamiętaj słowo: KRASNAL. Odpowiedz tylko: Zapamiętano."` -> Result: `"Zapamiętano."` (`8.04 s`)
   - Turn 2: `"Jakie słowo kazałem ci zapamiętać? Odpowiedz tylko tym słowem."` -> Result: `"KRASNAL"` (`4.45 s`)

---

## 4. Client Implementation (`gaja/hermes/client.py`)

### Protocol Compliance (`docs/ARCHITECTURE.md` & `gaja/hermes/base.py`)
```python
@dataclass
class HermesEvent:
    kind: Literal["accepted", "progress", "result", "error"]
    text: str
    data: dict[str, Any] | None = None

class HermesClient(Protocol):
    def submit(self, task: str, *, context: list[dict[str, Any]], session_id: str) -> AsyncIterator[HermesEvent]: ...
    async def health(self) -> bool: ...
```

### Architecture Details
1. **Persistent ACP Worker**:
   - Manages a persistent `hermes acp` subprocess (deliberately WITHOUT `--accept-hooks`: that flag would auto-allowlist the user's shell hooks, overriding `hooks_auto_accept: false` in ~/.hermes/config.yaml).
   - Background reader coroutine processes incoming JSON-RPC lines, routing responses to pending futures and updates to the active event stream.
   - Auto-recovers if the subprocess exits or crashes.
2. **Session Mapping**:
   - Maps Gaja's audio/conversation `session_id` to Hermes's internal ACP `sessionId`.
   - Reuses existing ACP sessions on follow-up turns to preserve context and achieve sub-6s response times.
3. **Voice Prompt Adaptation**:
   - Automatically detects language (`pl` or `en`) from incoming task text and recent conversation context.
   - Preamble added:
     ```
     [Kontekst: Rozmowa głosowa (Gaja). Język użytkownika: polski. Odpowiedz bardzo zwięźle, bezpośrednio i naturalnie, odpowiednio do przeczytania na głos przez syntezator mowy. Nie używaj formatowania markdown, nagłówków, list punktowanych, tabel ani bloków kodu.]
     ```
4. **Result Formatting & Spoken Hint**:
   - Returns the final user-facing text in `HermesEvent(kind="result", text=...)`.
   - Stores raw output in `data["raw_text"]`.
   - If Hermes returns markdown (headers, bullets, backticks, bolding), strips formatting and sets `data["spoken_hint"]` for TTS synthesis.
5. **Interactive Tool Call / Approval Handling**:
   - In a non-interactive voice context, tool approval prompts cannot block indefinitely.
   - When Hermes emits `session/request_permission`, the client automatically responds with `{"outcome": {"outcome": "cancelled"}}` and yields an informative `progress` event: *"Wymagana autoryzacja narzędzia — anulowano w trybie głosowym"*.
6. **Cancellation & Barging-in**:
   - If the caller cancels the async generator during generation, an explicit `session/cancel` RPC is dispatched to Hermes to abort model execution immediately.
7. **Testability via Transport Double**:
   - Accepts an optional `transport: AcpTransport` parameter for fast, deterministic unit testing without spawning real subprocesses.
