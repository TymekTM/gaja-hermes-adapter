# Shared LLM API (`/llm/`)

The adapter shares its llama-server (LFM3-1.7B) with other services on this machine and in the tailnet.
It does this through a reverse proxy under `/llm/` on its own HTTP port. Gaja's voice turns keep
priority: external requests get their own llama-server slot and never preempt Gaja.

## Address

| From | Base URL |
|---|---|
| the same machine | `http://127.0.0.1:8790/llm` |
| the tailnet | `https://<host>.<tailnet>.ts.net:8443/llm` (the existing `tailscale serve` HTTPS proxy of the adapter) |

llama-server itself stays on `127.0.0.1:8791`, so it is never reachable from the LAN. There is no new
listener and no firewall change: the tailnet path reuses the proxy that already serves the web client.

## Auth

- **Token**: send `Authorization: Bearer <token>`. The token is one line in `[llm_proxy] token_file`
  (default `~/.config/gaja/llm_api.token`, mode 600).
  - `uv run python scripts/llm_token.py` creates the token. `--rotate` replaces it, and old clients get
    401 at once; the adapter re-reads the file, so no restart is needed.
  - A service on this machine reads the file itself. A tailnet client gets a copy over a secure channel.
- **Direct loopback without a token**: allowed, unless the request carries an `Origin` the adapter does
  not allow, which keeps web pages out.
- **Requests relayed by `tailscale serve`** arrive from loopback too. They carry `X-Forwarded-For` and
  `Tailscale-User-Login`, so they always need the token. A wrong or missing token gets 401.

## Endpoints

Request and response bodies pass through unchanged; SSE streams (`"stream": true`) are relayed chunk
by chunk, without buffering.

| Endpoint | Kind |
|---|---|
| `POST /llm/v1/chat/completions` | generation (OpenAI-compatible) |
| `POST /llm/v1/completions` | generation (OpenAI-compatible) |
| `POST /llm/completion` | generation (llama.cpp native) |
| `GET /llm/v1/models` | pass-through |
| `GET /llm/health` | pass-through |
| `POST /llm/tokenize`, `/llm/detokenize`, `/llm/apply-template` | pass-through, no queue |
| `GET /llm/status` | the proxy's own state: slot, active, waiting, `gaja_busy`, limits |

Other llama-server paths (`/slots`, `/metrics`, `/props`, ...) return 404.

The model id is whatever `/llm/v1/models` lists. llama-server serves one model and ignores the `model` field.

## Slot policy and limits

llama-server runs `-c 36864 -np 3 --no-kv-unified`, which gives three slots of 12288 tokens each:

| Slot | Owner |
|---|---|
| 0 | Gaja's chat reply (its prompt cache holds the conversation) |
| 1 | Gaja's router |
| 2 | `/llm/` |

The proxy sets `id_slot` to 2 on every generation request and overrides any value the client sends.
External prompts therefore never evict Gaja's caches. A prompt plus its output must fit the slot's
12288 tokens.

Generation requests are handled as follows (all values are in `[llm_proxy]`):

- **Concurrency**: one external generation at a time (`max_concurrent = 1`). The next request waits in a
  FIFO queue of up to 16 (`max_waiting`). When the queue is full, the request gets `503` with
  `Retry-After: 10` at once.
- **Gaja first**: an external request starts only after Gaja has been idle for 4 s
  (`yield_to_gaja_s`). Gaja counts as busy during its own LLM calls, from a device wake, and from the
  start of speech to the end of the turn. A generation that is already running is not interrupted.
  The measurements below show what that costs Gaja.
- **Chunked prefill**: llama-server evaluates all slots in one batch step, so a big external prompt
  in slot 2 would hold Gaja's calls in slots 0 and 1 for seconds. The proxy therefore first feeds a
  long prompt into slot 2 in chunks of 128 tokens (`prefill_chunk_tokens`), and it pauses between
  chunks while Gaja is busy. The request itself then finds its prompt cached. This makes the external
  prefill about 30 % slower and limits what an in-flight chunk can cost Gaja to a fraction of a second.
  Prompts the proxy cannot render (several prompts, multimodal parts) prefill in one go.
- **Queue timeout**: a request that cannot start within 120 s (`queue_timeout_s`) gets `503` with
  `Retry-After`. Clients should honour `Retry-After` and retry.
- **Length cap**: `max_tokens` (and `max_completion_tokens`, or `n_predict` on `/completion`) is
  clamped to 2048. A missing, negative or larger value becomes 2048.
- **Errors**: if llama-server is unreachable, the proxy returns `502`. A client that hangs up
  mid-stream closes the upstream connection, so llama-server stops generating and the slot is freed.

Error bodies are OpenAI-shaped: `{"error": {"message", "type", "code"}}`.

## Effect on Gaja's latency

Measured with `scripts/bench_llm_share.py`. Each probe is one of Gaja's own call shapes on its pinned
slot, primed like the adapter primes it:

- **chat TTFT**: time to the first token of a reply over a cached history of about 4.9k tokens;
- **router**: a 1-token logprob decision over a cached prompt of about 1.2k tokens.

The background load runs one of two kinds in a loop:

- **decode**: a short prompt and a long streamed answer of up to 1024 tokens;
- **prefill**: a fresh prompt of about 5k tokens and 16 output tokens.

- **Before**: two slots, with the external client talking straight to llama-server.
- **After**: three slots, with the external client using `/llm/` and chunked prefill.

The numbers are medians, with p90 in parentheses, in ms. "Slot ready" is how long Gaja waits before its
slot holds its cached prefix again, i.e. before it can even start the call.

| Load | | chat TTFT | router | slot ready |
|---|---|---|---|---|
| none | before | 85 (87) | 76 (76) | |
| none | after | 94 (108) | 79 (82) | |
| decode | before | 176 (214) | 161 (175) | **30 861 (34 664)** |
| decode | after | 154 (154) | 140 (141) | 92 (100) |
| prefill | before | 194 (228) | 178 (192) | **11 646 (15 054)** |
| prefill | after | 418 (451) | 287 (296) | 93 (213) |

How to read the table:

- **Before**, an external request lands in one of Gaja's two slots and evicts its prompt cache. Gaja's
  next call on that slot queues behind the whole external request (12–35 s) and then prefills its
  prompt again. The TTFT columns look fine only because they start after that wait.
- **After**, Gaja's slots stay warm and free. A running external generation adds about 70 ms to
  Gaja's TTFT, because llama-server decodes both slots in the same steps.
- **After, prefill row**: the benchmark's probes do not go through the adapter, so the proxy does not
  pause for them. The prefill row is therefore the worst case: Gaja's call shares every step with an
  external chunk. In a real turn, the wake, the start of speech and Gaja's own calls pause the chunks,
  and at most the chunk already in flight (about 0.2 s) overlaps Gaja's first call.
- Without chunking, the same overlap measured 2–3 s (chat) and up to 1.4 s (router). A smaller llama
  `-b` did not help.

## Examples

```bash
TOKEN=$(cat ~/.config/gaja/llm_api.token)
BASE=http://127.0.0.1:8790/llm        # or https://<host>.<tailnet>.ts.net:8443/llm

curl -s "$BASE/v1/models" -H "Authorization: Bearer $TOKEN"

curl -s "$BASE/v1/chat/completions" -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Streść w jednym zdaniu: ..."}],"max_tokens":200}'

# streaming (SSE); -N turns off curl's own buffering
curl -sN "$BASE/v1/chat/completions" -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Wypisz trzy słowa kluczowe: ..."}],"stream":true}'
```

llama-server runs with `--reasoning-budget 0`, so replies carry no thinking.

## Configuration

```toml
[llm_proxy]
enabled = true
# token_file = "~/.config/gaja/llm_api.token"
# slot = 2, max_concurrent = 1, max_waiting = 16, queue_timeout_s = 120
# yield_to_gaja_s = 4, prefill_chunk_tokens = 128, max_tokens = 2048, timeout_s = 600
```

The proxy is off by default. Enabling it requires `-np 3` on llama-server (see `deploy/README.md`).
