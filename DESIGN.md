# ctxc — context compression for Copilot-dialect agent sessions

## Goal

A **functional context-compression workflow** for long coding-agent conversations in
the OpenAI Chat Completions dialect (what GitHub Copilot's backend and BYOK modes
speak), plus **plumbing that proves it works**: structural invariants, a
prefix-stability (cache-friendliness) check, and a replay harness that reports token
and AIC savings. Benchmarking other vendors' proxies is explicitly out of scope —
this is our own compressor and its verification.

Design lessons carried over from auditing `minmax-bench`:

1. **Cache-awareness is the whole game.** A strategy that rewrites history on every
   request torches the provider prompt cache and its token savings barely survive to
   cost (headroom-kompress: 12% tokens → 2% cost). So compression here happens at
   discrete **checkpoints**; between checkpoints the emitted prefix is byte-stable.
2. **Never emit an over-budget or structurally invalid chain.** The audited proxy
   failed 349/3258 turns with `prompt is too long` because compression didn't
   guarantee the cap. Our compressor escalates until the budget is met or raises an
   explicit error — never a silent overshoot.
3. **Verification must replay turn-by-turn and score from raw counts,**
   deterministically and offline, so anyone can re-run it without keys or spend.

## Message model

Native format is the OpenAI chat/completions request shape — plain dicts, passed
through untouched except for the parts we compress:

- `{"role": "system"|"user"|"assistant"|"tool", "content": str|list[parts]}`
- assistant messages may carry `tool_calls: [{id, type: "function", function: {name, arguments}}]`
- tool messages carry `tool_call_id`

Structural rules we validate (and must preserve):

- every `tool_calls[].id` is answered by `tool` message(s) with matching
  `tool_call_id` before the next non-tool message;
- no orphan `tool` messages (a `tool_call_id` that no prior assistant issued);
- system messages only at the head.

A **round** is the atomic eviction unit: one assistant message together with the
`tool` messages answering it, or a standalone user message. Evicting whole rounds
can never break pairing.

## Compression pipeline (deterministic, no LLM required)

`compress(messages, budget, config) -> CompressResult` — pure function.

Protected, always verbatim:

- the **head**: all leading system messages + the first non-system message (the
  task statement);
- the **tail**: the most recent `keep_recent` messages, extended backward to a round
  boundary so a protected tool result never loses its owning assistant message.

Stages, applied in order, recounting after each; stop as soon as `<= budget`:

1. **Tool-result truncation** — unprotected `tool` message contents are cut to
   head + tail excerpts with an explicit `[ctxc: truncated N chars]` marker. Tool
   results dominate coding-agent transcripts; this is the cheap 80%.
   Error-looking results (config regex) are kept longer — errors get referenced later.
2. **Duplicate-result elision** — identical tool contents (same hash) appearing
   more than once: later occurrences collapse to `[ctxc: duplicate of an earlier
   result]`. The *first* occurrence is kept because it is the one already sitting in
   the provider's cached prefix.
3. **Round eviction → digest** — oldest unprotected rounds are evicted and replaced
   by a single **digest** user message inserted right after the protected head:
   one line per evicted round (`assistant: first line… | tools: name(args…) -> first
   line of result`). The digest is deterministic and capped (share of budget). An
   existing digest (from a previous checkpoint) is folded into the new one, so
   digests never nest.
4. **Escalation loop** — if still over budget: tighter truncation caps, wider
   eviction window (everything unprotected), finally shrink the tail protection to
   the last round and truncate protected tool results too. If the irreducible core
   (system + task + last round + digest) still exceeds the budget, raise
   `BudgetImpossible` — an explicit failure, never a silent one.

Every `compress` result is re-validated structurally before being returned.

## Session state machine (cache checkpoints)

`SessionCompressor(budget, config)` wraps the pure function for a live conversation:

- keeps the last **emitted** message list;
- on each request: if the previous emission is a **prefix** of what emission would
  be now (i.e. only new tail messages arrived) and the total is under the
  **trigger** (`budget`), emit `previous + new tail` unchanged — byte-stable prefix,
  cache read for the provider;
- when the trigger is crossed, run one **checkpoint**: `compress()` the whole
  emitted+tail chain down to `budget * recompress_to` (e.g. 60%), emit the result,
  freeze it as the new prefix. Hysteresis (compress to well under the trigger)
  keeps checkpoints rare.

Contract (tested): for a monotonically growing source conversation, every emitted
request extends the previous emission exactly, except at checkpoints; the number of
checkpoints is `O(total_growth / (budget * (1 - recompress_to)))`.

## AIC cost model

Copilot now bills in **AICs at $0.01 each**. `aic.py` models this with configurable
per-model rates supporting both shapes, because token-metered AIC is what makes
compression save money while request-metered AIC makes it save *headroom*:

```python
AicRate(per_request=…, per_1m_input=…, per_1m_output=…)   # AIC units
usd = aic * 0.01
```

Defaults ship a small illustrative table plus `DEFAULT_RATE`; real rates are a
constructor arg / JSON file (`--rates rates.json`) since GitHub's numbers will
drift. The verify report shows both: AIC under token metering (savings) and
requests count (unchanged by compression — stated, not hidden).

## Verification plumbing ("confirm it works")

`verify.py` replays a recorded/synthetic session turn-by-turn (each assistant turn =
one request, prefix = everything before it, exactly a harness's call pattern):

1. **Invariants every turn** — structure valid, budget met, head/tail verbatim,
   digest well-formed. Any violation fails the run (exit 1, listed in the report).
2. **Prefix stability** — emissions between checkpoints must extend the previous
   emission; checkpoint count and positions are reported.
3. **Cache-aware accounting** — incremental cache model per turn: unchanged prefix
   = cache read, appended tail = cache write, prefix break at a checkpoint = cache
   write from the divergence point (the honest cost of recompression). Baseline =
   the same model over the uncompressed chain.
4. **Report** — per-turn and total: prompt tokens (orig vs emitted), cache
   read/write split, AIC + USD under the configured rates, checkpoint count,
   turns-over-model-cap before vs after (the headroom win that survives even
   request-metered billing).

Tests (`pytest`):

- structural invariants on synthetic + adversarial chains (orphan tools, huge
  single results, all-protected chains, empty content, list-form content);
- hard budget guarantee across a sweep of budgets, or `BudgetImpossible`;
- protected zone verbatim; dedupe keeps first; truncation markers present;
- SessionCompressor prefix stability + checkpoint hysteresis;
- AIC arithmetic;
- verify harness catches a deliberately broken compressor (mutation test);
- proxy end-to-end against a fake upstream (ASGI transport, no network).

## Proxy (the workflow glue)

`ctxc proxy --upstream URL --port 8790` — a small ASGI app (starlette + httpx):
accepts `POST /v1/chat/completions` (and `/chat/completions`), keys a
`SessionCompressor` per session (`x-ctxc-session-id` header, else a hash of the
head), compresses `messages`, forwards everything else verbatim (auth headers
included) to the upstream, streams the response back, and appends
`x-ctxc-original-tokens` / `x-ctxc-emitted-tokens` response headers. Point any
OpenAI-compatible client (Copilot BYOK endpoint, a local harness, curl) at it.

## CLI

- `ctxc compress session.json --budget 60000` — one-shot compress + stats.
- `ctxc verify session.json --budget 60000` — full replay verification report.
- `ctxc demo` — generate a synthetic long session, verify it, print the report.
- `ctxc proxy --upstream URL` — run the live proxy.

Session file format: `{"messages": [...]}` or a bare JSON array of messages.

## Layout

```
src/ctxc/
  models.py        chain validation, rounds, content helpers
  tokens.py        tiktoken counter (o200k_base, hash-cached)
  strategies.py    truncation / dedupe / eviction+digest stages
  compressor.py    pure compress() with escalation + CompressResult
  session.py       SessionCompressor (checkpoint state machine)
  aic.py           AIC rates and totals ($0.01/AIC)
  verify.py        replay harness: invariants + cache model + report
  synth.py         synthetic coding-agent session generator
  proxy.py         ASGI proxy app
  cli.py           argparse CLI (ctxc)
tests/             pytest suite (see above)
```

Dependencies: `tiktoken`, `httpx`, `starlette` (+ `uvicorn` to actually serve);
`pytest` for dev. Python ≥ 3.11.

## Non-goals

- LLM-generated summaries: the digest is deterministic-extractive. A
  `summarizer: Callable[[list[Message]], str]` hook exists in the config for later.
- Quality/trajectory evaluation of compressed sessions (same caveat minmax-bench
  carries) — the harness verifies structure, budget, stability, and cost, not
  whether an agent behaves identically.
- Anthropic-dialect rendering. OpenAI dialect only, since the target is Copilot.
