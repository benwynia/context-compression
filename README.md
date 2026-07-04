# ctxc — context compression with verification plumbing

A **cache-aware context-compression workflow** for OpenAI/Copilot-dialect agent
sessions, plus the plumbing that proves it works: structural invariants, a
prefix-stability check, and a replay harness with AIC cost accounting
(1 AIC = $0.01).

```
pip install -e .        # or: uv sync
ctxc demo               # synthesize a long session, verify, print the report
```

## What it does

Long coding-agent conversations blow past model context caps and get expensive.
`ctxc` compresses the message chain with three deterministic stages —

1. **tool-result truncation** (old tool outputs become head/tail excerpts;
   error results keep more),
2. **duplicate elision** (identical tool results collapse to a marker; the
   *last* copy survives, so eviction — which removes oldest rounds first — can
   never strand a marker pointing at content that was itself removed),
3. **round eviction → digest** (oldest assistant+tool rounds are replaced by a
   single deterministic summary line each) —

with an escalation loop that **guarantees** the result fits the token budget or
raises an explicit `BudgetImpossible` (never a silent overshoot), and always
returns a structurally valid chain (tool-call pairing, role order, system+task
head verbatim).

**Cache checkpoints, not per-turn rewriting.** Rewriting history on every
request destroys the provider prompt cache and the savings barely survive to
cost. `SessionCompressor` compresses only when the chain crosses the budget,
compresses *well below* it (hysteresis), and freezes the result — between
checkpoints every emitted request is a byte-stable extension of the previous
one (a cache read).

## Confirming it works

```
ctxc verify session.json --budget 60k          # exit 1 on any violation
ctxc demo --budget 40k --model-cap 50k
```

The verifier replays the session one request per assistant turn (exactly a
harness's call pattern) and checks **every** emission: structure valid, budget
met, head verbatim, prefix stable between checkpoints. It reports cache-aware
token accounting (checkpoint recompression is counted as the cache re-write it
really is) and AIC/USD totals. Example demo output:

```
ctxc verify — OK
  turns replayed          : 41
  checkpoints (recompress): 2
  prompt tokens  baseline : 1,467,879
  prompt tokens  emitted  : 985,312  (saved 32.9%)
  cache reads / writes    : 872,928 / 112,384  (88.6% of emitted read from cache)
  AIC saved               : 48.3 AIC ($0.48, 25.7%)
  turns over model cap (50,000): 12 before -> 0 after
```

The test suite (`uv run pytest`) covers the same invariants across budget
sweeps and adversarial chains, and includes mutation tests proving the harness
*catches* a broken compressor.

## The live workflow (proxy)

```
ctxc proxy --upstream https://your-openai-compatible-endpoint --budget 60k --port 8790
```

Point any OpenAI-compatible client at `http://localhost:8790/v1/chat/completions`.
The proxy keys one `SessionCompressor` per conversation (send an
`x-ctxc-session-id` header; the fallback hashes the chain's *first message*,
which is stable across turns but cannot distinguish concurrent conversations
with identical openings — send the header in multi-user deployments),
compresses `messages`, counts `tools` schemas against the same budget, forwards
everything else — auth headers and query string included — and returns the
upstream response plus `x-ctxc-original-tokens` / `x-ctxc-emitted-tokens`
headers. Compression runs off the event loop (per-session locked), so one big
checkpoint doesn't stall other sessions. `GET /healthz` shows live session
count. Streaming responses pass through chunk-by-chunk, so chat UIs render
tokens as they arrive.

## AIC cost model

Copilot bills in AI Credits at **$0.01 per AIC**. Rates are configurable per
model in both shapes, because they change what compression is worth:

- **token-metered AIC** → compression directly saves credits (reported);
- **request-metered AIC** → credits are unchanged (one request is one request);
  the win is **context headroom** — the report's `turns over model cap
  before/after` line — plus latency.

`--rates rates.json --model <name>` with
`{"<name>": {"per_request": 1.0, "per_1m_input": 100.0, "per_1m_output": 500.0}}`
overrides the illustrative default (both flags together — half-specified is an
error, never a silent fallback to placeholder pricing).

AIC metering has no cache tiers, so the cache read/write split in the report is
a cache-health signal, not a priced quantity: it tells you what checkpoint
recompression costs in provider-cache terms even though it doesn't change the
AIC bill.

## Getting a number you can defend

Do **not** quote the demo's percentage — it is measured on synthetic data. The
kit for measuring on *your* traffic, in increasing order of rigor:

1. **Shadow mode (zero risk).** `ctxc proxy --upstream URL --budget 60k
   --shadow --record ./sessions`. (There is also `--passthrough` — no
   compression at all, measurement only — used as the A/B control arm so both
   arms share identical instrumentation.) Traffic is forwarded **untouched**; the
   would-be savings are measured on the side and aggregated at `GET /stats`
   (including the upstream's own reported `usage` — provider-billed numbers,
   not tiktoken estimates, and the real cache-hit rate). Compression failures
   in shadow mode never fail a request; they're counted in `compress_errors`.
2. **Replay recorded sessions.** `--record` writes each conversation as a
   session file; `ctxc verify sessions/<f>.json --budget 60k` replays it with
   full invariant checking and per-turn accounting.
3. **Price it three ways.** Token savings ≠ dollar savings. With cache-tier
   rates set (`per_1m_cache_read` / `per_1m_cache_write` in the rates file),
   the report adds a cache-aware line where checkpoint recompressions are
   billed as the cache re-writes they really are.

What to expect (synthetic, haiku-class token rates, 40k budget — run your own):

| session depth | tokens saved | flat per-token AIC | cache-aware AIC |
|---|---|---|---|
| ~70k-token chain | 32.9% | 25.7% | **0.3%** |
| ~150k-token chain | 61.8% | 54.3% | **12.4%** |
| ~300k-token chain | 80.1% | 75.1% | **35.0%** |

The honest reading: against a *perfectly cached* baseline, short sessions save
almost no money — compression's dollar value comes from **long sessions**
(where avoided reads dwarf checkpoint re-writes), from **imperfect real-world
caching** (provider caches expire in minutes; idle gaps mean the true baseline
sits between the flat and cache-aware columns — shadow mode's `cached_tokens`
measures where), from **request-metered billing** (savings show up as context
headroom, not credits), and from **sessions that outlive the model's context
cap** — where the uncompressed baseline doesn't cost more, it simply dies.

4. **A/B test on a real benchmark.** The end-state proof — cost *and* task
   completion, paired per task on SWE-bench Verified (or Terminal-Bench):
   run the agent through the proxy vs direct, join grader verdicts with
   `GET /stats/sessions` cost rows, then `ctxc ab ctxc_results/
   control_results/` for resolved rates, an exact McNemar test, cost deltas
   with bootstrap CIs, and the compression-engaged segment. Full protocol in
   [docs/AB-TESTING.md](docs/AB-TESTING.md); step-by-step from a vanilla
   machine in [docs/QUICKSTART-SWEBENCH.md](docs/QUICKSTART-SWEBENCH.md); the
   one-engineer live trial under VS Code + Copilot in
   [docs/LIVE-COPILOT.md](docs/LIVE-COPILOT.md).

## Known limitations

- **Quality is not measured.** The harness proves compression is structurally
  safe, budget-bounded, cache-stable, and cheap — not that the agent performs
  identically on compressed context. Pilot with shadow mode → small active
  group, and watch task outcomes/turn counts.
- Token counts are tiktoken-based: exact for GPT-family, approximate for other
  models behind an OpenAI-compatible endpoint (the `/stats` upstream `usage`
  numbers are exact).
- Upstream-reported usage isn't parsed on streamed responses (it rides in the
  final SSE chunk); token accounting for streamed turns uses local counts.
- Session state is in-memory: run one proxy worker; a restart just means one
  extra checkpoint per live conversation (correctness unaffected).
- The proxy adds no auth of its own — deploy it inside your network boundary.
- The default AIC rate table is illustrative; plug in real rates via `--rates`.

## Library use

```python
from ctxc import SessionCompressor, compress, verify_session

res = compress(messages, budget=60_000)          # pure, one-shot
sc = SessionCompressor(budget=60_000)            # stateful, cache-checkpointed
emitted = sc.request(full_history)               # call per turn
report = verify_session(messages, budget=60_000) # replay verification
```

### In-house LLM compaction (optional, local)

The deterministic digest can be swapped for an LLM-written one — the in-house
equivalent of commercial "compaction" services, without routing your transcript
to a third party. Any OpenAI-compatible endpoint works, including a local 7B:

```bash
ollama pull qwen2.5:7b        # or vLLM / LM Studio / llama.cpp server
ctxc proxy --upstream $URL --budget 60k \
  --summarizer-url http://localhost:11434/v1 --summarizer-model qwen2.5:7b
```

The hook is deliberately constrained so it can only help, never hurt: it runs
**once per checkpoint** (a couple of times per long session, never per
request), its input is bounded (small-context 7Bs are safe), its output is
held to the same digest token cap, and *any* failure — endpoint down, empty
reply, over-cap output — falls back to the deterministic digest. The budget
guarantee is unconditional either way. In code:
`CompressConfig(summarizer=LlmSummarizer("http://localhost:11434/v1", "qwen2.5:7b"))`.

Honest expectations: the summarizer sees the digest *lines* (per-turn
extracts), not the full evicted text, so it improves coherence and merges
redundancy rather than recovering detail the extracts dropped. Whether that
buys task-completion points is an A/B question — run it as a third arm.

## Layout

```
src/ctxc/
  models.py      chain validation, rounds, protected-head logic
  tokens.py      tiktoken counter (deterministic, offline)
  strategies.py  truncation / dedupe / eviction+digest stages
  compressor.py  pure compress() with escalation + hard budget guarantee
  session.py     SessionCompressor (cache-checkpoint state machine)
  aic.py         AIC rates and USD conversion ($0.01/AIC)
  verify.py      replay harness: invariants + cache model + report
  synth.py       synthetic coding-agent session generator
  proxy.py       ASGI compression proxy
  cli.py         ctxc compress | verify | demo | proxy
DESIGN.md        full design rationale
```

## Non-goals (for now)

- Quality/trajectory evaluation — the harness verifies structure, budget,
  cache stability and cost, not that an agent behaves identically on the
  compressed context.
- Anthropic-dialect rendering; OpenAI dialect only, since the target is
  Copilot-style endpoints.
- Streaming responses are forwarded but buffered by the proxy.
