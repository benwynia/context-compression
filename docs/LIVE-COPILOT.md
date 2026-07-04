# Live test: ctxc under VS Code + GitHub Copilot (one engineer, one week)

Goal: a software engineer uses their normal editor all week; ctxc sits between
Copilot Chat and the model; `/stats` accumulates the real savings number.

## The one thing to understand first

Copilot has two kinds of model traffic:

1. **Standard Copilot requests** (your Copilot subscription's built-in models)
   go from VS Code to GitHub's servers over authenticated TLS. You **cannot**
   put a proxy in that path, and shouldn't try.
2. **BYOK models** ("bring your own key" — Anthropic/OpenAI/etc. keys you add
   in VS Code, and OpenAI-compatible custom endpoints) go to **whatever base
   URL you configure**. This is the path ctxc proxies.

So the live test is: the engineer uses a **BYOK model routed through ctxc**
for their agent-mode/chat work. Same model they'd use anyway — the only change
is the URL it's reached at.

## Setup (~10 minutes)

**1. Start the proxy on the engineer's machine** — in shadow mode first:

```bash
git clone https://github.com/benwynia/context-compression && cd context-compression
uv sync
uv run ctxc proxy --upstream https://api.your-provider.com \
  --budget 60k --shadow --record ~/ctxc-sessions --port 8790
```

Shadow = requests pass through **untouched** while ctxc measures what it
*would* save. Zero risk while trust builds.

**2. Point a Copilot BYOK model at it.** In VS Code: Copilot Chat → model
picker → **Manage models…** → add a provider. Pick the **OpenAI-compatible /
custom endpoint** option (menu names vary by VS Code version — any option
that lets you set a base URL works) and set:

- URL: `http://localhost:8790/v1`
- API key: your normal provider key (ctxc forwards auth headers verbatim;
  it never stores the key)
- Model id: the model you already use

If your VS Code build has no custom-endpoint option, any OpenAI-compatible
agent extension (Continue, Cline, …) configured with that base URL exercises
the identical path.

**3. Work normally.** Chat streams as usual (the proxy passes SSE through
chunk-by-chunk). Check the meter anytime:

```bash
curl -s localhost:8790/stats | python -m json.tool
```

`saved_pct` is the headline; `upstream_cached_tokens` shows your real
cache-hit rate (which decides how much of the token savings is money).

## Week 2: flip to active

Restart the proxy without `--shadow` (add `--summarizer-url/…-model` too if
you want the local-7B digests). Everything else stays the same. Now the
compression is real — the engineer should note anything that feels off:

- the agent re-reading files it already read (evicted context),
- forgetting earlier instructions or decisions,
- any request failing with `ctxc_budget_impossible` (a huge single message).

`--record` keeps every session replayable: `uv run ctxc verify
~/ctxc-sessions/<file>.json --budget 60k` reproduces exactly what compression
did to any conversation the engineer found suspicious.

## Rollback

Instant and total: pick a different model in the Copilot model picker, or
restart the proxy with `--passthrough`. Nothing else was changed.

## What this test can and cannot tell you

- **Can:** real savings on real work (`/stats`), real cache behavior, whether
  a working engineer *notices* quality loss, latency feel, rough AIC math.
- **Cannot:** statistically grounded quality numbers — one engineer's week is
  an anecdote. That's what the SWE-bench A/B is for
  (`docs/QUICKSTART-SWEBENCH.md`); run both.

Deployment notes: keep the proxy on localhost (it adds no auth of its own);
one uvicorn worker (session state is in-memory); a proxy restart just costs
one extra recompression per live conversation. **No Docker or containers
anywhere in this trial** — the proxy is a plain Python process installed with
uv; the only external processes are your editor and (optionally) Ollama.
