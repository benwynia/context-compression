# A/B testing ctxc: reproducible tasks, objective outcomes

The claim under test: *routing an agent through ctxc reduces cost without
reducing task success.* Both halves need numbers. This is the protocol.

## Design

Two arms, **one variable**:

| | ctxc arm | control arm |
|---|---|---|
| agent harness | identical, pinned version | identical, pinned version |
| model + params | identical (temperature 0 where supported) | identical |
| base URL | `http://localhost:8790` (ctxc proxy, **active** mode) | provider endpoint directly |

Outcomes, all objective:

- **resolved** — the benchmark's own grader (fail-to-pass tests), never a
  judge model, never human opinion;
- **cost** — provider-reported usage per task (prompt/output/cached tokens),
  priced as AIC flat and cache-aware;
- **requests, checkpoints, wall time, cap-failures** (tasks aborted for
  exceeding the context window).

Analysis is **paired per task** (same task in both arms): resolved-rate delta
with an exact McNemar test on the discordant pairs; per-task cost delta with a
bootstrap CI. `ctxc ab` computes all of it.

## Which benchmark

- **Primary: SWE-bench Verified (subset).** The industry-standard objective
  coding benchmark: real GitHub issues, resolution = the repo's own tests
  flipping from fail to pass under the official evaluation harness. Crucially,
  capable agents routinely build 100k+ token contexts on it — the regime where
  ctxc actually engages. Start with a fixed, pre-registered subset (e.g. the
  first 50 instances sorted by instance_id — pick the subset *before* seeing
  any results, and never re-pick after).
- **Secondary: Terminal-Bench** — objective terminal-task pass/fail, long
  agentic sessions, cheaper per instance.
- **Sanity check: Aider polyglot** — sessions are mostly short, so it mainly
  verifies ctxc does no harm when compression rarely engages (expect
  `engaged ≈ 0` and identical outcomes).

Any harness that speaks OpenAI-compatible chat/completions and lets you set a
base URL works: mini-swe-agent (simplest), SWE-agent, OpenHands (all
litellm-based — set the api_base to the proxy).

## Procedure

1. **Start the proxy for the ctxc arm** (one proxy per arm run):

   ```bash
   ctxc proxy --upstream $PROVIDER_URL --budget 60k --record ./runs/ctxc/sessions --port 8790
   ```

2. **Run each task with a per-task session id** so cost attributes cleanly.
   Most harnesses let you inject a header; otherwise the first-message hash
   fallback works when tasks have distinct issue texts (SWE-bench does):

   ```bash
   # per instance, conceptually:
   OPENAI_BASE_URL=http://localhost:8790/v1 \
   EXTRA_HEADER="x-ctxc-session-id: $INSTANCE_ID" \
   run-agent --instance $INSTANCE_ID ...
   ```

3. **Scrape cost after each task** from `GET /stats/sessions` (keyed by the
   session id) and join it with the grader verdict into one row per task:

   ```json
   {"task_id": "...", "resolved": true, "requests": 42,
    "prompt_tokens": 1830042, "output_tokens": 20411,
    "cache_read": 1520000, "cache_write": 310042, "checkpoints": 3}
   ```

   Write one such `*.json` per task into `runs/ctxc/results/`. For the control
   arm, take usage from the provider's responses (or run the proxy with a huge
   `--budget` so it never compresses — then `/stats/sessions` works identically
   and `checkpoints` stays 0).

4. **Grade with the benchmark's official harness** (e.g. `swebench` evaluate)
   — the `resolved` field must come from there, nowhere else.

5. **Compare:**

   ```bash
   ctxc ab runs/ctxc/results runs/control/results --rates rates.json --model claude-haiku-4-5
   ```

   The report gives resolved rates, McNemar p on the discordant pairs, token
   and AIC savings with a per-task bootstrap CI, and the **engaged segment**
   (tasks where compression actually fired).

## Pitfalls that invalidate runs

- **Compression never engaging.** If every chain stays under the budget, you
  measured nothing — the report warns loudly. Either the tasks are too short
  or the budget too high; drop the budget or pick harder instances.
- **Nondeterminism.** Temperature 0 still isn't deterministic server-side. Use
  enough tasks (≥50 pairs for a meaningful McNemar; discordant pairs are what
  carry power) or run pass@k per arm and compare rates.
- **Post-hoc subset selection.** Fix the instance list before the first run.
  Re-picking after seeing results is how you accidentally lie.
- **Asymmetric failures.** A task that crashes for infra reasons in one arm
  must be re-run or dropped from BOTH arms (`ctxc ab` drops unpaired tasks and
  says so).
- **Context-cap deaths count as unresolved, not as excluded.** If the control
  arm dies at the model cap on deep tasks, that IS the result — report it as
  `resolved: false`, don't filter it out (this is ctxc's strongest regime, and
  excluding it biases against ctxc's main benefit).
- **One proxy per arm, restarted between runs** — `/stats` is cumulative and
  in-memory; keep `max_sessions` above the task count so per-task stats aren't
  LRU-evicted before you scrape them (or scrape after every task).

## Reading the result

The pre-registered success criterion should look like: *"resolved-rate delta
not significantly negative (McNemar p ≥ 0.05, and ctxc-only wins ≥
control-only wins − ε), AND AIC saved > X% on the engaged segment."* Decide X
and the subset before running. If quality drops significantly, the savings
number is irrelevant — that's the point of measuring both.
