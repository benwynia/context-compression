# Rung 10: advisory eviction feasibility probe (offline, pre-shim)

Idea under test (proposed 2026-07-05): the frontier model emits pruning
directives against ID-labeled context blocks; a deterministic janitor
executes them in cache-aware batches. Before building the proxy shim, we
tested the load-bearing assumption: **can a model actually judge which
blocks are safe to evict?**

## Method

Replayed 15 recorded rung-8 control sessions (5 deepest + 10 random). At two
decision points per session (50% / 80% of assistant turns), every evictable
block (tool results outside the protected head and last-2 rounds, >=100
tokens) was labeled `[BLOCK id=tN tokens=X]`; the model was asked which are
no longer needed (and, in a second framing, which must be KEPT — the
complement gets evicted). 27 decision points per configuration.

Ground truth (conservative proxy): an eviction is HARMFUL if a distinctive
line unique to that block reappears verbatim in a later assistant message or
tool-call argument in the real transcript — the agent demonstrably used that
exact content afterward. Base rate: ~13% of evictable blocks were needed
later. Baselines matched to the model's evicted token volume per decision
point: random subsets (20 trials) and oldest-first.

## Results

| configuration | evicted volume | harm rate: model | random | oldest | verdict |
|---|---|---|---|---|---|
| gpt-5.4-mini, evict framing | 94% | 0.132 | 0.129 | 0.129 | rubber-stamps everything |
| gpt-5.4-mini, keep framing | ~14% overall (21/27 no-op) | — | — | — | hoards everything |
| **gpt-5.4, evict framing** | 75% | **0.031** | 0.071 | 0.051 | selective (2/53 harmful) |
| **gpt-5.4, keep framing** | 68% | **0.084** | 0.150 | 0.127 | model < random at 8/8 decided points, sign p=0.004 |

## Findings

1. **Eviction judgment is real but model-tier-gated.** gpt-5.4-mini shows no
   selectivity in either framing — the prompt frame, not the content, sets
   its eviction rate (evict-framing: evict ~everything; keep-framing: keep
   ~everything). Full gpt-5.4 evicts 2/3–3/4 of evictable volume while
   avoiding later-needed blocks significantly better than volume-matched
   blind baselines, with zero malformed directives across 54 calls.
2. This is compatible with the inline design: in real deployments the agent
   IS a frontier-tier model, and inline directives reuse the response call
   (no separate judging call needed). But it rules out delegating directive
   generation to a cheap sidecar model.
3. An oracle would evict 87% at zero harm; gpt-5.4 gets ~70–75% at 3–8%
   harm. The gap argues for the soft-delete/restore affordance in the shim
   design — wrong evictions must be cheap to undo, not fatal.

Caveats: n=27 points, one repo-benchmark domain, verbatim-reuse ground truth
(misses semantic dependence, forgives re-reads), retrospective judging of
another agent's transcript rather than the model pruning its own live
context (likely an underestimate of achievable quality).

## Decision

Advisory eviction **graduates to a shim prototype**: ctxc proxy injects a
`prune_context` tool + block ID markers, records directives to a pending
ledger, applies them only at checkpoint boundaries under the existing
invariant layer (head/recent/tool-pairing protected; soft-delete archive).
Evaluate as arm C against rung-8's A and B. Requires a frontier-tier agent
model. Probe harness: `scripts/advisory_probe.py` (rerunnable per model).
