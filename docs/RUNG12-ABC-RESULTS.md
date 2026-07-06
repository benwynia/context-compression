# Rung 12: three-arm A/B/C, gpt-5.4 — quality holds, economics invert, thrash discovered

Same frozen 50-instance list as rung 8, all arms on gpt-5.4 (the advisor is
frontier-tier-gated, and a fair comparison needs one model): A = passthrough,
B = deterministic 5k, C = advisor 5k. 4-way parallel arms, official grading.
Total agent spend ~$37 (+~$2 sidecar, billed outside litellm accounting).

## Results (official grader; paired vs A)

| | A (control) | B (deterministic) | C (advisor) |
|---|---|---|---|
| resolved | **20/50 (40%)** | 19/50 (38%) | 17/50 (34%) |
| McNemar vs A | — | p = 1.00 (5–6 discordant) | p = 0.55 (4–7) |
| total billed prompt tokens | 9.03M | 7.75M (−14.1%) | 9.85M (**+9.1%**) |
| model dollars (litellm, cache-aware) | **$7.05** | $13.04 (+85%) | $16.45 (+133%) |
| mean steps per task | **17.8** | 44.4 | 57.2 |
| advisor machinery | — | — | 347 sidecar calls, 1,263 directives, 0 invalid, 949 pruned, 0 errors |

## The honest findings

1. **No statistically detectable quality loss** in either compression arm
   (both p ≥ 0.55) — though C's 4–7 discordant split leans negative and
   deserves the follow-up below rather than a shrug.
2. **The rung-8 dollar savings inverted at frontier pricing.** B cost 85%
   more than doing nothing; C 133% more. Cause is not the compression ratio
   per se but **compression-induced wandering**: mean steps went 17.8 → 44.4
   → 57.2. Agents under a tight budget lose context, re-read, re-derive, and
   re-pay. Token-per-request savings were real; total requests exploded.
3. **The thrash loop is the discovery of this rung.** Worst cases in C:
   django-12589 ran 382 steps with 124 checkpoints; sphinx-11445 ran 326
   with 82. Mechanism: evict → agent re-reads the evicted file → budget
   pressure → evict again → repeat. The advisor amplifies it (its pruning
   frees space that immediately refills with re-reads it then re-prunes).
4. **Budget calibration was the trigger.** 5k was carried over from rung 8
   instead of re-derived per protocol; against gpt-5.4's chains it is 45% of
   the median (rule: 50–60%) — close — but the chain distribution has a
   heavy tail (p75 = 19k, max = 36k), so tail tasks were compressed 4–7×.
   All the death-spirals live in that tail. A single fixed budget per arm
   punishes exactly the tasks that need context most.
5. **The advisor mechanism itself scaled flawlessly** — 1,263 directives
   without a single malformed one, zero compressor errors across ~2,900
   requests. The plumbing is production-shaped; the *policy* around it is
   what failed.

## What this changes

- **Compression is not a dollar-saver for efficient frontier models on
  shallow tasks.** Its honest value propositions there are context headroom
  and session survival. Dollar savings were real at mini-tier pricing
  (rung 8: +27%) and remain plausible for genuinely deep sessions — shadow
  mode on real traffic is still the decisive test, now more clearly than ever.
- **ctxc needs a thrash guard**: detect eviction→re-read churn (checkpoints
  per turn, or re-read of recently evicted content) and respond by relaxing
  the budget or pinning re-read content. The 124-checkpoint session should
  be impossible by construction.
- **Budgets should adapt to the tail**, e.g. per-session escalation after N
  checkpoints, rather than one flat number per deployment.
- A clean rerun of B/C at the rule-derived budget (~6.5k) would separate
  "budget slightly tight" from "compression inherently causes wandering at
  this tier" — ~$25 if wanted, but the thrash guard is the higher-value fix.
