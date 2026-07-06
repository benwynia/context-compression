# Rung 13: the thrash guard

The fix for rung 12's discovery (compression-induced churn: evict → agent
re-reads → checkpoint → evict again; live worst case 124 checkpoints, 3x
cost). `ctxc proxy ... --thrash-guard`, or `SessionCompressor(guard=ThrashGuard())`.

## Mechanisms (all deterministic; the guard changes budgets, never content)

1. **Re-read pinning** — at every checkpoint the guard fingerprints the tool
   results that were dropped or rewritten (marker-stripped, whitespace-
   normalized, so the advisor's `[block tN]` labels don't defeat matching).
   When an incoming message matches an evicted fingerprint, the agent has
   gone back for something we threw away: that content is pinned — immune to
   truncation and eviction — so the same mistake can't repeat.
2. **Churn escalation** — every `escalate_after` (3) checkpoints, the
   effective budget grows by `escalate_factor` (1.5x), capped at `max_scale`
   (4x). A session that keeps checkpointing is measuring its own budget as
   too small; the guard believes it.
3. **Escalate-on-impossible** — when pinning pressure (or an oversized head)
   makes the budget unreachable, the budget escalates instead of the request
   failing; only at the ceiling does `BudgetImpossible` surface.

## Validation

- Closed-loop simulator (in tests): a synthetic agent that re-reads a
  mid-content fact whenever compression hides it. Ungoverned, it loops
  indefinitely (one re-read per checkpoint cycle); with the guard, the first
  re-read pins the fact, the loop dies, and the fact survives to the end.
- Replay of rung-12's real death-spiral sessions at the same 5k budget:

  | session | checkpoints (no guard) | with guard | re-reads pinned |
  |---|---|---|---|
  | django-12589 | 84 | **32** | 10 |
  | sphinx-11445 | 63 | **25** | 13 |
  | sphinx-8801 | 36 | **17** | 4 |

  Offline replay is the floor, not the ceiling: these transcripts contain
  every re-read the ungoverned agent actually made. Live, pinning removes
  the *reason* for most of those re-reads, so the loop never grows the
  transcript in the first place.

## What's next

The guard makes corpus-derived budgets trustworthy (the behavioral term is
now bounded), so the budget-sweep methodology from the rung-12 analysis can
be applied to real shadow-mode traffic. A live guarded rerun of arms B/C
(~$25) would quantify the cost recovery end-to-end when wanted.
