# Rung 11: advisory-eviction shim (arm C) — built, live-validated

The rung-10 idea productionized: `ctxc proxy --advisor`. What we learned
building it changed the design in one important way.

## What shipped

- **`[block tN]` markers** on evictable tool results — a pure function of
  the client's append-only history (same message, same marker, every turn),
  so annotation never breaks prefix stability or the provider cache.
- **`prune_context` tool injection** (non-streaming, tools-speaking requests
  only) with the protocol in the tool description; directives are stripped
  from responses before the agent harness sees them.
- **Per-session ledger**, applied ONLY inside compression checkpoints via a
  new `SessionCompressor.pre_checkpoint` hook — advice never adds cache
  rewrites of its own.
- **Soft delete**: application replaces a block's content with a one-line
  stub (structure untouched; tool pairing cannot break) and archives the
  original in memory. The invariant layer outranks every directive: the
  protected head, the recent window, and malformed/out-of-range ids are
  immune, silently counted.
- **Sidecar advisory queries** (see below), write-behind, fire-and-forget.
- 132 tests; `/stats` exposes directive/prune/reminder/sidecar counters.

## What the live runs taught us (django-16139, gpt-5.4, 5k budget)

1. **Smoke 1 — the inline channel is dead on arrival.** Tool injected on
   every turn: 0 calls in 45 turns. Live agents do not volunteer optional
   housekeeping mid-task, even at frontier tier.
2. **Smoke 2 — reminders don't fix it.** Just-in-time nudges appended to the
   newest tool result under budget pressure (sticky per message, so the
   annotated history stays append-only): 5 reminders delivered — verified
   present in 14/16 emissions, second-from-last position — still 0 calls.
   Offline the same model + same system prompt called the tool at 1/3
   decision points; the live harness's one-action discipline wins. The
   inline mechanism stays (other harnesses may differ) but cannot be the
   primary channel.
3. **Smoke 3 — the sidecar works.** Under budget pressure the PROXY asks the
   model directly (rung-10 style), write-behind, and feeds the ledger:
   11 sidecar calls -> 9 valid directives (0 malformed) -> 3 blocks pruned
   at checkpoints, patch submitted, 0 compressor errors. New problem: 11
   full-price calls in 22 turns roughly doubled the instance's model cost.
4. **Smoke 4 — cache-aligned and rate-limited: the shape that works.** Two
   fixes: the advisory query now sends the emission VERBATIM as its prefix
   with the instruction as a trailing user message, so it shares the agent's
   own cached prefix and bills at the cached tier (~10%); and the pressure
   gate moved from 0.70 to 0.85 of budget. Result: **5 sidecar calls in 51
   turns -> 22 directives (0 malformed) -> 16 blocks pruned, 8,590 tokens
   freed, 81.6% prompt-token savings, 0 compressor errors, patch
   submitted.** Five times the pruning of smoke 3 from half the calls — the
   model judges better when it reads the conversation natively as its own
   prefix than as a re-serialized transcript dump.

## Where this leaves arm C

The architecture that survived contact with reality: **deterministic
compressor + checkpoint-batched ledger + proxy-initiated frontier judgment**,
with the agent-inline channel as a free optional extra. Next step is the
rung-8 harness with three arms — A (passthrough), B (deterministic 5k),
C (advisor 5k) — scoring resolve rate, cost (sidecar included), and
harmful-eviction incidents (agent re-reads of pruned files).
