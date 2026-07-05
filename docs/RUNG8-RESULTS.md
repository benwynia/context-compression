# Rung 8 results: 50-instance graded A/B, SWE-agent, deep(er) chains

First adequately-powered quality test: 50 random SWE-bench Lite instances
(seed 42, 12 repos, rung-7's astropy five excluded), SWE-agent 1.1.0 with
filemap disabled, gpt-5.4-mini, per-instance ctxc sessions, official Docker
grading. Arm A = passthrough proxy; arm B = active at **5k budget** (rule:
50–60% of arm-A median chain, pre-registered before B ran).

## Results (official grader, ctxc ab)

| | ctxc (B, 5k) | control (A) |
|---|---|---|
| resolved | **14/50 (28.0%)** | 13/50 (26.0%) |
| discordant pairs | ctxc-only 5 | control-only 4 |
| McNemar exact p | 1.00 (no quality difference detected) | |
| prompt tokens (billed) | 4.57M | 23.4M (saved 80.4%) |
| **exact USD** (OpenAI cache-aware) | **$2.20** | **$3.02** (saved 27.2%) |
| compression engaged | 44/50 tasks | — |
| compressor errors | 0 | — |

## The honest findings

1. **No quality loss detected at n=50.** 14/50 vs 13/50, discordant 5–4 in
   ctxc's favor, and in the engaged-only subset 3–4 against — symmetric noise
   either way. n=50 can only rule out large effects (~15–20pp); it does that.
2. **Dollar savings survived cache-aware pricing this time: 27.2%.** The
   rung-7 inversion did not reproduce, because the budget rule guaranteed
   engagement (44/50 tasks, vs rung-7's near-none). 80% token savings became
   27% dollars — cache economics still ate two-thirds of the win, exactly as
   the depth table predicts for ~10k chains.
3. **Compressed runs wander more.** Control-only losses correlate with heavy
   checkpointing: sklearn-12471 took 32 steps + 10 checkpoints in B vs 13
   steps in A; matplotlib-25442 took 25 vs 15. Both submitted plausible-but-
   wrong patches. One B loss (django-11620) was an `exit_format` death at
   step 7 — but `exit_format` also occurred 3× in the control arm, so it is a
   baseline gpt-5.4-mini failure mode, not a compression signature.
4. **Chain depth: median 9.5k, p75 11.8k, max 48.8k** (filemap disabled).
   Even a verbose SWE-agent config on Lite with a competent model does not
   reach the 60k+ Copilot-agent regime; the deep-chain cost story still needs
   shadow mode on real Copilot/Claude-Code traffic.

## Protocol notes (what it took to get a clean run)

- SWE-agent's default `cache_control` history processor (Anthropic-only)
  moves markers every turn → every request looks like a rewritten history
  (57/74 session resets, constant recompression) — removed for both arms.
  Any history-comparing middleware must strip or tolerate such keys.
- SWE-agent's static system prompt collapses ctxc's first-message session
  fallback — per-instance `x-ctxc-session-id` headers are mandatory.
- Filemap condensation caps chains at ~5k; disabled for both arms.
- Docker Desktop's 60GB VM disk filled mid-run → instant `apt-get`/build
  failures that masquerade as harness bugs. 50-instance runs need ~150GB.
- Total model spend, both arms + pilots: **~$5.80**.

## What this changes

- The quality question has its first real answer: **at 2x compression on
  ~10k chains, task success is statistically indistinguishable** — and the
  per-task cost CI ([-8.17, +0.76] AIC) leans firmly cheaper.
- Next discriminating experiments: (a) tighter budgets (2–3k) to find where
  quality actually breaks; (b) shadow mode on genuinely deep traffic;
  (c) n=150 if a <10pp bound is ever needed.
