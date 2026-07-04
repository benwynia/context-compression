# Rung 7 results: 5-instance graded SWE-bench slice (in-sandbox)

First fully-graded A/B run: real agent (mini-swe-agent v2.4.4, gpt-5-mini),
real proxies, official SWE-bench Lite grading in Docker. **n=5 — a pipeline
validation and cost calibration, not a quality verdict.**

Setup: first 5 SWE-bench Lite instances by id (all astropy). Arm A = ctxc
proxy `--passthrough`; arm B = active, `--budget 8k` (chosen after observing
mini-swe-agent self-truncates observations, keeping final chains ~8–20k; the
initial 15k budget never engaged — the `ab` report's engagement check caught
that, and arm B was rerun at 8k).

## Results (ctxc ab, gpt-5-mini AIC rates: 25/M in, 200/M out)

| | ctxc (B, 8k budget) | control (A) |
|---|---|---|
| resolved (official grader) | **3/5** | 2/5 |
| discordant pairs | ctxc-only 1 | control-only 0 |
| McNemar exact p | 1.00 (n far too small) | |
| prompt tokens (billed) | 564,182 | 618,648 (saved 8.8%) |
| cache-hit rate (OpenAI-reported) | 79% | 88% |
| **exact USD** (cached@10%, out@$2/M) | **$0.077** | **$0.062** |
| compression engaged | 5/5 tasks, 1–6 checkpoints each | — |

## The honest findings

1. **No quality loss detected — and the one discordant pair favored ctxc**
   (astropy-12907: the compressed arm submitted a correct patch; the control
   arm submitted an empty one). At n=5 this is an anecdote, not evidence of
   improvement. The claim this run supports: *compression engaged on every
   task and nothing broke.*
2. **Token savings did NOT survive to dollars at these chain lengths.** 8.8%
   tokens saved became *negative* dollar savings: checkpoints cut the
   cache-hit rate from 88% to 79% (recompressed prefixes are cache re-writes),
   and the compressed arm ran more turns on some instances. This is the
   minmax-bench lesson reproduced in our own graded data — on short chains,
   compression costs more than it saves under cache-discounted pricing.
3. **Chain length is the whole game.** mini-swe-agent self-truncates
   observations, capping chains at ~8–20k tokens — the regime where our own
   depth table predicts ~zero cache-aware savings. Verbose harnesses (Copilot
   agent mode, SWE-agent, OpenHands) build 100k+ chains, where the same table
   predicts real savings. Shadow mode on real traffic measures which regime
   YOUR harness lives in before any rollout decision.

## Pipeline pieces validated end-to-end

Docker daemon bootstrap → image pulls → mini-swe-agent through per-instance
proxies (`api_base` override) → per-task cost rows from `/stats` → official
`swebench.harness.run_evaluation` grading → `ctxc resolve` → `ctxc ab`.
Total model spend for the slice: **~$0.14** (both arms, gpt-5-mini).
Note: gpt-4o-mini could not follow mini-swe-agent's response format
(RepeatedFormatError); gpt-5-mini worked.

## What this changes

- The 50-instance run should use a **verbose harness** (SWE-agent/OpenHands)
  or a benchmark whose sessions are naturally deep — with mini-swe-agent the
  experiment mostly measures the no-engagement regime.
- The `ab` report's engagement warning and the budget-vs-chain-length
  relationship are not theoretical: they caught a mis-set budget on the first
  attempt of this very run.
