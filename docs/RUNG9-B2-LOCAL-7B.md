# Rung 9: local-7B digest summarizer (B2) — performance yes, fidelity no

Question: can a locally-run 7B (qwen2.5:7b via Ollama, Apple M2 Max 32GB)
handle the digest-summarization hook fast enough, and does it improve
retention over the deterministic digest?

## Performance: a non-issue, as designed

Inputs captured from real rung-8 sessions (the exact lines the hook receives
— deterministic digest lines, not raw transcript): median ~430 tokens, max
~870. The 24k-char cap never came close to binding.

| metric | measured |
|---|---|
| prefill | ~611 tok/s (median) |
| generation | ~43 tok/s (median) |
| wall time per checkpoint | median 5.5s, min 1.5s, max 8.8s |
| cold start (model load) | 3.1s once |
| RAM | ~4.7GB resident |

At 1–10 checkpoints per task this is tolerable inline and would be invisible
behind a write-behind swap (serve deterministic digest now, swap the LLM
version in at the next checkpoint). Concurrency ceiling on one M2 Max:
~6–10 checkpoints/min; beyond that the hook's timeout fallback degrades
gracefully to deterministic.

## Fidelity: two disqualifying findings (as tested)

Retention probes on the real Copilot session, deterministic vs 7B digest:

| style | budget | deterministic | 7B summarizer |
|---|---|---|---|
| note (salient facts) | 15k | 8/8 | **7/8** |
| note | 12k | 8/8 | 8/8 |
| plain prose | 15k | 6/8 | 6/8 |
| plain prose | 12k | 4/8 | 4/8 |

1. **The 7B cannot rescue the plain-prose weak spot.** Plain facts lost by
   the deterministic layer are evicted *before* digest lines are built — the
   summarizer never sees them. Identical 6/8 and 4/8. Fixing that weak spot
   requires changing what reaches the digest, not who writes it.
2. **The 7B corrupts exact identifiers it does see.** Traced the note/15k
   miss end-to-end: planted code `RIDGE-4578` was present in the summarizer
   INPUT; the 7B's output listed clearance codes `FROST-5242, DELTA-7634,
   RIDGE-8808` — a confidently wrong mutation, not an omission. A compression
   layer that silently rewrites version numbers, ticket IDs, or error codes
   is strictly worse than one that visibly drops them.

## Verdict

B2 as "7B rewrites the digest" is **rejected on fidelity, not performance**
(one model, one prompt, temperature 0 — but silent token mutation at n=1 is
disqualifying for this failure class). If revisited, the viable shape is a
**hybrid**: pinned lines containing exact identifiers pass through verbatim
(deterministic), and the LLM only compresses the residual prose — plus
`probe --live` to measure semantic (not just string) retention, and a larger
local model. Until then, rung-8's deterministic-only result stands as the
recommended configuration.
