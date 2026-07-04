# Quickstart: A/B/B2 on SWE-bench, from a vanilla machine

Goal: three identical agent runs over the same SWE-bench tasks —
**A** (no compression), **B** (ctxc deterministic), **B2** (ctxc + local 7B
digests) — then one command that compares task success and cost.

Budget guide: a 50-instance subset × 3 arms costs roughly $50–200 in model
spend on a haiku-class model, plus a few hours of wall time.

## 0. Prerequisites (once)

- Linux or macOS, Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Docker — for the grading step only** (see "No Docker?" below; nothing in
  ctxc itself uses containers)
- Your model provider API key (exported as e.g. `OPENAI_API_KEY`)
- Only for B2: [Ollama](https://ollama.com) — `ollama pull qwen2.5:7b`

### No Docker?

Docker appears in exactly one step: SWE-bench's official grader (step 5),
which applies the agent's patch and runs each repo's test suite. That
container is the **sandbox for model-generated code** — dropping it moves the
risk somewhere else rather than deleting it. Options, most-recommended first:

1. **Rootless Podman.** Daemonless, runs as an unprivileged user, and exposes
   a Docker-compatible socket (`podman system service`; set
   `DOCKER_HOST=unix://$XDG_RUNTIME_DIR/podman/podman.sock`). The SWE-bench
   harness generally works unmodified, and security teams that reject the
   Docker daemon often accept rootless Podman.
2. **Grade in CI.** Run steps 1–4 (proxies + agent) anywhere with no
   containers, commit `preds.jsonl`, and let an ephemeral CI runner (GitHub
   Actions etc.) execute the grading job. Your organization already accepts
   CI running arbitrary repo code in throwaway environments — this is the
   same trust boundary. The `ctxc resolve` / `ctxc ab` steps consume the
   grader's output file wherever it was produced.
3. **A throwaway VM** dedicated to grading, wiped afterwards.
4. **Swap the benchmark.** Aider's polyglot benchmark can grade in local
   venvs — no containers, but model-generated code then runs directly on the
   host, and its sessions are mostly too short to engage compression. Only
   reasonable as a does-no-harm check, on a sacrificial machine.

Note the split: **arms A/B/B2, the proxies, cost capture, and the live
engineer trial need no containers at all.** Docker (or its substitute) is
confined to the offline grading of benchmark predictions.

## 1. Install ctxc and sanity-check it

```bash
git clone https://github.com/benwynia/context-compression
cd context-compression
uv sync
uv run ctxc demo          # should print "ctxc verify — OK"

# optional but recommended (~one cent): confirm your provider accepts
# compressed chains (digest message, truncation markers, tool pairing)
uv run ctxc smoke --upstream $UP --model <your-model>
```

## 2. Start the three proxies (three terminals, or `&` each)

All three point at the same provider; only compression differs:

```bash
export UP=https://api.your-provider.com          # your provider's base URL

uv run ctxc proxy --upstream $UP --budget 60k --passthrough \
  --record runs/A/sessions  --port 8791          # arm A: control
uv run ctxc proxy --upstream $UP --budget 60k \
  --record runs/B/sessions  --port 8790          # arm B: deterministic
uv run ctxc proxy --upstream $UP --budget 60k \
  --summarizer-url http://localhost:11434/v1 --summarizer-model qwen2.5:7b \
  --record runs/B2/sessions --port 8792          # arm B2: + local 7B digests
```

## 3. Pick the task list — BEFORE running anything

```bash
# e.g. first 50 SWE-bench Lite instance ids, fixed forever:
python -c "from datasets import load_dataset; \
  [print(r['instance_id']) for r in sorted(load_dataset('princeton-nlp/SWE-bench_Lite', split='test'), key=lambda r: r['instance_id'])[:50]]" \
  > instances.txt
```

Never change this list after seeing results.

## 4. Run the agent — once per arm

Any agent that speaks OpenAI-compatible chat/completions works
([mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) is the
simplest; SWE-agent and OpenHands also take a custom base URL — check your
harness's docs for the exact flag). The pattern, per arm:

```bash
export ARM=B PORT=8790                            # repeat with A/8791, B2/8792
export OPENAI_BASE_URL=http://localhost:$PORT/v1  # litellm-based agents honor this

while read ID; do
  run-your-agent --instance "$ID" \
    --header "x-ctxc-session-id: $ID" \
    --predictions runs/$ARM/preds.jsonl           # harness-specific flags
  uv run ctxc scrape --proxy http://localhost:$PORT \
    --task-id "$ID" --out runs/$ARM/results       # cost row for this task
done < instances.txt
```

(If your harness can't add a header, ctxc falls back to keying sessions by the
task's first message — distinct per SWE-bench instance, so this still works.)

## 5. Grade with the official harness, merge verdicts

```bash
# grade each arm's predictions (see swebench docs for your version):
python -m swebench.harness.run_evaluation \
  --predictions_path runs/$ARM/preds.jsonl --run_id $ARM ...
# produce one resolved id per line (jq path depends on swebench version), then:
uv run ctxc resolve runs/$ARM/results --ids-file runs/$ARM/resolved_ids.txt
```

The `resolved` field must come from the grader — nowhere else.

## 6. Compare

```bash
uv run ctxc ab runs/B/results  runs/A/results --rates rates.json --model <model>
uv run ctxc ab runs/B2/results runs/A/results --rates rates.json --model <model>
uv run ctxc ab runs/B2/results runs/B/results --rates rates.json --model <model>
```

Each report shows resolved rates with an exact McNemar test (did compression
hurt task success?), token/AIC savings with a per-task confidence interval,
and the **engaged segment** — tasks where compression actually fired. If it
says `WARNING: compression never engaged`, your budget was above every chain;
lower `--budget` and rerun.

Read `docs/AB-TESTING.md` before quoting results — it lists the pitfalls that
invalidate runs (post-hoc task selection, dropping infra failures from one arm
only, excluding context-cap deaths).
