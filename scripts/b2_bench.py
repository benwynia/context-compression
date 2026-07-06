"""B2 microbenchmark: capture the exact summarizer inputs from real rung-8
sessions, then time a local 7B on them via Ollama's native API (which reports
precise prefill/generation token counts and durations).

Usage: uv run python scripts/b2_bench.py [--model qwen2.5:7b] [--sessions N]
"""
import argparse
import glob
import json
import statistics

import httpx

from ctxc.compressor import CompressConfig
from ctxc.session import SessionCompressor
from ctxc.summarize import _SYSTEM_PROMPT
from ctxc.tokens import TokenCounter


class CaptureSummarizer:
    """Records the digest lines the compressor would hand a real summarizer,
    then raises so the deterministic fallback is used (no behavior change)."""

    def __init__(self):
        self.captured: list[list[str]] = []

    def __call__(self, lines: list[str]) -> str:
        self.captured.append(list(lines))
        raise RuntimeError("capture only")


def capture_inputs(session_files: list[str], budget: int) -> list[list[str]]:
    counter = TokenCounter()
    inputs: list[list[str]] = []
    for f in session_files:
        msgs = json.load(open(f))["messages"]
        cap = CaptureSummarizer()
        sc = SessionCompressor(budget, config=CompressConfig(summarizer=cap),
                               counter=counter)
        for i, m in enumerate(msgs):
            if m.get("role") == "assistant" and i > 0:
                try:
                    sc.request(msgs[:i])
                except Exception:
                    pass
        inputs.extend(cap.captured)
    return inputs


def bench_one(client: httpx.Client, model: str, lines: list[str],
              target_tokens: int = 300, max_input_chars: int = 24_000) -> dict:
    text = "\n".join(lines)
    if len(text) > max_input_chars:
        text = "(earliest evicted turns omitted)\n" + text[-max_input_chars:]
    r = client.post("http://localhost:11434/api/chat", json={
        "model": model,
        "stream": False,
        "options": {"temperature": 0, "num_predict": int(target_tokens * 1.5)},
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT.format(target=target_tokens)},
            {"role": "user", "content": text},
        ],
    }, timeout=300)
    r.raise_for_status()
    d = r.json()
    return {
        "input_chars": len(text),
        "prompt_tokens": d.get("prompt_eval_count", 0),
        "prefill_s": d.get("prompt_eval_duration", 0) / 1e9,
        "gen_tokens": d.get("eval_count", 0),
        "gen_s": d.get("eval_duration", 0) / 1e9,
        "total_s": d.get("total_duration", 0) / 1e9,
        "summary_preview": (d.get("message") or {}).get("content", "")[:200],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--budget", default=5000, type=int)
    args = ap.parse_args()

    files = sorted(glob.glob("runs/main/A/sessions/*.json"))[: args.sessions]
    inputs = capture_inputs(files, args.budget)
    if not inputs:
        print("no summarizer inputs captured — lower the budget")
        return
    sizes = sorted(len("\n".join(l)) for l in inputs)
    print(f"captured {len(inputs)} real summarizer inputs from {len(files)} sessions")
    print(f"input sizes (chars): min={sizes[0]} median={sizes[len(sizes)//2]} max={sizes[-1]}")

    client = httpx.Client()
    # warm the model (first call pays the load-from-disk cost; report it apart)
    warm = bench_one(client, args.model, ["warmup: read a file, edited a file"])
    print(f"cold-start call (model load + tiny input): {warm['total_s']:.1f}s")

    results = []
    for i, lines in enumerate(inputs):
        r = bench_one(client, args.model, lines)
        results.append(r)
        print(f"  [{i}] in={r['prompt_tokens']:5d} tok  prefill={r['prefill_s']:5.1f}s "
              f"({r['prompt_tokens']/max(r['prefill_s'],1e-9):6.0f} tok/s)  "
              f"gen={r['gen_tokens']:3d} tok in {r['gen_s']:4.1f}s "
              f"({r['gen_tokens']/max(r['gen_s'],1e-9):5.1f} tok/s)  total={r['total_s']:5.1f}s")

    tot = [r["total_s"] for r in results]
    print(f"\nper-checkpoint wall time: min={min(tot):.1f}s "
          f"median={statistics.median(tot):.1f}s max={max(tot):.1f}s")
    pf = [r["prompt_tokens"] / max(r["prefill_s"], 1e-9) for r in results]
    gn = [r["gen_tokens"] / max(r["gen_s"], 1e-9) for r in results]
    print(f"prefill: median {statistics.median(pf):.0f} tok/s | "
          f"generation: median {statistics.median(gn):.1f} tok/s")
    print("\nsample summary output:\n" + results[len(results)//2]["summary_preview"])


if __name__ == "__main__":
    main()
