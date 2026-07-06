"""Advisory-eviction feasibility probe (pre-shim, offline).

Question: when shown a real agent transcript-in-progress with evictable
blocks labeled by ID, does a frontier model's "safe to evict now" judgment
beat blind policies?

Method: replay recorded rung-8 control sessions. At decision points (50% and
80% of assistant turns), label every evictable block (tool results outside
the protected head and the recent window), ask gpt-5.4-mini which IDs are no
longer needed, then score each directive against the REAL rest of the
transcript: an eviction is HARMFUL if a distinctive line from that block
(present nowhere else in the surviving context) reappears verbatim in a
later assistant message or tool-call argument — i.e. the agent actually
used that exact content afterward.

Baselines at matched token volume: random subsets (20 trials) and
oldest-first. Ground truth is a conservative proxy: verbatim-reuse only,
so semantic dependence without quoting is missed — fine for a first read.

Usage: uv run python scripts/advisory_probe.py [--sessions 15] [--model gpt-5.4-mini]
"""
import argparse
import glob
import json
import os
import random
import re
import statistics

import httpx

from ctxc.tokens import TokenCounter

RECENT_ROUNDS = 2          # never offer the last N tool rounds for eviction
MIN_LINE = 24              # distinctive-line threshold (chars)
DECISION_FRACS = (0.5, 0.8)

_SYS = """You are the context manager for a coding agent working on a software task.
Below is the agent's conversation so far. Some blocks (old tool outputs) are
labeled like [BLOCK id=t12 tokens=840]. Identify blocks whose content is NO
LONGER NEEDED to finish the task (e.g. superseded reads, resolved errors,
exploration that led nowhere). The agent will NOT be able to see evicted
content again, so only evict blocks you are confident are done with.
Evicting nothing is a valid answer.
Reply with JSON only: {"evict": [{"id": "t12", "reason": "three words max"}, ...]}"""

_SYS_KEEP = """You are the context manager for a coding agent working on a software task.
Below is the agent's conversation so far. Some blocks (old tool outputs) are
labeled like [BLOCK id=t12 tokens=840]. Decide which blocks must be KEPT
because the agent may still quote, re-check, or rely on their exact content
to finish the task (file contents it will edit, error messages not yet fixed,
reference output). Every block you do NOT list will be permanently evicted.
Be careful: an over-aggressive eviction breaks the agent; keeping a block is
cheap.
Reply with JSON only: {"keep": [{"id": "t12", "reason": "three words max"}, ...]}"""


def norm(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def distinctive_lines(text: str) -> set[str]:
    return {norm(l) for l in text.splitlines() if len(norm(l)) >= MIN_LINE}


def later_text(messages: list, start: int) -> str:
    """Everything the AGENT produced after the decision point: assistant text
    and tool-call arguments (edits quote file content there)."""
    parts = []
    for m in messages[start:]:
        if m.get("role") != "assistant":
            continue
        if m.get("content"):
            parts.append(str(m["content"]))
        for tc in m.get("tool_calls") or []:
            parts.append(tc.get("function", {}).get("arguments") or "")
    return "\n".join(norm(l) for p in parts for l in p.splitlines())


def candidates(messages: list, upto: int, counter: TokenCounter) -> list[dict]:
    """Evictable blocks in messages[:upto]: tool results outside the head and
    outside the last RECENT_ROUNDS assistant rounds."""
    assistant_idx = [i for i, m in enumerate(messages[:upto]) if m.get("role") == "assistant"]
    recent_cut = assistant_idx[-RECENT_ROUNDS] if len(assistant_idx) >= RECENT_ROUNDS else upto
    out = []
    for i, m in enumerate(messages[:upto]):
        if m.get("role") != "tool" or i >= recent_cut:
            continue
        content = str(m.get("content") or "")
        toks = counter.count_text(content)
        if toks < 100:
            continue  # not worth a directive
        out.append({"id": f"t{i}", "idx": i, "tokens": toks, "content": content})
    return out


def build_prompt(messages: list, upto: int, cands: list[dict]) -> str:
    by_idx = {c["idx"]: c for c in cands}
    lines = []
    for i, m in enumerate(messages[:upto]):
        role = m.get("role")
        content = str(m.get("content") or "")
        if role == "assistant" and m.get("tool_calls"):
            calls = "; ".join(
                f"{tc['function']['name']}({tc['function']['arguments'][:200]})"
                for tc in m["tool_calls"])
            content = (content + "\n" if content else "") + f"<calls: {calls}>"
        if i in by_idx:
            c = by_idx[i]
            lines.append(f"[BLOCK id={c['id']} tokens={c['tokens']}]\n{role}: {content}")
        else:
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def harmful(block: dict, all_early: list[dict], messages: list, upto: int) -> bool:
    """A distinctive line of this block (unique among the pre-decision context)
    reappears in later agent output."""
    mine = distinctive_lines(block["content"])
    others = set()
    for j, m in enumerate(messages[:upto]):
        if m.get("role") == "tool" and f"t{j}" != block["id"]:
            others |= distinctive_lines(str(m.get("content") or ""))
        elif m.get("role") != "tool":
            others |= distinctive_lines(str(m.get("content") or ""))
    unique = mine - others
    if not unique:
        return False
    later = later_text(messages, upto)
    return any(l in later for l in unique)


def ask_model(client: httpx.Client, model: str, prompt: str,
              invert: bool, valid_ids: set[str]) -> tuple[list[str], int]:
    """Returns (ids_to_evict, n_invalid_ids_referenced)."""
    r = client.post("https://api.openai.com/v1/chat/completions", json={
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": _SYS_KEEP if invert else _SYS},
                     {"role": "user", "content": prompt}],
    }, headers={"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        timeout=180)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"] or "{}"
    try:
        key = "keep" if invert else "evict"
        listed = [e["id"] for e in (json.loads(raw).get(key) or [])
                  if isinstance(e, dict) and e.get("id")]
    except json.JSONDecodeError:
        listed = []
    bad = sum(1 for i in listed if i not in valid_ids)
    listed = [i for i in listed if i in valid_ids]
    if invert:
        return [i for i in sorted(valid_ids) if i not in listed], bad
    return listed, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=15)
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--invert", action="store_true",
                    help="ask for a KEEP list; evict the complement")
    ap.add_argument("--out", default="runs/advisory_probe_results.json")
    args = ap.parse_args()

    counter = TokenCounter()
    rng = random.Random(args.seed)
    files = sorted(glob.glob("runs/main/A/sessions/*.json"))
    # 5 deepest + random rest, for depth variety
    sized = sorted(files, key=lambda f: -counter.count_chain(json.load(open(f))["messages"]))
    pick = sized[:5] + rng.sample([f for f in sized[5:]], k=max(0, args.sessions - 5))

    client = httpx.Client()
    rows = []
    bad_ids = no_ops = calls = 0
    for f in pick:
        messages = json.load(open(f))["messages"]
        a_idx = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        if len(a_idx) < 6:
            continue
        for frac in DECISION_FRACS:
            upto = a_idx[int(len(a_idx) * frac)]
            cands = candidates(messages, upto, counter)
            if len(cands) < 3:
                continue
            calls += 1
            valid_ids = {c["id"] for c in cands}
            chosen, bad = ask_model(client, args.model,
                                    build_prompt(messages, upto, cands),
                                    args.invert, valid_ids)
            bad_ids += bad
            if not chosen:
                no_ops += 1
                continue
            by_id = {c["id"]: c for c in cands}
            harm = {c["id"]: harmful(c, cands, messages, upto) for c in cands}
            model_toks = sum(by_id[i]["tokens"] for i in chosen)
            model_harm = sum(harm[i] for i in chosen)

            def volume_match(ids_ordered):
                take, tot = [], 0
                for cid in ids_ordered:
                    if tot >= model_toks:
                        break
                    take.append(cid); tot += by_id[cid]["tokens"]
                return take

            rand_rates = []
            for _ in range(20):
                sel = volume_match(rng.sample(list(valid_ids), k=len(valid_ids)))
                rand_rates.append(sum(harm[i] for i in sel) / max(len(sel), 1))
            oldest = volume_match([c["id"] for c in cands])  # cands are in index order
            rows.append({
                "session": f.split("/")[-1].replace(".json", ""),
                "frac": frac,
                "cands": len(cands),
                "cand_tokens": sum(c["tokens"] for c in cands),
                "evicted": len(chosen),
                "evicted_tokens": model_toks,
                "harmful_model": model_harm,
                "harm_rate_model": model_harm / len(chosen),
                "harm_rate_random": statistics.mean(rand_rates),
                "harm_rate_oldest": sum(harm[i] for i in oldest) / max(len(oldest), 1),
            })
            r = rows[-1]
            print(f"{r['session'][:34]:34s} @{frac:.0%} cands={r['cands']:2d} "
                  f"evicted={r['evicted']:2d} ({r['evicted_tokens']:5d} tok, "
                  f"{100*r['evicted_tokens']/max(r['cand_tokens'],1):3.0f}% of evictable) "
                  f"harm: model={r['harmful_model']}/{r['evicted']} "
                  f"random={r['harm_rate_random']:.2f} oldest={r['harm_rate_oldest']:.2f}")

    print(f"\ndecision points: {calls} | no-op answers: {no_ops} | invalid IDs referenced: {bad_ids}")
    if rows:
        print(f"mean harm rate — model: {statistics.mean(r['harm_rate_model'] for r in rows):.3f} | "
              f"random: {statistics.mean(r['harm_rate_random'] for r in rows):.3f} | "
              f"oldest-first: {statistics.mean(r['harm_rate_oldest'] for r in rows):.3f}")
        print(f"mean evicted volume: {statistics.mean(r['evicted_tokens']/max(r['cand_tokens'],1) for r in rows):.1%} of evictable tokens")
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"rows -> {args.out}")


if __name__ == "__main__":
    main()
