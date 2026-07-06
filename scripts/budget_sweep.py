"""Budget sweep: find the prune point for YOUR sessions, offline and free.

Replays a directory of recorded sessions (ctxc session files — import real
VS Code Copilot chats with `ctxc import`, or record live traffic with
`ctxc proxy --shadow --record DIR`) through the deterministic compressor at
a range of budgets, and reports per budget:

  * engagement — how many sessions would compress at all
  * checkpoints per session (mean / max) — churn risk
  * token savings and cache-aware dollar savings vs. an uncompressed baseline

Replay is deterministic, so this costs nothing and is exact for the static
part of the trade-off. It does NOT capture behavioral feedback (a live agent
reacts to compression — rung 12) — deploy with the thrash guard, which
bounds that term, and validate the chosen point with a small live pilot.

Usage:
  uv run python scripts/budget_sweep.py SESSIONS_DIR \
      [--budgets 5k,8k,10k,15k,20k,30k] [--rates scripts/rates.json --model gpt-5.4]
"""
import argparse
import glob
import json
import statistics
import sys

from ctxc.aic import DEFAULT_RATE, load_rates
from ctxc.tokens import TokenCounter
from ctxc.verify import verify_session


def parse_budget(raw: str) -> int:
    raw = raw.strip().lower()
    return int(float(raw[:-1]) * 1000) if raw.endswith("k") else int(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions_dir")
    ap.add_argument("--budgets", default="5k,8k,10k,15k,20k,30k")
    ap.add_argument("--rates", help="JSON AIC rates file (see scripts/rates.json)")
    ap.add_argument("--model", help="model name to look up in --rates")
    ap.add_argument("--json", dest="json_out", help="also write results as JSON")
    args = ap.parse_args()

    rate = DEFAULT_RATE
    if args.rates and args.model:
        rate = load_rates(args.rates)[args.model]

    counter = TokenCounter()
    files = sorted(glob.glob(f"{args.sessions_dir.rstrip('/')}/*.json"))
    sessions = []
    for f in files:
        msgs = json.load(open(f)).get("messages") or []
        if sum(1 for m in msgs if m.get("role") == "assistant") >= 2:
            sessions.append(msgs)
    if not sessions:
        sys.exit(f"no usable sessions in {args.sessions_dir}")

    depths = sorted(counter.count_chain(m) for m in sessions)
    print(f"{len(sessions)} sessions | chain depth: "
          f"median={depths[len(depths)//2]:,} p75={depths[3*len(depths)//4]:,} "
          f"max={depths[-1]:,}")
    print(f"{'budget':>8} {'engaged':>9} {'ckpt mean':>9} {'ckpt max':>8} "
          f"{'tokens saved':>12} {'$ saved (cache-aware)':>21}")

    rows = []
    for budget in (parse_budget(b) for b in args.budgets.split(",")):
        engaged = 0
        ckpts, base, comp, tot_o, tot_e = [], 0.0, 0.0, 0, 0
        for msgs in sessions:
            r = verify_session(msgs, budget, counter=counter, rate=rate)
            ckpts.append(r.checkpoints)
            engaged += r.checkpoints > 0
            base += r.baseline_aic_cached or r.baseline_aic
            comp += r.compressed_aic_cached or r.compressed_aic
            tot_o += r.original_prompt_tokens
            tot_e += r.emitted_prompt_tokens
        row = {
            "budget": budget,
            "engaged": engaged,
            "sessions": len(sessions),
            "checkpoints_mean": round(statistics.mean(ckpts), 2),
            "checkpoints_max": max(ckpts),
            "tokens_saved_pct": round(100 * (tot_o - tot_e) / tot_o, 1) if tot_o else 0,
            "usd_saved_pct": round(100 * (base - comp) / base, 1) if base else 0,
        }
        rows.append(row)
        print(f"{budget:>8,} {engaged:>5}/{len(sessions):<3} "
              f"{row['checkpoints_mean']:>9.2f} {row['checkpoints_max']:>8} "
              f"{row['tokens_saved_pct']:>11.1f}% {row['usd_saved_pct']:>+20.1f}%")

    print("\nReading the table: pick the largest budget whose $ savings you still")
    print("like, with checkpoint max in low single digits. Deploy WITH the thrash")
    print("guard (--thrash-guard) and validate live before trusting the numbers.")
    if args.json_out:
        json.dump(rows, open(args.json_out, "w"), indent=1)
        print(f"rows -> {args.json_out}")


if __name__ == "__main__":
    main()
