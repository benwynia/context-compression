"""ctxc command line: compress | verify | demo | proxy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .aic import DEFAULT_RATE, load_rates
from .compressor import BudgetImpossible, CompressConfig, compress
from .synth import synth_session
from .tokens import TokenCounter
from .verify import render_report, verify_session


def _parse_budget(raw: str) -> int:
    r = raw.strip().lower().replace(",", "")
    mult = 1
    if r.endswith("k"):
        mult, r = 1_000, r[:-1]
    elif r.endswith("m"):
        mult, r = 1_000_000, r[:-1]
    return int(float(r) * mult)


def _load_messages(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("messages", [])
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a message list or {{'messages': [...]}}")
    return data


def _rate(args) -> object:
    if getattr(args, "rates", None) and getattr(args, "model", None):
        table = load_rates(args.rates)
        if args.model in table:
            return table[args.model]
        raise SystemExit(f"model {args.model!r} not in {args.rates}")
    return DEFAULT_RATE


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ctxc", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("compress", help="one-shot compress a session file")
    pc.add_argument("session")
    pc.add_argument("--budget", default="60k")
    pc.add_argument("--keep-recent", type=int, default=8)
    pc.add_argument("--out", help="write compressed messages JSON here")

    pv = sub.add_parser("verify", help="replay-verify a session file")
    pv.add_argument("session")
    pv.add_argument("--budget", default="60k")
    pv.add_argument("--model-cap", default=None)
    pv.add_argument("--rates", help="JSON AIC rates file")
    pv.add_argument("--model", help="model name to look up in --rates")

    pd = sub.add_parser("demo", help="synthesize a long session and verify it")
    pd.add_argument("--budget", default="40k")
    pd.add_argument("--rounds", type=int, default=40)
    pd.add_argument("--model-cap", default="50k")

    pp = sub.add_parser("proxy", help="run the live compression proxy")
    pp.add_argument("--upstream", required=True)
    pp.add_argument("--budget", default="60k")
    pp.add_argument("--port", type=int, default=8790)
    pp.add_argument("--host", default="127.0.0.1")

    args = p.parse_args(argv)
    budget = _parse_budget(args.budget)

    if args.cmd == "compress":
        messages = _load_messages(args.session)
        cfg = CompressConfig(keep_recent=args.keep_recent)
        try:
            res = compress(messages, budget, cfg)
        except BudgetImpossible as e:
            print(f"budget impossible: {e}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "original_tokens": res.original_tokens,
                    "compressed_tokens": res.compressed_tokens,
                    "saved_pct": round(
                        100 * (1 - res.compressed_tokens / max(1, res.original_tokens)), 1
                    ),
                    "stages": res.stages_applied,
                    "evicted_rounds": res.evicted_rounds,
                    "messages": len(res.messages),
                },
                indent=2,
            )
        )
        if args.out:
            Path(args.out).write_text(json.dumps({"messages": res.messages}, indent=1))
            print(f"wrote {args.out}", file=sys.stderr)
        return 0

    if args.cmd == "verify":
        messages = _load_messages(args.session)
        cap = _parse_budget(args.model_cap) if args.model_cap else None
        report = verify_session(messages, budget, rate=_rate(args), model_cap=cap)
        print(render_report(report))
        return 0 if report.ok else 1

    if args.cmd == "demo":
        messages = synth_session(rounds=args.rounds)
        counter = TokenCounter()
        total = counter.count_chain(messages)
        print(f"synthetic session: {len(messages)} messages, {total:,} tokens\n")
        cap = _parse_budget(args.model_cap) if args.model_cap else None
        report = verify_session(messages, budget, counter=counter, model_cap=cap)
        print(render_report(report))
        return 0 if report.ok else 1

    if args.cmd == "proxy":
        import uvicorn

        from .proxy import build_app

        app = build_app(args.upstream, budget)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
