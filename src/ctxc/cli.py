"""ctxc command line: compress | verify | demo | proxy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .aic import DEFAULT_RATE, AicRate, load_rates
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


def _rate(args) -> AicRate:
    rates = getattr(args, "rates", None)
    model = getattr(args, "model", None)
    if bool(rates) != bool(model):
        # never silently fall back to the illustrative default when the user
        # asked for real pricing — half-specified means a wrong dollar figure
        raise SystemExit("--rates and --model must be given together")
    if rates and model:
        table = load_rates(rates)
        if model in table:
            return table[model]
        raise SystemExit(f"model {model!r} not in {rates}")
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

    pa = sub.add_parser("ab", help="paired A/B report: ctxc arm vs control arm")
    pa.add_argument("ctxc_results", help="dir of *.json rows (or .jsonl) for the ctxc arm")
    pa.add_argument("control_results", help="dir of *.json rows (or .jsonl) for the control arm")
    pa.add_argument("--rates", help="JSON AIC rates file")
    pa.add_argument("--model", help="model name to look up in --rates")
    pa.add_argument("--json", dest="json_out", help="also write the full report as JSON")

    pp = sub.add_parser("proxy", help="run the live compression proxy")
    pp.add_argument("--upstream", required=True)
    pp.add_argument("--budget", default="60k")
    pp.add_argument("--port", type=int, default=8790)
    pp.add_argument("--host", default="127.0.0.1")
    pp.add_argument("--shadow", action="store_true",
                    help="forward requests UNTOUCHED; measure would-be savings on "
                         "the side (zero-risk pilot; read them at GET /stats)")
    pp.add_argument("--passthrough", action="store_true",
                    help="no compression at all — measurement/recording only. "
                         "Use for the A/B CONTROL arm so both arms share "
                         "identical instrumentation")
    pp.add_argument("--record", default=None, metavar="DIR",
                    help="capture each conversation as a replayable session file "
                         "for `ctxc verify`")

    args = p.parse_args(argv)
    budget = _parse_budget(args.budget) if hasattr(args, "budget") else 0

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

    if args.cmd == "ab":
        from dataclasses import asdict

        from .ab import compare, load_results, render_ab

        report = compare(
            load_results(args.ctxc_results), load_results(args.control_results),
            rate=_rate(args),
        )
        print(render_ab(report))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(asdict(report), indent=2))
        return 0

    if args.cmd == "proxy":
        import uvicorn

        from .proxy import build_app

        app = build_app(args.upstream, budget, shadow=args.shadow,
                        passthrough=args.passthrough, record_dir=args.record)
        mode = ("PASSTHROUGH (control arm: no compression, measuring only)"
                if args.passthrough
                else "SHADOW (traffic untouched, measuring only)" if args.shadow
                else "ACTIVE")
        print(f"ctxc proxy: {mode}; savings at http://{args.host}:{args.port}/stats",
              file=sys.stderr)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
