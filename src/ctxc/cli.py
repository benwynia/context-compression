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


def _add_summarizer_flags(sp) -> None:
    sp.add_argument("--summarizer-url", default=None, metavar="URL",
                    help="OpenAI-compatible endpoint for LLM-written digests "
                         "(e.g. a local Ollama/vLLM: http://localhost:11434/v1)")
    sp.add_argument("--summarizer-model", default=None, metavar="NAME",
                    help="model name at --summarizer-url (e.g. qwen2.5:7b)")
    sp.add_argument("--summarizer-key-env", default=None, metavar="ENV",
                    help="env var holding the endpoint's API key, if it needs one")


def _compress_config(args) -> "CompressConfig | None":
    url = getattr(args, "summarizer_url", None)
    model = getattr(args, "summarizer_model", None)
    if bool(url) != bool(model):
        raise SystemExit("--summarizer-url and --summarizer-model must be given together")
    if not url:
        return None
    from .summarize import LlmSummarizer

    return CompressConfig(
        summarizer=LlmSummarizer(url, model, api_key_env=args.summarizer_key_env)
    )


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
    _add_summarizer_flags(pv)

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

    ps = sub.add_parser("scrape", help="write one per-task result row from a running proxy")
    ps.add_argument("--proxy", required=True, help="proxy base url, e.g. http://localhost:8790")
    ps.add_argument("--task-id", required=True, help="the x-ctxc-session-id the task ran under")
    ps.add_argument("--out", required=True, metavar="DIR", help="results dir for this arm")

    pr = sub.add_parser("resolve", help="merge grader verdicts into result rows")
    pr.add_argument("results_dir")
    pr.add_argument("--ids-file", required=True,
                    help="file with one RESOLVED task_id per line (from the grader)")

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
                         "for `ctxc verify` (secret-looking strings are redacted)")
    pp.add_argument("--record-raw", action="store_true",
                    help="disable redaction in --record files (transcripts may "
                         "contain keys/passwords from tool output — handle with care)")
    _add_summarizer_flags(pp)

    pi = sub.add_parser("import", help="convert real transcripts (Claude Code "
                                       "JSONL, claude.ai export) to session files")
    pi.add_argument("source", help="a .jsonl session/subagent transcript, a "
                                   "conversations.json export, or a session file")
    pi.add_argument("--out", required=True, metavar="DIR",
                    help="directory for the converted session .json file(s)")

    pf = sub.add_parser("fleet", help="sweep a folder of transcripts: how would "
                                      "each session have done under compression?")
    pf.add_argument("root", help="directory to walk (e.g. ~/.claude/projects)")
    pf.add_argument("--budget", default="60k")
    pf.add_argument("--limit", type=int, default=None, help="max files to process")

    pb = sub.add_parser("probe", help="retention probes: plant facts, compress, "
                                      "measure survival (and retrieval with --live)")
    pb.add_argument("session")
    pb.add_argument("--budget", default="60k")
    pb.add_argument("--n", type=int, default=8, help="number of probes")
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--style", choices=["note", "plain"], default="note",
                    help="'note' = salient-shaped facts (codes, NOTE markers); "
                         "'plain' = pattern-free prose, measures residual loss "
                         "the salience heuristics cannot see")
    pb.add_argument("--live", action="store_true",
                    help="also ask a real model to retrieve each fact (paired "
                         "compressed vs original)")
    pb.add_argument("--upstream", default="https://api.openai.com")
    pb.add_argument("--model", default="gpt-4o-mini")
    pb.add_argument("--key-env", default="OPENAI_API_KEY")

    pk = sub.add_parser("smoke", help="one-cent live check that a real provider "
                                      "accepts compressed chains")
    pk.add_argument("--upstream", required=True)
    pk.add_argument("--model", required=True)
    pk.add_argument("--key-env", default="OPENAI_API_KEY",
                    help="env var holding the provider API key")
    pk.add_argument("--budget", default="800")

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
        report = verify_session(
            messages, budget, config=_compress_config(args), rate=_rate(args),
            model_cap=cap,
        )
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

    if args.cmd == "import":
        from .ingest import detect_and_convert

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        sessions = detect_and_convert(args.source)
        for name, msgs in sessions.items():
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[:80]
            path = out / f"{safe}.json"
            path.write_text(json.dumps({"messages": msgs}, ensure_ascii=False))
            print(f"wrote {path}  ({len(msgs)} messages)", file=sys.stderr)
        return 0

    if args.cmd == "fleet":
        from .fleet import render_fleet, sweep

        report = sweep(args.root, budget, limit=args.limit)
        print(render_fleet(report))
        return 0

    if args.cmd == "probe":
        import os

        import httpx

        from .probe import render_probe_report, run_probes

        messages = _load_messages(args.session)
        ask = None
        if args.live:
            base = args.upstream.rstrip("/")
            if not base.endswith("/v1"):
                base += "/v1"
            key = os.environ.get(args.key_env, "")
            http = httpx.Client(timeout=60.0)

            def ask(context, question, _base=base, _key=key, _http=http):
                body = {"model": args.model, "temperature": 0,
                        "max_completion_tokens": 30,
                        "messages": context + [{"role": "user", "content": question}]}
                headers = {"authorization": f"Bearer {_key}"} if _key else {}
                r = _http.post(f"{_base}/chat/completions", json=body, headers=headers)
                if r.status_code == 400 and "max_completion_tokens" in r.text:
                    body.pop("max_completion_tokens")
                    body["max_tokens"] = 30
                    r = _http.post(f"{_base}/chat/completions", json=body, headers=headers)
                r.raise_for_status()
                return (r.json()["choices"][0]["message"].get("content") or "")

        report = run_probes(messages, budget, n=args.n, seed=args.seed,
                            style=args.style, ask=ask)
        print(render_probe_report(report))
        return 0

    if args.cmd == "smoke":
        from .smoke import run_smoke

        result = run_smoke(args.upstream, args.model, key_env=args.key_env,
                           budget=budget)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1

    if args.cmd == "scrape":
        from .ab import scrape_row

        row = scrape_row(args.proxy, args.task_id)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{args.task_id}.json"
        path.write_text(json.dumps(row, indent=1))
        print(f"wrote {path}", file=sys.stderr)
        return 0

    if args.cmd == "resolve":
        from .ab import mark_resolved

        ids = {
            line.strip()
            for line in Path(args.ids_file).read_text().splitlines()
            if line.strip()
        }
        n = mark_resolved(args.results_dir, ids)
        print(f"updated {n} rows ({len(ids)} resolved ids)", file=sys.stderr)
        return 0

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

        app = build_app(args.upstream, budget, config=_compress_config(args),
                        shadow=args.shadow, passthrough=args.passthrough,
                        record_dir=args.record, record_raw=args.record_raw)
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
