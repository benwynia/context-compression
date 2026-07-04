"""Fleet sweep: assess a whole folder of recorded sessions at once.

``ctxc fleet ~/.claude/projects --budget 60k`` walks a directory tree, converts
anything it recognizes (Claude Code ``.jsonl`` transcripts, claude.ai exports,
ctxc session files), replays each session through the verifier, and renders one
row per session plus fleet totals — the "how would my history have done under
compression?" answer in a single command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .compressor import CompressConfig
from .ingest import detect_and_convert
from .tokens import TokenCounter
from .verify import verify_session

# sessions smaller than this aren't worth replaying (greetings, stubs)
MIN_MESSAGES = 6


@dataclass
class FleetRow:
    name: str
    messages: int
    final_chain_tokens: int
    turns: int = 0
    checkpoints: int = 0
    original_tokens: int = 0
    emitted_tokens: int = 0
    ok: bool = True
    error: str | None = None

    @property
    def engaged(self) -> bool:
        return self.checkpoints > 0

    @property
    def saved_pct(self) -> float:
        if not self.original_tokens:
            return 0.0
        return 100.0 * (1 - self.emitted_tokens / self.original_tokens)


@dataclass
class FleetReport:
    budget: int
    rows: list[FleetRow] = field(default_factory=list)
    skipped_small: int = 0
    skipped_unreadable: int = 0


def sweep(
    root: str | Path,
    budget: int,
    *,
    config: CompressConfig | None = None,
    counter: TokenCounter | None = None,
    limit: int | None = None,
) -> FleetReport:
    counter = counter or TokenCounter()
    report = FleetReport(budget=budget)
    paths = sorted(
        p for p in Path(root).rglob("*")
        if p.is_file() and p.suffix in (".jsonl", ".json")
        and "subagents" not in p.parts  # main sessions only; subagents inflate counts
    )
    if limit:
        paths = paths[:limit]
    for path in paths:
        try:
            sessions = detect_and_convert(path)
        except Exception:
            report.skipped_unreadable += 1
            continue
        for name, msgs in sessions.items():
            if len(msgs) < MIN_MESSAGES:
                report.skipped_small += 1
                continue
            row = FleetRow(
                name=name[:48],
                messages=len(msgs),
                final_chain_tokens=counter.count_chain(msgs),
            )
            try:
                r = verify_session(msgs, budget, config=config, counter=counter)
                row.turns = r.turns
                row.checkpoints = r.checkpoints
                row.original_tokens = r.original_prompt_tokens
                row.emitted_tokens = r.emitted_prompt_tokens
                row.ok = r.ok
                if not r.ok:
                    row.error = r.violations[0] if r.violations else "violations"
            except Exception as e:  # a bad session must not kill the sweep
                row.ok = False
                row.error = str(e)[:80]
            report.rows.append(row)
    report.rows.sort(key=lambda r: -r.final_chain_tokens)
    return report


def render_fleet(r: FleetReport) -> str:
    lines = [
        f"ctxc fleet — {len(r.rows)} sessions at budget {r.budget:,}"
        + (f"  (skipped: {r.skipped_small} tiny, {r.skipped_unreadable} unreadable)"
           if r.skipped_small or r.skipped_unreadable else ""),
        "",
        f"  {'session':<48} {'msgs':>5} {'chain':>9} {'turns':>5} "
        f"{'ckpts':>5} {'saved':>7} {'ok':>3}",
    ]
    for row in r.rows:
        lines.append(
            f"  {row.name:<48} {row.messages:>5} {row.final_chain_tokens:>9,} "
            f"{row.turns:>5} {row.checkpoints:>5} {row.saved_pct:>6.1f}% "
            f"{'ok' if row.ok else 'ERR'}"
        )
        if row.error:
            lines.append(f"      error: {row.error}")
    engaged = [x for x in r.rows if x.engaged]
    tot_o = sum(x.original_tokens for x in r.rows)
    tot_e = sum(x.emitted_tokens for x in r.rows)
    lines += [
        "",
        f"  engagement: {len(engaged)}/{len(r.rows)} sessions ever crossed the budget",
        f"  fleet prompt tokens: {tot_o:,} -> {tot_e:,}"
        + (f"  (saved {100 * (1 - tot_e / tot_o):.1f}%)" if tot_o else ""),
    ]
    if engaged:
        eo = sum(x.original_tokens for x in engaged)
        ee = sum(x.emitted_tokens for x in engaged)
        lines.append(
            f"  engaged sessions only: {eo:,} -> {ee:,}"
            + (f"  (saved {100 * (1 - ee / eo):.1f}%)" if eo else "")
        )
    else:
        lines.append(
            "  NOTE: nothing engaged — your sessions live below this budget; "
            "compression would be a no-op on this history."
        )
    return "\n".join(lines)
