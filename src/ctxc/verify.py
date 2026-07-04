"""Replay verification harness — the plumbing that proves the compressor works.

Replays a recorded (or synthetic) session the way a harness calls the model: one
request per assistant turn, request prefix = every message before it. Each
emission is checked against the invariants; token accounting is cache-aware.

Invariants checked every turn:
  1. structure — valid tool pairing / role order;
  2. budget — the emitted chain never exceeds the budget;
  3. head — system + task messages are verbatim;
  4. prefix stability — between checkpoints the emission extends the previous
     one exactly (this is what keeps the provider prompt cache alive).

Cache model per emission: the longest common message-prefix with the previous
emission counts as cache read; everything after it counts as cache write. A
checkpoint therefore pays a real re-write cost — recompression is not free, and
the report says so instead of hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .aic import DEFAULT_RATE, AicRate, aic_for, usd_for
from .compressor import BudgetImpossible, CompressConfig
from .models import Message, head_len, is_digest, protected_head_end, validate_chain
from .session import SessionCompressor
from .tokens import TokenCounter


@dataclass
class TurnStats:
    turn: int
    original_tokens: int
    emitted_tokens: int
    cache_read: int
    cache_write: int
    checkpoint: bool


@dataclass
class VerifyReport:
    ok: bool = True
    violations: list[str] = field(default_factory=list)
    turns: int = 0
    checkpoints: int = 0
    original_prompt_tokens: int = 0
    emitted_prompt_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    baseline_aic: float = 0.0
    compressed_aic: float = 0.0
    baseline_usd: float = 0.0
    compressed_usd: float = 0.0
    over_cap_before: int = 0
    over_cap_after: int = 0
    model_cap: int | None = None
    budget: int = 0
    per_turn: list[TurnStats] = field(default_factory=list)


def _common_prefix_len(a: list[Message], b: list[Message]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def verify_session(
    messages: list[Message],
    budget: int,
    *,
    config: CompressConfig | None = None,
    counter: TokenCounter | None = None,
    rate: AicRate = DEFAULT_RATE,
    model_cap: int | None = None,
    session_compressor=None,
) -> VerifyReport:
    counter = counter or TokenCounter()
    sc = session_compressor or SessionCompressor(budget, config=config, counter=counter)
    if not hasattr(sc, "checkpoints"):
        # fail at the source: a silent getattr default would misreport every
        # legitimate recompression as "prefix instability without a checkpoint"
        raise TypeError(
            "session_compressor must expose an integer 'checkpoints' attribute "
            "that increments whenever it rewrites (rather than extends) its emission"
        )
    report = VerifyReport(budget=budget, model_cap=model_cap)

    prev_emitted: list[Message] | None = None
    prev_checkpoints = sc.checkpoints

    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant" or i == 0:
            continue
        prefix = messages[:i]
        turn = report.turns
        report.turns += 1
        try:
            emitted = sc.request(prefix)
        except BudgetImpossible as e:
            # an explicit compressor failure is a finding, not a harness crash
            report.violations.append(f"turn {turn}: budget impossible: {e}")
            continue

        checkpoints = sc.checkpoints
        is_checkpoint = checkpoints > prev_checkpoints
        prev_checkpoints = checkpoints

        # -- invariants --------------------------------------------------------
        for err in validate_chain(emitted):
            report.violations.append(f"turn {turn}: structure: {err}")
        emitted_tokens = counter.count_chain(emitted)
        if emitted_tokens > budget:
            report.violations.append(
                f"turn {turn}: budget exceeded: {emitted_tokens} > {budget}"
            )
        h = head_len(prefix)
        if emitted[:h] != prefix[:h]:
            report.violations.append(f"turn {turn}: protected head was rewritten")
        digest_idx = [j for j, m in enumerate(emitted) if is_digest(m)]
        if len(digest_idx) > 1:
            report.violations.append(
                f"turn {turn}: digest malformed: {len(digest_idx)} digest messages"
            )
        elif digest_idx and digest_idx[0] != protected_head_end(emitted):
            report.violations.append(
                f"turn {turn}: digest malformed: at index {digest_idx[0]}, "
                f"expected directly after the protected head"
            )
        if prev_emitted is not None and not is_checkpoint:
            if emitted[: len(prev_emitted)] != prev_emitted:
                report.violations.append(
                    f"turn {turn}: prefix instability without a checkpoint"
                )

        # -- cache-aware accounting -------------------------------------------
        if prev_emitted is None:
            read = 0
        else:
            read = counter.count_chain(emitted[: _common_prefix_len(prev_emitted, emitted)])
        write = emitted_tokens - read
        original_tokens = counter.count_chain(prefix)

        report.original_prompt_tokens += original_tokens
        report.emitted_prompt_tokens += emitted_tokens
        report.cache_read_tokens += read
        report.cache_write_tokens += write
        if model_cap is not None:
            report.over_cap_before += original_tokens > model_cap
            report.over_cap_after += emitted_tokens > model_cap
        report.per_turn.append(
            TurnStats(turn, original_tokens, emitted_tokens, read, write, is_checkpoint)
        )
        prev_emitted = emitted

    report.checkpoints = sc.checkpoints
    report.baseline_aic = aic_for(
        rate, input_tokens=report.original_prompt_tokens, requests=report.turns
    )
    report.compressed_aic = aic_for(
        rate, input_tokens=report.emitted_prompt_tokens, requests=report.turns
    )
    report.baseline_usd = usd_for(report.baseline_aic)
    report.compressed_usd = usd_for(report.compressed_aic)
    report.ok = not report.violations
    return report


def render_report(r: VerifyReport) -> str:
    def pct(base: float, post: float) -> str:
        return f"{100.0 * (base - post) / base:.1f}%" if base else "n/a"

    lines = [
        f"ctxc verify — {'OK' if r.ok else 'FAILED'}",
        f"  turns replayed          : {r.turns}",
        f"  budget                  : {r.budget:,} tokens",
        f"  checkpoints (recompress): {r.checkpoints}",
        "",
        f"  prompt tokens  baseline : {r.original_prompt_tokens:,}",
        f"  prompt tokens  emitted  : {r.emitted_prompt_tokens:,}"
        f"  (saved {pct(r.original_prompt_tokens, r.emitted_prompt_tokens)})",
        f"  cache reads / writes    : {r.cache_read_tokens:,} / {r.cache_write_tokens:,}"
        f"  ({pct(r.emitted_prompt_tokens, r.cache_write_tokens)} of emitted read from cache)",
        "",
        f"  AIC baseline            : {r.baseline_aic:,.1f} AIC (${r.baseline_usd:,.2f})",
        f"  AIC compressed          : {r.compressed_aic:,.1f} AIC (${r.compressed_usd:,.2f})",
        f"  AIC saved               : {r.baseline_aic - r.compressed_aic:,.1f} AIC"
        f" (${r.baseline_usd - r.compressed_usd:,.2f}, {pct(r.baseline_aic, r.compressed_aic)})",
        "  note: requests count is unchanged by compression; under purely",
        "  per-request AIC metering the win is context headroom, not credits.",
    ]
    if r.model_cap is not None:
        lines += [
            "",
            f"  turns over model cap ({r.model_cap:,}): "
            f"{r.over_cap_before} before -> {r.over_cap_after} after",
        ]
    if r.violations:
        lines += ["", "  violations:"] + [f"    - {v}" for v in r.violations[:20]]
        if len(r.violations) > 20:
            lines.append(f"    … and {len(r.violations) - 20} more")
    return "\n".join(lines)
