"""Pure ``compress()`` with escalation until the budget is met.

The contract (enforced, and tested):

* the returned chain counts <= budget, or ``BudgetImpossible`` is raised —
  never a silent overshoot;
* the protected head (system + task) is verbatim;
* the returned chain is structurally valid (tool pairing, role order);
* at most one digest message exists, directly after the head — recompressing an
  already-compressed chain folds the old digest instead of nesting a new one.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .models import (
    Message,
    content_text,
    head_len,
    is_digest,
    round_boundary_at_or_before,
    validate_chain,
)
from .strategies import (
    build_digest_message,
    elide_duplicate_results,
    evict_rounds,
    truncate_tool_results,
)
from .tokens import TokenCounter


class BudgetImpossible(Exception):
    """Even the irreducible core (head + digest + last round) exceeds the budget."""


@dataclass
class CompressConfig:
    keep_recent: int = 8               # most recent messages kept verbatim
    tool_result_max_chars: int = 1200  # stage-1 excerpt size
    error_result_max_chars: int = 4000 # error results keep more context
    error_pattern: str = r"(?i)\berror\b|traceback|\bfailed\b|exit code [1-9]"
    digest_budget_share: float = 0.05  # digest size as a share of the budget
    recompress_to: float = 0.6         # checkpoint hysteresis (see session.py)
    # optional LLM summarizer hook: receives digest lines, returns the digest body
    summarizer: Callable[[list[str]], str] | None = None


@dataclass
class CompressResult:
    messages: list[Message]
    original_tokens: int
    compressed_tokens: int
    stages_applied: list[str] = field(default_factory=list)
    evicted_rounds: int = 0


@dataclass(frozen=True)
class _Level:
    trunc: int
    err_trunc: int
    tail_len: int | None  # None = protect only the last round
    tail_trunc: int | None = None
    digest_cap: int | None = None


def _levels(cfg: CompressConfig) -> list[_Level]:
    return [
        _Level(cfg.tool_result_max_chars, cfg.error_result_max_chars, cfg.keep_recent),
        _Level(max(200, cfg.tool_result_max_chars // 4),
               max(400, cfg.error_result_max_chars // 4), cfg.keep_recent),
        _Level(max(200, cfg.tool_result_max_chars // 4),
               max(400, cfg.error_result_max_chars // 4), None),
        _Level(200, 400, None, tail_trunc=400),
        _Level(120, 120, None, tail_trunc=120, digest_cap=100),
    ]


def compress(
    messages: list[Message],
    budget: int,
    config: CompressConfig | None = None,
    counter: TokenCounter | None = None,
) -> CompressResult:
    cfg = config or CompressConfig()
    counter = counter or TokenCounter()
    original_tokens = counter.count_chain(messages)
    if original_tokens <= budget:
        return CompressResult(list(messages), original_tokens, original_tokens)

    h = head_len(messages)
    body_start = h
    prior_digest_lines: list[str] = []
    if body_start < len(messages) and is_digest(messages[body_start]):
        # fold the previous checkpoint's digest instead of nesting
        prior = content_text(messages[body_start]).splitlines()
        prior_digest_lines = prior[1:]  # drop the marker line
        body_start += 1

    error_re = re.compile(cfg.error_pattern)
    last_result: CompressResult | None = None
    for level in _levels(cfg):
        result = _attempt(
            messages, budget, cfg, counter, level,
            h=h, body_start=body_start,
            prior_digest_lines=prior_digest_lines, error_re=error_re,
            original_tokens=original_tokens,
        )
        last_result = result
        if result.compressed_tokens <= budget:
            violations = validate_chain(result.messages)
            if violations:  # defensive: never return a broken chain
                raise AssertionError(f"compressor produced invalid chain: {violations}")
            return result
    raise BudgetImpossible(
        f"cannot reach budget={budget}: irreducible core is "
        f"{last_result.compressed_tokens if last_result else original_tokens} tokens"
    )


def _attempt(
    original: list[Message],
    budget: int,
    cfg: CompressConfig,
    counter: TokenCounter,
    level: _Level,
    *,
    h: int,
    body_start: int,
    prior_digest_lines: list[str],
    error_re: re.Pattern[str],
    original_tokens: int,
) -> CompressResult:
    msgs: list[Message] = [dict(m) for m in original]
    stages: list[str] = []
    digest_cap = level.digest_cap or max(200, int(budget * cfg.digest_budget_share))

    if level.tail_len is None:
        tail_start = round_boundary_at_or_before(msgs, len(msgs) - 1, body_start)
    else:
        tail_start = round_boundary_at_or_before(
            msgs, len(msgs) - level.tail_len, body_start
        )
    tail_start = max(tail_start, body_start)

    def assemble(work: list[Message], digest_lines: list[str]) -> list[Message]:
        head = work[:h]
        body = work[body_start:]
        if digest_lines:
            digest = build_digest_message(
                digest_lines, counter=counter, token_cap=digest_cap,
                summarizer=cfg.summarizer,
            )
            return head + [digest] + body
        return head + body

    def total(work: list[Message], digest_lines: list[str]) -> int:
        return counter.count_chain(assemble(work, digest_lines))

    # stage 1: truncate unprotected tool results
    if truncate_tool_results(
        msgs, body_start, tail_start,
        max_chars=level.trunc, error_max_chars=level.err_trunc, error_re=error_re,
    ):
        stages.append("truncate")
    digest_lines = list(prior_digest_lines)
    if total(msgs, digest_lines) <= budget:
        out = assemble(msgs, digest_lines)
        return CompressResult(out, original_tokens, counter.count_chain(out), stages)

    # stage 2: elide duplicate tool results (first occurrence survives)
    if elide_duplicate_results(msgs, body_start, tail_start):
        stages.append("dedupe")
    if total(msgs, digest_lines) <= budget:
        out = assemble(msgs, digest_lines)
        return CompressResult(out, original_tokens, counter.count_chain(out), stages)

    # stage 3: evict whole rounds oldest-first into the digest
    msgs, digest_lines, evicted = evict_rounds(
        msgs, body_start, tail_start,
        fits=lambda work, lines: total(work, lines) <= budget,
        digest_lines=digest_lines,
    )
    if evicted:
        stages.append("evict")

    # stage 4 (deep levels only): truncate the protected tail's tool results too
    if level.tail_trunc is not None and total(msgs, digest_lines) > budget:
        if truncate_tool_results(
            msgs, body_start, len(msgs),
            max_chars=level.tail_trunc, error_max_chars=level.tail_trunc,
            error_re=error_re,
        ):
            stages.append("truncate-tail")

    out = assemble(msgs, digest_lines)
    return CompressResult(
        out, original_tokens, counter.count_chain(out), stages, evicted_rounds=evicted
    )
