"""Pure ``compress()`` with escalation until the budget is met.

The contract (enforced, and tested):

* the returned chain counts <= budget, or ``BudgetImpossible`` is raised —
  never a silent overshoot;
* the protected head (system + task, extended through any tool results that
  answer a head-ending assistant) is verbatim;
* the returned chain is structurally valid (tool pairing, role order);
* at most one digest message exists, directly after the head — recompressing an
  already-compressed chain folds the old digest instead of nesting a new one;
* an optional ``summarizer`` hook is called at most once per compress, and its
  output is capped like the deterministic digest, so the hook can neither blow
  the budget nor get invoked per eviction probe.

Escalation floors scale with the budget so small effective budgets (e.g. after
the proxy subtracts tools-schema tokens) don't hit spurious ``BudgetImpossible``
on hardcoded rungs.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .models import (
    Message,
    content_text,
    copy_chain,
    is_digest,
    protected_head_end,
    round_boundary_at_or_before,
    validate_chain,
)
from .strategies import (
    DEFAULT_SALIENCE,
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
    # salience: lines matching this survive truncation (from the cut middle)
    # and eviction (as digest "keep:" lines) — retention by importance, not
    # position. None disables. See strategies.DEFAULT_SALIENCE.
    salience_pattern: str | None = DEFAULT_SALIENCE
    digest_salient_lines: int = 3      # max salient lines kept per evicted round
    # pinning: content matching this is IMMUNE to truncation and eviction — the
    # explicit escape hatch for critical facts ("ctxc:pin" anywhere in a
    # message). Over-pinning can make a budget impossible; that failure is
    # loud, never silent.
    pin_pattern: str | None = r"ctxc:pin"
    # optional LLM summarizer hook: receives digest lines, returns the digest
    # body. Called at most once per compress(); output capped at the digest cap.
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
    digest_cap: int = 0   # tokens; set per level in _levels()


def _levels(cfg: CompressConfig, budget: int) -> list[_Level]:
    # Char floors scale down with the budget (~4 chars/token) so a few-hundred-
    # token budget still has rungs below it instead of fixed 200-char cliffs.
    floor = max(32, min(200, budget // 20))
    err_floor = max(48, min(400, budget // 10))
    # salient "keep:" lines make digest tokens carry real information, so the
    # caps are budget-proportional rather than a fixed 100-token cliff
    digest_default = max(48, min(int(budget * cfg.digest_budget_share), budget // 3))
    digest_tight = max(32, min(300, budget // 8))
    lvl1 = max(floor, cfg.tool_result_max_chars // 4)
    lvl1e = max(err_floor, cfg.error_result_max_chars // 4)
    return [
        _Level(cfg.tool_result_max_chars, cfg.error_result_max_chars,
               cfg.keep_recent, digest_cap=digest_default),
        _Level(lvl1, lvl1e, cfg.keep_recent, digest_cap=digest_default),
        _Level(lvl1, lvl1e, None, digest_cap=digest_default),
        _Level(floor, err_floor, None, tail_trunc=err_floor, digest_cap=digest_tight),
        _Level(min(floor, 120), min(floor, 120), None,
               tail_trunc=min(floor, 120), digest_cap=digest_tight),
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
        # copies, so the no-op path has the same aliasing contract as the
        # compressed paths (callers may mutate the result safely)
        return CompressResult(copy_chain(messages), original_tokens, original_tokens)

    h = protected_head_end(messages)
    body_start = h
    prior_digest_lines: list[str] = []
    if body_start < len(messages) and is_digest(messages[body_start]):
        # fold the previous checkpoint's digest instead of nesting
        prior = content_text(messages[body_start]).splitlines()
        prior_digest_lines = prior[1:]  # drop the marker line
        body_start += 1

    error_re = re.compile(cfg.error_pattern)
    salience_re = re.compile(cfg.salience_pattern) if cfg.salience_pattern else None
    pin_re = re.compile(cfg.pin_pattern) if cfg.pin_pattern else None
    for level in _levels(cfg, budget):
        result = _attempt(
            messages, budget, cfg, counter, level,
            h=h, body_start=body_start,
            prior_digest_lines=prior_digest_lines, error_re=error_re,
            salience_re=salience_re, pin_re=pin_re,
            original_tokens=original_tokens,
        )
        if result is not None:
            violations = validate_chain(result.messages)
            if violations:  # defensive: never return a broken chain
                raise AssertionError(f"compressor produced invalid chain: {violations}")
            return result
    raise BudgetImpossible(
        f"cannot reach budget={budget}: the protected head + last round exceed "
        f"it even at the deepest escalation level"
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
    salience_re: re.Pattern[str] | None,
    pin_re: re.Pattern[str] | None,
    original_tokens: int,
) -> CompressResult | None:
    """One escalation level. Returns a result iff it fits the budget."""
    msgs: list[Message] = copy_chain(original)
    stages: list[str] = []
    evicted = 0

    if level.tail_len is None:
        tail_start = round_boundary_at_or_before(msgs, len(msgs) - 1, body_start)
    else:
        tail_start = round_boundary_at_or_before(
            msgs, len(msgs) - level.tail_len, body_start
        )
    tail_start = max(tail_start, body_start)

    def digest_for(lines: list[str], final: bool) -> Message | None:
        if not lines:
            return None
        # the summarizer runs only on the final build — never inside probes
        return build_digest_message(
            lines, counter=counter, token_cap=level.digest_cap,
            summarizer=cfg.summarizer if final else None,
            salience=salience_re,
        )

    def assemble(work: list[Message], lines: list[str], *, final: bool = False) -> list[Message]:
        digest = digest_for(lines, final)
        body = work[body_start:]
        return work[:h] + ([digest] if digest else []) + body

    def total(work: list[Message], lines: list[str]) -> int:
        return counter.count_chain(assemble(work, lines))

    def finish(work: list[Message], lines: list[str]) -> CompressResult | None:
        """Single exit: return a result iff the deterministic build fits. Only
        then is the summarizer invoked — once per compress, on the succeeding
        level — and its digest is kept only if it also fits, so the hook can
        improve the digest but never break the budget or run on doomed levels."""
        out = assemble(work, lines, final=False)
        n = counter.count_chain(out)
        if n > budget:
            return None
        if cfg.summarizer is not None and lines:
            candidate = assemble(work, lines, final=True)
            n2 = counter.count_chain(candidate)
            if n2 <= budget:
                out, n = candidate, n2
        return CompressResult(out, original_tokens, n, stages, evicted_rounds=evicted)

    # stage 1: truncate unprotected tool results
    if truncate_tool_results(
        msgs, body_start, tail_start,
        max_chars=level.trunc, error_max_chars=level.err_trunc, error_re=error_re,
        salience=salience_re, pin_re=pin_re,
    ):
        stages.append("truncate")
    digest_lines = list(prior_digest_lines)
    if total(msgs, digest_lines) <= budget:
        return finish(msgs, digest_lines)

    # stage 2: elide duplicate tool results (last occurrence survives)
    if elide_duplicate_results(msgs, body_start, tail_start):
        stages.append("dedupe")
    if total(msgs, digest_lines) <= budget:
        return finish(msgs, digest_lines)

    # stage 3: evict whole rounds oldest-first into the digest
    msgs, digest_lines, evicted = evict_rounds(
        msgs, body_start, tail_start,
        fits=lambda work, lines: total(work, lines) <= budget,
        digest_lines=digest_lines,
        salience=salience_re, max_salient=cfg.digest_salient_lines, pin_re=pin_re,
    )
    if evicted:
        stages.append("evict")

    # stage 4 (deep levels only): truncate the protected tail's tool results too
    if level.tail_trunc is not None and total(msgs, digest_lines) > budget:
        if truncate_tool_results(
            msgs, body_start, len(msgs),
            max_chars=level.tail_trunc, error_max_chars=level.tail_trunc,
            error_re=error_re, salience=salience_re, pin_re=pin_re,
        ):
            stages.append("truncate-tail")

    return finish(msgs, digest_lines)
