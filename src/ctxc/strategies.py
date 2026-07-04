"""Compression stages: truncation, duplicate elision, round eviction + digest.

Each stage is a pure helper operating on a mutable working copy of the chain,
touching only the unprotected body range. Structural safety comes from operating
on rounds (see :mod:`ctxc.models`), never on raw message slices.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .models import (
    DIGEST_MARKER,
    DUPLICATE_MARKER,
    Message,
    content_text,
    fingerprint,
    round_starts,
    set_content_text,
)
from .tokens import TokenCounter


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Head+tail excerpt with an explicit elision marker."""
    if len(text) <= max_chars:
        return text, False
    if max_chars <= 0:  # "keep nothing": marker only — never re-emit the body
        return f"[ctxc: truncated {len(text)} chars]", True
    head = int(max_chars * 0.6)
    tail = max_chars - head
    cut = len(text) - head - tail
    kept_tail = text[-tail:] if tail > 0 else ""
    return f"{text[:head]}\n[ctxc: truncated {cut} chars]\n{kept_tail}", True


def truncate_tool_results(
    msgs: list[Message],
    start: int,
    end: int,
    *,
    max_chars: int,
    error_max_chars: int,
    error_re: re.Pattern[str],
) -> bool:
    """Stage 1: cut unprotected tool results down to excerpts. Errors keep more —
    they tend to get referenced later in the session."""
    changed = False
    for i in range(start, min(end, len(msgs))):
        if msgs[i].get("role") != "tool":
            continue
        text = content_text(msgs[i])
        cap = error_max_chars if error_re.search(text) else max_chars
        new, did = truncate_text(text, cap)
        if did:
            msgs[i] = set_content_text(msgs[i], new)
            changed = True
    return changed


def elide_duplicate_results(msgs: list[Message], start: int, end: int) -> bool:
    """Stage 2: identical tool results collapse to a marker — the *last*
    occurrence survives.

    Keep-last, not keep-first, because stage 3 evicts oldest rounds first: with
    keep-first the surviving copy is exactly what eviction removes next, leaving
    markers pointing at content that no longer exists. Markers sit early (cheap
    to evict); the real content sits late, where it survives longest. Prefix
    stability is unaffected: dedupe only ever runs inside a checkpoint, which
    rewrites the emitted prefix anyway.
    """
    end = min(end, len(msgs))
    last_seen: dict[str, int] = {}
    for i in range(start, end):
        if msgs[i].get("role") == "tool":
            text = content_text(msgs[i])
            if text != DUPLICATE_MARKER:
                last_seen[fingerprint(text)] = i
    changed = False
    for i in range(start, end):
        if msgs[i].get("role") != "tool":
            continue
        text = content_text(msgs[i])
        if text == DUPLICATE_MARKER:
            continue
        if last_seen.get(fingerprint(text)) != i:
            msgs[i] = set_content_text(msgs[i], DUPLICATE_MARKER)
            changed = True
    return changed


def _first(text: str, n: int) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line[:n] + ("…" if len(line) > n else "")


def digest_round(msgs: list[Message], start: int, end: int) -> str:
    """One deterministic digest line for the round ``msgs[start:end]``."""
    lead = msgs[start]
    role = lead.get("role")
    if role == "user":
        return f"- user: {_first(content_text(lead), 100)}"
    parts = [f"- assistant: {_first(content_text(lead), 80)}"]
    calls = []
    for tc in lead.get("tool_calls") or []:
        fn = tc.get("function") or {}
        calls.append(f"{fn.get('name', '?')}({_first(str(fn.get('arguments', '')), 60)})")
    results = [
        _first(content_text(m), 80) for m in msgs[start + 1 : end] if m.get("role") == "tool"
    ]
    if calls:
        parts.append("tools: " + "; ".join(calls))
    if results:
        parts.append("-> " + " | ".join(results))
    return " | ".join(parts)


def build_digest_message(
    lines: list[str],
    *,
    counter: TokenCounter,
    token_cap: int,
    summarizer: Callable[[list[str]], str] | None = None,
) -> Message:
    """The single digest message replacing evicted rounds. Oldest lines are
    dropped first when the cap is exceeded; the drop is stated, never silent.

    A summarizer's output is held to the same ``token_cap`` as the
    deterministic path — an over-cap summary falls back to the deterministic
    digest so the budget guarantee cannot be broken by the hook.
    """
    body: str | None = None
    if summarizer is not None:
        try:
            candidate = summarizer(lines)
        except Exception:
            # a dead/misbehaving summarizer endpoint must never break the
            # budget guarantee — the deterministic digest always exists
            candidate = None
        if candidate is not None and counter.count_text(candidate) <= token_cap:
            body = candidate
    if body is None:
        kept = list(lines)
        dropped = 0
        while kept and counter.count_text("\n".join(kept)) > token_cap and len(kept) > 1:
            kept.pop(0)
            dropped += 1
        # a single line can still exceed the cap: cut by chars, then *verify in
        # tokens* — chars/token varies (code, CJK), so halve until it truly fits
        if kept and counter.count_text(kept[0]) > token_cap:
            original = kept[0]
            max_chars = max(80, token_cap * 3)
            cut, _ = truncate_text(original, max_chars)
            while counter.count_text(cut) > token_cap and max_chars > 16:
                max_chars //= 2
                cut, _ = truncate_text(original, max_chars)
            kept[0] = cut
        if dropped:
            kept.insert(0, f"(… {dropped} earlier rounds elided …)")
        body = "\n".join(kept)
    return {
        "role": "user",
        "content": f"{DIGEST_MARKER} Older turns were compressed. Summary of evicted activity:\n{body}",
    }


def evict_rounds(
    msgs: list[Message],
    start: int,
    end: int,
    *,
    fits: Callable[[list[Message], list[str]], bool],
    digest_lines: list[str],
) -> tuple[list[Message], list[str], int]:
    """Stage 3: evict whole rounds oldest-first from ``msgs[start:end]`` until
    ``fits`` says the chain (plus its digest-in-progress) is under budget.

    Returns (new_msgs, digest_lines, evicted_rounds). ``digest_lines`` arrives
    pre-seeded with any prior digest's lines so digests fold instead of nesting.
    """
    evicted = 0
    while True:
        if fits(msgs, digest_lines):
            return msgs, digest_lines, evicted
        if start >= end:
            return msgs, digest_lines, evicted
        starts = round_starts(msgs, start, end)
        if not starts:  # range holds only tool messages: nothing evictable here
            return msgs, digest_lines, evicted
        r_end = starts[1] if len(starts) > 1 else end
        digest_lines.append(digest_round(msgs, starts[0], r_end))
        msgs = msgs[: starts[0]] + msgs[r_end:]
        end -= r_end - starts[0]
        evicted += 1
