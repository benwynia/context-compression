"""Compression stages: truncation, duplicate elision, round eviction + digest.

Each stage is a pure helper operating on a mutable working copy of the chain,
touching only the unprotected body range. Structural safety comes from operating
on rounds (see :mod:`ctxc.models`), never on raw message slices.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

from .models import (
    DIGEST_MARKER,
    DUPLICATE_MARKER,
    Message,
    content_text,
    round_starts,
    set_content_text,
)
from .tokens import TokenCounter


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Head+tail excerpt with an explicit elision marker."""
    if len(text) <= max_chars:
        return text, False
    head = int(max_chars * 0.6)
    tail = max_chars - head
    cut = len(text) - head - tail
    return f"{text[:head]}\n[ctxc: truncated {cut} chars]\n{text[-tail:]}", True


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
    """Stage 2: identical tool results collapse to a marker — the *first*
    occurrence survives because it is the one already in the cached prefix."""
    seen: set[str] = set()
    changed = False
    for i in range(start, min(end, len(msgs))):
        if msgs[i].get("role") != "tool":
            continue
        text = content_text(msgs[i])
        if text == DUPLICATE_MARKER:
            continue
        key = hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=16).hexdigest()
        if key in seen:
            msgs[i] = set_content_text(msgs[i], DUPLICATE_MARKER)
            changed = True
        else:
            seen.add(key)
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
    dropped first when the cap is exceeded; the drop is stated, never silent."""
    if summarizer is not None:
        body = summarizer(lines)
    else:
        kept = list(lines)
        dropped = 0
        while kept and counter.count_text("\n".join(kept)) > token_cap and len(kept) > 1:
            kept.pop(0)
            dropped += 1
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
        starts = round_starts(msgs, start, end)
        if not starts or start >= end:
            return msgs, digest_lines, evicted
        r_end = starts[1] if len(starts) > 1 else end
        digest_lines.append(digest_round(msgs, starts[0], r_end))
        msgs = msgs[: starts[0]] + msgs[r_end:]
        end -= r_end - starts[0]
        evicted += 1
