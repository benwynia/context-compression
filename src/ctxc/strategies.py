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
    has_plain_text_content,
    is_opaque,
    round_starts,
    set_content_text,
)
from .tokens import TokenCounter


# Lines that look load-bearing: errors/warnings, decisions and instructions,
# notes, identifiers (codes like ABC-1234, long numbers, paths, URLs), and
# credential-shaped words. Used by salience-aware truncation and digests so
# what survives compression is chosen by importance, not by position.
DEFAULT_SALIENCE = (
    r"(?i)\berror\b|\bwarn(?:ing)?\b|\bfail(?:ed|ure)?\b|traceback"
    r"|\bnote[:\s]|\btodo\b|\bfixme\b|remember|important|ctxc:pin"
    r"|\bdecid\w+|\bagreed\b|\bmust\b|\bdo not\b|\bnever\b|\balways\b"
    r"|\b[A-Z]{2,}-\d{2,}\b|\b\d{5,}\b|(?:/[\w.-]+){2,}|https?://\S"
    r"|\bkey\b|\bcode\b|\btoken\b|\bpassword\b|\bsecret\b"
)


# Strong signals outrank weak keyword hits, so a log full of lines merely
# *containing* "error" or "token" can't crowd out an explicit NOTE/decision/
# identifier line. Tier 1: explicit markers and directives. Tier 2:
# identifier-shaped content (codes, URLs, paths) — case-sensitive on purpose.
_STRONG_MARKERS = re.compile(
    r"(?i)\bnote[:\s]|\btodo\b|\bfixme\b|\bremember\b|\bimportant\b|ctxc:pin"
    r"|\bdecid\w+|\bagreed\b|\bdo not\b|\bmust\b|\bnever\b|\balways\b"
)
_STRONG_IDENTIFIERS = re.compile(
    r"\b[A-Z]{2,}-\d{2,}\b|https?://\S|(?:/[\w.-]+){2,}|-----BEGIN"
)


def _salience_score(line: str, gate: re.Pattern[str]) -> int:
    score = 0
    if _STRONG_MARKERS.search(line):
        score += 2
    if _STRONG_IDENTIFIERS.search(line):
        score += 2
    if gate.search(line):
        score += 1
    return score


def salient_lines(
    text: str,
    pattern: re.Pattern[str],
    max_lines: int = 3,
    max_line_chars: int = 160,
) -> list[str]:
    """The most important-looking lines of ``text``: scored (strong markers and
    identifiers beat single weak keywords), top ``max_lines``, original order."""
    scored: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for pos, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line or line in seen:
            continue
        s = _salience_score(line, pattern)
        if s <= 0:
            continue
        seen.add(line)
        scored.append((-s, pos, line))
    scored.sort()
    top = sorted(scored[:max_lines], key=lambda t: t[1])  # back to text order
    return [
        ln[:max_line_chars] + ("…" if len(ln) > max_line_chars else "")
        for _, _, ln in top
    ]


def truncate_text(
    text: str,
    max_chars: int,
    salience: re.Pattern[str] | None = None,
) -> tuple[str, bool]:
    """Head+tail excerpt with an explicit elision marker. With ``salience``,
    important-looking lines from the cut middle survive alongside the excerpt —
    retention chosen by content, not position."""
    if len(text) <= max_chars:
        return text, False
    if max_chars <= 0:  # "keep nothing": marker only — never re-emit the body
        return f"[ctxc: truncated {len(text)} chars]", True
    head = int(max_chars * 0.6)
    tail = max_chars - head
    cut = len(text) - head - tail
    kept_tail = text[-tail:] if tail > 0 else ""
    kept = ""
    if salience is not None:
        middle = text[head: len(text) - tail if tail > 0 else len(text)]
        lines = [
            ln for ln in salient_lines(middle, salience)
            if ln.rstrip("…") not in text[:head] and ln.rstrip("…") not in kept_tail
        ]
        if lines:
            kept = "\n[ctxc: salient lines kept from cut]\n" + "\n".join(lines)
    return (
        f"{text[:head]}\n[ctxc: truncated {cut} chars]{kept}\n{kept_tail}",
        True,
    )


def truncate_tool_results(
    msgs: list[Message],
    start: int,
    end: int,
    *,
    max_chars: int,
    error_max_chars: int,
    error_re: re.Pattern[str],
    salience: re.Pattern[str] | None = None,
    pin_re: re.Pattern[str] | None = None,
) -> bool:
    """Stage 1: cut unprotected tool results down to excerpts. Errors keep more —
    they tend to get referenced later. Pinned messages are never touched."""
    changed = False
    for i in range(start, min(end, len(msgs))):
        if msgs[i].get("role") != "tool":
            continue
        if not has_plain_text_content(msgs[i]):
            continue  # list-form content may carry images — never rewrite it
        text = content_text(msgs[i])
        if pin_re is not None and pin_re.search(text):
            continue  # explicitly pinned: immune to truncation
        cap = error_max_chars if error_re.search(text) else max_chars
        new, did = truncate_text(text, cap, salience=salience)
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
        if msgs[i].get("role") == "tool" and has_plain_text_content(msgs[i]):
            text = content_text(msgs[i])
            if text != DUPLICATE_MARKER:
                last_seen[fingerprint(text)] = i
    changed = False
    for i in range(start, end):
        if msgs[i].get("role") != "tool" or not has_plain_text_content(msgs[i]):
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


def digest_round(
    msgs: list[Message],
    start: int,
    end: int,
    salience: re.Pattern[str] | None = None,
    max_salient: int = 3,
) -> str:
    """One deterministic digest line for the round ``msgs[start:end]``."""
    lead = msgs[start]
    role = lead.get("role")
    if role == "user":
        # user turns carry instructions — keep noticeably more than one line
        base = f"- user: {_first(content_text(lead), 240)}"
    else:
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
        base = " | ".join(parts)
    if salience is not None:
        # keep the round's important-looking lines (errors, decisions, codes,
        # paths) even when they weren't first lines — the rung-4 fix: what a
        # digest retains is chosen by content, not position
        full = "\n".join(content_text(m) for m in msgs[start:end])
        lines = [ln for ln in salient_lines(full, salience, max_lines=max_salient)
                 if ln.rstrip("…") not in base]
        if lines:
            base += "\n" + "\n".join(f"  keep: {ln}" for ln in lines)
    return base


def build_digest_message(
    lines: list[str],
    *,
    counter: TokenCounter,
    token_cap: int,
    summarizer: Callable[[list[str]], str] | None = None,
    salience: re.Pattern[str] | None = None,
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
        # line-granular priority trimming: when the cap bites, narrative lines
        # die first (oldest first), then salient "keep:" lines by ASCENDING
        # salience score — a fact's survival is decided by importance, then
        # age, never by position alone
        records: list[list] = []  # [entry_age, is_keep, score, line, alive]
        for age, entry in enumerate(lines):
            for ln in entry.splitlines():
                is_keep = ln.strip().startswith("keep:")
                score = (
                    _salience_score(ln, salience) if is_keep and salience else 0
                )
                records.append([age, is_keep, score, ln, True])

        def body_text() -> str:
            return "\n".join(r[3] for r in records if r[4])

        drop_order = sorted(
            range(len(records)),
            key=lambda j: (records[j][1], records[j][2], records[j][0]),
        )  # non-keep first (oldest first); then keeps: lowest score, oldest first
        for j in drop_order:
            if counter.count_text(body_text()) <= token_cap:
                break
            if sum(r[4] for r in records) <= 1:
                break
            records[j][4] = False
        dropped = len(
            {r[0] for r in records} - {r[0] for r in records if r[4]}
        )  # rounds that lost every line
        kept = [body_text()] if records else []
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
    salience: re.Pattern[str] | None = None,
    max_salient: int = 3,
    pin_re: re.Pattern[str] | None = None,
) -> tuple[list[Message], list[str], int]:
    """Stage 3: evict whole rounds oldest-first from ``msgs[start:end]`` until
    ``fits`` says the chain (plus its digest-in-progress) is under budget.

    Rounds carrying an opaque message shape (fail-open) or an explicit pin
    marker are retained and SKIPPED — later rounds keep evicting, so one
    protected round can't block the budget from being met.

    Returns (new_msgs, digest_lines, evicted_rounds). ``digest_lines`` arrives
    pre-seeded with any prior digest's lines so digests fold instead of nesting.
    """

    def protected(s: int, e: int) -> bool:
        for m in msgs[s:e]:
            if is_opaque(m):
                return True
            if pin_re is not None and has_plain_text_content(m) and pin_re.search(
                content_text(m)
            ):
                return True
        return False

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
        if protected(starts[0], r_end):
            start = r_end  # retain this round; keep evicting later ones
            continue
        digest_lines.append(digest_round(msgs, starts[0], r_end, salience, max_salient))
        msgs = msgs[: starts[0]] + msgs[r_end:]
        end -= r_end - starts[0]
        evicted += 1
