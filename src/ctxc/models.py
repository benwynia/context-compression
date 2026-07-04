"""Chain structure helpers for OpenAI chat/completions message lists.

Messages are plain dicts and are passed through untouched except for the parts
compression rewrites. This module knows the structural rules a valid chain obeys
and how to cut a chain into *rounds* — the atomic units eviction may remove
without ever breaking a tool_call/tool pairing.
"""

from __future__ import annotations

import hashlib
from typing import Any

Message = dict[str, Any]

DIGEST_MARKER = "[ctxc digest]"
TRUNCATION_MARKER = "[ctxc: truncated"
DUPLICATE_MARKER = "[ctxc: duplicate result elided; the full content appears later in this conversation]"


def fingerprint(text: str) -> str:
    """The one text-fingerprint used package-wide (counter cache keys, duplicate
    detection, session keys) so 'the same text' means the same thing everywhere."""
    return hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=16).hexdigest()


def copy_chain(messages: list[Message]) -> list[Message]:
    """Shallow per-message copy. Nested structures (tool_calls, content parts)
    stay aliased — safe because compression never mutates inside them, only
    replaces whole values via new dicts (see set_content_text)."""
    return [dict(m) for m in messages]


def content_text(msg: Message) -> str:
    """Flatten a message's content to text (string or list-of-parts form)."""
    c = msg.get("content")
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):  # content parts: [{"type": "text", "text": ...}, ...]
        out: list[str] = []
        for part in c:
            if isinstance(part, dict):
                out.append(str(part.get("text") or part.get("content") or ""))
            else:
                out.append(str(part))
        return "\n".join(out)
    return str(c)


def set_content_text(msg: Message, text: str) -> Message:
    """Copy of ``msg`` with its content replaced by plain text."""
    out = dict(msg)
    out["content"] = text
    return out


def tool_call_ids(msg: Message) -> list[str]:
    """IDs of the tool calls an assistant message issues."""
    return [
        tc.get("id", "")
        for tc in (msg.get("tool_calls") or [])
        if isinstance(tc, dict)
    ]


def is_digest(msg: Message) -> bool:
    return msg.get("role") == "user" and content_text(msg).startswith(DIGEST_MARKER)


def validate_chain(messages: list[Message]) -> list[str]:
    """Return a list of structural violations (empty = valid).

    Rules:
    - system messages only at the head;
    - every tool message answers a tool_call issued by the most recent assistant
      message (no orphans, no answering across a later non-tool message);
    - every tool_call is answered before the next non-tool message (except when
      it is the final message of the chain — a request mid-round is legal);
    - roles are one of system/user/assistant/tool.
    """
    errors: list[str] = []
    seen_non_system = False
    open_calls: set[str] = set()

    for i, m in enumerate(messages):
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            errors.append(f"msg[{i}]: unknown role {role!r}")
            continue
        if role == "system":
            if seen_non_system:
                errors.append(f"msg[{i}]: system message after non-system messages")
            continue
        seen_non_system = True

        if role == "tool":
            tcid = m.get("tool_call_id")
            if not tcid:
                errors.append(f"msg[{i}]: tool message without tool_call_id")
            elif tcid not in open_calls:
                errors.append(f"msg[{i}]: orphan tool message for {tcid!r}")
            else:
                open_calls.discard(tcid)
            continue

        # user / assistant: all previously issued calls must be answered by now
        if open_calls:
            errors.append(
                f"msg[{i}]: unanswered tool_calls before a {role} message: "
                f"{sorted(open_calls)}"
            )
            open_calls.clear()
        if role == "assistant":
            for tcid in tool_call_ids(m):
                if not tcid:
                    errors.append(f"msg[{i}]: tool_call with empty id")
                else:
                    open_calls.add(tcid)
    # trailing open calls are legal: the request is being made mid-round
    return errors


def head_len(messages: list[Message]) -> int:
    """Length of the protected head: leading system message(s) plus the first
    non-system message (the task statement)."""
    n = 0
    while n < len(messages) and messages[n].get("role") == "system":
        n += 1
    if n < len(messages):
        n += 1
    return n


def protected_head_end(messages: list[Message]) -> int:
    """``head_len`` extended through any tool messages answering a head-ending
    assistant (a transcript can open mid-round), so nothing — digest included —
    is ever inserted between a tool_call and its results."""
    h = head_len(messages)
    while h < len(messages) and messages[h].get("role") == "tool":
        h += 1
    return h


def round_starts(messages: list[Message], start: int, end: int) -> list[int]:
    """Indices in ``[start, end)`` where a round begins.

    A round is an assistant message plus the tool messages answering it, or a
    standalone user message. Tool messages never start a round, so the result
    can be empty (callers must handle that honestly rather than fabricate one).
    """
    return [
        i
        for i in range(start, end)
        if messages[i].get("role") in ("user", "assistant")
    ]


def round_boundary_at_or_before(messages: list[Message], idx: int, lo: int) -> int:
    """Largest round start <= idx (but >= lo), so a slice beginning there never
    opens with an orphan tool message."""
    i = max(lo, min(idx, len(messages)))
    while i > lo and (i >= len(messages) or messages[i].get("role") == "tool"):
        i -= 1
    return i
