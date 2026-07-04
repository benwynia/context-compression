"""Secret redaction for recorded sessions.

Coding-agent tool results routinely contain env vars, connection strings and
API keys. ``--record`` writes transcripts to disk, so recording redacts by
default (``--record-raw`` opts out). This is pattern-based hygiene, not a
guarantee — treat recorded sessions as sensitive artifacts regardless.

Only string *values* are rewritten (message content, text parts, tool_call
arguments); JSON structure is never touched, so recorded sessions stay
replayable by ``ctxc verify``.
"""

from __future__ import annotations

import re

from .models import Message

REDACTED = "[ctxc:redacted]"

_PATTERNS: list[re.Pattern[str]] = [
    # provider/API keys and tokens
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),                      # OpenAI-style
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                            # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),                  # GitHub tokens
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),               # Slack tokens
    re.compile(r"\beyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),  # JWTs
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s'\"]+:[^\s'\"]+@"),
]

# key = value / key: value assignments — keep the key name, redact the value
_ASSIGNMENT = re.compile(
    r"(?i)\b((?:api[_-]?key|secret|passwd|password|token|auth)[A-Za-z0-9_]*)"
    r"(\s*[:=]\s*['\"]?)([^\s'\"]{8,})"
)


def redact_text(text: str) -> tuple[str, int]:
    """Return (redacted_text, number_of_redactions)."""
    total = 0
    for pat in _PATTERNS:
        text, n = pat.subn(REDACTED, text)
        total += n
    text, n = _ASSIGNMENT.subn(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    total += n
    return text, total


def redact_messages(messages: list[Message]) -> tuple[list[Message], int]:
    """Deep-copy of the chain with secret-looking strings redacted."""
    total = 0
    out: list[Message] = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, str):
            m["content"], n = redact_text(content)
            total += n
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    part = dict(part)
                    part["text"], n = redact_text(part["text"])
                    total += n
                parts.append(part)
            m["content"] = parts
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        tc = dict(tc)
                        fn = dict(fn)
                        fn["arguments"], n = redact_text(args)
                        total += n
                        tc["function"] = fn
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out, total
