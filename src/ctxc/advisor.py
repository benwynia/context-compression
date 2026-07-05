"""Advisory eviction (arm C): the agent model nominates blocks to prune; a
deterministic janitor executes — at checkpoint boundaries only.

Division of labor (rung-10 probe: judgment is real at frontier tier):

* every evictable tool result in the client's history gets a stable marker
  ``[block tN]`` (N = index in the client's append-only history), so the
  model can address blocks it has seen;
* the request grows one extra tool, ``prune_context`` — its description
  carries the whole protocol, so the protected head is never touched;
* directives coming back are stripped from the response (the agent harness
  never sees them) and land in a per-session ledger;
* the ledger is applied ONLY inside a compression checkpoint — the moment
  the prefix is being rewritten anyway, so advice never adds cache rewrites;
* application is a soft delete: the tool message's content is replaced with
  a one-line stub (structure untouched, tool pairing can't break) and the
  original is archived in memory for the session's lifetime;
* the protected head and the recent window are immune, whatever the model
  says — the invariant layer outranks the advisor.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .models import Message, copy_chain, protected_head_end
from .tokens import TokenCounter

BLOCK_RE = re.compile(r"^\[block t(\d+)\]")
_STUB_RE = re.compile(r"^\[block t\d+ pruned by agent\b")
_ID_RE = re.compile(r"^t(\d+)$")
MIN_BLOCK_CHARS = 200      # smaller results aren't worth a directive
MAX_REASON_CHARS = 80

PRUNE_TOOL = {
    "type": "function",
    "function": {
        "name": "prune_context",
        "description": (
            "Free context space. Old tool outputs above are labeled like "
            "[block t12]. Call this ALONGSIDE your normal work (never as your "
            "only action) to nominate blocks whose exact content you will not "
            "need again: superseded file reads, output of commands you have "
            "acted on, resolved errors, dead-end exploration. Nominated blocks "
            "are removed later and you will not see their content again, so "
            "do not nominate anything you may still quote, re-check, or edit "
            "around. Nominating nothing is always safe."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "blocks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string",
                                   "description": "block id, e.g. t12"},
                            "reason": {"type": "string",
                                       "description": "three words max"},
                        },
                        "required": ["id"],
                    },
                }
            },
            "required": ["blocks"],
        },
    },
}


REMINDER = (
    "\n\n[context manager: window is filling up. If any [block tN] outputs "
    "above are no longer needed, also call prune_context with their ids in "
    "this same turn.]"
)
PRESSURE = 0.7  # remind when the last emission crossed this share of budget


def annotate(messages: list[Message]) -> list[Message]:
    """Prefix evictable tool results with their stable ``[block tN]`` marker.

    Pure function of the history: the same message always gets the same
    marker, so annotated histories stay append-only whenever the client's
    are (prefix stability — and therefore the provider cache — survives).
    Head messages and small results are left untouched.
    """
    out = copy_chain(messages)
    h = protected_head_end(out)
    for i, m in enumerate(out):
        if i < h or m.get("role") != "tool":
            continue
        content = m.get("content")
        if not isinstance(content, str) or len(content) < MIN_BLOCK_CHARS:
            continue
        if BLOCK_RE.match(content):  # defensive: never double-annotate
            continue
        m["content"] = f"[block t{i}] {content}"
    return out


@dataclass
class AdvisorState:
    """Per-session ledger + janitor hook. ``hook`` is wired into
    ``SessionCompressor.pre_checkpoint`` so advice applies exactly when a
    checkpoint rewrites the prefix."""

    keep_recent: int = 8
    counter: TokenCounter | None = None
    pending: dict[int, str] = field(default_factory=dict)
    applied: dict[int, str] = field(default_factory=dict)
    archive: dict[int, str] = field(default_factory=dict)
    # just-in-time reminders: sticky per message index so the annotated
    # history stays append-only turn over turn (cache stability)
    reminded: set[int] = field(default_factory=set)
    last_emitted_tokens: int = 0
    # sidecar: at most one advisory query in flight per session, and only
    # when the ledger is empty (fresh advice is pointless while old advice
    # is still waiting for a checkpoint)
    sidecar_inflight: bool = False
    # observability
    directives: int = 0
    invalid_directives: int = 0
    pruned_blocks: int = 0
    freed_tokens: int = 0
    prune_only_responses: int = 0
    reminders_sent: int = 0
    sidecar_calls: int = 0

    def wants_sidecar(self, budget: int) -> bool:
        return (self.last_emitted_tokens > budget * SIDECAR_PRESSURE
                and not self.pending and not self.sidecar_inflight)

    def annotate_input(self, messages: list[Message], budget: int) -> list[Message]:
        """Marker-annotate the client history and, under budget pressure,
        append a prune reminder to the NEWEST tool result. The live smoke run
        showed models never volunteer the optional tool (0 calls in 45
        turns); a just-in-time reminder in the latest observation triggers it
        without touching the protected head. Reminders are sticky: once
        message N carries one, it carries it on every later turn, so the
        annotated history extends its own past emissions byte-for-byte."""
        out = annotate(messages)
        pressure = self.last_emitted_tokens > budget * PRESSURE
        newest = next((i for i in range(len(out) - 1, -1, -1)
                       if out[i].get("role") == "tool"
                       and isinstance(out[i].get("content"), str)), None)
        if pressure and newest is not None and newest not in self.reminded:
            self.reminded.add(newest)
            self.reminders_sent += 1
        for idx in self.reminded:
            if idx < len(out) and isinstance(out[idx].get("content"), str) \
                    and not out[idx]["content"].endswith(REMINDER):
                out[idx]["content"] += REMINDER
        return out

    def add_directives(self, blocks: list[dict], history_len: int) -> None:
        for b in blocks:
            self.directives += 1
            m = _ID_RE.match(str(b.get("id") or ""))
            if not m:
                self.invalid_directives += 1
                continue
            idx = int(m.group(1))
            if idx >= history_len or idx in self.applied:
                if idx >= history_len:
                    self.invalid_directives += 1
                continue
            reason = re.sub(r"\s+", " ", str(b.get("reason") or "no reason"))
            self.pending[idx] = reason[:MAX_REASON_CHARS]

    def hook(self, msgs: list[Message]) -> list[Message]:
        """Apply the ledger to a checkpoint candidate (returns a copy; never
        mutates in place — the candidate aliases the frozen prior emission).

        Blocks are found by MARKER, not position: after earlier checkpoints
        the candidate is digest + survivors, so indices shift but markers
        don't. The protected head and the last ``keep_recent`` messages are
        immune regardless of what was advised."""
        wanted = {**self.applied, **self.pending}
        if not wanted:
            return msgs
        msgs = copy_chain(msgs)
        h = protected_head_end(msgs)
        recent_floor = max(h, len(msgs) - self.keep_recent)
        for pos, m in enumerate(msgs):
            if pos < h or pos >= recent_floor or m.get("role") != "tool":
                continue
            content = m.get("content")
            if not isinstance(content, str) or _STUB_RE.match(content):
                continue
            match = BLOCK_RE.match(content)
            if not match:
                continue
            idx = int(match.group(1))
            if idx not in wanted:
                continue
            reason = wanted[idx]
            if idx not in self.archive:
                self.archive[idx] = content
                self.pruned_blocks += 1
                if self.counter is not None:
                    self.freed_tokens += max(
                        0, self.counter.count_text(content)
                        - self.counter.count_text(""))
            m["content"] = f"[block t{idx} pruned by agent: {reason}]"
        for idx in list(self.pending):
            self.applied[idx] = self.pending.pop(idx)
        return msgs

    def stats(self) -> dict:
        return {
            "advisor_directives": self.directives,
            "advisor_invalid": self.invalid_directives,
            "advisor_pending": len(self.pending),
            "advisor_pruned_blocks": self.pruned_blocks,
            "advisor_freed_tokens": self.freed_tokens,
            "advisor_prune_only_responses": self.prune_only_responses,
            "advisor_reminders_sent": self.reminders_sent,
            "advisor_sidecar_calls": self.sidecar_calls,
        }


SIDECAR_PRESSURE = 0.85  # fire close to the checkpoint, ~once per cycle

SIDECAR_ASK = (
    "STOP. Do not continue the task. You are now acting as the context "
    "manager for this conversation. Old tool outputs above are labeled like "
    "[block t12]. Identify blocks whose content is NO LONGER NEEDED to "
    "finish the task (superseded reads, resolved errors, exploration that "
    "led nowhere). The agent will NOT see evicted content again, so only "
    "evict blocks you are confident are done with. Evicting nothing is a "
    "valid answer. Reply with JSON only: "
    '{"evict": [{"id": "t12", "reason": "three words max"}, ...]}'
)


def sidecar_request_body(emission: list[Message], model: str) -> dict:
    """The out-of-band advisory query. Inline cooperation proved unreliable
    (live agents never volunteer the optional tool, reminders or not), but
    the same model judges well when asked directly (rung 10: 2/53 harmful at
    75% volume) — so the proxy asks, write-behind, once per pressure window.

    The emission is sent VERBATIM as the prefix with the instruction as a
    trailing user message: the query then shares its whole prefix with the
    agent's own cached conversation, so the provider bills it at the cached
    tier (~10%) instead of full price."""
    msgs = list(emission)
    # a trailing assistant message with unanswered tool calls cannot be
    # followed by a user message (provider rejects the chain) — drop it
    while msgs and msgs[-1].get("role") == "assistant" and msgs[-1].get("tool_calls"):
        msgs.pop()
    return {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [*msgs, {"role": "user", "content": SIDECAR_ASK}],
    }


def parse_sidecar_reply(resp_body: bytes) -> list[dict]:
    try:
        content = json.loads(resp_body)["choices"][0]["message"]["content"]
        blocks = json.loads(content).get("evict") or []
        return [b for b in blocks if isinstance(b, dict)]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return []


def inject_tool(body: dict) -> None:
    """Add ``prune_context`` to the request's tools (idempotent)."""
    tools = list(body.get("tools") or [])
    if not any((t.get("function") or {}).get("name") == "prune_context"
               for t in tools if isinstance(t, dict)):
        tools.append(PRUNE_TOOL)
    body["tools"] = tools


def strip_prune_calls(resp_body: bytes) -> tuple[bytes, list[dict], bool]:
    """Remove ``prune_context`` tool calls from an upstream chat completion.

    Returns ``(new_body, directives, prune_only)`` where ``prune_only`` is
    True when a choice was left with neither content nor tool calls (the
    model disobeyed "never as your only action" — counted, not fatal).
    Anything unparseable passes through untouched: the agent's traffic is
    never held hostage by the advisor."""
    try:
        data = json.loads(resp_body)
        choices = data["choices"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return resp_body, [], False
    directives: list[dict] = []
    prune_only = False
    changed = False
    for choice in choices:
        msg = choice.get("message") or {}
        calls = msg.get("tool_calls")
        if not isinstance(calls, list):
            continue
        kept = []
        for tc in calls:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            if fn.get("name") != "prune_context":
                kept.append(tc)
                continue
            changed = True
            try:
                directives.extend(json.loads(fn.get("arguments") or "{}")
                                  .get("blocks") or [])
            except (json.JSONDecodeError, AttributeError):
                directives.append({"id": "malformed"})
        if len(kept) != len(calls):
            if kept:
                msg["tool_calls"] = kept
            else:
                msg.pop("tool_calls", None)
                if not (msg.get("content") or "").strip():
                    prune_only = True
                if choice.get("finish_reason") == "tool_calls":
                    choice["finish_reason"] = "stop"
    if not changed:
        return resp_body, [], False
    return json.dumps(data, ensure_ascii=False).encode(), directives, prune_only
