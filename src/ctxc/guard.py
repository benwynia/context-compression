"""ThrashGuard — the circuit breaker for compression-induced churn.

Rung 12 measured the failure mode this exists to prevent: a live agent under
a tight budget loses evicted context, re-reads it, grows the chain, forces
another checkpoint that evicts it again — a loop that produced sessions with
124 checkpoints and 3x cost. Offline replay of the same tasks showed max 8
checkpoints, so the loop is *interactive*: it cannot be tuned away with a
better static budget, only broken by feedback. Three mechanisms, all
deterministic:

* **re-read pinning** — content that was evicted at a checkpoint and then
  reappears in the incoming history was evicted wrongly; it gets pinned
  (immune to truncation and eviction) so the same mistake can't repeat;
* **churn escalation** — every ``escalate_after`` checkpoints the effective
  budget grows by ``escalate_factor`` (capped at ``max_scale``x): a session
  that keeps checkpointing is telling us its budget is too small for it;
* **escalate-on-impossible** — when pinning pressure makes a budget
  unreachable, the budget escalates instead of the request failing.

The guard changes budgets, never content: everything it does composes with
the existing invariant layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Message, content_text, fingerprint, has_plain_text_content

_BLOCK_MARKER = re.compile(r"^\[block t\d+\] ")
_WS = re.compile(r"\s+")
MIN_TRACK_CHARS = 200  # smaller results are cheap to lose and re-fetch


def _fp(text: str) -> str:
    """Marker-stripped, whitespace-normalized fingerprint, so a re-read of
    the same file matches its evicted twin even when the advisor's [block tN]
    marker differs (a re-read lands at a new index, hence a new marker)."""
    return fingerprint(_WS.sub(" ", _BLOCK_MARKER.sub("", text)).strip())


@dataclass
class ThrashGuard:
    escalate_after: int = 3     # checkpoints per escalation step
    escalate_factor: float = 1.5
    max_scale: float = 4.0      # budget never grows beyond this multiple
    pin_rereads: bool = True
    max_tracked: int = 512      # evicted-fingerprint memory bound

    # state
    evicted: dict[str, int] = field(default_factory=dict)  # fp -> checkpoint no.
    pinned: set[str] = field(default_factory=set)
    checkpoints_seen: int = 0
    forced_escalations: int = 0
    rereads_detected: int = 0

    def scale(self) -> float:
        steps = self.checkpoints_seen // self.escalate_after + self.forced_escalations
        return min(self.escalate_factor ** steps, self.max_scale)

    def effective_budget(self, budget: int) -> int:
        return int(budget * self.scale())

    def note_incoming(self, msgs: list[Message]) -> None:
        """New client messages: a tool result matching something we evicted
        is the thrash signature — the agent went back for it. Pin it."""
        for m in msgs:
            if m.get("role") != "tool" or not has_plain_text_content(m):
                continue
            text = content_text(m)
            if len(text) < MIN_TRACK_CHARS:
                continue
            fp = _fp(text)
            if fp in self.evicted:
                self.rereads_detected += 1
                del self.evicted[fp]
                if self.pin_rereads:
                    self.pinned.add(fp)

    def pin_check(self, text: str) -> bool:
        """Wired into CompressConfig.pin_check: pinned content is immune to
        truncation and eviction."""
        return bool(self.pinned) and _fp(text) in self.pinned

    def observe_checkpoint(self, before: list[Message], after: list[Message]) -> None:
        """Record what this checkpoint dropped or rewrote, so a later re-read
        of it is recognizable."""
        self.checkpoints_seen += 1
        kept = set()
        for m in after:
            if m.get("role") == "tool" and has_plain_text_content(m):
                kept.add(_fp(content_text(m)))
        for m in before:
            if m.get("role") != "tool" or not has_plain_text_content(m):
                continue
            text = content_text(m)
            if len(text) < MIN_TRACK_CHARS:
                continue
            fp = _fp(text)
            if fp not in kept and fp not in self.pinned:
                self.evicted[fp] = self.checkpoints_seen
        if len(self.evicted) > self.max_tracked:  # drop oldest entries
            for fp, _ in sorted(self.evicted.items(), key=lambda kv: kv[1])[
                : len(self.evicted) - self.max_tracked
            ]:
                del self.evicted[fp]

    def force_escalate(self) -> bool:
        """Budget unreachable (e.g. pinning pressure): grow it instead of
        failing — unless already at the ceiling."""
        if self.scale() >= self.max_scale:
            return False
        self.forced_escalations += 1
        return True

    def stats(self) -> dict:
        return {
            "guard_scale": round(self.scale(), 3),
            "guard_pinned": len(self.pinned),
            "guard_rereads": self.rereads_detected,
            "guard_forced_escalations": self.forced_escalations,
        }
