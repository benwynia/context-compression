"""SessionCompressor — the cache-checkpoint state machine.

Compression that rewrites history on every request destroys the provider's
prompt cache (the central lesson from cost-benchmarking compression proxies:
token savings barely survive to cost when every turn is a cache write). So this
wrapper compresses only at discrete *checkpoints*:

* between checkpoints, every emitted request is exactly the previous emission
  plus the newly arrived tail — a byte-stable prefix, i.e. a cache read;
* when the emitted chain would cross the budget, one checkpoint fires: the chain
  is compressed down to ``budget * recompress_to`` (hysteresis keeps checkpoints
  rare), the result is frozen, and appends resume on top of it.

Clients send the full uncompressed history each time (they don't know about the
compressor); we diff against the last seen source to find the new tail. A
non-append-only history (edited past) falls back to a from-scratch compress.
"""

from __future__ import annotations

import dataclasses

from .compressor import BudgetImpossible, CompressConfig, compress
from .guard import ThrashGuard
from .models import Message, copy_chain
from .tokens import TokenCounter


class SessionCompressor:
    def __init__(
        self,
        budget: int,
        config: CompressConfig | None = None,
        counter: TokenCounter | None = None,
        pre_checkpoint=None,
        guard: ThrashGuard | None = None,
    ):
        self.budget = budget
        self.config = config or CompressConfig()
        self.counter = counter or TokenCounter()
        # thrash guard: churn-based budget escalation + re-read pinning (see
        # guard.py). Its pin_check composes with any user-supplied one.
        self.guard = guard
        if guard is not None:
            user_check = self.config.pin_check
            combined = (guard.pin_check if user_check is None else
                        (lambda t, u=user_check, g=guard.pin_check: u(t) or g(t)))
            self.config = dataclasses.replace(self.config, pin_check=combined)
        # Called on the candidate chain right before a checkpoint compresses
        # it (chain -> chain). The one moment extra rewrites are free — the
        # prefix is being rebuilt anyway. Contract: must return a copy, never
        # mutate in place (the candidate aliases the frozen prior emission).
        self.pre_checkpoint = pre_checkpoint
        self.checkpoints = 0
        # non-append-only histories (user edited an earlier turn, retries,
        # session-key collisions) force a full rebuild — each one is a silent
        # cache rewrite, so it must be countable, not invisible
        self.resets = 0
        self._source: list[Message] = []   # last full history seen from the client
        self._emitted: list[Message] = []  # what we sent upstream for it
        # Set when the hysteresis target proved unreachable: skip the doomed
        # target ladder on later checkpoints until a compress lands under the
        # target again (self-healing — old content eventually becomes evictable).
        self._target_impossible = False

    def _is_append_only(self, history: list[Message]) -> bool:
        n = len(self._source)
        return len(history) >= n and history[:n] == self._source

    def request(self, history: list[Message], budget: int | None = None) -> list[Message]:
        """Map the client's full history to the chain to send upstream.

        ``budget`` overrides the session budget for this request — used by the
        proxy to subtract the request's ``tools`` schema tokens, which occupy
        the same context window the messages do.
        """
        budget = self.budget if budget is None else budget
        if self._source and self._is_append_only(history):
            tail = copy_chain(history[len(self._source):])
            candidate = self._emitted + tail
            new_msgs = tail
        else:
            if self._source:  # edited past: previous emission is unusable
                self._emitted = []
                self.resets += 1
            candidate = copy_chain(history)
            new_msgs = candidate
        base_budget = budget
        if self.guard is not None:
            # a re-read of previously evicted content is the thrash signature:
            # the guard pins it before this (or any later) checkpoint runs
            self.guard.note_incoming(new_msgs)
            budget = self.guard.effective_budget(base_budget)

        if self.counter.count_chain(candidate) > budget:
            if self.pre_checkpoint is not None:
                candidate = self.pre_checkpoint(candidate)
            pre_checkpoint_msgs = candidate
            while True:
                try:
                    result = self._compress_with_hysteresis(candidate, budget)
                    break
                except BudgetImpossible:
                    # pinning pressure can make a budget unreachable — with a
                    # guard, grow the budget instead of failing the request
                    if self.guard is not None and self.guard.force_escalate():
                        budget = self.guard.effective_budget(base_budget)
                        continue
                    raise
            candidate = result.messages
            self.checkpoints += 1
            if self.guard is not None:
                self.guard.observe_checkpoint(pre_checkpoint_msgs, candidate)

        self._source = copy_chain(history)
        self._emitted = candidate
        return candidate

    def _compress_with_hysteresis(self, candidate: list[Message], budget: int):
        target = max(1, int(budget * self.config.recompress_to))
        result = None
        if not self._target_impossible:
            try:
                result = compress(candidate, target, self.config, self.counter)
            except BudgetImpossible:
                self._target_impossible = True
        if result is None:
            # hysteresis target unreachable — the hard cap is what matters.
            # Skip the doomed target ladder on subsequent checkpoints too,
            # instead of paying two full escalation ladders every turn.
            result = compress(candidate, budget, self.config, self.counter)
            if result.compressed_tokens <= target:
                self._target_impossible = False  # target achievable again
        return result
