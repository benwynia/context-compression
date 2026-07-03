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

from .compressor import BudgetImpossible, CompressConfig, compress
from .models import Message
from .tokens import TokenCounter


class SessionCompressor:
    def __init__(
        self,
        budget: int,
        config: CompressConfig | None = None,
        counter: TokenCounter | None = None,
    ):
        self.budget = budget
        self.config = config or CompressConfig()
        self.counter = counter or TokenCounter()
        self.checkpoints = 0
        self._source: list[Message] = []   # last full history seen from the client
        self._emitted: list[Message] = []  # what we sent upstream for it

    def _is_append_only(self, history: list[Message]) -> bool:
        n = len(self._source)
        return len(history) >= n and history[:n] == self._source

    def request(self, history: list[Message]) -> list[Message]:
        """Map the client's full history to the chain to send upstream."""
        if self._source and self._is_append_only(history):
            tail = [dict(m) for m in history[len(self._source):]]
            candidate = self._emitted + tail
        else:
            if self._source:  # edited past: previous emission is unusable
                self._emitted = []
            candidate = [dict(m) for m in history]

        if self.counter.count_chain(candidate) > self.budget:
            target = max(1, int(self.budget * self.config.recompress_to))
            try:
                result = compress(candidate, target, self.config, self.counter)
            except BudgetImpossible:
                # hysteresis target unreachable — the hard cap is what matters
                result = compress(candidate, self.budget, self.config, self.counter)
            candidate = result.messages
            self.checkpoints += 1

        self._source = [dict(m) for m in history]
        self._emitted = candidate
        return candidate
