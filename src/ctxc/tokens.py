"""Token counting (tiktoken o200k_base, hash-cached).

An approximation for non-OpenAI models behind Copilot, but deterministic, offline
and consistent across baseline and compressed chains — which is what the
verification harness needs. The counter is pluggable everywhere it is used.
"""

from __future__ import annotations

import hashlib
import json

from .models import Message, content_text


class TokenCounter:
    def __init__(self, encoding: str = "o200k_base"):
        self._encoding_name = encoding
        self._enc = None
        self._cache: dict[str, int] = {}

    def _encoder(self):
        if self._enc is None:
            import tiktoken

            try:
                self._enc = tiktoken.get_encoding(self._encoding_name)
            except Exception:
                self._enc = tiktoken.get_encoding("cl100k_base")
        return self._enc

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        key = hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=16).hexdigest()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        n = len(self._encoder().encode(text, disallowed_special=()))
        self._cache[key] = n
        return n

    def count_message(self, msg: Message) -> int:
        total = 4  # rough per-message framing overhead
        total += self.count_text(content_text(msg))
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                total += self.count_text(str(fn.get("name") or ""))
                args = fn.get("arguments")
                if isinstance(args, str):
                    total += self.count_text(args)
                elif args is not None:
                    total += self.count_text(json.dumps(args, ensure_ascii=False))
        return total

    def count_chain(self, messages: list[Message]) -> int:
        return sum(self.count_message(m) for m in messages)
