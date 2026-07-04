"""In-house LLM compaction: a summarizer backed by any OpenAI-compatible
chat/completions endpoint — including a local 7B via Ollama, vLLM, LM Studio,
or llama.cpp server. No third-party compression service; your transcript goes
only to a model you run (or already pay for).

This plugs into the existing ``CompressConfig.summarizer`` hook, so all its
safety properties apply automatically:

* called at most **once per checkpoint** (a couple of times per long session,
  never per request or per eviction probe);
* input is the deterministic digest lines — already bounded — and is
  additionally hard-capped here, so a 7B's small context is never overflowed;
* output over the digest token cap falls back to the deterministic digest;
* any endpoint failure (down, timeout, garbage) also falls back — the
  compressor's budget guarantee cannot be broken by this hook.

Usage:

    ctxc proxy --upstream $URL --budget 60k \
      --summarizer-url http://localhost:11434/v1 --summarizer-model qwen2.5:7b

or in code::

    cfg = CompressConfig(summarizer=LlmSummarizer("http://localhost:11434/v1",
                                                  "qwen2.5:7b"))
"""

from __future__ import annotations

import os

import httpx

_SYSTEM_PROMPT = (
    "You compress the evicted history of a coding-agent conversation. The user "
    "message contains one line per evicted turn (assistant intent, tool calls, "
    "first lines of results). Rewrite them as a dense factual summary that "
    "preserves: file paths, commands run, key decisions, error messages, and "
    "what was tried with its outcome. Merge redundant lines. Plain text only, "
    "no preamble, no commentary. Stay under {target} tokens."
)


class LlmSummarizer:
    """Callable ``lines -> summary`` for ``CompressConfig.summarizer``."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        *,
        target_tokens: int = 300,
        max_input_chars: int = 24_000,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ):
        base = base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        self.url = base + "/chat/completions"
        self.model = model
        self.api_key_env = api_key_env
        self.target_tokens = target_tokens
        self.max_input_chars = max_input_chars
        self._client = client or httpx.Client(timeout=timeout)
        self.calls = 0  # observability: one per checkpoint, or something's wrong

    def __call__(self, lines: list[str]) -> str:
        self.calls += 1
        text = "\n".join(lines)
        if len(text) > self.max_input_chars:
            # keep the most recent lines — oldest history matters least
            text = "(earliest evicted turns omitted)\n" + text[-self.max_input_chars:]
        headers = {"content-type": "application/json"}
        key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        if key:
            headers["authorization"] = f"Bearer {key}"
        resp = self._client.post(
            self.url,
            json={
                "model": self.model,
                "temperature": 0,
                "max_tokens": int(self.target_tokens * 1.5),
                "messages": [
                    {
                        "role": "system",
                        "content": _SYSTEM_PROMPT.format(target=self.target_tokens),
                    },
                    {"role": "user", "content": text},
                ],
            },
            headers=headers,
        )
        resp.raise_for_status()  # any failure -> build_digest_message falls back
        content = resp.json()["choices"][0]["message"]["content"] or ""
        if not content.strip():
            raise ValueError("summarizer returned empty content")
        return content.strip()

    def close(self) -> None:
        self._client.close()
