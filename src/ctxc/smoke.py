"""Live-fire smoke test: send one *compressed* chain to a real provider.

Everything else in this repo runs against fakes. This is the ~one-cent check
that a real endpoint accepts the shapes compression produces — truncation
markers, the digest message, duplicate markers, tool-call pairing — before an
engineer discovers a dialect gap in the middle of their work.

    ctxc smoke --upstream https://api.openai.com --model gpt-5-mini
"""

from __future__ import annotations

import json
import os

import httpx

from .compressor import compress
from .models import DIGEST_MARKER, TRUNCATION_MARKER
from .tokens import TokenCounter

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def _session() -> list[dict]:
    """A tiny agent session with enough bulk that compress() at a small budget
    exercises truncation AND eviction+digest — the shapes we need accepted."""
    msgs: list[dict] = [
        {"role": "system", "content": "You are a coding agent. Be terse."},
        {"role": "user", "content": "Fix the retry logic in the payments client."},
    ]
    # distinct per-round content (so dedupe can't collapse it) and enough
    # rounds that truncation alone can't meet the default budget — eviction
    # must fire, so the digest message shape is exercised against the provider
    for n in range(10):
        filler = f"pay_{n} log: retry attempt backoff timing entry {n}\n" * 120
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"smoke_{n}",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": f"src/pay_{n}.py"}),
                        },
                    }
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"smoke_{n}", "content": filler})
        msgs.append({"role": "assistant", "content": f"Inspected src/pay_{n}.py."})
    msgs.append({"role": "user", "content": "Reply with the single word: ok"})
    return msgs


def run_smoke(
    upstream: str,
    model: str,
    key_env: str = "OPENAI_API_KEY",
    budget: int = 800,
    client: httpx.Client | None = None,
) -> dict:
    counter = TokenCounter()
    messages = _session()
    original = counter.count_chain(messages)
    res = compress(messages, budget, counter=counter)
    flat = json.dumps(res.messages)
    shapes = {
        "truncation_marker": TRUNCATION_MARKER in flat,
        "digest_message": DIGEST_MARKER in flat,
    }

    base = upstream.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    headers = {"content-type": "application/json"}
    key = os.environ.get(key_env, "")
    if key:
        headers["authorization"] = f"Bearer {key}"
    http = client or httpx.Client(timeout=60.0)
    body = {
        "model": model,
        # newer OpenAI models take max_completion_tokens; older ones and most
        # OpenAI-compatible servers take max_tokens — try new, fall back
        "max_completion_tokens": 16,
        "messages": res.messages,
        "tools": _TOOLS,
    }
    resp = http.post(f"{base}/chat/completions", json=body, headers=headers)
    if resp.status_code == 400 and "max_completion_tokens" in resp.text:
        body.pop("max_completion_tokens")
        body["max_tokens"] = 16
        resp = http.post(f"{base}/chat/completions", json=body, headers=headers)
    out = {
        "status": resp.status_code,
        "ok": resp.status_code == 200,
        "original_tokens": original,
        "compressed_tokens": res.compressed_tokens,
        "shapes_exercised": shapes,
    }
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        payload = {"raw": resp.text[:300]}
    if resp.status_code == 200:
        msg = (payload.get("choices") or [{}])[0].get("message", {})
        out["reply"] = (msg.get("content") or "")[:100]
        out["usage"] = payload.get("usage")
    else:
        out["error"] = payload
    return out
