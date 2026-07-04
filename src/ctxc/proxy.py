"""OpenAI-dialect compression proxy (the live workflow glue).

Point any OpenAI-compatible client at it; it compresses ``messages`` per session
and forwards everything else verbatim (auth headers and query string included)
to the upstream.

    ctxc proxy --upstream https://api.example.com --budget 60000 --port 8790

Two modes:

* **active** (default) — the compressed chain is what goes upstream.
* **shadow** (``--shadow``) — the ORIGINAL request goes upstream untouched;
  compression runs on the side and its would-be savings are recorded. Zero
  behavior change, real traffic, real numbers: run this first, quote the
  measured percentage, then flip to active.

Sessions are keyed by the ``x-ctxc-session-id`` header when present, else by a
hash of the chain's first message (stable across turns; concurrent
conversations with identical openings need the header). ``--record DIR``
captures each conversation as a replayable session file for ``ctxc verify``.
``GET /stats`` aggregates measured savings, including the upstream's own
reported ``usage`` (provider-billed numbers, not tiktoken estimates).
Responses come back unchanged plus ``x-ctxc-*`` accounting headers.
Non-streaming responses only (``stream: true`` is forwarded but buffered).
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .compressor import BudgetImpossible, CompressConfig
from .models import fingerprint
from .session import SessionCompressor
from .tokens import TokenCounter

_HOP_BY_HOP = {
    "host", "content-length", "transfer-encoding", "connection", "keep-alive",
    "proxy-authenticate", "proxy-authorization", "te", "trailer", "upgrade",
    "accept-encoding",
}


def _session_key(request: Request, messages: list) -> str:
    """Prefer the explicit header. The fallback hashes only the FIRST message —
    it never changes as the conversation grows, so one conversation keeps one
    session. It cannot distinguish concurrent conversations that share an
    identical first message; those callers must send x-ctxc-session-id (README).
    """
    explicit = request.headers.get("x-ctxc-session-id")
    if explicit:
        return explicit
    head = json.dumps(messages[:1], ensure_ascii=False, sort_keys=True)
    return fingerprint(head)


def tools_token_count(counter: TokenCounter, tools: list) -> int:
    """Token cost of the request's tools schemas — one recipe, shared with tests."""
    return counter.count_text(json.dumps(tools, ensure_ascii=False, sort_keys=True))


def _usage_from(resp_body: bytes) -> dict:
    """Best-effort extraction of the upstream's reported usage block."""
    try:
        u = json.loads(resp_body).get("usage") or {}
        details = u.get("prompt_tokens_details") or {}
        return {
            "prompt_tokens": int(u.get("prompt_tokens") or 0),
            "completion_tokens": int(u.get("completion_tokens") or 0),
            "cached_tokens": int(details.get("cached_tokens") or 0),
        }
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        return {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}


def build_app(
    upstream: str,
    budget: int,
    config: CompressConfig | None = None,
    client: httpx.AsyncClient | None = None,
    counter: TokenCounter | None = None,
    max_sessions: int = 256,
    shadow: bool = False,
    record_dir: str | Path | None = None,
) -> Starlette:
    upstream = upstream.rstrip("/")
    counter = counter or TokenCounter()
    sessions: OrderedDict[str, tuple[SessionCompressor, asyncio.Lock]] = OrderedDict()
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=120.0)
    rec_dir = Path(record_dir) if record_dir else None
    if rec_dir:
        rec_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "requests": 0,
        "compress_errors": 0,
        "original_tokens": 0,
        "emitted_tokens": 0,
        "upstream_prompt_tokens": 0,
        "upstream_completion_tokens": 0,
        "upstream_cached_tokens": 0,
    }

    def _record(key: str, messages: list) -> None:
        """Each request carries the FULL history, so overwriting with the latest
        leaves one complete, replayable session file per conversation."""
        if rec_dir is None:
            return
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", key)[:80]
        (rec_dir / f"{safe}.json").write_text(
            json.dumps({"messages": messages}, ensure_ascii=False)
        )

    async def chat(request: Request) -> Response:
        try:
            body = json.loads(await request.body())
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        messages = body.get("messages")
        if not isinstance(messages, list):
            return JSONResponse({"error": "missing messages"}, status_code=400)

        # tools schemas share the context window with the messages: they count
        # against the same budget, or a "60k budget" would overshoot the cap.
        tools_tokens = tools_token_count(counter, body["tools"]) if body.get("tools") else 0
        effective_budget = budget - tools_tokens
        if effective_budget <= 0 and not shadow:
            return JSONResponse(
                {"error": {"type": "ctxc_budget_impossible",
                           "message": f"tools schemas alone are {tools_tokens} tokens, "
                                      f"budget is {budget}"}},
                status_code=400,
            )

        key = _session_key(request, messages)
        entry = sessions.get(key)
        if entry is None:
            entry = sessions[key] = (
                SessionCompressor(budget, config=config, counter=counter),
                asyncio.Lock(),
            )
            while len(sessions) > max_sessions:  # LRU: drop the stalest session
                sessions.popitem(last=False)
        sessions.move_to_end(key)
        sc, lock = entry
        _record(key, messages)

        emitted = None
        try:
            # compression is CPU-bound (tokenizing + escalation ladder): run it
            # off the event loop so one big checkpoint doesn't stall every other
            # in-flight session; the per-session lock keeps sc state sequential.
            async with lock:
                original_tokens, emitted = await asyncio.to_thread(
                    lambda: (
                        counter.count_chain(messages),
                        sc.request(messages, budget=max(1, effective_budget)),
                    )
                )
        except BudgetImpossible as e:
            if not shadow:
                return JSONResponse(
                    {"error": {"type": "ctxc_budget_impossible", "message": str(e)}},
                    status_code=400,
                )
            # shadow must be zero-risk: log the failure, forward the original
            stats["compress_errors"] += 1
            original_tokens = counter.count_chain(messages)

        emitted_tokens = counter.count_chain(emitted) if emitted is not None else None
        if not shadow and emitted is not None:
            body["messages"] = emitted

        fwd_headers = {
            k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP
        }
        path = request.url.path
        if not path.startswith("/v1") and not upstream.endswith("/v1"):
            path = "/v1" + path
        elif path.startswith("/v1") and upstream.endswith("/v1"):
            path = path[len("/v1"):]
        url = f"{upstream}{path}"
        if request.url.query:  # forward query params (api-version=… etc.) verbatim
            url = f"{url}?{request.url.query}"
        resp = await http.post(url, json=body, headers=fwd_headers)

        usage = _usage_from(resp.content)
        stats["requests"] += 1
        stats["original_tokens"] += original_tokens
        stats["emitted_tokens"] += emitted_tokens if emitted_tokens is not None else original_tokens
        stats["upstream_prompt_tokens"] += usage["prompt_tokens"]
        stats["upstream_completion_tokens"] += usage["completion_tokens"]
        stats["upstream_cached_tokens"] += usage["cached_tokens"]

        out_headers = {
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in _HOP_BY_HOP and k.lower() != "content-encoding"
        }
        out_headers["x-ctxc-mode"] = "shadow" if shadow else "active"
        out_headers["x-ctxc-original-tokens"] = str(original_tokens)
        if emitted_tokens is not None:
            out_headers["x-ctxc-emitted-tokens"] = str(emitted_tokens)
        return Response(
            content=resp.content, status_code=resp.status_code, headers=out_headers
        )

    async def health(_: Request) -> Response:
        return JSONResponse({"ok": True, "sessions": len(sessions), "budget": budget})

    async def stats_view(_: Request) -> Response:
        o, e = stats["original_tokens"], stats["emitted_tokens"]
        return JSONResponse(
            {
                "mode": "shadow" if shadow else "active",
                "sessions": len(sessions),
                **stats,
                "saved_tokens": o - e,
                "saved_pct": round(100.0 * (o - e) / o, 2) if o else 0.0,
                # in shadow mode upstream_* is the real BASELINE spend; in
                # active mode it is the real COMPRESSED spend (provider-billed)
                "upstream_usage_is": "baseline" if shadow else "compressed",
            }
        )

    @asynccontextmanager
    async def lifespan(_: Starlette):
        try:
            yield
        finally:
            if owns_client:
                await http.aclose()

    return Starlette(
        routes=[
            Route("/v1/chat/completions", chat, methods=["POST"]),
            Route("/chat/completions", chat, methods=["POST"]),
            Route("/healthz", health, methods=["GET"]),
            Route("/stats", stats_view, methods=["GET"]),
        ],
        lifespan=lifespan,
    )
