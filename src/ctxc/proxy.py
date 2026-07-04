"""OpenAI-dialect compression proxy (the live workflow glue).

Point any OpenAI-compatible client at it; it compresses ``messages`` per session
and forwards everything else verbatim (auth headers included) to the upstream.

    ctxc proxy --upstream https://api.example.com --budget 60000 --port 8790

Sessions are keyed by the ``x-ctxc-session-id`` header when present, else by a
hash of the chain's head (system + task), so one conversation keeps one
SessionCompressor and its cache checkpoints. Responses come back unchanged plus
``x-ctxc-original-tokens`` / ``x-ctxc-emitted-tokens`` headers. Non-streaming
requests only (``stream: true`` is forwarded but the response is buffered).
"""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from contextlib import asynccontextmanager

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


def build_app(
    upstream: str,
    budget: int,
    config: CompressConfig | None = None,
    client: httpx.AsyncClient | None = None,
    counter: TokenCounter | None = None,
    max_sessions: int = 256,
) -> Starlette:
    upstream = upstream.rstrip("/")
    counter = counter or TokenCounter()
    sessions: OrderedDict[str, SessionCompressor] = OrderedDict()
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=120.0)

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
        if effective_budget <= 0:
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
        try:
            # compression is CPU-bound (tokenizing + escalation ladder): run it
            # off the event loop so one big checkpoint doesn't stall every other
            # in-flight session; the per-session lock keeps sc state sequential.
            async with lock:
                original_tokens, emitted = await asyncio.to_thread(
                    lambda: (
                        counter.count_chain(messages),
                        sc.request(messages, budget=effective_budget),
                    )
                )
        except BudgetImpossible as e:
            return JSONResponse(
                {"error": {"type": "ctxc_budget_impossible", "message": str(e)}},
                status_code=400,
            )
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

        out_headers = {
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in _HOP_BY_HOP and k.lower() != "content-encoding"
        }
        out_headers["x-ctxc-original-tokens"] = str(original_tokens)
        out_headers["x-ctxc-emitted-tokens"] = str(counter.count_chain(emitted))
        return Response(
            content=resp.content, status_code=resp.status_code, headers=out_headers
        )

    async def health(_: Request) -> Response:
        return JSONResponse({"ok": True, "sessions": len(sessions), "budget": budget})

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
        ],
        lifespan=lifespan,
    )
