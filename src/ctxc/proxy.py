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

import hashlib
import json
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .compressor import BudgetImpossible, CompressConfig
from .session import SessionCompressor
from .tokens import TokenCounter

_HOP_BY_HOP = {
    "host", "content-length", "transfer-encoding", "connection", "keep-alive",
    "proxy-authenticate", "proxy-authorization", "te", "trailer", "upgrade",
    "accept-encoding",
}


def _session_key(request: Request, messages: list) -> str:
    explicit = request.headers.get("x-ctxc-session-id")
    if explicit:
        return explicit
    head = json.dumps(messages[:2], ensure_ascii=False, sort_keys=True)
    return hashlib.blake2b(head.encode("utf-8", "ignore"), digest_size=16).hexdigest()


def build_app(
    upstream: str,
    budget: int,
    config: CompressConfig | None = None,
    client: httpx.AsyncClient | None = None,
    counter: TokenCounter | None = None,
) -> Starlette:
    upstream = upstream.rstrip("/")
    counter = counter or TokenCounter()
    sessions: dict[str, SessionCompressor] = {}
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

        key = _session_key(request, messages)
        sc = sessions.get(key)
        if sc is None:
            sc = sessions[key] = SessionCompressor(budget, config=config, counter=counter)
        original_tokens = counter.count_chain(messages)
        try:
            emitted = sc.request(messages)
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
        if not path.startswith("/v1"):
            path = "/v1" + path
        resp = await http.post(f"{upstream}{path}", json=body, headers=fwd_headers)

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
