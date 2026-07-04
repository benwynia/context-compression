"""OpenAI-dialect compression proxy (the live workflow glue).

Point any OpenAI-compatible client at it; it compresses ``messages`` per session
and forwards everything else verbatim (auth headers and query string included)
to the upstream.

    ctxc proxy --upstream https://api.example.com --budget 60000 --port 8790

Three modes:

* **active** (default) — the compressed chain is what goes upstream.
* **shadow** (``--shadow``) — the ORIGINAL request goes upstream untouched;
  compression runs on the side and its would-be savings are recorded. Zero
  behavior change, real traffic, real numbers: run this first, quote the
  measured percentage, then flip to active.
* **passthrough** (``--passthrough``) — no compression at all, measurement and
  recording only. This is the A/B *control arm*: both arms get the identical
  proxy hop, recording, and per-task stats, so the only variable left is
  compression itself.

Sessions are keyed by the ``x-ctxc-session-id`` header when present, else by a
hash of the chain's first message (stable across turns; concurrent
conversations with identical openings need the header). ``--record DIR``
captures each conversation as a replayable session file for ``ctxc verify``.
``GET /stats`` aggregates measured savings, including the upstream's own
reported ``usage`` (provider-billed numbers, not tiktoken estimates).
Responses come back unchanged plus ``x-ctxc-*`` accounting headers; streaming
(``stream: true``) responses are passed through chunk-by-chunk, so chat UIs
render tokens as they arrive (upstream-reported usage isn't parsed for
streamed turns — token accounting for those uses local counts).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .compressor import BudgetImpossible, CompressConfig
from .models import fingerprint
from .redact import redact_messages
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
    passthrough: bool = False,
    record_dir: str | Path | None = None,
    record_raw: bool = False,
) -> Starlette:
    if shadow and passthrough:
        raise ValueError("shadow and passthrough are mutually exclusive modes")
    mode = "passthrough" if passthrough else ("shadow" if shadow else "active")
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
        "session_resets": 0,       # non-append histories: silent cache rewrites
        "redactions": 0,           # secret-looking strings scrubbed from records
        "original_tokens": 0,
        "emitted_tokens": 0,
        "upstream_prompt_tokens": 0,
        "upstream_completion_tokens": 0,
        "upstream_cached_tokens": 0,
        "compress_ms_total": 0.0,  # CPU time inside compression/counting
        "compress_ms_max": 0.0,
    }
    # per-session (= per-task in an A/B run) attribution; evicted with the LRU
    session_stats: dict[str, dict] = {}

    def _fresh_session_stats() -> dict:
        return {k: 0 for k in stats if k != "compress_ms_max"} | {
            "checkpoints": 0, "compress_ms_max": 0.0,
        }

    def _record(key: str, messages: list) -> int:
        """Each request carries the FULL history, so overwriting with the latest
        leaves one complete, replayable session file per conversation. Secrets
        are redacted by default (``record_raw=True`` opts out)."""
        if rec_dir is None:
            return 0
        n = 0
        if not record_raw:
            messages, n = redact_messages(messages)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", key)[:80]
        (rec_dir / f"{safe}.json").write_text(
            json.dumps({"messages": messages}, ensure_ascii=False)
        )
        return n

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
        compress_error: str | None = None
        if effective_budget <= 0:
            # fail-open: an uncompressible request is forwarded as-is (the
            # provider's cap error beats a proxy-invented 400), and counted.
            compress_error = f"tools schemas alone are {tools_tokens} tokens, budget is {budget}"

        key = _session_key(request, messages)
        entry = sessions.get(key)
        if entry is None:
            entry = sessions[key] = (
                SessionCompressor(budget, config=config, counter=counter),
                asyncio.Lock(),
            )
            session_stats[key] = _fresh_session_stats()
            while len(sessions) > max_sessions:  # LRU: drop the stalest session
                dropped, _ = sessions.popitem(last=False)
                session_stats.pop(dropped, None)
        sessions.move_to_end(key)
        sc, lock = entry
        per = session_stats[key]
        redactions = _record(key, messages)
        stats["redactions"] += redactions
        per["redactions"] += redactions

        emitted = None
        resets_before = sc.resets
        t0 = time.perf_counter()
        if passthrough or compress_error is not None:
            original_tokens = await asyncio.to_thread(counter.count_chain, messages)
        else:
            try:
                # compression is CPU-bound (tokenizing + escalation ladder): run
                # it off the event loop so one big checkpoint doesn't stall every
                # other in-flight session; the per-session lock keeps sc state
                # sequential.
                async with lock:
                    original_tokens, emitted = await asyncio.to_thread(
                        lambda: (
                            counter.count_chain(messages),
                            sc.request(messages, budget=max(1, effective_budget)),
                        )
                    )
            except BudgetImpossible as e:
                # fail-open in EVERY mode: a compression failure must never
                # fail the request — forward the original and account for it
                compress_error = str(e)
                original_tokens = counter.count_chain(messages)
        compress_ms = (time.perf_counter() - t0) * 1000.0
        if compress_error is not None:
            stats["compress_errors"] += 1
            per["compress_errors"] += 1
        reset_delta = sc.resets - resets_before
        stats["session_resets"] += reset_delta
        per["session_resets"] += reset_delta
        stats["compress_ms_total"] += compress_ms
        per["compress_ms_total"] += compress_ms
        stats["compress_ms_max"] = max(stats["compress_ms_max"], compress_ms)
        per["compress_ms_max"] = max(per["compress_ms_max"], compress_ms)

        emitted_tokens = counter.count_chain(emitted) if emitted is not None else None
        if mode == "active" and emitted is not None:
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

        def _account(usage: dict) -> None:
            increments = {
                "requests": 1,
                "original_tokens": original_tokens,
                "emitted_tokens": emitted_tokens if emitted_tokens is not None else original_tokens,
                "upstream_prompt_tokens": usage["prompt_tokens"],
                "upstream_completion_tokens": usage["completion_tokens"],
                "upstream_cached_tokens": usage["cached_tokens"],
            }
            for k, v in increments.items():
                stats[k] += v
                per[k] += v
            per["checkpoints"] = sc.checkpoints

        def _out_headers(upstream_headers) -> dict:
            out = {
                k: v
                for k, v in upstream_headers.items()
                if k.lower() not in _HOP_BY_HOP and k.lower() != "content-encoding"
            }
            out["x-ctxc-mode"] = mode
            out["x-ctxc-original-tokens"] = str(original_tokens)
            if emitted_tokens is not None:
                out["x-ctxc-emitted-tokens"] = str(emitted_tokens)
            if compress_error is not None:
                out["x-ctxc-compress-error"] = compress_error[:200]
            return out

        if body.get("stream"):
            # SSE passthrough: chunks reach the client as they arrive (a chat
            # UI must not freeze behind a buffered response). Usage rides in
            # the final SSE chunk, which we don't parse — token accounting for
            # streamed turns uses local counts only.
            req = http.build_request("POST", url, json=body, headers=fwd_headers)
            upstream_resp = await http.send(req, stream=True)
            _account({"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0})
            return StreamingResponse(
                upstream_resp.aiter_raw(),
                status_code=upstream_resp.status_code,
                headers=_out_headers(upstream_resp.headers),
                background=BackgroundTask(upstream_resp.aclose),
            )

        resp = await http.post(url, json=body, headers=fwd_headers)
        _account(_usage_from(resp.content))
        return Response(
            content=resp.content, status_code=resp.status_code,
            headers=_out_headers(resp.headers),
        )

    async def health(_: Request) -> Response:
        return JSONResponse({"ok": True, "sessions": len(sessions), "budget": budget})

    async def stats_view(_: Request) -> Response:
        o, e = stats["original_tokens"], stats["emitted_tokens"]
        return JSONResponse(
            {
                "mode": mode,
                "sessions": len(sessions),
                **stats,
                "saved_tokens": o - e,
                "saved_pct": round(100.0 * (o - e) / o, 2) if o else 0.0,
                # in shadow/passthrough the upstream sees the ORIGINAL chain, so
                # upstream_* is the real BASELINE spend; in active mode it is
                # the real COMPRESSED spend (provider-billed)
                "upstream_usage_is": "compressed" if mode == "active" else "baseline",
            }
        )

    async def session_stats_view(_: Request) -> Response:
        """Per-session (= per-task) attribution for A/B runs: scrape after each
        task, keyed by the x-ctxc-session-id you drove the task with."""
        return JSONResponse({"mode": mode, "sessions": session_stats})

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
            Route("/stats/sessions", session_stats_view, methods=["GET"]),
        ],
        lifespan=lifespan,
    )
