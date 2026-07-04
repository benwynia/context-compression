import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from ctxc.synth import synth_session
from ctxc.tokens import TokenCounter


@pytest.fixture
def anyio_backend():
    return "asyncio"


def fake_upstream(received: list[dict], prefix: str = "/v1"):
    """Records each forwarded request (path, query, body, auth) and returns a
    canned completion. Shared by every proxy test so the upstream contract
    can't silently diverge between files."""

    async def chat(request):
        received.append(
            {
                "path": request.url.path,
                "query": request.url.query,
                "body": json.loads(await request.body()),
                "auth": request.headers.get("authorization", ""),
            }
        )
        return JSONResponse(
            {
                "id": "chatcmpl-fake",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )

    return Starlette(routes=[Route(f"{prefix}/chat/completions", chat, methods=["POST"])])


def asgi_client(app, base_url: str = "http://u") -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url)


@pytest.fixture(scope="session")
def counter():
    return TokenCounter()


@pytest.fixture(scope="session")
def session_messages():
    """A long synthetic coding-agent session (~hundreds of k tokens)."""
    return synth_session(rounds=40, seed=7)


@pytest.fixture(scope="session")
def small_session():
    return synth_session(rounds=8, seed=3, result_lines=(10, 40))
