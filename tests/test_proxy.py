import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from ctxc.proxy import build_app
from ctxc.synth import synth_session
from ctxc.tokens import TokenCounter


def _fake_upstream(received: list[dict]):
    async def chat(request):
        body = json.loads(await request.body())
        received.append(
            {"body": body, "auth": request.headers.get("authorization", "")}
        )
        return JSONResponse(
            {
                "id": "chatcmpl-fake",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}}
                ],
                "usage": {
                    "prompt_tokens": sum(len(str(m)) for m in body["messages"]) // 4,
                    "completion_tokens": 1,
                },
            }
        )

    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


@pytest.mark.anyio
async def test_proxy_compresses_and_forwards():
    received: list[dict] = []
    upstream = _fake_upstream(received)
    upstream_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=upstream), base_url="http://upstream"
    )
    app = build_app("http://upstream", budget=20_000, client=upstream_client)

    messages = synth_session(rounds=30, seed=5)
    counter = TokenCounter()
    orig_tokens = counter.count_chain(messages)
    assert orig_tokens > 20_000  # the fixture must actually need compression

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5", "messages": messages},
            headers={
                "authorization": "Bearer sekrit",
                "x-ctxc-session-id": "s1",
            },
        )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok"

    assert len(received) == 1
    fwd = received[0]
    assert fwd["auth"] == "Bearer sekrit"  # auth passthrough
    assert fwd["body"]["model"] == "gpt-5"  # non-message fields untouched
    assert counter.count_chain(fwd["body"]["messages"]) <= 20_000

    assert int(resp.headers["x-ctxc-original-tokens"]) == orig_tokens
    assert int(resp.headers["x-ctxc-emitted-tokens"]) <= 20_000


@pytest.mark.anyio
async def test_proxy_passes_small_requests_untouched():
    received: list[dict] = []
    upstream_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_fake_upstream(received)),
        base_url="http://upstream",
    )
    app = build_app("http://upstream", budget=1_000_000, client=upstream_client)
    messages = synth_session(rounds=3, seed=2, result_lines=(5, 10))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5", "messages": messages},
            headers={"x-ctxc-session-id": "s2"},
        )
    assert resp.status_code == 200
    assert received[0]["body"]["messages"] == messages


@pytest.fixture
def anyio_backend():
    return "asyncio"
