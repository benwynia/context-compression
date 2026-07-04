"""Tests for the live-use kit: streaming passthrough and scrape/resolve glue."""

import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import StreamingResponse
from starlette.routing import Route

from conftest import asgi_client, fake_upstream
from ctxc.ab import load_results, mark_resolved, scrape_row
from ctxc.proxy import build_app
from ctxc.synth import synth_session


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _sse_upstream(received: list[dict]):
    async def chat(request):
        received.append(json.loads(await request.body()))

        async def gen():
            for tok in ("Hel", "lo", " world"):
                yield f'data: {{"choices":[{{"delta":{{"content":"{tok}"}}}}]}}\n\n'.encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


@pytest.mark.anyio
async def test_streaming_passthrough():
    received: list[dict] = []
    app = build_app(
        "http://u", budget=20_000, client=asgi_client(_sse_upstream(received))
    )
    messages = synth_session(rounds=30, seed=5)  # over budget -> compression fires
    async with asgi_client(app) as client:
        async with client.stream(
            "POST", "/v1/chat/completions",
            json={"model": "gpt-5", "messages": messages, "stream": True},
            headers={"x-ctxc-session-id": "stream-1"},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["x-ctxc-mode"] == "active"
            assert int(resp.headers["x-ctxc-emitted-tokens"]) <= 20_000
            chunks = [c async for c in resp.aiter_bytes()]
        stats = (await client.get("/stats/sessions")).json()["sessions"]["stream-1"]

    body = b"".join(chunks).decode()
    assert "Hel" in body and "[DONE]" in body  # SSE passed through intact
    assert received[0]["stream"] is True  # stream flag forwarded
    assert stats["requests"] == 1
    assert stats["emitted_tokens"] <= 20_000  # local accounting still works
    assert stats["upstream_prompt_tokens"] == 0  # usage not parsed on streams


@pytest.mark.anyio
async def test_scrape_and_resolve_roundtrip(tmp_path):
    received: list[dict] = []
    app = build_app(
        "http://u", budget=20_000, client=asgi_client(fake_upstream(received))
    )
    messages = synth_session(rounds=30, seed=5)
    async with asgi_client(app) as client:
        await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5", "messages": messages},
            headers={"x-ctxc-session-id": "swe-001"},
        )
        # scrape via a sync-facing client backed by the same ASGI app
        sessions_payload = (await client.get("/stats/sessions")).json()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/stats/sessions"
        return httpx.Response(200, json=sessions_payload)

    row = scrape_row(
        "http://proxy", "swe-001",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert row["task_id"] == "swe-001"
    assert row["resolved"] is False
    assert row["checkpoints"] >= 1
    assert row["prompt_tokens"] > 0  # falls back to emitted when upstream is tiny/fake

    out = tmp_path / "results"
    out.mkdir()
    (out / "swe-001.json").write_text(json.dumps(row))
    (out / "swe-002.json").write_text(
        json.dumps({**row, "task_id": "swe-002"})
    )
    ids_updated = mark_resolved(out, {"swe-001"})
    assert ids_updated == 2
    rows = load_results(out)
    assert rows["swe-001"]["resolved"] is True
    assert rows["swe-002"]["resolved"] is False


def test_scrape_unknown_task_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"mode": "active", "sessions": {}})

    with pytest.raises(KeyError, match="no session"):
        scrape_row("http://proxy", "nope",
                   client=httpx.Client(transport=httpx.MockTransport(handler)))
