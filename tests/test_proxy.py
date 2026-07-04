import pytest

from conftest import asgi_client, fake_upstream
from ctxc.proxy import build_app
from ctxc.synth import synth_session
from ctxc.tokens import TokenCounter


@pytest.mark.anyio
async def test_proxy_compresses_and_forwards():
    received: list[dict] = []
    app = build_app(
        "http://u", budget=20_000, client=asgi_client(fake_upstream(received))
    )

    messages = synth_session(rounds=30, seed=5)
    counter = TokenCounter()
    orig_tokens = counter.count_chain(messages)
    assert orig_tokens > 20_000  # the fixture must actually need compression

    async with asgi_client(app, "http://proxy") as client:
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
    app = build_app(
        "http://u", budget=1_000_000, client=asgi_client(fake_upstream(received))
    )
    messages = synth_session(rounds=3, seed=2, result_lines=(5, 10))

    async with asgi_client(app, "http://proxy") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5", "messages": messages},
            headers={"x-ctxc-session-id": "s2"},
        )
    assert resp.status_code == 200
    assert received[0]["body"]["messages"] == messages
