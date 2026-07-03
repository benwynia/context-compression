"""Tests for the low-confidence areas: real-world message shapes, tools-schema
budget accounting, verify robustness on impossible budgets, proxy ops limits."""

import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from ctxc.compressor import compress
from ctxc.models import validate_chain
from ctxc.proxy import build_app
from ctxc.session import SessionCompressor
from ctxc.tokens import TokenCounter
from ctxc.verify import render_report, verify_session

BLOB = " ".join(f"token{i} filler payload" for i in range(400))


def _realistic_session(rounds: int = 12) -> list[dict]:
    """OpenAI-dialect shapes the synth generator doesn't produce: content:null
    assistants, multiple tool_calls per turn, list-form content parts."""
    msgs: list[dict] = [
        {"role": "system", "content": "You are an agent."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Fix the bug in the retry logic."},
                {"type": "text", "text": "Also add tests."},
            ],
        },
    ]
    n = 0
    for r in range(rounds):
        calls = []
        for _ in range(2 + r % 2):  # 2 or 3 calls per assistant turn
            n += 1
            calls.append(
                {
                    "id": f"c{n}",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"n": n})},
                }
            )
        msgs.append({"role": "assistant", "content": None, "tool_calls": calls})
        for c in calls:
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": f"{c['id']}: {BLOB}"})
        msgs.append({"role": "assistant", "content": f"Round {r} done."})
    return msgs


def test_realistic_shapes_validate():
    assert validate_chain(_realistic_session()) == []


def test_compress_none_content_and_multi_tool_calls(counter):
    msgs = _realistic_session(rounds=12)
    total = counter.count_chain(msgs)
    assert total > 12_000
    res = compress(msgs, budget=8_000, counter=counter)
    assert res.compressed_tokens <= 8_000
    assert validate_chain(res.messages) == []
    # None-content assistants must survive or be evicted whole, never mangled
    for m in res.messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            ids = {tc["id"] for tc in m["tool_calls"]}
            following = set()
            idx = res.messages.index(m) + 1
            while idx < len(res.messages) and res.messages[idx].get("role") == "tool":
                following.add(res.messages[idx]["tool_call_id"])
                idx += 1
            if idx < len(res.messages):  # not the trailing open round
                assert ids <= following


def test_session_compressor_on_realistic_shapes(counter):
    msgs = _realistic_session(rounds=12)
    sc = SessionCompressor(budget=10_000, counter=counter)
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant" or i == 0:
            continue
        emitted = sc.request(msgs[:i])
        assert validate_chain(emitted) == []
        assert counter.count_chain(emitted) <= 10_000
    assert sc.checkpoints >= 1


def test_verify_reports_budget_impossible_instead_of_crashing(counter):
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task " + "word " * 4000},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "b"},
    ]
    report = verify_session(msgs, budget=500, counter=counter)
    assert not report.ok
    assert any("budget impossible" in v for v in report.violations)
    render_report(report)  # must not raise


def test_per_request_budget_override(counter):
    msgs = _realistic_session(rounds=12)
    sc = SessionCompressor(budget=1_000_000, counter=counter)
    emitted = sc.request(msgs, budget=9_000)
    assert counter.count_chain(emitted) <= 9_000


# --------------------------------------------------------------------------- #
# proxy hardening
# --------------------------------------------------------------------------- #
def _fake_upstream(received: list[dict], prefix: str = "/v1"):
    async def chat(request):
        received.append({"path": request.url.path, "body": json.loads(await request.body())})
        return JSONResponse({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    return Starlette(routes=[Route(f"{prefix}/chat/completions", chat, methods=["POST"])])


def _client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://u")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_proxy_counts_tools_against_budget(counter):
    received: list[dict] = []
    app = build_app("http://u", budget=10_000, client=_client_for(_fake_upstream(received)))
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": "does things " * 60,
                "parameters": {"type": "object", "properties": {"a": {"type": "string"}}},
            },
        }
        for i in range(10)
    ]
    tools_tokens = counter.count_text(json.dumps(tools, ensure_ascii=False, sort_keys=True))
    assert 1_000 < tools_tokens < 9_000

    msgs = _realistic_session(rounds=10)
    async with _client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5", "messages": msgs, "tools": tools},
            headers={"x-ctxc-session-id": "t1"},
        )
    assert resp.status_code == 200
    fwd = received[0]["body"]
    assert counter.count_chain(fwd["messages"]) <= 10_000 - tools_tokens
    assert fwd["tools"] == tools  # schemas forwarded untouched


@pytest.mark.anyio
async def test_proxy_rejects_when_tools_alone_exceed_budget():
    received: list[dict] = []
    app = build_app("http://u", budget=200, client=_client_for(_fake_upstream(received)))
    tools = [{"type": "function", "function": {"name": "t", "description": "x " * 2000}}]
    async with _client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}], "tools": tools},
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "ctxc_budget_impossible"
    assert not received


@pytest.mark.anyio
async def test_proxy_upstream_already_has_v1():
    received: list[dict] = []
    app = build_app("http://u/v1", budget=1_000_000, client=_client_for(_fake_upstream(received)))
    async with _client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    assert received[0]["path"] == "/v1/chat/completions"  # not /v1/v1/...


@pytest.mark.anyio
async def test_proxy_session_lru_cap():
    received: list[dict] = []
    app = build_app(
        "http://u", budget=1_000_000, client=_client_for(_fake_upstream(received)),
        max_sessions=2,
    )
    async with _client_for(app) as client:
        for sid in ("a", "b", "c"):
            await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-5", "messages": [{"role": "user", "content": sid}]},
                headers={"x-ctxc-session-id": sid},
            )
        health = (await client.get("/healthz")).json()
    assert health["sessions"] == 2
