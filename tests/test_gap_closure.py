"""Tests for the boundary-gap fixes: fail-open message shapes, fail-open proxy,
record redaction, observability counters, and the smoke harness."""

import json

import httpx
import pytest

from conftest import asgi_client, fake_upstream
from ctxc.compressor import compress
from ctxc.models import validate_chain
from ctxc.proxy import build_app
from ctxc.redact import REDACTED, redact_messages, redact_text
from ctxc.session import SessionCompressor
from ctxc.smoke import run_smoke
from ctxc.synth import synth_session


@pytest.fixture
def anyio_backend():
    return "asyncio"


# --------------------------------------------------------------------------- #
# fail-open message shapes
# --------------------------------------------------------------------------- #
def test_developer_role_is_protected_head(counter):
    msgs = [{"role": "developer", "content": "You are an agent."}] + synth_session(
        rounds=30, seed=7
    )[1:]
    assert validate_chain(msgs) == []
    res = compress(msgs, budget=8_000, counter=counter)
    assert res.compressed_tokens <= 8_000
    assert res.messages[0] == msgs[0]  # developer message verbatim at the head


def test_unknown_role_survives_compression(counter):
    msgs = synth_session(rounds=30, seed=7)
    opaque = {"role": "reasoning_trace", "content": "provider-specific blob"}
    msgs.insert(4, dict(opaque))
    assert validate_chain(msgs) == []  # opaque roles never invalidate a chain
    res = compress(msgs, budget=8_000, counter=counter)
    assert res.compressed_tokens <= 8_000
    assert opaque in res.messages  # never truncated, deduped, or evicted


def test_image_tool_results_never_rewritten(counter):
    msgs = synth_session(rounds=25, seed=7)
    # replace one mid-chain tool result with multimodal content (a screenshot)
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and 4 < i < len(msgs) - 12:
            msgs[i] = {
                "role": "tool",
                "tool_call_id": m["tool_call_id"],
                "content": [
                    {"type": "text", "text": "screenshot follows " * 200},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
            image_msg = msgs[i]
            break
    res = compress(msgs, budget=12_000, counter=counter)
    assert res.compressed_tokens <= 12_000
    survivors = [m for m in res.messages if m.get("tool_call_id") == image_msg["tool_call_id"]]
    if survivors:  # if the round survived, the image content must be untouched
        assert survivors[0]["content"] == image_msg["content"]


# --------------------------------------------------------------------------- #
# fail-open proxy: compression failures never fail a request
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_active_mode_forwards_original_when_tools_exceed_budget():
    received: list[dict] = []
    app = build_app("http://u", budget=200, client=asgi_client(fake_upstream(received)))
    tools = [{"type": "function", "function": {"name": "t", "description": "x " * 2000}}]
    async with asgi_client(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}],
                  "tools": tools},
        )
        stats = (await client.get("/stats")).json()
    assert resp.status_code == 200  # never a proxy-invented 400
    assert "tools schemas" in resp.headers["x-ctxc-compress-error"]
    assert received[0]["body"]["tools"] == tools  # original forwarded untouched
    assert stats["compress_errors"] == 1


@pytest.mark.anyio
async def test_active_mode_forwards_original_on_budget_impossible():
    received: list[dict] = []
    app = build_app("http://u", budget=60, client=asgi_client(fake_upstream(received)))
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task " + "word " * 2000},  # head alone > budget
    ]
    async with asgi_client(app) as client:
        resp = await client.post(
            "/v1/chat/completions", json={"model": "gpt-5", "messages": messages}
        )
    assert resp.status_code == 200
    assert "x-ctxc-compress-error" in resp.headers
    assert received[0]["body"]["messages"] == messages


# --------------------------------------------------------------------------- #
# redaction
# --------------------------------------------------------------------------- #
def test_redact_text_patterns():
    text = (
        "export OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwx123456\n"
        "aws AKIAIOSFODNN7EXAMPLE\n"
        "Authorization: Bearer abcdef1234567890abcdef\n"
        "db postgres://admin:hunter2secret@db.internal:5432/prod\n"
        "password = supersecret99\n"
        "plain text stays put"
    )
    red, n = redact_text(text)
    assert n >= 5
    assert "sk-abcdefghijklmnop" not in red
    assert "AKIAIOSFODNN7EXAMPLE" not in red
    assert "hunter2secret" not in red
    assert "supersecret99" not in red
    assert "password" in red  # key names survive, values don't
    assert "plain text stays put" in red
    assert REDACTED in red


def test_redact_messages_walks_all_fields():
    msgs = [
        {"role": "user", "content": "my key is sk-abcdefghijklmnopqrstuvwx"},
        {"role": "user", "content": [{"type": "text", "text": "token=abcdefgh12345678"}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "bash", "arguments": '{"cmd": "curl -H \'Bearer aaaabbbbccccdddd1234\'"}'}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    red, n = redact_messages(msgs)
    assert n >= 3
    flat = json.dumps(red)
    assert "sk-abcdefghijklmnop" not in flat
    assert "aaaabbbbccccdddd1234" not in flat
    assert msgs[0]["content"].startswith("my key is sk-")  # input untouched
    assert validate_chain(red) == []  # structure preserved


@pytest.mark.anyio
async def test_recording_redacts_by_default(tmp_path):
    received: list[dict] = []
    app = build_app("http://u", budget=1_000_000, record_dir=tmp_path,
                    client=asgi_client(fake_upstream(received)))
    msgs = [{"role": "user", "content": "use sk-abcdefghijklmnopqrstuvwx please"}]
    async with asgi_client(app) as client:
        await client.post("/v1/chat/completions",
                          json={"model": "gpt-5", "messages": msgs},
                          headers={"x-ctxc-session-id": "r1"})
        stats = (await client.get("/stats")).json()
    recorded = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert "sk-abcdefghijklmnop" not in json.dumps(recorded)
    assert stats["redactions"] >= 1
    # the upstream still received the REAL content — redaction is record-only
    assert "sk-abcdefghijklmnop" in received[0]["body"]["messages"][0]["content"]


@pytest.mark.anyio
async def test_record_raw_opts_out(tmp_path):
    app = build_app("http://u", budget=1_000_000, record_dir=tmp_path,
                    record_raw=True, client=asgi_client(fake_upstream([])))
    msgs = [{"role": "user", "content": "use sk-abcdefghijklmnopqrstuvwx please"}]
    async with asgi_client(app) as client:
        await client.post("/v1/chat/completions",
                          json={"model": "gpt-5", "messages": msgs},
                          headers={"x-ctxc-session-id": "r2"})
    recorded = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert "sk-abcdefghijklmnop" in json.dumps(recorded)


# --------------------------------------------------------------------------- #
# observability
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_reset_counter_and_latency_in_stats():
    app = build_app("http://u", budget=1_000_000, client=asgi_client(fake_upstream([])))
    turn1 = [{"role": "user", "content": "original task"}]
    edited = [{"role": "user", "content": "a completely different task"},
              {"role": "assistant", "content": "ok"},
              {"role": "user", "content": "go on"}]
    async with asgi_client(app) as client:
        for msgs in (turn1, edited):  # same session id, edited history
            await client.post("/v1/chat/completions",
                              json={"model": "gpt-5", "messages": msgs},
                              headers={"x-ctxc-session-id": "edit-1"})
        stats = (await client.get("/stats")).json()
        per = (await client.get("/stats/sessions")).json()["sessions"]["edit-1"]
    assert stats["session_resets"] == 1  # the edit was counted, not invisible
    assert per["session_resets"] == 1
    assert stats["compress_ms_total"] > 0
    assert stats["compress_ms_max"] >= per["compress_ms_max"] > 0


def test_session_compressor_counts_resets():
    sc = SessionCompressor(budget=1_000_000)
    sc.request([{"role": "user", "content": "a"}])
    sc.request([{"role": "user", "content": "b"}])  # non-append
    assert sc.resets == 1


# --------------------------------------------------------------------------- #
# smoke harness
# --------------------------------------------------------------------------- #
def test_smoke_sends_valid_compressed_chain():
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 700, "completion_tokens": 1},
        })

    result = run_smoke("https://api.example.com", "gpt-5-mini",
                       client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert result["ok"] is True
    assert result["compressed_tokens"] <= 800 < result["original_tokens"]
    # the request must exercise the shapes we need providers to accept
    assert result["shapes_exercised"]["truncation_marker"]
    assert validate_chain(sent[0]["messages"]) == []
    assert sent[0]["tools"]
    assert result["reply"] == "ok"


def test_smoke_reports_provider_rejection():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad shape"}})

    result = run_smoke("https://api.example.com/v1", "m",
                       client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert result["ok"] is False
    assert result["error"]["error"]["message"] == "bad shape"
