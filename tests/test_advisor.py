import json

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from conftest import asgi_client
from ctxc.advisor import (
    PRUNE_TOOL,
    AdvisorState,
    annotate,
    inject_tool,
    strip_prune_calls,
)
from ctxc.models import validate_chain
from ctxc.proxy import build_app
from ctxc.session import SessionCompressor
from ctxc.synth import synth_session
from ctxc.tokens import TokenCounter


def _chain(rounds=6, chars=400):
    msgs = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
    ]
    for i in range(rounds):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "read", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "content": f"result {i} " + ("x" * chars)})
    msgs.append({"role": "assistant", "content": "done"})
    return msgs


# ---------------------------------------------------------------- annotate

def test_annotate_marks_tool_results_stably():
    msgs = _chain()
    a1, a2 = annotate(msgs), annotate(msgs)
    assert a1 == a2  # pure function: append-only in == append-only out
    assert a1[3]["content"].startswith("[block t3] result 0")
    assert a1[0] == msgs[0] and a1[1] == msgs[1]  # head untouched
    assert msgs[3]["content"].startswith("result 0")  # input not mutated
    # double annotation is a no-op
    assert annotate(a1)[3]["content"] == a1[3]["content"]


def test_annotate_skips_small_results():
    msgs = _chain(chars=20)
    assert all(not str(m.get("content", "")).startswith("[block")
               for m in annotate(msgs))


# ------------------------------------------------------------------ ledger

def test_directive_sanitization():
    st = AdvisorState()
    st.add_directives([{"id": "t3", "reason": "done"},
                       {"id": "t999", "reason": "out of range"},
                       {"id": "../etc", "reason": "hostile"},
                       {"id": "t5"}], history_len=20)
    assert set(st.pending) == {3, 5}
    assert st.pending[5] == "no reason"
    assert st.invalid_directives == 2


def test_hook_applies_outside_head_and_recent_only():
    counter = TokenCounter()
    msgs = annotate(_chain(rounds=8))
    st = AdvisorState(keep_recent=4, counter=counter)
    # advise everything, including head-adjacent and recent blocks
    st.add_directives([{"id": f"t{i}", "reason": "r"} for i in range(len(msgs))],
                      history_len=len(msgs))
    out = st.hook(msgs)
    assert validate_chain(out) == []
    assert msgs[3]["content"].startswith("[block t3] result 0")  # no mutation
    assert out[3]["content"] == "[block t3 pruned by agent: r]"
    # recent window survives whatever was advised
    assert out[-2]["content"].startswith("[block t")
    assert "pruned by agent" not in out[-2]["content"]
    # archive holds the original; ledger moved pending -> applied
    assert st.archive[3].startswith("[block t3] result 0")
    assert not st.pending and 3 in st.applied
    assert st.pruned_blocks > 0 and st.freed_tokens > 0
    # idempotent: re-applying to its own output changes nothing
    assert st.hook(out) == out


def test_advice_applies_only_at_checkpoint():
    counter = TokenCounter()
    msgs = annotate(_chain(rounds=12, chars=2000))
    st = AdvisorState(keep_recent=4, counter=counter)
    sc = SessionCompressor(100_000, counter=counter, pre_checkpoint=st.hook)
    st.add_directives([{"id": "t3", "reason": "done"}], history_len=len(msgs))
    # under budget: no checkpoint, no application — prefix must stay stable
    out1 = sc.request(msgs[:8])
    assert sc.checkpoints == 0
    assert any("result 0" in str(m.get("content")) for m in out1)
    out2 = sc.request(msgs)
    assert out2[: len(out1)] == out1  # cache-stable extension
    # force a checkpoint: now (and only now) the advice lands
    sc2 = SessionCompressor(3_000, counter=counter, pre_checkpoint=st.hook)
    st2 = st  # same ledger
    out3 = sc2.request(msgs)
    assert sc2.checkpoints == 1
    flat = json.dumps(out3)
    assert "[block t3 pruned by agent: done]" in flat or "result 0" not in flat


# --------------------------------------------------------------- stripping

def _completion(tool_calls=None, content="ok"):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return json.dumps({"choices": [{"index": 0, "message": msg,
                                    "finish_reason": "tool_calls" if tool_calls else "stop"}],
                       "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()


def test_strip_prune_calls_extracts_and_removes():
    prune = {"id": "x", "type": "function", "function": {
        "name": "prune_context",
        "arguments": json.dumps({"blocks": [{"id": "t3", "reason": "stale"}]})}}
    real = {"id": "y", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
    body, directives, prune_only = strip_prune_calls(_completion([prune, real]))
    assert directives == [{"id": "t3", "reason": "stale"}]
    assert not prune_only
    calls = json.loads(body)["choices"][0]["message"]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["bash"]


def test_strip_prune_only_response_flagged():
    prune = {"id": "x", "type": "function", "function": {
        "name": "prune_context", "arguments": json.dumps({"blocks": []})}}
    body, _, prune_only = strip_prune_calls(_completion([prune], content=""))
    assert prune_only
    choice = json.loads(body)["choices"][0]
    assert "tool_calls" not in choice["message"]
    assert choice["finish_reason"] == "stop"


def test_strip_passes_garbage_through():
    assert strip_prune_calls(b"not json") == (b"not json", [], False)


# ------------------------------------------------------------------- proxy

def _upstream_with_prune(received):
    async def chat(request):
        body = json.loads(await request.body())
        received.append(body)
        prune = {"id": "p1", "type": "function", "function": {
            "name": "prune_context",
            "arguments": json.dumps({"blocks": [{"id": "t3", "reason": "stale"}]})}}
        return JSONResponse({"choices": [{"index": 0, "message": {
            "role": "assistant", "content": "ok", "tool_calls": [prune]}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


@pytest.mark.anyio
async def test_advisor_proxy_end_to_end():
    received: list[dict] = []
    app = build_app("http://u", budget=100_000, advisor=True,
                    client=asgi_client(_upstream_with_prune(received)))
    messages = _chain(rounds=8)
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]

    async with asgi_client(app, "http://proxy") as client:
        resp = await client.post("/v1/chat/completions",
                                 json={"model": "m", "messages": messages,
                                       "tools": tools},
                                 headers={"x-ctxc-session-id": "s1"})
        assert resp.status_code == 200
        # prune tool was injected upstream; markers annotated
        sent = received[0]
        assert any(t["function"]["name"] == "prune_context" for t in sent["tools"])
        assert any(str(m.get("content", "")).startswith("[block t")
                   for m in sent["messages"])
        # prune call never reaches the agent
        msg = resp.json()["choices"][0]["message"]
        assert "tool_calls" not in msg or all(
            c["function"]["name"] != "prune_context" for c in msg["tool_calls"])
        # ledger recorded the directive
        stats = (await client.get("/stats")).json()
        assert stats["mode"] == "advisor"
        assert stats["advisor_directives"] == 1
        assert stats["advisor_pending"] == 1


@pytest.mark.anyio
async def test_advisor_prunes_at_checkpoint():
    received: list[dict] = []
    counter = TokenCounter()
    long = synth_session(rounds=25, seed=9)
    app = build_app("http://u", budget=8_000, advisor=True, counter=counter,
                    client=asgi_client(_upstream_with_prune(received)))
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]
    async with asgi_client(app, "http://proxy") as client:
        # turn 1 plants the directive (upstream always advises pruning t3)
        await client.post("/v1/chat/completions",
                          json={"model": "m", "messages": long[:10], "tools": tools},
                          headers={"x-ctxc-session-id": "s1"})
        # turn 2 blows the budget -> checkpoint -> advice applies
        r = await client.post("/v1/chat/completions",
                              json={"model": "m", "messages": long, "tools": tools},
                              headers={"x-ctxc-session-id": "s1"})
        assert r.status_code == 200
        stats = (await client.get("/stats")).json()
        assert stats["advisor_pruned_blocks"] >= 1
        assert stats["advisor_freed_tokens"] > 0
        sent = received[-1]["messages"]
        assert validate_chain(sent) == []
        # the advised block's content is gone from the emission — either as a
        # stub or evicted outright by the ladder afterwards; both are wins
        original_t3 = str(long[3].get("content", ""))
        assert original_t3 and original_t3 not in json.dumps(sent)


def test_inject_tool_idempotent():
    body = {"tools": [PRUNE_TOOL]}
    inject_tool(body)
    assert len(body["tools"]) == 1


# ----------------------------------------------------------------- sidecar

def _upstream_with_sidecar(received, sidecar_seen):
    """Serves the agent's normal completion AND answers the proxy's own
    advisory query (recognized by its response_format request)."""
    async def chat(request):
        body = json.loads(await request.body())
        if body.get("response_format", {}).get("type") == "json_object":
            sidecar_seen.append(body)
            return JSONResponse({"choices": [{"index": 0, "message": {
                "role": "assistant",
                "content": json.dumps({"evict": [{"id": "t3", "reason": "stale"}]})}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        received.append(body)
        return JSONResponse({"choices": [{"index": 0, "message": {
            "role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


@pytest.mark.anyio
async def test_sidecar_fires_under_pressure_and_feeds_ledger():
    import asyncio

    received, sidecar_seen = [], []
    counter = TokenCounter()
    long = synth_session(rounds=25, seed=11)
    # a prefix in the pressure window: near budget (>85%) but under it, so the
    # emission passes through uncompressed and pressure is immediate
    k = next(i for i in range(len(long), 0, -1)
             if counter.count_chain(long[:i]) < 19_800)
    prefix = long[:k]
    assert counter.count_chain(prefix) > 17_100  # > 0.85 * effective budget
    app = build_app("http://u", budget=20_000, advisor=True, counter=counter,
                    client=asgi_client(_upstream_with_sidecar(received, sidecar_seen)))
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]
    async with asgi_client(app, "http://proxy") as client:
        await client.post("/v1/chat/completions",
                          json={"model": "m", "messages": prefix, "tools": tools},
                          headers={"x-ctxc-session-id": "s1"})
        for _ in range(50):  # let the fire-and-forget task run
            await asyncio.sleep(0.01)
            if sidecar_seen:
                break
        stats = (await client.get("/stats")).json()
        assert stats["advisor_sidecar_calls"] == 1
        assert stats["advisor_pending"] == 1  # t3 waiting for next checkpoint
        # cache-aligned shape: emission verbatim as prefix, instruction last
        sq = sidecar_seen[0]["messages"]
        assert "context manager" in sq[-1]["content"]
        assert any("[block t" in str(m.get("content")) for m in sq[:-1])
        assert sq[0] == prefix[0]  # shares the agent's cached prefix
        assert sidecar_seen[0]["model"] == "m"


# ---------------------------------------------------------------- reminders

def test_reminder_only_under_pressure_and_sticky():
    from ctxc.advisor import REMINDER

    msgs = _chain(rounds=8)
    st = AdvisorState(keep_recent=4)
    # no pressure: plain annotation, no reminder anywhere
    a1 = st.annotate_input(msgs[:10], budget=10_000)
    assert not any(REMINDER in str(m.get("content")) for m in a1)
    # pressure: reminder lands on the NEWEST tool result
    st.last_emitted_tokens = 9_000
    a2 = st.annotate_input(msgs[:12], budget=10_000)
    newest_tool = max(i for i, m in enumerate(a2) if m.get("role") == "tool")
    assert a2[newest_tool]["content"].endswith(REMINDER)
    assert st.reminders_sent == 1
    # sticky: next turn (pressure gone) the same message still carries it,
    # so the annotated history extends the previous one byte-for-byte
    st.last_emitted_tokens = 0
    a3 = st.annotate_input(msgs, budget=10_000)
    assert a3[: len(a2)] == a2
    assert st.reminders_sent == 1  # no new reminder without pressure
