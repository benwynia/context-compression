import json

import pytest

from ctxc.guard import ThrashGuard, _fp
from ctxc.models import validate_chain
from ctxc.session import SessionCompressor
from ctxc.tokens import TokenCounter

# A long tool result with the load-bearing detail buried in the MIDDLE, in
# plain prose (no salience patterns) — exactly the shape rung 4 showed the
# heuristics can't protect: truncation keeps head+tail and cuts the middle.
NEEDLE = "the gateway retries four times with a backoff of forty-seven seconds"
FACT = ("alpha filler sentence about unrelated code. " * 20
        + NEEDLE + ". "
        + "omega filler sentence about other matters. " * 20)


def _round(i, content):
    return [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": f"c{i}", "content": content},
    ]


def _filler(i):
    return f"file chunk {i}: " + (f"line {i} of unremarkable code and text. " * 25)


# ------------------------------------------------------------------ unit

def test_scale_escalates_with_checkpoints():
    g = ThrashGuard(escalate_after=2, escalate_factor=1.5, max_scale=4.0)
    assert g.scale() == 1.0
    g.checkpoints_seen = 2
    assert g.scale() == 1.5
    g.checkpoints_seen = 8
    assert g.scale() == 4.0  # capped: 1.5^4 > 4.0
    assert g.effective_budget(1000) == 4000


def test_reread_detection_and_pinning():
    g = ThrashGuard()
    before = _round(0, FACT) + _round(1, _filler(1))
    after = _round(1, _filler(1))  # FACT's round evicted
    g.observe_checkpoint(before, after)
    assert _fp(FACT) in g.evicted
    # agent re-reads the same content (advisor marker differs — must not matter)
    g.note_incoming([{"role": "tool", "tool_call_id": "x",
                      "content": "[block t42] " + FACT}])
    assert g.rereads_detected == 1
    assert g.pin_check(FACT)
    assert g.pin_check("[block t99] " + FACT)  # any marker variant
    assert not g.pin_check(_filler(3))


def test_force_escalate_respects_ceiling():
    g = ThrashGuard(escalate_factor=2.0, max_scale=4.0)
    assert g.force_escalate() and g.scale() == 2.0
    assert g.force_escalate() and g.scale() == 4.0
    assert not g.force_escalate()  # at ceiling: caller must raise


# ------------------------------------------------ closed-loop simulator

def _simulate(guard, turns=30, budget=2600):
    """A miniature rung-12 agent: every turn it reads new filler, and any
    time the NEEDLE is missing from what it can see (truncated or evicted),
    it re-reads the whole FACT — that's the thrash loop."""
    counter = TokenCounter()
    sc = SessionCompressor(budget, counter=counter, guard=guard)
    history = [{"role": "system", "content": "agent rules"},
               {"role": "user", "content": "fix the retry bug ctxc-sim"}]
    history += _round(0, FACT)
    rereads = 0
    emitted = sc.request(history)
    r = 1
    for _ in range(turns):
        if NEEDLE not in json.dumps(emitted):
            history += _round(r, FACT)          # re-read: the thrash move
            rereads += 1
        else:
            history += _round(r, _filler(r))    # normal progress
        r += 1
        emitted = sc.request(history)
        assert validate_chain(emitted) == []
    return sc, rereads, json.dumps(emitted)


def test_closed_loop_thrash_is_broken_by_guard():
    sc_off, rereads_off, _ = _simulate(guard=None)
    guard = ThrashGuard(escalate_after=3, escalate_factor=1.5, max_scale=4.0)
    sc_on, rereads_on, final = _simulate(guard=guard)

    # the ungoverned loop must actually thrash for this test to mean anything
    # (one re-read per checkpoint cycle; ~3 cycles in 30 turns at this budget)
    assert rereads_off >= 2
    # with the guard: the re-read is detected, the fact pinned, the loop dies
    assert guard.rereads_detected >= 1
    assert guard.pin_check(FACT)
    assert rereads_on < rereads_off
    # and the pinned needle survives in the final emission
    assert NEEDLE in final


def test_budget_impossible_escalates_instead_of_failing():
    counter = TokenCounter()
    guard = ThrashGuard(escalate_factor=2.0, max_scale=8.0)
    sc = SessionCompressor(600, counter=counter, guard=guard)
    # the protected head ALONE exceeds the budget: without a guard this is
    # BudgetImpossible by definition (the head is verbatim)
    history = [{"role": "system", "content": "rules"},
               {"role": "user", "content": "task " + "words " * 800}]
    history += _round(0, "big result " * 400)
    out = sc.request(history)
    assert validate_chain(out) == []
    assert guard.forced_escalations >= 1  # grew the budget rather than raising


def test_guard_stats_shape():
    g = ThrashGuard()
    g.checkpoints_seen = 3
    s = g.stats()
    assert set(s) == {"guard_scale", "guard_pinned", "guard_rereads",
                      "guard_forced_escalations"}


# ------------------------------------------------------------------ proxy

@pytest.mark.anyio
async def test_proxy_thrash_guard_flag():
    from conftest import asgi_client, fake_upstream
    from ctxc.proxy import build_app
    from ctxc.synth import synth_session

    received: list[dict] = []
    app = build_app("http://u", budget=8_000, thrash_guard=True,
                    client=asgi_client(fake_upstream(received)))
    async with asgi_client(app, "http://proxy") as client:
        resp = await client.post("/v1/chat/completions",
                                 json={"model": "m",
                                       "messages": synth_session(rounds=25, seed=3)},
                                 headers={"x-ctxc-session-id": "s1"})
        assert resp.status_code == 200
        stats = (await client.get("/stats")).json()
        assert "guard_max_scale" in stats
