"""Tests for the measurement pilot kit: shadow mode, recording, /stats,
cache-tier AIC pricing — the pieces that turn synthetic percentages into
numbers measured on real traffic."""

import json

import pytest

from conftest import asgi_client, fake_upstream
from ctxc.aic import AicRate, aic_cached_for
from ctxc.proxy import build_app
from ctxc.synth import synth_session
from ctxc.tokens import TokenCounter
from ctxc.verify import render_report, verify_session


@pytest.mark.anyio
async def test_shadow_mode_forwards_original_and_measures():
    received: list[dict] = []
    app = build_app(
        "http://u", budget=20_000, shadow=True,
        client=asgi_client(fake_upstream(received)),
    )
    messages = synth_session(rounds=30, seed=5)
    counter = TokenCounter()
    assert counter.count_chain(messages) > 20_000

    async with asgi_client(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5", "messages": messages},
            headers={"x-ctxc-session-id": "sh1"},
        )
        stats = (await client.get("/stats")).json()

    assert resp.status_code == 200
    assert resp.headers["x-ctxc-mode"] == "shadow"
    # the defining property: upstream saw the ORIGINAL, untouched
    assert received[0]["body"]["messages"] == messages
    # ...while the would-be savings were still measured
    assert int(resp.headers["x-ctxc-emitted-tokens"]) <= 20_000
    assert stats["mode"] == "shadow"
    assert stats["saved_tokens"] > 0
    assert stats["saved_pct"] > 0
    assert stats["upstream_usage_is"] == "baseline"


@pytest.mark.anyio
async def test_shadow_mode_never_fails_a_request():
    received: list[dict] = []
    # budget so small that compression is impossible (protected head alone
    # exceeds it) — in shadow mode the request must still go through
    app = build_app(
        "http://u", budget=50, shadow=True,
        client=asgi_client(fake_upstream(received)),
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task " + "word " * 2000},
    ]
    async with asgi_client(app) as client:
        resp = await client.post(
            "/v1/chat/completions", json={"model": "gpt-5", "messages": messages}
        )
        stats = (await client.get("/stats")).json()
    assert resp.status_code == 200
    assert received[0]["body"]["messages"] == messages
    assert stats["compress_errors"] == 1


@pytest.mark.anyio
async def test_recording_captures_replayable_sessions(tmp_path):
    received: list[dict] = []
    app = build_app(
        "http://u", budget=1_000_000, record_dir=tmp_path,
        client=asgi_client(fake_upstream(received)),
    )
    messages = synth_session(rounds=6, seed=9, result_lines=(5, 10))
    turn1 = messages[:5]
    async with asgi_client(app) as client:
        for msgs in (turn1, messages):
            await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-5", "messages": msgs},
                headers={"x-ctxc-session-id": "rec-1"},
            )
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1  # one conversation -> one file, latest history wins
    recorded = json.loads(files[0].read_text())["messages"]
    assert recorded == messages
    # and the recorded file is directly verifiable
    report = verify_session(recorded, budget=50_000)
    assert report.ok


@pytest.mark.anyio
async def test_stats_capture_upstream_usage():
    received: list[dict] = []
    app = build_app(
        "http://u", budget=1_000_000, client=asgi_client(fake_upstream(received))
    )
    async with asgi_client(app) as client:
        await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]},
        )
        stats = (await client.get("/stats")).json()
    # conftest's fake upstream reports usage {prompt_tokens: 1, completion_tokens: 1}
    assert stats["upstream_prompt_tokens"] == 1
    assert stats["upstream_completion_tokens"] == 1
    assert stats["upstream_usage_is"] == "compressed"


def test_aic_cached_math():
    rate = AicRate(per_1m_input=100.0, per_1m_cache_read=10.0, per_1m_cache_write=125.0)
    assert rate.cache_aware
    aic = aic_cached_for(rate, cache_read=1_000_000, cache_write=1_000_000, requests=0)
    assert aic == pytest.approx(10.0 + 125.0)
    assert not AicRate(per_1m_input=100.0).cache_aware


def test_verify_cache_aware_pricing(session_messages):
    flat = AicRate(per_1m_input=100.0)
    cached = AicRate(
        per_1m_input=100.0, per_1m_cache_read=10.0, per_1m_cache_write=125.0
    )
    plain = verify_session(session_messages, budget=40_000, rate=flat)
    priced = verify_session(session_messages, budget=40_000, rate=cached)

    assert plain.baseline_aic_cached is None  # flat rate: no cache line
    assert priced.baseline_aic_cached is not None
    assert priced.compressed_aic_cached is not None
    # baseline read/write must partition the baseline prompt tokens exactly
    assert (
        priced.baseline_cache_read + priced.baseline_cache_write
        == priced.original_prompt_tokens
    )
    # compression still wins, but by less than the flat-token view suggests:
    # checkpoints turn cheap cache reads into dear cache writes
    def pct(base, post):
        return 100.0 * (base - post) / base

    flat_saved = pct(priced.baseline_aic, priced.compressed_aic)
    cached_saved = pct(priced.baseline_aic_cached, priced.compressed_aic_cached)
    assert cached_saved < flat_saved
    text = render_report(priced)
    assert "cache-aware pricing" in text
