"""Regression tests for the code-review findings: each test pins one fixed bug."""

import json

import pytest

import ctxc.session as session_mod
from conftest import asgi_client, fake_upstream
from ctxc.compressor import CompressConfig, compress
from ctxc.models import (
    DIGEST_MARKER,
    DUPLICATE_MARKER,
    is_digest,
    protected_head_end,
    validate_chain,
)
from ctxc.proxy import build_app
from ctxc.session import SessionCompressor
from ctxc.strategies import truncate_text
from ctxc.tokens import TokenCounter
from ctxc.verify import verify_session


def _round(n: int, content: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": f"step {n}",
            "tool_calls": [
                {
                    "id": f"c{n}",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"n": n})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"c{n}", "content": content},
    ]


# -- finding 1: head ending with an assistant tool_calls round ----------------- #
def test_head_ending_in_tool_call_round_does_not_crash(counter):
    msgs = [{"role": "system", "content": "sys"}]
    msgs += _round(0, "opening tool result " * 50)  # first non-system = assistant+tools
    for n in range(1, 40):
        msgs += _round(n, f"result {n} " + "filler content " * 120)
    assert validate_chain(msgs) == []

    res = compress(msgs, budget=4_000, counter=counter)
    assert res.compressed_tokens <= 4_000
    assert validate_chain(res.messages) == []
    # the digest may not split the head assistant from its tool results
    digests = [i for i, m in enumerate(res.messages) if is_digest(m)]
    if digests:
        assert digests == [protected_head_end(msgs)]


# -- finding 2: dedupe keep-last vs oldest-first eviction ---------------------- #
def test_no_dangling_duplicate_markers_after_eviction(counter):
    dup = "DUPDUP unique payload " * 150
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    for n in range(30):
        msgs += _round(n, dup if n in (1, 25) else f"blob {n} " + "words " * 200)

    res = compress(msgs, budget=6_000, counter=counter)
    assert validate_chain(res.messages) == []
    texts = [m.get("content") or "" for m in res.messages]
    marker_idx = [i for i, t in enumerate(texts) if t == DUPLICATE_MARKER]
    for mi in marker_idx:
        # a marker promises the content appears later — hold it to that
        assert any("DUPDUP" in t for t in texts[mi + 1 :]), (
            "duplicate marker dangles: full content was evicted"
        )


# -- finding 7: truncate_text never expands ------------------------------------ #
@pytest.mark.parametrize("cap", [0, 1, 5, 80])
def test_truncate_text_never_expands(cap):
    text = "abcdefghij" * 200
    out, did = truncate_text(text, cap)
    assert did
    assert len(out) < len(text)


# -- finding 9: budget-scaled escalation floors -------------------------------- #
def test_small_budget_compresses_instead_of_impossible(counter):
    msgs = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Fix the failing test."},
    ]
    for n in range(10):
        msgs += _round(n, f"result {n} " + "log line " * 100)
    res = compress(msgs, budget=500, counter=counter)
    assert res.compressed_tokens <= 500
    assert validate_chain(res.messages) == []


# -- finding 4: summarizer called once, output capped -------------------------- #
def test_summarizer_called_once_and_capped(counter):
    calls: list[list[str]] = []

    def huge_summarizer(lines):
        calls.append(lines)
        return "SUMMARY-TOKEN " * 50_000  # vastly over any digest cap

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    for n in range(30):
        msgs += _round(n, f"blob {n} " + "words " * 200)

    cfg = CompressConfig(summarizer=huge_summarizer)
    res = compress(msgs, budget=6_000, config=cfg, counter=counter)
    assert res.compressed_tokens <= 6_000  # hook cannot break the budget
    assert len(calls) == 1  # once per compress, not per level/probe
    digest = next(m for m in res.messages if is_digest(m))
    assert "SUMMARY-TOKEN" not in digest["content"]  # over-cap output rejected

    calls.clear()

    def small_summarizer(lines):
        calls.append(lines)
        return f"{len(lines)} rounds happened."

    cfg = CompressConfig(summarizer=small_summarizer)
    res = compress(msgs, budget=6_000, config=cfg, counter=counter)
    assert len(calls) == 1
    digest = next(m for m in res.messages if is_digest(m))
    assert "rounds happened." in digest["content"]  # in-cap output used


# -- finding 3: hysteresis double-ladder degenerate state ---------------------- #
def test_session_skips_doomed_hysteresis_target(counter, monkeypatch):
    compress_calls = []
    real_compress = session_mod.compress

    def counting_compress(messages, budget, config=None, cnt=None):
        compress_calls.append(budget)
        return real_compress(messages, budget, config, cnt)

    monkeypatch.setattr(session_mod, "compress", counting_compress)

    big_task = "task requirement detail " * 700  # ~2.8k tokens, protected head
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": big_task},
    ]
    sc = SessionCompressor(budget=3_500, counter=counter)

    n = 0
    checkpoints_seen = []
    while sc.checkpoints < 2 and n < 200:
        history = list(history)
        history += _round(n, f"result {n} " + "data " * 150)
        history.append({"role": "assistant", "content": f"done {n}"})
        emitted = sc.request(history)
        assert counter.count_chain(emitted) <= 3_500
        checkpoints_seen.append((sc.checkpoints, len(compress_calls)))
        n += 1

    assert sc.checkpoints >= 2
    # first checkpoint: target attempt + hard-budget fallback = 2 compress calls;
    # later checkpoints skip the doomed target: exactly 1 call each
    first_cp_calls = next(c for cp, c in checkpoints_seen if cp == 1)
    assert first_cp_calls == 2
    second_cp_calls = next(c for cp, c in checkpoints_seen if cp == 2)
    assert second_cp_calls == first_cp_calls + 1
    assert sc._target_impossible is True


# -- finding 8: CLI refuses half-specified rates ------------------------------- #
def test_cli_rates_requires_model(tmp_path):
    from ctxc.cli import main

    session_file = tmp_path / "s.json"
    session_file.write_text(json.dumps({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "ok"},
    ]}))
    rates_file = tmp_path / "rates.json"
    rates_file.write_text(json.dumps({"gpt-5": {"per_request": 1.0}}))

    with pytest.raises(SystemExit, match="together"):
        main(["verify", str(session_file), "--rates", str(rates_file)])
    with pytest.raises(SystemExit, match="together"):
        main(["verify", str(session_file), "--model", "gpt-5"])


# -- finding 10: verify protocol + digest well-formedness ----------------------- #
def test_verify_rejects_compressor_without_checkpoints(session_messages):
    class NoCheckpoints:
        def request(self, messages):
            return list(messages)

    with pytest.raises(TypeError, match="checkpoints"):
        verify_session(session_messages, budget=10_000_000,
                       session_compressor=NoCheckpoints())


def test_verify_flags_malformed_digest(session_messages):
    class DoubleDigest:
        checkpoints = 0

        def request(self, messages):
            out = list(messages)
            fake = {"role": "user", "content": f"{DIGEST_MARKER} bogus\n- x"}
            out.insert(2, dict(fake))
            out.insert(4, dict(fake))
            return out

    report = verify_session(session_messages, budget=10_000_000,
                            session_compressor=DoubleDigest())
    assert not report.ok
    assert any("digest malformed" in v for v in report.violations)


# -- finding 6: query string forwarded ------------------------------------------ #
@pytest.mark.anyio
async def test_proxy_forwards_query_string():
    received: list[dict] = []
    app = build_app("http://u", budget=1_000_000,
                    client=asgi_client(fake_upstream(received)))
    async with asgi_client(app) as client:
        resp = await client.post(
            "/v1/chat/completions?api-version=2024-06-01",
            json={"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    assert received[0]["query"] == "api-version=2024-06-01"


# -- finding 5: fallback session key stable across turns ------------------------ #
@pytest.mark.anyio
async def test_proxy_fallback_key_stable_across_turns():
    received: list[dict] = []
    app = build_app("http://u", budget=1_000_000,
                    client=asgi_client(fake_upstream(received)))
    turn1 = [{"role": "user", "content": "no system prompt here"}]
    turn2 = turn1 + [
        {"role": "assistant", "content": "sure"},
        {"role": "user", "content": "continue"},
    ]
    async with asgi_client(app) as client:
        for msgs in (turn1, turn2):
            await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-5", "messages": msgs},
            )
        health = (await client.get("/healthz")).json()
    assert health["sessions"] == 1  # one conversation -> one session


# -- no-op path aliasing --------------------------------------------------------- #
def test_noop_compress_returns_copies(counter):
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    res = compress(msgs, budget=10_000, counter=counter)
    assert res.messages == msgs
    assert res.messages[0] is not msgs[0]  # mutating the result can't corrupt input
