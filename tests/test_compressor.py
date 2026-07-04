import pytest

from ctxc.compressor import BudgetImpossible, CompressConfig, compress
from ctxc.models import (
    DUPLICATE_MARKER,
    TRUNCATION_MARKER,
    content_text,
    head_len,
    is_digest,
    validate_chain,
)
from ctxc.synth import synth_session


def test_noop_when_under_budget(small_session, counter):
    total = counter.count_chain(small_session)
    res = compress(small_session, budget=total + 1000, counter=counter)
    assert res.messages == small_session
    assert res.compressed_tokens == res.original_tokens


@pytest.mark.parametrize("budget", [3_000, 6_000, 12_000, 30_000, 60_000])
def test_budget_guarantee_across_sweep(session_messages, counter, budget):
    """The hard guarantee: under budget, or an explicit BudgetImpossible."""
    try:
        res = compress(session_messages, budget=budget, counter=counter)
    except BudgetImpossible:
        # acceptable only for budgets smaller than the irreducible core
        assert budget <= 3_000
        return
    assert counter.count_chain(res.messages) <= budget
    assert res.compressed_tokens <= budget
    assert validate_chain(res.messages) == []


def test_structure_valid_after_compression(session_messages, counter):
    res = compress(session_messages, budget=20_000, counter=counter)
    assert validate_chain(res.messages) == []


def test_head_preserved_verbatim(session_messages, counter):
    res = compress(session_messages, budget=20_000, counter=counter)
    h = head_len(session_messages)
    assert res.messages[:h] == session_messages[:h]


def test_recent_tail_preserved_verbatim(session_messages, counter):
    cfg = CompressConfig(keep_recent=6)
    res = compress(session_messages, budget=60_000, config=cfg, counter=counter)
    # the last keep_recent source messages must appear verbatim at the tail
    assert res.messages[-6:] == session_messages[-6:]


def test_truncation_marker_present(session_messages, counter):
    res = compress(session_messages, budget=60_000, counter=counter)
    texts = [content_text(m) for m in res.messages if m.get("role") == "tool"]
    assert any(TRUNCATION_MARKER in t for t in texts)


def test_duplicate_elision_keeps_last():
    msgs = synth_session(rounds=30, seed=11, duplicate_every=3)
    from ctxc.tokens import TokenCounter

    counter = TokenCounter()
    # budget between stage-1 and full: force dedupe to run
    res = compress(msgs, budget=25_000, counter=counter)
    texts = [content_text(m) for m in res.messages if m.get("role") == "tool"]
    marker_positions = [i for i, t in enumerate(texts) if t == DUPLICATE_MARKER]
    if marker_positions:
        # keep-LAST: every marker must have a surviving non-marker result after
        # it (so eviction, which removes oldest first, can never strand markers
        # pointing at content that was itself removed)
        last_real = max(
            i for i, t in enumerate(texts) if t != DUPLICATE_MARKER
        )
        assert max(marker_positions) < last_real
    assert validate_chain(res.messages) == []


def test_digest_inserted_after_head_when_evicting(session_messages, counter):
    res = compress(session_messages, budget=8_000, counter=counter)
    h = head_len(session_messages)
    digests = [i for i, m in enumerate(res.messages) if is_digest(m)]
    if res.evicted_rounds:
        assert digests == [h]


def test_digests_never_nest(session_messages, counter):
    res1 = compress(session_messages, budget=12_000, counter=counter)
    res2 = compress(res1.messages, budget=6_000, counter=counter)
    digests = [m for m in res2.messages if is_digest(m)]
    assert len(digests) <= 1
    assert validate_chain(res2.messages) == []


def test_budget_impossible_is_explicit(counter):
    huge_tail = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task " + "x " * 5000},
    ]
    with pytest.raises(BudgetImpossible):
        compress(huge_tail, budget=50, counter=counter)


def test_stats_are_consistent(session_messages, counter):
    res = compress(session_messages, budget=20_000, counter=counter)
    assert res.original_tokens == counter.count_chain(session_messages)
    assert res.compressed_tokens == counter.count_chain(res.messages)
    assert res.original_tokens > res.compressed_tokens
    assert res.stages_applied  # something happened
