from ctxc.compressor import CompressConfig
from ctxc.models import validate_chain
from ctxc.session import SessionCompressor
from ctxc.tokens import TokenCounter


def _replay(messages, sc):
    """Feed growing prefixes (one per assistant turn), collecting emissions.

    Each entry is (checkpoints_after_this_request, emitted): an increase over
    the previous entry means the checkpoint fired *during* this request.
    """
    emissions = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant" or i == 0:
            continue
        emitted = sc.request(messages[:i])
        emissions.append((sc.checkpoints, emitted))
    return emissions


def test_prefix_stable_between_checkpoints(session_messages):
    counter = TokenCounter()
    sc = SessionCompressor(budget=40_000, counter=counter)
    emissions = _replay(session_messages, sc)
    assert len(emissions) > 5
    for (cp_prev, prev), (cp_now, now) in zip(emissions, emissions[1:]):
        if cp_now == cp_prev:  # no checkpoint fired between these requests
            assert now[: len(prev)] == prev, "emitted prefix changed without a checkpoint"


def test_checkpoints_fire_and_are_rare(session_messages):
    counter = TokenCounter()
    sc = SessionCompressor(budget=40_000, counter=counter)
    emissions = _replay(session_messages, sc)
    assert sc.checkpoints >= 1, "a long session must cross the trigger"
    assert sc.checkpoints < len(emissions) / 2, "checkpoints must be rare, not per-turn"


def test_every_emission_under_trigger_after_checkpoint(session_messages):
    counter = TokenCounter()
    budget = 40_000
    sc = SessionCompressor(budget=budget, counter=counter)
    for _, emitted in _replay(session_messages, sc):
        assert validate_chain(emitted) == []
        # emissions may drift above the recompress target between checkpoints but
        # a request that crossed the trigger must have been compressed back under it
        assert counter.count_chain(emitted) <= budget


def test_hysteresis_compresses_below_trigger(session_messages):
    counter = TokenCounter()
    cfg = CompressConfig(recompress_to=0.5)
    sc = SessionCompressor(budget=40_000, config=cfg, counter=counter)
    prev_cp = 0
    checked = 0
    for cp, emitted in _replay(session_messages, sc):
        if cp > prev_cp:  # this emission is a fresh checkpoint
            assert counter.count_chain(emitted) <= 40_000 * 0.5
            checked += 1
        prev_cp = cp
    assert checked >= 1


def test_non_append_history_falls_back_to_recompress(small_session):
    counter = TokenCounter()
    sc = SessionCompressor(budget=100_000, counter=counter)
    sc.request(small_session[:4])
    mutated = [dict(m) for m in small_session[:6]]
    mutated[1] = {"role": "user", "content": "a completely different task"}
    out = sc.request(mutated)
    assert validate_chain(out) == []
    assert counter.count_chain(out) <= 100_000
