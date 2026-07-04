"""Tests for the rung-4 fixes: salience-aware retention and pinning."""

import json
import re

import pytest

from ctxc.compressor import BudgetImpossible, CompressConfig, compress
from ctxc.models import validate_chain
from ctxc.probe import run_probes
from ctxc.strategies import DEFAULT_SALIENCE, salient_lines, truncate_text
from ctxc.synth import synth_session

SAL = re.compile(DEFAULT_SALIENCE)


def test_salient_lines_extraction():
    text = (
        "plain narrative line one\n"
        "ERROR: connection refused on retry 3\n"
        "more filler text here\n"
        "we decided to use exponential backoff\n"
        "NOTE: the deploy code is ABC-4321\n"
        "trailing filler\n"
    )
    lines = salient_lines(text, SAL, max_lines=3)
    assert "ERROR: connection refused on retry 3" in lines
    assert "we decided to use exponential backoff" in lines
    assert "NOTE: the deploy code is ABC-4321" in lines
    assert "plain narrative line one" not in lines


def test_truncation_keeps_salient_middle():
    filler = "uninteresting log line about nothing\n" * 200
    text = filler + "AUDIT-NOTE: the master code is FROST-9911\n" + filler
    out, did = truncate_text(text, 800, salience=SAL)
    assert did
    assert "FROST-9911" in out  # deep-middle fact survives position-blind
    assert len(out) < len(text)


def test_evicted_round_facts_survive_in_digest(counter):
    msgs = synth_session(rounds=30, seed=7)
    # plant a salient-shaped fact mid-content of an EARLY round (will be evicted)
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and i < 8:
            body = m["content"]
            mid = len(body) // 2
            msgs[i] = dict(m)
            msgs[i]["content"] = (
                body[:mid] + "\nNOTE: the rollback code is EMBER-7777\n" + body[mid:]
            )
            break
    res = compress(msgs, budget=6_000, counter=counter)
    assert res.evicted_rounds > 0
    flat = json.dumps(res.messages)
    assert "EMBER-7777" in flat  # the rung-4 cliff, fixed
    assert validate_chain(res.messages) == []


def test_pinned_round_immune_to_eviction_and_truncation(counter):
    msgs = synth_session(rounds=30, seed=7)
    pinned_idx = None
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and i < 8:
            msgs[i] = dict(m)
            msgs[i]["content"] = m["content"] + "\nctxc:pin critical repro steps here"
            pinned_idx = i
            break
    res = compress(msgs, budget=8_000, counter=counter)
    assert res.compressed_tokens <= 8_000
    # the pinned message survives VERBATIM: not truncated, not evicted
    assert msgs[pinned_idx] in res.messages
    assert validate_chain(res.messages) == []


def test_eviction_skips_pinned_but_still_meets_budget(counter):
    """One pinned early round must not block later rounds from evicting."""
    msgs = synth_session(rounds=30, seed=7)
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and i < 8:
            msgs[i] = dict(m)
            msgs[i]["content"] = m["content"][:400] + "\nctxc:pin keep me"
            break
    res = compress(msgs, budget=6_000, counter=counter)
    assert res.compressed_tokens <= 6_000
    assert res.evicted_rounds > 0  # eviction continued past the pin
    assert "ctxc:pin keep me" in json.dumps(res.messages)


def test_overpinning_fails_loud(counter):
    """Pinning more than the budget can hold is an explicit error, not silence."""
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "task"}]
    for n in range(6):
        msgs += [
            {"role": "assistant", "content": f"s{n}", "tool_calls": [
                {"id": f"c{n}", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{n}",
             "content": "ctxc:pin " + f"critical {n} " * 400},
        ]
    with pytest.raises(BudgetImpossible):
        compress(msgs, budget=500, counter=counter)


def test_salience_off_restores_old_behavior(counter):
    cfg = CompressConfig(salience_pattern=None, pin_pattern=None)
    msgs = synth_session(rounds=30, seed=7)
    res = compress(msgs, budget=8_000, config=cfg, counter=counter)
    assert res.compressed_tokens <= 8_000
    assert validate_chain(res.messages) == []


# ---- the honest re-measure: probes by style --------------------------------- #
def test_note_probes_now_survive_eviction(counter):
    """Salient-shaped facts must survive at rates far above the pre-fix cliff."""
    msgs = synth_session(rounds=30, seed=7)
    report = run_probes(msgs, budget=6_000, n=8, style="note", counter=counter)
    assert report.compressed_tokens <= 6_000
    assert report.survival_rate >= 0.75  # was 25% at this budget before the fix


def test_plain_probes_measure_residual_loss(counter):
    """Pattern-free prose facts must NOT be flattered by the salience fix —
    deep-compression loss should still be visible for them."""
    msgs = synth_session(rounds=30, seed=7)
    plain = run_probes(msgs, budget=6_000, n=8, style="plain", counter=counter)
    note = run_probes(msgs, budget=6_000, n=8, style="note", counter=counter)
    assert plain.survival_rate <= note.survival_rate
    # the plain facts planted into evicted regions are genuinely lost
    assert any(not o.survived for o in plain.outcomes)
