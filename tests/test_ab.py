"""Tests for the paired A/B comparison tool and per-session proxy stats."""

import json

import pytest

from conftest import asgi_client, fake_upstream
from ctxc.ab import bootstrap_mean_ci, compare, load_results, mcnemar_exact_p, render_ab
from ctxc.aic import AicRate
from ctxc.proxy import build_app
from ctxc.synth import synth_session

RATE = AicRate(per_request=1.0, per_1m_input=100.0, per_1m_output=500.0)


def _row(tid, resolved, prompt, out=1000, requests=10, checkpoints=0, **extra):
    return {
        "task_id": tid, "resolved": resolved, "prompt_tokens": prompt,
        "output_tokens": out, "requests": requests, "checkpoints": checkpoints,
        **extra,
    }


def _write(dirpath, rows):
    dirpath.mkdir(parents=True, exist_ok=True)
    for r in rows:
        (dirpath / f"{r['task_id']}.json").write_text(json.dumps(r))
    return dirpath


def test_mcnemar_exact():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(3, 3) == 1.0
    # 0 vs 8 discordant: p = 2 * (1/2^8) = 0.0078125
    assert mcnemar_exact_p(0, 8) == pytest.approx(2 / 256)
    assert mcnemar_exact_p(8, 0) == mcnemar_exact_p(0, 8)  # symmetric


def test_bootstrap_ci_is_seeded_and_sane():
    deltas = [-10.0, -12.0, -8.0, -11.0, -9.0]
    lo1, hi1 = bootstrap_mean_ci(deltas, seed=0)
    lo2, hi2 = bootstrap_mean_ci(deltas, seed=0)
    assert (lo1, hi1) == (lo2, hi2)  # reproducible
    assert lo1 <= -10.0 <= hi1  # mean inside its own CI
    assert hi1 < 0  # clearly negative deltas -> CI excludes zero


def test_compare_pairs_and_stats(tmp_path):
    ctxc_rows = [
        _row("t1", True, 500_000, checkpoints=2),
        _row("t2", False, 400_000, checkpoints=1),
        _row("t3", True, 100_000, checkpoints=0),
        _row("t4", True, 300_000, checkpoints=1),
        _row("t9", True, 100_000),  # unpaired: control missing
    ]
    control_rows = [
        _row("t1", True, 900_000),
        _row("t2", True, 800_000),   # control-only win (discordant)
        _row("t3", True, 110_000),
        _row("t4", False, 700_000),  # ctxc-only win (discordant)
        _row("t8", False, 100_000),  # unpaired: ctxc missing
    ]
    a = load_results(_write(tmp_path / "a", ctxc_rows))
    b = load_results(_write(tmp_path / "b", control_rows))
    r = compare(a, b, RATE)

    assert r.paired_tasks == ["t1", "t2", "t3", "t4"]
    assert r.unpaired_a == ["t9"] and r.unpaired_b == ["t8"]
    assert (r.ctxc.resolved, r.control.resolved) == (3, 3)
    assert r.only_ctxc_resolved == 1 and r.only_control_resolved == 1
    assert r.mcnemar_p == 1.0  # perfectly balanced discordance
    assert r.ctxc.prompt_tokens == 1_300_000
    assert r.control.prompt_tokens == 2_510_000
    assert r.prompt_tokens_saved_pct == pytest.approx(100 * (1 - 1_300_000 / 2_510_000))
    assert r.aic_saved_pct > 0
    assert r.aic_delta_mean < 0  # ctxc cheaper per task
    assert r.engaged_tasks == 3  # t1, t2, t4
    assert r.engaged_only_ctxc == 1 and r.engaged_only_control == 1

    text = render_ab(r)
    assert "McNemar" in text
    assert "engaged" in text
    assert "WARNING" not in text


def test_compare_warns_when_compression_never_engaged(tmp_path):
    rows_a = [_row("t1", True, 10_000, checkpoints=0)]
    rows_b = [_row("t1", True, 10_000)]
    r = compare(
        load_results(_write(tmp_path / "a", rows_a)),
        load_results(_write(tmp_path / "b", rows_b)),
        RATE,
    )
    assert r.engaged_tasks == 0
    assert "WARNING" in render_ab(r)


def test_cache_aware_row_pricing(tmp_path):
    cached_rate = AicRate(
        per_request=1.0, per_1m_input=100.0, per_1m_output=500.0,
        per_1m_cache_read=10.0, per_1m_cache_write=125.0,
    )
    # same prompt totals; ctxc arm has worse cache behavior (more writes)
    a = {"t": _row("t", True, 1_000_000, cache_read=200_000, cache_write=800_000)}
    b = {"t": _row("t", True, 1_000_000, cache_read=900_000, cache_write=100_000)}
    r = compare(a, b, cached_rate)
    assert r.ctxc.aic > r.control.aic  # cache-aware pricing sees the difference
    flat = compare(a, b, RATE)
    assert flat.ctxc.aic == pytest.approx(flat.control.aic)  # flat pricing is blind


def test_load_results_rejects_duplicates(tmp_path):
    f = tmp_path / "r.jsonl"
    f.write_text('{"task_id": "x", "resolved": true}\n{"task_id": "x", "resolved": false}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_results(f)


def test_cli_ab_report(tmp_path, capsys):
    from ctxc.cli import main

    _write(tmp_path / "a", [_row("t1", True, 200_000, checkpoints=1)])
    _write(tmp_path / "b", [_row("t1", True, 500_000)])
    out_json = tmp_path / "report.json"
    rc = main(["ab", str(tmp_path / "a"), str(tmp_path / "b"), "--json", str(out_json)])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "A/B report" in printed
    data = json.loads(out_json.read_text())
    assert data["paired_tasks"] == ["t1"]


@pytest.mark.anyio
async def test_proxy_per_session_stats():
    received: list[dict] = []
    app = build_app(
        "http://u", budget=20_000, client=asgi_client(fake_upstream(received))
    )
    long_msgs = synth_session(rounds=30, seed=5)
    short_msgs = [{"role": "user", "content": "hi"}]
    async with asgi_client(app) as client:
        await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5", "messages": long_msgs},
            headers={"x-ctxc-session-id": "task-long"},
        )
        await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5", "messages": short_msgs},
            headers={"x-ctxc-session-id": "task-short"},
        )
        per = (await client.get("/stats/sessions")).json()["sessions"]

    assert set(per) == {"task-long", "task-short"}
    long_s, short_s = per["task-long"], per["task-short"]
    assert long_s["requests"] == 1
    assert long_s["checkpoints"] >= 1  # long chain crossed the budget
    assert long_s["emitted_tokens"] <= 20_000
    assert long_s["original_tokens"] > 20_000
    assert short_s["checkpoints"] == 0
    assert short_s["upstream_prompt_tokens"] == 1  # fake upstream's usage block


@pytest.mark.anyio
async def test_passthrough_control_arm():
    received: list[dict] = []
    app = build_app(
        "http://u", budget=20_000, passthrough=True,
        client=asgi_client(fake_upstream(received)),
    )
    long_msgs = synth_session(rounds=30, seed=5)
    async with asgi_client(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5", "messages": long_msgs},
            headers={"x-ctxc-session-id": "ctl-1"},
        )
        per = (await client.get("/stats/sessions")).json()
        top = (await client.get("/stats")).json()

    assert resp.status_code == 200
    assert resp.headers["x-ctxc-mode"] == "passthrough"
    # the defining property: NOTHING is compressed, even far over budget
    assert received[0]["body"]["messages"] == long_msgs
    assert per["mode"] == "passthrough"
    row = per["sessions"]["ctl-1"]
    assert row["checkpoints"] == 0
    assert row["emitted_tokens"] == row["original_tokens"]
    assert top["saved_tokens"] == 0
    assert top["upstream_usage_is"] == "baseline"


def test_shadow_and_passthrough_are_exclusive():
    with pytest.raises(ValueError, match="exclusive"):
        build_app("http://u", budget=1_000, shadow=True, passthrough=True)
