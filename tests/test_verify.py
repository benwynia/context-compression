from ctxc.aic import AicRate
from ctxc.verify import render_report, verify_session


def test_verify_green_on_synth(session_messages):
    report = verify_session(session_messages, budget=40_000)
    assert report.ok, report.violations
    assert report.violations == []
    assert report.turns >= 10
    assert report.checkpoints >= 1
    assert report.emitted_prompt_tokens < report.original_prompt_tokens
    assert report.compressed_aic < report.baseline_aic


def test_verify_reports_headroom(session_messages):
    # with a tiny model cap the uncompressed chain overflows and ours must not
    report = verify_session(session_messages, budget=40_000, model_cap=50_000)
    assert report.over_cap_before > 0
    assert report.over_cap_after == 0


def test_verify_cache_accounting_sane(session_messages):
    report = verify_session(session_messages, budget=40_000)
    # between checkpoints everything but the appended tail must be cache reads
    assert report.cache_read_tokens > 0
    assert report.cache_write_tokens > 0
    assert (
        report.cache_read_tokens + report.cache_write_tokens
        == report.emitted_prompt_tokens
    )


class _BrokenCompressor:
    """Emits the chain untouched: over budget and prefix-unstable never happens,
    so the harness must flag the budget violations."""

    def __init__(self):
        self.checkpoints = 0

    def request(self, messages):
        return list(messages)


def test_verify_catches_budget_violation(session_messages):
    report = verify_session(
        session_messages, budget=10_000, session_compressor=_BrokenCompressor()
    )
    assert not report.ok
    assert any("budget" in v for v in report.violations)


class _StructureBreaker:
    def __init__(self):
        self.checkpoints = 0

    def request(self, messages):
        out = list(messages)
        out.append({"role": "tool", "tool_call_id": "ghost", "content": "boo"})
        return out


def test_verify_catches_structure_violation(session_messages):
    report = verify_session(
        session_messages, budget=10_000_000, session_compressor=_StructureBreaker()
    )
    assert not report.ok
    assert any("orphan" in v for v in report.violations)


def test_render_report_mentions_aic(session_messages):
    rate = AicRate(per_request=1.0, per_1m_input=100.0, per_1m_output=500.0)
    report = verify_session(session_messages, budget=40_000, rate=rate)
    text = render_report(report)
    assert "AIC" in text
    assert "$" in text
    assert "checkpoints" in text.lower()
