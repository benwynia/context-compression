import pytest

from ctxc.aic import AIC_USD, AicRate, aic_for, usd_for


def test_aic_usd_is_one_cent():
    assert AIC_USD == 0.01


def test_token_metered_aic():
    rate = AicRate(per_request=0.0, per_1m_input=100.0, per_1m_output=500.0)
    aic = aic_for(rate, input_tokens=2_000_000, output_tokens=1_000_000, requests=10)
    assert aic == pytest.approx(2 * 100.0 + 1 * 500.0)


def test_request_metered_aic():
    rate = AicRate(per_request=1.0, per_1m_input=0.0, per_1m_output=0.0)
    aic = aic_for(rate, input_tokens=5_000_000, output_tokens=0, requests=42)
    assert aic == pytest.approx(42.0)


def test_usd_conversion():
    assert usd_for(250.0) == pytest.approx(2.50)


def test_mixed_rate():
    rate = AicRate(per_request=1.0, per_1m_input=100.0, per_1m_output=0.0)
    aic = aic_for(rate, input_tokens=1_000_000, output_tokens=0, requests=3)
    assert aic == pytest.approx(3.0 + 100.0)
