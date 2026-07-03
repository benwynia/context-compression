"""AIC cost model. 1 AIC = $0.01.

Copilot bills in AI Credits. Whether a given model meters AICs per request, per
token, or both is a moving target, so the rate is fully configurable and both
shapes are computed. Token-metered AIC is where compression saves money;
request-metered AIC is unchanged by compression (the win there is context
headroom, which the verify report states separately).

``DEFAULT_RATE`` is illustrative only — override it with real numbers via the
constructor or a JSON rates file (``{"model": {"per_request": 1.0, ...}}``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

AIC_USD = 0.01


@dataclass(frozen=True)
class AicRate:
    per_request: float = 0.0
    per_1m_input: float = 0.0
    per_1m_output: float = 0.0


# Illustrative haiku-class token metering (~$1/M in, ~$5/M out) plus one AIC per
# request. Replace with real Copilot numbers via --rates.
DEFAULT_RATE = AicRate(per_request=1.0, per_1m_input=100.0, per_1m_output=500.0)


def aic_for(
    rate: AicRate, *, input_tokens: int, output_tokens: int = 0, requests: int = 1
) -> float:
    return (
        requests * rate.per_request
        + input_tokens / 1_000_000 * rate.per_1m_input
        + output_tokens / 1_000_000 * rate.per_1m_output
    )


def usd_for(aic: float) -> float:
    return aic * AIC_USD


def load_rates(path: str | Path) -> dict[str, AicRate]:
    """Load ``{"model-name": {"per_request": .., "per_1m_input": .., ...}}``."""
    data = json.loads(Path(path).read_text())
    return {name: AicRate(**fields) for name, fields in data.items()}
