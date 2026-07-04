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
    # Optional cache-tier rates (per-token upstreams price cached prefix reads
    # at ~10% and cache writes at ~125% of input). When either is nonzero, the
    # verify report adds a cache-aware cost line — token savings that trade
    # cheap cache reads for recompression cache writes stop looking free.
    per_1m_cache_read: float = 0.0
    per_1m_cache_write: float = 0.0

    @property
    def cache_aware(self) -> bool:
        return bool(self.per_1m_cache_read or self.per_1m_cache_write)


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


def aic_cached_for(
    rate: AicRate, *, cache_read: int, cache_write: int, requests: int = 1
) -> float:
    """Cache-aware prompt cost: every prompt token is either a cache read (the
    unchanged prefix) or a cache write (appended/rewritten). This is where a
    checkpoint's recompression shows up as the re-write it really is."""
    return (
        requests * rate.per_request
        + cache_read / 1_000_000 * rate.per_1m_cache_read
        + cache_write / 1_000_000 * rate.per_1m_cache_write
    )


def usd_for(aic: float) -> float:
    return aic * AIC_USD


def load_rates(path: str | Path) -> dict[str, AicRate]:
    """Load ``{"model-name": {"per_request": .., "per_1m_input": .., ...}}``."""
    data = json.loads(Path(path).read_text())
    return {name: AicRate(**fields) for name, fields in data.items()}
