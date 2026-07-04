"""ctxc — cache-aware context compression for OpenAI/Copilot-dialect agent chains."""

from .compressor import BudgetImpossible, CompressConfig, CompressResult, compress
from .session import SessionCompressor
from .verify import verify_session

__all__ = [
    "BudgetImpossible",
    "CompressConfig",
    "CompressResult",
    "SessionCompressor",
    "compress",
    "verify_session",
]
