"""Autolyse - prescriptive automated EDA with health scoring."""

from .core import Autolyse
from .findings import Finding, FindingsEngine, HealthScore, Severity
from .insights import GeminiProvider, InsightEngine, LLMProvider, Narrator

__version__ = "2.0.0"

__all__ = [
    "Autolyse",
    "Finding",
    "FindingsEngine",
    "HealthScore",
    "Severity",
    "InsightEngine",
    "Narrator",
    "LLMProvider",
    "GeminiProvider",
    "__version__",
]
