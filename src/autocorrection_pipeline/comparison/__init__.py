"""
Response comparison subpackage.

Provides comparison strategies for evaluating LLM-generated responses
against ideal responses. Supports cosine similarity (embedding-based)
and LLM-as-judge comparison methods.
"""

# Re-export primary comparison interfaces for convenient imports
from autocorrection_pipeline.comparison.base import BaseComparator, ComparisonResult
from autocorrection_pipeline.comparison.cosine_similarity import CosineSimilarityComparator
from autocorrection_pipeline.comparison.llm_judge import LLMJudgeComparator

# Define public API for this subpackage
__all__ = [
    "BaseComparator",
    "ComparisonResult",
    "CosineSimilarityComparator",
    "LLMJudgeComparator",
]
