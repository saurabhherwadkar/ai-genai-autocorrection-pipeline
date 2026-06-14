"""
Unit tests for the cosine similarity comparator module.

Tests embedding-based comparison including vector math, edge cases,
and provider error handling.
"""

from unittest.mock import AsyncMock  # Mocking utilities for async code

import pytest  # Test framework

from autocorrection_pipeline.comparison.base import (  # Result types
    ComparisonError,
    ComparisonResult,
)
from autocorrection_pipeline.comparison.cosine_similarity import (
    CosineSimilarityComparator,  # Class under test
)
from autocorrection_pipeline.providers.base import ProviderError  # Provider error type


class TestCosineSimilarityCompare:
    """Tests for the cosine similarity compare method."""

    @pytest.mark.asyncio
    async def test_identical_embeddings_score_one(self) -> None:
        """Verify identical embeddings produce a score of 1.0."""
        # Create mock provider returning identical embeddings
        mock_provider: AsyncMock = AsyncMock()
        mock_provider.generate_embedding.return_value = [1.0, 0.0, 0.0]
        # Create comparator and compare
        comparator: CosineSimilarityComparator = CosineSimilarityComparator(mock_provider)
        result: ComparisonResult = await comparator.compare("same text", "same text")
        # Verify score is 1.0 for identical vectors
        assert result.confidence_score == pytest.approx(1.0, abs=0.001)
        assert result.method == "cosine_similarity"

    @pytest.mark.asyncio
    async def test_orthogonal_embeddings_score_zero(self) -> None:
        """Verify orthogonal embeddings produce a score of 0.0."""
        # Create mock provider returning orthogonal embeddings
        mock_provider: AsyncMock = AsyncMock()
        # First call returns [1, 0], second returns [0, 1]
        mock_provider.generate_embedding.side_effect = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        # Create comparator and compare
        comparator: CosineSimilarityComparator = CosineSimilarityComparator(mock_provider)
        result: ComparisonResult = await comparator.compare("text a", "text b")
        # Verify score is 0.0 for orthogonal vectors
        assert result.confidence_score == pytest.approx(0.0, abs=0.001)

    @pytest.mark.asyncio
    async def test_similar_embeddings_score_between_zero_and_one(self) -> None:
        """Verify partially similar embeddings produce intermediate score."""
        # Create mock provider returning similar but not identical embeddings
        mock_provider: AsyncMock = AsyncMock()
        mock_provider.generate_embedding.side_effect = [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
        # Create comparator and compare
        comparator: CosineSimilarityComparator = CosineSimilarityComparator(mock_provider)
        result: ComparisonResult = await comparator.compare("text a", "text b")
        # Verify score is between 0 and 1
        assert 0.0 < result.confidence_score < 1.0

    @pytest.mark.asyncio
    async def test_provider_error_raises_comparison_error(self) -> None:
        """Verify provider errors are wrapped in ComparisonError."""
        # Create mock provider that raises ProviderError
        mock_provider: AsyncMock = AsyncMock()
        mock_provider.generate_embedding.side_effect = ProviderError("test", "Failed")
        # Create comparator and attempt comparison
        comparator: CosineSimilarityComparator = CosineSimilarityComparator(mock_provider)
        with pytest.raises(ComparisonError, match="Embedding generation failed"):
            await comparator.compare("text a", "text b")

    @pytest.mark.asyncio
    async def test_result_contains_explanation(self) -> None:
        """Verify result explanation includes score and dimensions."""
        # Create mock provider with known embeddings
        mock_provider: AsyncMock = AsyncMock()
        mock_provider.generate_embedding.return_value = [0.5, 0.5, 0.5]
        # Create comparator and compare
        comparator: CosineSimilarityComparator = CosineSimilarityComparator(mock_provider)
        result: ComparisonResult = await comparator.compare("text a", "text b")
        # Verify explanation contains useful information
        assert "Cosine similarity score" in result.explanation
        assert "dimension" in result.explanation


class TestComputeCosineSimilarity:
    """Tests for the static cosine similarity computation."""

    def test_identical_vectors(self) -> None:
        """Verify identical vectors produce similarity of 1.0."""
        # Compute similarity for identical vectors
        score: float = CosineSimilarityComparator._compute_cosine_similarity(
            [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]
        )
        # Verify result is 1.0
        assert score == pytest.approx(1.0, abs=0.0001)

    def test_opposite_vectors(self) -> None:
        """Verify opposite vectors produce similarity of -1.0."""
        # Compute similarity for opposite vectors
        score: float = CosineSimilarityComparator._compute_cosine_similarity(
            [1.0, 0.0], [-1.0, 0.0]
        )
        # Verify result is -1.0
        assert score == pytest.approx(-1.0, abs=0.0001)

    def test_zero_vector_returns_zero(self) -> None:
        """Verify zero vector produces similarity of 0.0."""
        # Compute similarity with a zero vector
        score: float = CosineSimilarityComparator._compute_cosine_similarity(
            [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]
        )
        # Verify result is 0.0 (not NaN)
        assert score == 0.0

    def test_both_zero_vectors_returns_zero(self) -> None:
        """Verify two zero vectors produce similarity of 0.0."""
        # Compute similarity with both vectors as zero
        score: float = CosineSimilarityComparator._compute_cosine_similarity(
            [0.0, 0.0], [0.0, 0.0]
        )
        # Verify result is 0.0 (not NaN or inf)
        assert score == 0.0

    def test_unit_vectors(self) -> None:
        """Verify unit vectors produce correct cosine angle."""
        # Use 45-degree angle vectors
        import math
        vec_a: list[float] = [1.0, 0.0]
        vec_b: list[float] = [math.cos(math.pi / 4), math.sin(math.pi / 4)]
        score: float = CosineSimilarityComparator._compute_cosine_similarity(vec_a, vec_b)
        # cos(45°) ≈ 0.7071
        assert score == pytest.approx(0.7071, abs=0.001)
