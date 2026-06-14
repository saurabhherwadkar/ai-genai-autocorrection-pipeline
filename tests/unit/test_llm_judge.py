"""
Unit tests for the LLM judge comparator module.

Tests LLM-based evaluation including prompt construction, JSON response
parsing, and error handling for malformed responses.
"""

from unittest.mock import AsyncMock  # Mocking utilities for async code

import pytest  # Test framework

from autocorrection_pipeline.comparison.base import (  # Result types
    ComparisonError,
    ComparisonResult,
)
from autocorrection_pipeline.comparison.llm_judge import LLMJudgeComparator  # Class under test
from autocorrection_pipeline.providers.base import ProviderError  # Provider error type


class TestLLMJudgeCompare:
    """Tests for the LLM judge compare method."""

    @pytest.mark.asyncio
    async def test_successful_judgment_with_valid_json(self) -> None:
        """Verify successful judgment with properly formatted JSON response."""
        # Create mock provider returning valid JSON
        mock_provider: AsyncMock = AsyncMock()
        mock_provider.generate_response.return_value = (
            '{"score": 0.85, "explanation": "The responses are very similar."}'
        )
        # Create comparator and compare
        comparator: LLMJudgeComparator = LLMJudgeComparator(mock_provider)
        result: ComparisonResult = await comparator.compare("actual", "ideal")
        # Verify the parsed result
        assert result.confidence_score == pytest.approx(0.85, abs=0.001)
        assert result.method == "llm_judge"
        assert "very similar" in result.explanation

    @pytest.mark.asyncio
    async def test_score_clamped_to_max_one(self) -> None:
        """Verify scores above 1.0 are clamped to 1.0."""
        # Create mock provider returning score above 1.0
        mock_provider: AsyncMock = AsyncMock()
        mock_provider.generate_response.return_value = (
            '{"score": 1.5, "explanation": "Exceeded maximum."}'
        )
        # Create comparator and compare
        comparator: LLMJudgeComparator = LLMJudgeComparator(mock_provider)
        result: ComparisonResult = await comparator.compare("actual", "ideal")
        # Verify score is clamped to 1.0
        assert result.confidence_score == 1.0

    @pytest.mark.asyncio
    async def test_score_clamped_to_min_zero(self) -> None:
        """Verify scores below 0.0 are clamped to 0.0."""
        # Create mock provider returning negative score
        mock_provider: AsyncMock = AsyncMock()
        mock_provider.generate_response.return_value = (
            '{"score": -0.5, "explanation": "Negative score test."}'
        )
        # Create comparator and compare
        comparator: LLMJudgeComparator = LLMJudgeComparator(mock_provider)
        result: ComparisonResult = await comparator.compare("actual", "ideal")
        # Verify score is clamped to 0.0
        assert result.confidence_score == 0.0

    @pytest.mark.asyncio
    async def test_json_embedded_in_text(self) -> None:
        """Verify JSON extraction from surrounding text."""
        # Create mock provider returning JSON with surrounding text
        mock_provider: AsyncMock = AsyncMock()
        mock_provider.generate_response.return_value = (
            'Here is my evaluation:\n'
            '{"score": 0.72, "explanation": "Partially correct."}\n'
            'That is my assessment.'
        )
        # Create comparator and compare
        comparator: LLMJudgeComparator = LLMJudgeComparator(mock_provider)
        result: ComparisonResult = await comparator.compare("actual", "ideal")
        # Verify the extracted JSON was parsed correctly
        assert result.confidence_score == pytest.approx(0.72, abs=0.001)
        assert "Partially correct" in result.explanation

    @pytest.mark.asyncio
    async def test_malformed_json_raises_comparison_error(self) -> None:
        """Verify unparseable response raises ComparisonError."""
        # Create mock provider returning invalid JSON
        mock_provider: AsyncMock = AsyncMock()
        mock_provider.generate_response.return_value = "This is not valid JSON at all"
        # Create comparator and attempt comparison
        comparator: LLMJudgeComparator = LLMJudgeComparator(mock_provider)
        with pytest.raises(ComparisonError, match="Failed to parse"):
            await comparator.compare("actual", "ideal")

    @pytest.mark.asyncio
    async def test_provider_error_raises_comparison_error(self) -> None:
        """Verify provider errors are wrapped in ComparisonError."""
        # Create mock provider that raises ProviderError
        mock_provider: AsyncMock = AsyncMock()
        mock_provider.generate_response.side_effect = ProviderError("test", "API failed")
        # Create comparator and attempt comparison
        comparator: LLMJudgeComparator = LLMJudgeComparator(mock_provider)
        with pytest.raises(ComparisonError, match="Judge LLM call failed"):
            await comparator.compare("actual", "ideal")

    @pytest.mark.asyncio
    async def test_json_missing_score_key_raises_error(self) -> None:
        """Verify JSON without 'score' key raises ComparisonError."""
        # Create mock provider returning JSON without score key
        mock_provider: AsyncMock = AsyncMock()
        mock_provider.generate_response.return_value = (
            '{"rating": 0.9, "explanation": "Missing score key."}'
        )
        # Create comparator and attempt comparison
        comparator: LLMJudgeComparator = LLMJudgeComparator(mock_provider)
        with pytest.raises(ComparisonError, match="Failed to parse"):
            await comparator.compare("actual", "ideal")


class TestBuildJudgePrompt:
    """Tests for the judge prompt construction."""

    def test_prompt_contains_both_responses(self) -> None:
        """Verify built prompt includes actual and ideal responses."""
        # Create the comparator
        mock_provider: AsyncMock = AsyncMock()
        comparator: LLMJudgeComparator = LLMJudgeComparator(mock_provider)
        # Build the prompt
        prompt: str = comparator._build_judge_prompt("actual text", "ideal text")
        # Verify both responses are in the prompt
        assert "actual text" in prompt
        assert "ideal text" in prompt
        # Verify section headers are present
        assert "Actual Response" in prompt
        assert "Ideal Response" in prompt

    def test_prompt_contains_json_instructions(self) -> None:
        """Verify prompt instructs JSON output format."""
        # Create the comparator
        mock_provider: AsyncMock = AsyncMock()
        comparator: LLMJudgeComparator = LLMJudgeComparator(mock_provider)
        # Build the prompt
        prompt: str = comparator._build_judge_prompt("actual", "ideal")
        # Verify JSON format instructions are present
        assert "JSON" in prompt
        assert "score" in prompt


class TestParseJudgeResponse:
    """Tests for the judge response parser."""

    def test_parse_valid_json(self) -> None:
        """Verify valid JSON is parsed correctly."""
        # Parse a valid JSON response
        score, explanation = LLMJudgeComparator._parse_judge_response(
            '{"score": 0.9, "explanation": "Very good match."}'
        )
        # Verify parsed values
        assert score == pytest.approx(0.9)
        assert explanation == "Very good match."

    def test_parse_json_with_whitespace(self) -> None:
        """Verify JSON with surrounding whitespace is parsed."""
        # Parse JSON with extra whitespace
        score, explanation = LLMJudgeComparator._parse_judge_response(
            '  \n  {"score": 0.5, "explanation": "Average."} \n '
        )
        # Verify parsed values
        assert score == pytest.approx(0.5)
        assert explanation == "Average."

    def test_parse_invalid_json_raises_error(self) -> None:
        """Verify completely invalid input raises ValueError."""
        # Attempt to parse non-JSON text
        with pytest.raises(ValueError, match="Could not parse"):
            LLMJudgeComparator._parse_judge_response("Not JSON at all")

    def test_parse_json_missing_keys_raises_error(self) -> None:
        """Verify JSON without required keys raises ValueError."""
        # Attempt to parse JSON without required keys
        with pytest.raises(ValueError, match="Could not parse"):
            LLMJudgeComparator._parse_judge_response('{"other": "value"}')
