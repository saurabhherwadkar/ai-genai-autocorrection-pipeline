"""
Unit tests for the input validation utilities module.

Tests query validation, threshold validation, and API key
validation with both valid and invalid inputs.
"""

import pytest  # Test framework

from autocorrection_pipeline.utils.validators import (
    validate_api_key,
    validate_confidence_threshold,
    validate_query,
)


class TestValidateQuery:
    """Tests for the query validation function."""

    def test_valid_query_returns_stripped(self) -> None:
        """Verify valid query is returned with whitespace stripped."""
        # Test with a normal query containing leading/trailing spaces
        result: str = validate_query("  What is Python?  ")
        # Verify whitespace was stripped
        assert result == "What is Python?"

    def test_valid_query_unchanged(self) -> None:
        """Verify a clean query passes through unchanged."""
        # Test with a query that needs no modification
        result: str = validate_query("What is machine learning?")
        # Verify the query is unchanged
        assert result == "What is machine learning?"

    def test_empty_query_raises_error(self) -> None:
        """Verify empty string raises ValueError."""
        # Test with empty string
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_query("")

    def test_whitespace_only_query_raises_error(self) -> None:
        """Verify whitespace-only string raises ValueError."""
        # Test with only whitespace characters
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_query("   \t\n   ")

    def test_query_exceeds_max_length_raises_error(self) -> None:
        """Verify query exceeding max length raises ValueError."""
        # Create a query that exceeds the 10000 character limit
        long_query: str = "a" * 10001
        # Test with the oversized query
        with pytest.raises(ValueError, match="exceeds maximum length"):
            validate_query(long_query)

    def test_query_at_max_length_passes(self) -> None:
        """Verify query at exactly max length passes validation."""
        # Create a query at exactly the maximum length
        max_query: str = "a" * 10000
        # Verify it passes validation
        result: str = validate_query(max_query)
        assert len(result) == 10000

    def test_control_characters_removed(self) -> None:
        """Verify control characters are stripped from query."""
        # Create a query with embedded control characters
        query_with_controls: str = "Hello\x00World\x01Test"
        # Validate the query
        result: str = validate_query(query_with_controls)
        # Verify control characters were removed
        assert "\x00" not in result
        assert "\x01" not in result
        assert result == "HelloWorldTest"

    def test_non_string_input_raises_error(self) -> None:
        """Verify non-string input raises ValueError."""
        # Test with integer input
        with pytest.raises(ValueError, match="must be a string"):
            validate_query(12345)  # type: ignore[arg-type]

    def test_newlines_and_tabs_preserved(self) -> None:
        """Verify normal whitespace within query is preserved."""
        # Test with a query containing newlines and tabs
        query: str = "Line 1\nLine 2\tTabbed"
        result: str = validate_query(query)
        # Verify internal whitespace is preserved
        assert "\n" in result
        assert "\t" in result


class TestValidateConfidenceThreshold:
    """Tests for the confidence threshold validation function."""

    def test_valid_float_threshold(self) -> None:
        """Verify valid float threshold returns unchanged."""
        # Test with a typical threshold value
        result: float = validate_confidence_threshold(0.8)
        # Verify the value is unchanged
        assert result == 0.8

    def test_zero_threshold_valid(self) -> None:
        """Verify 0.0 is a valid threshold value."""
        # Test with zero
        result: float = validate_confidence_threshold(0.0)
        assert result == 0.0

    def test_one_threshold_valid(self) -> None:
        """Verify 1.0 is a valid threshold value."""
        # Test with one
        result: float = validate_confidence_threshold(1.0)
        assert result == 1.0

    def test_integer_converted_to_float(self) -> None:
        """Verify integer input is accepted and converted."""
        # Test with integer input
        result: float = validate_confidence_threshold(1)
        assert result == 1.0
        assert isinstance(result, float)

    def test_negative_threshold_raises_error(self) -> None:
        """Verify negative value raises ValueError."""
        # Test with negative threshold
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            validate_confidence_threshold(-0.5)

    def test_above_one_threshold_raises_error(self) -> None:
        """Verify value above 1.0 raises ValueError."""
        # Test with value exceeding maximum
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            validate_confidence_threshold(1.5)

    def test_non_numeric_raises_type_error(self) -> None:
        """Verify non-numeric input raises TypeError."""
        # Test with string input
        with pytest.raises(TypeError, match="must be numeric"):
            validate_confidence_threshold("0.8")  # type: ignore[arg-type]


class TestValidateApiKey:
    """Tests for the API key validation function."""

    def test_valid_api_key_passes(self) -> None:
        """Verify valid API key passes validation without error."""
        # Test with a typical API key format
        validate_api_key("sk-1234567890abcdef", "openai")

    def test_empty_key_raises_error(self) -> None:
        """Verify empty string raises ValueError."""
        # Test with empty key
        with pytest.raises(ValueError, match="is empty"):
            validate_api_key("", "openai")

    def test_whitespace_only_key_raises_error(self) -> None:
        """Verify whitespace-only key raises ValueError."""
        # Test with whitespace string
        with pytest.raises(ValueError, match="is empty"):
            validate_api_key("   ", "anthropic")

    def test_none_key_raises_error(self) -> None:
        """Verify None key raises ValueError."""
        # Test with None value
        with pytest.raises(ValueError, match="is None"):
            validate_api_key(None, "openai")  # type: ignore[arg-type]

    def test_non_string_key_raises_error(self) -> None:
        """Verify non-string key raises ValueError."""
        # Test with integer input
        with pytest.raises(ValueError, match="must be a string"):
            validate_api_key(12345, "openai")  # type: ignore[arg-type]

    def test_error_message_includes_provider_name(self) -> None:
        """Verify error message contains the provider name."""
        # Test that provider name appears in the error
        with pytest.raises(ValueError, match="anthropic"):
            validate_api_key("", "anthropic")
