"""
Input validation utilities for the autocorrection pipeline.

Provides security-focused validation and sanitization functions
for user inputs, configuration values, and API credentials.
All external inputs pass through these validators before processing.

Responsibilities:
    - Validate and sanitize user query strings
    - Validate confidence threshold ranges
    - Validate API key presence and format
    - Prevent injection and resource exhaustion attacks
"""

import logging  # Standard library logging for validation events
import re  # Regular expressions for pattern matching

# Module-level logger for validation operations
logger: logging.Logger = logging.getLogger(__name__)

# Maximum allowed character length for user queries
_MAX_QUERY_LENGTH: int = 10000

# Minimum allowed character length for user queries
_MIN_QUERY_LENGTH: int = 1

# Pattern for detecting potential injection attempts in queries
_INJECTION_PATTERN: re.Pattern = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def validate_query(query: str) -> str:
    """
    Validate and sanitize a user query string.

    Checks the query for emptiness, excessive length, and potentially
    dangerous characters. Strips leading/trailing whitespace and removes
    null bytes and control characters.

    Args:
        query: The raw user query input string to validate.

    Returns:
        str: The sanitized query string with whitespace trimmed
            and control characters removed.

    Raises:
        ValueError: If the query is empty after stripping whitespace.
        ValueError: If the query exceeds the maximum allowed length.
    """
    # Check that the query is a string type
    if not isinstance(query, str):
        raise ValueError("Query must be a string type")
    # Strip leading and trailing whitespace from the query
    stripped_query: str = query.strip()
    # Check that the query is not empty after stripping
    if len(stripped_query) < _MIN_QUERY_LENGTH:
        raise ValueError("Query cannot be empty or contain only whitespace")
    # Check that the query does not exceed maximum length
    if len(stripped_query) > _MAX_QUERY_LENGTH:
        raise ValueError(
            f"Query exceeds maximum length of {_MAX_QUERY_LENGTH} characters "
            f"(got {len(stripped_query)})"
        )
    # Remove control characters that could indicate injection attempts
    sanitized_query: str = _INJECTION_PATTERN.sub("", stripped_query)
    # Log a warning if control characters were found and removed
    if len(sanitized_query) != len(stripped_query):
        logger.warning("Control characters removed from query input")
    # Return the validated and sanitized query string
    return sanitized_query


def validate_confidence_threshold(threshold: float) -> float:
    """
    Validate that the confidence threshold is within valid range.

    The confidence threshold must be a float between 0.0 and 1.0
    inclusive, representing a valid probability score.

    Args:
        threshold: The confidence threshold value to validate.

    Returns:
        float: The validated threshold value (unchanged if valid).

    Raises:
        TypeError: If the threshold is not a numeric type.
        ValueError: If the threshold is outside the range [0.0, 1.0].
    """
    # Check that the threshold is a numeric type
    if not isinstance(threshold, int | float):
        raise TypeError(f"Threshold must be numeric, got: {type(threshold).__name__}")
    # Convert to float for consistent type handling
    threshold_float: float = float(threshold)
    # Validate the threshold is within the valid probability range
    if not 0.0 <= threshold_float <= 1.0:
        raise ValueError(
            f"Confidence threshold must be between 0.0 and 1.0, got: {threshold_float}"
        )
    # Return the validated threshold value
    return threshold_float


def validate_api_key(key: str, provider_name: str) -> None:
    """
    Validate that an API key is present and non-empty.

    Checks that the API key string is not None, not empty, and not
    just whitespace. Does not validate the key format against provider
    specifications to avoid false negatives.

    Args:
        key: The API key string to validate for presence.
        provider_name: Name of the provider for error messaging.

    Raises:
        ValueError: If the key is None, empty, or whitespace-only.
    """
    # Check that the key is not None
    if key is None:
        raise ValueError(f"API key for provider '{provider_name}' is None")
    # Check that the key is a string type
    if not isinstance(key, str):
        raise ValueError(f"API key for provider '{provider_name}' must be a string")
    # Check that the key is not empty or whitespace-only
    if not key.strip():
        raise ValueError(
            f"API key for provider '{provider_name}' is empty. "
            f"Set the appropriate environment variable in your .env file."
        )
    # Log successful validation (without revealing the key)
    logger.debug("API key validated for provider: %s", provider_name)
