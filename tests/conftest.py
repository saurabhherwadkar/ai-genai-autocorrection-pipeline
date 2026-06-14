"""
Shared pytest fixtures for the autocorrection pipeline test suite.

Provides reusable test fixtures including mock providers, sample
settings, test data, and temporary directories used across all
unit and integration tests.

Responsibilities:
    - Provide a fully populated test settings object
    - Provide mocked LLM provider instances
    - Provide sample ideal response data
    - Provide temporary output directories
"""

from datetime import UTC, datetime  # Timestamp for test records
from pathlib import Path  # Cross-platform path handling
from unittest.mock import AsyncMock  # Mocking utilities for async code

import pytest  # Test framework and fixture decorator

from autocorrection_pipeline.comparison.base import ComparisonResult  # Comparison result type
from autocorrection_pipeline.config.settings import (  # Configuration types
    AppSettings,
    ProviderConfig,
)
from autocorrection_pipeline.correction.capture import CorrectionRecord  # Correction record type


@pytest.fixture
def sample_provider_config() -> ProviderConfig:
    """
    Provide a sample provider configuration for testing.

    Returns:
        ProviderConfig: A test configuration with typical OpenAI settings.
    """
    # Create and return a test provider configuration
    return ProviderConfig(
        model="gpt-4o-test",
        max_tokens=4096,
        temperature=0.0,
        embedding_model="text-embedding-3-small",
        api_version=None,
    )


@pytest.fixture
def sample_settings(tmp_path: Path, sample_provider_config: ProviderConfig) -> AppSettings:
    """
    Provide a fully populated test settings object.

    Uses tmp_path for file paths to avoid polluting real directories.

    Args:
        tmp_path: Pytest-provided temporary directory path.
        sample_provider_config: Sample provider configuration fixture.

    Returns:
        AppSettings: A complete test settings object with all fields populated.
    """
    # Create the ideal responses file in the temporary directory
    ideal_path: Path = tmp_path / "ideal_responses.json"
    # Write sample ideal responses content to the temp file
    ideal_path.write_text(
        '{"queries": [{"query": "What is Python?", '
        '"ideal_response": "Python is a high-level programming language."}]}',
        encoding="utf-8",
    )
    # Create and return the test settings object
    return AppSettings(
        active_provider="openai",
        confidence_threshold=0.8,
        comparison_method="cosine_similarity",
        providers={
            "openai": sample_provider_config,
            "anthropic": ProviderConfig(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                temperature=0.0,
                embedding_model="text-embedding-3-small",
            ),
            "azure_openai": ProviderConfig(
                model="gpt-4o",
                max_tokens=4096,
                temperature=0.0,
                embedding_model="text-embedding-3-small",
                api_version="2024-06-01",
            ),
        },
        ideal_responses_path=ideal_path,
        output_directory=tmp_path / "output",
        log_level="DEBUG",
        openai_api_key="sk-test-key-openai-12345",
        anthropic_api_key="sk-ant-test-key-12345",
        azure_openai_api_key="test-azure-key-12345",
        azure_openai_endpoint="https://test.openai.azure.com/",
    )


@pytest.fixture
def mock_provider() -> AsyncMock:
    """
    Provide a mocked LLM provider with preset responses.

    Returns an AsyncMock that simulates the BaseLLMProvider interface
    with configurable return values for generate_response and
    generate_embedding methods.

    Returns:
        AsyncMock: A mocked provider with default response behaviors.
    """
    # Create the async mock provider instance
    provider: AsyncMock = AsyncMock()
    # Configure the default response for generate_response
    provider.generate_response.return_value = "This is a test response from the mock provider."
    # Configure the default embedding vector (small for testing)
    provider.generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    # Return the configured mock provider
    return provider


@pytest.fixture
def sample_ideal_responses() -> dict[str, str]:
    """
    Provide sample query-ideal response pairs for testing.

    Returns:
        dict[str, str]: Dictionary mapping test queries to ideal responses.
    """
    # Return a dictionary of test query-response pairs
    return {
        "What is Python?": "Python is a high-level programming language.",
        "What is 2+2?": "The sum of 2 and 2 is 4.",
        "What is AI?": "AI stands for Artificial Intelligence.",
    }


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """
    Provide a temporary directory for correction output files.

    Creates the directory within pytest's tmp_path fixture to ensure
    automatic cleanup after test completion.

    Args:
        tmp_path: Pytest-provided temporary directory path.

    Returns:
        Path: Path to the temporary output directory.
    """
    # Create the output subdirectory within tmp_path
    output_dir: Path = tmp_path / "output"
    # Create the directory on the filesystem
    output_dir.mkdir(parents=True, exist_ok=True)
    # Return the directory path
    return output_dir


@pytest.fixture
def sample_comparison_result() -> ComparisonResult:
    """
    Provide a sample comparison result for testing.

    Returns a below-threshold result to trigger correction capture.

    Returns:
        ComparisonResult: A test comparison result with score below 0.8.
    """
    # Create and return a below-threshold comparison result
    return ComparisonResult(
        confidence_score=0.65,
        method="cosine_similarity",
        explanation="Test comparison result with low confidence.",
    )


@pytest.fixture
def sample_correction_record(sample_comparison_result: ComparisonResult) -> CorrectionRecord:
    """
    Provide a sample correction record for testing.

    Args:
        sample_comparison_result: A below-threshold comparison result.

    Returns:
        CorrectionRecord: A complete test correction record.
    """
    # Create and return a test correction record
    return CorrectionRecord(
        query="What is Python?",
        actual_response="Python is a snake.",
        ideal_response="Python is a high-level programming language.",
        comparison_result=sample_comparison_result,
        timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
    )
