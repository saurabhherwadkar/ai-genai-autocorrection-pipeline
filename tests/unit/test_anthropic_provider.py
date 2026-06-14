"""
Unit tests for the Anthropic LLM provider implementation.

Tests response generation, embedding fallback, and error handling
using mocked Anthropic client responses.
"""

from unittest.mock import AsyncMock, MagicMock, patch  # Mocking utilities

import pytest  # Test framework

from autocorrection_pipeline.config.settings import ProviderConfig  # Config type
from autocorrection_pipeline.providers.anthropic_provider import (
    AnthropicProvider,  # Class under test
)
from autocorrection_pipeline.providers.base import ProviderError  # Error type


@pytest.fixture
def anthropic_config() -> ProviderConfig:
    """Provide a test configuration for Anthropic provider."""
    # Create and return test Anthropic configuration
    return ProviderConfig(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        temperature=0.0,
        embedding_model="text-embedding-3-small",
    )


class TestAnthropicProviderInit:
    """Tests for Anthropic provider initialization."""

    def test_init_with_valid_key(self, anthropic_config: ProviderConfig) -> None:
        """Verify successful initialization with valid API key."""
        # Create provider with mocked client
        with (
            patch("autocorrection_pipeline.providers.anthropic_provider.AsyncAnthropic"),
            patch("autocorrection_pipeline.providers.anthropic_provider.AsyncOpenAI"),
        ):
            provider: AnthropicProvider = AnthropicProvider(
                api_key="sk-ant-test-key",
                config=anthropic_config,
                embedding_fallback_key="sk-openai-fallback",
            )
        # Verify the provider was created
        assert provider is not None

    def test_init_with_empty_key_raises_error(self, anthropic_config: ProviderConfig) -> None:
        """Verify empty API key raises ProviderError."""
        # Attempt to create with empty key
        with pytest.raises(ProviderError, match="API key is required"):
            AnthropicProvider(api_key="", config=anthropic_config)

    def test_init_without_fallback_key(self, anthropic_config: ProviderConfig) -> None:
        """Verify initialization without embedding fallback key succeeds."""
        # Create provider without fallback key
        with patch("autocorrection_pipeline.providers.anthropic_provider.AsyncAnthropic"):
            provider: AnthropicProvider = AnthropicProvider(
                api_key="sk-ant-test-key",
                config=anthropic_config,
            )
        # Verify the provider was created (embedding will fail later)
        assert provider is not None


class TestAnthropicGenerateResponse:
    """Tests for Anthropic response generation."""

    @pytest.mark.asyncio
    async def test_successful_response_generation(self, anthropic_config: ProviderConfig) -> None:
        """Verify successful generation returns response content."""
        # Mock both Anthropic and OpenAI clients
        with (
            patch("autocorrection_pipeline.providers.anthropic_provider.AsyncAnthropic") as mock_anthro_cls,
            patch("autocorrection_pipeline.providers.anthropic_provider.AsyncOpenAI"),
        ):
            # Configure the mock Anthropic response
            mock_client: AsyncMock = AsyncMock()
            mock_anthro_cls.return_value = mock_client
            mock_response: MagicMock = MagicMock()
            mock_content: MagicMock = MagicMock()
            mock_content.text = "Claude test response"
            mock_response.content = [mock_content]
            mock_client.messages.create.return_value = mock_response
            # Create provider and generate response
            provider: AnthropicProvider = AnthropicProvider(
                api_key="sk-ant-test",
                config=anthropic_config,
            )
            result: str = await provider.generate_response("Test query")
        # Verify the response content
        assert result == "Claude test response"

    @pytest.mark.asyncio
    async def test_empty_content_returns_empty_string(self, anthropic_config: ProviderConfig) -> None:
        """Verify empty content list returns empty string."""
        # Mock the client with empty content
        with (
            patch("autocorrection_pipeline.providers.anthropic_provider.AsyncAnthropic") as mock_anthro_cls,
            patch("autocorrection_pipeline.providers.anthropic_provider.AsyncOpenAI"),
        ):
            mock_client: AsyncMock = AsyncMock()
            mock_anthro_cls.return_value = mock_client
            mock_response: MagicMock = MagicMock()
            mock_response.content = []
            mock_client.messages.create.return_value = mock_response
            # Create provider and generate response
            provider: AnthropicProvider = AnthropicProvider(
                api_key="sk-ant-test",
                config=anthropic_config,
            )
            result: str = await provider.generate_response("Test query")
        # Verify empty string is returned
        assert result == ""


class TestAnthropicGenerateEmbedding:
    """Tests for Anthropic embedding generation (OpenAI fallback)."""

    @pytest.mark.asyncio
    async def test_embedding_with_fallback_key(self, anthropic_config: ProviderConfig) -> None:
        """Verify embedding generation works with OpenAI fallback."""
        # Mock both clients
        with (
            patch("autocorrection_pipeline.providers.anthropic_provider.AsyncAnthropic"),
            patch("autocorrection_pipeline.providers.anthropic_provider.AsyncOpenAI") as mock_openai_cls,
        ):
            # Configure mock embedding response
            mock_openai_client: AsyncMock = AsyncMock()
            mock_openai_cls.return_value = mock_openai_client
            mock_embedding: MagicMock = MagicMock()
            mock_embedding.data = [MagicMock(embedding=[0.5, 0.6, 0.7])]
            mock_openai_client.embeddings.create.return_value = mock_embedding
            # Create provider with fallback key
            provider: AnthropicProvider = AnthropicProvider(
                api_key="sk-ant-test",
                config=anthropic_config,
                embedding_fallback_key="sk-openai-fallback",
            )
            result: list[float] = await provider.generate_embedding("Test text")
        # Verify the embedding vector from fallback
        assert result == [0.5, 0.6, 0.7]

    @pytest.mark.asyncio
    async def test_embedding_without_fallback_raises_error(self, anthropic_config: ProviderConfig) -> None:
        """Verify missing fallback key raises ProviderError for embeddings."""
        # Create provider without fallback key
        with patch("autocorrection_pipeline.providers.anthropic_provider.AsyncAnthropic"):
            provider: AnthropicProvider = AnthropicProvider(
                api_key="sk-ant-test",
                config=anthropic_config,
            )
            # Attempt to generate embedding should fail
            with pytest.raises(ProviderError, match="require an OpenAI API key"):
                await provider.generate_embedding("Test text")
