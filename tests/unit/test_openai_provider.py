"""
Unit tests for the OpenAI LLM provider implementation.

Tests response generation, embedding generation, and error handling
using mocked OpenAI client responses.
"""

from unittest.mock import AsyncMock, MagicMock, patch  # Mocking utilities

import pytest  # Test framework

from autocorrection_pipeline.config.settings import ProviderConfig  # Config type
from autocorrection_pipeline.providers.base import ProviderError  # Error type
from autocorrection_pipeline.providers.openai_provider import OpenAIProvider  # Class under test


@pytest.fixture
def openai_config() -> ProviderConfig:
    """Provide a test configuration for OpenAI provider."""
    # Create and return test OpenAI configuration
    return ProviderConfig(
        model="gpt-4o",
        max_tokens=4096,
        temperature=0.0,
        embedding_model="text-embedding-3-small",
    )


class TestOpenAIProviderInit:
    """Tests for OpenAI provider initialization."""

    def test_init_with_valid_key(self, openai_config: ProviderConfig) -> None:
        """Verify successful initialization with valid API key."""
        # Create provider with valid key (client creation is mocked implicitly)
        with patch("autocorrection_pipeline.providers.openai_provider.AsyncOpenAI"):
            provider: OpenAIProvider = OpenAIProvider(
                api_key="sk-test-key",
                config=openai_config,
            )
        # Verify the provider was created
        assert provider is not None

    def test_init_with_empty_key_raises_error(self, openai_config: ProviderConfig) -> None:
        """Verify empty API key raises ProviderError."""
        # Attempt to create with empty key
        with pytest.raises(ProviderError, match="API key is required"):
            OpenAIProvider(api_key="", config=openai_config)


class TestOpenAIGenerateResponse:
    """Tests for OpenAI response generation."""

    @pytest.mark.asyncio
    async def test_successful_response_generation(self, openai_config: ProviderConfig) -> None:
        """Verify successful generation returns response content."""
        # Mock the OpenAI client
        with patch("autocorrection_pipeline.providers.openai_provider.AsyncOpenAI") as mock_client_cls:
            # Configure the mock response
            mock_client: AsyncMock = AsyncMock()
            mock_client_cls.return_value = mock_client
            mock_response: MagicMock = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
            mock_client.chat.completions.create.return_value = mock_response
            # Create provider and generate response
            provider: OpenAIProvider = OpenAIProvider(api_key="sk-test", config=openai_config)
            result: str = await provider.generate_response("Test query")
        # Verify the response content
        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_empty_response_returns_empty_string(self, openai_config: ProviderConfig) -> None:
        """Verify None content is returned as empty string."""
        # Mock the client with None content
        with patch("autocorrection_pipeline.providers.openai_provider.AsyncOpenAI") as mock_client_cls:
            mock_client: AsyncMock = AsyncMock()
            mock_client_cls.return_value = mock_client
            mock_response: MagicMock = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content=None))]
            mock_client.chat.completions.create.return_value = mock_response
            # Create provider and generate response
            provider: OpenAIProvider = OpenAIProvider(api_key="sk-test", config=openai_config)
            result: str = await provider.generate_response("Test query")
        # Verify empty string is returned
        assert result == ""

    @pytest.mark.asyncio
    async def test_api_error_raises_provider_error(self, openai_config: ProviderConfig) -> None:
        """Verify API errors are wrapped in ProviderError."""
        from openai import APIError
        # Mock the client to raise an API error
        with patch("autocorrection_pipeline.providers.openai_provider.AsyncOpenAI") as mock_client_cls:
            mock_client: AsyncMock = AsyncMock()
            mock_client_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = APIError(
                message="Test error",
                request=MagicMock(),
                body=None,
            )
            # Create provider and attempt generation
            provider: OpenAIProvider = OpenAIProvider(api_key="sk-test", config=openai_config)
            with pytest.raises(ProviderError, match="API error"):
                await provider.generate_response("Test query")


class TestOpenAIGenerateEmbedding:
    """Tests for OpenAI embedding generation."""

    @pytest.mark.asyncio
    async def test_successful_embedding_generation(self, openai_config: ProviderConfig) -> None:
        """Verify successful embedding returns vector list."""
        # Mock the OpenAI client
        with patch("autocorrection_pipeline.providers.openai_provider.AsyncOpenAI") as mock_client_cls:
            mock_client: AsyncMock = AsyncMock()
            mock_client_cls.return_value = mock_client
            # Configure mock embedding response
            mock_embedding: MagicMock = MagicMock()
            mock_embedding.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
            mock_client.embeddings.create.return_value = mock_embedding
            # Create provider and generate embedding
            provider: OpenAIProvider = OpenAIProvider(api_key="sk-test", config=openai_config)
            result: list[float] = await provider.generate_embedding("Test text")
        # Verify the embedding vector
        assert result == [0.1, 0.2, 0.3]
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_no_embedding_model_raises_error(self) -> None:
        """Verify missing embedding model raises ProviderError."""
        # Create config without embedding model
        config: ProviderConfig = ProviderConfig(
            model="gpt-4o",
            max_tokens=4096,
            temperature=0.0,
            embedding_model=None,
        )
        # Mock the client
        with patch("autocorrection_pipeline.providers.openai_provider.AsyncOpenAI") as mock_client_cls:
            mock_client_cls.return_value = AsyncMock()
            provider: OpenAIProvider = OpenAIProvider(api_key="sk-test", config=config)
            with pytest.raises(ProviderError, match="No embedding model"):
                await provider.generate_embedding("Test text")
