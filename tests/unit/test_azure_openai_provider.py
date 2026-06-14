"""
Unit tests for the Azure OpenAI LLM provider implementation.

Tests response generation, embedding generation, and error handling
using mocked Azure OpenAI client responses.
"""

from unittest.mock import AsyncMock, MagicMock, patch  # Mocking utilities

import pytest  # Test framework

from autocorrection_pipeline.config.settings import ProviderConfig  # Config type
from autocorrection_pipeline.providers.azure_openai_provider import (
    AzureOpenAIProvider,  # Class under test
)
from autocorrection_pipeline.providers.base import ProviderError  # Error type


@pytest.fixture
def azure_config() -> ProviderConfig:
    """Provide a test configuration for Azure OpenAI provider."""
    # Create and return test Azure configuration
    return ProviderConfig(
        model="gpt-4o",
        max_tokens=4096,
        temperature=0.0,
        embedding_model="text-embedding-3-small",
        api_version="2024-06-01",
    )


class TestAzureOpenAIProviderInit:
    """Tests for Azure OpenAI provider initialization."""

    def test_init_with_valid_credentials(self, azure_config: ProviderConfig) -> None:
        """Verify successful initialization with valid credentials."""
        # Create provider with mocked client
        with patch("autocorrection_pipeline.providers.azure_openai_provider.AsyncAzureOpenAI"):
            provider: AzureOpenAIProvider = AzureOpenAIProvider(
                api_key="test-azure-key",
                endpoint="https://test.openai.azure.com/",
                config=azure_config,
            )
        # Verify the provider was created
        assert provider is not None

    def test_init_with_empty_key_raises_error(self, azure_config: ProviderConfig) -> None:
        """Verify empty API key raises ProviderError."""
        # Attempt to create with empty key
        with pytest.raises(ProviderError, match="API key is required"):
            AzureOpenAIProvider(
                api_key="",
                endpoint="https://test.openai.azure.com/",
                config=azure_config,
            )

    def test_init_with_empty_endpoint_raises_error(self, azure_config: ProviderConfig) -> None:
        """Verify empty endpoint raises ProviderError."""
        # Attempt to create with empty endpoint
        with pytest.raises(ProviderError, match="endpoint URL is required"):
            AzureOpenAIProvider(
                api_key="test-key",
                endpoint="",
                config=azure_config,
            )


class TestAzureOpenAIGenerateResponse:
    """Tests for Azure OpenAI response generation."""

    @pytest.mark.asyncio
    async def test_successful_response_generation(self, azure_config: ProviderConfig) -> None:
        """Verify successful generation returns response content."""
        # Mock the Azure OpenAI client
        with patch("autocorrection_pipeline.providers.azure_openai_provider.AsyncAzureOpenAI") as mock_cls:
            mock_client: AsyncMock = AsyncMock()
            mock_cls.return_value = mock_client
            # Configure mock response
            mock_response: MagicMock = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Azure response"))]
            mock_client.chat.completions.create.return_value = mock_response
            # Create provider and generate response
            provider: AzureOpenAIProvider = AzureOpenAIProvider(
                api_key="test-key",
                endpoint="https://test.openai.azure.com/",
                config=azure_config,
            )
            result: str = await provider.generate_response("Test query")
        # Verify the response content
        assert result == "Azure response"

    @pytest.mark.asyncio
    async def test_api_error_raises_provider_error(self, azure_config: ProviderConfig) -> None:
        """Verify API errors are wrapped in ProviderError."""
        from openai import APIError
        # Mock the client to raise an API error
        with patch("autocorrection_pipeline.providers.azure_openai_provider.AsyncAzureOpenAI") as mock_cls:
            mock_client: AsyncMock = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = APIError(
                message="Azure test error",
                request=MagicMock(),
                body=None,
            )
            # Create provider and attempt generation
            provider: AzureOpenAIProvider = AzureOpenAIProvider(
                api_key="test-key",
                endpoint="https://test.openai.azure.com/",
                config=azure_config,
            )
            with pytest.raises(ProviderError, match="API error"):
                await provider.generate_response("Test query")


class TestAzureOpenAIGenerateEmbedding:
    """Tests for Azure OpenAI embedding generation."""

    @pytest.mark.asyncio
    async def test_successful_embedding_generation(self, azure_config: ProviderConfig) -> None:
        """Verify successful embedding returns vector list."""
        # Mock the Azure OpenAI client
        with patch("autocorrection_pipeline.providers.azure_openai_provider.AsyncAzureOpenAI") as mock_cls:
            mock_client: AsyncMock = AsyncMock()
            mock_cls.return_value = mock_client
            # Configure mock embedding response
            mock_embedding: MagicMock = MagicMock()
            mock_embedding.data = [MagicMock(embedding=[0.2, 0.4, 0.6])]
            mock_client.embeddings.create.return_value = mock_embedding
            # Create provider and generate embedding
            provider: AzureOpenAIProvider = AzureOpenAIProvider(
                api_key="test-key",
                endpoint="https://test.openai.azure.com/",
                config=azure_config,
            )
            result: list[float] = await provider.generate_embedding("Test text")
        # Verify the embedding vector
        assert result == [0.2, 0.4, 0.6]

    @pytest.mark.asyncio
    async def test_no_embedding_model_raises_error(self) -> None:
        """Verify missing embedding model raises ProviderError."""
        # Create config without embedding model
        config: ProviderConfig = ProviderConfig(
            model="gpt-4o",
            max_tokens=4096,
            temperature=0.0,
            embedding_model=None,
            api_version="2024-06-01",
        )
        # Mock the client
        with patch("autocorrection_pipeline.providers.azure_openai_provider.AsyncAzureOpenAI") as mock_cls:
            mock_cls.return_value = AsyncMock()
            provider: AzureOpenAIProvider = AzureOpenAIProvider(
                api_key="test-key",
                endpoint="https://test.openai.azure.com/",
                config=config,
            )
            with pytest.raises(ProviderError, match="No embedding model"):
                await provider.generate_embedding("Test text")
