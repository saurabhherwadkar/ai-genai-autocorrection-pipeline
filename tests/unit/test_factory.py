"""
Unit tests for the provider factory module.

Tests provider creation, registry lookup, and dynamic registration
of new provider classes.
"""

from unittest.mock import AsyncMock, patch  # Mocking utilities

import pytest  # Test framework

from autocorrection_pipeline.config.settings import AppSettings  # Config types
from autocorrection_pipeline.providers.base import (  # Interface and error
    BaseLLMProvider,
    ProviderError,
)
from autocorrection_pipeline.providers.factory import ProviderFactory  # Class under test


class TestProviderFactoryCreate:
    """Tests for the factory create method."""

    def test_create_openai_provider(self, sample_settings: AppSettings) -> None:
        """Verify factory creates OpenAI provider correctly."""
        # Mock the OpenAI client to prevent actual API connection
        with patch("autocorrection_pipeline.providers.openai_provider.AsyncOpenAI"):
            # Create the provider via factory
            provider: BaseLLMProvider = ProviderFactory.create(sample_settings)
        # Verify an OpenAI provider was created
        assert provider is not None
        from autocorrection_pipeline.providers.openai_provider import OpenAIProvider
        assert isinstance(provider, OpenAIProvider)

    def test_create_anthropic_provider(self, sample_settings: AppSettings) -> None:
        """Verify factory creates Anthropic provider correctly."""
        # Modify settings to use Anthropic
        anthro_settings: AppSettings = AppSettings(
            active_provider="anthropic",
            confidence_threshold=sample_settings.confidence_threshold,
            comparison_method=sample_settings.comparison_method,
            providers=sample_settings.providers,
            ideal_responses_path=sample_settings.ideal_responses_path,
            output_directory=sample_settings.output_directory,
            log_level=sample_settings.log_level,
            openai_api_key=sample_settings.openai_api_key,
            anthropic_api_key=sample_settings.anthropic_api_key,
            azure_openai_api_key=sample_settings.azure_openai_api_key,
            azure_openai_endpoint=sample_settings.azure_openai_endpoint,
        )
        # Mock the Anthropic and OpenAI clients
        with (
            patch("autocorrection_pipeline.providers.anthropic_provider.AsyncAnthropic"),
            patch("autocorrection_pipeline.providers.anthropic_provider.AsyncOpenAI"),
        ):
            provider: BaseLLMProvider = ProviderFactory.create(anthro_settings)
        # Verify an Anthropic provider was created
        from autocorrection_pipeline.providers.anthropic_provider import AnthropicProvider
        assert isinstance(provider, AnthropicProvider)

    def test_create_azure_openai_provider(self, sample_settings: AppSettings) -> None:
        """Verify factory creates Azure OpenAI provider correctly."""
        # Create settings with Azure provider active
        azure_settings: AppSettings = AppSettings(
            active_provider="azure_openai",
            confidence_threshold=sample_settings.confidence_threshold,
            comparison_method=sample_settings.comparison_method,
            providers=sample_settings.providers,
            ideal_responses_path=sample_settings.ideal_responses_path,
            output_directory=sample_settings.output_directory,
            log_level=sample_settings.log_level,
            openai_api_key=sample_settings.openai_api_key,
            anthropic_api_key=sample_settings.anthropic_api_key,
            azure_openai_api_key=sample_settings.azure_openai_api_key,
            azure_openai_endpoint=sample_settings.azure_openai_endpoint,
        )
        # Mock the Azure OpenAI client
        with patch("autocorrection_pipeline.providers.azure_openai_provider.AsyncAzureOpenAI"):
            provider: BaseLLMProvider = ProviderFactory.create(azure_settings)
        # Verify an Azure provider was created
        from autocorrection_pipeline.providers.azure_openai_provider import AzureOpenAIProvider
        assert isinstance(provider, AzureOpenAIProvider)

    def test_unknown_provider_raises_error(self, sample_settings: AppSettings) -> None:
        """Verify unknown provider name raises ProviderError."""
        # Create settings with an invalid provider name
        invalid_settings: AppSettings = AppSettings(
            active_provider="unknown_provider",
            confidence_threshold=sample_settings.confidence_threshold,
            comparison_method=sample_settings.comparison_method,
            providers=sample_settings.providers,
            ideal_responses_path=sample_settings.ideal_responses_path,
            output_directory=sample_settings.output_directory,
            log_level=sample_settings.log_level,
        )
        # Attempt to create should raise ProviderError
        with pytest.raises(ProviderError, match="Unknown provider"):
            ProviderFactory.create(invalid_settings)


class TestProviderFactoryRegister:
    """Tests for the factory register method."""

    def test_register_new_provider(self) -> None:
        """Verify registering a new provider adds it to the registry."""
        # Create a mock provider class
        class MockProvider(BaseLLMProvider):
            async def generate_response(self, query: str) -> str:
                return "mock"
            async def generate_embedding(self, text: str) -> list[float]:
                return [0.0]
        # Register the mock provider
        ProviderFactory.register_provider("mock_provider", MockProvider)
        # Verify it's in the registry
        assert "mock_provider" in ProviderFactory._registry
        # Clean up: remove the test registration
        del ProviderFactory._registry["mock_provider"]

    def test_register_empty_name_raises_error(self) -> None:
        """Verify empty provider name raises ValueError."""
        # Attempt to register with empty name
        with pytest.raises(ValueError, match="cannot be empty"):
            ProviderFactory.register_provider("", AsyncMock)  # type: ignore[arg-type]

    def test_register_none_class_raises_error(self) -> None:
        """Verify None class raises ValueError."""
        # Attempt to register with None class
        with pytest.raises(ValueError, match="cannot be None"):
            ProviderFactory.register_provider("test", None)  # type: ignore[arg-type]
