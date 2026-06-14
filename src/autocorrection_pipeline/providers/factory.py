"""
Factory module for creating LLM provider instances.

Implements the factory pattern for instantiating the correct LLM
provider based on the application configuration. Supports dynamic
registration of new providers without modifying existing code.

Responsibilities:
    - Map provider name strings to concrete provider classes
    - Instantiate providers with correct credentials and configuration
    - Support dynamic registration of additional providers
"""

import logging  # Standard library logging for factory operations

from autocorrection_pipeline.config.settings import AppSettings  # Application settings type
from autocorrection_pipeline.providers.anthropic_provider import (
    AnthropicProvider,  # Anthropic concrete class
)
from autocorrection_pipeline.providers.azure_openai_provider import (
    AzureOpenAIProvider,  # Azure concrete class
)
from autocorrection_pipeline.providers.base import (  # Interface and error types
    BaseLLMProvider,
    ProviderError,
)
from autocorrection_pipeline.providers.openai_provider import (
    OpenAIProvider,  # OpenAI concrete class
)

# Module-level logger for factory operations
logger: logging.Logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory class that creates the appropriate LLM provider instance.

    Uses a registry pattern to map provider name strings to their
    concrete implementation classes. The factory creates providers
    with the correct credentials and configuration from AppSettings.
    """

    # Registry mapping provider name strings to their concrete classes
    _registry: dict[str, type[BaseLLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "azure_openai": AzureOpenAIProvider,
    }

    @classmethod
    def create(cls, settings: AppSettings) -> BaseLLMProvider:
        """
        Create and return an LLM provider based on active configuration.

        Reads the active_provider setting to determine which provider
        class to instantiate, then passes the appropriate credentials
        and configuration to the constructor.

        Args:
            settings: The application settings containing the active
                     provider name, credentials, and provider configs.

        Returns:
            BaseLLMProvider: An initialized instance of the configured
                           LLM provider ready for use.

        Raises:
            ProviderError: If the configured provider name is not found
                         in the registry or instantiation fails.
        """
        # Get the active provider name from settings
        provider_name: str = settings.active_provider
        # Log the provider being created
        logger.info("Creating LLM provider: %s", provider_name)
        # Look up the provider class in the registry
        provider_class: type[BaseLLMProvider] | None = cls._registry.get(provider_name)
        # Validate that the provider name maps to a registered class
        if provider_class is None:
            raise ProviderError(
                provider_name,
                f"Unknown provider '{provider_name}'. "
                f"Registered providers: {list(cls._registry.keys())}",
            )
        # Get the provider-specific configuration
        provider_config = settings.providers.get(provider_name)
        # Validate that provider configuration exists
        if provider_config is None:
            raise ProviderError(
                provider_name,
                f"No configuration found for provider '{provider_name}'",
            )
        # Create the appropriate provider based on the name
        if provider_name == "openai":
            # Create OpenAI provider with API key and config
            return OpenAIProvider(
                api_key=settings.openai_api_key,
                config=provider_config,
            )
        elif provider_name == "anthropic":
            # Create Anthropic provider with API key, config, and fallback
            return AnthropicProvider(
                api_key=settings.anthropic_api_key,
                config=provider_config,
                embedding_fallback_key=settings.openai_api_key,
            )
        elif provider_name == "azure_openai":
            # Create Azure OpenAI provider with key, endpoint, and config
            return AzureOpenAIProvider(
                api_key=settings.azure_openai_api_key,
                endpoint=settings.azure_openai_endpoint,
                config=provider_config,
            )
        else:
            # This branch handles dynamically registered providers
            raise ProviderError(
                provider_name,
                f"Provider '{provider_name}' is registered but has no "
                f"instantiation logic in the factory",
            )

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseLLMProvider]) -> None:
        """
        Register a new provider class in the factory registry.

        Allows extending the factory with additional providers without
        modifying existing code. Registered providers must implement
        the BaseLLMProvider interface.

        Args:
            name: The string identifier for the new provider (used
                 in settings.yaml as the active_provider value).
            provider_class: The concrete class implementing the
                          BaseLLMProvider abstract interface.

        Raises:
            ValueError: If name is empty or provider_class is None.
        """
        # Validate the provider name is not empty
        if not name:
            raise ValueError("Provider name cannot be empty")
        # Validate the provider class is provided
        if provider_class is None:
            raise ValueError("Provider class cannot be None")
        # Add the provider to the registry
        cls._registry[name] = provider_class
        # Log the successful registration
        logger.info("Registered new provider: %s -> %s", name, provider_class.__name__)
