"""
Azure OpenAI LLM provider implementation.

Handles all interactions with Azure-hosted OpenAI models including
chat completions and text embeddings. Uses the Azure-specific
async client configuration.

Responsibilities:
    - Generate text responses via Azure OpenAI chat completion API
    - Generate text embeddings via Azure OpenAI embeddings API
    - Handle Azure-specific authentication and endpoint configuration
"""

import logging  # Standard library logging for operation tracking

from openai import (  # Azure OpenAI async client and errors
    APIConnectionError,
    APIError,
    AsyncAzureOpenAI,
    RateLimitError,
)

from autocorrection_pipeline.config.settings import ProviderConfig  # Provider configuration type
from autocorrection_pipeline.providers.base import (  # Interface and error types
    BaseLLMProvider,
    ProviderError,
)

# Module-level logger for Azure OpenAI provider operations
logger: logging.Logger = logging.getLogger(__name__)

# Provider identifier constant for error messages
_PROVIDER_NAME: str = "azure_openai"


class AzureOpenAIProvider(BaseLLMProvider):
    """
    Concrete LLM provider for Azure OpenAI Service.

    Implements the BaseLLMProvider interface using the Azure-specific
    OpenAI client configuration with endpoint and API version support.
    """

    def __init__(self, api_key: str, endpoint: str, config: ProviderConfig) -> None:
        """
        Initialize the Azure OpenAI provider with credentials and config.

        Creates the async Azure OpenAI client with the provided API key,
        endpoint URL, and API version from the configuration.

        Args:
            api_key: The Azure OpenAI API authentication key string.
            endpoint: The Azure OpenAI service endpoint URL
                     (e.g., https://your-resource.openai.azure.com/).
            config: Provider-specific configuration containing model name,
                   max tokens, temperature, and API version.

        Raises:
            ProviderError: If the API key or endpoint is empty.
        """
        # Validate that the API key is not empty
        if not api_key:
            raise ProviderError(_PROVIDER_NAME, "API key is required but was empty")
        # Validate that the endpoint URL is not empty
        if not endpoint:
            raise ProviderError(_PROVIDER_NAME, "Azure endpoint URL is required but was empty")
        # Store the provider configuration for API calls
        self._config: ProviderConfig = config
        # Create the async Azure OpenAI client with credentials
        self._client: AsyncAzureOpenAI = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=config.api_version or "2024-06-01",
        )
        # Log successful initialization with endpoint info
        logger.info(
            "Azure OpenAI provider initialized with model: %s, endpoint: %s",
            config.model,
            endpoint,
        )

    async def generate_response(self, query: str) -> str:
        """
        Generate a response using Azure OpenAI's chat completion API.

        Sends the query as a user message to the configured deployment
        model and returns the assistant's response content.

        Args:
            query: The user's input query string.

        Returns:
            str: The generated response text from the Azure model.

        Raises:
            ProviderError: If the API call fails for any reason.
        """
        try:
            # Log the query being sent to Azure OpenAI
            logger.debug("Sending query to Azure OpenAI model %s", self._config.model)
            # Make the async API call to chat completions endpoint
            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=[{"role": "user", "content": query}],
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )
            # Extract the response content from the first choice
            content: str = response.choices[0].message.content or ""
            # Log successful response generation
            logger.debug("Azure OpenAI response received, length: %d chars", len(content))
            # Return the extracted response text
            return content
        except RateLimitError as rate_error:
            # Handle rate limiting from Azure service
            logger.error("Azure OpenAI rate limit exceeded: %s", rate_error)
            raise ProviderError(
                _PROVIDER_NAME, f"Rate limit exceeded: {rate_error}"
            ) from rate_error
        except APIConnectionError as conn_error:
            # Handle network connectivity issues to Azure
            logger.error("Azure OpenAI connection error: %s", conn_error)
            raise ProviderError(
                _PROVIDER_NAME, f"Connection failed: {conn_error}"
            ) from conn_error
        except APIError as api_error:
            # Handle general API errors from Azure OpenAI
            logger.error("Azure OpenAI API error: %s", api_error)
            raise ProviderError(_PROVIDER_NAME, f"API error: {api_error}") from api_error

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate a vector embedding using Azure OpenAI's embedding API.

        Converts the input text into a dense vector using the configured
        embedding deployment model.

        Args:
            text: The input text string to generate an embedding for.

        Returns:
            list[float]: The embedding vector as a list of floats.

        Raises:
            ProviderError: If the embedding API call fails or no
                          embedding model is configured.
        """
        try:
            # Verify that an embedding model is configured
            if not self._config.embedding_model:
                raise ProviderError(_PROVIDER_NAME, "No embedding model configured")
            # Log the embedding generation request
            logger.debug("Generating Azure embedding with model: %s", self._config.embedding_model)
            # Make the async API call to the embeddings endpoint
            response = await self._client.embeddings.create(
                model=self._config.embedding_model,
                input=text,
            )
            # Extract the embedding vector from the response
            embedding: list[float] = response.data[0].embedding
            # Log successful embedding generation with dimensions
            logger.debug("Azure embedding generated, dimensions: %d", len(embedding))
            # Return the embedding vector
            return embedding
        except RateLimitError as rate_error:
            # Handle rate limiting for embedding requests
            logger.error("Azure OpenAI embedding rate limit: %s", rate_error)
            raise ProviderError(
                _PROVIDER_NAME, f"Rate limit exceeded: {rate_error}"
            ) from rate_error
        except APIConnectionError as conn_error:
            # Handle network issues for embedding requests
            logger.error("Azure OpenAI embedding connection error: %s", conn_error)
            raise ProviderError(
                _PROVIDER_NAME, f"Connection failed: {conn_error}"
            ) from conn_error
        except APIError as api_error:
            # Handle general API errors for embedding requests
            logger.error("Azure OpenAI embedding API error: %s", api_error)
            raise ProviderError(_PROVIDER_NAME, f"API error: {api_error}") from api_error
