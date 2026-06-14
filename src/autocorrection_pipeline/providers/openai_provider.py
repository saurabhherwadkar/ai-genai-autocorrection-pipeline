"""
OpenAI LLM provider implementation.

Handles all interactions with the OpenAI API including chat completions
and text embeddings. Uses the async client for non-blocking operations.

Responsibilities:
    - Generate text responses via OpenAI chat completion API
    - Generate text embeddings via OpenAI embeddings API
    - Handle OpenAI-specific errors and translate to ProviderError
"""

import logging  # Standard library logging for operation tracking

from openai import (  # OpenAI async client and errors
    APIConnectionError,
    APIError,
    AsyncOpenAI,
    RateLimitError,
)

from autocorrection_pipeline.config.settings import ProviderConfig  # Provider configuration type
from autocorrection_pipeline.providers.base import (  # Interface and error types
    BaseLLMProvider,
    ProviderError,
)

# Module-level logger for OpenAI provider operations
logger: logging.Logger = logging.getLogger(__name__)

# Provider identifier constant for error messages
_PROVIDER_NAME: str = "openai"


class OpenAIProvider(BaseLLMProvider):
    """
    Concrete LLM provider for OpenAI's API (GPT models).

    Implements the BaseLLMProvider interface using the OpenAI Python
    SDK's async client for both chat completions and embeddings.
    """

    def __init__(self, api_key: str, config: ProviderConfig) -> None:
        """
        Initialize the OpenAI provider with credentials and configuration.

        Creates the async client instance with the provided API key
        and stores the provider configuration for use in API calls.

        Args:
            api_key: The OpenAI API authentication key string.
            config: Provider-specific configuration containing model
                   name, max tokens, temperature, and embedding model.

        Raises:
            ProviderError: If the API key is empty or None.
        """
        # Validate that the API key is not empty
        if not api_key:
            raise ProviderError(_PROVIDER_NAME, "API key is required but was empty")
        # Store the provider configuration for API calls
        self._config: ProviderConfig = config
        # Create the async OpenAI client with the provided key
        self._client: AsyncOpenAI = AsyncOpenAI(api_key=api_key)
        # Log successful initialization
        logger.info("OpenAI provider initialized with model: %s", config.model)

    async def generate_response(self, query: str) -> str:
        """
        Generate a response using OpenAI's chat completion API.

        Sends the query as a user message to the configured model
        and returns the assistant's response content.

        Args:
            query: The user's input query string.

        Returns:
            str: The generated response text from the model.

        Raises:
            ProviderError: If the API call fails for any reason.
        """
        try:
            # Log the query being sent to OpenAI
            logger.debug("Sending query to OpenAI model %s", self._config.model)
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
            logger.debug("OpenAI response received, length: %d chars", len(content))
            # Return the extracted response text
            return content
        except RateLimitError as rate_error:
            # Handle rate limiting specifically with clear error message
            logger.error("OpenAI rate limit exceeded: %s", rate_error)
            raise ProviderError(
                _PROVIDER_NAME, f"Rate limit exceeded: {rate_error}"
            ) from rate_error
        except APIConnectionError as conn_error:
            # Handle network connectivity issues
            logger.error("OpenAI connection error: %s", conn_error)
            raise ProviderError(
                _PROVIDER_NAME, f"Connection failed: {conn_error}"
            ) from conn_error
        except APIError as api_error:
            # Handle general API errors from OpenAI
            logger.error("OpenAI API error: %s", api_error)
            raise ProviderError(_PROVIDER_NAME, f"API error: {api_error}") from api_error

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate a vector embedding using OpenAI's embedding API.

        Converts the input text into a dense vector using the configured
        embedding model (e.g., text-embedding-3-small).

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
            logger.debug("Generating embedding with model: %s", self._config.embedding_model)
            # Make the async API call to the embeddings endpoint
            response = await self._client.embeddings.create(
                model=self._config.embedding_model,
                input=text,
            )
            # Extract the embedding vector from the response
            embedding: list[float] = response.data[0].embedding
            # Log successful embedding generation with vector dimension
            logger.debug("Embedding generated, dimensions: %d", len(embedding))
            # Return the embedding vector
            return embedding
        except RateLimitError as rate_error:
            # Handle rate limiting for embedding requests
            logger.error("OpenAI embedding rate limit: %s", rate_error)
            raise ProviderError(
                _PROVIDER_NAME, f"Rate limit exceeded: {rate_error}"
            ) from rate_error
        except APIConnectionError as conn_error:
            # Handle network issues for embedding requests
            logger.error("OpenAI embedding connection error: %s", conn_error)
            raise ProviderError(
                _PROVIDER_NAME, f"Connection failed: {conn_error}"
            ) from conn_error
        except APIError as api_error:
            # Handle general API errors for embedding requests
            logger.error("OpenAI embedding API error: %s", api_error)
            raise ProviderError(_PROVIDER_NAME, f"API error: {api_error}") from api_error
