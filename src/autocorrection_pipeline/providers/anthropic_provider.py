"""
Anthropic LLM provider implementation.

Handles all interactions with the Anthropic API for text generation.
Since Anthropic does not provide a native embedding API, embeddings
fall back to OpenAI's embedding endpoint when a fallback key is configured.

Responsibilities:
    - Generate text responses via Anthropic messages API
    - Provide embedding fallback via OpenAI embeddings API
    - Handle Anthropic-specific errors and translate to ProviderError
"""

import logging  # Standard library logging for operation tracking

from anthropic import (  # Anthropic async client and errors
    APIConnectionError,
    APIError,
    AsyncAnthropic,
    RateLimitError,
)
from openai import AsyncOpenAI  # OpenAI client for embedding fallback

from autocorrection_pipeline.config.settings import ProviderConfig  # Provider configuration type
from autocorrection_pipeline.providers.base import (  # Interface and error types
    BaseLLMProvider,
    ProviderError,
)

# Module-level logger for Anthropic provider operations
logger: logging.Logger = logging.getLogger(__name__)

# Provider identifier constant for error messages
_PROVIDER_NAME: str = "anthropic"


class AnthropicProvider(BaseLLMProvider):
    """
    Concrete LLM provider for Anthropic's API (Claude models).

    Implements the BaseLLMProvider interface using the Anthropic Python
    SDK's async client for text generation. Embeddings are handled via
    an OpenAI fallback since Anthropic lacks native embedding support.
    """

    def __init__(
        self,
        api_key: str,
        config: ProviderConfig,
        embedding_fallback_key: str = "",
    ) -> None:
        """
        Initialize the Anthropic provider with credentials and config.

        Creates the async Anthropic client and optionally sets up an
        OpenAI client for embedding fallback functionality.

        Args:
            api_key: The Anthropic API authentication key string.
            config: Provider-specific configuration containing model
                   name, max tokens, and temperature settings.
            embedding_fallback_key: Optional OpenAI API key for embedding
                                   generation (Anthropic has no native
                                   embedding endpoint).

        Raises:
            ProviderError: If the Anthropic API key is empty or None.
        """
        # Validate that the API key is not empty
        if not api_key:
            raise ProviderError(_PROVIDER_NAME, "API key is required but was empty")
        # Store the provider configuration for API calls
        self._config: ProviderConfig = config
        # Create the async Anthropic client with the provided key
        self._client: AsyncAnthropic = AsyncAnthropic(api_key=api_key)
        # Store the embedding fallback key for later use
        self._embedding_fallback_key: str = embedding_fallback_key
        # Create OpenAI client for embeddings if fallback key provided
        self._embedding_client: AsyncOpenAI | None = (
            AsyncOpenAI(api_key=embedding_fallback_key) if embedding_fallback_key else None
        )
        # Log successful initialization with model info
        logger.info("Anthropic provider initialized with model: %s", config.model)

    async def generate_response(self, query: str) -> str:
        """
        Generate a response using Anthropic's messages API.

        Sends the query as a user message to the configured Claude model
        and returns the assistant's response content.

        Args:
            query: The user's input query string.

        Returns:
            str: The generated response text from the Claude model.

        Raises:
            ProviderError: If the API call fails for any reason.
        """
        try:
            # Log the query being sent to Anthropic
            logger.debug("Sending query to Anthropic model %s", self._config.model)
            # Make the async API call to the messages endpoint
            response = await self._client.messages.create(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                messages=[{"role": "user", "content": query}],
            )
            # Extract the text content from the first content block
            content: str = response.content[0].text if response.content else ""
            # Log successful response generation
            logger.debug("Anthropic response received, length: %d chars", len(content))
            # Return the extracted response text
            return content
        except RateLimitError as rate_error:
            # Handle rate limiting specifically with clear messaging
            logger.error("Anthropic rate limit exceeded: %s", rate_error)
            raise ProviderError(
                _PROVIDER_NAME, f"Rate limit exceeded: {rate_error}"
            ) from rate_error
        except APIConnectionError as conn_error:
            # Handle network connectivity issues
            logger.error("Anthropic connection error: %s", conn_error)
            raise ProviderError(
                _PROVIDER_NAME, f"Connection failed: {conn_error}"
            ) from conn_error
        except APIError as api_error:
            # Handle general API errors from Anthropic
            logger.error("Anthropic API error: %s", api_error)
            raise ProviderError(_PROVIDER_NAME, f"API error: {api_error}") from api_error

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding using OpenAI fallback.

        Anthropic does not provide a native embedding API, so this
        method uses an OpenAI client as a fallback for embedding
        generation when cosine similarity comparison is needed.

        Args:
            text: The input text string to generate an embedding for.

        Returns:
            list[float]: The embedding vector as a list of floats.

        Raises:
            ProviderError: If no fallback key is configured or the
                          embedding API call fails.
        """
        # Check if embedding fallback client is available
        if not self._embedding_client:
            # Raise error indicating embeddings are not supported
            raise ProviderError(
                _PROVIDER_NAME,
                "Embeddings require an OpenAI API key as fallback "
                "(set OPENAI_API_KEY environment variable)",
            )
        try:
            # Determine the embedding model to use
            embedding_model: str = self._config.embedding_model or "text-embedding-3-small"
            # Log the embedding generation request
            logger.debug("Generating embedding via OpenAI fallback: %s", embedding_model)
            # Make the API call to OpenAI's embeddings endpoint
            response = await self._embedding_client.embeddings.create(
                model=embedding_model,
                input=text,
            )
            # Extract the embedding vector from the response
            embedding: list[float] = response.data[0].embedding
            # Log successful embedding generation
            logger.debug("Fallback embedding generated, dimensions: %d", len(embedding))
            # Return the embedding vector
            return embedding
        except Exception as embed_error:
            # Handle any embedding generation failures
            logger.error("Embedding fallback error: %s", embed_error)
            raise ProviderError(
                _PROVIDER_NAME, f"Embedding fallback failed: {embed_error}"
            ) from embed_error
