"""
Abstract base class for LLM providers.

Defines the interface contract that all LLM provider implementations
must fulfill. This enables the factory pattern and strategy pattern
for swapping providers without changing consuming code.

Responsibilities:
    - Define the generate_response method signature
    - Define the generate_embedding method signature
    - Provide a common exception hierarchy for provider errors
"""

from abc import ABC, abstractmethod  # Abstract base class mechanism


class ProviderError(Exception):
    """
    Base exception for all LLM provider errors.

    Raised when an LLM provider encounters an error during
    API communication, authentication, or response processing.

    Attributes:
        provider_name: The name of the provider that raised the error.
        message: Human-readable description of the error.
    """

    def __init__(self, provider_name: str, message: str) -> None:
        """
        Initialize the provider error with context information.

        Args:
            provider_name: Identifier of the provider that failed.
            message: Descriptive error message explaining the failure.
        """
        # Store the provider name for programmatic access
        self.provider_name: str = provider_name
        # Construct the full error message with provider context
        super().__init__(f"[{provider_name}] {message}")


class BaseLLMProvider(ABC):
    """
    Abstract interface for language model providers.

    All concrete LLM provider implementations must inherit from this
    class and implement both the generate_response and generate_embedding
    methods. This ensures a consistent interface across providers.
    """

    @abstractmethod
    async def generate_response(self, query: str) -> str:
        """
        Generate an LLM response for the given input query.

        Sends the query to the configured language model and returns
        the generated text response.

        Args:
            query: The user's input query string to send to the LLM.

        Returns:
            str: The LLM-generated response text content.

        Raises:
            ProviderError: If the API call fails due to authentication,
                          rate limiting, network issues, or other errors.
        """
        ...

    @abstractmethod
    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate a vector embedding for the given text input.

        Converts the input text into a dense vector representation
        suitable for similarity computations.

        Args:
            text: The input text string to embed into a vector.

        Returns:
            list[float]: A list of floating point numbers representing
                        the embedding vector for the input text.

        Raises:
            ProviderError: If the embedding API call fails or the
                          provider does not support embeddings natively.
        """
        ...
