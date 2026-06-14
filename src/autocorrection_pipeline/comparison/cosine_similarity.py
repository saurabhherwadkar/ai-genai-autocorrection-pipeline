"""
Cosine similarity comparison implementation.

Compares LLM responses by computing the cosine similarity between
their embedding vectors. Uses the configured LLM provider's embedding
endpoint to generate vectors for both responses.

Responsibilities:
    - Generate embeddings for actual and ideal responses
    - Compute cosine similarity between embedding vectors
    - Return a ComparisonResult with the similarity score
"""

import logging  # Standard library logging for operation tracking

import numpy as np  # NumPy for efficient vector mathematics

from autocorrection_pipeline.comparison.base import (  # Base interface and result type
    BaseComparator,
    ComparisonError,
    ComparisonResult,
)
from autocorrection_pipeline.providers.base import (  # Provider interface
    BaseLLMProvider,
    ProviderError,
)

# Module-level logger for cosine similarity operations
logger: logging.Logger = logging.getLogger(__name__)

# Comparison method identifier string
_METHOD_NAME: str = "cosine_similarity"


class CosineSimilarityComparator(BaseComparator):
    """
    Compares responses using embedding-based cosine similarity.

    Generates vector embeddings for both the actual and ideal responses,
    then computes the cosine similarity between them. A score of 1.0
    indicates identical semantic content, while 0.0 indicates no similarity.
    """

    def __init__(self, provider: BaseLLMProvider) -> None:
        """
        Initialize with an LLM provider capable of generating embeddings.

        The provider must implement the generate_embedding method to
        convert text into vector representations.

        Args:
            provider: An LLM provider instance with embedding capability.
        """
        # Store the provider reference for embedding generation
        self._provider: BaseLLMProvider = provider
        # Log initialization of the cosine similarity comparator
        logger.info("Cosine similarity comparator initialized")

    async def compare(self, actual_response: str, ideal_response: str) -> ComparisonResult:
        """
        Compare responses by computing cosine similarity of their embeddings.

        Generates embeddings for both responses and computes the cosine
        similarity between the resulting vectors.

        Args:
            actual_response: The LLM-generated response text to evaluate.
            ideal_response: The ground-truth ideal response text.

        Returns:
            ComparisonResult: Contains the cosine similarity score (0.0 to 1.0),
                            method identifier, and explanation.

        Raises:
            ComparisonError: If embedding generation or computation fails.
        """
        try:
            # Log the comparison operation beginning
            logger.debug("Computing cosine similarity between responses")
            # Generate embedding vector for the actual response
            actual_embedding: list[float] = await self._provider.generate_embedding(
                actual_response
            )
            # Generate embedding vector for the ideal response
            ideal_embedding: list[float] = await self._provider.generate_embedding(
                ideal_response
            )
            # Compute cosine similarity between the two vectors
            similarity_score: float = self._compute_cosine_similarity(
                actual_embedding, ideal_embedding
            )
            # Clamp the score to valid range [0.0, 1.0]
            clamped_score: float = max(0.0, min(1.0, similarity_score))
            # Generate human-readable explanation of the score
            explanation: str = (
                f"Cosine similarity score: {clamped_score:.4f}. "
                f"Vectors of dimension {len(actual_embedding)} compared."
            )
            # Log the computed similarity score
            logger.info("Cosine similarity computed: %.4f", clamped_score)
            # Return the comparison result with score and explanation
            return ComparisonResult(
                confidence_score=clamped_score,
                method=_METHOD_NAME,
                explanation=explanation,
            )
        except ProviderError as provider_err:
            # Handle embedding generation failures from the provider
            logger.error("Embedding generation failed: %s", provider_err)
            raise ComparisonError(
                _METHOD_NAME, f"Embedding generation failed: {provider_err}"
            ) from provider_err

    @staticmethod
    def _compute_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """
        Compute cosine similarity between two vectors using NumPy.

        Calculates the dot product divided by the product of magnitudes.
        Returns 0.0 if either vector has zero magnitude to avoid division
        by zero.

        Args:
            vec_a: First embedding vector as a list of floats.
            vec_b: Second embedding vector as a list of floats.

        Returns:
            float: Cosine similarity value between -1.0 and 1.0.
                  Returns 0.0 for zero-magnitude vectors.
        """
        # Convert input lists to NumPy arrays for vectorized math
        array_a: np.ndarray = np.array(vec_a, dtype=np.float64)
        # Convert second input list to NumPy array
        array_b: np.ndarray = np.array(vec_b, dtype=np.float64)
        # Compute the dot product of the two vectors
        dot_product: float = float(np.dot(array_a, array_b))
        # Compute the L2 norm (magnitude) of vector A
        magnitude_a: float = float(np.linalg.norm(array_a))
        # Compute the L2 norm (magnitude) of vector B
        magnitude_b: float = float(np.linalg.norm(array_b))
        # Check for zero-magnitude vectors to prevent division by zero
        if magnitude_a == 0.0 or magnitude_b == 0.0:
            # Return zero similarity for degenerate vectors
            return 0.0
        # Compute and return the cosine similarity
        return dot_product / (magnitude_a * magnitude_b)
