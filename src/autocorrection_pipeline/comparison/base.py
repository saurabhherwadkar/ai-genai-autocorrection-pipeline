"""
Abstract base class for response comparison strategies.

Defines the interface contract and result type for all comparison
method implementations. Enables the strategy pattern for swapping
comparison algorithms via configuration.

Responsibilities:
    - Define the ComparisonResult data structure
    - Define the compare method signature
    - Provide a common exception type for comparison errors
"""

from abc import ABC, abstractmethod  # Abstract base class mechanism
from dataclasses import dataclass  # Structured immutable data container


class ComparisonError(Exception):
    """
    Exception raised when a response comparison operation fails.

    Wraps underlying errors from embedding generation or LLM
    evaluation into a consistent exception type for the pipeline.

    Attributes:
        method: The comparison method that encountered the error.
        message: Human-readable description of the failure.
    """

    def __init__(self, method: str, message: str) -> None:
        """
        Initialize the comparison error with context information.

        Args:
            method: Name of the comparison method that failed.
            message: Descriptive error message explaining the failure.
        """
        # Store the comparison method name for programmatic access
        self.method: str = method
        # Construct the full error message with method context
        super().__init__(f"[{method}] {message}")


@dataclass(frozen=True)
class ComparisonResult:
    """
    Immutable result of comparing an actual response to an ideal response.

    Contains the confidence score, the method used, and a human-readable
    explanation of why the score was assigned.

    Attributes:
        confidence_score: Float between 0.0 and 1.0 where 1.0 means
                         the actual response perfectly matches the ideal.
        method: String identifier of the comparison method used.
        explanation: Human-readable explanation of the confidence score.
    """

    # Confidence score between 0.0 (no match) and 1.0 (perfect match)
    confidence_score: float
    # Identifier of the comparison method that produced this result
    method: str
    # Human-readable explanation of the assigned confidence score
    explanation: str


class BaseComparator(ABC):
    """
    Abstract interface for response comparison strategies.

    All concrete comparison implementations must inherit from this
    class and implement the compare method. This enables the pipeline
    to swap comparison strategies via configuration.
    """

    @abstractmethod
    async def compare(self, actual_response: str, ideal_response: str) -> ComparisonResult:
        """
        Compare an actual LLM response against the ideal response.

        Evaluates how closely the actual response matches the ideal
        response and returns a confidence score with explanation.

        Args:
            actual_response: The LLM-generated response text to evaluate.
            ideal_response: The ground-truth ideal response to compare against.

        Returns:
            ComparisonResult: Contains the confidence score (0.0 to 1.0),
                            the method identifier, and an explanation string.

        Raises:
            ComparisonError: If the comparison process encounters an
                           unrecoverable error during evaluation.
        """
        ...
