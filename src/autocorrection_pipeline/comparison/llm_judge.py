"""
LLM-as-judge comparison implementation.

Uses an LLM to evaluate the quality of an actual response compared
to the ideal response. The LLM acts as a judge, providing a confidence
score and explanation in structured JSON format.

Responsibilities:
    - Construct evaluation prompts for the judge LLM
    - Send evaluation requests to the LLM provider
    - Parse structured JSON responses into ComparisonResult
    - Handle malformed judge responses gracefully
"""

import json  # JSON parsing for structured LLM output
import logging  # Standard library logging for operation tracking
import re  # Regular expressions for JSON extraction

from autocorrection_pipeline.comparison.base import (  # Base interface and result type
    BaseComparator,
    ComparisonError,
    ComparisonResult,
)
from autocorrection_pipeline.providers.base import (  # Provider interface
    BaseLLMProvider,
    ProviderError,
)

# Module-level logger for LLM judge operations
logger: logging.Logger = logging.getLogger(__name__)

# Comparison method identifier string
_METHOD_NAME: str = "llm_judge"

# System prompt instructing the LLM to act as a response quality evaluator
_JUDGE_SYSTEM_PROMPT: str = (
    "You are a response quality evaluator. Compare the actual response "
    "against the ideal response and provide a confidence score between "
    "0.0 and 1.0, where 1.0 means the actual response perfectly matches "
    "the ideal in meaning, accuracy, and completeness.\n\n"
    "Return your evaluation as JSON with exactly these keys:\n"
    '- "score": a float between 0.0 and 1.0\n'
    '- "explanation": a brief string explaining why you assigned this score\n\n'
    "Return ONLY the JSON object, no additional text."
)


class LLMJudgeComparator(BaseComparator):
    """
    Compares responses using an LLM as a quality judge.

    Sends both the actual and ideal responses to an LLM with instructions
    to evaluate similarity and return a structured confidence score.
    """

    def __init__(self, provider: BaseLLMProvider) -> None:
        """
        Initialize with an LLM provider for judging responses.

        The provider's generate_response method is used to send
        evaluation prompts and receive structured judgments.

        Args:
            provider: An LLM provider instance for making judge calls.
        """
        # Store the provider reference for judge evaluation calls
        self._provider: BaseLLMProvider = provider
        # Log initialization of the LLM judge comparator
        logger.info("LLM judge comparator initialized")

    async def compare(self, actual_response: str, ideal_response: str) -> ComparisonResult:
        """
        Use an LLM to judge the quality of the actual response.

        Constructs an evaluation prompt with both responses, sends it
        to the LLM, and parses the structured JSON response into a
        ComparisonResult.

        Args:
            actual_response: The LLM-generated response text to evaluate.
            ideal_response: The ground-truth ideal response text.

        Returns:
            ComparisonResult: Contains the LLM-evaluated confidence score,
                            method identifier, and judge's explanation.

        Raises:
            ComparisonError: If the LLM call fails or response parsing fails.
        """
        try:
            # Log the judge evaluation operation beginning
            logger.debug("Sending responses to LLM judge for evaluation")
            # Build the evaluation prompt with both responses
            judge_prompt: str = self._build_judge_prompt(actual_response, ideal_response)
            # Send the evaluation prompt to the LLM provider
            raw_response: str = await self._provider.generate_response(judge_prompt)
            # Parse the structured JSON response from the judge
            score, explanation = self._parse_judge_response(raw_response)
            # Clamp the score to valid range [0.0, 1.0]
            clamped_score: float = max(0.0, min(1.0, score))
            # Log the judge's evaluation score
            logger.info("LLM judge score: %.4f", clamped_score)
            # Return the comparison result with judge's evaluation
            return ComparisonResult(
                confidence_score=clamped_score,
                method=_METHOD_NAME,
                explanation=explanation,
            )
        except ProviderError as provider_err:
            # Handle LLM provider failures during judge evaluation
            logger.error("LLM judge provider error: %s", provider_err)
            raise ComparisonError(
                _METHOD_NAME, f"Judge LLM call failed: {provider_err}"
            ) from provider_err
        except ValueError as parse_err:
            # Handle response parsing failures
            logger.error("LLM judge response parsing failed: %s", parse_err)
            raise ComparisonError(
                _METHOD_NAME, f"Failed to parse judge response: {parse_err}"
            ) from parse_err

    def _build_judge_prompt(self, actual_response: str, ideal_response: str) -> str:
        """
        Construct the evaluation prompt for the LLM judge.

        Formats both responses into a structured prompt that instructs
        the LLM to compare them and return a JSON evaluation.

        Args:
            actual_response: The generated response to be evaluated.
            ideal_response: The ground-truth ideal response.

        Returns:
            str: The complete formatted evaluation prompt string.
        """
        # Construct the prompt combining system instructions and responses
        prompt: str = (
            f"{_JUDGE_SYSTEM_PROMPT}\n\n"
            f"## Actual Response:\n{actual_response}\n\n"
            f"## Ideal Response:\n{ideal_response}\n\n"
            f"Provide your JSON evaluation:"
        )
        # Return the constructed evaluation prompt
        return prompt

    @staticmethod
    def _parse_judge_response(raw_response: str) -> tuple[float, str]:
        """
        Parse the LLM judge's JSON response into score and explanation.

        Attempts to parse the response as JSON directly, then falls back
        to regex extraction if the response contains extra text around
        the JSON object.

        Args:
            raw_response: The raw text output from the judge LLM.

        Returns:
            tuple[float, str]: A tuple of (confidence_score, explanation).

        Raises:
            ValueError: If the response cannot be parsed into valid
                       JSON with required 'score' and 'explanation' keys.
        """
        # First attempt: try direct JSON parsing of the full response
        try:
            # Parse the entire response as JSON
            parsed: dict = json.loads(raw_response.strip())
            # Extract the score value from the parsed JSON
            score: float = float(parsed["score"])
            # Extract the explanation text from the parsed JSON
            explanation: str = str(parsed["explanation"])
            # Return the extracted score and explanation
            return score, explanation
        except (json.JSONDecodeError, KeyError, TypeError):
            # Direct parsing failed, try regex extraction
            pass
        # Second attempt: extract JSON object using regex pattern
        json_pattern: re.Pattern = re.compile(r"\{[^{}]*\}", re.DOTALL)
        # Search for JSON-like content in the response
        match: re.Match | None = json_pattern.search(raw_response)
        # Check if a JSON-like pattern was found
        if match:
            try:
                # Parse the extracted JSON substring
                parsed = json.loads(match.group())
                # Extract the score value from the extracted JSON
                score = float(parsed["score"])
                # Extract the explanation from the extracted JSON
                explanation = str(parsed["explanation"])
                # Return the extracted values
                return score, explanation
            except (json.JSONDecodeError, KeyError, TypeError):
                # Regex extraction also failed
                pass
        # Both parsing attempts failed, raise a descriptive error
        raise ValueError(
            f"Could not parse judge response as JSON. "
            f"Raw response: {raw_response[:200]}"
        )
