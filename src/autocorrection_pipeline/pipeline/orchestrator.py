"""
Pipeline orchestrator module for the autocorrection pipeline.

Coordinates the full autocorrection workflow: generating LLM responses,
comparing them against ideal responses, and capturing corrections for
responses that fall below the confidence threshold.

Responsibilities:
    - Load ideal responses from the JSON reference file
    - Create and configure the LLM provider via factory
    - Create the appropriate comparison strategy
    - Process individual queries through the full pipeline
    - Run batch processing for all queries in the reference file
    - Determine when corrections should be captured
"""

import json  # JSON parsing for ideal responses file
import logging  # Standard library logging for pipeline operations
from dataclasses import dataclass  # Structured result container
from datetime import UTC, datetime  # UTC timestamp generation
from pathlib import Path  # Cross-platform filesystem path handling

from autocorrection_pipeline.comparison.base import (  # Comparison interfaces
    BaseComparator,
    ComparisonResult,
)
from autocorrection_pipeline.comparison.cosine_similarity import (
    CosineSimilarityComparator,  # Cosine strategy
)
from autocorrection_pipeline.comparison.llm_judge import LLMJudgeComparator  # LLM judge strategy
from autocorrection_pipeline.config.settings import AppSettings  # Application settings type
from autocorrection_pipeline.correction.capture import (  # Capture module
    CorrectionCapture,
    CorrectionRecord,
)
from autocorrection_pipeline.providers.base import BaseLLMProvider  # Provider interface
from autocorrection_pipeline.providers.factory import ProviderFactory  # Provider creation

# Module-level logger for pipeline operations
logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Result of processing a single query through the pipeline.

    Contains all information about the processing outcome including
    the query, responses, comparison result, and correction status.

    Attributes:
        query: The input query that was processed.
        actual_response: The LLM-generated response for the query.
        ideal_response: The ideal response from the reference file.
        comparison_result: The full comparison evaluation result.
        correction_captured: Whether a correction file was generated.
        correction_file_path: Path to the correction file if written.
    """

    # The original input query string
    query: str
    # The LLM-generated response text
    actual_response: str
    # The ideal response from the reference file
    ideal_response: str
    # The comparison result with confidence score
    comparison_result: ComparisonResult
    # Flag indicating if a correction was captured
    correction_captured: bool
    # Path to the correction file (None if not captured)
    correction_file_path: Path | None


class PipelineOrchestrator:
    """
    Orchestrates the full autocorrection pipeline workflow.

    Manages the lifecycle of processing queries through LLM generation,
    comparison against ideal responses, and correction capture for
    responses below the configured confidence threshold.
    """

    def __init__(self, settings: AppSettings) -> None:
        """
        Initialize the orchestrator with application settings.

        Stores settings for later use during async initialization
        and pipeline execution.

        Args:
            settings: The loaded application configuration containing
                     all parameters needed for pipeline operation.
        """
        # Store the application settings for component initialization
        self._settings: AppSettings = settings
        # Initialize provider reference as None (set during initialize)
        self._provider: BaseLLMProvider | None = None
        # Initialize comparator reference as None (set during initialize)
        self._comparator: BaseComparator | None = None
        # Initialize correction capture reference as None
        self._capture: CorrectionCapture | None = None
        # Initialize ideal responses dict as None (loaded during initialize)
        self._ideal_responses: dict[str, str] | None = None
        # Log orchestrator construction with key settings
        logger.info(
            "Pipeline orchestrator created (provider: %s, method: %s, threshold: %.2f)",
            settings.active_provider,
            settings.comparison_method,
            settings.confidence_threshold,
        )

    async def initialize(self) -> None:
        """
        Perform async initialization of all pipeline components.

        Must be called before process_query or run_batch. Creates the
        LLM provider, comparator, and correction capture instances.

        Raises:
            FileNotFoundError: If the ideal responses file doesn't exist.
            json.JSONDecodeError: If the ideal responses file is invalid.
            ProviderError: If provider creation fails.
        """
        # Create the LLM provider using the factory
        self._provider = ProviderFactory.create(self._settings)
        # Create the comparison strategy based on configuration
        self._comparator = self._create_comparator()
        # Create the correction capture instance with output directory
        self._capture = CorrectionCapture(self._settings.output_directory)
        # Load ideal responses from the JSON reference file
        self._ideal_responses = self._load_ideal_responses()
        # Log successful initialization with query count
        logger.info(
            "Pipeline initialized with %d ideal response(s)",
            len(self._ideal_responses),
        )

    async def process_query(self, query: str) -> PipelineResult:
        """
        Process a single query through the full autocorrection pipeline.

        Executes the complete pipeline workflow:
        1. Generate an LLM response for the query
        2. Look up the ideal response from the reference data
        3. Compare the actual response against the ideal
        4. If confidence is below threshold, capture the correction

        Args:
            query: The user's input query string to process.

        Returns:
            PipelineResult: Complete result containing all processing
                          details and correction status.

        Raises:
            RuntimeError: If initialize() has not been called first.
            KeyError: If the query is not found in ideal responses.
            ProviderError: If LLM response generation fails.
            ComparisonError: If response comparison fails.
        """
        # Verify that initialization has been performed
        self._ensure_initialized()
        # Log the query being processed
        logger.info("Processing query: %s", query[:100])
        # Generate an LLM response for the input query
        actual_response: str = await self._provider.generate_response(query)  # type: ignore[union-attr]
        # Look up the ideal response for this query
        ideal_response: str = self._get_ideal_response(query)
        # Compare the actual response against the ideal response
        comparison_result: ComparisonResult = await self._comparator.compare(  # type: ignore[union-attr]
            actual_response, ideal_response
        )
        # Determine if a correction should be captured
        should_capture: bool = self._should_capture_correction(comparison_result)
        # Initialize correction file path as None
        correction_file_path: Path | None = None
        # Capture the correction if confidence is below threshold
        if should_capture:
            # Create the correction record with all relevant data
            record: CorrectionRecord = CorrectionRecord(
                query=query,
                actual_response=actual_response,
                ideal_response=ideal_response,
                comparison_result=comparison_result,
                timestamp=datetime.now(UTC),
            )
            # Write the correction to a markdown file
            correction_file_path = await self._capture.capture(record)  # type: ignore[union-attr]
            # Log that a correction was captured
            logger.warning(
                "Correction captured for query (score: %.4f < %.2f)",
                comparison_result.confidence_score,
                self._settings.confidence_threshold,
            )
        else:
            # Log that the response passed the threshold
            logger.info(
                "Response passed threshold (score: %.4f >= %.2f)",
                comparison_result.confidence_score,
                self._settings.confidence_threshold,
            )
        # Construct and return the pipeline result
        return PipelineResult(
            query=query,
            actual_response=actual_response,
            ideal_response=ideal_response,
            comparison_result=comparison_result,
            correction_captured=should_capture,
            correction_file_path=correction_file_path,
        )

    async def run_batch(self) -> list[PipelineResult]:
        """
        Run the pipeline for all queries in the ideal responses file.

        Processes every query in the loaded ideal responses dictionary
        sequentially and collects all results.

        Returns:
            list[PipelineResult]: List of results for every processed query.

        Raises:
            RuntimeError: If initialize() has not been called first.
        """
        # Verify that initialization has been performed
        self._ensure_initialized()
        # Get all queries from the ideal responses dictionary
        queries: list[str] = list(self._ideal_responses.keys())  # type: ignore[union-attr]
        # Log the batch processing start with query count
        logger.info("Starting batch processing of %d queries", len(queries))
        # Initialize the results list
        results: list[PipelineResult] = []
        # Process each query sequentially through the pipeline
        for index, query in enumerate(queries, start=1):
            # Log progress for the current query
            logger.info("Processing query %d/%d", index, len(queries))
            # Process the query and collect the result
            result: PipelineResult = await self.process_query(query)
            # Append the result to the collection
            results.append(result)
        # Calculate and log batch summary statistics
        captured_count: int = sum(1 for r in results if r.correction_captured)
        # Log the batch completion summary
        logger.info(
            "Batch processing complete: %d/%d queries produced corrections",
            captured_count,
            len(results),
        )
        # Return all collected pipeline results
        return results

    def _load_ideal_responses(self) -> dict[str, str]:
        """
        Load query-response pairs from the ideal responses JSON file.

        Reads the JSON file and constructs a dictionary mapping query
        strings to their corresponding ideal response strings.

        Returns:
            dict[str, str]: Dictionary mapping queries to ideal responses.

        Raises:
            FileNotFoundError: If the ideal responses file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
            ValueError: If the JSON structure is unexpected.
        """
        # Get the path to the ideal responses file
        file_path: Path = self._settings.ideal_responses_path
        # Verify the file exists before attempting to read
        if not file_path.exists():
            raise FileNotFoundError(f"Ideal responses file not found: {file_path}")
        # Read and parse the JSON file content
        raw_content: str = file_path.read_text(encoding="utf-8")
        # Parse the JSON string into a Python dictionary
        data: dict = json.loads(raw_content)
        # Extract the queries list from the JSON structure
        queries_list: list[dict] = data.get("queries", [])
        # Build the query-to-ideal-response mapping dictionary
        ideal_responses: dict[str, str] = {}
        # Iterate over each query entry in the JSON
        for entry in queries_list:
            # Extract the query string from the entry
            query: str = entry.get("query", "")
            # Extract the ideal response string from the entry
            ideal: str = entry.get("ideal_response", "")
            # Add the mapping only if both values are present
            if query and ideal:
                ideal_responses[query] = ideal
        # Log the number of ideal responses loaded
        logger.debug("Loaded %d ideal response(s) from: %s", len(ideal_responses), file_path)
        # Return the constructed mapping dictionary
        return ideal_responses

    def _create_comparator(self) -> BaseComparator:
        """
        Create the appropriate comparator based on settings configuration.

        Reads the comparison_method setting and instantiates the
        corresponding comparator strategy with the LLM provider.

        Returns:
            BaseComparator: An initialized comparator instance matching
                          the configured comparison method.

        Raises:
            ValueError: If the comparison method is not recognized.
        """
        # Get the configured comparison method name
        method: str = self._settings.comparison_method
        # Create the appropriate comparator based on method
        if method == "cosine_similarity":
            # Create cosine similarity comparator with provider
            return CosineSimilarityComparator(self._provider)  # type: ignore[arg-type]
        elif method == "llm_judge":
            # Create LLM judge comparator with provider
            return LLMJudgeComparator(self._provider)  # type: ignore[arg-type]
        else:
            # Raise error for unrecognized comparison methods
            raise ValueError(f"Unknown comparison method: {method}")

    def _should_capture_correction(self, comparison_result: ComparisonResult) -> bool:
        """
        Determine if a correction should be captured based on the score.

        Compares the confidence score against the configured threshold.
        Returns True if the score is strictly below the threshold.

        Args:
            comparison_result: The comparison result to evaluate.

        Returns:
            bool: True if confidence is below threshold (needs correction).
        """
        # Compare the confidence score against the threshold
        return comparison_result.confidence_score < self._settings.confidence_threshold

    def _get_ideal_response(self, query: str) -> str:
        """
        Look up the ideal response for a given query.

        Searches the loaded ideal responses dictionary for the query
        and returns the corresponding ideal response.

        Args:
            query: The query string to look up.

        Returns:
            str: The ideal response text for the given query.

        Raises:
            KeyError: If the query is not found in the ideal responses.
        """
        # Attempt to find the query in ideal responses
        if query not in self._ideal_responses:  # type: ignore[operator]
            raise KeyError(f"Query not found in ideal responses: {query[:100]}")
        # Return the ideal response for the query
        return self._ideal_responses[query]  # type: ignore[index]

    def _ensure_initialized(self) -> None:
        """
        Verify that the pipeline has been initialized.

        Checks that all required components have been created by
        the initialize() method before allowing pipeline operations.

        Raises:
            RuntimeError: If initialize() has not been called.
        """
        # Check if the provider has been created
        if self._provider is None:
            raise RuntimeError(
                "Pipeline not initialized. Call initialize() before processing queries."
            )
