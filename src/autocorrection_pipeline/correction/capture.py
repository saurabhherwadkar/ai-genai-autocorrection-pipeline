"""
Correction capture module for the autocorrection pipeline.

Writes low-confidence query/response pairs to structured markdown
files that can be used for future model training and correction.
Each correction is saved as an individual markdown file with metadata.

Responsibilities:
    - Generate unique, descriptive filenames for correction records
    - Format correction data as structured markdown content
    - Write correction files to the configured output directory
    - Ensure the output directory exists before writing
"""

import hashlib  # SHA-256 hashing for unique filename generation
import logging  # Standard library logging for operation tracking
from dataclasses import dataclass  # Structured immutable data container
from datetime import datetime  # UTC timestamp generation
from pathlib import Path  # Cross-platform filesystem path handling

from autocorrection_pipeline.comparison.base import ComparisonResult  # Comparison result type

# Module-level logger for correction capture operations
logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorrectionRecord:
    """
    Immutable record of a correction that needs to be captured.

    Contains all information about a low-confidence response including
    the original query, actual response, ideal response, comparison
    details, and the timestamp of identification.

    Attributes:
        query: The original user query that was processed.
        actual_response: The LLM-generated response that scored low.
        ideal_response: The ground-truth ideal response for comparison.
        comparison_result: The full comparison result with score details.
        timestamp: UTC datetime when the correction was identified.
    """

    # The original user query string that was processed
    query: str
    # The LLM-generated response that fell below confidence threshold
    actual_response: str
    # The ground-truth ideal response from the reference file
    ideal_response: str
    # The full comparison result containing score and explanation
    comparison_result: ComparisonResult
    # UTC timestamp when this correction was identified
    timestamp: datetime


class CorrectionCapture:
    """
    Handles writing correction records to structured markdown files.

    Creates individual markdown files for each low-confidence response,
    formatted for easy review and potential use in model training.
    """

    def __init__(self, output_directory: Path) -> None:
        """
        Initialize the capture module with the output directory path.

        Creates the output directory if it does not already exist to
        ensure write operations will succeed.

        Args:
            output_directory: Filesystem path to the directory where
                            correction markdown files will be written.
        """
        # Store the output directory path for file writing
        self._output_directory: Path = output_directory
        # Ensure the output directory exists, creating if necessary
        self._output_directory.mkdir(parents=True, exist_ok=True)
        # Log initialization with the output directory path
        logger.info("Correction capture initialized, output: %s", output_directory)

    async def capture(self, record: CorrectionRecord) -> Path:
        """
        Write a correction record to a structured markdown file.

        Generates a unique filename, formats the record as markdown,
        and writes it to the output directory.

        Args:
            record: The CorrectionRecord containing all correction data
                   to be persisted to a markdown file.

        Returns:
            Path: The absolute filesystem path to the generated markdown file.

        Raises:
            IOError: If the file cannot be written to the output directory
                    due to permissions or disk space issues.
        """
        # Generate a unique filename for this correction record
        filename: str = self._generate_filename(record)
        # Construct the full file path in the output directory
        file_path: Path = self._output_directory / filename
        # Format the correction record as structured markdown
        markdown_content: str = self._format_markdown(record)
        # Write the markdown content to the file
        file_path.write_text(markdown_content, encoding="utf-8")
        # Log the successful file creation
        logger.info(
            "Correction captured: %s (score: %.4f)",
            filename,
            record.comparison_result.confidence_score,
        )
        # Return the path to the created file
        return file_path

    def _generate_filename(self, record: CorrectionRecord) -> str:
        """
        Generate a unique, descriptive filename for the correction.

        Combines a timestamp with a short hash of the query content
        to create a filename that is both unique and sortable by time.

        Args:
            record: The correction record to generate a filename for.

        Returns:
            str: A unique filename string in the format:
                 correction_YYYY-MM-DDTHH-MM-SS_<hash>.md
        """
        # Format the timestamp as a filesystem-safe string
        timestamp_str: str = record.timestamp.strftime("%Y-%m-%dT%H-%M-%S")
        # Generate a short hash from the query for uniqueness
        query_hash: str = hashlib.sha256(
            record.query.encode("utf-8")
        ).hexdigest()[:8]
        # Construct the filename combining timestamp and hash
        filename: str = f"correction_{timestamp_str}_{query_hash}.md"
        # Return the generated filename
        return filename

    def _format_markdown(self, record: CorrectionRecord) -> str:
        """
        Format the correction record as structured markdown content.

        Creates a well-organized markdown document with metadata,
        query, actual response, and ideal response sections.

        Args:
            record: The correction record to format as markdown.

        Returns:
            str: The complete markdown-formatted string content
                ready to be written to a file.
        """
        # Format the timestamp as ISO 8601 string
        timestamp_iso: str = record.timestamp.isoformat()
        # Get the confidence score for display
        score: float = record.comparison_result.confidence_score
        # Get the comparison method name
        method: str = record.comparison_result.method
        # Get the explanation from the comparison result
        explanation: str = record.comparison_result.explanation
        # Construct the structured markdown document
        markdown: str = (
            f"# Correction Record\n"
            f"\n"
            f"## Metadata\n"
            f"- **Timestamp**: {timestamp_iso}\n"
            f"- **Confidence Score**: {score:.4f}\n"
            f"- **Comparison Method**: {method}\n"
            f"- **Explanation**: {explanation}\n"
            f"\n"
            f"## Query\n"
            f"{record.query}\n"
            f"\n"
            f"## Actual Response (LLM Generated)\n"
            f"{record.actual_response}\n"
            f"\n"
            f"## Ideal Response (Ground Truth)\n"
            f"{record.ideal_response}\n"
            f"\n"
            f"---\n"
            f"*Captured for model training/correction purposes*\n"
        )
        # Return the formatted markdown string
        return markdown
