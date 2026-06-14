"""
Unit tests for the correction capture module.

Tests markdown file generation, filename creation, and content
formatting for correction records.
"""

from datetime import UTC, datetime  # Timestamp handling
from pathlib import Path  # Path handling

import pytest  # Test framework

from autocorrection_pipeline.comparison.base import ComparisonResult  # Comparison result type
from autocorrection_pipeline.correction.capture import (  # Classes under test
    CorrectionCapture,
    CorrectionRecord,
)


class TestCorrectionCaptureInit:
    """Tests for correction capture initialization."""

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        """Verify initialization creates the output directory."""
        # Define a non-existent subdirectory
        output_dir: Path = tmp_path / "new_output_dir"
        # Create the capture instance (should create directory)
        CorrectionCapture(output_dir)
        # Verify the directory was created
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_existing_directory_no_error(self, tmp_output_dir: Path) -> None:
        """Verify initialization with existing directory succeeds."""
        # Create capture with existing directory
        CorrectionCapture(tmp_output_dir)
        # Should not raise any error
        assert tmp_output_dir.exists()


class TestCorrectionCaptureCapture:
    """Tests for the capture method."""

    @pytest.mark.asyncio
    async def test_creates_markdown_file(
        self, tmp_output_dir: Path, sample_correction_record: CorrectionRecord
    ) -> None:
        """Verify capture creates a markdown file in output directory."""
        # Create capture instance
        capture: CorrectionCapture = CorrectionCapture(tmp_output_dir)
        # Capture the correction record
        file_path: Path = await capture.capture(sample_correction_record)
        # Verify the file was created
        assert file_path.exists()
        assert file_path.suffix == ".md"
        assert file_path.parent == tmp_output_dir

    @pytest.mark.asyncio
    async def test_file_contains_query(
        self, tmp_output_dir: Path, sample_correction_record: CorrectionRecord
    ) -> None:
        """Verify captured file contains the original query."""
        # Create capture and write record
        capture: CorrectionCapture = CorrectionCapture(tmp_output_dir)
        file_path: Path = await capture.capture(sample_correction_record)
        # Read the file content
        content: str = file_path.read_text(encoding="utf-8")
        # Verify query is in the content
        assert sample_correction_record.query in content

    @pytest.mark.asyncio
    async def test_file_contains_actual_response(
        self, tmp_output_dir: Path, sample_correction_record: CorrectionRecord
    ) -> None:
        """Verify captured file contains the actual LLM response."""
        # Create capture and write record
        capture: CorrectionCapture = CorrectionCapture(tmp_output_dir)
        file_path: Path = await capture.capture(sample_correction_record)
        # Read the file content
        content: str = file_path.read_text(encoding="utf-8")
        # Verify actual response is in the content
        assert sample_correction_record.actual_response in content

    @pytest.mark.asyncio
    async def test_file_contains_ideal_response(
        self, tmp_output_dir: Path, sample_correction_record: CorrectionRecord
    ) -> None:
        """Verify captured file contains the ideal response."""
        # Create capture and write record
        capture: CorrectionCapture = CorrectionCapture(tmp_output_dir)
        file_path: Path = await capture.capture(sample_correction_record)
        # Read the file content
        content: str = file_path.read_text(encoding="utf-8")
        # Verify ideal response is in the content
        assert sample_correction_record.ideal_response in content

    @pytest.mark.asyncio
    async def test_file_contains_confidence_score(
        self, tmp_output_dir: Path, sample_correction_record: CorrectionRecord
    ) -> None:
        """Verify captured file contains the confidence score."""
        # Create capture and write record
        capture: CorrectionCapture = CorrectionCapture(tmp_output_dir)
        file_path: Path = await capture.capture(sample_correction_record)
        # Read the file content
        content: str = file_path.read_text(encoding="utf-8")
        # Verify score is in the content
        assert "0.6500" in content

    @pytest.mark.asyncio
    async def test_file_contains_markdown_headers(
        self, tmp_output_dir: Path, sample_correction_record: CorrectionRecord
    ) -> None:
        """Verify captured file has proper markdown structure."""
        # Create capture and write record
        capture: CorrectionCapture = CorrectionCapture(tmp_output_dir)
        file_path: Path = await capture.capture(sample_correction_record)
        # Read the file content
        content: str = file_path.read_text(encoding="utf-8")
        # Verify markdown headers are present
        assert "# Correction Record" in content
        assert "## Metadata" in content
        assert "## Query" in content
        assert "## Actual Response" in content
        assert "## Ideal Response" in content


class TestGenerateFilename:
    """Tests for filename generation."""

    def test_filename_contains_timestamp(
        self, tmp_output_dir: Path, sample_correction_record: CorrectionRecord
    ) -> None:
        """Verify filename includes the timestamp."""
        # Create capture instance
        capture: CorrectionCapture = CorrectionCapture(tmp_output_dir)
        # Generate filename
        filename: str = capture._generate_filename(sample_correction_record)
        # Verify timestamp format is in filename
        assert "2024-01-15T10-30-00" in filename

    def test_filename_ends_with_md_extension(
        self, tmp_output_dir: Path, sample_correction_record: CorrectionRecord
    ) -> None:
        """Verify filename has .md extension."""
        # Create capture instance
        capture: CorrectionCapture = CorrectionCapture(tmp_output_dir)
        # Generate filename
        filename: str = capture._generate_filename(sample_correction_record)
        # Verify .md extension
        assert filename.endswith(".md")

    def test_filename_starts_with_correction_prefix(
        self, tmp_output_dir: Path, sample_correction_record: CorrectionRecord
    ) -> None:
        """Verify filename starts with 'correction_' prefix."""
        # Create capture instance
        capture: CorrectionCapture = CorrectionCapture(tmp_output_dir)
        # Generate filename
        filename: str = capture._generate_filename(sample_correction_record)
        # Verify prefix
        assert filename.startswith("correction_")

    def test_different_queries_produce_different_filenames(
        self, tmp_output_dir: Path, sample_comparison_result: ComparisonResult
    ) -> None:
        """Verify different queries generate unique filenames."""
        # Create two records with different queries
        record_a: CorrectionRecord = CorrectionRecord(
            query="Query A",
            actual_response="Response A",
            ideal_response="Ideal A",
            comparison_result=sample_comparison_result,
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        )
        record_b: CorrectionRecord = CorrectionRecord(
            query="Query B",
            actual_response="Response B",
            ideal_response="Ideal B",
            comparison_result=sample_comparison_result,
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        )
        # Generate filenames for each
        capture: CorrectionCapture = CorrectionCapture(tmp_output_dir)
        filename_a: str = capture._generate_filename(record_a)
        filename_b: str = capture._generate_filename(record_b)
        # Verify they are different
        assert filename_a != filename_b


class TestFormatMarkdown:
    """Tests for markdown formatting."""

    def test_format_includes_training_purpose_note(
        self, tmp_output_dir: Path, sample_correction_record: CorrectionRecord
    ) -> None:
        """Verify markdown includes the training purpose note."""
        # Create capture instance
        capture: CorrectionCapture = CorrectionCapture(tmp_output_dir)
        # Format the markdown
        markdown: str = capture._format_markdown(sample_correction_record)
        # Verify training note is present
        assert "model training/correction purposes" in markdown

    def test_format_includes_comparison_method(
        self, tmp_output_dir: Path, sample_correction_record: CorrectionRecord
    ) -> None:
        """Verify markdown includes the comparison method used."""
        # Create capture instance
        capture: CorrectionCapture = CorrectionCapture(tmp_output_dir)
        # Format the markdown
        markdown: str = capture._format_markdown(sample_correction_record)
        # Verify comparison method is present
        assert "cosine_similarity" in markdown
