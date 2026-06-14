"""
Unit tests for the pipeline orchestrator module.

Tests the full pipeline workflow including query processing,
comparison evaluation, and correction capture decisions.
"""

import json  # JSON for creating test files
from pathlib import Path  # Path handling
from unittest.mock import AsyncMock, patch  # Mocking utilities

import pytest  # Test framework

from autocorrection_pipeline.comparison.base import ComparisonResult  # Result type
from autocorrection_pipeline.config.settings import AppSettings, ProviderConfig  # Config types
from autocorrection_pipeline.pipeline.orchestrator import (  # Classes under test
    PipelineOrchestrator,
    PipelineResult,
)


@pytest.fixture
def orchestrator_settings(tmp_path: Path) -> AppSettings:
    """Provide settings configured for orchestrator testing."""
    # Create ideal responses file
    ideal_path: Path = tmp_path / "ideal_responses.json"
    ideal_data: dict = {
        "queries": [
            {"query": "What is Python?", "ideal_response": "Python is a programming language."},
            {"query": "What is AI?", "ideal_response": "AI is artificial intelligence."},
        ]
    }
    ideal_path.write_text(json.dumps(ideal_data), encoding="utf-8")
    # Create output directory
    output_dir: Path = tmp_path / "output"
    output_dir.mkdir()
    # Return configured settings
    return AppSettings(
        active_provider="openai",
        confidence_threshold=0.8,
        comparison_method="cosine_similarity",
        providers={
            "openai": ProviderConfig(
                model="gpt-4o",
                max_tokens=4096,
                temperature=0.0,
                embedding_model="text-embedding-3-small",
            ),
        },
        ideal_responses_path=ideal_path,
        output_directory=output_dir,
        log_level="DEBUG",
        openai_api_key="sk-test-key",
    )


class TestPipelineOrchestratorInit:
    """Tests for orchestrator initialization."""

    def test_init_stores_settings(self, orchestrator_settings: AppSettings) -> None:
        """Verify orchestrator stores settings on construction."""
        # Create orchestrator
        orchestrator: PipelineOrchestrator = PipelineOrchestrator(orchestrator_settings)
        # Verify settings are stored
        assert orchestrator._settings == orchestrator_settings

    def test_init_components_are_none(self, orchestrator_settings: AppSettings) -> None:
        """Verify components are None before initialize()."""
        # Create orchestrator without initializing
        orchestrator: PipelineOrchestrator = PipelineOrchestrator(orchestrator_settings)
        # Verify components are not yet created
        assert orchestrator._provider is None
        assert orchestrator._comparator is None
        assert orchestrator._capture is None


class TestPipelineOrchestratorInitialize:
    """Tests for the async initialize method."""

    @pytest.mark.asyncio
    async def test_initialize_creates_components(self, orchestrator_settings: AppSettings) -> None:
        """Verify initialize creates all pipeline components."""
        # Mock the provider creation
        with patch("autocorrection_pipeline.pipeline.orchestrator.ProviderFactory") as mock_factory:
            mock_provider: AsyncMock = AsyncMock()
            mock_factory.create.return_value = mock_provider
            # Create and initialize orchestrator
            orchestrator: PipelineOrchestrator = PipelineOrchestrator(orchestrator_settings)
            await orchestrator.initialize()
        # Verify components were created
        assert orchestrator._provider is not None
        assert orchestrator._comparator is not None
        assert orchestrator._capture is not None
        assert orchestrator._ideal_responses is not None

    @pytest.mark.asyncio
    async def test_initialize_loads_ideal_responses(self, orchestrator_settings: AppSettings) -> None:
        """Verify initialize loads ideal responses from file."""
        # Mock the provider creation
        with patch("autocorrection_pipeline.pipeline.orchestrator.ProviderFactory") as mock_factory:
            mock_factory.create.return_value = AsyncMock()
            # Create and initialize orchestrator
            orchestrator: PipelineOrchestrator = PipelineOrchestrator(orchestrator_settings)
            await orchestrator.initialize()
        # Verify ideal responses were loaded
        assert len(orchestrator._ideal_responses) == 2
        assert "What is Python?" in orchestrator._ideal_responses


class TestPipelineOrchestratorProcessQuery:
    """Tests for the process_query method."""

    @pytest.mark.asyncio
    async def test_process_query_not_initialized_raises_error(
        self, orchestrator_settings: AppSettings
    ) -> None:
        """Verify process_query raises RuntimeError if not initialized."""
        # Create orchestrator without initializing
        orchestrator: PipelineOrchestrator = PipelineOrchestrator(orchestrator_settings)
        # Attempt to process should raise RuntimeError
        with pytest.raises(RuntimeError, match="not initialized"):
            await orchestrator.process_query("What is Python?")

    @pytest.mark.asyncio
    async def test_process_query_below_threshold_captures_correction(
        self, orchestrator_settings: AppSettings
    ) -> None:
        """Verify low-confidence responses trigger correction capture."""
        # Mock the provider to return a poor response
        with patch("autocorrection_pipeline.pipeline.orchestrator.ProviderFactory") as mock_factory:
            mock_provider: AsyncMock = AsyncMock()
            mock_provider.generate_response.return_value = "Python is a snake."
            mock_provider.generate_embedding.side_effect = [
                [1.0, 0.0, 0.0],  # actual embedding
                [0.0, 1.0, 0.0],  # ideal embedding (orthogonal = score 0)
            ]
            mock_factory.create.return_value = mock_provider
            # Create, initialize, and process
            orchestrator: PipelineOrchestrator = PipelineOrchestrator(orchestrator_settings)
            await orchestrator.initialize()
            result: PipelineResult = await orchestrator.process_query("What is Python?")
        # Verify correction was captured
        assert result.correction_captured is True
        assert result.correction_file_path is not None
        assert result.correction_file_path.exists()

    @pytest.mark.asyncio
    async def test_process_query_above_threshold_skips_capture(
        self, orchestrator_settings: AppSettings
    ) -> None:
        """Verify high-confidence responses do not trigger correction."""
        # Mock the provider to return a good response
        with patch("autocorrection_pipeline.pipeline.orchestrator.ProviderFactory") as mock_factory:
            mock_provider: AsyncMock = AsyncMock()
            mock_provider.generate_response.return_value = "Python is a programming language."
            mock_provider.generate_embedding.return_value = [1.0, 1.0, 1.0]
            mock_factory.create.return_value = mock_provider
            # Create, initialize, and process
            orchestrator: PipelineOrchestrator = PipelineOrchestrator(orchestrator_settings)
            await orchestrator.initialize()
            result: PipelineResult = await orchestrator.process_query("What is Python?")
        # Verify correction was NOT captured (score is 1.0 for identical vectors)
        assert result.correction_captured is False
        assert result.correction_file_path is None

    @pytest.mark.asyncio
    async def test_process_unknown_query_raises_key_error(
        self, orchestrator_settings: AppSettings
    ) -> None:
        """Verify unknown query raises KeyError."""
        # Mock the provider
        with patch("autocorrection_pipeline.pipeline.orchestrator.ProviderFactory") as mock_factory:
            mock_provider: AsyncMock = AsyncMock()
            mock_provider.generate_response.return_value = "Some response"
            mock_factory.create.return_value = mock_provider
            # Create, initialize, and process unknown query
            orchestrator: PipelineOrchestrator = PipelineOrchestrator(orchestrator_settings)
            await orchestrator.initialize()
            with pytest.raises(KeyError, match="not found"):
                await orchestrator.process_query("Unknown question not in file")


class TestPipelineOrchestratorRunBatch:
    """Tests for the batch processing method."""

    @pytest.mark.asyncio
    async def test_batch_processes_all_queries(self, orchestrator_settings: AppSettings) -> None:
        """Verify batch mode processes every query in the file."""
        # Mock the provider
        with patch("autocorrection_pipeline.pipeline.orchestrator.ProviderFactory") as mock_factory:
            mock_provider: AsyncMock = AsyncMock()
            mock_provider.generate_response.return_value = "Test response"
            mock_provider.generate_embedding.return_value = [1.0, 1.0, 1.0]
            mock_factory.create.return_value = mock_provider
            # Create, initialize, and run batch
            orchestrator: PipelineOrchestrator = PipelineOrchestrator(orchestrator_settings)
            await orchestrator.initialize()
            results: list[PipelineResult] = await orchestrator.run_batch()
        # Verify all queries were processed
        assert len(results) == 2


class TestShouldCaptureCorrection:
    """Tests for the correction capture decision logic."""

    def test_below_threshold_returns_true(self, orchestrator_settings: AppSettings) -> None:
        """Verify score below threshold triggers capture."""
        # Create orchestrator
        orchestrator: PipelineOrchestrator = PipelineOrchestrator(orchestrator_settings)
        # Create a low-confidence result
        result: ComparisonResult = ComparisonResult(
            confidence_score=0.5, method="test", explanation="Low score"
        )
        # Verify capture is triggered
        assert orchestrator._should_capture_correction(result) is True

    def test_above_threshold_returns_false(self, orchestrator_settings: AppSettings) -> None:
        """Verify score above threshold does not trigger capture."""
        # Create orchestrator
        orchestrator: PipelineOrchestrator = PipelineOrchestrator(orchestrator_settings)
        # Create a high-confidence result
        result: ComparisonResult = ComparisonResult(
            confidence_score=0.9, method="test", explanation="High score"
        )
        # Verify capture is not triggered
        assert orchestrator._should_capture_correction(result) is False

    def test_at_threshold_returns_false(self, orchestrator_settings: AppSettings) -> None:
        """Verify score exactly at threshold does not trigger capture."""
        # Create orchestrator
        orchestrator: PipelineOrchestrator = PipelineOrchestrator(orchestrator_settings)
        # Create a result at exactly the threshold
        result: ComparisonResult = ComparisonResult(
            confidence_score=0.8, method="test", explanation="At threshold"
        )
        # Verify capture is NOT triggered (threshold is strict less-than)
        assert orchestrator._should_capture_correction(result) is False
