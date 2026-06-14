"""
Integration tests for the full autocorrection pipeline flow.

Tests the complete pipeline workflow with mocked LLM providers
but real file I/O, configuration loading, and component interaction.
Verifies end-to-end behavior from query input to correction output.
"""

import json  # JSON for creating test data files
from pathlib import Path  # Path handling
from unittest.mock import AsyncMock, patch  # Mocking utilities

import pytest  # Test framework

from autocorrection_pipeline.config.settings import AppSettings, ProviderConfig  # Config types
from autocorrection_pipeline.pipeline.orchestrator import (  # Pipeline types
    PipelineOrchestrator,
    PipelineResult,
)


@pytest.fixture
def integration_settings(tmp_path: Path) -> AppSettings:
    """
    Provide a fully configured settings object for integration testing.

    Creates real files in a temporary directory for the pipeline to
    read and write, with mocked API keys.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        AppSettings: Complete settings for integration testing.
    """
    # Create ideal responses file with multiple test entries
    ideal_path: Path = tmp_path / "ideal_responses.json"
    ideal_data: dict = {
        "queries": [
            {
                "query": "What is the speed of light?",
                "ideal_response": "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
            },
            {
                "query": "What is water made of?",
                "ideal_response": "Water is made of two hydrogen atoms and one oxygen atom, with the chemical formula H2O.",
            },
            {
                "query": "What is gravity?",
                "ideal_response": "Gravity is a fundamental force of nature that attracts objects with mass toward each other.",
            },
        ]
    }
    # Write the ideal responses file
    ideal_path.write_text(json.dumps(ideal_data), encoding="utf-8")
    # Create output directory
    output_dir: Path = tmp_path / "corrections"
    output_dir.mkdir()
    # Return complete settings
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
        openai_api_key="sk-integration-test-key",
    )


class TestPipelineIntegration:
    """Integration tests exercising the full pipeline with mocked providers."""

    @pytest.mark.asyncio
    async def test_below_threshold_produces_correction_file(
        self, integration_settings: AppSettings
    ) -> None:
        """Verify that a low-confidence response generates a markdown correction file."""
        # Mock the provider to simulate a poor response
        with patch("autocorrection_pipeline.pipeline.orchestrator.ProviderFactory") as mock_factory:
            mock_provider: AsyncMock = AsyncMock()
            # Return a clearly wrong response
            mock_provider.generate_response.return_value = "Light goes really fast."
            # Return dissimilar embeddings (orthogonal vectors = 0 similarity)
            mock_provider.generate_embedding.side_effect = [
                [1.0, 0.0, 0.0, 0.0],  # actual response embedding
                [0.0, 0.0, 1.0, 0.0],  # ideal response embedding
            ]
            mock_factory.create.return_value = mock_provider
            # Run the pipeline for a single query
            orchestrator: PipelineOrchestrator = PipelineOrchestrator(integration_settings)
            await orchestrator.initialize()
            result: PipelineResult = await orchestrator.process_query("What is the speed of light?")
        # Verify correction was captured
        assert result.correction_captured is True
        assert result.correction_file_path is not None
        # Verify the file exists and has content
        assert result.correction_file_path.exists()
        content: str = result.correction_file_path.read_text(encoding="utf-8")
        # Verify file content includes expected sections
        assert "What is the speed of light?" in content
        assert "Light goes really fast." in content
        assert "299,792,458" in content
        assert "Correction Record" in content

    @pytest.mark.asyncio
    async def test_above_threshold_skips_capture(
        self, integration_settings: AppSettings
    ) -> None:
        """Verify that a high-confidence response does not generate a correction file."""
        # Mock the provider to simulate a good response
        with patch("autocorrection_pipeline.pipeline.orchestrator.ProviderFactory") as mock_factory:
            mock_provider: AsyncMock = AsyncMock()
            # Return a good response
            mock_provider.generate_response.return_value = (
                "The speed of light in a vacuum is approximately 299,792,458 meters per second."
            )
            # Return identical embeddings (similarity = 1.0)
            mock_provider.generate_embedding.return_value = [0.8, 0.6, 0.0, 0.0]
            mock_factory.create.return_value = mock_provider
            # Run the pipeline for a single query
            orchestrator: PipelineOrchestrator = PipelineOrchestrator(integration_settings)
            await orchestrator.initialize()
            result: PipelineResult = await orchestrator.process_query("What is the speed of light?")
        # Verify no correction was captured
        assert result.correction_captured is False
        assert result.correction_file_path is None
        # Verify no files were created in output directory
        output_files: list[Path] = list(integration_settings.output_directory.glob("*.md"))
        assert len(output_files) == 0

    @pytest.mark.asyncio
    async def test_batch_processing_handles_all_queries(
        self, integration_settings: AppSettings
    ) -> None:
        """Verify batch mode processes every query in the ideal responses file."""
        # Mock the provider with varying responses
        with patch("autocorrection_pipeline.pipeline.orchestrator.ProviderFactory") as mock_factory:
            mock_provider: AsyncMock = AsyncMock()
            # Return different response each call
            mock_provider.generate_response.side_effect = [
                "Light is fast.",  # Bad response
                "H2O molecules.",  # Bad response
                "Gravity is a fundamental force attracting objects with mass.",  # Good response
            ]
            # Return embeddings that alternate between dissimilar and similar
            mock_provider.generate_embedding.side_effect = [
                [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],  # Orthogonal (score 0)
                [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],  # Orthogonal (score 0)
                [0.9, 0.1, 0.0], [0.9, 0.1, 0.0],  # Identical (score 1.0)
            ]
            mock_factory.create.return_value = mock_provider
            # Run batch processing
            orchestrator: PipelineOrchestrator = PipelineOrchestrator(integration_settings)
            await orchestrator.initialize()
            results: list[PipelineResult] = await orchestrator.run_batch()
        # Verify all three queries were processed
        assert len(results) == 3
        # Verify first two triggered corrections
        assert results[0].correction_captured is True
        assert results[1].correction_captured is True
        # Verify third passed the threshold
        assert results[2].correction_captured is False
        # Verify two correction files were created
        output_files: list[Path] = list(integration_settings.output_directory.glob("*.md"))
        assert len(output_files) == 2

    @pytest.mark.asyncio
    async def test_llm_judge_comparison_method(self, integration_settings: AppSettings) -> None:
        """Verify pipeline works with LLM judge comparison method."""
        # Override comparison method to llm_judge
        judge_settings: AppSettings = AppSettings(
            active_provider="openai",
            confidence_threshold=0.8,
            comparison_method="llm_judge",
            providers=integration_settings.providers,
            ideal_responses_path=integration_settings.ideal_responses_path,
            output_directory=integration_settings.output_directory,
            log_level="DEBUG",
            openai_api_key="sk-test",
        )
        # Mock the provider
        with patch("autocorrection_pipeline.pipeline.orchestrator.ProviderFactory") as mock_factory:
            mock_provider: AsyncMock = AsyncMock()
            # First call is the query response
            # Second call is the judge evaluation
            mock_provider.generate_response.side_effect = [
                "Light travels at 300000 km/s.",  # query response
                '{"score": 0.75, "explanation": "Close but not precise."}',  # judge response
            ]
            mock_factory.create.return_value = mock_provider
            # Run the pipeline
            orchestrator: PipelineOrchestrator = PipelineOrchestrator(judge_settings)
            await orchestrator.initialize()
            result: PipelineResult = await orchestrator.process_query("What is the speed of light?")
        # Verify LLM judge result
        assert result.comparison_result.method == "llm_judge"
        assert result.comparison_result.confidence_score == pytest.approx(0.75, abs=0.01)
        # Score 0.75 < 0.8 threshold so correction should be captured
        assert result.correction_captured is True
