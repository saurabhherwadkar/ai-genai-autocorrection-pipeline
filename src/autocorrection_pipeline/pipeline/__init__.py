"""
Pipeline orchestration subpackage.

Provides the main pipeline orchestrator that coordinates the full
autocorrection workflow from query processing through correction capture.
"""

# Re-export primary pipeline interface for convenient imports
from autocorrection_pipeline.pipeline.orchestrator import PipelineOrchestrator, PipelineResult

# Define public API for this subpackage
__all__ = ["PipelineOrchestrator", "PipelineResult"]
