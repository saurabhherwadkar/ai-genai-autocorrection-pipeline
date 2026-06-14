"""
Application entry point for the autocorrection pipeline.

Provides the CLI interface and bootstraps the pipeline execution
including configuration loading, logging setup, and pipeline
orchestration.

Responsibilities:
    - Bootstrap application configuration and logging
    - Initialize the pipeline orchestrator
    - Run batch processing of all configured queries
    - Report results and handle top-level errors gracefully
"""

import asyncio  # Async event loop management
import logging  # Standard library logging
import sys  # System exit code handling

from autocorrection_pipeline.config.settings import (  # Configuration loading
    AppSettings,
    load_settings,
)
from autocorrection_pipeline.logging_config.setup import configure_logging  # Logging setup
from autocorrection_pipeline.pipeline.orchestrator import (  # Pipeline components
    PipelineOrchestrator,
    PipelineResult,
)

# Module-level logger for main entry point operations
logger: logging.Logger = logging.getLogger(__name__)


async def main() -> int:
    """
    Main async entry point for the autocorrection pipeline.

    Bootstraps the application by loading configuration, setting up
    logging, initializing the pipeline, and running batch processing.

    Returns:
        int: Exit code where 0 indicates success and 1 indicates failure.
    """
    try:
        # Load application settings from YAML and environment
        settings: AppSettings = load_settings()
        # Configure logging with the loaded settings
        configure_logging(level_override=settings.log_level)
        # Log application startup
        logger.info("=" * 60)
        logger.info("AI GenAI Auto-Correction Pipeline Starting")
        logger.info("=" * 60)
        # Log the active configuration summary
        logger.info("Active provider: %s", settings.active_provider)
        logger.info("Comparison method: %s", settings.comparison_method)
        logger.info("Confidence threshold: %.2f", settings.confidence_threshold)
        # Create the pipeline orchestrator with settings
        orchestrator: PipelineOrchestrator = PipelineOrchestrator(settings)
        # Perform async initialization of pipeline components
        await orchestrator.initialize()
        # Run batch processing for all configured queries
        results: list[PipelineResult] = await orchestrator.run_batch()
        # Report the batch processing results
        _report_results(results)
        # Log successful completion
        logger.info("Pipeline execution completed successfully")
        # Return success exit code
        return 0
    except FileNotFoundError as file_err:
        # Handle missing configuration or data files
        logger.error("File not found: %s", file_err)
        # Return failure exit code
        return 1
    except ValueError as val_err:
        # Handle configuration validation errors
        logger.error("Configuration error: %s", val_err)
        # Return failure exit code
        return 1
    except Exception as unexpected_err:
        # Handle any unforeseen errors at the top level
        logger.critical("Unexpected error: %s", unexpected_err, exc_info=True)
        # Return failure exit code
        return 1


def _report_results(results: list[PipelineResult]) -> None:
    """
    Report a summary of batch processing results to the log.

    Calculates statistics about the batch run and logs them
    including total queries, corrections captured, and pass rate.

    Args:
        results: List of PipelineResult objects from batch processing.
    """
    # Calculate the total number of queries processed
    total_queries: int = len(results)
    # Count the number of corrections that were captured
    corrections_captured: int = sum(1 for r in results if r.correction_captured)
    # Calculate the number of queries that passed the threshold
    passed_count: int = total_queries - corrections_captured
    # Log the summary header
    logger.info("-" * 40)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("-" * 40)
    # Log total queries processed
    logger.info("Total queries processed: %d", total_queries)
    # Log the number that passed the threshold
    logger.info("Passed threshold: %d", passed_count)
    # Log the number of corrections captured
    logger.info("Corrections captured: %d", corrections_captured)
    # Calculate and log the pass rate percentage
    if total_queries > 0:
        # Compute pass rate as a percentage
        pass_rate: float = (passed_count / total_queries) * 100.0
        # Log the pass rate
        logger.info("Pass rate: %.1f%%", pass_rate)
    # Log the summary footer
    logger.info("-" * 40)


def cli() -> None:
    """
    CLI entry point invoked by Poetry script configuration.

    Runs the async main function and exits with the returned code.
    This function is referenced in pyproject.toml as the CLI command.
    """
    # Run the async main function and capture the exit code
    exit_code: int = asyncio.run(main())
    # Exit the process with the returned code
    sys.exit(exit_code)


# Allow direct execution of this module as a script
if __name__ == "__main__":
    # Invoke the CLI entry point
    cli()
