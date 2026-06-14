"""
Logging configuration subpackage.

Provides centralized logging setup from YAML configuration
with support for log level override via settings or environment.
"""

# Re-export the primary logging configuration function
from autocorrection_pipeline.logging_config.setup import configure_logging

# Define public API for this subpackage
__all__ = ["configure_logging"]
