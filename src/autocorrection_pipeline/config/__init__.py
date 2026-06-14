"""
Configuration management subpackage.

Provides settings loading from YAML configuration files
and environment variables for the autocorrection pipeline.
"""

# Re-export primary configuration interface for convenient imports
from autocorrection_pipeline.config.settings import AppSettings, load_settings

# Define public API for this subpackage
__all__ = ["AppSettings", "load_settings"]
