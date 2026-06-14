"""
Logging configuration module for the autocorrection pipeline.

Configures Python's logging framework from a YAML configuration file
with support for runtime log level overrides. Ensures log directories
exist before file handlers attempt to write.

Responsibilities:
    - Load logging configuration from config/logging.yaml
    - Create log output directories if they do not exist
    - Apply log level overrides from settings or environment
    - Configure rotating file handlers for production use
"""

import logging  # Standard library logging framework
import logging.config  # Dictionary-based logging configuration
from pathlib import Path  # Cross-platform path handling

import yaml  # YAML parser for logging config file

# Module-level logger for logging setup diagnostics
logger: logging.Logger = logging.getLogger(__name__)


def configure_logging(
    config_path: Path | None = None,
    level_override: str | None = None,
) -> None:
    """
    Configure application logging from YAML configuration file.

    Loads the logging configuration from the specified YAML file and
    applies it to Python's logging framework. Optionally overrides
    the log level for the application logger.

    Args:
        config_path: Path to the logging.yaml configuration file.
                     If None, defaults to config/logging.yaml relative
                     to the project root directory.
        level_override: Optional log level string to override the
                       configured level (e.g., "DEBUG", "WARNING").
                       When provided, this takes precedence over the
                       level specified in the YAML configuration.

    Raises:
        FileNotFoundError: If the logging configuration file does not
                          exist at the specified or default path.
        yaml.YAMLError: If the logging configuration file contains
                       invalid YAML syntax.
    """
    # Resolve the project root directory for default path construction
    project_root: Path = _resolve_project_root()
    # Determine the logging configuration file path
    resolved_path: Path = config_path or (project_root / "config" / "logging.yaml")
    # Verify the logging configuration file exists
    if not resolved_path.exists():
        # Fall back to basic configuration if file is missing
        logging.basicConfig(level=logging.INFO)
        # Log a warning about the missing configuration file
        logger.warning("Logging config not found at %s, using basic config", resolved_path)
        # Exit early since there is no file to parse
        return
    # Open and parse the logging YAML configuration file
    with resolved_path.open(mode="r", encoding="utf-8") as config_file:
        # Parse YAML content using safe loader to prevent code execution
        log_config: dict = yaml.safe_load(config_file)
    # Ensure the log output directory exists for file handlers
    _ensure_log_directory(project_root / "logs")
    # Update file handler paths to be absolute relative to project root
    _resolve_handler_paths(log_config, project_root)
    # Apply the parsed logging configuration to Python's logging system
    logging.config.dictConfig(log_config)
    # Apply log level override if one was specified
    if level_override:
        # Get the application-specific logger instance
        app_logger: logging.Logger = logging.getLogger("autocorrection_pipeline")
        # Set the overridden log level on the application logger
        app_logger.setLevel(getattr(logging, level_override.upper(), logging.INFO))
        # Log the level override for diagnostic purposes
        logger.debug("Log level overridden to: %s", level_override.upper())


def _resolve_project_root() -> Path:
    """
    Determine the project root by traversing up from this file.

    Walks the directory tree upward looking for pyproject.toml
    which serves as the project root marker.

    Returns:
        Path: Absolute path to the project root directory.

    Raises:
        FileNotFoundError: If pyproject.toml is not found in any
                          parent directory.
    """
    # Start from the directory containing this module file
    current_directory: Path = Path(__file__).resolve().parent
    # Traverse up through parent directories
    for parent in [current_directory, *current_directory.parents]:
        # Check if this directory contains the project marker
        if (parent / "pyproject.toml").exists():
            # Return the project root path
            return parent
    # Project root marker not found anywhere in hierarchy
    raise FileNotFoundError("Could not locate project root: no pyproject.toml found")


def _ensure_log_directory(log_path: Path) -> None:
    """
    Create the log output directory if it does not already exist.

    Uses exist_ok=True to avoid errors when the directory already
    exists, and parents=True to create intermediate directories.

    Args:
        log_path: The directory path where log files will be written.
    """
    # Create the directory and any required parent directories
    log_path.mkdir(parents=True, exist_ok=True)


def _resolve_handler_paths(log_config: dict, project_root: Path) -> None:
    """
    Convert relative file handler paths to absolute paths.

    Iterates through all configured handlers and resolves any
    relative filename paths to be absolute relative to the project root.

    Args:
        log_config: The parsed logging configuration dictionary.
        project_root: The project root path for resolving relatives.
    """
    # Get the handlers section from the configuration
    handlers: dict = log_config.get("handlers", {})
    # Iterate over each handler configuration
    for handler_config in handlers.values():
        # Check if this handler has a filename property
        if "filename" in handler_config:
            # Resolve the filename relative to the project root
            handler_config["filename"] = str(
                project_root / handler_config["filename"]
            )
