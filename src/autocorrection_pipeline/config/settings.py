"""
Configuration management module for the autocorrection pipeline.

Loads application settings from YAML configuration files and environment
variables. Implements immutable dataclasses for type-safe configuration
access throughout the application.

Responsibilities:
    - Parse config/settings.yaml for application settings
    - Load sensitive API keys from environment variables via .env
    - Validate all configuration values at startup
    - Provide a single immutable AppSettings object to the application
"""

from __future__ import annotations  # Enable forward references for type hints

import logging  # Standard library logging for configuration errors
import os  # Operating system interface for environment variable access
from dataclasses import dataclass, field  # Structured immutable data containers
from pathlib import Path  # Cross-platform filesystem path handling
from typing import Any  # Generic type annotation support

import yaml  # YAML parser for configuration file loading
from dotenv import load_dotenv  # Load .env files into process environment

# Module-level logger for configuration-related messages
logger: logging.Logger = logging.getLogger(__name__)

# Maximum allowed length for configuration string values
_MAX_CONFIG_STRING_LENGTH: int = 10000

# Valid provider identifiers recognized by the factory
_VALID_PROVIDERS: set[str] = {"openai", "anthropic", "azure_openai"}

# Valid comparison method identifiers
_VALID_COMPARISON_METHODS: set[str] = {"cosine_similarity", "llm_judge"}


@dataclass(frozen=True)
class ProviderConfig:
    """
    Immutable configuration for a single LLM provider.

    Holds model identifiers, token limits, and provider-specific
    parameters loaded from the settings YAML file.

    Attributes:
        model: The primary model identifier (e.g., 'gpt-4o').
        max_tokens: Maximum number of tokens in LLM response.
        temperature: Sampling temperature (0.0 for deterministic output).
        embedding_model: Optional embedding model name for vector operations.
        api_version: Optional API version string (used by Azure OpenAI).
    """

    # The primary model identifier string for this provider
    model: str
    # Maximum number of tokens the model should generate
    max_tokens: int
    # Sampling temperature controlling response randomness
    temperature: float
    # Optional embedding model name for vector generation
    embedding_model: str | None = None
    # Optional API version identifier (Azure-specific)
    api_version: str | None = None


@dataclass(frozen=True)
class AppSettings:
    """
    Top-level immutable application settings container.

    Aggregates all configuration values needed by the pipeline including
    provider settings, comparison parameters, file paths, and API keys.
    API key fields use repr=False to prevent accidental logging of secrets.

    Attributes:
        active_provider: Name of the currently active LLM provider.
        confidence_threshold: Score below which corrections are captured.
        comparison_method: Algorithm used for response comparison.
        providers: Dictionary of provider-specific configurations.
        ideal_responses_path: Path to the ideal responses JSON file.
        output_directory: Directory path for correction output files.
        log_level: Application logging level string.
        openai_api_key: OpenAI API authentication key.
        anthropic_api_key: Anthropic API authentication key.
        azure_openai_api_key: Azure OpenAI API authentication key.
        azure_openai_endpoint: Azure OpenAI service endpoint URL.
    """

    # Name of the currently active LLM provider
    active_provider: str
    # Confidence score threshold for triggering correction capture
    confidence_threshold: float
    # Comparison algorithm identifier
    comparison_method: str
    # Mapping of provider names to their configuration objects
    providers: dict[str, ProviderConfig]
    # Filesystem path to the ideal responses JSON file
    ideal_responses_path: Path
    # Filesystem path to the correction output directory
    output_directory: Path
    # Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_level: str
    # OpenAI API key loaded from environment (hidden from repr)
    openai_api_key: str = field(default="", repr=False)
    # Anthropic API key loaded from environment (hidden from repr)
    anthropic_api_key: str = field(default="", repr=False)
    # Azure OpenAI API key loaded from environment (hidden from repr)
    azure_openai_api_key: str = field(default="", repr=False)
    # Azure OpenAI endpoint URL loaded from environment (hidden from repr)
    azure_openai_endpoint: str = field(default="", repr=False)


def load_settings(config_path: Path | None = None) -> AppSettings:
    """
    Load application settings from YAML config and environment variables.

    This is the primary entry point for configuration loading. It resolves
    the project root, loads the YAML configuration, reads environment
    variables for sensitive values, and constructs a validated AppSettings.

    Args:
        config_path: Optional explicit path to settings.yaml file.
                     If None, defaults to config/settings.yaml relative
                     to the resolved project root directory.

    Returns:
        AppSettings: A fully populated and validated immutable settings object
                     containing all configuration needed by the pipeline.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist
                          on the filesystem.
        ValueError: If required configuration values are missing, out of
                   valid range, or contain invalid identifiers.
        yaml.YAMLError: If the YAML file contains syntax errors.
    """
    # Load environment variables from .env file into process environment
    load_dotenv()
    # Determine the root directory of the project
    project_root: Path = _resolve_project_root()
    # Resolve the configuration file path
    resolved_config_path: Path = config_path or (project_root / "config" / "settings.yaml")
    # Log the configuration file being loaded
    logger.info("Loading configuration from: %s", resolved_config_path)
    # Parse the YAML configuration file into a dictionary
    raw_config: dict[str, Any] = _load_yaml_config(resolved_config_path)
    # Extract the LLM section from raw configuration
    llm_config: dict[str, Any] = raw_config.get("llm", {})
    # Extract the paths section from raw configuration
    paths_config: dict[str, Any] = raw_config.get("paths", {})
    # Extract the logging section from raw configuration
    logging_config: dict[str, Any] = raw_config.get("logging", {})
    # Build provider configuration objects from raw YAML data
    provider_configs: dict[str, ProviderConfig] = _build_provider_configs(
        raw_config.get("providers", {})
    )
    # Load sensitive API keys from environment variables
    env_keys: dict[str, str] = _load_env_keys()
    # Retrieve the active provider identifier
    active_provider: str = llm_config.get("active_provider", "openai")
    # Validate the active provider is a recognized identifier
    _validate_provider_name(active_provider)
    # Retrieve the confidence threshold value
    threshold: float = float(llm_config.get("confidence_threshold", 0.8))
    # Validate the confidence threshold is within valid range
    _validate_threshold(threshold)
    # Retrieve the comparison method identifier
    comparison_method: str = llm_config.get("comparison_method", "cosine_similarity")
    # Validate the comparison method is recognized
    _validate_comparison_method(comparison_method)
    # Resolve the ideal responses file path relative to project root
    ideal_responses_path: Path = project_root / paths_config.get(
        "ideal_responses", "config/ideal_responses.json"
    )
    # Resolve the output directory path relative to project root
    output_directory: Path = project_root / paths_config.get("output_directory", "output")
    # Determine log level from config or environment variable override
    log_level: str = os.environ.get("LOG_LEVEL", logging_config.get("level", "INFO"))
    # Construct and return the immutable settings object
    return AppSettings(
        active_provider=active_provider,
        confidence_threshold=threshold,
        comparison_method=comparison_method,
        providers=provider_configs,
        ideal_responses_path=ideal_responses_path,
        output_directory=output_directory,
        log_level=log_level,
        openai_api_key=env_keys.get("openai_api_key", ""),
        anthropic_api_key=env_keys.get("anthropic_api_key", ""),
        azure_openai_api_key=env_keys.get("azure_openai_api_key", ""),
        azure_openai_endpoint=env_keys.get("azure_openai_endpoint", ""),
    )


def _resolve_project_root() -> Path:
    """
    Determine the project root directory by traversing up from this file.

    Walks up the directory tree from the current file location until it
    finds a directory containing pyproject.toml, which indicates the
    project root.

    Returns:
        Path: The absolute path to the project root directory.

    Raises:
        FileNotFoundError: If pyproject.toml cannot be found in any
                          parent directory up to the filesystem root.
    """
    # Start from the directory containing this settings module
    current_directory: Path = Path(__file__).resolve().parent
    # Traverse up the directory tree looking for pyproject.toml
    for parent in [current_directory, *current_directory.parents]:
        # Check if pyproject.toml exists in this directory
        if (parent / "pyproject.toml").exists():
            # Found the project root directory
            return parent
    # No pyproject.toml found in any parent directory
    raise FileNotFoundError(
        "Could not locate project root: no pyproject.toml found in parent directories"
    )


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """
    Parse and return the YAML configuration file contents.

    Opens the specified YAML file using safe loading to prevent
    arbitrary code execution from malicious YAML content.

    Args:
        config_path: Absolute filesystem path to the YAML config file.

    Returns:
        dict[str, Any]: Parsed YAML content as a nested dictionary.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If the file contains invalid YAML syntax.
    """
    # Verify the configuration file exists before attempting to read
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    # Open and parse the YAML file using safe loader
    with config_path.open(mode="r", encoding="utf-8") as config_file:
        # Use safe_load to prevent arbitrary code execution
        parsed_config: dict[str, Any] = yaml.safe_load(config_file) or {}
    # Log successful configuration loading
    logger.debug("Successfully parsed configuration file: %s", config_path)
    # Return the parsed configuration dictionary
    return parsed_config


def _build_provider_configs(raw_providers: dict[str, Any]) -> dict[str, ProviderConfig]:
    """
    Construct ProviderConfig instances from raw YAML dictionary data.

    Iterates over each provider entry in the configuration and creates
    a typed ProviderConfig dataclass instance with validated values.

    Args:
        raw_providers: Dictionary mapping provider names to their raw
                      configuration dictionaries from YAML parsing.

    Returns:
        dict[str, ProviderConfig]: Mapping of provider names to validated
                                   ProviderConfig instances.
    """
    # Initialize empty dictionary for provider configurations
    provider_configs: dict[str, ProviderConfig] = {}
    # Iterate over each provider name and its raw configuration
    for provider_name, provider_data in raw_providers.items():
        # Skip entries that are not valid dictionaries
        if not isinstance(provider_data, dict):
            continue
        # Create a typed ProviderConfig from the raw dictionary
        provider_configs[provider_name] = ProviderConfig(
            model=str(provider_data.get("model", "")),
            max_tokens=int(provider_data.get("max_tokens", 4096)),
            temperature=float(provider_data.get("temperature", 0.0)),
            embedding_model=provider_data.get("embedding_model"),
            api_version=provider_data.get("api_version"),
        )
    # Return the constructed provider configurations
    return provider_configs


def _load_env_keys() -> dict[str, str]:
    """
    Load sensitive API keys from environment variables.

    Reads API keys and endpoints from the process environment which
    should have been populated from the .env file via python-dotenv.
    Returns empty strings for missing keys rather than raising errors
    to allow partial configuration for testing.

    Returns:
        dict[str, str]: Dictionary mapping key identifiers to their
                       string values from environment variables.
    """
    # Read OpenAI API key from environment, default to empty string
    openai_key: str = os.environ.get("OPENAI_API_KEY", "")
    # Read Anthropic API key from environment, default to empty string
    anthropic_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    # Read Azure OpenAI API key from environment, default to empty string
    azure_key: str = os.environ.get("AZURE_OPENAI_API_KEY", "")
    # Read Azure OpenAI endpoint from environment, default to empty string
    azure_endpoint: str = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    # Return all environment keys as a dictionary
    return {
        "openai_api_key": openai_key,
        "anthropic_api_key": anthropic_key,
        "azure_openai_api_key": azure_key,
        "azure_openai_endpoint": azure_endpoint,
    }


def _validate_provider_name(provider_name: str) -> None:
    """
    Validate that the provider name is a recognized identifier.

    Checks the given provider name against the set of valid providers
    and raises a ValueError if it is not recognized.

    Args:
        provider_name: The provider identifier string to validate.

    Raises:
        ValueError: If the provider name is not in the valid set.
    """
    # Check if the provider name is in the allowed set
    if provider_name not in _VALID_PROVIDERS:
        # Raise descriptive error with valid options listed
        raise ValueError(
            f"Invalid provider '{provider_name}'. Must be one of: {_VALID_PROVIDERS}"
        )


def _validate_threshold(threshold: float) -> None:
    """
    Validate that the confidence threshold is within valid range.

    The confidence threshold must be between 0.0 and 1.0 inclusive
    to represent a valid probability score.

    Args:
        threshold: The confidence threshold value to validate.

    Raises:
        ValueError: If the threshold is outside the valid range [0.0, 1.0].
    """
    # Check if threshold is within the valid probability range
    if not 0.0 <= threshold <= 1.0:
        # Raise error with the invalid value and valid range
        raise ValueError(
            f"Confidence threshold must be between 0.0 and 1.0, got: {threshold}"
        )


def _validate_comparison_method(method: str) -> None:
    """
    Validate that the comparison method is a recognized algorithm.

    Checks the given method string against the set of valid comparison
    methods supported by the pipeline.

    Args:
        method: The comparison method identifier string to validate.

    Raises:
        ValueError: If the method is not in the valid set.
    """
    # Check if the comparison method is in the allowed set
    if method not in _VALID_COMPARISON_METHODS:
        # Raise descriptive error with valid options listed
        raise ValueError(
            f"Invalid comparison method '{method}'. Must be one of: {_VALID_COMPARISON_METHODS}"
        )
