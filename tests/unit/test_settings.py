"""
Unit tests for the configuration settings module.

Tests loading, parsing, and validation of application settings
from YAML configuration files and environment variables.
"""

from pathlib import Path  # Cross-platform path handling
from unittest.mock import patch  # Mocking utilities

import pytest  # Test framework

from autocorrection_pipeline.config.settings import (
    AppSettings,
    ProviderConfig,
    _build_provider_configs,
    _load_env_keys,
    _load_yaml_config,
    _validate_comparison_method,
    _validate_provider_name,
    _validate_threshold,
)


class TestProviderConfig:
    """Tests for the ProviderConfig dataclass."""

    def test_create_provider_config_with_all_fields(self) -> None:
        """Verify ProviderConfig stores all fields correctly."""
        # Create a config with all fields populated
        config: ProviderConfig = ProviderConfig(
            model="gpt-4o",
            max_tokens=4096,
            temperature=0.5,
            embedding_model="text-embedding-3-small",
            api_version="2024-06-01",
        )
        # Verify each field matches the input values
        assert config.model == "gpt-4o"
        assert config.max_tokens == 4096
        assert config.temperature == 0.5
        assert config.embedding_model == "text-embedding-3-small"
        assert config.api_version == "2024-06-01"

    def test_create_provider_config_with_defaults(self) -> None:
        """Verify ProviderConfig optional fields default to None."""
        # Create a config with only required fields
        config: ProviderConfig = ProviderConfig(
            model="gpt-4o",
            max_tokens=4096,
            temperature=0.0,
        )
        # Verify optional fields are None
        assert config.embedding_model is None
        assert config.api_version is None

    def test_provider_config_is_immutable(self) -> None:
        """Verify ProviderConfig cannot be modified after creation."""
        # Create a frozen config instance
        config: ProviderConfig = ProviderConfig(
            model="gpt-4o",
            max_tokens=4096,
            temperature=0.0,
        )
        # Attempt to modify should raise an error
        with pytest.raises(AttributeError):
            config.model = "different-model"  # type: ignore[misc]


class TestAppSettings:
    """Tests for the AppSettings dataclass."""

    def test_api_keys_hidden_from_repr(self, sample_settings: AppSettings) -> None:
        """Verify API keys are not exposed in string representation."""
        # Get the repr of settings
        repr_str: str = repr(sample_settings)
        # Verify API keys are not in the repr output
        assert "sk-test-key-openai" not in repr_str
        assert "sk-ant-test-key" not in repr_str
        assert "test-azure-key" not in repr_str

    def test_settings_stores_all_fields(self, sample_settings: AppSettings) -> None:
        """Verify AppSettings stores all configuration fields."""
        # Verify core settings are stored
        assert sample_settings.active_provider == "openai"
        assert sample_settings.confidence_threshold == 0.8
        assert sample_settings.comparison_method == "cosine_similarity"
        assert sample_settings.log_level == "DEBUG"

    def test_settings_is_immutable(self, sample_settings: AppSettings) -> None:
        """Verify AppSettings cannot be modified after creation."""
        # Attempt to modify should raise an error
        with pytest.raises(AttributeError):
            sample_settings.active_provider = "anthropic"  # type: ignore[misc]


class TestLoadYamlConfig:
    """Tests for the YAML configuration loading function."""

    def test_load_existing_yaml_file(self, tmp_path: Path) -> None:
        """Verify successful loading of a valid YAML config file."""
        # Create a test YAML file
        yaml_content: str = "app:\n  name: test\n  version: '1.0'"
        yaml_path: Path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")
        # Load the config
        result: dict = _load_yaml_config(yaml_path)
        # Verify the parsed content
        assert result["app"]["name"] == "test"
        assert result["app"]["version"] == "1.0"

    def test_load_missing_yaml_file_raises_error(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError for missing config file."""
        # Attempt to load a non-existent file
        missing_path: Path = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            _load_yaml_config(missing_path)

    def test_load_empty_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        """Verify empty YAML file returns empty dictionary."""
        # Create an empty YAML file
        yaml_path: Path = tmp_path / "empty.yaml"
        yaml_path.write_text("", encoding="utf-8")
        # Load should return empty dict
        result: dict = _load_yaml_config(yaml_path)
        assert result == {}


class TestBuildProviderConfigs:
    """Tests for the provider configuration builder."""

    def test_build_multiple_provider_configs(self) -> None:
        """Verify building configs for multiple providers."""
        # Define raw provider data as from YAML
        raw_providers: dict = {
            "openai": {
                "model": "gpt-4o",
                "max_tokens": 4096,
                "temperature": 0.0,
                "embedding_model": "text-embedding-3-small",
            },
            "anthropic": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2048,
                "temperature": 0.1,
            },
        }
        # Build the configs
        configs: dict = _build_provider_configs(raw_providers)
        # Verify two configs were created
        assert len(configs) == 2
        # Verify OpenAI config values
        assert configs["openai"].model == "gpt-4o"
        assert configs["openai"].embedding_model == "text-embedding-3-small"
        # Verify Anthropic config values
        assert configs["anthropic"].model == "claude-sonnet-4-20250514"
        assert configs["anthropic"].max_tokens == 2048

    def test_build_empty_provider_configs(self) -> None:
        """Verify empty input returns empty dictionary."""
        # Build with empty input
        configs: dict = _build_provider_configs({})
        # Verify empty result
        assert configs == {}

    def test_skip_invalid_provider_data(self) -> None:
        """Verify non-dict provider entries are skipped."""
        # Include an invalid entry alongside a valid one
        raw_providers: dict = {
            "openai": {"model": "gpt-4o", "max_tokens": 4096, "temperature": 0.0},
            "invalid": "not a dict",
        }
        # Build the configs
        configs: dict = _build_provider_configs(raw_providers)
        # Only valid entry should be present
        assert len(configs) == 1
        assert "openai" in configs


class TestLoadEnvKeys:
    """Tests for environment variable loading."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-123"}, clear=False)
    def test_load_openai_key_from_env(self) -> None:
        """Verify OpenAI API key is loaded from environment."""
        # Load environment keys
        keys: dict = _load_env_keys()
        # Verify OpenAI key was loaded
        assert keys["openai_api_key"] == "sk-test-123"

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_env_vars_return_empty_strings(self) -> None:
        """Verify missing environment variables return empty strings."""
        # Load with no env vars set
        keys: dict = _load_env_keys()
        # All keys should be empty strings
        assert keys["openai_api_key"] == ""
        assert keys["anthropic_api_key"] == ""
        assert keys["azure_openai_api_key"] == ""
        assert keys["azure_openai_endpoint"] == ""


class TestValidateProviderName:
    """Tests for provider name validation."""

    def test_valid_provider_names_pass(self) -> None:
        """Verify all valid provider names pass validation."""
        # Test each valid provider name
        _validate_provider_name("openai")
        _validate_provider_name("anthropic")
        _validate_provider_name("azure_openai")

    def test_invalid_provider_name_raises_error(self) -> None:
        """Verify invalid provider name raises ValueError."""
        # Test with an unrecognized provider name
        with pytest.raises(ValueError, match="Invalid provider"):
            _validate_provider_name("invalid_provider")

    def test_empty_provider_name_raises_error(self) -> None:
        """Verify empty string raises ValueError."""
        # Test with empty string
        with pytest.raises(ValueError, match="Invalid provider"):
            _validate_provider_name("")


class TestValidateThreshold:
    """Tests for confidence threshold validation."""

    def test_valid_threshold_values_pass(self) -> None:
        """Verify valid threshold values pass without error."""
        # Test boundary and mid-range values
        _validate_threshold(0.0)
        _validate_threshold(0.5)
        _validate_threshold(0.8)
        _validate_threshold(1.0)

    def test_threshold_below_zero_raises_error(self) -> None:
        """Verify negative threshold raises ValueError."""
        # Test with negative value
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            _validate_threshold(-0.1)

    def test_threshold_above_one_raises_error(self) -> None:
        """Verify threshold above 1.0 raises ValueError."""
        # Test with value exceeding maximum
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            _validate_threshold(1.1)


class TestValidateComparisonMethod:
    """Tests for comparison method validation."""

    def test_valid_methods_pass(self) -> None:
        """Verify valid comparison methods pass validation."""
        # Test each valid method name
        _validate_comparison_method("cosine_similarity")
        _validate_comparison_method("llm_judge")

    def test_invalid_method_raises_error(self) -> None:
        """Verify invalid method raises ValueError."""
        # Test with unrecognized method
        with pytest.raises(ValueError, match="Invalid comparison method"):
            _validate_comparison_method("invalid_method")
