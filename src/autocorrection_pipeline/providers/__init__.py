"""
LLM providers subpackage.

Contains the abstract provider interface, concrete implementations
for OpenAI, Anthropic, and Azure OpenAI, and the factory class
for provider instantiation based on configuration.
"""

# Re-export primary provider interfaces for convenient imports
from autocorrection_pipeline.providers.base import BaseLLMProvider
from autocorrection_pipeline.providers.factory import ProviderFactory

# Define public API for this subpackage
__all__ = ["BaseLLMProvider", "ProviderFactory"]
