"""Model registry for provider validation.

This module maintains a registry of supported models for each provider
with validation logic and helpful error messages.
Updated for August 2025 models.
"""

from __future__ import annotations

import os
import re
from re import Pattern

from .exceptions import ConfigurationError
from .models import ProviderName

# Model patterns for each provider (updated to latest models as of August 2025)
_MODEL_PATTERNS: dict[ProviderName, list[Pattern[str]]] = {
    "openai": [
        re.compile(r"^gpt-5.*$"),  # GPT-5 released August 2025
        re.compile(r"^gpt-4o.*$"),
        re.compile(r"^gpt-4.*$"),
        re.compile(r"^gpt-3\.5-turbo.*$"),
        re.compile(r"^o1-.*$"),  # Reasoning models
        re.compile(r"^o2-.*$"),  # Next-gen reasoning models
        re.compile(r"^text-.*$"),
        re.compile(r"^davinci-.*$"),
    ],
    "anthropic": [
        re.compile(r"^claude-opus-4.*$"),  # Claude Opus 4.1 (August 2025)
        re.compile(r"^claude-3.*$"),
        re.compile(r"^claude-2.*$"),
        re.compile(r"^claude-instant.*$"),
    ],
    "gemini": [
        re.compile(r"^gemini-2\.5-.*$"),  # Gemini 2.5 Pro/Flash (March 2025)
        re.compile(r"^gemini-2\.0-.*$"),
        re.compile(r"^gemini-1\.5-.*$"),
        re.compile(r"^gemini-.*$"),
        re.compile(r"^models/gemini-.*$"),
    ],
    "xai": [
        re.compile(r"^grok-3.*$"),  # Current available model
        re.compile(r"^grok-2.*$"),  # Vision model
        re.compile(r"^grok-.*$"),  # Future models
    ],
    "openrouter": [
        # OpenRouter supports many models, including proxied versions
        re.compile(r"^.*$"),  # Allow all for OpenRouter due to extensive catalog
    ],
}

# Example models for helpful error messages (updated for 2025)
_EXAMPLE_MODELS: dict[ProviderName, list[str]] = {
    "openai": ["gpt-5", "gpt-5-turbo", "gpt-4o", "gpt-4-turbo", "o2-preview"],
    "anthropic": ["claude-opus-4.1", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
    "gemini": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash-lite"],
    "xai": ["grok-3", "grok-2-vision", "grok-2"],
    "openrouter": [
        "meta-llama/llama-4-maverick",
        "meta-llama/llama-3.3-70b",
        "meta-llama/llama-3.2-90b",
        "openai/gpt-5",
        "anthropic/claude-opus-4.1",
    ],
}


def validate_model(provider: ProviderName, model: str) -> None:
    """Validate that a model is supported by the given provider.

    Args:
        provider: The provider name to validate against.
        model: The model identifier to validate.

    Raises:
        ConfigurationError: If the model is not supported by the provider.

    Note:
        Validation can be bypassed by setting ALLOW_UNLISTED_MODELS=1
        in the environment.
    """
    # Allow bypass for testing or new models
    if os.getenv("ALLOW_UNLISTED_MODELS") == "1":
        return

    patterns = _MODEL_PATTERNS.get(provider, [])
    if not patterns:
        raise ConfigurationError(
            f"Unknown provider: {provider}",
            provider=provider,
        )

    # Check if model matches any pattern
    for pattern in patterns:
        if pattern.match(model):
            return

    # Model not found, provide helpful error
    examples = _EXAMPLE_MODELS.get(provider, [])
    example_text = f" Examples: {', '.join(examples[:3])}" if examples else ""

    raise ConfigurationError(
        f"Unsupported model '{model}' for provider '{provider}'.{example_text}",
        provider=provider,
    )


def get_supported_providers() -> list[ProviderName]:
    """Get a list of all supported provider names.

    Returns:
        List of supported provider names.
    """
    return list(_MODEL_PATTERNS.keys())


def get_example_models(provider: ProviderName) -> list[str]:
    """Get example models for a given provider.

    Args:
        provider: The provider to get examples for.

    Returns:
        List of example model names.

    Raises:
        ConfigurationError: If the provider is not supported.
    """
    if provider not in _MODEL_PATTERNS:
        raise ConfigurationError(f"Unknown provider: {provider}")

    return _EXAMPLE_MODELS.get(provider, [])
