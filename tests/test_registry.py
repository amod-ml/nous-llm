"""
Tests for model registry and validation.
"""

from __future__ import annotations

import os

import pytest

from nous_llm.core.exceptions import ConfigurationError
from nous_llm.core.registry import (
    get_example_models,
    get_supported_providers,
    validate_model,
)


class TestModelValidation:
    """Tests for model validation logic."""

    def test_valid_openai_models(self) -> None:
        """Test validation of valid OpenAI models."""
        valid_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "o1-preview",
            "o1-mini",
        ]

        for model in valid_models:
            validate_model("openai", model)  # Should not raise

    def test_valid_anthropic_models(self) -> None:
        """Test validation of valid Anthropic models."""
        valid_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-instant-1.2",
        ]

        for model in valid_models:
            validate_model("anthropic", model)  # Should not raise

    def test_valid_gemini_models(self) -> None:
        """Test validation of valid Gemini models."""
        valid_models = [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "models/gemini-pro",
        ]

        for model in valid_models:
            validate_model("gemini", model)  # Should not raise

    def test_valid_xai_models(self) -> None:
        """Test validation of valid xAI models."""
        valid_models = [
            "grok-beta",
            "grok-2",
        ]

        for model in valid_models:
            validate_model("xai", model)  # Should not raise

    def test_openrouter_allows_all(self) -> None:
        """Test that OpenRouter allows any model."""
        models = [
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "meta-llama/llama-3.1-405b",
            "random/model-name",
        ]

        for model in models:
            validate_model("openrouter", model)  # Should not raise

    def test_invalid_model_raises_error(self) -> None:
        """Test that invalid models raise ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            validate_model("openai", "invalid-model")

        assert "Unsupported model 'invalid-model' for provider 'openai'" in str(exc_info.value)
        assert "Examples:" in str(exc_info.value)

    def test_invalid_provider_raises_error(self) -> None:
        """Test that invalid providers raise ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            validate_model("invalid-provider", "some-model")

        assert "Unknown provider: invalid-provider" in str(exc_info.value)

    def test_bypass_validation_with_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that validation can be bypassed with environment variable."""
        monkeypatch.setenv("ALLOW_UNLISTED_MODELS", "1")

        # Should not raise even for invalid model
        validate_model("openai", "completely-invalid-model")

    def test_bypass_disabled_by_default(self) -> None:
        """Test that bypass is disabled by default."""
        # Ensure env var is not set
        if "ALLOW_UNLISTED_MODELS" in os.environ:
            del os.environ["ALLOW_UNLISTED_MODELS"]

        with pytest.raises(ConfigurationError):
            validate_model("openai", "invalid-model")


class TestRegistryUtilities:
    """Tests for registry utility functions."""

    def test_get_supported_providers(self) -> None:
        """Test getting list of supported providers."""
        providers = get_supported_providers()

        expected_providers = ["openai", "anthropic", "gemini", "xai", "openrouter"]
        assert set(providers) == set(expected_providers)

    def test_get_example_models_openai(self) -> None:
        """Test getting example models for OpenAI."""
        examples = get_example_models("openai")

        assert isinstance(examples, list)
        assert len(examples) > 0
        assert "gpt-4o" in examples

    def test_get_example_models_anthropic(self) -> None:
        """Test getting example models for Anthropic."""
        examples = get_example_models("anthropic")

        assert isinstance(examples, list)
        assert len(examples) > 0
        assert any("claude" in model for model in examples)

    def test_get_example_models_invalid_provider(self) -> None:
        """Test getting examples for invalid provider raises error."""
        with pytest.raises(ConfigurationError):
            get_example_models("invalid-provider")
