"""
Tests for core data models.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nous_llm import GenParams, LLMResponse, Prompt, ProviderConfig, Usage


class TestProviderConfig:
    """Tests for ProviderConfig model."""

    def test_valid_config(self) -> None:
        """Test creating a valid provider configuration."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            base_url="https://custom.api.com",
        )

        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.api_key == "test-key"
        assert config.base_url == "https://custom.api.com"

    def test_minimal_config(self) -> None:
        """Test creating minimal configuration."""
        config = ProviderConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")

        assert config.provider == "anthropic"
        assert config.model == "claude-3-5-sonnet-20241022"
        assert config.api_key is None
        assert config.base_url is None

    def test_invalid_provider(self) -> None:
        """Test validation fails for invalid provider."""
        with pytest.raises(ValidationError):
            ProviderConfig(provider="invalid", model="test-model")

    def test_empty_model(self) -> None:
        """Test validation fails for empty model."""
        with pytest.raises(ValidationError):
            ProviderConfig(provider="openai", model="")

    def test_config_is_frozen(self) -> None:
        """Test that config is immutable."""
        config = ProviderConfig(provider="openai", model="gpt-4o")

        with pytest.raises(ValidationError):
            config.model = "different-model"


class TestPrompt:
    """Tests for Prompt model."""

    def test_valid_prompt(self) -> None:
        """Test creating a valid prompt."""
        prompt = Prompt(
            instructions="You are a helpful assistant.",
            input="What is the capital of France?",
        )

        assert prompt.instructions == "You are a helpful assistant."
        assert prompt.input == "What is the capital of France?"

    def test_empty_instructions(self) -> None:
        """Test validation fails for empty instructions."""
        with pytest.raises(ValidationError):
            Prompt(instructions="", input="test")

    def test_empty_input(self) -> None:
        """Test validation fails for empty input."""
        with pytest.raises(ValidationError):
            Prompt(instructions="test", input="")

    def test_prompt_is_frozen(self) -> None:
        """Test that prompt is immutable."""
        prompt = Prompt(instructions="test", input="test input")

        with pytest.raises(ValidationError):
            prompt.input = "different input"


class TestGenParams:
    """Tests for GenParams model."""

    def test_default_params(self) -> None:
        """Test default parameter values."""
        params = GenParams()

        assert params.max_tokens == 512
        assert params.temperature == 0.7
        assert params.top_p is None
        assert params.stop is None
        assert params.extra == {}

    def test_custom_params(self) -> None:
        """Test custom parameter values."""
        params = GenParams(
            max_tokens=1000,
            temperature=0.5,
            top_p=0.9,
            stop=["END", "STOP"],
            extra={"reasoning": True},
        )

        assert params.max_tokens == 1000
        assert params.temperature == 0.5
        assert params.top_p == 0.9
        assert params.stop == ["END", "STOP"]
        assert params.extra == {"reasoning": True}

    def test_invalid_max_tokens(self) -> None:
        """Test validation for invalid max_tokens."""
        with pytest.raises(ValidationError):
            GenParams(max_tokens=0)

        with pytest.raises(ValidationError):
            GenParams(max_tokens=50000)  # Too high

    def test_invalid_temperature(self) -> None:
        """Test validation for invalid temperature."""
        with pytest.raises(ValidationError):
            GenParams(temperature=-0.1)

        with pytest.raises(ValidationError):
            GenParams(temperature=2.1)

    def test_invalid_top_p(self) -> None:
        """Test validation for invalid top_p."""
        with pytest.raises(ValidationError):
            GenParams(top_p=-0.1)

        with pytest.raises(ValidationError):
            GenParams(top_p=1.1)


class TestUsage:
    """Tests for Usage model."""

    def test_complete_usage(self) -> None:
        """Test usage with all fields."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_partial_usage(self) -> None:
        """Test usage with only some fields."""
        usage = Usage(input_tokens=100)

        assert usage.input_tokens == 100
        assert usage.output_tokens is None
        assert usage.total_tokens is None

    def test_empty_usage(self) -> None:
        """Test empty usage."""
        usage = Usage()

        assert usage.input_tokens is None
        assert usage.output_tokens is None
        assert usage.total_tokens is None

    def test_negative_tokens(self) -> None:
        """Test validation fails for negative token counts."""
        with pytest.raises(ValidationError):
            Usage(input_tokens=-1)


class TestLLMResponse:
    """Tests for LLMResponse model."""

    def test_complete_response(self) -> None:
        """Test response with all fields."""
        usage = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        response = LLMResponse(
            provider="openai",
            model="gpt-4o",
            text="Paris is the capital of France.",
            usage=usage,
            raw={"id": "test-123"},
        )

        assert response.provider == "openai"
        assert response.model == "gpt-4o"
        assert response.text == "Paris is the capital of France."
        assert response.usage == usage
        assert response.raw == {"id": "test-123"}

    def test_minimal_response(self) -> None:
        """Test response with required fields only."""
        response = LLMResponse(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            text="Hello world!",
        )

        assert response.provider == "anthropic"
        assert response.model == "claude-3-5-sonnet-20241022"
        assert response.text == "Hello world!"
        assert response.usage is None
        assert response.raw is None

    def test_response_is_frozen(self) -> None:
        """Test that response is immutable."""
        response = LLMResponse(
            provider="openai",
            model="gpt-4o",
            text="test",
        )

        with pytest.raises(ValidationError):
            response.text = "different text"
