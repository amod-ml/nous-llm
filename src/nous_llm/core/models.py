"""Core data models for the Universal LLM Wrapper.

This module defines all the Pydantic models used throughout the library,
following strict typing and validation requirements.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Type alias for supported providers
ProviderName = Literal["openai", "openrouter", "gemini", "xai", "anthropic"]


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider.

    Args:
        provider: The name of the LLM provider to use.
        model: The specific model identifier for the provider.
        api_key: API key for authentication. If None, will attempt to read from environment.
        base_url: Custom base URL for the provider (useful for proxies or OpenRouter).

    Examples:
        >>> config = ProviderConfig(
        ...     provider="openai",
        ...     model="gpt-4",
        ...     api_key="sk-..."
        ... )
    """

    provider: ProviderName
    model: str = Field(min_length=1, description="Model identifier")
    api_key: str | None = Field(default=None, description="API key for authentication")
    base_url: str | None = Field(default=None, description="Custom base URL")

    model_config = {"frozen": True}


class Prompt(BaseModel):
    """Input prompt structure following the instructions + input pattern.

    Args:
        instructions: System-level instructions or context for the model.
        input: The actual user input or query to process.

    Examples:
        >>> prompt = Prompt(
        ...     instructions="You are a helpful assistant.",
        ...     input="What is the capital of France?"
        ... )
    """

    instructions: str = Field(min_length=1, description="System instructions")
    input: str = Field(min_length=1, description="User input")

    model_config = {"frozen": True}


class GenParams(BaseModel):
    """Generation parameters for controlling LLM output.

    Args:
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 to 2.0).
        top_p: Nucleus sampling parameter (0.0 to 1.0).
        stop: List of stop sequences to halt generation.
        extra: Provider-specific parameters as key-value pairs.

    Examples:
        >>> params = GenParams(
        ...     max_tokens=512,
        ...     temperature=0.7,
        ...     extra={"reasoning": True}  # OpenAI-specific
        ... )
    """

    max_tokens: int = Field(default=512, ge=1, le=32000, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling")
    stop: list[str] | None = Field(default=None, description="Stop sequences")
    extra: dict[str, Any] = Field(default_factory=dict, description="Provider-specific parameters")

    model_config = {"frozen": True}


class Usage(BaseModel):
    """Token usage information from the LLM response.

    Args:
        input_tokens: Number of tokens in the input prompt.
        output_tokens: Number of tokens in the generated output.
        total_tokens: Total tokens used (input + output).

    Note:
        Some providers may not return all usage fields. Fields will be None
        if not provided by the provider.
    """

    input_tokens: int | None = Field(default=None, ge=0, description="Input token count")
    output_tokens: int | None = Field(default=None, ge=0, description="Output token count")
    total_tokens: int | None = Field(default=None, ge=0, description="Total token count")

    model_config = {"frozen": True}


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider.

    Args:
        provider: The provider that generated this response.
        model: The specific model used for generation.
        text: The generated text content.
        usage: Token usage information, if available.
        raw: Raw response data from the provider for debugging.

    Examples:
        >>> response = LLMResponse(
        ...     provider="openai",
        ...     model="gpt-4",
        ...     text="Paris is the capital of France.",
        ...     usage=Usage(input_tokens=10, output_tokens=8, total_tokens=18)
        ... )
    """

    provider: ProviderName
    model: str
    text: str = Field(description="Generated text content")
    usage: Usage | None = Field(default=None, description="Token usage information")
    raw: dict[str, Any] | None = Field(default=None, description="Raw provider response")

    model_config = {"frozen": True}
