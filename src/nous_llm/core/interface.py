"""Main interface functions for the Universal LLM Wrapper.

This module provides the primary functional interface following RORO
(Receive Object, Return Object) patterns for consistent usage.
"""

from __future__ import annotations

from .adapters import get_adapter
from .models import GenParams, LLMResponse, Prompt, ProviderConfig
from .registry import validate_model


def generate(
    config: ProviderConfig,
    prompt: Prompt,
    params: GenParams | None = None,
) -> LLMResponse:
    """Generate text using the specified LLM provider synchronously.

    This is the main synchronous interface for text generation. It validates
    the configuration, selects the appropriate adapter, and returns a
    standardized response.

    Args:
        config: Provider configuration including provider name, model, and API key.
        prompt: Input prompt containing instructions and user input.
        params: Optional generation parameters. Uses defaults if not provided.

    Returns:
        Standardized response containing generated text and metadata.

    Raises:
        ConfigurationError: If the provider or model configuration is invalid.
        AuthError: If authentication with the provider fails.
        RateLimitError: If rate limits are exceeded.
        ProviderError: If the provider returns an error.
        ValidationError: If input validation fails.

    Examples:
        >>> from unillm import generate, ProviderConfig, Prompt, GenParams
        >>>
        >>> config = ProviderConfig(
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     api_key="your-api-key"
        ... )
        >>>
        >>> prompt = Prompt(
        ...     instructions="You are a helpful assistant.",
        ...     input="What is the capital of France?"
        ... )
        >>>
        >>> params = GenParams(temperature=0.7, max_tokens=100)
        >>>
        >>> response = generate(config, prompt, params)
        >>> print(response.text)
        "Paris is the capital of France."
    """
    # Use default parameters if none provided
    if params is None:
        params = GenParams()

    # Validate model compatibility with provider
    validate_model(config.provider, config.model)

    # Get appropriate adapter and generate
    adapter = get_adapter(config.provider)
    return adapter.generate(config, prompt, params)


async def agenenerate(
    config: ProviderConfig,
    prompt: Prompt,
    params: GenParams | None = None,
) -> LLMResponse:
    """Generate text using the specified LLM provider asynchronously.

    This is the main asynchronous interface for text generation. It validates
    the configuration, selects the appropriate adapter, and returns a
    standardized response.

    Args:
        config: Provider configuration including provider name, model, and API key.
        prompt: Input prompt containing instructions and user input.
        params: Optional generation parameters. Uses defaults if not provided.

    Returns:
        Standardized response containing generated text and metadata.

    Raises:
        ConfigurationError: If the provider or model configuration is invalid.
        AuthError: If authentication with the provider fails.
        RateLimitError: If rate limits are exceeded.
        ProviderError: If the provider returns an error.
        ValidationError: If input validation fails.

    Examples:
        >>> import asyncio
        >>> from unillm import agenenerate, ProviderConfig, Prompt, GenParams
        >>>
        >>> async def main():
        ...     config = ProviderConfig(
        ...         provider="anthropic",
        ...         model="claude-3-5-sonnet-20241022",
        ...         api_key="your-api-key"
        ...     )
        ...
        ...     prompt = Prompt(
        ...         instructions="You are a helpful assistant.",
        ...         input="Explain quantum computing in simple terms."
        ...     )
        ...
        ...     response = await agenenerate(config, prompt)
        ...     print(response.text)
        >>>
        >>> asyncio.run(main())
    """
    # Use default parameters if none provided
    if params is None:
        params = GenParams()

    # Validate model compatibility with provider
    validate_model(config.provider, config.model)

    # Get appropriate adapter and generate
    adapter = get_adapter(config.provider)
    return await adapter.agenenerate(config, prompt, params)
