"""Convenience client class for the Universal LLM Wrapper.

This module provides an optional client class that can cache configuration
and provide a more object-oriented interface while maintaining the underlying
RORO pattern.
"""

from __future__ import annotations

from .interface import agenenerate, generate
from .models import GenParams, LLMResponse, Prompt, ProviderConfig


class LLMClient:
    """Convenience client for LLM operations.

    This class provides a convenient wrapper around the functional interface,
    allowing you to cache provider configuration and reuse it for multiple
    generation requests.

    Examples:
        >>> from unillm import LLMClient, ProviderConfig, Prompt, GenParams
        >>>
        >>> # Create client with configuration
        >>> client = LLMClient(ProviderConfig(
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     api_key="your-api-key"
        ... ))
        >>>
        >>> # Generate text using the cached config
        >>> prompt = Prompt(
        ...     instructions="You are a helpful assistant.",
        ...     input="What is machine learning?"
        ... )
        >>>
        >>> response = client.generate(prompt)
        >>> print(response.text)
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize the client with provider configuration.

        Args:
            config: Provider configuration to use for all requests.
        """
        self.config = config

    def generate(
        self,
        prompt: Prompt,
        params: GenParams | None = None,
    ) -> LLMResponse:
        """Generate text using the client's configuration synchronously.

        Args:
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
        """
        return generate(self.config, prompt, params)

    async def agenenerate(
        self,
        prompt: Prompt,
        params: GenParams | None = None,
    ) -> LLMResponse:
        """Generate text using the client's configuration asynchronously.

        Args:
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
        """
        return await agenenerate(self.config, prompt, params)

    def update_config(self, **kwargs) -> LLMClient:
        """Create a new client with updated configuration.

        Args:
            **kwargs: Configuration fields to update.

        Returns:
            New client instance with updated configuration.

        Examples:
            >>> original_client = LLMClient(ProviderConfig(
            ...     provider="openai",
            ...     model="gpt-4o",
            ...     api_key="your-api-key"
            ... ))
            >>>
            >>> # Create new client with different model
            >>> new_client = original_client.update_config(model="gpt-3.5-turbo")
        """
        # Create a new config with updated fields
        config_dict = self.config.model_dump()
        config_dict.update(kwargs)
        new_config = ProviderConfig.model_validate(config_dict)

        return LLMClient(new_config)

    def __repr__(self) -> str:
        """Return string representation of the client."""
        return f"LLMClient(provider={self.config.provider}, model={self.config.model})"


# Alias for backward compatibility and brand naming
NousLLM = LLMClient
