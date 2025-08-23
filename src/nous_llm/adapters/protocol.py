"""Protocol definition for provider adapters.

This module defines the interface that all provider adapters must implement
to ensure consistent behavior across different LLM services.
"""

from __future__ import annotations

from typing import Protocol

from ..core.models import GenParams, LLMResponse, Prompt, ProviderConfig


class AdapterProtocol(Protocol):
    """Protocol that all provider adapters must implement.

    This ensures a consistent interface across all LLM providers while
    allowing for provider-specific optimizations and features.
    """

    def generate(
        self,
        config: ProviderConfig,
        prompt: Prompt,
        params: GenParams,
    ) -> LLMResponse:
        """Generate text synchronously using the provider's API.

        Args:
            config: Provider configuration including model and API key.
            prompt: The input prompt with instructions and user input.
            params: Generation parameters like temperature and max_tokens.

        Returns:
            Standardized response containing the generated text and metadata.

        Raises:
            AuthError: When authentication fails.
            RateLimitError: When rate limits are exceeded.
            ProviderError: When the provider returns an error.
            ValidationError: When input validation fails.
        """
        ...

    async def agenenerate(
        self,
        config: ProviderConfig,
        prompt: Prompt,
        params: GenParams,
    ) -> LLMResponse:
        """Generate text asynchronously using the provider's API.

        Args:
            config: Provider configuration including model and API key.
            prompt: The input prompt with instructions and user input.
            params: Generation parameters like temperature and max_tokens.

        Returns:
            Standardized response containing the generated text and metadata.

        Raises:
            AuthError: When authentication fails.
            RateLimitError: When rate limits are exceeded.
            ProviderError: When the provider returns an error.
            ValidationError: When input validation fails.
        """
        ...
