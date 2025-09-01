"""Token limit validation utilities for provider-specific constraints.

This module provides dynamic token limit validation based on provider and model
capabilities, replacing the static 32k token limit with flexible, accurate limits.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import ClassVar

from .models import ProviderName

logger = logging.getLogger(__name__)


class TokenLimitValidator:
    """Validates token limits based on provider and model specifications.

    This class provides dynamic token limit validation by querying provider-specific
    model metadata and caching the results for performance.
    """

    # Default fallback limits for unknown models (conservative estimates)
    DEFAULT_LIMITS: ClassVar[dict[str, int]] = {
        "openai": 16384,  # Most OpenAI models support at least 16k
        "openrouter": 16384,  # Conservative default for OpenRouter
        "gemini": 8192,  # Conservative default for Gemini
        "xai": 32768,  # xAI models generally support higher limits
        "anthropic": 16384,  # Conservative default for Anthropic
    }

    # Known model limits (cached to avoid repeated API calls)
    KNOWN_LIMITS: ClassVar[dict[str, dict[str, int]]] = {
        "openai": {
            # GPT-4 series
            "gpt-4": 8192,
            "gpt-4-turbo": 16384,
            "gpt-4o": 16384,
            "gpt-4o-mini": 16384,
            "gpt-4o-realtime": 4096,
            # O-series (reasoning models)
            "o1-preview": 32768,
            "o1-mini": 65536,
            "o3-mini": 65536,
            # GPT-OSS series
            "gpt-oss-20b": 131072,
            "gpt-oss-120b": 131072,
            # GPT-3.5 series
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            # Base models
            "davinci-002": 16384,
            "babbage-002": 16384,
        },
        "gemini": {
            # Gemini 1.5 series
            "gemini-1.5-pro": 8192,
            "gemini-1.5-flash": 8192,
            # Gemini 2.0 series
            "gemini-2.0-flash": 2048,
            "gemini-2.0-flash-preview": 8192,
            # Gemini 2.5 series
            "gemini-2.5-flash": 65536,
            "gemini-2.5-pro": 65536,
            "gemini-live-2.5-flash": 8192,
        },
        "xai": {
            # Grok series - xAI doesn't specify strict max_tokens limits
            # These are reasonable defaults based on their rate limits
            "grok-1": 32768,
            "grok-2": 32768,
            "grok-3": 32768,
            "grok-4": 32768,
            "grok-beta": 32768,
            "grok-code-fast-1": 32768,
        },
        "anthropic": {
            # Claude series
            "claude-3-opus": 16384,
            "claude-3-sonnet": 16384,
            "claude-3-haiku": 16384,
            "claude-3-5-sonnet": 16384,
            "claude-3-5-haiku": 16384,
        },
        "openrouter": {
            # OpenRouter limits depend on the underlying model
            # These are conservative defaults
            "default": 16384,
        },
    }

    @classmethod
    @lru_cache(maxsize=128)
    def get_max_tokens_limit(cls, provider: ProviderName, model: str, api_key: str | None = None) -> int:
        """Get the maximum token limit for a specific provider and model.

        Args:
            provider: The LLM provider name
            model: The specific model identifier
            api_key: Optional API key for querying model metadata

        Returns:
            Maximum token limit for the model

        Examples:
            >>> TokenLimitValidator.get_max_tokens_limit("openai", "gpt-4o")
            16384
            >>> TokenLimitValidator.get_max_tokens_limit("gemini", "gemini-2.5-flash")
            65536
        """
        # Normalize model name for lookup
        normalized_model = cls._normalize_model_name(model)

        # Check known limits first (fastest path)
        if provider in cls.KNOWN_LIMITS:
            provider_limits = cls.KNOWN_LIMITS[provider]
            if normalized_model in provider_limits:
                return provider_limits[normalized_model]

            # Try partial matches for versioned models
            for known_model, limit in provider_limits.items():
                if normalized_model.startswith(known_model):
                    return limit

        # Try to query provider-specific metadata (if API key provided)
        if api_key:
            try:
                dynamic_limit = cls._query_provider_metadata(provider, model, api_key)
                if dynamic_limit:
                    # Cache the result for future use
                    if provider not in cls.KNOWN_LIMITS:
                        cls.KNOWN_LIMITS[provider] = {}
                    cls.KNOWN_LIMITS[provider][normalized_model] = dynamic_limit
                    return dynamic_limit
            except Exception as e:
                logger.warning(f"Failed to query {provider} metadata for {model}: {e}")

        # Fall back to provider default
        default_limit = cls.DEFAULT_LIMITS.get(provider, 16384)
        logger.info(f"Using default limit {default_limit} for {provider}/{model}")
        return default_limit

    @classmethod
    def validate_max_tokens(
        cls, max_tokens: int | None, provider: ProviderName, model: str, api_key: str | None = None
    ) -> int | None:
        """Validate and potentially adjust max_tokens parameter.

        Args:
            max_tokens: Requested max_tokens value
            provider: The LLM provider name
            model: The specific model identifier
            api_key: Optional API key for querying model metadata

        Returns:
            Validated max_tokens value (may be adjusted if too high)

        Raises:
            ValueError: If max_tokens exceeds model capabilities
        """
        if max_tokens is None:
            return None

        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        # Get the actual limit for this provider/model
        model_limit = cls.get_max_tokens_limit(provider, model, api_key)

        if max_tokens > model_limit:
            raise ValueError(
                f"max_tokens ({max_tokens}) exceeds model limit ({model_limit}) "
                f"for {provider}/{model}. Maximum supported: {model_limit}"
            )

        return max_tokens

    @classmethod
    def _normalize_model_name(cls, model: str) -> str:
        """Normalize model name for consistent lookup.

        Args:
            model: Raw model name

        Returns:
            Normalized model name
        """
        # Remove provider prefixes (e.g., "openai/" from OpenRouter)
        if "/" in model:
            model = model.split("/", 1)[1]

        # Convert to lowercase for consistent matching
        return model.lower().strip()

    @classmethod
    def _query_provider_metadata(cls, provider: ProviderName, model: str, api_key: str) -> int | None:
        """Query provider-specific APIs for model metadata.

        Args:
            provider: The LLM provider name
            model: The specific model identifier
            api_key: API key for authentication

        Returns:
            Maximum token limit if available, None otherwise
        """
        try:
            if provider == "openai":
                return cls._query_openai_metadata(model, api_key)
            elif provider == "gemini":
                return cls._query_gemini_metadata(model, api_key)
            elif provider == "anthropic":
                return cls._query_anthropic_metadata(model, api_key)
            # xAI and OpenRouter don't provide reliable metadata APIs yet

        except Exception as e:
            logger.debug(f"Failed to query {provider} metadata: {e}")

        return None

    @classmethod
    def _query_openai_metadata(cls, model: str, api_key: str) -> int | None:
        """Query OpenAI API for model metadata."""
        try:
            import openai

            client = openai.OpenAI(api_key=api_key)

            # OpenAI doesn't provide a direct endpoint for model limits
            # We rely on known limits for now
            return None

        except ImportError:
            logger.debug("OpenAI library not available for metadata query")
            return None

    @classmethod
    def _query_gemini_metadata(cls, model: str, api_key: str) -> int | None:
        """Query Gemini API for model metadata."""
        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            model_info = client.models.get(model=model)

            if hasattr(model_info, "output_token_limit"):
                return model_info.output_token_limit

        except ImportError:
            logger.debug("Google GenAI library not available for metadata query")
        except Exception as e:
            logger.debug(f"Failed to query Gemini metadata: {e}")

        return None

    @classmethod
    def _query_anthropic_metadata(cls, model: str, api_key: str) -> int | None:
        """Query Anthropic API for model metadata."""
        try:
            import anthropic

            # Anthropic doesn't provide a direct metadata endpoint
            # We rely on known limits for now
            return None

        except ImportError:
            logger.debug("Anthropic library not available for metadata query")
            return None
