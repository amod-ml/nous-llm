"""Adapter factory for getting the correct provider adapter.

This module provides a centralized way to get adapter instances
for different LLM providers.
"""

from __future__ import annotations

from ..adapters.anthropic_adapter import AnthropicAdapter
from ..adapters.gemini_adapter import GeminiAdapter
from ..adapters.openai_adapter import OpenAIAdapter
from ..adapters.openrouter_adapter import OpenRouterAdapter
from ..adapters.protocol import AdapterProtocol
from ..adapters.xai_adapter import XAIAdapter
from .exceptions import ConfigurationError
from .models import ProviderName

# Registry of adapter classes
_ADAPTER_REGISTRY: dict[ProviderName, type[AdapterProtocol]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
    "xai": XAIAdapter,
    "openrouter": OpenRouterAdapter,
}

# Cache for adapter instances
_ADAPTER_CACHE: dict[ProviderName, AdapterProtocol] = {}


def get_adapter(provider: ProviderName) -> AdapterProtocol:
    """Get an adapter instance for the specified provider.

    Args:
        provider: The name of the provider to get an adapter for.

    Returns:
        Adapter instance for the provider.

    Raises:
        ConfigurationError: If the provider is not supported.

    Note:
        Adapter instances are cached for reuse to avoid unnecessary
        initialization overhead.
    """
    if provider not in _ADAPTER_REGISTRY:
        supported = list(_ADAPTER_REGISTRY.keys())
        raise ConfigurationError(
            f"Unsupported provider: {provider}. Supported providers: {supported}",
            provider=provider,
        )

    # Return cached instance if available
    if provider in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[provider]

    # Create new instance and cache it
    adapter_class = _ADAPTER_REGISTRY[provider]
    adapter_instance = adapter_class()
    _ADAPTER_CACHE[provider] = adapter_instance

    return adapter_instance


def register_adapter(provider: ProviderName, adapter_class: type[AdapterProtocol]) -> None:
    """Register a custom adapter for a provider.

    Args:
        provider: The provider name to register the adapter for.
        adapter_class: The adapter class to register.

    Note:
        This allows for custom provider implementations or overriding
        existing adapters for testing purposes.
    """
    _ADAPTER_REGISTRY[provider] = adapter_class

    # Clear cached instance if it exists
    _ADAPTER_CACHE.pop(provider, None)


def clear_adapter_cache() -> None:
    """Clear the adapter cache.

    This is useful for testing or when you want to ensure fresh
    adapter instances are created.
    """
    _ADAPTER_CACHE.clear()


def get_supported_providers() -> list[ProviderName]:
    """Get a list of all supported provider names.

    Returns:
        List of supported provider names.
    """
    return list(_ADAPTER_REGISTRY.keys())
