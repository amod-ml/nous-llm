"""Nous LLM - Intelligent No Frills LLM Router.

This package provides a consistent interface for interacting with OpenAI, Gemini,
OpenRouter, xAI, and Anthropic APIs through typed models and RORO patterns.
"""

from __future__ import annotations

from .core.client import LLMClient, NousLLM
from .core.exceptions import (
    AuthError,
    ConfigurationError,
    ProviderError,
    RateLimitError,
    ValidationError,
)
from .core.interface import agenenerate, generate
from .core.models import (
    GenParams,
    LLMResponse,
    Prompt,
    ProviderConfig,
    ProviderName,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    "AuthError",
    "ConfigurationError",
    "GenParams",
    "LLMClient",
    "LLMResponse",
    "NousLLM",
    "Prompt",
    "ProviderConfig",
    "ProviderError",
    "ProviderName",
    "RateLimitError",
    "Usage",
    "ValidationError",
    "agenenerate",
    "generate",
    "__version__",
]
