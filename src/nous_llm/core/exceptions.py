"""Exception hierarchy for the Universal LLM Wrapper.

This module defines all custom exceptions with consistent error handling
and provider context preservation.
"""

from __future__ import annotations

from typing import Any


class UnillmError(Exception):
    """Base exception for all Universal LLM Wrapper errors.

    Args:
        message: Human-readable error description.
        provider: The provider where the error occurred, if applicable.
        raw: Raw error data from the provider for debugging.
    """

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.raw = raw or {}

    def __str__(self) -> str:
        """Return a formatted error message with provider context."""
        if self.provider:
            return f"[{self.provider}] {self.message}"
        return self.message


class ConfigurationError(UnillmError):
    """Raised when there are configuration issues.

    Examples:
        - Invalid provider name
        - Missing API keys
        - Unsupported model for provider
        - Invalid base URL format
    """

    pass


class AuthError(UnillmError):
    """Raised when authentication fails.

    Examples:
        - Invalid API key (401)
        - Insufficient permissions (403)
        - Expired token
    """

    pass


class RateLimitError(UnillmError):
    """Raised when rate limits are exceeded.

    Examples:
        - Too many requests per minute
        - Quota exceeded
        - Concurrent request limit reached
    """

    pass


class ProviderError(UnillmError):
    """Raised when the provider returns an error.

    Examples:
        - Server errors (5xx)
        - Service unavailable
        - Model overloaded
        - Invalid request format
    """

    pass


class ValidationError(UnillmError):
    """Raised when input validation fails.

    Examples:
        - Invalid parameter values
        - Missing required fields
        - Type mismatches
        - Constraint violations
    """

    pass
