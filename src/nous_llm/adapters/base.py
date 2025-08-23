"""Base adapter with common functionality for all providers.

This module provides shared utilities and error handling logic
that can be reused across different provider adapters.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..core.exceptions import AuthError, ProviderError, RateLimitError
from ..core.models import ProviderConfig, Usage


class BaseAdapter:
    """Base class providing common functionality for provider adapters.

    This class handles common tasks like API key resolution, HTTP client
    management, error mapping, and retry logic.
    """

    def __init__(self) -> None:
        """Initialize the base adapter."""
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    def _get_api_key(self, config: ProviderConfig, env_var: str) -> str:
        """Get API key from config or environment.

        Args:
            config: Provider configuration that may contain an API key.
            env_var: Environment variable name to check if config key is None.

        Returns:
            The API key to use for authentication.

        Raises:
            AuthError: If no API key is found.
        """
        api_key = config.api_key or os.getenv(env_var)
        if not api_key:
            raise AuthError(
                f"No API key found. Provide via config.api_key or {env_var} environment variable.",
                provider=config.provider,
            )
        return api_key

    def _get_sync_client(self, timeout: float = 30.0) -> httpx.Client:
        """Get or create a synchronous HTTP client.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            Configured HTTP client instance.
        """
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                timeout=timeout,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._sync_client

    def _get_async_client(self, timeout: float = 30.0) -> httpx.AsyncClient:
        """Get or create an asynchronous HTTP client.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            Configured async HTTP client instance.
        """
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._async_client

    def _map_http_error(self, response: httpx.Response, provider: str) -> Exception:
        """Map HTTP response codes to appropriate exceptions.

        Args:
            response: The HTTP response that contains an error.
            provider: Name of the provider for error context.

        Returns:
            Appropriate exception instance for the error.
        """
        try:
            error_data = response.json()
        except Exception:
            error_data = {"error": response.text}

        if response.status_code == 401:
            return AuthError(
                "Authentication failed. Check your API key.",
                provider=provider,
                raw=error_data,
            )
        elif response.status_code == 403:
            return AuthError(
                "Access forbidden. Check your permissions.",
                provider=provider,
                raw=error_data,
            )
        elif response.status_code == 429:
            return RateLimitError(
                "Rate limit exceeded. Please try again later.",
                provider=provider,
                raw=error_data,
            )
        elif response.status_code >= 500:
            return ProviderError(
                f"Server error ({response.status_code}). Please try again.",
                provider=provider,
                raw=error_data,
            )
        else:
            return ProviderError(
                f"Request failed with status {response.status_code}",
                provider=provider,
                raw=error_data,
            )

    def _extract_usage(self, raw_usage: dict[str, Any] | None) -> Usage | None:
        """Extract usage information from provider response.

        Args:
            raw_usage: Raw usage data from provider response.

        Returns:
            Standardized usage information, or None if not available.
        """
        if not raw_usage:
            return None

        # Handle different provider usage formats
        return Usage(
            input_tokens=raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens"),
            output_tokens=raw_usage.get("completion_tokens") or raw_usage.get("output_tokens"),
            total_tokens=raw_usage.get("total_tokens"),
        )

    @retry(
        retry=retry_if_exception_type((RateLimitError, ProviderError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def _make_request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make HTTP request with automatic retry logic.

        Args:
            method: HTTP method (GET, POST, etc.).
            url: Request URL.
            **kwargs: Additional arguments for the request.

        Returns:
            HTTP response object.

        Raises:
            Various exceptions based on response status.
        """
        client = self._get_sync_client()
        response = client.request(method, url, **kwargs)

        if not response.is_success:
            raise self._map_http_error(response, "unknown")

        return response

    @retry(
        retry=retry_if_exception_type((RateLimitError, ProviderError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _make_async_request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make async HTTP request with automatic retry logic.

        Args:
            method: HTTP method (GET, POST, etc.).
            url: Request URL.
            **kwargs: Additional arguments for the request.

        Returns:
            HTTP response object.

        Raises:
            Various exceptions based on response status.
        """
        client = self._get_async_client()
        response = await client.request(method, url, **kwargs)

        if not response.is_success:
            raise self._map_http_error(response, "unknown")

        return response

    def close(self) -> None:
        """Close HTTP clients and clean up resources."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
        if self._async_client:
            # Note: This is sync close, async close should be handled in context manager
            pass

    async def aclose(self) -> None:
        """Asynchronously close HTTP clients and clean up resources."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
