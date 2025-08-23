"""xAI adapter using the xAI SDK.

This adapter implements the xAI provider using the official
xai-sdk for Grok models with the correct API patterns.
"""

from __future__ import annotations

from typing import Any

import anyio

from ..core.exceptions import AuthError, ProviderError, RateLimitError
from ..core.models import GenParams, LLMResponse, Prompt, ProviderConfig, Usage
from .base import BaseAdapter


class XAIAdapter(BaseAdapter):
    """Adapter for xAI Grok models using the xAI SDK."""

    def __init__(self) -> None:
        """Initialize the xAI adapter."""
        super().__init__()
        self._client = None
        self._async_client = None

    def _get_client(self, api_key: str):
        """Get or create synchronous xAI client.

        Args:
            api_key: xAI API key.

        Returns:
            xAI client instance.
        """
        if self._client is None:
            try:
                import xai_sdk

                self._client = xai_sdk.Client(api_key=api_key)
            except ImportError as e:
                raise ProviderError(
                    "xai-sdk package not installed. Install with: pip install xai-sdk",
                    provider="xai",
                ) from e
        return self._client

    def _get_async_client(self, api_key: str):
        """Get or create asynchronous xAI client.

        Args:
            api_key: xAI API key.

        Returns:
            Async xAI client instance.
        """
        if self._async_client is None:
            try:
                import xai_sdk

                self._async_client = xai_sdk.AsyncClient(api_key=api_key)
            except ImportError as e:
                raise ProviderError(
                    "xai-sdk package not installed. Install with: pip install xai-sdk",
                    provider="xai",
                ) from e
        return self._async_client

    def _create_chat_messages(self, prompt: Prompt):
        """Create chat messages using xAI SDK helper functions.

        Args:
            prompt: Input prompt with instructions and user input.

        Returns:
            List of message objects using xAI SDK format.
        """
        try:
            from xai_sdk.chat import system, user

            return [
                system(prompt.instructions),
                user(prompt.input),
            ]
        except ImportError as e:
            raise ProviderError(
                "xai-sdk package not installed or outdated. Install with: pip install xai-sdk",
                provider="xai",
            ) from e

    def _parse_response(
        self,
        response: Any,
        config: ProviderConfig,
    ) -> LLMResponse:
        """Parse xAI response into standardized format.

        Args:
            response: Raw response from xAI API.
            config: Provider configuration.

        Returns:
            Standardized LLM response.
        """
        try:
            # Extract text content from the response
            text_content = ""
            if hasattr(response, "content"):
                text_content = response.content or ""
            elif hasattr(response, "text"):
                text_content = response.text or ""

            # xAI SDK may not provide detailed usage information in all responses
            usage = None
            if hasattr(response, "usage"):
                usage_data = response.usage
                usage = Usage(
                    input_tokens=getattr(usage_data, "prompt_tokens", None),
                    output_tokens=getattr(usage_data, "completion_tokens", None),
                    total_tokens=getattr(usage_data, "total_tokens", None),
                )

            # Build raw data for debugging
            raw_data = {}
            if hasattr(response, "proto"):
                # xAI SDK provides access to the raw protobuf object
                raw_data["proto"] = str(response.proto)

            # Add any other accessible attributes
            for attr in ["id", "model", "created", "finish_reason"]:
                if hasattr(response, attr):
                    raw_data[attr] = getattr(response, attr)

            return LLMResponse(
                provider="xai",
                model=config.model,
                text=text_content,
                usage=usage,
                raw=raw_data,
            )

        except Exception as e:
            raise ProviderError(
                f"Failed to parse xAI response: {e}",
                provider="xai",
                raw={"error": str(e)},
            ) from e

    def _handle_xai_exception(self, e: Exception) -> Exception:
        """Map xAI exceptions to standard exceptions.

        Args:
            e: Original exception from xAI API.

        Returns:
            Mapped exception.
        """
        error_str = str(e).lower()

        # Check for gRPC status codes mentioned in the documentation
        if "unauthenticated" in error_str or "invalid api key" in error_str:
            return AuthError(
                "xAI authentication failed. Check your API key.",
                provider="xai",
                raw={"error": str(e)},
            )
        elif "resource_exhausted" in error_str or "rate limit" in error_str or "quota" in error_str:
            return RateLimitError(
                "xAI rate limit or quota exceeded.",
                provider="xai",
                raw={"error": str(e)},
            )
        elif "not_found" in error_str:
            return ProviderError(
                "xAI model not found. Check your model name.",
                provider="xai",
                raw={"error": str(e)},
            )
        elif "permission_denied" in error_str:
            return AuthError(
                "xAI permission denied. Check your API key permissions.",
                provider="xai",
                raw={"error": str(e)},
            )
        else:
            return ProviderError(
                f"xAI API error: {e}",
                provider="xai",
                raw={"error": str(e)},
            )

    def generate(
        self,
        config: ProviderConfig,
        prompt: Prompt,
        params: GenParams,
    ) -> LLMResponse:
        """Generate text using xAI synchronously.

        Args:
            config: xAI configuration.
            prompt: Input prompt.
            params: Generation parameters.

        Returns:
            Generated response.
        """
        api_key = self._get_api_key(config, "XAI_API_KEY")
        client = self._get_client(api_key)

        try:
            # Create chat using the xAI SDK pattern
            messages = self._create_chat_messages(prompt)
            chat = client.chat.create(
                model=config.model,
                messages=messages,
            )

            # Sample a response from the chat
            response = chat.sample()
            return self._parse_response(response, config)

        except Exception as e:
            raise self._handle_xai_exception(e) from e

    async def agenenerate(
        self,
        config: ProviderConfig,
        prompt: Prompt,
        params: GenParams,
    ) -> LLMResponse:
        """Generate text using xAI asynchronously.

        Args:
            config: xAI configuration.
            prompt: Input prompt.
            params: Generation parameters.

        Returns:
            Generated response.
        """
        api_key = self._get_api_key(config, "XAI_API_KEY")

        try:
            client = self._get_async_client(api_key)

            # Create chat using the async xAI SDK pattern
            messages = self._create_chat_messages(prompt)
            chat = client.chat.create(
                model=config.model,
                messages=messages,
            )

            # Sample a response from the chat asynchronously
            response = await chat.sample()
            return self._parse_response(response, config)

        except Exception as e:
            # Try fallback to sync in thread if async fails
            if "async" in str(e).lower() or "awaitable" in str(e).lower():
                return await anyio.to_thread.run_sync(self.generate, config, prompt, params)

            raise self._handle_xai_exception(e) from e
