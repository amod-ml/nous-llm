"""Anthropic adapter using the Messages API.

This adapter implements the Anthropic provider using the official
anthropic Python SDK for Claude models.
"""

from __future__ import annotations

from typing import Any

import anyio

from ..core.exceptions import AuthError, ProviderError
from ..core.models import GenParams, LLMResponse, Prompt, ProviderConfig, Usage
from .base import BaseAdapter


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic Claude models using the Messages API."""

    def __init__(self) -> None:
        """Initialize the Anthropic adapter."""
        super().__init__()
        self._client = None
        self._async_client = None

    def _get_client(self, api_key: str):
        """Get or create synchronous Anthropic client.

        Args:
            api_key: Anthropic API key.

        Returns:
            Anthropic client instance.
        """
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError as e:
                raise ProviderError(
                    "anthropic package not installed. Install with: pip install anthropic",
                    provider="anthropic",
                ) from e
        return self._client

    def _get_async_client(self, api_key: str):
        """Get or create asynchronous Anthropic client.

        Args:
            api_key: Anthropic API key.

        Returns:
            Async Anthropic client instance.
        """
        if self._async_client is None:
            try:
                import anthropic

                self._async_client = anthropic.AsyncAnthropic(api_key=api_key)
            except ImportError as e:
                raise ProviderError(
                    "anthropic package not installed. Install with: pip install anthropic",
                    provider="anthropic",
                ) from e
        return self._async_client

    def _build_request_params(
        self,
        config: ProviderConfig,
        prompt: Prompt,
        params: GenParams,
    ) -> dict[str, Any]:
        """Build request parameters for Anthropic Messages API.

        Args:
            config: Provider configuration.
            prompt: Input prompt.
            params: Generation parameters.

        Returns:
            Request parameters dictionary.
        """
        request_params = {
            "model": config.model,
            "system": prompt.instructions,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt.input,
                        }
                    ],
                }
            ],
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
        }

        # Add optional parameters
        if params.top_p is not None:
            request_params["top_p"] = params.top_p

        if params.stop:
            request_params["stop_sequences"] = params.stop

        # Add Anthropic-specific parameters
        for key, value in params.extra.items():
            if key in ["thinking", "metadata", "tools", "tool_choice"]:
                request_params[key] = value

        return request_params

    def _parse_response(
        self,
        response: Any,
        config: ProviderConfig,
    ) -> LLMResponse:
        """Parse Anthropic response into standardized format.

        Args:
            response: Raw response from Anthropic API.
            config: Provider configuration.

        Returns:
            Standardized LLM response.
        """
        try:
            # Extract text content
            text_content = ""
            for content_block in response.content:
                if hasattr(content_block, "text"):
                    text_content += content_block.text
                elif hasattr(content_block, "type") and content_block.type == "text":
                    text_content += content_block.text

            # Extract usage information
            usage = None
            if hasattr(response, "usage") and response.usage:
                input_tokens = getattr(response.usage, "input_tokens", None)
                output_tokens = getattr(response.usage, "output_tokens", None)
                total_tokens = None

                # Calculate total if both input and output are available
                if input_tokens and output_tokens:
                    total_tokens = input_tokens + output_tokens

                usage = Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                )

            # Convert response to dict for raw storage
            raw_data = {
                "id": getattr(response, "id", None),
                "type": getattr(response, "type", None),
                "role": getattr(response, "role", None),
                "model": getattr(response, "model", None),
                "stop_reason": getattr(response, "stop_reason", None),
                "stop_sequence": getattr(response, "stop_sequence", None),
                "usage": getattr(response, "usage", None),
            }

            return LLMResponse(
                provider="anthropic",
                model=config.model,
                text=text_content,
                usage=usage,
                raw=raw_data,
            )

        except Exception as e:
            raise ProviderError(
                f"Failed to parse Anthropic response: {e}",
                provider="anthropic",
                raw={"error": str(e)},
            ) from e

    def generate(
        self,
        config: ProviderConfig,
        prompt: Prompt,
        params: GenParams,
    ) -> LLMResponse:
        """Generate text using Anthropic synchronously.

        Args:
            config: Anthropic configuration.
            prompt: Input prompt.
            params: Generation parameters.

        Returns:
            Generated response.
        """
        api_key = self._get_api_key(config, "ANTHROPIC_API_KEY")
        client = self._get_client(api_key)

        request_params = self._build_request_params(config, prompt, params)

        try:
            response = client.messages.create(**request_params)
            return self._parse_response(response, config)

        except Exception as e:
            # Map common Anthropic exceptions
            if "authentication" in str(e).lower() or "401" in str(e):
                raise AuthError(
                    "Anthropic authentication failed. Check your API key.",
                    provider="anthropic",
                    raw={"error": str(e)},
                ) from e
            elif "rate limit" in str(e).lower() or "429" in str(e):
                from ..core.exceptions import RateLimitError

                raise RateLimitError(
                    "Anthropic rate limit exceeded.",
                    provider="anthropic",
                    raw={"error": str(e)},
                ) from e
            else:
                raise ProviderError(
                    f"Anthropic API error: {e}",
                    provider="anthropic",
                    raw={"error": str(e)},
                ) from e

    async def agenenerate(
        self,
        config: ProviderConfig,
        prompt: Prompt,
        params: GenParams,
    ) -> LLMResponse:
        """Generate text using Anthropic asynchronously.

        Args:
            config: Anthropic configuration.
            prompt: Input prompt.
            params: Generation parameters.

        Returns:
            Generated response.
        """
        api_key = self._get_api_key(config, "ANTHROPIC_API_KEY")

        # Use async client if available, otherwise run sync in thread
        try:
            client = self._get_async_client(api_key)
            request_params = self._build_request_params(config, prompt, params)

            response = await client.messages.create(**request_params)
            return self._parse_response(response, config)

        except Exception as e:
            # Try fallback to sync in thread if async fails
            if "async" in str(e).lower():
                return await anyio.to_thread.run_sync(self.generate, config, prompt, params)

            # Map common Anthropic exceptions
            if "authentication" in str(e).lower() or "401" in str(e):
                raise AuthError(
                    "Anthropic authentication failed. Check your API key.",
                    provider="anthropic",
                    raw={"error": str(e)},
                ) from e
            elif "rate limit" in str(e).lower() or "429" in str(e):
                from ..core.exceptions import RateLimitError

                raise RateLimitError(
                    "Anthropic rate limit exceeded.",
                    provider="anthropic",
                    raw={"error": str(e)},
                ) from e
            else:
                raise ProviderError(
                    f"Anthropic API error: {e}",
                    provider="anthropic",
                    raw={"error": str(e)},
                ) from e
