"""OpenAI adapter using the official OpenAI Python SDK.

This adapter implements the OpenAI provider using the official
openai Python package for Chat Completions API.
"""

from __future__ import annotations

from typing import Any

import anyio

from ..core.exceptions import AuthError, ProviderError, RateLimitError
from ..core.models import GenParams, LLMResponse, Prompt, ProviderConfig, Usage
from .base import BaseAdapter


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI using the official Python SDK."""

    def __init__(self) -> None:
        """Initialize the OpenAI adapter."""
        super().__init__()
        self._client = None
        self._async_client = None

    def _get_client(self, api_key: str, base_url: str | None = None):
        """Get or create synchronous OpenAI client.

        Args:
            api_key: OpenAI API key.
            base_url: Optional base URL override.

        Returns:
            OpenAI client instance.
        """
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
            except ImportError as e:
                raise ProviderError(
                    "openai package not installed. Install with: pip install openai",
                    provider="openai",
                ) from e
        return self._client

    def _get_async_client(self, api_key: str, base_url: str | None = None):
        """Get or create asynchronous OpenAI client.

        Args:
            api_key: OpenAI API key.
            base_url: Optional base URL override.

        Returns:
            Async OpenAI client instance.
        """
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI

                self._async_client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
            except ImportError as e:
                raise ProviderError(
                    "openai package not installed. Install with: pip install openai",
                    provider="openai",
                ) from e
        return self._async_client

    def _build_messages(self, prompt: Prompt) -> list[dict[str, str]]:
        """Build messages list for OpenAI chat API.

        Args:
            prompt: Input prompt with instructions and user input.

        Returns:
            List of message dictionaries.
        """
        return [
            {"role": "system", "content": prompt.instructions},
            {"role": "user", "content": prompt.input},
        ]

    def _build_request_params(
        self,
        config: ProviderConfig,
        prompt: Prompt,
        params: GenParams,
    ) -> dict[str, Any]:
        """Build request parameters for OpenAI Chat Completions API.

        Args:
            config: Provider configuration.
            prompt: Input prompt.
            params: Generation parameters.

        Returns:
            Request parameters dictionary.
        """
        messages = self._build_messages(prompt)

        request_params = {
            "model": config.model,
            "messages": messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
        }

        # Add optional parameters
        if params.top_p is not None:
            request_params["top_p"] = params.top_p

        if params.stop:
            request_params["stop"] = params.stop

        # Add OpenAI-specific parameters
        for key, value in params.extra.items():
            if key in ["stream", "response_format", "seed", "logprobs", "top_logprobs", "tools", "tool_choice"]:
                request_params[key] = value

        return request_params

    def _parse_response(
        self,
        response: Any,
        config: ProviderConfig,
    ) -> LLMResponse:
        """Parse OpenAI response into standardized format.

        Args:
            response: Raw response from OpenAI API.
            config: Provider configuration.

        Returns:
            Standardized LLM response.
        """
        try:
            # Extract text content from choices
            text_content = ""
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message") and choice.message:
                    text_content = choice.message.content or ""

            # Extract usage information
            usage = None
            if hasattr(response, "usage") and response.usage:
                usage_data = response.usage
                usage = Usage(
                    input_tokens=getattr(usage_data, "prompt_tokens", None),
                    output_tokens=getattr(usage_data, "completion_tokens", None),
                    total_tokens=getattr(usage_data, "total_tokens", None),
                )

            # Build raw data for debugging
            raw_data = {
                "id": getattr(response, "id", None),
                "object": getattr(response, "object", None),
                "created": getattr(response, "created", None),
                "model": getattr(response, "model", None),
                "choices": [],
                "usage": getattr(response, "usage", None),
            }

            if hasattr(response, "choices"):
                for choice in response.choices:
                    raw_data["choices"].append(
                        {
                            "index": getattr(choice, "index", None),
                            "message": {
                                "role": (
                                    getattr(choice.message, "role", None)
                                    if hasattr(choice, "message")
                                    else None
                                ),
                                "content": (
                                    getattr(choice.message, "content", None)
                                    if hasattr(choice, "message")
                                    else None
                                ),
                            },
                            "finish_reason": getattr(choice, "finish_reason", None),
                        }
                    )

            return LLMResponse(
                provider="openai",
                model=config.model,
                text=text_content,
                usage=usage,
                raw=raw_data,
            )

        except Exception as e:
            raise ProviderError(
                f"Failed to parse OpenAI response: {e}",
                provider="openai",
                raw={"error": str(e)},
            ) from e

    def _handle_openai_exception(self, e: Exception) -> Exception:
        """Map OpenAI exceptions to standard exceptions.

        Args:
            e: Original exception from OpenAI API.

        Returns:
            Mapped exception.
        """
        error_str = str(e).lower()

        if "api key" in error_str or "authentication" in error_str or "401" in error_str:
            return AuthError(
                "Authentication failed. Check your API key.",
                provider="openai",
                raw={"error": str(e)},
            )
        elif "rate limit" in error_str or "quota" in error_str or "429" in error_str:
            return RateLimitError(
                "Rate limit exceeded.",
                provider="openai",
                raw={"error": str(e)},
            )
        else:
            return ProviderError(
                f"OpenAI API error: {e}",
                provider="openai",
                raw={"error": str(e)},
            )

    def generate(
        self,
        config: ProviderConfig,
        prompt: Prompt,
        params: GenParams,
    ) -> LLMResponse:
        """Generate text using OpenAI synchronously.

        Args:
            config: OpenAI configuration.
            prompt: Input prompt.
            params: Generation parameters.

        Returns:
            Generated response.
        """
        api_key = self._get_api_key(config, "OPENAI_API_KEY")
        client = self._get_client(api_key, config.base_url)

        request_params = self._build_request_params(config, prompt, params)

        try:
            response = client.chat.completions.create(**request_params)
            return self._parse_response(response, config)

        except Exception as e:
            raise self._handle_openai_exception(e) from e

    async def agenenerate(
        self,
        config: ProviderConfig,
        prompt: Prompt,
        params: GenParams,
    ) -> LLMResponse:
        """Generate text using OpenAI asynchronously.

        Args:
            config: OpenAI configuration.
            prompt: Input prompt.
            params: Generation parameters.

        Returns:
            Generated response.
        """
        api_key = self._get_api_key(config, "OPENAI_API_KEY")

        try:
            client = self._get_async_client(api_key, config.base_url)
            request_params = self._build_request_params(config, prompt, params)

            response = await client.chat.completions.create(**request_params)
            return self._parse_response(response, config)

        except Exception as e:
            # Try fallback to sync in thread if async fails
            if "async" in str(e).lower():
                return await anyio.to_thread.run_sync(self.generate, config, prompt, params)

            raise self._handle_openai_exception(e) from e