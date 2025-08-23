"""OpenRouter adapter using the OpenAI SDK with custom base URL.

This adapter implements the OpenRouter provider using the OpenAI SDK
with a custom base URL to access OpenRouter's extensive model catalog.
"""

from __future__ import annotations

from typing import Any

import anyio

from ..core.exceptions import AuthError, ProviderError, RateLimitError
from ..core.models import GenParams, LLMResponse, Prompt, ProviderConfig, Usage
from .base import BaseAdapter


class OpenRouterAdapter(BaseAdapter):
    """Adapter for OpenRouter using OpenAI SDK with custom base URL."""

    def __init__(self) -> None:
        """Initialize the OpenRouter adapter."""
        super().__init__()
        self.base_url = "https://openrouter.ai/api/v1"
        self._client = None
        self._async_client = None

    def _get_client(self, api_key: str, base_url: str):
        """Get or create synchronous OpenAI client for OpenRouter.

        Args:
            api_key: OpenRouter API key.
            base_url: OpenRouter base URL.

        Returns:
            OpenAI client configured for OpenRouter.
        """
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
            except ImportError as e:
                raise ProviderError(
                    "openai package not installed. Install with: pip install openai",
                    provider="openrouter",
                ) from e
        return self._client

    def _get_async_client(self, api_key: str, base_url: str):
        """Get or create asynchronous OpenAI client for OpenRouter.

        Args:
            api_key: OpenRouter API key.
            base_url: OpenRouter base URL.

        Returns:
            Async OpenAI client configured for OpenRouter.
        """
        if self._async_client is None:
            try:
                import openai

                self._async_client = openai.AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
            except ImportError as e:
                raise ProviderError(
                    "openai package not installed. Install with: pip install openai",
                    provider="openrouter",
                ) from e
        return self._async_client

    def _build_messages(self, prompt: Prompt) -> list[dict[str, str]]:
        """Build messages list for OpenRouter chat API.

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
        """Build request parameters for OpenRouter API.

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

        # Add OpenRouter-specific parameters
        for key, value in params.extra.items():
            if key in ["stream", "seed", "tools", "tool_choice", "transforms"]:
                request_params[key] = value

        return request_params

    def _parse_response(
        self,
        response: Any,
        config: ProviderConfig,
    ) -> LLMResponse:
        """Parse OpenRouter response into standardized format.

        Args:
            response: Raw response from OpenRouter API.
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
                            "message": getattr(choice, "message", None),
                            "finish_reason": getattr(choice, "finish_reason", None),
                        }
                    )

            return LLMResponse(
                provider="openrouter",
                model=config.model,
                text=text_content,
                usage=usage,
                raw=raw_data,
            )

        except Exception as e:
            raise ProviderError(
                f"Failed to parse OpenRouter response: {e}",
                provider="openrouter",
                raw={"error": str(e)},
            ) from e

    def _handle_openrouter_exception(self, e: Exception) -> Exception:
        """Map OpenRouter exceptions to standard exceptions.

        Args:
            e: Original exception from OpenRouter API.

        Returns:
            Mapped exception.
        """
        error_str = str(e).lower()

        if "api key" in error_str or "authentication" in error_str or "401" in error_str:
            return AuthError(
                "OpenRouter authentication failed. Check your API key.",
                provider="openrouter",
                raw={"error": str(e)},
            )
        elif "rate limit" in error_str or "quota" in error_str or "429" in error_str:
            return RateLimitError(
                "OpenRouter rate limit exceeded.",
                provider="openrouter",
                raw={"error": str(e)},
            )
        else:
            return ProviderError(
                f"OpenRouter API error: {e}",
                provider="openrouter",
                raw={"error": str(e)},
            )

    def generate(
        self,
        config: ProviderConfig,
        prompt: Prompt,
        params: GenParams,
    ) -> LLMResponse:
        """Generate text using OpenRouter synchronously.

        Args:
            config: OpenRouter configuration.
            prompt: Input prompt.
            params: Generation parameters.

        Returns:
            Generated response.
        """
        api_key = self._get_api_key(config, "OPENROUTER_API_KEY")
        base_url = config.base_url or self.base_url

        client = self._get_client(api_key, base_url)
        request_params = self._build_request_params(config, prompt, params)

        try:
            response = client.chat.completions.create(**request_params)
            return self._parse_response(response, config)

        except Exception as e:
            raise self._handle_openrouter_exception(e) from e

    async def agenenerate(
        self,
        config: ProviderConfig,
        prompt: Prompt,
        params: GenParams,
    ) -> LLMResponse:
        """Generate text using OpenRouter asynchronously.

        Args:
            config: OpenRouter configuration.
            prompt: Input prompt.
            params: Generation parameters.

        Returns:
            Generated response.
        """
        api_key = self._get_api_key(config, "OPENROUTER_API_KEY")
        base_url = config.base_url or self.base_url

        try:
            client = self._get_async_client(api_key, base_url)
            request_params = self._build_request_params(config, prompt, params)

            response = await client.chat.completions.create(**request_params)
            return self._parse_response(response, config)

        except Exception as e:
            # Try fallback to sync in thread if async fails
            if "async" in str(e).lower():
                return await anyio.to_thread.run_sync(self.generate, config, prompt, params)

            raise self._handle_openrouter_exception(e) from e
