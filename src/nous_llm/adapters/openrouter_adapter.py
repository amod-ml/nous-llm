"""OpenRouter adapter using the OpenAI SDK with custom base URL.

This adapter implements the OpenRouter provider using the OpenAI SDK
with a custom base URL to access OpenRouter's extensive model catalog.
"""

from __future__ import annotations

import re
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

        # Model patterns for reasoning capability detection
        self._reasoning_model_patterns = {
            # OpenAI o-series and GPT-5 models (use effort levels)
            "effort_based": [
                r"openai/o[1-9].*",
                r"openai/o3.*",
                r"openai/gpt-5.*",
                r"xai/grok.*",
            ],
            # Anthropic models (use max_tokens)
            "max_tokens_based": [
                r"anthropic/claude.*",
            ],
            # Gemini thinking models (use max_tokens)
            "gemini_thinking": [
                r"google/gemini.*thinking.*",
                r"google/gemini.*flash.*thinking.*",
            ],
        }

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

    def _detect_model_reasoning_type(self, model: str) -> str | None:
        """Detect the reasoning parameter type supported by a model.

        Args:
            model: Model identifier.

        Returns:
            Reasoning type: 'effort_based', 'max_tokens_based', 'gemini_thinking', or None.
        """
        for reasoning_type, patterns in self._reasoning_model_patterns.items():
            for pattern in patterns:
                if re.match(pattern, model, re.IGNORECASE):
                    return reasoning_type
        return None

    def _build_reasoning_config(self, params: GenParams, model: str) -> dict[str, Any] | None:
        """Build OpenRouter reasoning configuration based on model type and parameters.

        Args:
            params: Generation parameters.
            model: Model identifier.

        Returns:
            Reasoning configuration dict or None if not applicable.
        """
        reasoning_type = self._detect_model_reasoning_type(model)
        if not reasoning_type:
            return None

        reasoning_config = {}

        # Check for reasoning parameters in extra
        reasoning_effort = params.extra.get("reasoning_effort")
        reasoning_max_tokens = params.extra.get("reasoning_max_tokens")
        reasoning_exclude = params.extra.get("reasoning_exclude", False)
        reasoning_enabled = params.extra.get("reasoning_enabled")

        # Legacy parameter support
        include_thoughts = params.extra.get("include_thoughts")
        thinking_budget = params.extra.get("thinking_budget")

        # Map legacy parameters to new format
        if include_thoughts is not None:
            reasoning_exclude = not include_thoughts
        if thinking_budget is not None:
            reasoning_max_tokens = thinking_budget

        # Build reasoning config based on model type
        if reasoning_type == "effort_based":
            # OpenAI o-series, GPT-5, Grok models
            if reasoning_effort:
                reasoning_config["effort"] = reasoning_effort
            elif reasoning_max_tokens:
                # Convert max_tokens to effort level
                if reasoning_max_tokens >= 8000:
                    reasoning_config["effort"] = "high"
                elif reasoning_max_tokens >= 4000:
                    reasoning_config["effort"] = "medium"
                else:
                    reasoning_config["effort"] = "low"
            elif reasoning_enabled:
                reasoning_config["effort"] = "medium"  # Default

        elif reasoning_type in ["max_tokens_based", "gemini_thinking"]:
            # Anthropic and Gemini thinking models
            if reasoning_max_tokens:
                reasoning_config["max_tokens"] = reasoning_max_tokens
            elif reasoning_effort:
                # Convert effort to approximate max_tokens
                effort_to_tokens = {
                    "high": int(params.max_tokens * 0.8) if params.max_tokens else 8000,
                    "medium": int(params.max_tokens * 0.5) if params.max_tokens else 4000,
                    "low": int(params.max_tokens * 0.2) if params.max_tokens else 1000,
                }
                reasoning_config["max_tokens"] = effort_to_tokens.get(reasoning_effort, 4000)
            elif reasoning_enabled:
                reasoning_config["max_tokens"] = 4000  # Default

        # Add exclude parameter if specified
        if reasoning_exclude:
            reasoning_config["exclude"] = True

        # Add enabled parameter if specified and no other parameters
        if reasoning_enabled and not reasoning_config:
            reasoning_config["enabled"] = True

        return reasoning_config if reasoning_config else None

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

        # Add reasoning configuration if supported
        reasoning_config = self._build_reasoning_config(params, config.model)
        if reasoning_config:
            request_params["reasoning"] = reasoning_config

        # Add OpenRouter-specific parameters
        reasoning_keys = {
            "reasoning_effort",
            "reasoning_max_tokens",
            "reasoning_exclude",
            "reasoning_enabled",
            "include_thoughts",
            "thinking_budget",
        }
        for key, value in params.extra.items():
            if key in ["stream", "seed", "tools", "tool_choice", "transforms"] or key not in reasoning_keys:
                # Only add non-reasoning parameters to avoid duplication
                if key not in [
                    "reasoning_effort",
                    "reasoning_max_tokens",
                    "reasoning_exclude",
                    "reasoning_enabled",
                    "include_thoughts",
                    "thinking_budget",
                ]:
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
            # Extract text content and reasoning from choices
            text_content = ""
            reasoning_content = ""

            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message") and choice.message:
                    text_content = choice.message.content or ""

                    # Check for reasoning content in message
                    if hasattr(choice.message, "reasoning") and choice.message.reasoning:
                        reasoning_content = choice.message.reasoning

            # Combine reasoning and regular content if both exist
            if reasoning_content:
                if text_content:
                    # Add reasoning content as a separate section
                    text_content = f"**Reasoning:**\n{reasoning_content}\n\n**Response:**\n{text_content}"
                else:
                    # Only reasoning content available
                    text_content = f"**Reasoning:**\n{reasoning_content}"

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

            # Add reasoning details if present
            if hasattr(response, "reasoning_details"):
                raw_data["reasoning_details"] = getattr(response, "reasoning_details", None)

            if hasattr(response, "choices"):
                for choice in response.choices:
                    choice_data = {
                        "index": getattr(choice, "index", None),
                        "message": getattr(choice, "message", None),
                        "finish_reason": getattr(choice, "finish_reason", None),
                    }

                    # Add reasoning content if present
                    if hasattr(choice, "message") and hasattr(choice.message, "reasoning"):
                        choice_data["reasoning"] = getattr(choice.message, "reasoning", None)

                    raw_data["choices"].append(choice_data)

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
