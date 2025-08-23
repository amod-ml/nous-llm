"""Gemini adapter using the Google Gen AI SDK.

This adapter implements the Google Gemini provider using the official
google-genai Python SDK for the latest API patterns (2025).
"""

from __future__ import annotations

from typing import Any

import anyio

from ..core.exceptions import AuthError, ProviderError, RateLimitError
from ..core.models import GenParams, LLMResponse, Prompt, ProviderConfig, Usage
from .base import BaseAdapter


class GeminiAdapter(BaseAdapter):
    """Adapter for Google Gemini models using the Gen AI SDK."""

    def __init__(self) -> None:
        """Initialize the Gemini adapter."""
        super().__init__()
        self._client = None
        self._async_client = None

    def _get_client(self, api_key: str):
        """Get or create synchronous Gemini client.

        Args:
            api_key: Google API key for Gemini.

        Returns:
            Gemini client instance.
        """
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=api_key)
            except ImportError as e:
                raise ProviderError(
                    "google-genai package not installed. Install with: pip install google-genai",
                    provider="gemini",
                ) from e
        return self._client

    def _get_async_client(self, api_key: str):
        """Get or create asynchronous Gemini client.

        Args:
            api_key: Google API key for Gemini.

        Returns:
            Async Gemini client instance.
        """
        if self._async_client is None:
            try:
                from google import genai

                # Note: google-genai SDK uses the same client for sync/async
                self._async_client = genai.Client(api_key=api_key)
            except ImportError as e:
                raise ProviderError(
                    "google-genai package not installed. Install with: pip install google-genai",
                    provider="gemini",
                ) from e
        return self._async_client

    def _build_generation_config(self, params: GenParams) -> dict[str, Any]:
        """Build generation configuration for Gemini.

        Args:
            params: Generation parameters.

        Returns:
            Gemini generation configuration.
        """
        config = {
            "max_output_tokens": params.max_tokens,
            "temperature": params.temperature,
        }

        if params.top_p is not None:
            config["top_p"] = params.top_p

        if params.stop:
            config["stop_sequences"] = params.stop

        # Add Gemini-specific parameters
        for key, value in params.extra.items():
            if key in ["top_k", "candidate_count", "response_mime_type"]:
                config[key] = value

        return config

    def _build_content(self, prompt: Prompt) -> str:
        """Build content string for Gemini API.

        Args:
            prompt: Input prompt with instructions and user input.

        Returns:
            Formatted content string.
        """
        return f"System: {prompt.instructions}\n\nUser: {prompt.input}"

    def _parse_response(
        self,
        response: Any,
        config: ProviderConfig,
    ) -> LLMResponse:
        """Parse Gemini response into standardized format.

        Args:
            response: Raw response from Gemini API.
            config: Provider configuration.

        Returns:
            Standardized LLM response.
        """
        try:
            # Extract text content - new SDK returns text directly
            text_content = ""
            if hasattr(response, "text") and response.text:
                text_content = response.text
            elif hasattr(response, "candidates") and response.candidates:
                # Handle structured response format
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    if hasattr(candidate.content, "parts"):
                        for part in candidate.content.parts:
                            if hasattr(part, "text"):
                                text_content += part.text
                    elif hasattr(candidate.content, "text"):
                        text_content = candidate.content.text

            # Extract usage information
            usage = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage_data = response.usage_metadata
                usage = Usage(
                    input_tokens=getattr(usage_data, "prompt_token_count", None),
                    output_tokens=getattr(usage_data, "candidates_token_count", None),
                    total_tokens=getattr(usage_data, "total_token_count", None),
                )

            # Build raw data for debugging
            raw_data = {
                "model": config.model,
                "text": text_content,
            }

            if hasattr(response, "usage_metadata"):
                raw_data["usage_metadata"] = getattr(response, "usage_metadata", None)

            if hasattr(response, "candidates"):
                raw_data["candidates"] = []
                for candidate in response.candidates:
                    raw_data["candidates"].append(
                        {
                            "content": getattr(candidate, "content", None),
                            "finish_reason": getattr(candidate, "finish_reason", None),
                            "safety_ratings": getattr(candidate, "safety_ratings", None),
                        }
                    )

            return LLMResponse(
                provider="gemini",
                model=config.model,
                text=text_content,
                usage=usage,
                raw=raw_data,
            )

        except Exception as e:
            raise ProviderError(
                f"Failed to parse Gemini response: {e}",
                provider="gemini",
                raw={"error": str(e)},
            ) from e

    def _handle_gemini_exception(self, e: Exception) -> Exception:
        """Map Gemini exceptions to standard exceptions.

        Args:
            e: Original exception from Gemini API.

        Returns:
            Mapped exception.
        """
        error_str = str(e).lower()

        if "api key" in error_str or "authentication" in error_str or "401" in error_str:
            return AuthError(
                "Gemini authentication failed. Check your API key.",
                provider="gemini",
                raw={"error": str(e)},
            )
        elif "quota" in error_str or "rate limit" in error_str or "429" in error_str:
            return RateLimitError(
                "Gemini rate limit or quota exceeded.",
                provider="gemini",
                raw={"error": str(e)},
            )
        else:
            return ProviderError(
                f"Gemini API error: {e}",
                provider="gemini",
                raw={"error": str(e)},
            )

    def generate(
        self,
        config: ProviderConfig,
        prompt: Prompt,
        params: GenParams,
    ) -> LLMResponse:
        """Generate text using Gemini synchronously.

        Args:
            config: Gemini configuration.
            prompt: Input prompt.
            params: Generation parameters.

        Returns:
            Generated response.
        """
        api_key = self._get_api_key(config, "GEMINI_API_KEY")
        client = self._get_client(api_key)

        try:
            # Build generation config
            generation_config = self._build_generation_config(params)

            # Generate content using the new SDK pattern
            content = self._build_content(prompt)
            response = client.models.generate_content(
                model=config.model,
                contents=content,
                config=generation_config,
            )

            return self._parse_response(response, config)

        except Exception as e:
            raise self._handle_gemini_exception(e) from e

    async def agenenerate(
        self,
        config: ProviderConfig,
        prompt: Prompt,
        params: GenParams,
    ) -> LLMResponse:
        """Generate text using Gemini asynchronously.

        Args:
            config: Gemini configuration.
            prompt: Input prompt.
            params: Generation parameters.

        Returns:
            Generated response.
        """
        api_key = self._get_api_key(config, "GEMINI_API_KEY")
        client = self._get_async_client(api_key)

        try:
            # Build generation config
            generation_config = self._build_generation_config(params)

            # Generate content
            content = self._build_content(prompt)

            # Check if async method exists
            if hasattr(client.models, "generate_content_async"):
                response = await client.models.generate_content_async(
                    model=config.model,
                    contents=content,
                    config=generation_config,
                )
            else:
                # Fallback to running sync in thread
                def _sync_generate():
                    return client.models.generate_content(
                        model=config.model,
                        contents=content,
                        config=generation_config,
                    )

                response = await anyio.to_thread.run_sync(_sync_generate)

            return self._parse_response(response, config)

        except Exception as e:
            raise self._handle_gemini_exception(e) from e
