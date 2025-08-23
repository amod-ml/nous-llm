"""
Tests for provider adapters.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx
from httpx import Response

from nous_llm import GenParams, Prompt, ProviderConfig
from nous_llm.adapters.anthropic_adapter import AnthropicAdapter
from nous_llm.adapters.gemini_adapter import GeminiAdapter
from nous_llm.adapters.openai_adapter import OpenAIAdapter
from nous_llm.adapters.openrouter_adapter import OpenRouterAdapter
from nous_llm.adapters.xai_adapter import XAIAdapter
from nous_llm.core.exceptions import AuthError, ProviderError, RateLimitError


class TestOpenAIAdapter:
    """Tests for OpenAI adapter."""

    @pytest.fixture
    def adapter(self) -> OpenAIAdapter:
        """Create OpenAI adapter instance."""
        return OpenAIAdapter()

    @pytest.fixture
    def config(self) -> ProviderConfig:
        """OpenAI configuration."""
        return ProviderConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
        )

    def test_sync_generate_success(
        self,
        adapter: OpenAIAdapter,
        config: ProviderConfig,
        sample_prompt: Prompt,
        sample_params: GenParams,
        openai_success_response: dict,
    ) -> None:
        """Test successful synchronous generation."""
        with respx.mock:
            respx.post("https://api.openai.com/v1/responses").mock(
                return_value=Response(200, json=openai_success_response)
            )

            response = adapter.generate(config, sample_prompt, sample_params)

            assert response.provider == "openai"
            assert response.model == "gpt-4o"
            assert response.text == "Paris is the capital of France."
            assert response.usage is not None
            assert response.usage.input_tokens == 15
            assert response.usage.output_tokens == 8
            assert response.usage.total_tokens == 23

    @pytest.mark.asyncio
    async def test_async_generate_success(
        self,
        adapter: OpenAIAdapter,
        config: ProviderConfig,
        sample_prompt: Prompt,
        sample_params: GenParams,
        openai_success_response: dict,
    ) -> None:
        """Test successful asynchronous generation."""
        with respx.mock:
            respx.post("https://api.openai.com/v1/responses").mock(
                return_value=Response(200, json=openai_success_response)
            )

            response = await adapter.agenenerate(config, sample_prompt, sample_params)

            assert response.provider == "openai"
            assert response.model == "gpt-4o"
            assert response.text == "Paris is the capital of France."

    def test_auth_error_handling(
        self,
        adapter: OpenAIAdapter,
        config: ProviderConfig,
        sample_prompt: Prompt,
        sample_params: GenParams,
    ) -> None:
        """Test authentication error handling."""
        error_response = {
            "error": {
                "message": "Invalid API key",
                "type": "invalid_api_key",
            }
        }

        with respx.mock:
            respx.post("https://api.openai.com/v1/responses").mock(return_value=Response(401, json=error_response))

            with pytest.raises(AuthError) as exc_info:
                adapter.generate(config, sample_prompt, sample_params)

            assert "Authentication failed" in str(exc_info.value)
            assert exc_info.value.provider == "openai"

    def test_rate_limit_error_handling(
        self,
        adapter: OpenAIAdapter,
        config: ProviderConfig,
        sample_prompt: Prompt,
        sample_params: GenParams,
    ) -> None:
        """Test rate limit error handling."""
        error_response = {
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_exceeded",
            }
        }

        with respx.mock:
            respx.post("https://api.openai.com/v1/responses").mock(return_value=Response(429, json=error_response))

            with pytest.raises(RateLimitError) as exc_info:
                adapter.generate(config, sample_prompt, sample_params)

            assert "Rate limit exceeded" in str(exc_info.value)
            assert exc_info.value.provider == "openai"

    def test_custom_base_url(
        self,
        adapter: OpenAIAdapter,
        sample_prompt: Prompt,
        sample_params: GenParams,
        openai_success_response: dict,
    ) -> None:
        """Test using custom base URL."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            base_url="https://custom.api.com/v1",
        )

        with respx.mock:
            respx.post("https://custom.api.com/v1/responses").mock(
                return_value=Response(200, json=openai_success_response)
            )

            response = adapter.generate(config, sample_prompt, sample_params)
            assert response.text == "Paris is the capital of France."


class TestAnthropicAdapter:
    """Tests for Anthropic adapter."""

    @pytest.fixture
    def adapter(self) -> AnthropicAdapter:
        """Create Anthropic adapter instance."""
        return AnthropicAdapter()

    @pytest.fixture
    def config(self) -> ProviderConfig:
        """Anthropic configuration."""
        return ProviderConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
        )

    def test_sync_generate_success(
        self,
        adapter: AnthropicAdapter,
        config: ProviderConfig,
        sample_prompt: Prompt,
        sample_params: GenParams,
    ) -> None:
        """Test successful synchronous generation."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Paris is the capital of France.")]
        mock_response.usage = MagicMock(input_tokens=15, output_tokens=8)
        mock_response.id = "msg_test_123"
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.stop_reason = "end_turn"

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_client.messages.create.return_value = mock_response

            response = adapter.generate(config, sample_prompt, sample_params)

            assert response.provider == "anthropic"
            assert response.model == "claude-3-5-sonnet-20241022"
            assert response.text == "Paris is the capital of France."
            assert response.usage is not None
            assert response.usage.input_tokens == 15
            assert response.usage.output_tokens == 8
            assert response.usage.total_tokens == 23

    @pytest.mark.asyncio
    async def test_async_generate_success(
        self,
        adapter: AnthropicAdapter,
        config: ProviderConfig,
        sample_prompt: Prompt,
        sample_params: GenParams,
    ) -> None:
        """Test successful asynchronous generation."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Paris is the capital of France.")]
        mock_response.usage = MagicMock(input_tokens=15, output_tokens=8)

        with patch("anthropic.AsyncAnthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_client.messages.create = AsyncMock(return_value=mock_response)

            response = await adapter.agenenerate(config, sample_prompt, sample_params)

            assert response.provider == "anthropic"
            assert response.text == "Paris is the capital of France."

    def test_missing_anthropic_package(
        self,
        adapter: AnthropicAdapter,
        config: ProviderConfig,
        sample_prompt: Prompt,
        sample_params: GenParams,
    ) -> None:
        """Test error when anthropic package is missing."""
        with patch("anthropic.Anthropic", side_effect=ImportError("No module named 'anthropic'")):
            with pytest.raises(ProviderError) as exc_info:
                adapter.generate(config, sample_prompt, sample_params)

            assert "anthropic package not installed" in str(exc_info.value)

    def test_auth_error_handling(
        self,
        adapter: AnthropicAdapter,
        config: ProviderConfig,
        sample_prompt: Prompt,
        sample_params: GenParams,
    ) -> None:
        """Test authentication error handling."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("authentication failed")

            with pytest.raises(AuthError):
                adapter.generate(config, sample_prompt, sample_params)


class TestGeminiAdapter:
    """Tests for Gemini adapter."""

    @pytest.fixture
    def adapter(self) -> GeminiAdapter:
        """Create Gemini adapter instance."""
        return GeminiAdapter()

    @pytest.fixture
    def config(self) -> ProviderConfig:
        """Gemini configuration."""
        return ProviderConfig(
            provider="gemini",
            model="gemini-1.5-pro",
            api_key="test-key",
        )

    def test_sync_generate_success(
        self,
        adapter: GeminiAdapter,
        config: ProviderConfig,
        sample_prompt: Prompt,
        sample_params: GenParams,
    ) -> None:
        """Test successful synchronous generation."""
        mock_response = MagicMock()
        mock_response.text = "Paris is the capital of France."
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=15,
            candidates_token_count=8,
            total_token_count=23,
        )

        with patch("google.generativeai.configure"), patch("google.generativeai.GenerativeModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            mock_model.generate_content.return_value = mock_response

            response = adapter.generate(config, sample_prompt, sample_params)

            assert response.provider == "gemini"
            assert response.model == "gemini-1.5-pro"
            assert response.text == "Paris is the capital of France."
            assert response.usage is not None
            assert response.usage.input_tokens == 15
            assert response.usage.output_tokens == 8
            assert response.usage.total_tokens == 23

    def test_missing_gemini_package(
        self,
        adapter: GeminiAdapter,
        config: ProviderConfig,
        sample_prompt: Prompt,
        sample_params: GenParams,
    ) -> None:
        """Test error when google-generativeai package is missing."""
        with patch("google.generativeai.configure", side_effect=ImportError("No module named 'google.generativeai'")):
            with pytest.raises(ProviderError) as exc_info:
                adapter.generate(config, sample_prompt, sample_params)

            assert "google-generativeai package not installed" in str(exc_info.value)


class TestXAIAdapter:
    """Tests for xAI adapter."""

    @pytest.fixture
    def adapter(self) -> XAIAdapter:
        """Create xAI adapter instance."""
        return XAIAdapter()

    @pytest.fixture
    def config(self) -> ProviderConfig:
        """xAI configuration."""
        return ProviderConfig(
            provider="xai",
            model="grok-beta",
            api_key="test-key",
        )

    def test_sync_generate_success(
        self,
        adapter: XAIAdapter,
        config: ProviderConfig,
        sample_prompt: Prompt,
        sample_params: GenParams,
    ) -> None:
        """Test successful synchronous generation."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Paris is the capital of France."
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=15,
            completion_tokens=8,
            total_tokens=23,
        )

        with patch("xai_sdk.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.create.return_value = mock_response

            response = adapter.generate(config, sample_prompt, sample_params)

            assert response.provider == "xai"
            assert response.model == "grok-beta"
            assert response.text == "Paris is the capital of France."
            assert response.usage is not None
            assert response.usage.input_tokens == 15
            assert response.usage.output_tokens == 8
            assert response.usage.total_tokens == 23


class TestOpenRouterAdapter:
    """Tests for OpenRouter adapter."""

    @pytest.fixture
    def adapter(self) -> OpenRouterAdapter:
        """Create OpenRouter adapter instance."""
        return OpenRouterAdapter()

    @pytest.fixture
    def config(self) -> ProviderConfig:
        """OpenRouter configuration."""
        return ProviderConfig(
            provider="openrouter",
            model="openai/gpt-4o",
            api_key="test-key",
        )

    def test_sync_generate_success(
        self,
        adapter: OpenRouterAdapter,
        config: ProviderConfig,
        sample_prompt: Prompt,
        sample_params: GenParams,
    ) -> None:
        """Test successful synchronous generation."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Paris is the capital of France."
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=15,
            completion_tokens=8,
            total_tokens=23,
        )

        with patch("openai.OpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            response = adapter.generate(config, sample_prompt, sample_params)

            assert response.provider == "openrouter"
            assert response.model == "openai/gpt-4o"
            assert response.text == "Paris is the capital of France."
            assert response.usage is not None
            assert response.usage.input_tokens == 15
            assert response.usage.output_tokens == 8
            assert response.usage.total_tokens == 23

    def test_custom_base_url(
        self,
        adapter: OpenRouterAdapter,
        sample_prompt: Prompt,
        sample_params: GenParams,
    ) -> None:
        """Test using custom base URL."""
        config = ProviderConfig(
            provider="openrouter",
            model="openai/gpt-4o",
            api_key="test-key",
            base_url="https://custom.openrouter.ai/api/v1",
        )

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        with patch("openai.OpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            response = adapter.generate(config, sample_prompt, sample_params)

            # Verify the client was initialized with custom base URL
            mock_client_class.assert_called_with(
                api_key="test-key",
                base_url="https://custom.openrouter.ai/api/v1",
            )

            assert response.text == "Test response"
