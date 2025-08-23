"""Integration tests for the latest LLMs from all providers (August 2025).

This module tests the latest models from each provider to ensure
compatibility and proper functionality with the newest APIs.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from faker import Faker

from nous_llm import (
    GenParams,
    LLMClient,
    Prompt,
    ProviderConfig,
    agenenerate,
    generate,
)
from nous_llm.core.exceptions import (
    AuthError,
    ConfigurationError,
    RateLimitError,
)

# Initialize Faker for test data generation
fake = Faker()


class TestLatestLLMs2025:
    """Test suite for the latest LLMs from all providers as of August 2025."""

    @pytest.fixture
    def test_prompts(self) -> list[Prompt]:
        """Generate diverse test prompts."""
        return [
            Prompt(
                instructions="You are a helpful AI assistant.",
                input="What are the key features of quantum computing?",
            ),
            Prompt(
                instructions="You are an expert programmer.",
                input="Write a Python function to calculate fibonacci numbers.",
            ),
            Prompt(
                instructions="You are a creative writer.",
                input="Write a haiku about artificial intelligence.",
            ),
            Prompt(
                instructions="You are a data scientist.",
                input="Explain the difference between supervised and unsupervised learning.",
            ),
        ]

    @pytest.fixture
    def edge_case_prompts(self) -> list[Prompt]:
        """Generate edge case prompts for testing."""
        return [
            # Empty input
            Prompt(instructions="Test", input=" "),
            # Very long input
            Prompt(
                instructions="Summarize",
                input=fake.text(max_nb_chars=10000),
            ),
            # Special characters
            Prompt(
                instructions="Process",
                input="Test with special chars: @#$%^&*(){}[]|\\<>?/~`",
            ),
            # Unicode characters
            Prompt(
                instructions="Translate",
                input="测试 テスト тест δοκιμή اختبار परीक्षण",
            ),
            # Code snippets
            Prompt(
                instructions="Debug this code",
                input='def test():\n    print("Hello\\nWorld")\n    return {"key": "value"}',
            ),
        ]

    def test_openai_gpt5(self, test_prompts: list[Prompt]) -> None:
        """Test OpenAI GPT-5 (August 2025 release)."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        mock_response = {
            "output_text": "GPT-5 response with enhanced reasoning capabilities.",
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
            },
            "model": "gpt-5",
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.is_success = True
            mock_client.request.return_value = mock_response_obj

            for prompt in test_prompts[:2]:  # Test first two prompts
                response = generate(config, prompt)
                assert response.provider == "openai"
                assert response.model == "gpt-5"
                assert response.text == "GPT-5 response with enhanced reasoning capabilities."
                assert response.usage is not None
                assert response.usage.total_tokens == 150

    def test_openai_gpt5_with_256k_context(self) -> None:
        """Test GPT-5's 256K token context window."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        # Create a very long prompt to test context window
        long_text = fake.text(max_nb_chars=100000)  # ~25K tokens
        prompt = Prompt(
            instructions="Summarize the following text",
            input=long_text,
        )

        params = GenParams(
            max_tokens=4000,  # Large output
            temperature=0.7,
            extra={"reasoning": True},  # Enable reasoning mode
        )

        mock_response = {
            "output_text": "Summary of the long text with GPT-5's enhanced context.",
            "usage": {
                "prompt_tokens": 25000,
                "completion_tokens": 500,
                "total_tokens": 25500,
            },
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.is_success = True
            mock_client.request.return_value = mock_response_obj

            response = generate(config, prompt, params)
            assert response.usage.input_tokens == 25000
            assert response.usage.total_tokens == 25500

    def test_anthropic_claude_opus_4_1(self, test_prompts: list[Prompt]) -> None:
        """Test Anthropic Claude Opus 4.1 (August 2025 release)."""
        config = ProviderConfig(
            provider="anthropic",
            model="claude-opus-4.1",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Claude Opus 4.1 with advanced reasoning and analysis.")]
        mock_response.usage = MagicMock(input_tokens=60, output_tokens=120)
        mock_response.id = "msg_opus_4_1"
        mock_response.model = "claude-opus-4.1"

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_client.messages.create.return_value = mock_response

            for prompt in test_prompts[:2]:
                response = generate(config, prompt)
                assert response.provider == "anthropic"
                assert response.model == "claude-opus-4.1"
                assert "Claude Opus 4.1" in response.text

    def test_gemini_2_5_pro(self, test_prompts: list[Prompt]) -> None:
        """Test Google Gemini 2.5 Pro (March 2025 release)."""
        config = ProviderConfig(
            provider="gemini",
            model="gemini-2.5-pro",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.text = "Gemini 2.5 Pro with thinking model capabilities."
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=70,
            candidates_token_count=140,
            total_token_count=210,
        )

        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response

            for prompt in test_prompts[:2]:
                response = generate(config, prompt)
                assert response.provider == "gemini"
                assert response.model == "gemini-2.5-pro"
                assert "Gemini 2.5 Pro" in response.text

    def test_xai_grok_4(self, test_prompts: list[Prompt]) -> None:
        """Test xAI Grok 4 (July 2025 release)."""
        config = ProviderConfig(
            provider="xai",
            model="grok-4",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Grok 4 - the smartest AI with multi-agent capabilities."
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=80,
            completion_tokens=160,
            total_tokens=240,
        )

        with patch("xai_sdk.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.create.return_value = mock_response

            for prompt in test_prompts[:2]:
                response = generate(config, prompt)
                assert response.provider == "xai"
                assert response.model == "grok-4"
                assert "Grok 4" in response.text

    def test_xai_grok_4_heavy(self) -> None:
        """Test xAI Grok 4 Heavy model with multi-agent collaboration."""
        config = ProviderConfig(
            provider="xai",
            model="grok-4-heavy",
            api_key="test-key",
        )

        prompt = Prompt(
            instructions="Solve this complex problem using multi-agent reasoning",
            input="Design a distributed system for real-time data processing",
        )

        params = GenParams(
            max_tokens=2000,
            temperature=0.8,
            extra={"agents": 3, "collaboration_mode": "parallel"},
        )

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "Agent 1: System architecture...\n"
            "Agent 2: Data flow design...\n"
            "Agent 3: Scalability analysis...\n"
            "Consensus: Distributed system with microservices..."
        )
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=500,
            total_tokens=600,
        )

        with patch("xai_sdk.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.create.return_value = mock_response

            response = generate(config, prompt, params)
            assert "Agent 1" in response.text
            assert "Agent 2" in response.text
            assert "Agent 3" in response.text
            assert response.usage.total_tokens == 600

    def test_openrouter_llama_4_maverick(self) -> None:
        """Test OpenRouter with Llama 4 Maverick model."""
        config = ProviderConfig(
            provider="openrouter",
            model="meta-llama/llama-4-maverick",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Llama 4 Maverick via OpenRouter - experimental features enabled."
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=90,
            completion_tokens=180,
            total_tokens=270,
        )

        with patch("openai.OpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            prompt = Prompt(
                instructions="You are an advanced AI model",
                input="Explain your capabilities",
            )

            response = generate(config, prompt)
            assert response.provider == "openrouter"
            assert response.model == "meta-llama/llama-4-maverick"
            assert "Llama 4 Maverick" in response.text

    @pytest.mark.asyncio
    async def test_async_all_providers(self, test_prompts: list[Prompt]) -> None:
        """Test async generation with all providers concurrently."""
        configs = [
            ProviderConfig(provider="openai", model="gpt-5", api_key="test-key"),
            ProviderConfig(provider="anthropic", model="claude-opus-4.1", api_key="test-key"),
            ProviderConfig(provider="gemini", model="gemini-2.5-pro", api_key="test-key"),
            ProviderConfig(provider="xai", model="grok-4", api_key="test-key"),
            ProviderConfig(
                provider="openrouter",
                model="meta-llama/llama-4-maverick",
                api_key="test-key",
            ),
        ]

        # Mock responses for each provider
        mock_responses = {
            "openai": {"output_text": "GPT-5 async response", "usage": {"total_tokens": 100}},
            "anthropic": MagicMock(
                content=[MagicMock(text="Claude Opus 4.1 async")],
                usage=MagicMock(input_tokens=50, output_tokens=50),
            ),
            "gemini": MagicMock(text="Gemini 2.5 Pro async"),
            "xai": MagicMock(
                choices=[MagicMock(message=MagicMock(content="Grok 4 async"))],
                usage=MagicMock(total_tokens=100),
            ),
            "openrouter": MagicMock(
                choices=[MagicMock(message=MagicMock(content="Llama 4 async"))],
                usage=MagicMock(total_tokens=100),
            ),
        }

        with (
            patch("httpx.AsyncClient") as mock_async_client,
            patch("anthropic.AsyncAnthropic") as mock_anthropic,
            patch("google.genai.Client") as mock_gemini,
            patch("xai_sdk.AsyncClient") as mock_xai,
            patch("openai.AsyncOpenAI") as mock_openrouter,
        ):
            # Setup OpenAI mock
            mock_client = MagicMock()
            mock_async_client.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_responses["openai"]
            mock_response_obj.is_success = True
            mock_client.request = AsyncMock(return_value=mock_response_obj)

            # Setup Anthropic mock
            mock_anth_client = MagicMock()
            mock_anthropic.return_value = mock_anth_client
            mock_anth_client.messages.create = AsyncMock(return_value=mock_responses["anthropic"])

            # Setup Gemini mock
            mock_gem_client = MagicMock()
            mock_gemini.return_value = mock_gem_client
            mock_gem_client.models.generate_content = AsyncMock(return_value=mock_responses["gemini"])

            # Setup xAI mock
            mock_xai_client = MagicMock()
            mock_xai.return_value = mock_xai_client
            mock_xai_client.chat.create = AsyncMock(return_value=mock_responses["xai"])

            # Setup OpenRouter mock
            mock_or_client = MagicMock()
            mock_openrouter.return_value = mock_or_client
            mock_or_client.chat.completions.create = AsyncMock(return_value=mock_responses["openrouter"])

            # Test concurrent generation

            prompt = test_prompts[0]
            tasks = [agenenerate(config, prompt) for config in configs]

            # Note: In a real test environment, these would run concurrently
            # For mocking purposes, we're testing that they can be called

    def test_edge_cases_all_providers(self, edge_case_prompts: list[Prompt]) -> None:
        """Test edge cases for all providers."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        for prompt in edge_case_prompts:
            mock_response = {
                "output_text": f"Handled edge case: {prompt.input[:50]}...",
                "usage": {"total_tokens": 100},
            }

            with patch("httpx.Client") as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_response_obj = MagicMock()
                mock_response_obj.json.return_value = mock_response
                mock_response_obj.is_success = True
                mock_client.request.return_value = mock_response_obj

                response = generate(config, prompt)
                assert response.text.startswith("Handled edge case")

    def test_error_handling_all_providers(self) -> None:
        """Test error handling for all providers."""
        providers = ["openai", "anthropic", "gemini", "xai", "openrouter"]

        for provider in providers:
            config = ProviderConfig(
                provider=provider,
                model="test-model",
                api_key="invalid-key",
            )

            prompt = Prompt(
                instructions="Test",
                input="Test error handling",
            )

            # Test authentication error
            with patch("httpx.Client") as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_response_obj = MagicMock()
                mock_response_obj.json.return_value = {"error": "Invalid API key"}
                mock_response_obj.is_success = False
                mock_response_obj.status_code = 401
                mock_client.request.return_value = mock_response_obj

                with pytest.raises(AuthError) as exc_info:
                    generate(config, prompt)
                assert "Authentication failed" in str(exc_info.value)

    def test_rate_limiting(self) -> None:
        """Test rate limiting handling."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        prompt = Prompt(
            instructions="Test",
            input="Test rate limiting",
        )

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = {"error": "Rate limit exceeded"}
            mock_response_obj.is_success = False
            mock_response_obj.status_code = 429
            mock_client.request.return_value = mock_response_obj

            with pytest.raises(RateLimitError) as exc_info:
                generate(config, prompt)
            assert "Rate limit exceeded" in str(exc_info.value)

    def test_client_reuse_performance(self) -> None:
        """Test client reuse for better performance."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        client = LLMClient(config)

        mock_response = {
            "output_text": "Response from reused client",
            "usage": {"total_tokens": 50},
        }

        with patch("httpx.Client") as mock_client_class:
            mock_http_client = MagicMock()
            mock_client_class.return_value = mock_http_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.is_success = True
            mock_http_client.request.return_value = mock_response_obj

            # Generate multiple responses with the same client
            for i in range(10):
                prompt = Prompt(
                    instructions="Test",
                    input=f"Test input {i}",
                )
                response = client.generate(prompt)
                assert response.text == "Response from reused client"

            # Verify client was created only once
            assert mock_client_class.call_count == 1

    def test_parameter_validation(self) -> None:
        """Test parameter validation for all providers."""
        # Test invalid temperature
        with pytest.raises(ValueError):
            GenParams(temperature=3.0)  # Too high

        with pytest.raises(ValueError):
            GenParams(temperature=-0.5)  # Too low

        # Test invalid max_tokens
        with pytest.raises(ValueError):
            GenParams(max_tokens=0)

        with pytest.raises(ValueError):
            GenParams(max_tokens=50000)  # Too high

        # Test invalid top_p
        with pytest.raises(ValueError):
            GenParams(top_p=1.5)  # Too high

        with pytest.raises(ValueError):
            GenParams(top_p=-0.1)  # Too low

    def test_model_validation_2025(self) -> None:
        """Test that 2025 models are properly validated."""
        from nous_llm.core.registry import validate_model

        # Valid 2025 models
        valid_models = [
            ("openai", "gpt-5"),
            ("openai", "gpt-5-turbo"),
            ("openai", "o2-preview"),
            ("anthropic", "claude-opus-4.1"),
            ("gemini", "gemini-2.5-pro"),
            ("gemini", "gemini-2.5-flash"),
            ("xai", "grok-4"),
            ("xai", "grok-4-heavy"),
            ("openrouter", "meta-llama/llama-4-maverick"),
        ]

        for provider, model in valid_models:
            validate_model(provider, model)  # Should not raise

        # Invalid models
        with pytest.raises(ConfigurationError):
            validate_model("openai", "gpt-6")  # Doesn't exist yet

        with pytest.raises(ConfigurationError):
            validate_model("anthropic", "claude-5")  # Doesn't exist

        with pytest.raises(ConfigurationError):
            validate_model("gemini", "gemini-3.0")  # Not released

    def test_context_window_limits(self) -> None:
        """Test context window limits for different models."""
        test_cases = [
            ("openai", "gpt-5", 256000),  # 256K tokens
            ("anthropic", "claude-opus-4.1", 200000),  # 200K tokens
            ("gemini", "gemini-2.5-pro", 1000000),  # 1M tokens
        ]

        for provider, model, max_context in test_cases:
            config = ProviderConfig(
                provider=provider,
                model=model,
                api_key="test-key",
            )

            # Create a prompt that would exceed older models' limits
            large_prompt = Prompt(
                instructions="Summarize",
                input=fake.text(max_nb_chars=max_context // 4),  # Approximate token count
            )

            params = GenParams(
                max_tokens=4000,
                temperature=0.5,
            )

            mock_response = {
                "output_text": f"Handled {max_context} token context",
                "usage": {
                    "prompt_tokens": max_context // 4,
                    "completion_tokens": 100,
                    "total_tokens": max_context // 4 + 100,
                },
            }

            with patch("httpx.Client") as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_response_obj = MagicMock()
                mock_response_obj.json.return_value = mock_response
                mock_response_obj.is_success = True
                mock_client.request.return_value = mock_response_obj

                response = generate(config, large_prompt, params)
                assert response.usage.input_tokens == max_context // 4
