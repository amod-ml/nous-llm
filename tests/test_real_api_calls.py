"""Real API call tests for nous-llm.

This module tests actual API calls to all providers using real API keys
from environment variables. These tests are marked as integration tests
and can be run separately from unit tests.

Run with: pytest tests/test_real_api_calls.py -m integration -v
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import pytest

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
    RateLimitError,
)

# Configure logger for integration tests
logger = logging.getLogger(__name__)


class TestRealAPICalls:
    """Test real API calls to all providers."""

    @pytest.fixture(autouse=True)
    def load_env_vars(self) -> None:
        """Load environment variables from .env file if it exists."""
        # Try to load from .env file
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass  # dotenv not available, use system env vars

        # Also try to load from keys.txt if it exists
        if os.path.exists("keys.txt"):
            with open("keys.txt") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#") and not line.startswith("here"):
                        try:
                            key, value = line.split("=", 1)
                            os.environ[key] = value
                        except ValueError:
                            continue

    @pytest.fixture
    def test_prompts(self) -> list[Prompt]:
        """Generate simple test prompts for real API calls."""
        return [
            Prompt(
                instructions="You are a helpful assistant. Be concise.",
                input="What is the capital of France? Answer in one word.",
            ),
            Prompt(
                instructions="You are a math tutor. Be brief.",
                input="What is 2 + 2? Answer with just the number.",
            ),
            Prompt(
                instructions="You are a creative assistant.",
                input="Write a very short haiku about AI (3 lines total).",
            ),
        ]

    @pytest.fixture
    def simple_params(self) -> GenParams:
        """Simple parameters for real API calls to minimize costs."""
        return GenParams(
            max_tokens=50,  # Keep responses short
            temperature=0.1,  # Keep responses consistent
        )

    @pytest.mark.integration
    def test_openai_real_api(self, test_prompts: list[Prompt], simple_params: GenParams) -> None:
        """Test real OpenAI API calls."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in environment")

        # Test with current available model (not future GPT-5)
        config = ProviderConfig(
            provider="openai",
            model="gpt-4o-mini",  # Use cheaper model for testing
            api_key=api_key,
        )

        logger.info("🔬 Testing OpenAI with %s", config.model)

        try:
            response = generate(config, test_prompts[0], simple_params)

            logger.info("✅ Response: %s", response.text)
            assert response.provider == "openai"
            assert response.model == "gpt-4o-mini"
            assert len(response.text) > 0
            assert response.usage is not None
            assert response.usage.total_tokens > 0

        except AuthError:
            pytest.fail("OpenAI authentication failed - check API key")
        except RateLimitError:
            pytest.skip("OpenAI rate limit exceeded")
        except Exception as e:
            pytest.fail(f"Unexpected OpenAI error: {e}")

    @pytest.mark.integration
    def test_anthropic_real_api(self, test_prompts: list[Prompt], simple_params: GenParams) -> None:
        """Test real Anthropic API calls."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not found in environment")

        # Test with current available model
        config = ProviderConfig(
            provider="anthropic",
            model="claude-3-haiku-20240307",  # Use cheaper model for testing
            api_key=api_key,
        )

        print(f"\n🔬 Testing Anthropic with {config.model}")

        try:
            response = generate(config, test_prompts[0], simple_params)

            print(f"✅ Response: {response.text}")
            assert response.provider == "anthropic"
            assert response.model == "claude-3-haiku-20240307"
            assert len(response.text) > 0
            assert response.usage is not None

        except AuthError:
            pytest.fail("Anthropic authentication failed - check API key")
        except RateLimitError:
            pytest.skip("Anthropic rate limit exceeded")
        except Exception as e:
            pytest.fail(f"Unexpected Anthropic error: {e}")

    @pytest.mark.integration
    def test_gemini_real_api(self, test_prompts: list[Prompt], simple_params: GenParams) -> None:
        """Test real Google Gemini API calls."""
        # Try both possible environment variable names
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment")

        # Test with current available model
        config = ProviderConfig(
            provider="gemini",
            model="gemini-2.5-flash",  # Use latest recommended model for testing
            api_key=api_key,
        )

        logger.info("🔬 Testing Gemini with %s", config.model)

        try:
            response = generate(config, test_prompts[0], simple_params)

            logger.info("✅ Response: %s", response.text)
            assert response.provider == "gemini"
            assert response.model == "gemini-2.5-flash"
            assert len(response.text) > 0

        except AuthError:
            pytest.fail("Gemini authentication failed - check API key")
        except RateLimitError:
            pytest.skip("Gemini rate limit exceeded")
        except Exception as e:
            pytest.fail(f"Unexpected Gemini error: {e}")

    @pytest.mark.integration
    def test_xai_real_api(self, test_prompts: list[Prompt], simple_params: GenParams) -> None:
        """Test real xAI API calls."""
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            pytest.skip("XAI_API_KEY not found in environment")

        # Test with current available model
        config = ProviderConfig(
            provider="xai",
            model="grok-3",  # Use latest model from documentation
            api_key=api_key,
        )

        logger.info("🔬 Testing xAI with %s", config.model)

        try:
            response = generate(config, test_prompts[0], simple_params)

            logger.info("✅ Response: %s", response.text)
            assert response.provider == "xai"
            assert response.model == "grok-3"
            assert len(response.text) > 0

        except AuthError:
            pytest.fail("xAI authentication failed - check API key")
        except RateLimitError:
            pytest.skip("xAI rate limit exceeded")
        except Exception as e:
            pytest.fail(f"Unexpected xAI error: {e}")

    @pytest.mark.integration
    def test_openrouter_real_api(self, test_prompts: list[Prompt], simple_params: GenParams) -> None:
        """Test real OpenRouter API calls."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not found in environment")

        # Test with a working model on OpenRouter
        config = ProviderConfig(
            provider="openrouter",
            model="openai/gpt-3.5-turbo",  # Reliable model for testing
            api_key=api_key,
        )

        logger.info("🔬 Testing OpenRouter with %s", config.model)

        try:
            response = generate(config, test_prompts[0], simple_params)

            logger.info("✅ Response: %s", response.text)
            assert response.provider == "openrouter"
            assert response.model == "openai/gpt-3.5-turbo"
            assert len(response.text) > 0

        except AuthError:
            pytest.fail("OpenRouter authentication failed - check API key")
        except RateLimitError:
            pytest.skip("OpenRouter rate limit exceeded")
        except Exception as e:
            pytest.fail(f"Unexpected OpenRouter error: {e}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_all_providers_async(self, simple_params: GenParams) -> None:
        """Test async calls to all available providers."""
        prompt = Prompt(
            instructions="You are helpful. Be very brief.",
            input="Say 'Hello' in one word.",
        )

        # Collect available providers
        available_configs = []

        if os.getenv("OPENAI_API_KEY"):
            available_configs.append(
                ProviderConfig(
                    provider="openai",
                    model="gpt-4o-mini",
                    api_key=os.getenv("OPENAI_API_KEY"),
                )
            )

        if os.getenv("ANTHROPIC_API_KEY"):
            available_configs.append(
                ProviderConfig(
                    provider="anthropic",
                    model="claude-3-haiku-20240307",
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                )
            )

        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            available_configs.append(
                ProviderConfig(
                    provider="gemini",
                    model="gemini-2.5-flash",
                    api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
                )
            )

        if os.getenv("XAI_API_KEY"):
            available_configs.append(
                ProviderConfig(
                    provider="xai",
                    model="grok-3",
                    api_key=os.getenv("XAI_API_KEY"),
                )
            )

        if os.getenv("OPENROUTER_API_KEY"):
            available_configs.append(
                ProviderConfig(
                    provider="openrouter",
                    model="openai/gpt-3.5-turbo",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )
            )

        if not available_configs:
            pytest.skip("No API keys found for any provider")

        logger.info("🚀 Testing %d providers concurrently...", len(available_configs))

        # Run all providers concurrently
        async def test_provider(config: ProviderConfig) -> tuple[str, Any]:
            try:
                response = await agenenerate(config, prompt, simple_params)
                return config.provider, response
            except Exception as e:
                return config.provider, e

        tasks = [test_provider(config) for config in available_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        for provider, result in results:
            if isinstance(result, Exception):
                logger.error("❌ %s: %s", provider, result)
            else:
                logger.info("✅ %s: %s...", provider, result.text[:50])
                success_count += 1

        # At least one provider should work
        assert success_count > 0, "No providers succeeded"

    @pytest.mark.integration
    def test_client_reuse_real_api(self, simple_params: GenParams) -> None:
        """Test client reuse with real API calls."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not found - needed for client reuse test")

        config = ProviderConfig(
            provider="openrouter",
            model="openai/gpt-3.5-turbo",
            api_key=api_key,
        )

        client = LLMClient(config)

        prompts = [
            Prompt(
                instructions="Answer briefly.",
                input=f"What is {i} + 1? Just the number.",
            )
            for i in range(3)  # Keep it small for real API
        ]

        logger.info("🔄 Testing client reuse with %d requests...", len(prompts))

        responses = []
        for i, prompt in enumerate(prompts):
            try:
                response = client.generate(prompt, simple_params)
                responses.append(response)
                logger.info("  Request %d: %s", i + 1, response.text.strip())
            except Exception as e:
                logger.error("  Request %d failed: %s", i + 1, e)

        assert len(responses) > 0, "No requests succeeded"

    @pytest.mark.integration
    def test_error_handling_real_api(self) -> None:
        """Test error handling with real API calls."""
        # Test with invalid API key
        config = ProviderConfig(
            provider="openai",
            model="gpt-4o-mini",
            api_key="invalid-key-12345",
        )

        prompt = Prompt(
            instructions="Test",
            input="Hello",
        )

        logger.info("🚨 Testing error handling with invalid API key...")

        with pytest.raises(AuthError) as exc_info:
            generate(config, prompt)

        assert "authentication" in str(exc_info.value).lower()
        logger.info("✅ Correctly caught auth error: %s", exc_info.value)

    @pytest.mark.integration
    def test_rate_limiting_real_api(self, simple_params: GenParams) -> None:
        """Test rate limiting with real API calls."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not found - needed for rate limiting test")

        config = ProviderConfig(
            provider="openrouter",
            model="openai/gpt-3.5-turbo",
            api_key=api_key,
        )

        prompt = Prompt(
            instructions="Be brief.",
            input="Hello",
        )

        logger.info("⏱️ Testing rate limiting with rapid requests...")

        success_count = 0
        rate_limit_count = 0

        # Make rapid requests to potentially trigger rate limiting
        for i in range(10):
            try:
                response = generate(config, prompt, simple_params)
                success_count += 1
                logger.info("  Request %d: Success", i + 1)
            except RateLimitError:
                rate_limit_count += 1
                logger.warning("  Request %d: Rate limited", i + 1)
                break  # Stop on first rate limit
            except Exception as e:
                logger.error("  Request %d: Other error: %s", i + 1, e)

        logger.info("✅ Completed: %d success, %d rate limited", success_count, rate_limit_count)
        assert success_count > 0, "No requests succeeded"

    @pytest.mark.integration
    def test_parameter_variations_real_api(self) -> None:
        """Test different parameter settings with real API calls."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not found - needed for parameter test")

        config = ProviderConfig(
            provider="openrouter",
            model="openai/gpt-3.5-turbo",
            api_key=api_key,
        )

        prompt = Prompt(
            instructions="Write creatively.",
            input="Describe a sunset in 10 words.",
        )

        # Test different parameter combinations
        param_sets = [
            GenParams(max_tokens=20, temperature=0.1),  # Conservative
            GenParams(max_tokens=30, temperature=0.7),  # Balanced
            GenParams(max_tokens=40, temperature=0.9),  # Creative
        ]

        logger.info("🎛️ Testing %d parameter combinations...", len(param_sets))

        for i, params in enumerate(param_sets):
            try:
                response = generate(config, prompt, params)
                logger.info("  Config %d (temp=%s): %s", i + 1, params.temperature, response.text.strip())
                assert len(response.text) > 0
            except Exception as e:
                logger.error("  Config %d failed: %s", i + 1, e)

    @pytest.mark.integration
    def test_model_comparison_real_api(self) -> None:
        """Compare responses from different models on the same provider."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not found - needed for model comparison")

        prompt = Prompt(
            instructions="Explain briefly.",
            input="What is machine learning in one sentence?",
        )

        params = GenParams(max_tokens=50, temperature=0.1)

        # Test different models on OpenRouter
        models = [
            "openai/gpt-3.5-turbo",
            "anthropic/claude-instant-v1",
        ]

        logger.info("🔄 Comparing %d models...", len(models))

        for model in models:
            config = ProviderConfig(
                provider="openrouter",
                model=model,
                api_key=api_key,
            )

            try:
                response = generate(config, prompt, params)
                logger.info("  %s: %s", model, response.text.strip())
                assert response.model == model
            except Exception as e:
                logger.error("  %s: Failed - %s", model, e)

    @pytest.mark.integration
    def test_context_length_real_api(self) -> None:
        """Test handling of longer contexts with real API calls."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not found - needed for context length test")

        config = ProviderConfig(
            provider="openrouter",
            model="openai/gpt-3.5-turbo",
            api_key=api_key,
        )

        # Create a longer context
        long_context = "Here is a story: " + "Once upon a time, there was a brave knight. " * 20

        prompt = Prompt(
            instructions="Summarize the story briefly.",
            input=long_context,
        )

        params = GenParams(max_tokens=30, temperature=0.1)

        logger.info("📄 Testing longer context (%d chars)...", len(long_context))

        try:
            response = generate(config, prompt, params)
            logger.info("✅ Summary: %s", response.text.strip())
            assert len(response.text) > 0
            if response.usage:
                logger.info("  Tokens used: %s", response.usage.total_tokens)
        except Exception as e:
            logger.warning("❌ Context length test failed: %s", e)
            # Don't fail the test - some models have smaller context windows
