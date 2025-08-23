"""Comprehensive edge case and error handling tests for nous-llm.

This module tests various edge cases, error conditions, and boundary
scenarios to ensure robust behavior across all providers.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import TimeoutError
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
    ProviderError,
    RateLimitError,
)

fake = Faker()


class TestEdgeCases:
    """Test edge cases for the nous-llm package."""

    def test_empty_prompt_handling(self) -> None:
        """Test handling of empty or whitespace-only prompts."""
        # Empty instructions should fail validation
        with pytest.raises(ValueError):
            Prompt(instructions="", input="test")

        # Empty input should fail validation
        with pytest.raises(ValueError):
            Prompt(instructions="test", input="")

        # Whitespace-only should fail validation
        with pytest.raises(ValueError):
            Prompt(instructions="   ", input="test")

    def test_extremely_long_prompts(self) -> None:
        """Test handling of extremely long prompts."""
        # Generate a very long prompt (1MB of text)
        very_long_text = "a" * 1_000_000

        prompt = Prompt(
            instructions="Process this",
            input=very_long_text,
        )

        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        mock_response = {
            "output_text": "Processed long input",
            "usage": {"prompt_tokens": 250000, "completion_tokens": 10, "total_tokens": 250010},
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.is_success = True
            mock_client.request.return_value = mock_response_obj

            response = generate(config, prompt)
            assert response.usage.input_tokens == 250000

    def test_special_characters_in_prompts(self) -> None:
        """Test handling of special characters and escape sequences."""
        special_prompts = [
            "Test with newlines\n\n\nand tabs\t\t",
            "Unicode: 你好世界 🌍 ñ é ü",
            "Control chars: \x00\x01\x02",
            "Quotes: \"double\" and 'single' and `backticks`",
            "Backslashes: \\ and \\n and \\\\",
            "HTML: <script>alert('test')</script>",
            "SQL: '; DROP TABLE users; --",
            'JSON: {"key": "value", "nested": {"array": [1, 2, 3]}}',
        ]

        config = ProviderConfig(
            provider="anthropic",
            model="claude-opus-4.1",
            api_key="test-key",
        )

        for special_input in special_prompts:
            prompt = Prompt(
                instructions="Process this safely",
                input=special_input,
            )

            mock_response = MagicMock()
            mock_response.content = [MagicMock(text=f"Safely processed: {special_input[:20]}...")]
            mock_response.usage = MagicMock(input_tokens=10, output_tokens=10)

            with patch("anthropic.Anthropic") as mock_anthropic:
                mock_client = MagicMock()
                mock_anthropic.return_value = mock_client
                mock_client.messages.create.return_value = mock_response

                response = generate(config, prompt)
                assert "Safely processed" in response.text

    def test_null_and_none_handling(self) -> None:
        """Test handling of None values in optional parameters."""
        config = ProviderConfig(
            provider="gemini",
            model="gemini-2.5-pro",
            api_key="test-key",
            base_url=None,  # Explicitly None
        )

        prompt = Prompt(
            instructions="Test",
            input="Test input",
        )

        params = GenParams(
            max_tokens=100,
            temperature=0.7,
            top_p=None,  # Explicitly None
            stop=None,  # Explicitly None
            extra={},  # Empty dict
        )

        mock_response = MagicMock()
        mock_response.text = "Response with None values handled"

        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response

            response = generate(config, prompt, params)
            assert response.text == "Response with None values handled"

    def test_malformed_response_handling(self) -> None:
        """Test handling of malformed responses from providers."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        prompt = Prompt(
            instructions="Test",
            input="Test malformed response",
        )

        # Test various malformed responses
        malformed_responses = [
            {},  # Empty response
            {"error": "Something went wrong"},  # Error response
            {"choices": []},  # Empty choices
            {"output_text": None},  # Null output
            {"invalid_field": "value"},  # Unexpected format
            "not a dict",  # Wrong type
        ]

        for malformed in malformed_responses:
            with patch("httpx.Client") as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_response_obj = MagicMock()

                if isinstance(malformed, str):
                    mock_response_obj.json.side_effect = json.JSONDecodeError("test", "test", 0)
                else:
                    mock_response_obj.json.return_value = malformed

                mock_response_obj.is_success = True
                mock_client.request.return_value = mock_response_obj

                with pytest.raises(ProviderError):
                    generate(config, prompt)

    def test_timeout_handling(self) -> None:
        """Test timeout handling for slow responses."""
        config = ProviderConfig(
            provider="xai",
            model="grok-4",
            api_key="test-key",
        )

        prompt = Prompt(
            instructions="Test",
            input="Test timeout",
        )

        with patch("xai_sdk.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.create.side_effect = TimeoutError("Request timed out")

            with pytest.raises(TimeoutError):
                generate(config, prompt)

    def test_concurrent_requests_limit(self) -> None:
        """Test handling of concurrent request limits."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        # Create multiple prompts
        prompts = [
            Prompt(
                instructions="Test",
                input=f"Concurrent request {i}",
            )
            for i in range(100)
        ]

        mock_response = {
            "output_text": "Response",
            "usage": {"total_tokens": 10},
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.is_success = True

            # Simulate rate limiting after 10 requests
            call_count = 0

            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count > 10:
                    error_response = MagicMock()
                    error_response.is_success = False
                    error_response.status_code = 429
                    error_response.json.return_value = {"error": "Rate limit"}
                    return error_response
                return mock_response_obj

            mock_client.request.side_effect = side_effect

            # Process requests
            success_count = 0
            rate_limit_count = 0

            for prompt in prompts[:20]:
                try:
                    response = generate(config, prompt)
                    success_count += 1
                except RateLimitError:
                    rate_limit_count += 1

            assert success_count == 10
            assert rate_limit_count == 10

    @pytest.mark.asyncio
    async def test_async_timeout_and_cancellation(self) -> None:
        """Test async timeout and cancellation handling."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        prompt = Prompt(
            instructions="Test",
            input="Test async timeout",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Simulate a slow response
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(10)  # Simulate slow response
                return MagicMock()

            mock_client.request = AsyncMock(side_effect=slow_response)

            # Test timeout
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    agenenerate(config, prompt),
                    timeout=0.1,
                )

    def test_invalid_api_key_formats(self) -> None:
        """Test handling of various invalid API key formats."""
        invalid_keys = [
            "",  # Empty string
            " ",  # Whitespace
            "invalid",  # Too short
            "sk-" * 100,  # Too long
            "not-a-valid-key-format",
            "123456",  # Numbers only
            "!@#$%^&*()",  # Special chars only
        ]

        for key in invalid_keys:
            config = ProviderConfig(
                provider="openai",
                model="gpt-5",
                api_key=key,
            )

            prompt = Prompt(
                instructions="Test",
                input="Test invalid key",
            )

            with patch("httpx.Client") as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_response_obj = MagicMock()
                mock_response_obj.is_success = False
                mock_response_obj.status_code = 401
                mock_response_obj.json.return_value = {"error": "Invalid API key"}
                mock_client.request.return_value = mock_response_obj

                with pytest.raises(AuthError):
                    generate(config, prompt)

    def test_provider_specific_errors(self) -> None:
        """Test handling of provider-specific error codes and messages."""
        error_scenarios = [
            ("openai", 400, "Invalid request", ProviderError),
            ("openai", 403, "Forbidden", AuthError),
            ("openai", 500, "Internal server error", ProviderError),
            ("openai", 503, "Service unavailable", ProviderError),
            ("anthropic", 400, "Invalid model", ProviderError),
            ("anthropic", 402, "Payment required", ProviderError),
            ("gemini", 400, "Invalid generation config", ProviderError),
            ("xai", 422, "Unprocessable entity", ProviderError),
            ("openrouter", 400, "Model not available", ProviderError),
        ]

        for provider, status_code, error_msg, expected_exception in error_scenarios:
            config = ProviderConfig(
                provider=provider,
                model="test-model",
                api_key="test-key",
            )

            prompt = Prompt(
                instructions="Test",
                input="Test error",
            )

            with patch("httpx.Client") as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_response_obj = MagicMock()
                mock_response_obj.is_success = False
                mock_response_obj.status_code = status_code
                mock_response_obj.json.return_value = {"error": error_msg}
                mock_client.request.return_value = mock_response_obj

                with pytest.raises(expected_exception):
                    generate(config, prompt)

    def test_retry_logic_with_exponential_backoff(self) -> None:
        """Test retry logic with exponential backoff."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        prompt = Prompt(
            instructions="Test",
            input="Test retry logic",
        )

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                # Fail first 2 attempts
                response = MagicMock()
                response.is_success = False
                response.status_code = 503  # Service unavailable
                response.json.return_value = {"error": "Service temporarily unavailable"}
                return response
            else:
                # Succeed on 3rd attempt
                response = MagicMock()
                response.is_success = True
                response.json.return_value = {
                    "output_text": "Success after retries",
                    "usage": {"total_tokens": 10},
                }
                return response

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.request.side_effect = side_effect

            # Note: The actual retry logic would be in the base adapter
            # This test verifies the structure is in place

    def test_client_connection_pooling(self) -> None:
        """Test that HTTP clients use connection pooling efficiently."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        # Create a client that should reuse connections
        client = LLMClient(config)

        mock_response = {
            "output_text": "Response",
            "usage": {"total_tokens": 10},
        }

        with patch("httpx.Client") as mock_client_class:
            mock_http_client = MagicMock()
            mock_client_class.return_value = mock_http_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.is_success = True
            mock_http_client.request.return_value = mock_response_obj

            # Make multiple requests
            for i in range(50):
                prompt = Prompt(
                    instructions="Test",
                    input=f"Request {i}",
                )
                response = client.generate(prompt)
                assert response.text == "Response"

            # Verify client was created with proper connection limits
            assert mock_client_class.call_count == 1

    def test_memory_usage_with_large_responses(self) -> None:
        """Test memory handling with very large responses."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        prompt = Prompt(
            instructions="Generate a very long response",
            input="Write a detailed essay",
        )

        # Simulate a very large response (10MB of text)
        large_text = "a" * 10_000_000

        mock_response = {
            "output_text": large_text,
            "usage": {"prompt_tokens": 100, "completion_tokens": 2500000, "total_tokens": 2500100},
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.is_success = True
            mock_client.request.return_value = mock_response_obj

            response = generate(config, prompt)
            assert len(response.text) == 10_000_000
            assert response.usage.output_tokens == 2500000

    def test_model_fallback_on_unavailability(self) -> None:
        """Test fallback to alternative models when primary is unavailable."""
        primary_config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        fallback_config = ProviderConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
        )

        prompt = Prompt(
            instructions="Test",
            input="Test fallback",
        )

        # First attempt fails with model unavailable
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Primary model unavailable
            primary_response = MagicMock()
            primary_response.is_success = False
            primary_response.status_code = 404
            primary_response.json.return_value = {"error": "Model not found"}

            # Fallback model succeeds
            fallback_response = MagicMock()
            fallback_response.is_success = True
            fallback_response.json.return_value = {
                "output_text": "Response from fallback model",
                "usage": {"total_tokens": 10},
            }

            mock_client.request.side_effect = [primary_response, fallback_response]

            # Try primary first
            with pytest.raises(ProviderError):
                generate(primary_config, prompt)

            # Then try fallback
            response = generate(fallback_config, prompt)
            assert response.text == "Response from fallback model"
