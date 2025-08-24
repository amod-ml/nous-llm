"""Tests for OpenAI parameter mapping and fallback mechanism.

This module specifically tests the logic for handling max_tokens vs max_completion_tokens
parameters across different OpenAI model families (GPT-4, GPT-5, O-series).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nous_llm import GenParams, Prompt, ProviderConfig, generate
from nous_llm.adapters.openai_adapter import OpenAIAdapter
from nous_llm.core.exceptions import ProviderError


class TestOpenAIParameterMapping:
    """Test parameter mapping logic for OpenAI models."""

    @pytest.fixture
    def adapter(self) -> OpenAIAdapter:
        """Create OpenAI adapter instance."""
        return OpenAIAdapter()

    @pytest.fixture
    def sample_prompt(self) -> Prompt:
        """Create sample prompt for testing."""
        return Prompt(
            instructions="You are a helpful assistant.",
            input="What is the capital of France?",
        )

    @pytest.fixture
    def sample_params(self) -> GenParams:
        """Create sample generation parameters."""
        return GenParams(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )

    def test_uses_completion_tokens_gpt5_models(self, adapter: OpenAIAdapter) -> None:
        """Test that GPT-5 series models use max_completion_tokens."""
        gpt5_models = [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5-chat",
            "gpt-5-turbo",
        ]

        for model in gpt5_models:
            assert adapter._uses_completion_tokens(model), f"Model {model} should use max_completion_tokens"

    def test_uses_completion_tokens_o_series_models(self, adapter: OpenAIAdapter) -> None:
        """Test that O-series models use max_completion_tokens."""
        o_series_models = [
            "o1",
            "o1-mini",
            "o3",
            "o3-mini",
            "o3-pro",
            "o4-mini",
            "o9-experimental",  # Future model
        ]

        for model in o_series_models:
            assert adapter._uses_completion_tokens(model), f"Model {model} should use max_completion_tokens"

    def test_uses_max_tokens_legacy_models(self, adapter: OpenAIAdapter) -> None:
        """Test that legacy models still use max_tokens."""
        legacy_models = [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "text-davinci-003",
            "text-ada-001",
        ]

        for model in legacy_models:
            assert not adapter._uses_completion_tokens(model), f"Model {model} should use max_tokens"

    def test_build_request_params_gpt5_uses_completion_tokens(
        self, adapter: OpenAIAdapter, sample_prompt: Prompt, sample_params: GenParams
    ) -> None:
        """Test that request params for GPT-5 models use max_completion_tokens."""
        config = ProviderConfig(provider="openai", model="gpt-5", api_key="test")

        params = adapter._build_request_params(config, sample_prompt, sample_params)

        assert "max_completion_tokens" in params
        assert "max_tokens" not in params
        assert params["max_completion_tokens"] == 100

    def test_build_request_params_gpt4_uses_max_tokens(
        self, adapter: OpenAIAdapter, sample_prompt: Prompt, sample_params: GenParams
    ) -> None:
        """Test that request params for GPT-4 models use max_tokens."""
        config = ProviderConfig(provider="openai", model="gpt-4", api_key="test")

        params = adapter._build_request_params(config, sample_prompt, sample_params)

        assert "max_tokens" in params
        assert "max_completion_tokens" not in params
        assert params["max_tokens"] == 100

    def test_build_request_params_o_series_uses_completion_tokens(
        self, adapter: OpenAIAdapter, sample_prompt: Prompt, sample_params: GenParams
    ) -> None:
        """Test that request params for O-series models use max_completion_tokens."""
        config = ProviderConfig(provider="openai", model="o3-mini", api_key="test")

        params = adapter._build_request_params(config, sample_prompt, sample_params)

        assert "max_completion_tokens" in params
        assert "max_tokens" not in params
        assert params["max_completion_tokens"] == 100

    @patch("openai.OpenAI")
    def test_fallback_mechanism_success(self, mock_openai, sample_prompt: Prompt, sample_params: GenParams) -> None:
        """Test that fallback mechanism works when max_tokens is not supported."""
        # Mock the first call to fail with the specific error
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # First call fails with max_tokens error
        first_error = Exception(
            "Error code: 400 - {'error': {'message': \"Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.\", 'type': 'invalid_request_error', 'param': 'max_tokens', 'code': 'unsupported_parameter'}}"
        )

        # Second call (with max_completion_tokens) succeeds
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Paris is the capital of France."))]
        mock_response.usage = MagicMock(prompt_tokens=15, completion_tokens=8, total_tokens=23)

        mock_client.chat.completions.create.side_effect = [first_error, mock_response]

        config = ProviderConfig(provider="openai", model="gpt-4", api_key="test")

        response = generate(config, sample_prompt, sample_params)

        # Verify it made two calls - first with max_tokens, then with max_completion_tokens
        assert mock_client.chat.completions.create.call_count == 2

        # Check first call had max_tokens
        first_call_kwargs = mock_client.chat.completions.create.call_args_list[0][1]
        assert "max_tokens" in first_call_kwargs
        assert "max_completion_tokens" not in first_call_kwargs

        # Check second call had max_completion_tokens
        second_call_kwargs = mock_client.chat.completions.create.call_args_list[1][1]
        assert "max_completion_tokens" in second_call_kwargs
        assert "max_tokens" not in second_call_kwargs

        # Verify response was parsed correctly
        assert response.text == "Paris is the capital of France."

    def test_fallback_mechanism_both_fail(self, adapter: OpenAIAdapter, sample_prompt: Prompt, sample_params: GenParams) -> None:
        """Test behavior when both max_tokens and max_completion_tokens fail."""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            # Both calls fail
            first_error = Exception(
                "Error code: 400 - {'error': {'message': \"Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.\", 'type': 'invalid_request_error', 'param': 'max_tokens', 'code': 'unsupported_parameter'}}"
            )
            second_error = Exception("Some other error with max_completion_tokens")

            mock_client.chat.completions.create.side_effect = [first_error, second_error]

            config = ProviderConfig(provider="openai", model="gpt-4", api_key="test")

            with pytest.raises(ProviderError):
                adapter.generate(config, sample_prompt, sample_params)

            # Verify it made two calls
            assert mock_client.chat.completions.create.call_count == 2

    def test_no_fallback_for_other_errors(self, adapter: OpenAIAdapter, sample_prompt: Prompt, sample_params: GenParams) -> None:
        """Test that fallback is not triggered for non-parameter errors."""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            # Error that should not trigger fallback
            auth_error = Exception("Error code: 401 - Invalid API key")
            mock_client.chat.completions.create.side_effect = auth_error

            config = ProviderConfig(provider="openai", model="gpt-4", api_key="test")

            with pytest.raises(Exception):  # Should raise the auth error, not try fallback
                adapter.generate(config, sample_prompt, sample_params)

            # Verify it only made one call (no fallback)
            assert mock_client.chat.completions.create.call_count == 1

    def test_parameter_mapping_case_insensitive(self, adapter: OpenAIAdapter) -> None:
        """Test that parameter mapping works regardless of case."""
        models_upper = ["GPT-5", "O3-MINI", "GPT-4"]
        models_mixed = ["GpT-5-MiNi", "o3-Pro", "gPt-4O"]

        # GPT-5 and O-series should use completion tokens regardless of case
        assert adapter._uses_completion_tokens("GPT-5")
        assert adapter._uses_completion_tokens("GpT-5-MiNi")
        assert adapter._uses_completion_tokens("O3-MINI")
        assert adapter._uses_completion_tokens("o3-Pro")

        # GPT-4 should not use completion tokens regardless of case
        assert not adapter._uses_completion_tokens("GPT-4")
        assert not adapter._uses_completion_tokens("gPt-4O")

    @patch("openai.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_async_fallback_mechanism(self, mock_async_openai, sample_prompt: Prompt, sample_params: GenParams) -> None:
        """Test that fallback mechanism works for async calls."""
        mock_client = MagicMock()
        mock_async_openai.return_value = mock_client

        # First call fails with max_tokens error
        first_error = Exception(
            "Error code: 400 - {'error': {'message': \"Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.\", 'type': 'invalid_request_error', 'param': 'max_tokens', 'code': 'unsupported_parameter'}}"
        )

        # Second call succeeds
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Paris is the capital of France."))]
        mock_response.usage = MagicMock(prompt_tokens=15, completion_tokens=8, total_tokens=23)

        # Create async mock
        async def mock_create_side_effect(*args, **kwargs):
            if mock_client.chat.completions.create.call_count == 0:
                mock_client.chat.completions.create.call_count += 1
                raise first_error
            else:
                return mock_response

        mock_client.chat.completions.create = MagicMock(side_effect=mock_create_side_effect)
        mock_client.chat.completions.create.call_count = 0

        adapter = OpenAIAdapter()
        config = ProviderConfig(provider="openai", model="gpt-4", api_key="test")

        response = await adapter.agenenerate(config, sample_prompt, sample_params)

        # Verify response was parsed correctly
        assert response.text == "Paris is the capital of France."
