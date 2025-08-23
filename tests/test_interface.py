"""
Tests for main interface functions.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nous_llm import LLMResponse, Usage, agenenerate, generate
from nous_llm.core.exceptions import ConfigurationError


class TestGenerate:
    """Tests for the generate function."""

    def test_generate_with_valid_config(
        self,
        sample_config,
        sample_prompt,
        sample_params,
    ) -> None:
        """Test generate function with valid configuration."""
        expected_response = LLMResponse(
            provider="openai",
            model="gpt-4o",
            text="Paris is the capital of France.",
            usage=Usage(input_tokens=15, output_tokens=8, total_tokens=23),
        )

        with (
            patch("unillm.core.interface.get_adapter") as mock_get_adapter,
            patch("unillm.core.interface.validate_model") as mock_validate,
        ):
            mock_adapter = MagicMock()
            mock_adapter.generate.return_value = expected_response
            mock_get_adapter.return_value = mock_adapter

            response = generate(sample_config, sample_prompt, sample_params)

            # Verify model validation was called
            mock_validate.assert_called_once_with("openai", "gpt-4o")

            # Verify adapter was obtained and called correctly
            mock_get_adapter.assert_called_once_with("openai")
            mock_adapter.generate.assert_called_once_with(
                sample_config,
                sample_prompt,
                sample_params,
            )

            assert response == expected_response

    def test_generate_with_default_params(
        self,
        sample_config,
        sample_prompt,
    ) -> None:
        """Test generate function with default parameters."""
        expected_response = LLMResponse(
            provider="openai",
            model="gpt-4o",
            text="Hello world!",
        )

        with (
            patch("unillm.core.interface.get_adapter") as mock_get_adapter,
            patch("unillm.core.interface.validate_model"),
        ):
            mock_adapter = MagicMock()
            mock_adapter.generate.return_value = expected_response
            mock_get_adapter.return_value = mock_adapter

            response = generate(sample_config, sample_prompt)

            # Verify adapter was called with default GenParams
            args, kwargs = mock_adapter.generate.call_args
            assert args[0] == sample_config
            assert args[1] == sample_prompt
            # Third argument should be default GenParams
            assert args[2].max_tokens == 512
            assert args[2].temperature == 0.7

            assert response == expected_response

    def test_generate_with_invalid_model(
        self,
        sample_config,
        sample_prompt,
    ) -> None:
        """Test generate function with invalid model."""
        with patch("unillm.core.interface.validate_model") as mock_validate:
            mock_validate.side_effect = ConfigurationError("Invalid model")

            with pytest.raises(ConfigurationError):
                generate(sample_config, sample_prompt)


class TestAgenerate:
    """Tests for the agenenerate function."""

    @pytest.mark.asyncio
    async def test_agenenerate_with_valid_config(
        self,
        sample_config,
        sample_prompt,
        sample_params,
    ) -> None:
        """Test agenenerate function with valid configuration."""
        expected_response = LLMResponse(
            provider="openai",
            model="gpt-4o",
            text="Paris is the capital of France.",
            usage=Usage(input_tokens=15, output_tokens=8, total_tokens=23),
        )

        with (
            patch("unillm.core.interface.get_adapter") as mock_get_adapter,
            patch("unillm.core.interface.validate_model") as mock_validate,
        ):
            mock_adapter = MagicMock()
            mock_adapter.agenenerate = AsyncMock(return_value=expected_response)
            mock_get_adapter.return_value = mock_adapter

            response = await agenenerate(sample_config, sample_prompt, sample_params)

            # Verify model validation was called
            mock_validate.assert_called_once_with("openai", "gpt-4o")

            # Verify adapter was obtained and called correctly
            mock_get_adapter.assert_called_once_with("openai")
            mock_adapter.agenenerate.assert_called_once_with(
                sample_config,
                sample_prompt,
                sample_params,
            )

            assert response == expected_response

    @pytest.mark.asyncio
    async def test_agenenerate_with_default_params(
        self,
        sample_config,
        sample_prompt,
    ) -> None:
        """Test agenenerate function with default parameters."""
        expected_response = LLMResponse(
            provider="openai",
            model="gpt-4o",
            text="Hello world!",
        )

        with (
            patch("unillm.core.interface.get_adapter") as mock_get_adapter,
            patch("unillm.core.interface.validate_model"),
        ):
            mock_adapter = MagicMock()
            mock_adapter.agenenerate = AsyncMock(return_value=expected_response)
            mock_get_adapter.return_value = mock_adapter

            response = await agenenerate(sample_config, sample_prompt)

            # Verify adapter was called with default GenParams
            args, kwargs = mock_adapter.agenenerate.call_args
            assert args[0] == sample_config
            assert args[1] == sample_prompt
            # Third argument should be default GenParams
            assert args[2].max_tokens == 512
            assert args[2].temperature == 0.7

            assert response == expected_response

    @pytest.mark.asyncio
    async def test_agenenerate_with_invalid_model(
        self,
        sample_config,
        sample_prompt,
    ) -> None:
        """Test agenenerate function with invalid model."""
        with patch("unillm.core.interface.validate_model") as mock_validate:
            mock_validate.side_effect = ConfigurationError("Invalid model")

            with pytest.raises(ConfigurationError):
                await agenenerate(sample_config, sample_prompt)
