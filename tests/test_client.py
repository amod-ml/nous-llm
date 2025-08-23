"""
Tests for the LLMClient convenience class.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nous_llm import LLMClient, LLMResponse, Usage


class TestLLMClient:
    """Tests for LLMClient class."""

    def test_client_initialization(self, sample_config) -> None:
        """Test client initialization with configuration."""
        client = LLMClient(sample_config)

        assert client.config == sample_config

    def test_client_generate(
        self,
        sample_config,
        sample_prompt,
        sample_params,
    ) -> None:
        """Test client generate method."""
        expected_response = LLMResponse(
            provider="openai",
            model="gpt-4o",
            text="Paris is the capital of France.",
            usage=Usage(input_tokens=15, output_tokens=8, total_tokens=23),
        )

        with patch("unillm.core.client.generate") as mock_generate:
            mock_generate.return_value = expected_response

            client = LLMClient(sample_config)
            response = client.generate(sample_prompt, sample_params)

            # Verify generate was called with correct arguments
            mock_generate.assert_called_once_with(
                sample_config,
                sample_prompt,
                sample_params,
            )

            assert response == expected_response

    def test_client_generate_with_default_params(
        self,
        sample_config,
        sample_prompt,
    ) -> None:
        """Test client generate method with default parameters."""
        expected_response = LLMResponse(
            provider="openai",
            model="gpt-4o",
            text="Hello world!",
        )

        with patch("unillm.core.client.generate") as mock_generate:
            mock_generate.return_value = expected_response

            client = LLMClient(sample_config)
            response = client.generate(sample_prompt)

            # Verify generate was called with None params
            mock_generate.assert_called_once_with(
                sample_config,
                sample_prompt,
                None,
            )

            assert response == expected_response

    @pytest.mark.asyncio
    async def test_client_agenenerate(
        self,
        sample_config,
        sample_prompt,
        sample_params,
    ) -> None:
        """Test client agenenerate method."""
        expected_response = LLMResponse(
            provider="openai",
            model="gpt-4o",
            text="Paris is the capital of France.",
            usage=Usage(input_tokens=15, output_tokens=8, total_tokens=23),
        )

        with patch("unillm.core.client.agenenerate") as mock_agenenerate:
            mock_agenenerate.return_value = expected_response

            client = LLMClient(sample_config)
            response = await client.agenenerate(sample_prompt, sample_params)

            # Verify agenenerate was called with correct arguments
            mock_agenenerate.assert_called_once_with(
                sample_config,
                sample_prompt,
                sample_params,
            )

            assert response == expected_response

    @pytest.mark.asyncio
    async def test_client_agenenerate_with_default_params(
        self,
        sample_config,
        sample_prompt,
    ) -> None:
        """Test client agenenerate method with default parameters."""
        expected_response = LLMResponse(
            provider="openai",
            model="gpt-4o",
            text="Hello world!",
        )

        with patch("unillm.core.client.agenenerate") as mock_agenenerate:
            mock_agenenerate.return_value = expected_response

            client = LLMClient(sample_config)
            response = await client.agenenerate(sample_prompt)

            # Verify agenenerate was called with None params
            mock_agenenerate.assert_called_once_with(
                sample_config,
                sample_prompt,
                None,
            )

            assert response == expected_response

    def test_update_config(self, sample_config) -> None:
        """Test updating client configuration."""
        client = LLMClient(sample_config)

        # Update model
        new_client = client.update_config(model="gpt-3.5-turbo")

        # Original client should be unchanged
        assert client.config.model == "gpt-4o"

        # New client should have updated model
        assert new_client.config.model == "gpt-3.5-turbo"
        assert new_client.config.provider == "openai"  # Other fields unchanged
        assert new_client.config.api_key == "test-api-key"

    def test_update_config_multiple_fields(self, sample_config) -> None:
        """Test updating multiple configuration fields."""
        client = LLMClient(sample_config)

        new_client = client.update_config(
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            api_key="new-key",
        )

        assert new_client.config.model == "claude-3-5-sonnet-20241022"
        assert new_client.config.provider == "anthropic"
        assert new_client.config.api_key == "new-key"

    def test_client_repr(self, sample_config) -> None:
        """Test client string representation."""
        client = LLMClient(sample_config)

        repr_str = repr(client)
        assert "LLMClient" in repr_str
        assert "provider=openai" in repr_str
        assert "model=gpt-4o" in repr_str
