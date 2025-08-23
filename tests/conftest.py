"""
Pytest configuration and shared fixtures for the test suite.
"""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from nous_llm import GenParams, Prompt, ProviderConfig


@pytest.fixture
def sample_config() -> ProviderConfig:
    """Sample provider configuration for testing."""
    return ProviderConfig(
        provider="openai",
        model="gpt-4o",
        api_key="test-api-key",
    )


@pytest.fixture
def sample_prompt() -> Prompt:
    """Sample prompt for testing."""
    return Prompt(
        instructions="You are a helpful assistant.",
        input="What is the capital of France?",
    )


@pytest.fixture
def sample_params() -> GenParams:
    """Sample generation parameters for testing."""
    return GenParams(
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        stop=["END"],
    )


@pytest.fixture
def openai_success_response() -> dict:
    """Mock successful OpenAI response."""
    return {
        "output_text": "Paris is the capital of France.",
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 8,
            "total_tokens": 23,
        },
    }


@pytest.fixture
def anthropic_success_response() -> dict:
    """Mock successful Anthropic response structure."""
    return {
        "id": "msg_test_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [
            {
                "type": "text",
                "text": "Paris is the capital of France.",
            }
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 15,
            "output_tokens": 8,
        },
    }


@pytest.fixture
def gemini_success_response() -> dict:
    """Mock successful Gemini response structure."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "Paris is the capital of France.",
                        }
                    ]
                },
                "finish_reason": "STOP",
                "safety_ratings": [],
            }
        ],
        "usage_metadata": {
            "prompt_token_count": 15,
            "candidates_token_count": 8,
            "total_token_count": 23,
        },
    }


@pytest.fixture
def mock_http_client():
    """Mock HTTP client using respx."""
    with respx.mock:
        yield respx


@pytest.fixture
def openai_error_response_401() -> Response:
    """Mock 401 error response from OpenAI."""
    return Response(
        status_code=401,
        json={"error": {"message": "Invalid API key", "type": "invalid_api_key"}},
    )


@pytest.fixture
def openai_error_response_429() -> Response:
    """Mock 429 rate limit response from OpenAI."""
    return Response(
        status_code=429,
        json={"error": {"message": "Rate limit exceeded", "type": "rate_limit_exceeded"}},
    )
