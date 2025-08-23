#!/usr/bin/env python3
"""AWS Lambda example for the Universal LLM Wrapper.

This example shows how to use the unillm library in an AWS Lambda function
for serverless LLM inference with minimal cold start overhead.
"""

from __future__ import annotations

import json
import os
from typing import Any

from nous_llm import GenParams, LLMClient, Prompt, ProviderConfig
from nous_llm.core.exceptions import UnillmError

# Global client instance for connection reuse across invocations
_client: LLMClient | None = None


def get_client() -> LLMClient:
    """Get or create a cached LLM client for connection reuse."""
    global _client

    if _client is None:
        # Read configuration from environment variables
        provider = os.environ["LLM_PROVIDER"]  # e.g., "openai"
        model = os.environ["LLM_MODEL"]  # e.g., "gpt-4o"
        api_key = os.environ["LLM_API_KEY"]  # Provider API key

        config = ProviderConfig(
            provider=provider,
            model=model,
            api_key=api_key,
        )

        _client = LLMClient(config)

    return _client


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda handler for LLM text generation.

    Expected event format:
    {
        "instructions": "You are a helpful assistant.",
        "input": "What is the capital of France?",
        "max_tokens": 100,
        "temperature": 0.7
    }

    Returns:
    {
        "statusCode": 200,
        "body": {
            "text": "Generated response text",
            "provider": "openai",
            "model": "gpt-4o",
            "usage": {
                "input_tokens": 15,
                "output_tokens": 8,
                "total_tokens": 23
            }
        }
    }
    """
    try:
        # Parse request
        instructions = event.get("instructions", "You are a helpful assistant.")
        user_input = event.get("input", "")

        if not user_input:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing required field: input"})}

        # Create prompt
        prompt = Prompt(
            instructions=instructions,
            input=user_input,
        )

        # Parse optional parameters
        params = GenParams(
            max_tokens=event.get("max_tokens", 512),
            temperature=event.get("temperature", 0.7),
            top_p=event.get("top_p"),
            stop=event.get("stop"),
            extra=event.get("extra", {}),
        )

        # Generate response
        client = get_client()
        response = client.generate(prompt, params)

        # Format response
        response_body = {
            "text": response.text,
            "provider": response.provider,
            "model": response.model,
        }

        # Include usage if available
        if response.usage:
            response_body["usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",  # Enable CORS if needed
            },
            "body": json.dumps(response_body),
        }

    except UnillmError as e:
        # Handle library-specific errors
        error_response = {
            "error": str(e),
            "error_type": type(e).__name__,
            "provider": getattr(e, "provider", None),
        }

        # Map error types to HTTP status codes
        if "Auth" in type(e).__name__:
            status_code = 401
        elif "RateLimit" in type(e).__name__:
            status_code = 429
        elif "Configuration" in type(e).__name__ or "Validation" in type(e).__name__:
            status_code = 400
        else:
            status_code = 500

        return {
            "statusCode": status_code,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(error_response),
        }

    except Exception as e:
        # Handle unexpected errors
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(
                {
                    "error": "Internal server error",
                    "details": str(e),
                }
            ),
        }


# Example for testing locally
def test_locally() -> None:
    """Test the Lambda function locally."""
    # Set environment variables for testing
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["LLM_MODEL"] = "gpt-4o-mini"
    os.environ["LLM_API_KEY"] = "your-api-key-here"

    # Test event
    test_event = {
        "instructions": "You are a helpful assistant that provides concise answers.",
        "input": "What is the capital of France?",
        "max_tokens": 50,
        "temperature": 0.7,
    }

    # Call handler
    result = lambda_handler(test_event, None)

    print("Lambda Response:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    test_locally()
