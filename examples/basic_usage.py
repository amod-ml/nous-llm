#!/usr/bin/env python3
"""Basic usage examples for the Universal LLM Wrapper.

This script demonstrates the core functionality of the unillm library
with different providers and configurations.
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from nous_llm import (
    GenParams,
    LLMClient,
    Prompt,
    ProviderConfig,
    agenenerate,
    generate,
)

# Load environment variables from .env file
load_dotenv()


def basic_openai_example() -> None:
    """Basic OpenAI example using functional interface."""
    print("=== Basic OpenAI Example ===")

    config = ProviderConfig(
        provider="openai",
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = Prompt(
        instructions="You are a helpful assistant that provides concise answers.",
        input="What is the capital of France?",
    )

    params = GenParams(
        max_tokens=50,
        temperature=0.7,
    )

    try:
        response = generate(config, prompt, params)
        print(f"Response: {response.text}")
        print(f"Usage: {response.usage}")
        print(f"Provider: {response.provider}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()


def client_based_example() -> None:
    """Example using the LLMClient convenience class."""
    print("=== Client-Based Example ===")

    # Create client with configuration
    client = LLMClient(
        ProviderConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    )

    prompt = Prompt(
        instructions="You are an expert in explaining complex topics simply.",
        input="Explain quantum computing in one sentence.",
    )

    try:
        response = client.generate(prompt)
        print(f"Response: {response.text}")
        print(f"Model: {response.model}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()


async def async_example() -> None:
    """Example using asynchronous interface."""
    print("=== Async Example ===")

    config = ProviderConfig(
        provider="gemini",
        model="gemini-1.5-pro",
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    prompt = Prompt(
        instructions="You are a creative writing assistant.",
        input="Write a haiku about coding.",
    )

    try:
        response = await agenenerate(config, prompt)
        print(f"Response: {response.text}")
        print(f"Provider: {response.provider}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()


def openrouter_example() -> None:
    """Example using OpenRouter to access multiple models."""
    print("=== OpenRouter Example ===")

    config = ProviderConfig(
        provider="openrouter",
        model="meta-llama/llama-3.1-405b",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    prompt = Prompt(
        instructions="You are a knowledgeable assistant.",
        input="What are the benefits of open-source AI models?",
    )

    params = GenParams(
        max_tokens=150,
        temperature=0.8,
    )

    try:
        response = generate(config, prompt, params)
        print(f"Response: {response.text}")
        print(f"Model: {response.model}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()


def xai_example() -> None:
    """Example using xAI's Grok model."""
    print("=== xAI Example ===")

    config = ProviderConfig(
        provider="xai",
        model="grok-beta",
        api_key=os.getenv("XAI_API_KEY"),
    )

    prompt = Prompt(
        instructions="You are Grok, a witty and helpful AI assistant.",
        input="What's the meaning of life?",
    )

    try:
        response = generate(config, prompt)
        print(f"Response: {response.text}")
        print(f"Provider: {response.provider}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()


def provider_specific_features() -> None:
    """Example demonstrating provider-specific features."""
    print("=== Provider-Specific Features ===")

    # OpenAI with reasoning (if supported)
    config = ProviderConfig(
        provider="openai",
        model="o1-preview",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = Prompt(
        instructions="Think step by step about this problem.",
        input="How would you design a simple chat application?",
    )

    params = GenParams(
        max_tokens=200,
        temperature=0.5,
        extra={"reasoning": True},  # OpenAI-specific feature
    )

    try:
        response = generate(config, prompt, params)
        print(f"Response: {response.text}")
        print(f"Raw data: {response.raw}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()


async def concurrent_requests() -> None:
    """Example showing concurrent requests to different providers."""
    print("=== Concurrent Requests ===")

    configs = [
        ProviderConfig(
            provider="openai",
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        ProviderConfig(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        ),
    ]

    prompt = Prompt(
        instructions="You are a helpful assistant.",
        input="Name one interesting fact about Python programming.",
    )

    # Run requests concurrently
    tasks = []
    for config in configs:
        if config.api_key:  # Only run if API key is available
            task = agenenerate(config, prompt)
            tasks.append(task)

    if tasks:
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    print(f"Provider {configs[i].provider} error: {response}")
                else:
                    print(f"Provider {response.provider}: {response.text}")
            print()
        except Exception as e:
            print(f"Error in concurrent requests: {e}")
            print()


def main() -> None:
    """Run all examples."""
    print("Universal LLM Wrapper Examples")
    print("=" * 40)
    print()

    # Check for required environment variables
    required_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "XAI_API_KEY",
        "OPENROUTER_API_KEY",
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Warning: Missing environment variables: {', '.join(missing_vars)}")
        print("Some examples may fail. Set these in your .env file or environment.")
        print()

    # Run synchronous examples
    basic_openai_example()
    client_based_example()
    openrouter_example()
    xai_example()
    provider_specific_features()

    # Run asynchronous examples
    asyncio.run(async_example())
    asyncio.run(concurrent_requests())

    print("Examples completed!")


if __name__ == "__main__":
    main()
