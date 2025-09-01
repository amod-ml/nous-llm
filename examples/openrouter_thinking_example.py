#!/usr/bin/env python3
"""Example demonstrating OpenRouter thinking functionality with nous-llm."""

import os

from src.nous_llm.adapters.openrouter_adapter import OpenRouterAdapter
from src.nous_llm.core.models import GenParams, Prompt, ProviderConfig


def main():
    """Demonstrate OpenRouter thinking functionality across different model types."""
    # Set up API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable")
        return

    # Initialize OpenRouter adapter
    adapter = OpenRouterAdapter()

    print("🧠 OpenRouter Thinking Functionality Examples")
    print("=" * 60)

    # Example 1: OpenAI o1 with effort-based reasoning
    print("\n🤖 Example 1: OpenAI o1 with Effort-Based Reasoning")
    print("-" * 50)

    config = ProviderConfig(model="openai/o1-preview", api_key=api_key)

    prompt = Prompt(
        instructions="You are a math tutor. Show your reasoning process clearly.",
        input="A train travels 120 km in 2 hours, then 180 km in 3 hours. What is the average speed for the entire journey?",
    )

    # Use effort-based reasoning (OpenAI style)
    params = GenParams(
        max_tokens=2000,
        temperature=1.0,  # O-series models require temperature=1.0
        extra={
            "reasoning_effort": "high",  # High reasoning effort
            "reasoning_exclude": False,  # Include reasoning in response
        },
    )

    try:
        response = adapter.generate(config, prompt, params)
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        print("\nResponse:")
        print(response.text[:1000] + "..." if len(response.text) > 1000 else response.text)
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Anthropic Claude with max_tokens reasoning
    print("\n\n🤖 Example 2: Anthropic Claude with Max Tokens Reasoning")
    print("-" * 50)

    config = ProviderConfig(model="anthropic/claude-3-5-sonnet", api_key=api_key)

    prompt = Prompt(
        instructions="You are a strategic consultant. Think through this problem systematically.",
        input="A startup has $100k funding and burns $15k/month. They expect revenue to grow 20% monthly starting at $5k. When will they break even?",
    )

    # Use max_tokens reasoning (Anthropic style)
    params = GenParams(
        max_tokens=1500,
        temperature=0.3,
        extra={
            "reasoning_max_tokens": 6000,  # Specific token budget for reasoning
            "reasoning_exclude": False,  # Show reasoning process
        },
    )

    try:
        response = adapter.generate(config, prompt, params)
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        print("\nResponse:")
        print(response.text[:1000] + "..." if len(response.text) > 1000 else response.text)
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: xAI Grok with effort levels
    print("\n\n🤖 Example 3: xAI Grok with Effort Levels")
    print("-" * 50)

    config = ProviderConfig(model="xai/grok-beta", api_key=api_key)

    prompt = Prompt(
        instructions="You are a creative problem solver. Think outside the box.",
        input="Design a sustainable transportation system for a city of 1 million people. Consider environmental, economic, and social factors.",
    )

    # Use medium effort reasoning
    params = GenParams(
        max_tokens=2000,
        temperature=0.7,
        extra={
            "reasoning_effort": "medium",  # Medium reasoning effort
            "reasoning_exclude": False,  # Include reasoning
        },
    )

    try:
        response = adapter.generate(config, prompt, params)
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        print("\nResponse:")
        print(response.text[:1000] + "..." if len(response.text) > 1000 else response.text)
    except Exception as e:
        print(f"Error: {e}")

    # Example 4: Legacy parameter support (backward compatibility)
    print("\n\n🤖 Example 4: Legacy Parameter Support")
    print("-" * 50)

    config = ProviderConfig(model="openai/o3-mini", api_key=api_key)

    prompt = Prompt(
        instructions="You are a data analyst. Show your analytical process.",
        input="Analyze the trend: Sales were $10k, $12k, $15k, $18k over 4 months. Predict next month's sales.",
    )

    # Use legacy parameters (for backward compatibility)
    params = GenParams(
        max_tokens=1500,
        temperature=1.0,
        extra={
            "include_thoughts": True,  # Legacy: show thoughts
            "thinking_budget": 4000,  # Legacy: thinking token budget
        },
    )

    try:
        response = adapter.generate(config, prompt, params)
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        print("\nResponse:")
        print(response.text[:1000] + "..." if len(response.text) > 1000 else response.text)
    except Exception as e:
        print(f"Error: {e}")

    # Example 5: Excluding reasoning from response
    print("\n\n🤖 Example 5: Hidden Reasoning (Exclude from Response)")
    print("-" * 50)

    config = ProviderConfig(model="openai/o1-mini", api_key=api_key)

    prompt = Prompt(
        instructions="You are a concise assistant.", input="What are the top 3 benefits of renewable energy?"
    )

    # Use reasoning but exclude from response
    params = GenParams(
        max_tokens=800,
        temperature=1.0,
        extra={
            "reasoning_effort": "medium",  # Use reasoning internally
            "reasoning_exclude": True,  # But don't show it in response
        },
    )

    try:
        response = adapter.generate(config, prompt, params)
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        print("\nResponse (reasoning hidden):")
        print(response.text)
    except Exception as e:
        print(f"Error: {e}")

    # Example 6: Dynamic model detection demo
    print("\n\n🔍 Example 6: Dynamic Model Detection")
    print("-" * 50)

    test_models = [
        "openai/o1-preview",
        "openai/o3-mini",
        "openai/gpt-5-turbo",
        "anthropic/claude-3-5-sonnet",
        "google/gemini-2.0-flash-thinking-exp",
        "xai/grok-beta",
        "meta-llama/llama-3.3-70b-instruct",
    ]

    print("Model reasoning capability detection:")
    for model in test_models:
        reasoning_type = adapter._detect_model_reasoning_type(model)
        status = "✅" if reasoning_type else "❌"
        print(f"{status} {model:<35} → {reasoning_type or 'No reasoning support'}")

    print("\n🎯 Parameter Mapping Examples:")
    print("-" * 30)

    # Show how different parameters get mapped
    test_params = [
        GenParams(max_tokens=1000, extra={"reasoning_effort": "high"}),
        GenParams(max_tokens=1000, extra={"reasoning_max_tokens": 5000}),
        GenParams(max_tokens=1000, extra={"include_thoughts": True, "thinking_budget": 3000}),
        GenParams(max_tokens=1000, extra={"reasoning_enabled": True}),
    ]

    for i, params in enumerate(test_params, 1):
        config_result = adapter._build_reasoning_config(params, "openai/o1-preview")
        print(f"Config {i}: {params.extra} → {config_result}")


if __name__ == "__main__":
    main()
