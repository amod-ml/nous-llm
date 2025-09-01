#!/usr/bin/env python3
"""Example demonstrating Gemini thinking functionality with nous-llm."""

import os

from src.nous_llm.adapters.gemini_adapter import GeminiAdapter
from src.nous_llm.core.models import GenParams, Prompt, ProviderConfig


def main():
    """Demonstrate Gemini thinking functionality."""
    # Set up API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        return

    # Initialize Gemini adapter
    adapter = GeminiAdapter()

    # Configuration for Gemini thinking model
    config = ProviderConfig(
        model="gemini-2.0-flash-thinking-exp",  # Use thinking-enabled model
        api_key=api_key,
    )

    # Example 1: Math problem with thinking
    print("🧮 Example 1: Math Problem with Thinking")
    print("=" * 50)

    prompt = Prompt(
        instructions="You are a math tutor. Show your step-by-step reasoning.",
        input="Calculate the area of a circle with radius 7 cm, then find what percentage this is of a square with side length 15 cm.",
    )

    # Enable thinking with parameters
    params = GenParams(
        max_tokens=1500,
        temperature=0.3,
        extra={
            "include_thoughts": True,  # Show the model's reasoning process
            "thinking_budget": 8000,  # Allow up to 8000 tokens for thinking
        },
    )

    try:
        response = adapter.generate(config, prompt, params)
        print(response.text)
        print(f"\nUsage: {response.usage}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 70 + "\n")

    # Example 2: Logic puzzle with thinking
    print("🧩 Example 2: Logic Puzzle with Thinking")
    print("=" * 50)

    prompt = Prompt(
        instructions="You are a logic puzzle solver. Think through each step carefully.",
        input="""Three friends - Alice, Bob, and Carol - each have a different pet (cat, dog, fish) and live in different colored houses (red, blue, green).

Clues:
1. Alice doesn't live in the red house
2. The person with the cat lives in the blue house
3. Bob doesn't have a fish
4. Carol doesn't live in the green house
5. The person in the red house has a dog

Who has which pet and lives in which house?""",
    )

    params = GenParams(
        max_tokens=2000,
        temperature=0.1,  # Lower temperature for logical reasoning
        extra={
            "include_thoughts": True,
            "thinking_budget": 10000,  # More thinking budget for complex logic
        },
    )

    try:
        response = adapter.generate(config, prompt, params)
        print(response.text)
        print(f"\nUsage: {response.usage}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 70 + "\n")

    # Example 3: Comparison - with vs without thinking
    print("🔄 Example 3: Comparison - With vs Without Thinking")
    print("=" * 50)

    prompt = Prompt(
        instructions="Explain the concept of quantum entanglement in simple terms.",
        input="How does quantum entanglement work and why is it important?",
    )

    # With thinking
    print("🧠 WITH thinking:")
    print("-" * 30)
    params_with_thinking = GenParams(
        max_tokens=1000, temperature=0.5, extra={"include_thoughts": True, "thinking_budget": 5000}
    )

    try:
        response_with = adapter.generate(config, prompt, params_with_thinking)
        print(response_with.text[:500] + "..." if len(response_with.text) > 500 else response_with.text)
    except Exception as e:
        print(f"Error: {e}")

    print("\n🤖 WITHOUT thinking:")
    print("-" * 30)
    # Without thinking
    params_without_thinking = GenParams(
        max_tokens=1000,
        temperature=0.5,
        extra={},  # No thinking parameters
    )

    try:
        response_without = adapter.generate(config, prompt, params_without_thinking)
        print(response_without.text[:500] + "..." if len(response_without.text) > 500 else response_without.text)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
