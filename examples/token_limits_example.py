#!/usr/bin/env python3
"""Example demonstrating dynamic token limits across different providers."""

import os

from src.nous_llm.adapters.gemini_adapter import GeminiAdapter
from src.nous_llm.adapters.openai_adapter import OpenAIAdapter
from src.nous_llm.adapters.openrouter_adapter import OpenRouterAdapter
from src.nous_llm.core.models import GenParams, ProviderConfig
from src.nous_llm.core.token_validator import TokenLimitValidator


def demonstrate_token_limits():
    """Demonstrate dynamic token limit validation across providers."""
    print("🔢 Dynamic Token Limits Example")
    print("=" * 50)

    # Test different models and their limits
    test_cases = [
        # OpenAI models
        ("openai", "gpt-4o", 16384),
        ("openai", "o1-mini", 65536),
        ("openai", "gpt-oss-120b", 131072),
        ("openai", "gpt-4o-realtime", 4096),
        # Gemini models
        ("gemini", "gemini-2.5-flash", 65536),
        ("gemini", "gemini-2.0-flash", 2048),
        ("gemini", "gemini-1.5-flash", 8192),
        # xAI models
        ("xai", "grok-4", 32768),
        ("xai", "grok-code-fast-1", 32768),
        # OpenRouter (depends on underlying model)
        ("openrouter", "openai/gpt-4o", 16384),
        ("openrouter", "anthropic/claude-3-5-sonnet", 16384),
    ]

    print("\n📊 Model Token Limits:")
    print("-" * 60)
    print(f"{'Provider':<12} {'Model':<25} {'Max Tokens':<12} {'Status'}")
    print("-" * 60)

    for provider, model, expected_limit in test_cases:
        try:
            actual_limit = TokenLimitValidator.get_max_tokens_limit(provider, model)
            status = "✅ Known" if actual_limit == expected_limit else f"📝 {actual_limit}"
            print(f"{provider:<12} {model:<25} {expected_limit:<12} {status}")
        except Exception as e:
            print(f"{provider:<12} {model:<25} {expected_limit:<12} ❌ Error: {e}")

    print("\n🧪 Testing Token Validation:")
    print("-" * 40)

    # Test validation with different scenarios
    validation_tests = [
        # (provider, model, max_tokens, should_pass)
        ("openai", "gpt-4o", 16384, True),  # At limit
        ("openai", "gpt-4o", 16385, False),  # Over limit
        ("openai", "o1-mini", 65536, True),  # At higher limit
        ("gemini", "gemini-2.0-flash", 2048, True),  # At limit
        ("gemini", "gemini-2.0-flash", 3000, False),  # Over limit
        ("gemini", "gemini-2.5-flash", 65536, True),  # At much higher limit
    ]

    for provider, model, max_tokens, should_pass in validation_tests:
        try:
            result = TokenLimitValidator.validate_max_tokens(max_tokens, provider, model)
            status = "✅ Pass" if should_pass else "❌ Unexpected pass"
            print(f"{provider}/{model}: {max_tokens} tokens → {status}")
        except ValueError as e:
            status = "✅ Correctly rejected" if not should_pass else f"❌ Unexpected error: {e}"
            print(f"{provider}/{model}: {max_tokens} tokens → {status}")

    print("\n🔄 Testing with Real Adapters:")
    print("-" * 35)

    # Test with actual adapter integration
    test_adapters = [
        ("OpenAI", OpenAIAdapter(), "openai", "gpt-4o-mini", os.getenv("OPENAI_API_KEY")),
        ("Gemini", GeminiAdapter(), "gemini", "gemini-2.0-flash", os.getenv("GEMINI_API_KEY")),
        ("OpenRouter", OpenRouterAdapter(), "openrouter", "openai/gpt-4o", os.getenv("OPENROUTER_API_KEY")),
    ]

    for name, adapter, provider, model, api_key in test_adapters:
        if not api_key:
            print(f"{name}: ⚠️  No API key - skipping")
            continue

        try:
            config = ProviderConfig(provider=provider, model=model, api_key=api_key)

            # Test with valid token count
            valid_params = GenParams(max_tokens=1000)
            adapter.generate(config, Prompt(instructions="test", input="test"), valid_params)
            print(f"{name}: ✅ Valid token count (1000) accepted")

            # Test with excessive token count
            try:
                invalid_params = GenParams(max_tokens=999999)
                adapter.generate(config, Prompt(instructions="test", input="test"), invalid_params)
                print(f"{name}: ❌ Excessive tokens unexpectedly accepted")
            except ValueError:
                print(f"{name}: ✅ Excessive tokens correctly rejected")

        except Exception as e:
            print(f"{name}: ❌ Error: {e}")

    print("\n📈 Benefits of Dynamic Token Limits:")
    print("-" * 40)
    print("• ✅ No more artificial 32k limit")
    print("• ✅ Model-specific accurate limits")
    print("• ✅ Automatic validation before API calls")
    print("• ✅ Clear error messages when limits exceeded")
    print("• ✅ Support for high-capacity models (131k+ tokens)")
    print("• ✅ Cached limits for performance")

    print("\n🎯 Usage in Your Code:")
    print("-" * 25)
    print("""
# The library now automatically validates token limits
from nous_llm import generate, ProviderConfig, Prompt, GenParams

config = ProviderConfig(
    provider="openai",
    model="gpt-oss-120b",  # Supports 131k tokens!
    api_key="your-api-key"
)

params = GenParams(
    max_tokens=100000,  # No longer limited to 32k
    temperature=0.7
)

# This will work without issues
response = generate(config, prompt, params)
""")


if __name__ == "__main__":
    demonstrate_token_limits()
