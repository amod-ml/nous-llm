#!/usr/bin/env python3
"""Script to test real API calls with all providers."""

import os
import subprocess
import sys


def main():
    """Run real API tests."""
    print("🚀 Nous LLM - Real API Testing")
    print("=" * 50)

    # Check if .env or keys.txt exists
    env_files = []
    if os.path.exists(".env"):
        env_files.append(".env")
    if os.path.exists("keys.txt"):
        env_files.append("keys.txt")

    if not env_files:
        print("⚠️  No .env or keys.txt file found!")
        print("   Create one with your API keys:")
        print("   OPENAI_API_KEY=your_key_here")
        print("   ANTHROPIC_API_KEY=your_key_here")
        print("   GEMINI_API_KEY=your_key_here")
        print("   XAI_API_KEY=your_key_here")
        print("   OPENROUTER_API_KEY=your_key_here")
        return 1

    print(f"📁 Found environment files: {', '.join(env_files)}")

    # Check available API keys
    available_keys = []
    for key in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "XAI_API_KEY",
        "OPENROUTER_API_KEY",
    ]:
        if os.getenv(key):
            available_keys.append(key)

    if not available_keys:
        # Try loading from keys.txt
        if os.path.exists("keys.txt"):
            with open("keys.txt") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#") and not line.startswith("here"):
                        try:
                            key, value = line.split("=", 1)
                            if key in [
                                "OPENAI_API_KEY",
                                "ANTHROPIC_API_KEY",
                                "GEMINI_API_KEY",
                                "GOOGLE_API_KEY",
                                "XAI_API_KEY",
                                "OPENROUTER_API_KEY",
                            ]:
                                os.environ[key] = value
                                available_keys.append(key)
                        except ValueError:
                            continue

    if not available_keys:
        print("❌ No API keys found in environment!")
        return 1

    print(f"🔑 Found API keys: {', '.join(available_keys)}")
    print()

    # Run the tests
    cmd = [
        "uv",
        "run",
        "pytest",
        "tests/test_real_api_calls.py",
        "-m",
        "integration",
        "-v",
        "--tb=short",
        "-s",  # Don't capture output so we can see real-time results
    ]

    print(f"🧪 Running command: {' '.join(cmd)}")
    print("=" * 50)

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
