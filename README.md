# Nous LLM - Intelligent No Frills LLM Router

A unified Python interface for multiple Large Language Model providers including OpenAI, Anthropic Claude, Google Gemini, xAI Grok, and OpenRouter.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## 🔒 Security & Authentication

### GPG Signing Required

**ALL commits to this repository MUST be GPG-signed.** This is automatically enforced by a pre-commit hook that cannot be bypassed.

#### 🛡️ What This Means:
- **Authentication**: Every commit is cryptographically verified
- **Integrity**: Commits cannot be tampered with after signing
- **Non-repudiation**: Contributors cannot deny authorship of signed commits
- **Supply Chain Security**: Protection against commit spoofing attacks

#### 🚀 Quick Setup for Contributors:

**New to the project? Run this first:**
```bash
# Automated setup - installs hook and guides through GPG configuration
./scripts/setup-gpg-hook.sh
```

**Already have GPG configured? Just ensure it's enabled:**
```bash
# Enable GPG signing for this repository
git config commit.gpgsign true
git config user.signingkey YOUR_KEY_ID
```

#### ⚠️ Important Notes:
- **Unsigned commits will be automatically rejected**
- **The pre-commit hook validates your GPG setup before every commit**
- **You must add your GPG public key to your GitHub account**
- **The hook cannot be bypassed with `--no-verify`**

#### 📚 Need Help?
- **Full Setup Guide**: [GPG Signing Documentation](docs/GPG-SIGNING.md)
- **Troubleshooting**: Run `./scripts/setup-gpg-hook.sh` for diagnostics
- **Quick Test**: Try making a commit - the hook will guide you if anything's wrong

---

## Features

- **Unified Interface**: Single API for multiple LLM providers
- **Async Support**: Both synchronous and asynchronous interfaces
- **Type Safety**: Full typing with Pydantic v2 validation
- **Provider Flexibility**: Easy switching between providers and models
- **Serverless Ready**: Optimized for AWS Lambda and Google Cloud Run
- **Error Handling**: Comprehensive error taxonomy with provider context
- **Extensible**: Plugin architecture for custom providers

## Supported Providers

| Provider | Models | Status |
|----------|--------|--------|
| **OpenAI** | GPT-5, GPT-4o, GPT-4, GPT-3.5-turbo, o1, o2 | ✅ |
| **Anthropic** | Claude Opus 4.1, Claude 3.5 Sonnet, Claude 3 Haiku | ✅ |
| **Google Gemini** | Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.0 Flash Lite | ✅ |
| **xAI** | Grok 4, Grok 4 Heavy, Grok Beta | ✅ |
| **OpenRouter** | Llama 4 Maverick, Llama 3.3 70B, 100+ models via proxy | ✅ |

## Installation

Install the base package:

```bash
pip install nous-llm
```

Install with specific provider support:

```bash
# Install with OpenAI support
pip install nous-llm[openai]

# Install with all providers
pip install nous-llm[all]

# Install for development
pip install nous-llm[dev]
```

Or using uv (recommended):

```bash
# Base installation
uv add nous-llm

# With all providers
uv add nous-llm[all]
```

## Quick Start

### Basic Usage

```python
from nous_llm import generate, ProviderConfig, Prompt, GenParams

# Configure your provider
config = ProviderConfig(
    provider="openai",
    model="gpt-4o",
    api_key="your-api-key"  # or set OPENAI_API_KEY env var
)

# Create a prompt
prompt = Prompt(
    instructions="You are a helpful assistant.",
    input="What is the capital of France?"
)

# Generate response
response = generate(config, prompt)
print(response.text)  # "Paris is the capital of France."
```

### Async Usage

```python
import asyncio
from nous_llm import agenenerate, ProviderConfig, Prompt

async def main():
    config = ProviderConfig(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        api_key="your-api-key"
    )
    
    prompt = Prompt(
        instructions="You are a creative writing assistant.",
        input="Write a haiku about coding."
    )
    
    response = await agenenerate(config, prompt)
    print(response.text)

asyncio.run(main())
```

### Client-Based Approach

```python
from nous_llm import LLMClient, ProviderConfig, Prompt, GenParams

# Create a client for reuse
client = LLMClient(ProviderConfig(
    provider="gemini",
    model="gemini-1.5-pro",
    api_key="your-api-key"
))

# Generate multiple responses
prompts = [
    Prompt(instructions="You are helpful.", input="What is AI?"),
    Prompt(instructions="You are creative.", input="Write a poem."),
]

for prompt in prompts:
    response = client.generate(prompt)
    print(f"{response.provider}: {response.text}")
```

## Environment Variables

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AIza..."
export XAI_API_KEY="xai-..."
export OPENROUTER_API_KEY="sk-or-..."
```

Or create a `.env` file:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
XAI_API_KEY=xai-...
OPENROUTER_API_KEY=sk-or-...
```

## Advanced Features

### Provider-Specific Parameters

```python
from nous_llm import generate, ProviderConfig, Prompt, GenParams

# OpenAI with reasoning
config = ProviderConfig(provider="openai", model="o1-preview")
params = GenParams(
    max_tokens=1000,
    temperature=0.7,
    extra={"reasoning": True}  # OpenAI-specific
)

# Anthropic with thinking tokens
config = ProviderConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
params = GenParams(
    extra={"thinking": True}  # Anthropic-specific
)
```

### Custom Base URLs

```python
# Use OpenRouter for OpenAI models
config = ProviderConfig(
    provider="openrouter",
    model="openai/gpt-4o",
    base_url="https://openrouter.ai/api/v1",
    api_key="your-openrouter-key"
)
```

### Error Handling

```python
from nous_llm import generate, AuthError, RateLimitError, ProviderError

try:
    response = generate(config, prompt)
except AuthError as e:
    print(f"Authentication failed: {e}")
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except ProviderError as e:
    print(f"Provider error: {e}")
```

## Framework Integration

### FastAPI Service

```python
from fastapi import FastAPI
from nous_llm import agenenerate, ProviderConfig, Prompt

app = FastAPI()

@app.post("/generate")
async def generate_text(request: dict):
    config = ProviderConfig(**request["config"])
    prompt = Prompt(**request["prompt"])
    
    response = await agenenerate(config, prompt)
    return {"text": response.text, "usage": response.usage}
```

### AWS Lambda

```python
import json
from nous_llm import LLMClient, ProviderConfig, Prompt

# Global client for connection reuse
client = LLMClient(ProviderConfig(
    provider="openai",
    model="gpt-4o-mini"
))

def lambda_handler(event, context):
    prompt = Prompt(
        instructions=event["instructions"],
        input=event["input"]
    )
    
    response = client.generate(prompt)
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "text": response.text,
            "usage": response.usage.model_dump() if response.usage else None
        })
    }
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/amod-ml/nous-llm.git
cd nous-llm

# Install with development dependencies
uv sync --group dev

# Install pre-commit hooks (includes GPG validation)
./scripts/setup-gpg-hook.sh
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=nous_llm

# Run specific test file
uv run pytest tests/test_models.py
```

### Code Quality

```bash
# Format code
uv run ruff format

# Lint code
uv run ruff check

# Type checking
uv run mypy src/nous_llm
```

### Adding a New Provider

1. Create adapter in `src/unillm/adapters/`
2. Implement the `AdapterProtocol`
3. Register in `src/unillm/core/adapters.py`
4. Add model patterns to `src/unillm/core/registry.py`
5. Add tests in `tests/`

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.py` - Core functionality demos
- `fastapi_service.py` - REST API service
- `lambda_example.py` - AWS Lambda function

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Requirements

- Python 3.12+
- All commits must be GPG-signed
- Code must pass all tests and linting
- Follow the established patterns and conventions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 [Documentation](https://github.com/amod-ml/unillm#readme)
- 🐛 [Issue Tracker](https://github.com/amod-ml/unillm/issues)
- 💬 [Discussions](https://github.com/amod-ml/unillm/discussions)

---

*This security measure ensures the authenticity and integrity of all code contributions.*