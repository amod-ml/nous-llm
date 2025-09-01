# Nous LLM

> **Intelligent No Frills LLM Router** - A unified Python interface for multiple Large Language Model providers

[![PyPI version](https://badge.fury.io/py/nous-llm.svg)](https://badge.fury.io/py/nous-llm)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Issues](https://img.shields.io/github/issues/amod-ml/nous-llm)](https://github.com/amod-ml/nous-llm/issues)

## Why Nous LLM?

Switch between LLM providers with a single line of code. Build AI applications without vendor lock-in.

```python
# Same interface, different providers
config = ProviderConfig(provider="openai", model="gpt-4o")     # OpenAI
config = ProviderConfig(provider="anthropic", model="claude-3-5-sonnet")  # Anthropic
config = ProviderConfig(provider="gemini", model="gemini-2.5-pro")  # Google
```

## ✨ Key Features

- **🔄 Unified Interface**: Single API for multiple LLM providers
- **⚡ Async Support**: Both synchronous and asynchronous interfaces  
- **🛡️ Type Safety**: Full typing with Pydantic v2 validation
- **🔀 Provider Flexibility**: Easy switching between providers and models
- **☁️ Serverless Ready**: Optimized for AWS Lambda and Google Cloud Run
- **🚨 Error Handling**: Comprehensive error taxonomy with provider context
- **🔌 Extensible**: Plugin architecture for custom providers

## 🚀 Quick Start

### Install

```bash
pip install nous-llm
```

### Use in 3 Lines

```python
from nous_llm import generate, ProviderConfig, Prompt

config = ProviderConfig(provider="openai", model="gpt-4o")
response = generate(config, Prompt(input="What is the capital of France?"))
print(response.text)  # "Paris is the capital of France."
```

## 📦 Supported Providers

| Provider | Popular Models | Latest Models |
|----------|---------------|---------------|
| **OpenAI** | GPT-4o, GPT-4-turbo, GPT-3.5-turbo | GPT-5, o3, o4-mini |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Haiku | Claude Opus 4.1 |
| **Google** | Gemini 1.5 Pro, Gemini 1.5 Flash | Gemini 2.5 Pro |
| **xAI** | Grok Beta | Grok 4, Grok 4 Heavy |
| **OpenRouter** | Llama 3.3 70B, Mixtral | Llama 4 Maverick |

## Installation

### Quick Install

```bash
# Using pip
pip install nous-llm

# Using uv (recommended)
uv add nous-llm
```

### Installation Options

```bash
# Install with specific provider support
pip install nous-llm[openai]      # OpenAI only
pip install nous-llm[anthropic]   # Anthropic only
pip install nous-llm[all]         # All providers

# Development installation
pip install nous-llm[dev]         # Includes testing tools
```

### Environment Setup

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

## Usage Examples

### 1. Basic Synchronous Usage

```python
from nous_llm import generate, ProviderConfig, Prompt

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

### 2. Asynchronous Usage

```python
import asyncio
from nous_llm import agenenerate, ProviderConfig, Prompt

async def main():
    config = ProviderConfig(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022"
    )
    
    prompt = Prompt(
        instructions="You are a creative writing assistant.",
        input="Write a haiku about coding."
    )
    
    response = await agenenerate(config, prompt)
    print(response.text)

asyncio.run(main())
```

### 3. Client-Based Approach (Recommended for Multiple Calls)

```python
from nous_llm import LLMClient, ProviderConfig, Prompt

# Create a reusable client
client = LLMClient(ProviderConfig(
    provider="gemini",
    model="gemini-1.5-pro"
))

# Generate multiple responses efficiently
prompts = [
    Prompt(instructions="You are helpful.", input="What is AI?"),
    Prompt(instructions="You are creative.", input="Write a poem."),
]

for prompt in prompts:
    response = client.generate(prompt)
    print(f"{response.provider}: {response.text}")
```

## Advanced Features

### 4. Provider-Specific Parameters

```python
from nous_llm import generate, ProviderConfig, Prompt, GenParams

# OpenAI GPT-5 with reasoning mode
config = ProviderConfig(provider="openai", model="gpt-5")
params = GenParams(
    max_tokens=1000,
    temperature=0.7,
    extra={"reasoning": True}  # OpenAI-specific
)

# OpenAI O-series reasoning model
config = ProviderConfig(provider="openai", model="o3-mini")
params = GenParams(
    max_tokens=1000,
    temperature=0.7,  # Will be automatically set to 1.0 with a warning
)

# Anthropic with thinking tokens
config = ProviderConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
params = GenParams(
    extra={"thinking": True}  # Anthropic-specific
)

response = generate(config, prompt, params)
```

### 5. Gemini Thinking Functionality

```python
from nous_llm import generate, ProviderConfig, Prompt, GenParams

# Enable thinking mode for enhanced reasoning
config = ProviderConfig(
    provider="gemini", 
    model="gemini-2.5-pro"  # Use thinking-enabled model
)

prompt = Prompt(
    instructions="You are a math tutor. Show your step-by-step reasoning.",
    input="Calculate the area of a circle with radius 7 cm, then find what percentage this is of a square with side length 15 cm."
)

# Configure thinking parameters
params = GenParams(
    max_tokens=1500,
    temperature=0.3,
    extra={
        "include_thoughts": True,      # Show the model's reasoning process
        "thinking_budget": 8000        # Allow up to 8000 tokens for thinking
    }
)

response = generate(config, prompt, params)
print(response.text)

# Output format:
# **Thinking:**
# Let me break this down step by step...
# First, I need to calculate the area of the circle...
# 
# **Response:**
# The area of the circle is approximately 153.94 cm²...
```

**Thinking Parameters:**
- `include_thoughts`: Boolean to enable/disable thinking output
- `thinking_budget`: Integer token budget for the thinking process
- Works with thinking-enabled models like `gemini-2.5-pro`

> **Note for Developers**: 
> 
> **Parameter Changes in OpenAI's Latest Models:**
> - **Token Limits**: GPT-5 series and O-series models (o1, o3, o4-mini) use `max_completion_tokens` instead of `max_tokens`. The library automatically handles this with intelligent parameter mapping and fallback mechanisms.
> - **Temperature**: O-series reasoning models (o1, o3, o4-mini) and GPT-5 thinking/reasoning variants require `temperature=1.0`. The library automatically adjusts this and warns you if a different value is requested.
> 
> You can continue using the standard parameters in `GenParams` - they will be automatically converted to the correct parameter for each model.

### 5. Custom Base URLs & Proxies

```python
# Use OpenRouter as a proxy for OpenAI models
config = ProviderConfig(
    provider="openrouter",
    model="openai/gpt-4o",
    base_url="https://openrouter.ai/api/v1",
    api_key="your-openrouter-key"
)
```

### 6. Error Handling

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

### 6. OpenRouter Thinking Functionality

OpenRouter supports thinking/reasoning functionality across multiple model families with different parameter configurations:

```python
from nous_llm import generate, ProviderConfig, Prompt, GenParams

# OpenAI o-series models (effort-based reasoning)
config = ProviderConfig(
    provider="openrouter",
    model="openai/o1-preview",
    api_key="your-openrouter-key"
)

prompt = Prompt(
    instructions="You are a math tutor. Show your reasoning clearly.",
    input="Calculate compound interest on $1000 at 5% for 3 years."
)

# Effort-based reasoning (OpenAI o1/o3/GPT-5 models)
params = GenParams(
    max_tokens=2000,
    temperature=1.0,  # Required for o-series models
    extra={
        "reasoning_effort": "high",      # "low", "medium", "high"
        "reasoning_exclude": False       # Include reasoning in response
    }
)

response = generate(config, prompt, params)
print(response.text)
```

**Different Model Types:**

```python
# Anthropic Claude (max_tokens-based reasoning)
config = ProviderConfig(
    provider="openrouter",
    model="anthropic/claude-3-5-sonnet",
    api_key="your-openrouter-key"
)

params = GenParams(
    max_tokens=1500,
    extra={
        "reasoning_max_tokens": 6000,    # Token budget for reasoning
        "reasoning_exclude": False       # Show reasoning process
    }
)

# xAI Grok (effort-based reasoning)
config = ProviderConfig(
    provider="openrouter", 
    model="xai/grok-beta",
    api_key="your-openrouter-key"
)

params = GenParams(
    max_tokens=2000,
    extra={
        "reasoning_effort": "medium",    # Reasoning effort level
        "reasoning_exclude": True        # Hide reasoning, show only final answer
    }
)

# Legacy parameter support (backward compatibility)
params = GenParams(
    max_tokens=1500,
    extra={
        "include_thoughts": True,        # Enable thinking
        "thinking_budget": 4000          # Token budget (maps to appropriate param)
    }
)
```

**Supported Models:**
- **OpenAI**: o1-preview, o1-mini, o3-mini, gpt-5-turbo (effort-based)
- **Anthropic**: claude-3-5-sonnet, claude-3-5-haiku (max_tokens-based)  
- **xAI**: grok-beta, grok-2 (effort-based)
- **Google**: gemini-2.0-flash-thinking-exp (max_tokens-based)

The adapter automatically detects model capabilities and applies the correct reasoning parameters.

### Dynamic Token Limits

The library now supports dynamic token limits based on actual provider and model capabilities, replacing the previous static 32k limit:

```python
from nous_llm import generate, ProviderConfig, Prompt, GenParams

# High-capacity models now supported
config = ProviderConfig(
    provider="openai",
    model="gpt-oss-120b",  # Supports 131,072 tokens
    api_key="your-api-key"
)

params = GenParams(
    max_tokens=100000,  # No longer limited to 32k
    temperature=0.7
)

response = generate(config, prompt, params)
```

**Model-Specific Limits:**
- **OpenAI**: 4,096 (GPT-4o Realtime) to 131,072 (GPT-OSS series)
- **Gemini**: 2,048 (Gemini 2.0 Flash) to 65,536 (Gemini 2.5 series)
- **xAI**: 32,768 tokens (Grok series)
- **Anthropic**: 16,384 tokens (Claude series)
- **OpenRouter**: Varies by underlying model

The library automatically validates token limits and provides clear error messages:

```python
# This will raise ValueError with helpful message
params = GenParams(max_tokens=200000)  # Exceeds model limit
response = generate(config, prompt, params)
# ValueError: max_tokens (200000) exceeds model limit (131072) for openai/gpt-oss-120b
```

**Benefits:**
- ✅ No artificial 32k limit restriction
- ✅ Model-specific accurate validation
- ✅ Support for high-capacity models
- ✅ Automatic limit detection and caching
- ✅ Clear error messages when limits exceeded

## Production Integration

### FastAPI Web Service

```python
from fastapi import FastAPI, HTTPException
from nous_llm import agenenerate, ProviderConfig, Prompt, AuthError

app = FastAPI(title="Nous LLM API")

@app.post("/generate")
async def generate_text(request: dict):
    try:
        config = ProviderConfig(**request["config"])
        prompt = Prompt(**request["prompt"])
        
        response = await agenenerate(config, prompt)
        return {
            "text": response.text, 
            "usage": response.usage,
            "provider": response.provider
        }
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
```

### AWS Lambda Function

```python
import json
from nous_llm import LLMClient, ProviderConfig, Prompt

# Global client for connection reuse across invocations
client = LLMClient(ProviderConfig(
    provider="openai",
    model="gpt-4o-mini"
))

def lambda_handler(event, context):
    try:
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
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
```

---

## Development

### Project Setup

```bash
# Clone the repository
git clone https://github.com/amod-ml/nous-llm.git
cd nous-llm

# Install with development dependencies
uv sync --group dev

# Install pre-commit hooks (includes GPG validation)
./scripts/setup-gpg-hook.sh
```

### Testing & Quality

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=nous_llm

# Format and lint code
uv run ruff format
uv run ruff check

# Type checking
uv run mypy src/nous_llm
```

### Adding a New Provider

1. Create adapter in `src/nous_llm/adapters/`
2. Implement the `AdapterProtocol` 
3. Register in `src/nous_llm/core/adapters.py`
4. Add model patterns to `src/nous_llm/core/registry.py`
5. Add comprehensive tests in `tests/`

## Examples & Resources

### Complete Examples
- 📁 `examples/basic_usage.py` - Core functionality demos
- 📁 `examples/fastapi_service.py` - REST API service  
- 📁 `examples/lambda_example.py` - AWS Lambda function

### Documentation & Support
- 📖 [Full Documentation](https://github.com/amod-ml/nous-llm#readme)
- 🐛 [Issue Tracker](https://github.com/amod-ml/nous-llm/issues)
- 💬 [Discussions](https://github.com/amod-ml/nous-llm/discussions)

## 🐛 Found an Issue?

We'd love to hear from you! Please [report any issues](https://github.com/amod-ml/nous-llm/issues/new) you encounter. When reporting issues, please include:

- Python version
- Nous LLM version (`pip show nous-llm`)
- Minimal code to reproduce the issue
- Full error traceback

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### 🔒 Security Requirements for Contributors

**ALL commits to this repository MUST be GPG-signed.** This is automatically enforced by a pre-commit hook.

#### Why GPG Signing?
- **🔐 Authentication**: Every commit is cryptographically verified
- **🛡️ Integrity**: Commits cannot be tampered with after signing  
- **📝 Non-repudiation**: Contributors cannot deny authorship of signed commits
- **🔗 Supply Chain Security**: Protection against commit spoofing attacks

#### Quick Setup for Contributors

**New to the project?**
```bash
# Automated setup - installs hook and guides through GPG configuration
./scripts/setup-gpg-hook.sh
```

**Already have GPG configured?**
```bash
# Enable GPG signing for this repository
git config commit.gpgsign true
git config user.signingkey YOUR_KEY_ID
```

#### Important Notes
- ❌ Unsigned commits will be automatically rejected
- ✅ The pre-commit hook validates your GPG setup before every commit
- 📋 You must add your GPG public key to your GitHub account
- 🚫 The hook cannot be bypassed with `--no-verify`

#### Need Help?
- 📖 **Full Setup Guide**: [GPG Signing Documentation](docs/GPG-SIGNING.md)
- 🔧 **Troubleshooting**: Run `./scripts/setup-gpg-hook.sh` for diagnostics
- 🧪 **Quick Test**: Try making a commit - the hook will guide you if anything's wrong

### Development Requirements
- ✅ Python 3.12+
- 🔐 All commits must be GPG-signed
- 🧪 Code must pass all tests and linting
- 📋 Follow established patterns and conventions

## 📄 License

This project is licensed under the **Mozilla Public License 2.0** - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Built with ❤️ for the AI community</strong><br>
  <em>🔒 GPG signing ensures the authenticity and integrity of all code contributions</em>
</p>
<!-- Cache bust to update PyPI badge -->
