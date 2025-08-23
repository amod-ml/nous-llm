# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-08-XX

### Added
- Initial release of Nous LLM - Intelligent No Frills LLM Router
- Support for multiple LLM providers with latest 2025 models:
  - OpenAI (GPT-5, GPT-4o, GPT-4, GPT-3.5-turbo, o1, o2 models)
  - Anthropic (Claude Opus 4.1, Claude 3.5 Sonnet, Claude 3 Haiku)
  - Google Gemini (Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.0 Flash Lite) using google-genai SDK
  - xAI (Grok 4, Grok 4 Heavy, Grok Beta)
  - OpenRouter (Llama 4 Maverick, Llama 3.3 70B, 100+ models via proxy)
- Unified interface with `generate()` and `agenenerate()` functions
- Type-safe models with Pydantic v2 validation
- Comprehensive error handling and provider-specific error mapping
- Client class for configuration reuse
- Model validation and registry system
- Async support with proper error handling
- Extensible adapter architecture
- Complete test suite with >95% coverage
- Examples for AWS Lambda, FastAPI, and basic usage
- Full documentation with setup guides

### Features
- **RORO Pattern**: Receive Object, Return Object for consistent interfaces
- **Provider Flexibility**: Easy switching between providers and models
- **Type Safety**: Full typing with strict validation
- **Serverless Ready**: Optimized for AWS Lambda and Google Cloud Run
- **Error Taxonomy**: Comprehensive error classification with provider context
- **Extensible**: Plugin architecture for custom providers
- **Async First**: Both sync and async interfaces with identical signatures

### Technical
- Python 3.12+ support
- GPG-signed commits required
- Comprehensive linting and formatting with Ruff
- Type checking with mypy
- Modern packaging with uv and hatchling
- Optional dependencies for specific providers
