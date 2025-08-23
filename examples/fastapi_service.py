#!/usr/bin/env python3
"""FastAPI service example for the Universal LLM Wrapper.

This example shows how to build a REST API service using FastAPI
and the unillm library for multi-provider LLM inference.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from nous_llm import GenParams, LLMClient, Prompt, ProviderConfig, agenenerate
from nous_llm.core.exceptions import AuthError, ProviderError, RateLimitError, ValidationError


# Request/Response models
class GenerateRequest(BaseModel):
    """Request model for text generation."""

    instructions: str = Field(
        description="System instructions for the LLM",
        example="You are a helpful assistant.",
    )
    input: str = Field(
        description="User input to process",
        example="What is the capital of France?",
    )
    provider: str = Field(
        description="LLM provider to use",
        example="openai",
    )
    model: str = Field(
        description="Model to use for generation",
        example="gpt-4o",
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        le=4000,
        description="Maximum tokens to generate",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter",
    )
    stop: list[str] | None = Field(
        default=None,
        description="Stop sequences",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific parameters",
    )


class GenerateResponse(BaseModel):
    """Response model for text generation."""

    text: str = Field(description="Generated text")
    provider: str = Field(description="Provider used")
    model: str = Field(description="Model used")
    usage: dict[str, int | None] | None = Field(
        default=None,
        description="Token usage information",
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(description="Error message")
    error_type: str = Field(description="Error type")
    provider: str | None = Field(default=None, description="Provider where error occurred")


# Global client cache for connection reuse
_client_cache: dict[str, LLMClient] = {}


def get_client(provider: str, model: str, api_key: str) -> LLMClient:
    """Get or create a cached client for the provider/model combination."""
    cache_key = f"{provider}:{model}:{hash(api_key)}"

    if cache_key not in _client_cache:
        config = ProviderConfig(
            provider=provider,
            model=model,
            api_key=api_key,
        )
        _client_cache[cache_key] = LLMClient(config)

    return _client_cache[cache_key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    print("Starting Universal LLM API service...")
    yield
    # Shutdown
    print("Shutting down Universal LLM API service...")
    _client_cache.clear()


# Create FastAPI app
app = FastAPI(
    title="Universal LLM API",
    description="Multi-provider LLM inference API using unillm",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Universal LLM API",
        "version": "0.1.0",
        "description": "Multi-provider LLM inference API",
        "supported_providers": ["openai", "anthropic", "gemini", "xai", "openrouter"],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
)
async def generate_text(request: GenerateRequest):
    """Generate text using the specified LLM provider."""
    try:
        # Get API key from environment
        api_key_env = f"{request.provider.upper()}_API_KEY"
        api_key = os.getenv(api_key_env)

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"API key not found for provider {request.provider}. Set {api_key_env} environment variable.",
            )

        # Create configuration
        config = ProviderConfig(
            provider=request.provider,
            model=request.model,
            api_key=api_key,
        )

        # Create prompt
        prompt = Prompt(
            instructions=request.instructions,
            input=request.input,
        )

        # Create parameters
        params = GenParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            extra=request.extra,
        )

        # Generate response
        response = await agenenerate(config, prompt, params)

        # Format response
        result = GenerateResponse(
            text=response.text,
            provider=response.provider,
            model=response.model,
        )

        # Include usage if available
        if response.usage:
            result.usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return result

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {e}",
        )
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {e}",
        )
    except RateLimitError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {e}",
        )
    except ProviderError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Provider error: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}",
        )


@app.post("/generate/client")
async def generate_with_client(request: GenerateRequest):
    """Generate text using cached client for better performance."""
    try:
        # Get API key from environment
        api_key_env = f"{request.provider.upper()}_API_KEY"
        api_key = os.getenv(api_key_env)

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"API key not found for provider {request.provider}",
            )

        # Get cached client
        client = get_client(request.provider, request.model, api_key)

        # Create prompt and parameters
        prompt = Prompt(
            instructions=request.instructions,
            input=request.input,
        )

        params = GenParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            extra=request.extra,
        )

        # Generate response using cached client
        response = await client.agenenerate(prompt, params)

        return GenerateResponse(
            text=response.text,
            provider=response.provider,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens if response.usage else None,
                "output_tokens": response.usage.output_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            }
            if response.usage
            else None,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# Run with: uvicorn examples.fastapi_service:app --reload
if __name__ == "__main__":
    import uvicorn

    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    uvicorn.run(
        "examples.fastapi_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
