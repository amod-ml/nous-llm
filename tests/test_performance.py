"""Performance and stress tests for nous-llm.

This module tests performance characteristics, resource usage,
and behavior under stress conditions.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nous_llm import (
    LLMClient,
    Prompt,
    ProviderConfig,
    agenenerate,
    generate,
)


class TestPerformance:
    """Performance tests for the nous-llm package."""

    def test_response_time_single_request(self) -> None:
        """Test response time for a single request."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        prompt = Prompt(
            instructions="You are a helpful assistant",
            input="What is 2+2?",
        )

        mock_response = {
            "output_text": "4",
            "usage": {"total_tokens": 10},
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.is_success = True
            mock_client.request.return_value = mock_response_obj

            start_time = time.perf_counter()
            response = generate(config, prompt)
            end_time = time.perf_counter()

            elapsed = end_time - start_time
            assert elapsed < 1.0  # Should complete in under 1 second
            assert response.text == "4"

    def test_batch_processing_performance(self) -> None:
        """Test performance when processing multiple requests in batch."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        # Create 100 prompts
        prompts = [
            Prompt(
                instructions="Calculate",
                input=f"What is {i} + {i}?",
            )
            for i in range(100)
        ]

        mock_response = {
            "output_text": "Answer",
            "usage": {"total_tokens": 10},
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.is_success = True
            mock_client.request.return_value = mock_response_obj

            start_time = time.perf_counter()

            responses = []
            for prompt in prompts:
                response = generate(config, prompt)
                responses.append(response)

            end_time = time.perf_counter()
            elapsed = end_time - start_time

            assert len(responses) == 100
            assert elapsed < 5.0  # Should process 100 requests in under 5 seconds

            # Calculate throughput
            throughput = len(responses) / elapsed
            assert throughput > 20  # Should process at least 20 requests per second

    @pytest.mark.asyncio
    async def test_async_concurrent_performance(self) -> None:
        """Test performance with concurrent async requests."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        # Create 50 prompts for concurrent processing
        prompts = [
            Prompt(
                instructions="Process",
                input=f"Task {i}",
            )
            for i in range(50)
        ]

        mock_response = {
            "output_text": "Processed",
            "usage": {"total_tokens": 10},
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.is_success = True
            mock_client.request = AsyncMock(return_value=mock_response_obj)

            start_time = time.perf_counter()

            # Process all prompts concurrently
            tasks = [agenenerate(config, prompt) for prompt in prompts]
            responses = await asyncio.gather(*tasks)

            end_time = time.perf_counter()
            elapsed = end_time - start_time

            assert len(responses) == 50
            assert elapsed < 2.0  # Concurrent processing should be fast

            # Concurrent throughput should be much higher
            throughput = len(responses) / elapsed
            assert throughput > 25  # At least 25 requests per second

    def test_thread_pool_performance(self) -> None:
        """Test performance with thread pool executor."""
        config = ProviderConfig(
            provider="anthropic",
            model="claude-opus-4.1",
            api_key="test-key",
        )

        prompts = [
            Prompt(
                instructions="Process",
                input=f"Thread task {i}",
            )
            for i in range(30)
        ]

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Processed")]
        mock_response.usage = MagicMock(input_tokens=5, output_tokens=5)

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_client.messages.create.return_value = mock_response

            start_time = time.perf_counter()

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(generate, config, prompt) for prompt in prompts]
                responses = [future.result() for future in futures]

            end_time = time.perf_counter()
            elapsed = end_time - start_time

            assert len(responses) == 30
            assert elapsed < 3.0  # Thread pool should handle this quickly

    def test_client_reuse_performance_benefit(self) -> None:
        """Test performance benefit of reusing client instances."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        prompts = [
            Prompt(
                instructions="Test",
                input=f"Input {i}",
            )
            for i in range(20)
        ]

        mock_response = {
            "output_text": "Response",
            "usage": {"total_tokens": 10},
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.is_success = True
            mock_client.request.return_value = mock_response_obj

            # Test with client reuse
            client = LLMClient(config)

            start_reuse = time.perf_counter()
            for prompt in prompts:
                client.generate(prompt)
            end_reuse = time.perf_counter()
            time_with_reuse = end_reuse - start_reuse

            # Test without client reuse (new client each time)
            start_no_reuse = time.perf_counter()
            for prompt in prompts:
                generate(config, prompt)
            end_no_reuse = time.perf_counter()
            time_without_reuse = end_no_reuse - start_no_reuse

            # Client reuse should be faster (or at least not slower)
            assert time_with_reuse <= time_without_reuse * 1.1  # Allow 10% margin

    def test_memory_efficiency_with_streaming(self) -> None:
        """Test memory efficiency when handling streaming responses."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        prompt = Prompt(
            instructions="Generate a long response",
            input="Write a detailed analysis",
        )

        # Simulate streaming chunks
        chunks = [f"Chunk {i} " for i in range(1000)]
        full_text = "".join(chunks)

        mock_response = {
            "output_text": full_text,
            "usage": {"total_tokens": 5000},
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.is_success = True
            mock_client.request.return_value = mock_response_obj

            response = generate(config, prompt)
            assert len(response.text) == len(full_text)

    def test_rate_limiting_performance(self) -> None:
        """Test performance under rate limiting conditions."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        prompts = [
            Prompt(
                instructions="Test",
                input=f"Request {i}",
            )
            for i in range(20)
        ]

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            response = MagicMock()
            if call_count % 5 == 0:
                # Every 5th request hits rate limit
                response.is_success = False
                response.status_code = 429
                response.json.return_value = {"error": "Rate limit"}
            else:
                response.is_success = True
                response.json.return_value = {
                    "output_text": "Success",
                    "usage": {"total_tokens": 10},
                }
            return response

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.request.side_effect = side_effect

            start_time = time.perf_counter()

            success_count = 0
            for prompt in prompts:
                try:
                    generate(config, prompt)
                    success_count += 1
                except:
                    pass  # Rate limited

            end_time = time.perf_counter()
            elapsed = end_time - start_time

            assert success_count == 16  # 20 - 4 rate limited
            assert elapsed < 5.0  # Should handle rate limiting efficiently

    @pytest.mark.asyncio
    async def test_async_cancellation_performance(self) -> None:
        """Test performance when cancelling async operations."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        prompt = Prompt(
            instructions="Test",
            input="Test cancellation",
        )

        async def slow_response(*args, **kwargs):
            await asyncio.sleep(5)  # Simulate slow response
            return MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.request = AsyncMock(side_effect=slow_response)

            start_time = time.perf_counter()

            # Create task and cancel it quickly
            task = asyncio.create_task(agenenerate(config, prompt))
            await asyncio.sleep(0.1)  # Let it start
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            end_time = time.perf_counter()
            elapsed = end_time - start_time

            assert elapsed < 0.5  # Cancellation should be quick

    def test_cache_performance(self) -> None:
        """Test performance with caching of common requests."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        # Same prompt repeated multiple times
        prompt = Prompt(
            instructions="Calculate",
            input="What is 2+2?",
        )

        mock_response = {
            "output_text": "4",
            "usage": {"total_tokens": 10},
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.is_success = True
            mock_client.request.return_value = mock_response_obj

            # First request (cache miss)
            start_first = time.perf_counter()
            response1 = generate(config, prompt)
            end_first = time.perf_counter()
            first_time = end_first - start_first

            # Subsequent requests (would benefit from cache if implemented)
            start_subsequent = time.perf_counter()
            for _ in range(10):
                response = generate(config, prompt)
                assert response.text == "4"
            end_subsequent = time.perf_counter()
            subsequent_time = (end_subsequent - start_subsequent) / 10

            # Each subsequent request should be at least as fast
            assert subsequent_time <= first_time * 1.5  # Allow some variance

    def test_large_batch_stress_test(self) -> None:
        """Stress test with a large batch of requests."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
        )

        # Create 1000 prompts
        prompts = [
            Prompt(
                instructions="Process",
                input=f"Large batch item {i}",
            )
            for i in range(1000)
        ]

        mock_response = {
            "output_text": "Processed",
            "usage": {"total_tokens": 10},
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.is_success = True
            mock_client.request.return_value = mock_response_obj

            client = LLMClient(config)

            start_time = time.perf_counter()

            responses = []
            for i, prompt in enumerate(prompts):
                response = client.generate(prompt)
                responses.append(response)

                # Print progress every 100 requests
                if (i + 1) % 100 == 0:
                    elapsed = time.perf_counter() - start_time
                    rate = (i + 1) / elapsed
                    print(f"Processed {i + 1}/1000 - Rate: {rate:.1f} req/s")

            end_time = time.perf_counter()
            total_elapsed = end_time - start_time

            assert len(responses) == 1000

            # Calculate overall performance metrics
            throughput = len(responses) / total_elapsed
            avg_latency = total_elapsed / len(responses)

            print(f"Total time: {total_elapsed:.2f}s")
            print(f"Throughput: {throughput:.1f} req/s")
            print(f"Avg latency: {avg_latency * 1000:.1f}ms")

            # Performance assertions
            assert throughput > 100  # Should handle at least 100 req/s
            assert avg_latency < 0.01  # Average latency under 10ms
