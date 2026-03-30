#!/usr/bin/env python3
"""Benchmark client for rvllm and Python vLLM comparison.

Non-streaming mode: each request returns a complete JSON response.
Wall-clock latency = actual generation time. No TCP buffering artifacts.

Measures:
- Throughput (tokens/sec) = total completion tokens / total wall time
- Request latency (end-to-end per request)
- TTFT approximated as full-request latency / completion_tokens (first token)
"""

import asyncio
import aiohttp
import json
import time
import argparse
import sys

PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to sort a list of integers.",
    "What are the main differences between TCP and UDP?",
    "Describe the process of photosynthesis step by step.",
    "Write a short story about a robot learning to paint.",
    "Explain how a transformer neural network works.",
    "What are the advantages of Rust over C++?",
    "Describe the water cycle in detail.",
    "Write a haiku about machine learning.",
    "Explain the concept of recursion with an example.",
    "What is the difference between a stack and a queue?",
    "Describe how HTTPS encryption works.",
    "Write a SQL query to find duplicate records in a table.",
    "Explain the CAP theorem in distributed systems.",
    "What are the main principles of object-oriented programming?",
    "Describe the architecture of a modern CPU.",
    "Write a regular expression to validate email addresses.",
    "Explain how garbage collection works in Java.",
    "What is the difference between concurrency and parallelism?",
    "Describe the MapReduce programming model.",
    "Explain how a B-tree index works in databases.",
    "What are the trade-offs between microservices and monoliths?",
    "Describe the process of DNS resolution.",
    "Write pseudocode for the A* pathfinding algorithm.",
]


async def send_request(session, url, prompt, max_tokens=128):
    """Send a non-streaming completion request and measure wall-clock latency."""
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.8,
        "stream": False,
    }

    start = time.perf_counter()
    try:
        async with session.post(f"{url}/v1/completions", json=payload) as resp:
            result = await resp.json()
    except Exception as e:
        return None

    end = time.perf_counter()
    latency_ms = (end - start) * 1000

    usage = result.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)

    if completion_tokens == 0:
        return None

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": latency_ms,
        "tokens_per_sec": completion_tokens / (latency_ms / 1000),
    }


async def run_benchmark(url, num_prompts, concurrency, max_tokens=128):
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_prompts)]
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    errors = 0

    async def limited_request(session, prompt):
        nonlocal errors
        async with semaphore:
            result = await send_request(session, url, prompt, max_tokens)
            if result is None:
                errors += 1
            else:
                results.append(result)

    start = time.perf_counter()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as session:
        tasks = [limited_request(session, p) for p in prompts]
        await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start

    if not results:
        print(f"ERROR: All {num_prompts} requests failed!")
        return None

    latencies = sorted([r["latency_ms"] for r in results])
    total_completion_tokens = sum(r["completion_tokens"] for r in results)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)

    def percentile(arr, p):
        idx = int(len(arr) * p / 100)
        return arr[min(idx, len(arr) - 1)]

    return {
        "server_url": url,
        "num_requests": num_prompts,
        "successful_requests": len(results),
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "total_time_sec": total_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "throughput_tok_per_sec": total_completion_tokens / total_time,
        "requests_per_sec": len(results) / total_time,
        "avg_latency_ms": sum(latencies) / len(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "min_latency_ms": latencies[0],
        "max_latency_ms": latencies[-1],
        "avg_tokens_per_request": total_completion_tokens / len(results),
        "avg_tps_per_request": sum(r["tokens_per_sec"] for r in results) / len(results),
        "num_errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark client for rvllm and Python vLLM (non-streaming)"
    )
    parser.add_argument("--url", required=True, help="Server URL")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--concurrent", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    print(f"Benchmarking {args.url} (non-streaming)")
    print(f"  Prompts: {args.num_prompts}, Concurrency: {args.concurrent}, Max tokens: {args.max_tokens}")

    result = asyncio.run(
        run_benchmark(args.url, args.num_prompts, args.concurrent, args.max_tokens)
    )

    if result:
        print(f"\nResults:")
        print(f"  Total time:  {result['total_time_sec']:.2f}s")
        print(f"  Throughput:  {result['throughput_tok_per_sec']:.1f} tok/s")
        print(f"  Requests/s:  {result['requests_per_sec']:.1f}")
        print(f"  Avg latency: {result['avg_latency_ms']:.1f} ms")
        print(f"  P50 latency: {result['p50_latency_ms']:.1f} ms")
        print(f"  P95 latency: {result['p95_latency_ms']:.1f} ms")
        print(f"  P99 latency: {result['p99_latency_ms']:.1f} ms")
        print(f"  Avg tok/req: {result['avg_tokens_per_request']:.1f}")
        print(f"  Errors:      {result['num_errors']}")

        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
