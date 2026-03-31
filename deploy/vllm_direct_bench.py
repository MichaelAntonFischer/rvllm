#!/usr/bin/env python3
"""Direct engine benchmark for Python vLLM (no HTTP).

Matches rvLLM's direct engine measurement: call the engine API directly,
bypass all HTTP/networking overhead. Measures pure generation throughput.

Usage:
    python3 deploy/vllm_direct_bench.py --model Qwen/Qwen2.5-7B --max-tokens 128
"""

import time
import json
import argparse

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


def bench_concurrency(llm, sampling_params, n, max_tokens, warmup=True):
    """Benchmark at concurrency N by batching N prompts."""
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n)]

    if warmup:
        llm.generate(prompts[:min(n, 4)], sampling_params)

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tok_per_sec = total_tokens / elapsed

    return {
        "n": n,
        "total_tokens": total_tokens,
        "elapsed_sec": round(elapsed, 3),
        "tok_per_sec": round(tok_per_sec, 1),
        "avg_tokens_per_req": round(total_tokens / n, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM direct engine benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--output", default="/root/results_vllm_direct.json")
    parser.add_argument("--concurrency", type=str, default="1,4,16,32,64,128",
                        help="Comma-separated concurrency levels")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    print(f"Loading model: {args.model}")
    load_start = time.perf_counter()
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        max_model_len=4096,
    )
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=args.max_tokens,
    )

    concurrency_levels = [int(x) for x in args.concurrency.split(",")]
    results = []

    print(f"\nDirect engine benchmark (no HTTP)")
    print(f"Model: {args.model}, max_tokens={args.max_tokens}")
    print(f"{'N':>6} | {'tok/s':>10} | {'tokens':>8} | {'elapsed':>8}")
    print("-" * 45)

    for n in concurrency_levels:
        r = bench_concurrency(llm, sampling_params, n, args.max_tokens, warmup=(n == concurrency_levels[0]))
        results.append(r)
        print(f"{r['n']:>6} | {r['tok_per_sec']:>10,.1f} | {r['total_tokens']:>8,} | {r['elapsed_sec']:>7.2f}s")

    output = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "load_time_sec": round(load_time, 1),
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
