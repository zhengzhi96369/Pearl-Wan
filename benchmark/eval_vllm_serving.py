import argparse
import concurrent.futures
import json
import os
import statistics
import time
from typing import Dict, List

import requests


PROMPTS = {
    "speed": [
        "Explain speculative decoding in one paragraph.",
        "Describe how cloud-edge LLM inference can handle WAN latency.",
        "List three causes of high tail latency in model serving.",
        "Write a short note about batching in LLM serving.",
    ],
    "humaneval": [
        "Write a Python function fibonacci(n) that returns the nth Fibonacci number.",
        "Write a Python function is_palindrome(s) that ignores case and spaces.",
    ],
    "gsm8k": [
        "A store sells 12 apples and then buys 30 more. It sells 9. How many apples remain?",
        "If a train travels 60 miles in 2 hours, what is its average speed?",
    ],
    "math500": [
        "If 4x - 5 = 19, solve for x.",
        "What is the area of a triangle with base 10 and height 7?",
    ],
    "mtbench": [
        "Compare speculative decoding and ordinary autoregressive decoding.",
        "Give a concise plan for evaluating an LLM serving system.",
    ],
}


def request_once(base_url: str, model: str, prompt: str, max_tokens: int, temperature: float, timeout: float) -> Dict:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    start = time.time()
    response = requests.post(url, json=payload, timeout=timeout)
    elapsed = time.time() - start
    response.raise_for_status()
    data = response.json()
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    return {
        "prompt": prompt,
        "latency_sec": elapsed,
        "completion_tokens": completion_tokens,
        "tokens_per_sec": completion_tokens / elapsed if elapsed > 0 else 0.0,
        "usage": usage,
        "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
    }


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((pct / 100.0) * (len(values) - 1)))))
    return values[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", default="speed", choices=sorted(PROMPTS))
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--num_requests", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--exp_name", default="vllm_serving_test")
    args = parser.parse_args()

    exp_dir = os.path.join(os.getcwd(), "exp", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    prompts = (PROMPTS[args.task] * ((args.num_requests // len(PROMPTS[args.task])) + 1))[: args.num_requests]

    errors = []
    results = []
    wall_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(request_once, args.base_url, args.model, prompt, args.max_tokens, args.temperature, args.timeout)
            for prompt in prompts
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append(str(exc))
    wall_time = time.time() - wall_start

    latencies = [row["latency_sec"] for row in results]
    tokens = sum(row["completion_tokens"] for row in results)
    summary = {
        "config": vars(args),
        "cloud_backend": "vllm_openai",
        "num_success": len(results),
        "num_errors": len(errors),
        "wall_time_sec": wall_time,
        "total_completion_tokens": tokens,
        "aggregate_tokens_per_sec": tokens / wall_time if wall_time > 0 else 0.0,
        "mean_latency_sec": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency_sec": percentile(latencies, 50),
        "p95_latency_sec": percentile(latencies, 95),
        "errors": errors[:10],
        "runs": results,
    }
    out_path = os.path.join(exp_dir, "eval_vllm_serving_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
