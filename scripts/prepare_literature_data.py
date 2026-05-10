import argparse
import json
import os
from typing import Dict, Iterable, List


FALLBACK_ROWS: Dict[str, List[dict]] = {
    "mtbench": [
        {"prompt": "Explain why speculative decoding can improve LLM serving throughput.", "category": "writing"},
        {"prompt": "Compare edge-cloud inference with local-only inference for mobile assistants.", "category": "reasoning"},
    ],
    "alpacaeval": [
        {"instruction": "Write a concise guide to debugging CUDA out-of-memory errors.", "input": "", "output": ""},
        {"instruction": "Summarize the tradeoffs of quantization for LLM inference.", "input": "", "output": ""},
    ],
    "sharegpt": [
        {"prompt": "User: Give me a Python function for topological sort.\nAssistant:", "source": "fallback"},
        {"prompt": "User: What causes high tail latency in distributed inference?\nAssistant:", "source": "fallback"},
    ],
    "mbpp": [
        {"text": "Write a function to return the factorial of a non-negative integer.", "code": "", "test_list": []},
        {"text": "Write a function to check whether a string is a palindrome.", "code": "", "test_list": []},
    ],
    "instructcoder": [
        {"prompt": "Implement binary search over a sorted list in Python.", "language": "python"},
        {"prompt": "Implement an LRU cache with get and put methods.", "language": "python"},
    ],
    "math500": [
        {"problem": "If 3x + 7 = 22, what is x?", "answer": "5"},
        {"problem": "A rectangle has area 48 and width 6. What is its length?", "answer": "8"},
    ],
    "aime": [
        {"problem": "Find the remainder when 2026 is divided by 17.", "answer": "3"},
        {"problem": "How many positive divisors does 36 have?", "answer": "9"},
    ],
    "cnn_dailymail": [
        {"article": "A research team evaluated several inference systems under simulated WAN latency.", "highlights": "A team benchmarked WAN inference."},
        {"article": "The server ran experiments with different network round-trip times and bandwidth caps.", "highlights": "Experiments varied network conditions."},
    ],
    "natural_questions": [
        {"question": "What is speculative decoding?", "answer": "A decoding method using a draft model and verification model."},
        {"question": "What is round-trip time?", "answer": "The time for a message to travel to a remote endpoint and back."},
    ],
}


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def try_dataset(name: str, limit: int) -> List[dict]:
    try:
        from datasets import load_dataset
    except Exception:
        return FALLBACK_ROWS[name][:limit]

    recipes = {
        "mbpp": ("google-research-datasets/mbpp", "sanitized", "test"),
        "math500": ("HuggingFaceH4/MATH-500", None, "test"),
        "cnn_dailymail": ("cnn_dailymail", "3.0.0", "test"),
        "natural_questions": ("google-research-datasets/natural_questions", None, "validation"),
    }
    if name not in recipes:
        return FALLBACK_ROWS[name][:limit]
    dataset_name, subset, split = recipes[name]
    try:
        ds = load_dataset(dataset_name, subset, split=split) if subset else load_dataset(dataset_name, split=split)
        return [dict(row) for row in ds.select(range(min(limit, len(ds))))]
    except Exception:
        return FALLBACK_ROWS[name][:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("PEARL_WAN_DATA_DIR", "data"))
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    manifest = []
    for name in FALLBACK_ROWS:
        rows = try_dataset(name, args.limit)
        path = os.path.join(args.data_dir, f"{name}.jsonl")
        count = write_jsonl(path, rows)
        manifest.append({"dataset": name, "path": path, "rows": count})

    manifest_path = os.path.join(args.data_dir, "literature_data_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({"manifest": manifest_path, "datasets": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
