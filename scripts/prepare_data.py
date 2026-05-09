#!/usr/bin/env python3
import argparse
import json
import os
from datasets import load_dataset


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_n(dataset, limit):
    if limit and limit > 0:
        return dataset.select(range(min(limit, len(dataset))))
    return dataset


def prepare_humaneval(output_dir, limit):
    data = first_n(load_dataset("openai/openai_humaneval", split="test"), limit)
    rows = [{"task_id": item["task_id"], "prompt": item["prompt"]} for item in data]
    write_jsonl(os.path.join(output_dir, "humaneval.jsonl"), rows)


def prepare_gsm8k(output_dir, limit):
    data = first_n(load_dataset("openai/gsm8k", "main", split="test"), limit)
    rows = [{"question": item["question"], "answer": item["answer"]} for item in data]
    write_jsonl(os.path.join(output_dir, "gsm8k.jsonl"), rows)


def prepare_mgsm(output_dir, limit):
    data = first_n(load_dataset("juletxara/mgsm", "en", split="test", trust_remote_code=True), limit)
    rows = []
    for item in data:
        rows.append({
            "question": item["question"],
            "answer": str(item.get("answer_number", item.get("answer", ""))),
            "category": "en",
        })
    write_jsonl(os.path.join(output_dir, "mgsm.jsonl"), rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare PEARL-WAN benchmark JSONL files.")
    parser.add_argument("--output-dir", default=os.environ.get("PEARL_WAN_DATA_DIR", "data"))
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    prepare_humaneval(args.output_dir, args.limit)
    prepare_gsm8k(args.output_dir, args.limit)
    prepare_mgsm(args.output_dir, args.limit)

    for name in ["humaneval.jsonl", "gsm8k.jsonl", "mgsm.jsonl"]:
        path = os.path.join(args.output_dir, name)
        with open(path, encoding="utf-8") as f:
            count = sum(1 for _ in f)
        print(f"{path}: {count} rows")


if __name__ == "__main__":
    main()
