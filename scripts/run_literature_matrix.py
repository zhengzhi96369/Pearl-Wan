import argparse
import json
import os
import subprocess
import sys
import time
from itertools import count
from typing import Dict, Iterable, List


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(ROOT, "configs", "literature_matrix.json")


TASK_TO_SCRIPT = {
    "humaneval": "benchmark/eval_humaneval_wan.py",
    "gsm8k": "benchmark/eval_gsm8k_wan.py",
    "mgsm": "benchmark/eval_mgsm_wan.py",
    "speed": "benchmark/eval_wan.py",
}


def load_config() -> Dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def pair_key(pair: Dict) -> str:
    return f"{pair['draft']}->{pair['target']}"


def selected_pairs(config: Dict, profile_name: str) -> List[Dict]:
    profile = config["profiles"][profile_name]
    if "strict_pairs" in profile:
        wanted = set(profile["strict_pairs"])
        return [pair for pair in config["strict_sd_pairs"] if pair_key(pair) in wanted]
    tiers = set(profile.get("strict_pair_tiers", []))
    return [pair for pair in config["strict_sd_pairs"] if pair["tier"] in tiers]


def sweep_cases(config: Dict, profile_name: str) -> Iterable[Dict]:
    variables = config["variables"]
    profile = config["profiles"][profile_name]
    base = {
        "gamma": 4,
        "rtt_ms": 50,
        "bandwidth_mbps": 100,
        "packet_loss_rate": 0.0,
        "max_tokens": 64,
        "compression_top_k": 50,
        "network_simulator": "research",
        "loss_model": "random",
        "jitter_model": "uniform",
    }
    if profile_name == "tiny":
        yield {
            **base,
            "gamma": profile["gamma"][0],
            "rtt_ms": profile["rtt_ms"][0],
            "bandwidth_mbps": profile["bandwidth_mbps"][0],
            "packet_loss_rate": profile["packet_loss_rate"][0],
            "max_tokens": profile["max_tokens"][0],
            "sweep": "tiny",
        }
        return

    yield {**base, "sweep": "baseline"}
    for gamma in variables["gamma"]:
        yield {**base, "gamma": gamma, "sweep": f"gamma{gamma}"}
    for rtt in variables["rtt_ms"]:
        yield {**base, "rtt_ms": rtt, "sweep": f"rtt{rtt}"}
    for bandwidth in variables["bandwidth_mbps"]:
        yield {**base, "bandwidth_mbps": bandwidth, "sweep": f"bw{bandwidth}"}
    for loss in variables["packet_loss_rate"]:
        yield {
            **base,
            "packet_loss_rate": loss,
            "loss_model": "gilbert_elliott" if loss > 0 else "random",
            "sweep": f"loss{loss}",
        }
    for max_tokens in variables["max_tokens"]:
        yield {**base, "max_tokens": max_tokens, "sweep": f"tok{max_tokens}"}
    for top_k in variables["compression_top_k"]:
        yield {**base, "compression_top_k": top_k, "sweep": f"topk{top_k}"}


def strict_commands(config: Dict, profile_name: str, run_id: str) -> Iterable[List[str]]:
    profile = config["profiles"][profile_name]
    tasks = profile.get("tasks") or profile.get("strict_tasks") or ["speed"]
    limit = int(profile.get("limit", 20))
    exp_counter = count(1)
    for pair in selected_pairs(config, profile_name):
        for task in tasks:
            script = TASK_TO_SCRIPT.get(task)
            if not script:
                continue
            for case in sweep_cases(config, profile_name):
                exp_id = next(exp_counter)
                exp_name = (
                    f"{run_id}/strict_{exp_id:04d}_{pair['family']}_{task}_"
                    f"{pair['draft']}_{pair['target']}_{case['sweep']}"
                )
                cmd = [
                    sys.executable,
                    script,
                    "--draft_model",
                    pair["draft"],
                    "--target_model",
                    pair["target"],
                    "--exp_name",
                    exp_name,
                    "--eval_mode",
                    "wan",
                    "--max_tokens",
                    str(case["max_tokens"]),
                    "--temp",
                    "0.0",
                    "--gamma",
                    str(case["gamma"]),
                    "--rtt_ms",
                    str(case["rtt_ms"]),
                    "--bandwidth_mbps",
                    str(case["bandwidth_mbps"]),
                    "--packet_loss_rate",
                    str(case["packet_loss_rate"]),
                    "--network_simulator",
                    case["network_simulator"],
                    "--loss_model",
                    case["loss_model"],
                    "--jitter_model",
                    case["jitter_model"],
                    "--compression_top_k",
                    str(case["compression_top_k"]),
                    "--device_edge",
                    os.environ.get("PEARL_DEVICE_EDGE", "cuda"),
                    "--device_cloud",
                    os.environ.get("PEARL_DEVICE_CLOUD", "cuda"),
                    "--enable_adaptive_window",
                    "--enable_compression",
                    "--enable_fallback",
                ]
                if task == "speed":
                    cmd.extend(["--num_samples", str(max(1, min(4, limit)))])
                else:
                    cmd.extend(["--limit", str(limit)])
                yield cmd


def serving_commands(config: Dict, profile_name: str, run_id: str) -> Iterable[List[str]]:
    if profile_name == "tiny":
        return
    profile = config["profiles"][profile_name]
    variables = config["variables"]
    model_names = set(profile.get("serving_models", []))
    tasks = profile.get("serving_tasks", ["speed"])
    exp_counter = count(1)
    for model in config["serving_cloud_models"]:
        if model["name"] not in model_names:
            continue
        for task in tasks:
            if task not in model.get("tasks", []) and task != "speed":
                continue
            for concurrency in variables["concurrency"]:
                exp_id = next(exp_counter)
                exp_name = f"{run_id}/serving_{exp_id:04d}_{model['name']}_{task}_c{concurrency}"
                yield [
                    sys.executable,
                    "benchmark/eval_vllm_serving.py",
                    "--model",
                    model["name"],
                    "--task",
                    task if task in {"speed", "humaneval", "gsm8k", "math500", "mtbench"} else "speed",
                    "--concurrency",
                    str(concurrency),
                    "--num_requests",
                    str(max(concurrency, 4)),
                    "--max_tokens",
                    "128",
                    "--base_url",
                    os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
                    "--exp_name",
                    exp_name,
                ]


def run_command(cmd: List[str], log_path: str, dry_run: bool) -> Dict:
    started = time.time()
    row = {"cmd": cmd, "log_path": log_path, "started_at": started}
    if dry_run:
        row.update({"status": "dry_run", "returncode": 0, "ended_at": time.time()})
        return row
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    row.update({"status": "ok" if proc.returncode == 0 else "failed", "returncode": proc.returncode, "ended_at": time.time()})
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="tiny", choices=["tiny", "paper"])
    parser.add_argument("--run-id", default=time.strftime("literature_%Y%m%d-%H%M%S"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-serving", action="store_true")
    parser.add_argument("--max-experiments", type=int, default=0)
    args = parser.parse_args()

    config = load_config()
    archive_dir = os.path.join(ROOT, "exp", args.run_id)
    os.makedirs(archive_dir, exist_ok=True)
    commands = list(strict_commands(config, args.profile, args.run_id))
    if args.include_serving:
        commands.extend(list(serving_commands(config, args.profile, args.run_id) or []))
    if args.max_experiments > 0:
        commands = commands[: args.max_experiments]

    manifest = {
        "run_id": args.run_id,
        "profile": args.profile,
        "include_serving": args.include_serving,
        "dry_run": args.dry_run,
        "num_experiments": len(commands),
        "references": config["references"],
        "commands": [],
    }
    manifest_path = os.path.join(archive_dir, "manifest.json")
    for i, cmd in enumerate(commands, 1):
        log_path = os.path.join(archive_dir, "logs", f"experiment_{i:04d}.log")
        print(f"[{i}/{len(commands)}] {' '.join(cmd)}")
        manifest["commands"].append(run_command(cmd, log_path, args.dry_run))
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
