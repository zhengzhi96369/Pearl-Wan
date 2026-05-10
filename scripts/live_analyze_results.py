import argparse
import json
import os
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


RESULT_RE = re.compile(r"strict_(\d+)_([^_]+)_([^_]+)_")


def load_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def result_rows(exp_dir: Path) -> List[Dict]:
    rows = []
    for path in sorted(exp_dir.rglob("*_results.json"), key=lambda p: p.stat().st_mtime):
        data = load_json(path)
        config = data.get("config", {})
        parent = path.parent.name
        match = RESULT_RE.search(parent)
        task = match.group(3) if match else path.stem.replace("_results", "")
        speeds = {run.get("mode"): run.get("avg_speed", 0.0) for run in data.get("runs", [])}
        row = {
            "name": parent,
            "path": str(path),
            "task": task,
            "draft_model": config.get("draft_model", ""),
            "target_model": config.get("target_model", ""),
            "rtt_ms": config.get("rtt_ms", ""),
            "bandwidth_mbps": config.get("bandwidth_mbps", ""),
            "gamma": config.get("gamma", ""),
            "ar_speed": speeds.get("autoregressive", 0.0),
            "sd_speed": speeds.get("speculative_decoding", 0.0),
            "wan_speed": speeds.get("wan", 0.0),
        }
        row["wan_vs_ar"] = row["wan_speed"] / row["ar_speed"] if row["ar_speed"] else 0.0
        row["wan_vs_sd"] = row["wan_speed"] / row["sd_speed"] if row["sd_speed"] else 0.0
        rows.append(row)
    return rows


def error_rows(exp_dir: Path) -> List[Dict]:
    rows = []
    log_dir = exp_dir / "logs"
    for path in sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime) if log_dir.exists() else []:
        text = path.read_text(encoding="utf-8", errors="replace")
        marker = ""
        for needle in ["Traceback", "RuntimeError", "CUDA out of memory", "ERROR", "Error"]:
            if needle in text:
                marker = needle
                break
        if marker:
            lines = [line.strip() for line in text.splitlines() if marker in line or "Exception" in line or "Error" in line]
            rows.append({"log": path.name, "message": (lines[-1] if lines else marker)[:240]})
    return rows


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def build_analysis(root: Path, run_id: str) -> Dict:
    exp_dir = root / "exp" / run_id
    rows = result_rows(exp_dir)
    errors = error_rows(exp_dir)
    mode_speeds = {
        "autoregressive": mean([row["ar_speed"] for row in rows if row["ar_speed"]]),
        "speculative_decoding": mean([row["sd_speed"] for row in rows if row["sd_speed"]]),
        "wan": mean([row["wan_speed"] for row in rows if row["wan_speed"]]),
    }
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row["wan_speed"])
    bullets = []
    if rows:
        bullets.append(f"Completed {len(rows)} strict SD result files across {len(by_task)} task groups.")
        if mode_speeds["wan"] and mode_speeds["autoregressive"]:
            bullets.append(f"Mean WAN/AR ratio so far is {mode_speeds['wan'] / mode_speeds['autoregressive']:.3f}.")
        fastest = max(rows, key=lambda row: row["wan_speed"])
        bullets.append(f"Fastest WAN run so far is {fastest['task']} at {fastest['wan_speed']:.2f} tok/s.")
        if errors:
            bullets.append(f"{len(errors)} logs contain error markers; inspect Error Signals before final conclusions.")
        else:
            bullets.append("No error markers have appeared in experiment logs so far.")
    else:
        bullets.append("Waiting for the first strict SD result JSON.")

    task_summary = {
        task: {
            "count": len(vals),
            "mean_wan_speed": mean([v for v in vals if v]),
        }
        for task, vals in sorted(by_task.items())
    }
    summary = (
        f"{len(rows)} results available. "
        f"Mean WAN speed {mode_speeds['wan']:.2f} tok/s; "
        f"mean AR speed {mode_speeds['autoregressive']:.2f} tok/s."
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "summary": summary,
        "completed_results": len(rows),
        "mode_speeds": mode_speeds,
        "task_summary": task_summary,
        "bullets": bullets,
        "recent": rows[-12:],
        "errors": errors[-20:],
    }


def render_html(data: Dict) -> str:
    bullets = "\n".join(f"<li>{item}</li>" for item in data["bullets"])
    tasks = "\n".join(
        f"<tr><td>{task}</td><td>{row['count']}</td><td>{row['mean_wan_speed']:.2f}</td></tr>"
        for task, row in data["task_summary"].items()
    )
    recent = "\n".join(
        f"<tr><td>{row['name']}</td><td>{row['task']}</td><td>{row['wan_speed']:.2f}</td><td>{row['wan_vs_ar']:.3f}</td></tr>"
        for row in data["recent"]
    )
    errors = "\n".join(f"<tr><td>{row['log']}</td><td>{row['message']}</td></tr>" for row in data["errors"])
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="20">
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
td, th {{ border-bottom: 1px solid #ddd; padding: 6px; text-align: left; }}
</style></head><body>
<h1>Live Analysis Agent</h1>
<p>{data['summary']}</p>
<ul>{bullets}</ul>
<h2>Task Summary</h2><table><tr><th>Task</th><th>Count</th><th>Mean WAN tok/s</th></tr>{tasks}</table>
<h2>Recent Results</h2><table><tr><th>Name</th><th>Task</th><th>WAN tok/s</th><th>WAN/AR</th></tr>{recent}</table>
<h2>Errors</h2><table><tr><th>Log</th><th>Message</th></tr>{errors}</table>
</body></html>"""


def write_outputs(data: Dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "live_analysis.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "live_analysis.html").write_text(render_html(data), encoding="utf-8")
    with (output_dir / "analysis_agent.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": data["timestamp"], "completed_results": data["completed_results"]}) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Continuously analyze Pearl-Wan results as they arrive.")
    parser.add_argument("--root", default=os.getcwd())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else root / "archives" / args.run_id / "analysis_agent"
    while True:
        data = build_analysis(root, args.run_id)
        write_outputs(data, output_dir)
        print(f"[{data['timestamp']}] analyzed {data['completed_results']} results", flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
