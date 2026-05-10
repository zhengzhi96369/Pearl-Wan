import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


EXPERIMENT_RE = re.compile(r"\[(\d+)/(\d+)\]\s+(.+)")


def run_cmd(cmd: List[str], timeout: int = 10) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception as exc:
        return str(exc)


def latest_run(root: Path) -> Optional[Path]:
    archives = root / "archives"
    candidates = sorted(archives.glob("literature_full_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def parse_driver_log(log_path: Path) -> Dict:
    if not log_path.exists():
        return {"current_index": 0, "total_experiments": 0, "current_command": "", "tail": ""}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    current = {"current_index": 0, "total_experiments": 0, "current_command": ""}
    for line in lines:
        match = EXPERIMENT_RE.match(line.strip())
        if match:
            current = {
                "current_index": int(match.group(1)),
                "total_experiments": int(match.group(2)),
                "current_command": match.group(3),
            }
    current["tail"] = "\n".join(lines[-80:])
    return current


def pid_status(pid_path: Path) -> Dict:
    if not pid_path.exists():
        return {"pid": "", "alive": False, "ps": ""}
    pid = pid_path.read_text(encoding="utf-8").strip()
    if not pid:
        return {"pid": "", "alive": False, "ps": ""}
    ps = run_cmd(["ps", "-p", pid, "-o", "pid,etime,stat,cmd"], timeout=5)
    alive = pid in ps and "CMD" in ps
    return {"pid": pid, "alive": alive, "ps": ps.strip()}


def count_results(exp_dir: Path) -> Dict:
    results = list(exp_dir.rglob("*_results.json")) if exp_dir.exists() else []
    logs = list((exp_dir / "logs").glob("*.log")) if (exp_dir / "logs").exists() else []
    failed = 0
    ok = 0
    for log in logs:
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "Traceback" in text or "ERROR" in text or "Error" in text:
            failed += 1
        elif "Results saved to:" in text:
            ok += 1
    return {
        "result_json_count": len(results),
        "experiment_log_count": len(logs),
        "log_success_count": ok,
        "log_error_count": failed,
    }


def gpu_status() -> List[Dict]:
    out = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    rows = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 7:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_used_mib": parts[2],
                    "memory_total_mib": parts[3],
                    "utilization_gpu_pct": parts[4],
                    "power_w": parts[5],
                    "temperature_c": parts[6],
                }
            )
    return rows


def docker_status() -> List[Dict]:
    out = run_cmd(["docker", "ps", "--format", "{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Command}}"], timeout=10)
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 5:
            rows.append(
                {
                    "id": parts[0],
                    "name": parts[1],
                    "status": parts[2],
                    "image": parts[3],
                    "command": parts[4],
                }
            )
    return rows


def model_download_manifest(root: Path, run_id: str) -> List[Dict]:
    path = root / "exp" / f"{run_id}_model_downloads.jsonl"
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def build_snapshot(root: Path, run_dir: Path) -> Dict:
    run_id = run_dir.name
    exp_dir = root / "exp" / run_id
    log_info = parse_driver_log(run_dir / "driver.log")
    counts = count_results(exp_dir)
    progress = 0.0
    if log_info["total_experiments"]:
        progress = min(1.0, counts["result_json_count"] / max(log_info["total_experiments"], 1))
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "exp_dir": str(exp_dir),
        "pid": pid_status(run_dir / "driver.pid"),
        "driver": log_info,
        "counts": counts,
        "progress_fraction": progress,
        "gpu": gpu_status(),
        "docker": docker_status(),
        "model_downloads": model_download_manifest(root, run_id),
    }
    return snapshot


def render_html(snapshot: Dict) -> str:
    gpu_rows = "\n".join(
        f"<tr><td>{g['index']}</td><td>{g['name']}</td><td>{g['memory_used_mib']}/{g['memory_total_mib']} MiB</td>"
        f"<td>{g['utilization_gpu_pct']}%</td><td>{g['power_w']} W</td><td>{g['temperature_c']} C</td></tr>"
        for g in snapshot["gpu"]
    )
    docker_rows = "\n".join(
        f"<tr><td>{d['name']}</td><td>{d['status']}</td><td>{d['image']}</td><td><code>{d['command']}</code></td></tr>"
        for d in snapshot["docker"]
    )
    downloads = "\n".join(
        f"<tr><td>{m.get('local_name')}</td><td>{m.get('status')}</td><td>{m.get('source')}</td><td>{int(m.get('bytes', 0)) / 1e9:.2f} GB</td></tr>"
        for m in snapshot["model_downloads"][-20:]
    )
    progress_pct = snapshot["progress_fraction"] * 100
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="30">
  <title>Pearl-Wan Monitor - {snapshot['run_id']}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f7f7f4; color: #202020; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    section {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 16px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td, th {{ border-bottom: 1px solid #eee; padding: 6px; text-align: left; font-size: 13px; }}
    pre {{ white-space: pre-wrap; max-height: 520px; overflow: auto; background: #111; color: #eee; padding: 12px; border-radius: 6px; }}
    progress {{ width: 100%; height: 18px; }}
    code {{ font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Pearl-Wan Experiment Monitor</h1>
  <p><b>Run:</b> {snapshot['run_id']} | <b>Updated:</b> {snapshot['timestamp']}</p>
  <progress max="100" value="{progress_pct:.2f}"></progress>
  <p>{snapshot['counts']['result_json_count']} result JSON / {snapshot['driver']['total_experiments'] or '?'} planned strict experiments</p>
  <div class="grid">
    <section>
      <h2>Process</h2>
      <pre>{snapshot['pid']['ps']}</pre>
    </section>
    <section>
      <h2>Current Experiment</h2>
      <p><b>{snapshot['driver']['current_index']}/{snapshot['driver']['total_experiments']}</b></p>
      <pre>{snapshot['driver']['current_command']}</pre>
    </section>
    <section>
      <h2>GPU</h2>
      <table><tr><th>ID</th><th>Name</th><th>Memory</th><th>Util</th><th>Power</th><th>Temp</th></tr>{gpu_rows}</table>
    </section>
    <section>
      <h2>Docker</h2>
      <table><tr><th>Name</th><th>Status</th><th>Image</th><th>Command</th></tr>{docker_rows}</table>
    </section>
    <section>
      <h2>Model Downloads</h2>
      <table><tr><th>Model</th><th>Status</th><th>Source</th><th>Size</th></tr>{downloads}</table>
    </section>
  </div>
  <h2>Driver Log Tail</h2>
  <pre>{snapshot['driver']['tail']}</pre>
</body>
</html>"""


def write_snapshot(snapshot: Dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "monitor.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "monitor.html").write_text(render_html(snapshot), encoding="utf-8")
    line = {
        "timestamp": snapshot["timestamp"],
        "run_id": snapshot["run_id"],
        "current": snapshot["driver"]["current_index"],
        "total": snapshot["driver"]["total_experiments"],
        "results": snapshot["counts"]["result_json_count"],
        "alive": snapshot["pid"]["alive"],
    }
    with (output_dir / "monitor.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Monitor Pearl-Wan literature experiment progress.")
    parser.add_argument("--root", default=os.getcwd())
    parser.add_argument("--run-id", default="")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    while True:
        run_dir = root / "archives" / args.run_id if args.run_id else latest_run(root)
        if not run_dir:
            raise SystemExit("no literature_full run found")
        output_dir = Path(args.output_dir) if args.output_dir else run_dir / "monitor"
        snapshot = build_snapshot(root, run_dir)
        write_snapshot(snapshot, output_dir)
        print(
            f"[{snapshot['timestamp']}] {snapshot['run_id']} "
            f"{snapshot['driver']['current_index']}/{snapshot['driver']['total_experiments']} "
            f"results={snapshot['counts']['result_json_count']} alive={snapshot['pid']['alive']}",
            flush=True,
        )
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
