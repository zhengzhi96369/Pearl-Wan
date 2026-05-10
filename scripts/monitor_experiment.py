import argparse
import html
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


def live_analysis(run_dir: Path) -> Dict:
    path = run_dir / "analysis_agent" / "live_analysis.json"
    if not path.exists():
        return {
            "available": False,
            "summary": "Live analysis agent has not produced a snapshot yet.",
            "bullets": [],
            "mode_speeds": {},
            "recent": [],
            "errors": [],
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["available"] = True
        return data
    except Exception as exc:
        return {
            "available": False,
            "summary": f"Could not read live analysis: {exc}",
            "bullets": [],
            "mode_speeds": {},
            "recent": [],
            "errors": [],
        }


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
        "analysis": live_analysis(run_dir),
    }
    return snapshot


def render_html(snapshot: Dict) -> str:
    current_command = html.escape(snapshot["driver"]["current_command"])
    log_tail = html.escape(snapshot["driver"]["tail"])
    process_text = html.escape(snapshot["pid"]["ps"])
    run_id = html.escape(snapshot["run_id"])
    updated = html.escape(snapshot["timestamp"])
    progress_pct = snapshot["progress_fraction"] * 100
    current = snapshot["driver"]["current_index"]
    total = snapshot["driver"]["total_experiments"]
    results = snapshot["counts"]["result_json_count"]
    failed = snapshot["counts"]["log_error_count"]
    analysis = snapshot["analysis"]
    phase = "Running strict SD matrix" if total else "Preparing experiment queue"
    if not snapshot["pid"]["alive"]:
        phase = "Driver stopped"
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
    mode_rows = "\n".join(
        f"<tr><td>{html.escape(str(mode))}</td><td>{float(speed):.2f} tok/s</td></tr>"
        for mode, speed in analysis.get("mode_speeds", {}).items()
    )
    bullets = "\n".join(f"<li>{html.escape(str(item))}</li>" for item in analysis.get("bullets", []))
    recent_rows = "\n".join(
        f"<tr><td>{html.escape(str(row.get('name', '')))}</td><td>{html.escape(str(row.get('task', '')))}</td><td>{html.escape(str(row.get('wan_speed', '')))}</td></tr>"
        for row in analysis.get("recent", [])[-12:]
    )
    error_rows = "\n".join(
        f"<tr><td>{html.escape(str(row.get('log', '')))}</td><td>{html.escape(str(row.get('message', '')))}</td></tr>"
        for row in analysis.get("errors", [])[-8:]
    )
    analysis_summary = html.escape(str(analysis.get("summary", "No analysis yet.")))
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="10">
  <title>Pearl-Wan Setup - {run_id}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; min-height: 100vh; display: grid; place-items: center;
      background: #2f3437; font-family: "Segoe UI", system-ui, sans-serif; color: #202020;
    }}
    .window {{
      width: min(980px, calc(100vw - 32px)); min-height: 680px; background: #f4f4f4;
      border: 1px solid #1f2427; box-shadow: 0 28px 90px rgba(0,0,0,.45);
    }}
    .titlebar {{
      height: 38px; display: flex; align-items: center; justify-content: space-between;
      padding: 0 12px; color: #f8f8f8; background: linear-gradient(#3c4246, #272c2f); font-size: 13px;
    }}
    .controls span {{
      display: inline-grid; place-items: center; width: 30px; height: 22px; margin-left: 4px;
      border: 1px solid rgba(255,255,255,.18); color: #ddd;
    }}
    .hero {{
      display: grid; grid-template-columns: 180px 1fr; min-height: 172px;
      background: linear-gradient(90deg, #d9e4ee, #fbfbfb); border-bottom: 1px solid #d2d2d2;
    }}
    .brand {{
      display: grid; place-items: center; background: #24445f; color: white;
      font-size: 54px; font-weight: 700; letter-spacing: 0;
    }}
    .intro {{ padding: 28px 34px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; font-weight: 600; }}
    .muted {{ color: #60686e; font-size: 13px; }}
    .content {{ padding: 26px 34px 18px; }}
    .progress-shell {{
      width: 100%; height: 28px; padding: 3px; border: 1px solid #8d969c;
      background: #ffffff; box-shadow: inset 0 1px 2px rgba(0,0,0,.18); margin: 12px 0 8px;
    }}
    .progress-fill {{
      height: 100%; width: {progress_pct:.2f}%; min-width: 2px;
      background: repeating-linear-gradient(45deg, #2e8b57 0, #2e8b57 14px, #37a46a 14px, #37a46a 28px);
      transition: width .4s ease;
    }}
    .status-line {{ display: flex; justify-content: space-between; gap: 16px; font-size: 13px; color: #30383d; margin-bottom: 22px; }}
    .steps {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 18px; }}
    .step {{ border: 1px solid #c8ced3; background: white; padding: 10px; min-height: 74px; }}
    .step b {{ display: block; font-size: 18px; margin-bottom: 4px; }}
    .panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    section {{ background: white; border: 1px solid #c8ced3; padding: 12px; min-height: 150px; }}
    h2 {{ margin: 0 0 10px; font-size: 14px; font-weight: 600; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td, th {{ border-bottom: 1px solid #eceff1; padding: 6px; text-align: left; font-size: 12px; }}
    pre {{ margin: 0; white-space: pre-wrap; max-height: 210px; overflow: auto; background: #111820; color: #e8edf2; padding: 10px; font-size: 12px; }}
    .footer {{ height: 58px; display: flex; align-items: center; justify-content: flex-end; gap: 10px; padding: 0 34px; border-top: 1px solid #d0d0d0; background: #ededed; }}
    button {{ min-width: 90px; height: 30px; border: 1px solid #9aa4aa; background: #fafafa; color: #777; }}
    .tabs {{ display: flex; gap: 4px; margin-bottom: 0; }}
    .tab {{ padding: 9px 16px; border: 1px solid #b8c0c6; border-bottom: 0; background: #e7e9eb; font-size: 13px; }}
    .tab.active {{ background: white; font-weight: 600; }}
    .tab-body {{ border: 1px solid #b8c0c6; background: white; padding: 14px; margin-bottom: 18px; }}
    .analysis-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    ul {{ margin: 8px 0 0 18px; padding: 0; }}
  </style>
</head>
<body>
  <main class="window">
    <div class="titlebar"><div>Pearl-Wan Experiment Setup</div><div class="controls"><span>-</span><span>□</span><span>×</span></div></div>
    <div class="hero">
      <div class="brand">PW</div>
      <div class="intro">
        <h1>Running literature-driven WAN experiments</h1>
        <div class="muted">Run: {run_id}</div>
        <div class="muted">Last update: {updated}</div>
      </div>
    </div>
    <div class="content">
      <div><b>{html.escape(phase)}</b></div>
      <div class="progress-shell"><div class="progress-fill"></div></div>
      <div class="status-line"><span>{results} result files completed from {total or "?"} planned strict experiments</span><span>{progress_pct:.1f}%</span></div>
      <div class="tabs">
        <div class="tab active">Status</div>
        <div class="tab">Live Analysis</div>
      </div>
      <div class="tab-body">
        <div class="steps">
          <div class="step"><b>{current}</b><span>Current experiment</span></div>
          <div class="step"><b>{results}</b><span>Result JSON files</span></div>
          <div class="step"><b>{failed}</b><span>Logged errors</span></div>
          <div class="step"><b>{"Alive" if snapshot["pid"]["alive"] else "Stopped"}</b><span>Driver status</span></div>
        </div>
      </div>
      <div class="tab-body">
        <h2>Live Analysis Agent</h2>
        <p class="muted">{analysis_summary}</p>
        <div class="analysis-grid">
          <section><h2>Early Findings</h2><ul>{bullets or "<li>Waiting for more completed results.</li>"}</ul></section>
          <section><h2>Average Speeds</h2><table><tr><th>Mode</th><th>Average</th></tr>{mode_rows}</table></section>
          <section><h2>Recent Results</h2><table><tr><th>Experiment</th><th>Task</th><th>WAN speed</th></tr>{recent_rows}</table></section>
          <section><h2>Error Signals</h2><table><tr><th>Log</th><th>Message</th></tr>{error_rows}</table></section>
        </div>
      </div>
      <div class="panels">
        <section><h2>Current Command</h2><pre>{current_command}</pre></section>
        <section><h2>Process</h2><pre>{process_text}</pre></section>
        <section><h2>GPU</h2><table><tr><th>ID</th><th>Name</th><th>Memory</th><th>Util</th><th>Power</th><th>Temp</th></tr>{gpu_rows}</table></section>
        <section><h2>Docker</h2><table><tr><th>Name</th><th>Status</th><th>Image</th><th>Command</th></tr>{docker_rows}</table></section>
        <section><h2>Model Downloads</h2><table><tr><th>Model</th><th>Status</th><th>Source</th><th>Size</th></tr>{downloads}</table></section>
        <section><h2>Driver Log Tail</h2><pre>{log_tail}</pre></section>
      </div>
    </div>
    <div class="footer"><button disabled>Back</button><button disabled>Next</button><button disabled>Cancel</button></div>
  </main>
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
