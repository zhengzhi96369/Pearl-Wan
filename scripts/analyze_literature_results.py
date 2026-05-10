import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


def find_json_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*_results.json")


def parse_strict(path: Path, data: Dict) -> List[Dict]:
    config = data.get("config", {})
    rows = []
    speeds = {run.get("mode"): run.get("avg_speed", 0.0) for run in data.get("runs", [])}
    for run in data.get("runs", []):
        if "mode" not in run:
            continue
        rows.append(
            {
                "kind": "strict_sd",
                "path": str(path),
                "task": path.name.replace("_results.json", ""),
                "draft_model": config.get("draft_model"),
                "target_model": config.get("target_model"),
                "mode": run.get("mode"),
                "speed": run.get("avg_speed", 0.0),
                "accuracy": run.get("accuracy", ""),
                "rtt_ms": config.get("rtt_ms"),
                "bandwidth_mbps": config.get("bandwidth_mbps"),
                "packet_loss_rate": config.get("packet_loss_rate"),
                "gamma": config.get("gamma"),
                "network_simulator": config.get("network_simulator", "legacy"),
                "compression_top_k": config.get("compression_top_k", ""),
                "speedup_vs_ar": (run.get("avg_speed", 0.0) / speeds.get("autoregressive", 0.0)) if speeds.get("autoregressive", 0.0) else "",
                "aggregate_tokens_per_sec": "",
                "mean_latency_sec": "",
                "p95_latency_sec": "",
                "status": "ok",
            }
        )
    return rows


def parse_serving(path: Path, data: Dict) -> List[Dict]:
    config = data.get("config", {})
    return [
        {
            "kind": "cloud_serving",
            "path": str(path),
            "task": config.get("task"),
            "draft_model": "",
            "target_model": config.get("model"),
            "mode": "vllm_openai",
            "speed": "",
            "accuracy": "",
            "rtt_ms": "",
            "bandwidth_mbps": "",
            "packet_loss_rate": "",
            "gamma": "",
            "network_simulator": "",
            "compression_top_k": "",
            "speedup_vs_ar": "",
            "aggregate_tokens_per_sec": data.get("aggregate_tokens_per_sec", 0.0),
            "mean_latency_sec": data.get("mean_latency_sec", 0.0),
            "p95_latency_sec": data.get("p95_latency_sec", 0.0),
            "status": "ok" if data.get("num_errors", 0) == 0 else "partial",
        }
    ]


def load_rows(exp_dir: Path) -> List[Dict]:
    rows = []
    for path in find_json_files(exp_dir):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("cloud_backend") == "vllm_openai":
            rows.extend(parse_serving(path, data))
        else:
            rows.extend(parse_strict(path, data))
    return rows


def write_csv(rows: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "kind",
        "task",
        "draft_model",
        "target_model",
        "mode",
        "speed",
        "speedup_vs_ar",
        "accuracy",
        "rtt_ms",
        "bandwidth_mbps",
        "packet_loss_rate",
        "gamma",
        "network_simulator",
        "compression_top_k",
        "aggregate_tokens_per_sec",
        "mean_latency_sec",
        "p95_latency_sec",
        "status",
        "path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_strict_speed(rows: List[Dict], out_dir: Path):
    strict = [r for r in rows if r["kind"] == "strict_sd" and r["mode"] in {"autoregressive", "speculative_decoding", "wan"}]
    if not strict:
        return
    labels = []
    values = []
    colors = []
    for row in strict[:60]:
        labels.append(f"{row['target_model']}\n{row['mode']}")
        values.append(float(row["speed"] or 0))
        colors.append({"autoregressive": "#4C78A8", "speculative_decoding": "#F58518", "wan": "#54A24B"}.get(row["mode"], "#999999"))
    plt.figure(figsize=(max(10, len(labels) * 0.35), 5))
    plt.bar(range(len(values)), values, color=colors)
    plt.xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=7)
    plt.ylabel("tokens / second")
    plt.title("Strict speculative decoding speed")
    plt.tight_layout()
    plt.savefig(out_dir / "strict_sd_speed.png", dpi=180)
    plt.close()


def plot_network(rows: List[Dict], out_dir: Path):
    wan_rows = [r for r in rows if r["kind"] == "strict_sd" and r["mode"] == "wan" and r.get("rtt_ms") not in {"", None}]
    if not wan_rows:
        return
    groups = {}
    for row in wan_rows:
        key = row["target_model"]
        groups.setdefault(key, []).append((float(row["rtt_ms"]), float(row["speed"] or 0)))
    plt.figure(figsize=(8, 5))
    for key, vals in groups.items():
        vals = sorted(vals)
        plt.plot([v[0] for v in vals], [v[1] for v in vals], marker="o", label=key)
    plt.xlabel("RTT (ms)")
    plt.ylabel("WAN tokens / second")
    plt.title("WAN speed under RTT sweep")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "wan_rtt_curve.png", dpi=180)
    plt.close()


def plot_serving(rows: List[Dict], out_dir: Path):
    serving = [r for r in rows if r["kind"] == "cloud_serving"]
    if not serving:
        return
    labels = [f"{r['target_model']}\nc{Path(r['path']).parent.name.split('_c')[-1]}" for r in serving[:40]]
    values = [float(r["aggregate_tokens_per_sec"] or 0) for r in serving[:40]]
    plt.figure(figsize=(max(8, len(labels) * 0.4), 5))
    plt.bar(range(len(values)), values, color="#B279A2")
    plt.xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=7)
    plt.ylabel("aggregate tokens / second")
    plt.title("Cloud serving throughput")
    plt.tight_layout()
    plt.savefig(out_dir / "cloud_serving_throughput.png", dpi=180)
    plt.close()


def write_report(rows: List[Dict], out_dir: Path, run_id: str):
    strict_count = sum(1 for r in rows if r["kind"] == "strict_sd")
    serving_count = sum(1 for r in rows if r["kind"] == "cloud_serving")
    wan = [float(r["speed"] or 0) for r in rows if r["kind"] == "strict_sd" and r["mode"] == "wan"]
    ar = [float(r["speed"] or 0) for r in rows if r["kind"] == "strict_sd" and r["mode"] == "autoregressive"]
    serving = [float(r["aggregate_tokens_per_sec"] or 0) for r in rows if r["kind"] == "cloud_serving"]
    md = f"""# Pearl-Wan Literature-Driven Experiment Report

Run id: `{run_id}`

## Scope

This report separates strict speculative decoding experiments from cloud serving simulation experiments. Strict runs use local Transformers models and report AR/SD/WAN speed. Serving runs use OpenAI-compatible vLLM endpoints and report latency/throughput.

## Summary

- Strict SD rows: {strict_count}
- Cloud serving rows: {serving_count}
- Mean WAN speed: {(sum(wan) / len(wan)) if wan else 0:.2f} tok/s
- Mean AR speed: {(sum(ar) / len(ar)) if ar else 0:.2f} tok/s
- Mean serving throughput: {(sum(serving) / len(serving)) if serving else 0:.2f} tok/s

## Figures

![Strict SD speed](plots/strict_sd_speed.png)

![WAN RTT curve](plots/wan_rtt_curve.png)

![Cloud serving throughput](plots/cloud_serving_throughput.png)

## Interpretation Guide

- Strict speedup and cloud serving throughput are different metrics and should not be mixed.
- WAN results include packet protocol overhead when `network_simulator=research`.
- Failed, OOM, or unsupported model profiles are expected outcomes for the large-model part of the matrix and should be used to delimit feasible bserver configurations.
"""
    (out_dir / "EXTENDED_RESULTS.md").write_text(md, encoding="utf-8")
    html = "<html><body>" + md.replace("\n", "<br>\n") + "</body></html>"
    (out_dir / "EXTENDED_RESULTS.html").write_text(html, encoding="utf-8")
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf_path = out_dir / "EXTENDED_RESULTS.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        text = c.beginText(50, 750)
        text.setFont("Helvetica", 10)
        for line in md.splitlines():
            if text.getY() < 60:
                c.drawText(text)
                c.showPage()
                text = c.beginText(50, 750)
                text.setFont("Helvetica", 10)
            text.textLine(line[:110])
        c.drawText(text)
        c.save()
    except Exception as exc:
        (out_dir / "PDF_SKIPPED.txt").write_text(str(exc), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default="exp")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    run_id = args.run_id or exp_dir.name
    out_dir = Path(args.output_dir) if args.output_dir else exp_dir / "analysis"
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(exp_dir)
    write_csv(rows, out_dir / "summary.csv")
    plot_strict_speed(rows, plot_dir)
    plot_network(rows, plot_dir)
    plot_serving(rows, plot_dir)
    write_report(rows, out_dir, run_id)
    print(json.dumps({"rows": len(rows), "output_dir": str(out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
