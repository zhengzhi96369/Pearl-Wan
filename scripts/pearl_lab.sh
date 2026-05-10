#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_ID="${PEARL_RUN_ID:-literature_$(date +%Y%m%d-%H%M%S)}"
ARCHIVE_ROOT="${PEARL_ARCHIVE_ROOT:-$ROOT/archives}"
ARCHIVE_DIR="$ARCHIVE_ROOT/$RUN_ID"
GPU_DEVICES="${PEARL_GPU_DEVICES:-all}"
MAIL_TO="${PEARL_MAIL_TO:-2251645084@qq.com}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

docker_run() {
    docker run --rm --gpus "$GPU_DEVICES" \
        -e HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
        -e HF_HOME=/workspace/.cache/huggingface \
        -e CUDA_VISIBLE_DEVICES="${PEARL_CUDA_VISIBLE_DEVICES:-0}" \
        -e PEARL_WAN_MODEL_DIR=/workspace/models \
        -e PEARL_WAN_DATA_DIR=/workspace/pearl_wan/data \
        -e PEARL_DEVICE_EDGE="${PEARL_DEVICE_EDGE:-cuda}" \
        -e PEARL_DEVICE_CLOUD="${PEARL_DEVICE_CLOUD:-cuda}" \
        -e VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}" \
        -v /home/b/models/pearl-wan:/workspace/models \
        -v /home/b/models/huggingface:/workspace/.cache/huggingface \
        -v "$ROOT/data:/workspace/pearl_wan/data" \
        -v "$ROOT/exp:/workspace/pearl_wan/exp" \
        -v "$ROOT/archives:/workspace/pearl_wan/archives" \
        --entrypoint "" \
        pearl-wan:latest "$@"
}

status() {
    mkdir -p "$ARCHIVE_ROOT"
    {
        echo "run_id=$RUN_ID"
        echo "root=$ROOT"
        echo "archive_dir=$ARCHIVE_DIR"
        echo "--- git ---"
        git rev-parse HEAD || true
        git status --short || true
        echo "--- gpu ---"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
        echo "--- docker ---"
        docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true
        echo "--- disk ---"
        df -h "$ROOT" /home/b/models || true
    } | tee "$ARCHIVE_ROOT/status_${RUN_ID}.txt"
}

preempt() {
    local dry="${1:-}"
    mkdir -p "$ARCHIVE_DIR"
    local out="$ARCHIVE_DIR/preempt.jsonl"
    : > "$out"
    docker ps --format '{{.ID}} {{.Names}}' | while read -r cid name; do
        [ -n "${cid:-}" ] || continue
        if [[ "$name" == pearl-wan* ]]; then
            continue
        fi
        local inspect
        inspect="$(docker inspect "$cid" 2>/dev/null || true)"
        "$PYTHON_BIN" - "$cid" "$name" "$dry" "$out" <<'PY'
import json, subprocess, sys
cid, name, dry, out = sys.argv[1:]
should_stop = False
try:
    smi = subprocess.check_output([
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True)
    for line in smi.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) >= 2048:
            should_stop = True
except Exception:
    pass
row = {"container": name, "id": cid, "action": "stop" if should_stop else "keep", "dry_run": bool(dry)}
with open(out, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
if should_stop and not dry:
    subprocess.run(["docker", "stop", cid], check=False)
PY
    done
    echo "Preempt log: $out"
}

smoke() {
    "$PYTHON_BIN" tests/test_research_network.py
    docker build -t pearl-wan:latest .
    docker_run python tests/test_research_network.py
    docker_run python scripts/run_literature_matrix.py --profile tiny --run-id "$RUN_ID" --max-experiments 1
}

prepare() {
    docker build -t pearl-wan:latest .
    if [ "${PEARL_SKIP_DATA_PREP:-0}" != "1" ]; then
        docker_run python scripts/prepare_literature_data.py --limit "${PEARL_DATA_LIMIT:-200}"
    else
        echo "Skipping data preparation because PEARL_SKIP_DATA_PREP=1"
    fi
    docker_run bash scripts/prepare_literature_models.sh "${1:-strict-small}" "exp/${RUN_ID}_model_downloads.jsonl"
}

run_matrix() {
    local profile="${1:-paper}"
    local max="${2:-0}"
    mkdir -p "$ARCHIVE_DIR"
    docker build -t pearl-wan:latest .
    local args=(python scripts/run_literature_matrix.py --profile "$profile" --run-id "$RUN_ID")
    if [ "${PEARL_AUTO_ADJUST:-0}" = "1" ]; then
        args+=(--auto-adjust)
    fi
    if [ "${PEARL_INCLUDE_SERVING:-0}" = "1" ]; then
        args+=(--include-serving)
    fi
    if [ "$max" != "0" ]; then
        args+=(--max-experiments "$max")
    fi
    docker_run "${args[@]}"
}

analyze() {
    mkdir -p "$ARCHIVE_DIR"
    docker_run python scripts/analyze_literature_results.py \
        --exp-dir "exp/$RUN_ID" \
        --output-dir "archives/$RUN_ID/analysis" \
        --run-id "$RUN_ID"
    cp -a "exp/$RUN_ID" "$ARCHIVE_DIR/raw" 2>/dev/null || true
    ln -sfn "$ARCHIVE_DIR" "$ARCHIVE_ROOT/latest"
    tar --zstd -cf "$ARCHIVE_ROOT/pearl-wan-${RUN_ID}.tar.zst" -C "$ARCHIVE_ROOT" "$RUN_ID" 2>/dev/null || \
        tar -czf "$ARCHIVE_ROOT/pearl-wan-${RUN_ID}.tar.gz" -C "$ARCHIVE_ROOT" "$RUN_ID"
    echo "Archive: $ARCHIVE_DIR"
}

mail_payload() {
    local payload="$ARCHIVE_DIR/mail_payload.json"
    "$PYTHON_BIN" - "$ARCHIVE_DIR" "$MAIL_TO" "$payload" <<'PY'
import json, os, sys
archive, to, payload = sys.argv[1:]
analysis = os.path.join(archive, "analysis")
attachments = [
    os.path.join(analysis, "EXTENDED_RESULTS.pdf"),
    os.path.join(analysis, "EXTENDED_RESULTS.html"),
    os.path.join(analysis, "summary.csv"),
]
attachments = [p for p in attachments if os.path.exists(p)]
body = f"""Pearl-Wan 文献驱动扩展实验已完成。

归档目录: {archive}

附件包含 PDF/HTML 报告和 summary.csv。报告区分 strict speculative decoding 与 cloud serving simulation，避免混用 speedup 口径。
"""
data = {
    "to": to,
    "subject": "Pearl-Wan 文献驱动扩展实验报告",
    "body": body,
    "attachments": attachments,
}
with open(payload, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    print(payload)
PY
}

mail_payload() {
    local payload="$ARCHIVE_DIR/mail_payload.json"
    "$PYTHON_BIN" - "$ARCHIVE_DIR" "$MAIL_TO" "$payload" <<'PY'
import json, os, sys
archive, to, payload = sys.argv[1:]
analysis = os.path.join(archive, "analysis")
attachments = [
    os.path.join(analysis, "EXTENDED_RESULTS.pdf"),
    os.path.join(analysis, "EXTENDED_RESULTS.html"),
    os.path.join(analysis, "summary.csv"),
]
attachments = [p for p in attachments if os.path.exists(p)]
body = f"""Pearl-Wan single RTX 5090 auto-adjusted experiment has completed.

Archive directory: {archive}

Attachments include the PDF/HTML report and summary.csv. The report separates strict speculative decoding, WAN protocol simulation, and failed/skipped model configurations so the conclusions keep a consistent experimental scope."""
data = {
    "to": to,
    "subject": "Pearl-Wan single-5090 auto-adjusted experiment report",
    "body": body,
    "attachments": attachments,
}
with open(payload, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(payload)
PY
}

case "${1:-status}" in
    status) status ;;
    preempt) preempt "${2:-}" ;;
    smoke) smoke ;;
    prepare) prepare "${2:-strict-small}" ;;
    run) run_matrix "${2:-paper}" "${3:-0}" ;;
    serving) bash scripts/run_vllm_serving_matrix.sh ;;
    analyze) analyze ;;
    mail-payload) mail_payload ;;
    *) echo "Usage: $0 {status|preempt [--dry-run]|smoke|prepare [profile]|run [profile] [max]|serving|analyze|mail-payload}" >&2; exit 2 ;;
esac
