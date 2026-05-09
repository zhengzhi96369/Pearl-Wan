#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export PEARL_WAN_MODEL_DIR="${PEARL_WAN_MODEL_DIR:-/workspace/models}"
export PEARL_WAN_DATA_DIR="${PEARL_WAN_DATA_DIR:-/workspace/pearl_wan/data}"

mkdir -p "$HF_HOME" "$PEARL_WAN_MODEL_DIR" "$PEARL_WAN_DATA_DIR"

download_model() {
    local repo_id="$1"
    local local_name="$2"
    local local_dir="$PEARL_WAN_MODEL_DIR/$local_name"

    if [ -f "$local_dir/config.json" ]; then
        echo "[assets] Model already present: $local_dir"
        return
    fi

    echo "[assets] Downloading $repo_id to $local_dir via $HF_ENDPOINT"
    if hf download "$repo_id" --local-dir "$local_dir"; then
        return
    fi

    echo "[assets] hf download failed for $repo_id; trying ModelScope fallback"
    python - "$repo_id" "$local_dir" <<'PY'
import os
import sys
from modelscope import snapshot_download

repo_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(repo_id, local_dir=local_dir)
if not os.path.exists(os.path.join(local_dir, "config.json")):
    raise SystemExit(f"ModelScope download did not create config.json in {local_dir}")
PY
}

download_model "Qwen/Qwen2.5-0.5B-Instruct" "qwen2.5-0.5b-instruct"
download_model "Qwen/Qwen2.5-1.5B-Instruct" "qwen2.5-1.5b-instruct"
download_model "Qwen/Qwen2.5-7B-Instruct" "qwen2.5-7b-instruct"

echo "[assets] Preparing benchmark data"
python scripts/prepare_data.py --output-dir "$PEARL_WAN_DATA_DIR" --limit 8
