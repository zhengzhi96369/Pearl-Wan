#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${PEARL_WAN_MODEL_DIR:-/workspace/models}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PROFILE="${1:-strict-small}"
MANIFEST="${2:-$ROOT/exp/model_download_manifest_$(date +%Y%m%d-%H%M%S).jsonl}"

mkdir -p "$MODEL_DIR" "$HF_HOME" "$(dirname "$MANIFEST")"

download_hf() {
    local repo_id="$1"
    local local_name="$2"
    local local_dir="$MODEL_DIR/$local_name"
    local started ended status source
    started="$(date -Is)"
    status="ok"
    source="hf-mirror"
    if [ -f "$local_dir/config.json" ] && find "$local_dir" -maxdepth 2 \( -name '*.safetensors' -o -name 'pytorch_model*.bin' -o -name '*.gguf' \) | grep -q .; then
        ended="$(date -Is)"
        python - "$MANIFEST" "$repo_id" "$local_name" "$local_dir" "local-cache" "ok" "$started" "$ended" <<'PY'
import json, os, sys
manifest, repo_id, local_name, local_dir, source, status, started, ended = sys.argv[1:]
size = 0
for root, _, files in os.walk(local_dir):
    for name in files:
        try:
            size += os.path.getsize(os.path.join(root, name))
        except OSError:
            pass
row = {"repo_id": repo_id, "local_name": local_name, "local_dir": local_dir, "source": source, "status": status, "started_at": started, "ended_at": ended, "bytes": size}
with open(manifest, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
        echo ">>> using local cache $repo_id -> $local_dir"
        return 0
    fi
    echo ">>> downloading $repo_id -> $local_dir"
    if ! HF_ENDPOINT="$HF_ENDPOINT" hf download "$repo_id" \
        --local-dir "$local_dir" \
        --include "*.json" "*.txt" "*.model" "*.safetensors" "pytorch_model*.bin" "merges.txt" "vocab.json"; then
        status="hf_failed"
        source="modelscope"
        echo ">>> hf download failed for $repo_id; trying ModelScope fallback"
        if ! modelscope download --model "$repo_id" --local_dir "$local_dir"; then
            status="failed"
        else
            status="ok"
        fi
    fi
    ended="$(date -Is)"
    python - "$MANIFEST" "$repo_id" "$local_name" "$local_dir" "$source" "$status" "$started" "$ended" <<'PY'
import json, os, sys
manifest, repo_id, local_name, local_dir, source, status, started, ended = sys.argv[1:]
size = 0
for root, _, files in os.walk(local_dir):
    for name in files:
        try:
            size += os.path.getsize(os.path.join(root, name))
        except OSError:
            pass
row = {
    "repo_id": repo_id,
    "local_name": local_name,
    "local_dir": local_dir,
    "source": source,
    "status": status,
    "started_at": started,
    "ended_at": ended,
    "bytes": size,
}
with open(manifest, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

case "$PROFILE" in
    strict-small)
        MODELS=(
            "Qwen/Qwen2.5-0.5B-Instruct|qwen2.5-0.5b-instruct"
            "Qwen/Qwen2.5-1.5B-Instruct|qwen2.5-1.5b-instruct"
            "Qwen/Qwen2.5-7B-Instruct|qwen2.5-7b-instruct"
        )
        ;;
    strict-paper)
        MODELS=(
            "Qwen/Qwen2.5-0.5B-Instruct|qwen2.5-0.5b-instruct"
            "Qwen/Qwen2.5-1.5B-Instruct|qwen2.5-1.5b-instruct"
            "Qwen/Qwen2.5-7B-Instruct|qwen2.5-7b-instruct"
            "Qwen/Qwen2.5-Coder-1.5B-Instruct|qwen2.5-coder-1.5b-instruct"
            "Qwen/Qwen2.5-Coder-7B-Instruct|qwen2.5-coder-7b-instruct"
        )
        ;;
    single5090)
        MODELS=(
            "Qwen/Qwen2.5-0.5B-Instruct|qwen2.5-0.5b-instruct"
            "Qwen/Qwen2.5-1.5B-Instruct|qwen2.5-1.5b-instruct"
            "Qwen/Qwen2.5-7B-Instruct|qwen2.5-7b-instruct"
            "Qwen/Qwen2.5-Coder-1.5B-Instruct|qwen2.5-coder-1.5b-instruct"
            "Qwen/Qwen2.5-Coder-7B-Instruct|qwen2.5-coder-7b-instruct"
            "facebook/opt-125m|opt-125m"
            "facebook/opt-1.3b|opt-1.3b"
            "facebook/opt-6.7b|opt-6.7b"
        )
        ;;
    serving-paper)
        MODELS=(
            "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ|qwen2.5-coder-14b-instruct-awq"
            "Qwen/Qwen2.5-32B-Instruct|qwen2.5-32b-instruct"
            "Qwen/Qwen2.5-72B-Instruct|qwen2.5-72b-instruct"
            "Qwen/Qwen3-30B-A3B|qwen3-30b-a3b"
            "mistralai/Mixtral-8x7B-Instruct-v0.1|mixtral-8x7b"
            "meta-llama/Llama-3.1-70B-Instruct|llama-3.1-70b"
        )
        ;;
    all)
        "$0" strict-paper "$MANIFEST"
        "$0" serving-paper "$MANIFEST"
        exit 0
        ;;
    *)
        echo "Unknown profile: $PROFILE" >&2
        exit 2
        ;;
esac

for item in "${MODELS[@]}"; do
    IFS='|' read -r repo_id local_name <<< "$item"
    download_hf "$repo_id" "$local_name"
done

echo "Model download manifest: $MANIFEST"
