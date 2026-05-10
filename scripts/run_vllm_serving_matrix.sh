#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_ID="${PEARL_RUN_ID:-serving_$(date +%Y%m%d-%H%M%S)}"
PORT="${VLLM_PORT:-18000}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
CONTAINER="${VLLM_CONTAINER:-pearl-wan-vllm}"
OUT_DIR="$ROOT/exp/$RUN_ID/serving_services"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME_HOST="${HF_HOME_HOST:-/home/b/models/huggingface}"
MODEL_DIR_HOST="${MODEL_DIR_HOST:-/home/b/models/pearl-wan}"
mkdir -p "$OUT_DIR"

models_json() {
    python3 - "$ROOT/configs/literature_matrix.json" <<'PY'
import json, sys
cfg=json.load(open(sys.argv[1], encoding="utf-8"))
for row in cfg["serving_cloud_models"]:
    print(json.dumps(row, ensure_ascii=False))
PY
}

stop_service() {
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}

wait_ready() {
    local url="http://127.0.0.1:${PORT}/v1/models"
    for _ in $(seq 1 120); do
        if curl -fsS "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    return 1
}

run_eval_container() {
    local model="$1"
    local task="$2"
    local concurrency="$3"
    local exp_name="$RUN_ID/serving_${model}_${task}_c${concurrency}"
    docker run --rm --network host \
        -e VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1" \
        -v "$ROOT/exp:/workspace/pearl_wan/exp" \
        --entrypoint "" pearl-wan:latest \
        python benchmark/eval_vllm_serving.py \
            --base_url "http://127.0.0.1:${PORT}/v1" \
            --model "$model" \
            --task "$task" \
            --concurrency "$concurrency" \
            --num_requests "$(( concurrency > 4 ? concurrency : 4 ))" \
            --max_tokens "${SERVING_MAX_TOKENS:-128}" \
            --exp_name "$exp_name"
}

trap stop_service EXIT

models_json | while read -r model_row; do
    name="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["name"])' "$model_row")"
    repo="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["repo_id"])' "$model_row")"
    quant="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("quantization",""))' "$model_row")"
    tasks="$(python3 -c 'import json,sys; print(" ".join(json.loads(sys.argv[1]).get("tasks",["speed"])))' "$model_row")"
    service_dir="$OUT_DIR/$name"
    mkdir -p "$service_dir"
    stop_service
    extra_args=()
    if [[ "$quant" == "awq" ]]; then
        extra_args+=(--quantization awq)
    fi
    echo ">>> starting $name ($repo)"
    if ! docker run -d --name "$CONTAINER" --gpus all --network host \
        -e HF_ENDPOINT="$HF_ENDPOINT" \
        -v "$HF_HOME_HOST:/root/.cache/huggingface" \
        -v "$MODEL_DIR_HOST:/models" \
        "$IMAGE" \
        --model "$repo" \
        --served-model-name "$name" \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port "$PORT" \
        --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
        "${extra_args[@]}" > "$service_dir/container_id.txt"; then
        echo '{"status":"launch_failed"}' > "$service_dir/status.json"
        continue
    fi
    if ! wait_ready; then
        docker logs "$CONTAINER" > "$service_dir/vllm.log" 2>&1 || true
        echo '{"status":"health_failed"}' > "$service_dir/status.json"
        stop_service
        continue
    fi
    curl -fsS "http://127.0.0.1:${PORT}/v1/models" > "$service_dir/models.json" || true
    for task in $tasks; do
        case "$task" in
            humaneval|gsm8k|math500|mtbench|speed) ;;
            *) task="speed" ;;
        esac
        for concurrency in ${SERVING_CONCURRENCY:-1 2 4 8 16 32}; do
            echo ">>> serving eval model=$name task=$task concurrency=$concurrency"
            run_eval_container "$name" "$task" "$concurrency" > "$service_dir/${task}_c${concurrency}.log" 2>&1 || true
        done
    done
    docker logs "$CONTAINER" > "$service_dir/vllm.log" 2>&1 || true
    echo '{"status":"ok"}' > "$service_dir/status.json"
    stop_service
done
