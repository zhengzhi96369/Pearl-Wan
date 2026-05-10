#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_ID="${1:-single5090_auto_$(date +%Y%m%d-%H%M%S)}"
ARCHIVE_DIR="$ROOT/archives/$RUN_ID"
mkdir -p "$ARCHIVE_DIR"

cat > "$ARCHIVE_DIR/driver.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export PEARL_RUN_ID="$RUN_ID"
export PEARL_GPU_DEVICES="device=1"
export PEARL_CUDA_VISIBLE_DEVICES="0"
export PEARL_DEVICE_EDGE="cuda"
export PEARL_DEVICE_CLOUD="cuda"
export PEARL_INCLUDE_SERVING=0
export PEARL_AUTO_ADJUST=1
export PEARL_SKIP_DATA_PREP=1
export PEARL_MAIL_TO="\${PEARL_MAIL_TO:-2251645084@qq.com}"
export HF_ENDPOINT="\${HF_ENDPOINT:-https://hf-mirror.com}"
bash scripts/pearl_lab.sh preempt
bash scripts/pearl_lab.sh prepare single5090
bash scripts/pearl_lab.sh run single5090
bash scripts/pearl_lab.sh analyze
bash scripts/pearl_lab.sh mail-payload
EOF
chmod +x "$ARCHIVE_DIR/driver.sh"

nohup bash "$ARCHIVE_DIR/driver.sh" > "$ARCHIVE_DIR/driver.log" 2>&1 < /dev/null &
echo "$!" > "$ARCHIVE_DIR/driver.pid"

echo "$RUN_ID"
cat "$ARCHIVE_DIR/driver.pid"
