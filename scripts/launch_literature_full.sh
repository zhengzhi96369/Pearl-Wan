#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_ID="${1:-literature_full_$(date +%Y%m%d-%H%M%S)}"
ARCHIVE_DIR="$ROOT/archives/$RUN_ID"
mkdir -p "$ARCHIVE_DIR"

cat > "$ARCHIVE_DIR/driver.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export PEARL_RUN_ID="$RUN_ID"
export PEARL_INCLUDE_SERVING=0
bash scripts/pearl_lab.sh preempt
bash scripts/pearl_lab.sh prepare strict-paper
bash scripts/pearl_lab.sh run paper
bash scripts/pearl_lab.sh prepare serving-paper
bash scripts/pearl_lab.sh serving
bash scripts/pearl_lab.sh analyze
bash scripts/pearl_lab.sh mail-payload
EOF
chmod +x "$ARCHIVE_DIR/driver.sh"

nohup bash "$ARCHIVE_DIR/driver.sh" > "$ARCHIVE_DIR/driver.log" 2>&1 < /dev/null &
echo "$!" > "$ARCHIVE_DIR/driver.pid"

echo "$RUN_ID"
cat "$ARCHIVE_DIR/driver.pid"
