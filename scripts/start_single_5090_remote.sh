#!/usr/bin/env bash
set -euo pipefail

ROOT="${PEARL_ROOT:-/home/b/research/pearl-wan}"
cd "$ROOT"

stop_pidfile() {
    local pidfile="$1"
    [ -f "$pidfile" ] || return 0
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
    fi
}

for pidfile in archives/*/driver.pid archives/*/monitor.pid archives/*/analysis_agent.pid archives/*/monitor/http.pid; do
    stop_pidfile "$pidfile"
done
sleep 2

for pidfile in archives/*/driver.pid archives/*/monitor.pid archives/*/analysis_agent.pid archives/*/monitor/http.pid; do
    [ -f "$pidfile" ] || continue
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
    fi
done

cleanup_targets=(
    exp/literature_full_*
    exp/literature_20260510-*
    exp/dry_remote_single5090
    exp/dry_single5090_check
    archives/literature_full_*
    archives/literature_20260510-*
    archives/dry_remote_single5090
    archives/latest
    archives/pearl-wan-literature*.tar.*
)
rm -rf "${cleanup_targets[@]}" 2>/tmp/pearl-wan-cleanup.err || {
    if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        sudo rm -rf "${cleanup_targets[@]}"
    else
        quarantine="archives/invalid_quarantine_$(date +%Y%m%d-%H%M%S)"
        mkdir -p "$quarantine"
        for target in "${cleanup_targets[@]}"; do
            for path in $target; do
                [ -e "$path" ] || continue
                mv "$path" "$quarantine/" 2>/dev/null || {
                    base="$(basename "$path")"
                    parent="$(dirname "$path")"
                    mv "$path" "$parent/invalid_$base" 2>/dev/null || true
                }
            done
        done
        echo "Some root-owned invalid data could not be deleted; quarantined what was movable under $quarantine"
        cat /tmp/pearl-wan-cleanup.err || true
    fi
}

run_info="$(bash scripts/launch_single_5090_auto.sh)"
run_id="$(printf "%s\n" "$run_info" | head -1)"
driver_pid="$(printf "%s\n" "$run_info" | sed -n '2p')"

mkdir -p "archives/$run_id/analysis_agent" "archives/$run_id/monitor"
nohup python3 scripts/live_analyze_results.py \
    --root "$ROOT" \
    --run-id "$run_id" \
    --interval 20 \
    > "archives/$run_id/analysis_agent/analysis_agent.log" 2>&1 < /dev/null &
echo "$!" > "archives/$run_id/analysis_agent.pid"

nohup python3 scripts/monitor_experiment.py \
    --root "$ROOT" \
    --run-id "$run_id" \
    --interval 10 \
    > "archives/$run_id/monitor/monitor_stdout.log" 2>&1 < /dev/null &
echo "$!" > "archives/$run_id/monitor.pid"

(
    cd "archives/$run_id/monitor"
    nohup python3 -m http.server 18088 > http.log 2>&1 < /dev/null &
    echo "$!" > http.pid
)

ln -sfn "$ROOT/archives/$run_id" archives/latest

cat <<EOF
run_id=$run_id
driver_pid=$driver_pid
analysis_pid=$(cat "archives/$run_id/analysis_agent.pid")
monitor_pid=$(cat "archives/$run_id/monitor.pid")
monitor_url=http://127.0.0.1:18088/monitor.html
archive=$ROOT/archives/$run_id
EOF
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
