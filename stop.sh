#!/usr/bin/env bash
# =============================================================================
# stop.sh  — Stop API + vector orchestrator
# =============================================================================

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$APP_DIR/.pids"

stop_pid() {
    local name="$1"
    local pidfile="$PID_DIR/$name.pid"
    if [ -f "$pidfile" ]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Stopping $name (PID $pid) …"
            kill "$pid" 2>/dev/null || true
            for i in 1 2 3; do
                kill -0 "$pid" 2>/dev/null || break
                sleep 1
            done
            kill -9 "$pid" 2>/dev/null || true
        else
            echo "  $name (PID $pid) already stopped."
        fi
        rm -f "$pidfile"
    else
        echo "  No PID file for $name — skipping."
    fi
}

echo "Stopping News Recommendation Engine …"
stop_pid "vector"
stop_pid "api"
echo "✅ Done."
