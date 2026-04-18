#!/usr/bin/env bash
# =============================================================================
# start.sh  — Start API + vector orchestrator on EC2 (no frontend)
#
#   1. FastAPI backend     (Python / uvicorn)      → port 8000
#   2. Vector orchestrator (Node.js / index.js)    → hourly S3→Qdrant job
#
# Usage:
#   bash start.sh           # production
#   bash start.sh --dev     # dev mode (uvicorn --reload + node --watch)
# =============================================================================

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$APP_DIR/venv"
LOG_DIR="/var/log/news-rec"
PID_DIR="$APP_DIR/.pids"

DEV_MODE=false
[[ "${1:-}" == "--dev" ]] && DEV_MODE=true

mkdir -p "$LOG_DIR" "$PID_DIR"

# ── Load .env ──────────────────────────────────────────────────────────────────
if [ -f "$APP_DIR/.env" ]; then
    set -a; source "$APP_DIR/.env"; set +a
fi

BACKEND_PORT="${PORT:-8000}"
NODE_BIN="$(command -v node)"

echo "======================================================"
echo "  News Recommendation Engine (EC2 — no frontend)"
echo "  Mode   : $([ "$DEV_MODE" = true ] && echo 'DEVELOPMENT' || echo 'PRODUCTION')"
echo "  API    : http://0.0.0.0:$BACKEND_PORT"
echo "  Vector : hourly S3→Qdrant job (orchestrator)"
echo "======================================================"

# Stop any running instances first
bash "$APP_DIR/stop.sh" 2>/dev/null || true
sleep 1

# ── Resolve Python ─────────────────────────────────────────────────────────────
PYTHON="$VENV_DIR/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
    echo "⚠  venv not found at $VENV_DIR — using system $PYTHON"
fi

# ── 1. FastAPI Backend ─────────────────────────────────────────────────────────
echo ""
echo "[1/2] Starting FastAPI backend …"
cd "$APP_DIR"

UVICORN_EXTRA="--workers 1"
[ "$DEV_MODE" = true ] && UVICORN_EXTRA="--reload"

nohup "$PYTHON" -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "$BACKEND_PORT" \
    $UVICORN_EXTRA \
    >> "$LOG_DIR/api.log" 2>&1 &

echo "$!" > "$PID_DIR/api.pid"
echo "  ✓ PID $(cat "$PID_DIR/api.pid") → $LOG_DIR/api.log"

# ── 2. Vector Orchestrator (Node.js) ──────────────────────────────────────────
echo ""
echo "[2/2] Starting vector orchestrator (index.js) …"
cd "$APP_DIR"

export EMBED_API_URL="http://localhost:$BACKEND_PORT"

if [ "$DEV_MODE" = true ]; then
    nohup "$NODE_BIN" --watch index.js >> "$LOG_DIR/vector.log" 2>&1 &
else
    nohup "$NODE_BIN" index.js >> "$LOG_DIR/vector.log" 2>&1 &
fi

echo "$!" > "$PID_DIR/vector.pid"
echo "  ✓ PID $(cat "$PID_DIR/vector.pid") → $LOG_DIR/vector.log"

# ── Wait for API readiness ─────────────────────────────────────────────────────
echo ""
echo "Waiting for API to become ready …"
for i in $(seq 1 40); do
    if curl -sf "http://localhost:$BACKEND_PORT/health" > /dev/null 2>&1; then
        echo "  ✅ API is ready."
        break
    fi
    printf "."
    sleep 3
done
echo ""

echo "======================================================"
echo "  All services started."
echo "  API  → http://localhost:$BACKEND_PORT"
echo "  Docs → http://localhost:$BACKEND_PORT/docs"
echo ""
echo "  Logs:"
echo "    tail -f $LOG_DIR/api.log"
echo "    tail -f $LOG_DIR/vector.log"
echo ""
echo "  Stop:  bash stop.sh  |  npm run stop"
echo "======================================================"
