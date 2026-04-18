#!/usr/bin/env bash
# =============================================================================
# start.sh  — Start the News Recommendation API via PM2
# =============================================================================
# Usage:
#   bash start.sh           # production (pm2 start)
#   bash start.sh --dev     # dev mode   (uvicorn --reload, no PM2)
# =============================================================================

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$APP_DIR/venv"
LOG_DIR="/var/log/news-rec"

DEV_MODE=false
[[ "${1:-}" == "--dev" ]] && DEV_MODE=true

mkdir -p "$LOG_DIR"

if [ -f "$APP_DIR/.env" ]; then
    set -a; source "$APP_DIR/.env"; set +a
fi

BACKEND_PORT="${PORT:-8000}"

if [ "$DEV_MODE" = true ]; then
    # ── Dev mode: run uvicorn directly with --reload ───────────────────────
    echo "Starting API in DEV mode (uvicorn --reload) on port $BACKEND_PORT …"
    PYTHON="$VENV_DIR/bin/python"
    [ ! -f "$PYTHON" ] && PYTHON="$(command -v python3)"
    exec "$PYTHON" -m uvicorn app.main:app \
        --host 0.0.0.0 \
        --port "$BACKEND_PORT" \
        --reload
else
    # ── Production: run via PM2 ────────────────────────────────────────────
    echo "Starting API via PM2 …"
    pm2 start "$APP_DIR/ecosystem.config.cjs"

    echo ""
    echo "  API  → http://localhost:$BACKEND_PORT"
    echo "  Docs → http://localhost:$BACKEND_PORT/docs"
    echo ""
    echo "  pm2 status              — show process table"
    echo "  pm2 logs news-rec-api   — tail logs"
    echo "  pm2 stop news-rec-api   — stop"
    echo "  pm2 restart news-rec-api — restart"
fi
