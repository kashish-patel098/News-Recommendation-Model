#!/usr/bin/env bash
# =============================================================================
# setup_ec2.sh — One-time EC2 environment setup (no Docker, no frontend)
# =============================================================================
# Run this once after SSHing into a fresh Ubuntu 22.04 EC2 instance.
#
# What it does:
#   1. Installs system packages (Python 3.11, Node 20, pip, git)
#   2. Creates a Python virtualenv at /opt/news-rec/venv
#   3. Installs Python dependencies (FastAPI, Qdrant, BGE-m3, boto3 …)
#   4. Installs Node.js dependencies (vector orchestrator only)
#   5. Creates log directories
#   6. Registers systemd services: news-rec-api + news-rec-vector
#   7. Sets up crontab for monthly training
#
# Usage:
#   chmod +x setup_ec2.sh
#   sudo bash setup_ec2.sh
# =============================================================================

set -euo pipefail

APP_DIR="/opt/news-rec"
VENV_DIR="$APP_DIR/venv"
LOG_DIR="/var/log/news-rec"
SERVICE_USER="${SUDO_USER:-ubuntu}"

echo "======================================================"
echo "  News Recommendation Engine — EC2 Setup"
echo "  (API + vector orchestrator only — no frontend)"
echo "======================================================"

# ── 1. System packages ─────────────────────────────────────────────────────────
echo "[1/6] Installing system packages …"
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip build-essential \
    libpq-dev curl git

# Node.js 20 LTS via NodeSource
if ! command -v node &>/dev/null; then
    echo "  Installing Node.js 20 LTS …"
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi
echo "  Node $(node -v) | npm $(npm -v)"

# ── 2. Clone repo (if not already present) ────────────────────────────────────
if [ ! -d "$APP_DIR/.git" ]; then
    echo "[2/6] Cloning repository into $APP_DIR …"
    git clone https://github.com/kashish-patel098/News-Recommendation-Model.git "$APP_DIR"
else
    echo "[2/6] Repo already at $APP_DIR — pulling latest …"
    cd "$APP_DIR" && git pull
fi

# ── 3. Python virtualenv + dependencies ───────────────────────────────────────
echo "[3/6] Creating Python virtualenv at $VENV_DIR …"
python3.11 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip wheel
"$VENV_DIR/bin/pip" install -r "$APP_DIR/app/requirements.txt"

# ── 4. Node.js dependencies (vector orchestrator only) ────────────────────────
echo "[4/6] Installing npm packages for vector orchestrator …"
cd "$APP_DIR"
npm install
# (frontend is NOT deployed on EC2)

# ── 5. Log + data directories ─────────────────────────────────────────────────
echo "[5/6] Creating directories …"
mkdir -p "$LOG_DIR" "$APP_DIR/local_store"
chown -R "$SERVICE_USER":"$SERVICE_USER" "$LOG_DIR" "$APP_DIR"

# ── 6. systemd services ────────────────────────────────────────────────────────
echo "[6/6] Installing systemd services …"

# ── FastAPI backend ────────────────────────────────────────────────────────────
cat > /etc/systemd/system/news-rec-api.service <<EOF
[Unit]
Description=News Recommendation API (FastAPI + uvicorn)
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$VENV_DIR/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=5
StandardOutput=append:$LOG_DIR/api.log
StandardError=append:$LOG_DIR/api.log

[Install]
WantedBy=multi-user.target
EOF

# ── Vector orchestrator ────────────────────────────────────────────────────────
cat > /etc/systemd/system/news-rec-vector.service <<EOF
[Unit]
Description=News Recommendation Vector Orchestrator (Node.js / index.js)
After=network.target news-rec-api.service

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
Environment=EMBED_API_URL=http://localhost:8000
ExecStart=$(command -v node) $APP_DIR/index.js
Restart=always
RestartSec=10
StandardOutput=append:$LOG_DIR/vector.log
StandardError=append:$LOG_DIR/vector.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable news-rec-api news-rec-vector

echo ""
echo "  systemd services registered:"
echo "    news-rec-api     (port 8000)"
echo "    news-rec-vector  (hourly S3→Qdrant job)"

# ── Crontab — monthly training only ───────────────────────────────────────────
CRON_FILE="/tmp/news-rec-cron"
crontab -u "$SERVICE_USER" -l 2>/dev/null > "$CRON_FILE" || true
grep -v "news-rec" "$CRON_FILE" > "${CRON_FILE}.clean" || true
mv "${CRON_FILE}.clean" "$CRON_FILE"

cat >> "$CRON_FILE" <<EOF

# ── News Recommendation Engine ────────────────────────────────────────────────
# Monthly: train ranker on new articles at 03:00 UTC on the 1st
0 3 1 * * cd $APP_DIR && $VENV_DIR/bin/python scripts/monthly_train.py >> $LOG_DIR/train.log 2>&1
EOF
# Note: hourly S3→Qdrant ingestion is handled by news-rec-vector (runs 24/7).

crontab -u "$SERVICE_USER" "$CRON_FILE"
rm -f "$CRON_FILE"

echo ""
echo "======================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Configure environment:"
echo "       cp $APP_DIR/.env.ec2.example $APP_DIR/.env"
echo "       nano $APP_DIR/.env"
echo ""
echo "  2. Run one-shot vector ingestion (catches up current hour):"
echo "       cd $APP_DIR && node index.js --once"
echo ""
echo "  3. Run first-time model training:"
echo "       $VENV_DIR/bin/python scripts/monthly_train.py --all-data"
echo ""
echo "  4. Start services:"
echo "       sudo systemctl start news-rec-api"
echo "       sudo systemctl start news-rec-vector"
echo ""
echo "  5. Check status:"
echo "       sudo systemctl status news-rec-api"
echo "       sudo systemctl status news-rec-vector"
echo ""
echo "  6. Dev mode (on this machine):"
echo "       npm run dev     # starts API + vector in dev/watch mode"
echo "======================================================"
