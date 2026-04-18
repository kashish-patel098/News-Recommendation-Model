#!/usr/bin/env bash
# =============================================================================
# setup_ec2.sh — One-time EC2 setup for the News Recommendation API
# =============================================================================
# Uses PM2 to keep the FastAPI process running forever.
#
# Steps:
#   1. Installs Python 3.11, Node 20, pip
#   2. Creates Python virtualenv + installs dependencies
#   3. Installs PM2 globally
#   4. Creates log directories
#   5. Starts the API via PM2 + configures PM2 to survive reboots
#   6. Sets up crontab for monthly model training
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
echo "  News Recommendation API — EC2 Setup (PM2)"
echo "======================================================"

# ── 1. System packages ─────────────────────────────────────────────────────────
echo "[1/6] Installing system packages …"
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip build-essential libpq-dev curl git

# Node.js 20 LTS (needed for PM2)
if ! command -v node &>/dev/null; then
    echo "  Installing Node.js 20 LTS …"
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi
echo "  Node $(node -v) | npm $(npm -v)"

# ── 2. Clone / pull repo ──────────────────────────────────────────────────────
if [ ! -d "$APP_DIR/.git" ]; then
    echo "[2/6] Cloning repository …"
    git clone https://github.com/kashish-patel098/News-Recommendation-Model.git "$APP_DIR"
else
    echo "[2/6] Repo already present — pulling latest …"
    cd "$APP_DIR" && git pull
fi

# ── 3. Python virtualenv + dependencies ───────────────────────────────────────
echo "[3/6] Setting up Python virtualenv …"
python3.11 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip wheel
"$VENV_DIR/bin/pip" install -r "$APP_DIR/app/requirements.txt"

# ── 4. Install PM2 globally ────────────────────────────────────────────────────
echo "[4/6] Installing PM2 …"
npm install -g pm2
pm2 --version

# ── 5. Log + data directories ─────────────────────────────────────────────────
echo "[5/6] Creating directories …"
mkdir -p "$LOG_DIR" "$APP_DIR/local_store"
chown -R "$SERVICE_USER":"$SERVICE_USER" "$LOG_DIR" "$APP_DIR"

# ── 6. Start API via PM2 + configure startup ──────────────────────────────────
echo "[6/6] Starting API with PM2 …"

# Run as the service user (not root)
sudo -u "$SERVICE_USER" bash -c "
    cd $APP_DIR
    pm2 start ecosystem.config.cjs
    pm2 save
"

# Register PM2 startup hook so it survives reboots
PM2_STARTUP=$(sudo -u "$SERVICE_USER" pm2 startup systemd -u "$SERVICE_USER" --hp "/home/$SERVICE_USER" | tail -1)
echo "  Running startup hook: $PM2_STARTUP"
eval "$PM2_STARTUP"

# ── Crontab — monthly training ────────────────────────────────────────────────
CRON_FILE="/tmp/news-rec-cron"
crontab -u "$SERVICE_USER" -l 2>/dev/null > "$CRON_FILE" || true
grep -v "news-rec" "$CRON_FILE" > "${CRON_FILE}.clean" || true
mv "${CRON_FILE}.clean" "$CRON_FILE"

cat >> "$CRON_FILE" <<EOF

# Monthly: retrain NewsRanker on the 1st at 03:00 UTC
0 3 1 * * cd $APP_DIR && $VENV_DIR/bin/python scripts/monthly_train.py >> $LOG_DIR/train.log 2>&1
EOF

crontab -u "$SERVICE_USER" "$CRON_FILE"
rm -f "$CRON_FILE"

echo ""
echo "======================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Configure .env:"
echo "       cp $APP_DIR/.env.ec2.example $APP_DIR/.env"
echo "       nano $APP_DIR/.env"
echo "       pm2 restart news-rec-api   # apply new .env"
echo ""
echo "  2. Check status:"
echo "       pm2 status"
echo "       pm2 logs news-rec-api"
echo "       curl http://localhost:8000/health"
echo ""
echo "  3. Common PM2 commands:"
echo "       pm2 restart news-rec-api   # restart after code change"
echo "       pm2 reload news-rec-api    # zero-downtime reload"
echo "       pm2 stop news-rec-api      # stop"
echo "       pm2 delete news-rec-api    # remove from PM2"
echo "======================================================"
