#!/usr/bin/env bash
# stop.sh — Stop the API via PM2
set -euo pipefail
echo "Stopping news-rec-api …"
pm2 stop news-rec-api 2>/dev/null || echo "  (not running)"
echo "✅ Done.  Run 'pm2 delete news-rec-api' to remove it from PM2 list."
