// ecosystem.config.cjs
// PM2 process configuration for the News Recommendation API
//
// Commands:
//   pm2 start ecosystem.config.cjs          # start
//   pm2 restart news-rec-api                # restart
//   pm2 stop news-rec-api                   # stop
//   pm2 logs news-rec-api                   # tail logs
//   pm2 save && pm2 startup                 # survive reboots

module.exports = {
  apps: [
    {
      name: 'news-rec-api',
      script: 'venv/bin/python',
      args: '-m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1',
      cwd: '/opt/news-rec',
      interpreter: 'none',           // run script directly, not via node
      env_file: '/opt/news-rec/.env',

      // ── Restart policy ───────────────────────────────────────────────────
      autorestart: true,
      watch: false,                  // don't watch files in production
      max_memory_restart: '4G',      // restart if BGE-m3 causes memory spike
      restart_delay: 5000,           // wait 5s before restarting

      // ── Logging ──────────────────────────────────────────────────────────
      out_file: '/var/log/news-rec/api.log',
      error_file: '/var/log/news-rec/api-error.log',
      merge_logs: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      log_type: 'json',
    },
  ],
};
