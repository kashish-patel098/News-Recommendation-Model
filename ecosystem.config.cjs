/* ecosystem.config.cjs */
/* PM2 process configuration for the News Recommendation API */

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
      watch: false,
      max_memory_restart: '4G',
      restart_delay: 5000,

      // ── Logging ──────────────────────────────────────────────────────────
      // Using relative paths so it works on Windows without /var/log/
      out_file: './logs/api.log',
      error_file: './logs/api-error.log',
      merge_logs: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
    },
  ],
};
