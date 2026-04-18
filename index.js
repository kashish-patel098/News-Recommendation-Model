/**
 * index.js — Orchestrator
 * ────────────────────────
 * Schedules all recurring jobs:
 *   • Vector job  — runs every hour (reads S3, embeds, upserts Qdrant)
 *
 * Also exports helpers so the process can be started and stopped cleanly.
 *
 * On EC2 (no Docker):
 *   node index.js               # production
 *   node --watch index.js       # dev (restarts on file change)
 *
 * Via npm scripts (root package.json):
 *   npm run vector              # start orchestrator
 *   npm run vector:once         # run one cycle immediately then exit
 */

import dotenv from 'dotenv';
dotenv.config({ quiet: true });

import { VECTOR_JOB_INTERVAL_MS } from './shared/config.js';
import { initVectorService, runVectorJob } from './ec2_vector.js';

// ── Startup ────────────────────────────────────────────────────────────────────

let _timer = null;

async function start() {
  console.log('🟢 [orchestrator] Starting up …');
  console.log(`   Vector job interval: ${VECTOR_JOB_INTERVAL_MS / 1000 / 60} min`);

  // One-time init (ensure Qdrant collection, etc.)
  await initVectorService();

  // Run immediately on startup, then on schedule
  await runVectorJob();

  _timer = setInterval(async () => {
    try {
      await runVectorJob();
    } catch (err) {
      console.error('❌ [orchestrator] Unhandled error in vector job:', err);
    }
  }, VECTOR_JOB_INTERVAL_MS);

  console.log('✅ [orchestrator] Running. Press Ctrl-C to stop.');
}

// ── Graceful shutdown ──────────────────────────────────────────────────────────

function shutdown(signal) {
  console.log(`\n🔴 [orchestrator] Received ${signal} — shutting down …`);
  if (_timer) clearInterval(_timer);
  process.exit(0);
}

process.on('SIGINT',  () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));

// ── CLI entry-point ────────────────────────────────────────────────────────────

const runOnce = process.argv.includes('--once');

if (runOnce) {
  // Used by:  npm run vector:once
  initVectorService()
    .then(() => runVectorJob())
    .then(() => {
      console.log('✅ [orchestrator] One-shot complete. Exiting.');
      process.exit(0);
    })
    .catch(err => {
      console.error('❌ [orchestrator] One-shot failed:', err);
      process.exit(1);
    });
} else {
  start().catch(err => {
    console.error('❌ [orchestrator] Fatal startup error:', err);
    process.exit(1);
  });
}
