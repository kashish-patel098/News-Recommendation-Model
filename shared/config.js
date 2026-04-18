/**
 * shared/config.js
 * ─────────────────
 * Single source of truth for all service configuration.
 * Values are read from .env (loaded by the entry-point).
 *
 * AWS credentials are NOT configured here.
 * On EC2, the IAM Instance Role provides automatic access to S3,
 * Athena, and Glue — no keys required.
 */

export const QDRANT_URL   = process.env.QDRANT_URL   || 'http://localhost:6333';
export const COLLECTION   = process.env.COLLECTION   || 'news_embeddings';
export const VECTOR_SIZE  = parseInt(process.env.VECTOR_SIZE || '386', 10);

export const S3_BUCKET    = process.env.S3_BUCKET    || '';
export const S3_REGION    = process.env.S3_REGION    || process.env.AWS_REGION || 'us-east-1';

/**
 * How often (ms) the vector job fires.
 * Default: 3_600_000 ms = 1 hour, matching the hourly et_{hh}.json S3 files.
 * The job reads the current IST hour's file and upserts it to Qdrant.
 * index.js keeps the process alive and calls runVectorJob() on this interval.
 */
export const VECTOR_JOB_INTERVAL_MS = parseInt(
  process.env.VECTOR_JOB_INTERVAL_MS || String(60 * 60 * 1000), 10
);
