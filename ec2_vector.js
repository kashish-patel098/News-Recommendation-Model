/**
 * ec2_vector.js — Vectorise Economic Times articles and upsert them into Qdrant.
 *
 * Scheduling is handled entirely by index.js.
 */
import { QdrantClient } from '@qdrant/js-client-rest';
import { S3Client, GetObjectCommand } from '@aws-sdk/client-s3';

import { QDRANT_URL, COLLECTION, VECTOR_SIZE, S3_BUCKET, S3_REGION } from './shared/config.js';
import { getISTDateParts } from './shared/utils.js';
import { embed, isValidVector } from './shared/embedder.js';
import dotenv from 'dotenv';
dotenv.config({ quiet: true });

// ── CLIENTS ───────────────────────────────────────────────
const s3 = new S3Client({ region: S3_REGION });
const qdrant = new QdrantClient({
  url: QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
  checkCompatibility: false,
});

// ── QDRANT SETUP ──────────────────────────────────────────
async function ensureCollection() {
  const { collections } = await qdrant.getCollections();
  if (!collections.some(c => c.name === COLLECTION)) {
    await qdrant.createCollection(COLLECTION, {
      vectors: {
        title:       { size: VECTOR_SIZE, distance: 'Cosine' },
        description: { size: VECTOR_SIZE, distance: 'Cosine' },
        tags:        { size: VECTOR_SIZE, distance: 'Cosine' },
      },
    });
    console.log(`📂 Created Qdrant collection "${COLLECTION}"`);
  }
}

// ── S3 ────────────────────────────────────────────────────
async function readJsonFromS3(key) {
  const res = await s3.send(new GetObjectCommand({ Bucket: S3_BUCKET, Key: key }));
  const body = await res.Body.transformToString();
  return JSON.parse(body);
}

// ── PROCESS ONE S3 FILE ───────────────────────────────────
async function processFile(key) {
  const json = await readJsonFromS3(key);

  if (!Array.isArray(json.articles)) {
    console.log('— No articles array in', key);
    return;
  }

  // ⚠️  points MUST be local to each run — never module-level!
  const points = [];

  for (const article of json.articles) {
    try {
      const fd = article.formatted_data || {};
      const title       = fd.title       || article.original_title || '';
      const description = fd.descriptive_paragraph || '';
      const tagsText    = (fd.tags || []).join(' ');

      // Embed all three fields in parallel
      const [tVec, dVec, gVec] = await Promise.all([
        embed(title),
        embed(description),
        embed(tagsText),
      ]);

      if (!isValidVector(tVec) || !isValidVector(dVec) || !isValidVector(gVec)) {
        console.error('❌ Invalid vector, skipping article:', article.id);
        continue;
      }

      points.push({
        id:      article.id || `${key}-${points.length}`,
        vectors: { title: tVec, description: dVec, tags: gVec },
        payload: {
          published_time: article.published_time ?? null,
          economy_impact: fd.economyImpact?.score ?? null,
          tags:           fd.tags || [],
        },
      });
    } catch (e) {
      console.error('❌ Article error, skipped:', article.id, e.message);
    }
  }

  if (!points.length) {
    console.log('— No valid points to upsert for', key);
    return;
  }

  console.log(`📤 Upserting ${points.length} points from ${key}`);
  await qdrant.upsert(COLLECTION, { points, wait: true });
  console.log(`✔  Stored ${points.length} articles from ${key}`);
}

// ── JOB ───────────────────────────────────────────────────
async function job() {
  const { yyyy, mm, dd, hh } = getISTDateParts();
  const key = `curated/news/economic-times/${yyyy}/${mm}/${dd}/et_${hh}.json`;

  console.log(`\n🕒 [vector] IST ${yyyy}-${mm}-${dd} ${hh}:xx  →  s3://${S3_BUCKET}/${key}`);

  try {
    await processFile(key);
    console.log('✅ [vector] SUCCESS');
  } catch (err) {
    console.error('❌ [vector] ERROR:', err.name, '—', err.message);
  }
}

// ── EXPORTS ────────────────────────────────────────────────
/** One-time startup: ensure Qdrant collection exists. */
export async function initVectorService() {
  await ensureCollection();
  console.log('🚀 News vector service ready');
}

/** Run one vectorisation cycle. Called by the orchestrator in index.js. */
export async function runVectorJob() {
  await job();
}
