/**
 * shared/embedder.js
 * ───────────────────
 * Lightweight embedding client.
 *
 * Strategy (in priority order):
 *   1. If EMBED_API_URL is set → call your own FastAPI /embed endpoint
 *      (the Python backend already loads BGE-m3; reuse it to avoid loading
 *       the model twice on the same machine).
 *   2. If @xenova/transformers is installed → run BGE-m3 in-process (ONNX).
 *   3. Throws a clear error so the missing dependency is obvious.
 *
 * Environment variables:
 *   EMBED_API_URL   e.g. http://localhost:8000   (use the FastAPI backend)
 *   VECTOR_SIZE     386  (must match Qdrant collection)
 */

import { VECTOR_SIZE } from './config.js';

const EMBED_API_URL = process.env.EMBED_API_URL || 'http://localhost:8000';

// ── Strategy 1: call the Python /embed endpoint ───────────────────────────────

async function embedViaAPI(text) {
  const url = `${EMBED_API_URL}/api/v1/embed`;
  const res  = await fetch(url, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ text }),
  });

  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText);
    throw new Error(`Embed API error ${res.status}: ${msg}`);
  }

  const data = await res.json();
  // Expected response: { vector: [0.1, 0.2, ... ] }
  if (!Array.isArray(data.vector)) {
    throw new Error(`Embed API returned unexpected shape: ${JSON.stringify(data)}`);
  }
  return data.vector;
}

// ── Strategy 2: in-process ONNX via @xenova/transformers ─────────────────────

let _xenovaPipeline = null;

async function embedViaXenova(text) {
  if (!_xenovaPipeline) {
    const { pipeline } = await import('@xenova/transformers');
    _xenovaPipeline = await pipeline('feature-extraction', 'Xenova/bge-m3', {
      quantized: false,
    });
  }
  const output = await _xenovaPipeline(text, { pooling: 'cls', normalize: true });
  return Array.from(output.data);
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Embed a single text string → number[] of length VECTOR_SIZE.
 *
 * @param {string} text
 * @returns {Promise<number[]>}
 */
export async function embed(text) {
  // Sanitise input
  const t = (text || '').trim().slice(0, 4000);

  // Try the FastAPI endpoint first (preferred when API is running)
  try {
    return await embedViaAPI(t);
  } catch (apiErr) {
    // If API is not reachable, try in-process Xenova
    let xenovaErr;
    try {
      return await embedViaXenova(t);
    } catch (e) {
      xenovaErr = e;
    }

    throw new Error(
      `embed() failed via both strategies.\n` +
      `  API (${EMBED_API_URL}): ${apiErr.message}\n` +
      `  Xenova: ${xenovaErr.message}`
    );
  }
}

/**
 * Validate that a vector is a non-empty array of the expected size.
 *
 * @param {any} vec
 * @returns {boolean}
 */
export function isValidVector(vec) {
  return (
    Array.isArray(vec) &&
    vec.length === VECTOR_SIZE &&
    vec.every(v => typeof v === 'number' && isFinite(v))
  );
}
