/**
 * shared/utils.js
 * ────────────────
 * Utility helpers shared across all services.
 */

/**
 * Returns the current IST date-time parts as zero-padded strings.
 *
 * @returns {{ yyyy: string, mm: string, dd: string, hh: string }}
 *
 * Example:
 *   { yyyy: '2026', mm: '04', dd: '18', hh: '09' }
 */
export function getISTDateParts() {
  const now = new Date();

  // IST = UTC + 5:30
  const istOffsetMs = 5.5 * 60 * 60 * 1000;
  const ist = new Date(now.getTime() + istOffsetMs);

  const yyyy = String(ist.getUTCFullYear());
  const mm   = String(ist.getUTCMonth() + 1).padStart(2, '0');
  const dd   = String(ist.getUTCDate()).padStart(2, '0');
  const hh   = String(ist.getUTCHours()).padStart(2, '0');

  return { yyyy, mm, dd, hh };
}

/**
 * Sleep for `ms` milliseconds.
 * @param {number} ms
 */
export function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Clamp a number between min and max (inclusive).
 */
export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
