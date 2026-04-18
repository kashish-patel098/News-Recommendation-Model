"""
scripts/ingest_from_athena.py
──────────────────────────────
Incremental ingestion of news articles from AWS Athena (Iceberg table)
into PostgreSQL + Qdrant.

What it does
────────────
  1. Reads the highest published_time_unix already in PostgreSQL
  2. Queries Athena for all articles published AFTER that timestamp
  3. Skips articles already present in BOTH stores (idempotent)
  4. Embeds title / description / tags separately → 386-dim named vectors
  5. Upserts to Qdrant (named-vector collection)
  6. Writes full article to PostgreSQL
  7. Saves the watermark timestamp so the next run only pulls new data

Environment variables
─────────────────────
  See .env.ec2.example for the complete list.

Usage (on EC2)
──────────────
  # Activate venv
  source /opt/news-rec/venv/bin/activate

  # Run ingestion (pulls only records newer than the stored watermark)
  python scripts/ingest_from_athena.py

  # Override watermark to re-process last N days
  python scripts/ingest_from_athena.py --since-days 7

  # Dry run
  python scripts/ingest_from_athena.py --dry-run
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.athena_client import AthenaClient
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.utils.text_utils import parse_tags, strip_html
from local_store.news_store import NewsStore

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ingest_from_athena")

# ── Config ────────────────────────────────────────────────────────────────────
COLLECTION_NAME   = os.getenv("COLLECTION_NAME",    "news_embeddings")
QDRANT_HOST       = os.getenv("QDRANT_HOST",        "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT",    "6333"))
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")     or None
MODEL_NAME        = os.getenv("MODEL_NAME",          "BAAI/bge-m3")
DATABASE_URL      = os.getenv("DATABASE_URL",        "postgresql://postgres:postgres@localhost:5432/newsdb")
BATCH_SIZE        = int(os.getenv("INGEST_BATCH_SIZE", "16"))

# Path to persist the last-ingested timestamp watermark
WATERMARK_FILE    = ROOT / "local_store" / "athena_watermark.txt"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_str(val) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def _safe_int(val):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _row_to_article(row: dict) -> dict:
    return {
        "id":                     _safe_str(row.get("id")),
        "published_time":         _safe_str(row.get("published_time")),
        "published_time_unix":    _safe_int(row.get("published_time_unix")),
        "title":                  _safe_str(row.get("title")),
        "introductory_paragraph": _safe_str(row.get("introductory_paragraph")),
        "descriptive_paragraph":  _safe_str(row.get("descriptive_paragraph")),
        "historical_context":     _safe_str(row.get("historical_context")),
        "economyimpact":          _safe_str(row.get("economyimpact")),
        "impact_matrix":          _safe_str(row.get("impact_matrix")),
        "perception_lines":       _safe_str(row.get("perception_lines")),
        "tags":                   _safe_str(row.get("tags")),
        "ai_image_prompt":        _safe_str(row.get("ai_image_prompt")),
        "processed_at":           _safe_str(row.get("processed_at")),
    }


def _tags_as_text(tags_raw: str) -> str:
    """Convert tags field (comma/JSON list) to a space-joined string for embedding."""
    return " ".join(parse_tags(tags_raw))


def _load_watermark() -> int:
    """Load the last successfully ingested published_time_unix from disk."""
    if WATERMARK_FILE.exists():
        try:
            ts = int(WATERMARK_FILE.read_text().strip())
            logger.info("Loaded watermark timestamp: %d", ts)
            return ts
        except ValueError:
            pass
    # Default: epoch 0 → fetch everything
    logger.info("No watermark found — will fetch all articles.")
    return 0


def _save_watermark(ts: int) -> None:
    WATERMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    WATERMARK_FILE.write_text(str(ts))
    logger.info("Watermark saved: %d", ts)


def _build_qdrant_url() -> str:
    if QDRANT_HOST in ("", "local", "embedded", "localhost"):
        return "local://qdrant_storage"
    if QDRANT_HOST.startswith("http://") or QDRANT_HOST.startswith("https://"):
        import re
        clean = re.sub(r":\d+$", "", QDRANT_HOST.rstrip("/"))
        return f"{clean}:{QDRANT_PORT}"
    return f"http://{QDRANT_HOST}:{QDRANT_PORT}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False, since_days: int = None) -> None:
    logger.info("=" * 60)
    logger.info("  Athena → Qdrant / PostgreSQL Incremental Ingestion")
    logger.info("=" * 60)

    # Determine watermark
    if since_days is not None:
        cutoff_dt  = datetime.now(timezone.utc) - timedelta(days=since_days)
        since_unix = int(cutoff_dt.timestamp() * 1000)
        logger.info("Override: fetching articles from last %d days (since %s)", since_days, cutoff_dt.isoformat())
    else:
        since_unix = _load_watermark()

    # ── Athena ────────────────────────────────────────────────────────────────
    athena = AthenaClient()
    logger.info("Fetching articles from Athena (published_time_unix > %d) …", since_unix)
    df = athena.fetch_new_articles(since_unix_ms=since_unix)

    if df.empty:
        logger.info("No new articles in Athena. Nothing to do.")
        return

    logger.info("Athena returned %d candidate articles.", len(df))

    if dry_run:
        logger.info("[DRY RUN] Would process %d articles. Exiting.", len(df))
        return

    # ── Stores ────────────────────────────────────────────────────────────────
    logger.info("Connecting to PostgreSQL …")
    news_store = NewsStore(db_url=DATABASE_URL)
    logger.info("PostgreSQL has %d articles.", news_store.count())

    qdrant_url = _build_qdrant_url()
    logger.info("Connecting to Qdrant at %s …", qdrant_url)
    qdrant_svc = QdrantService(
        url=qdrant_url,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
    )
    qdrant_svc.ensure_collection(vector_size=386)

    logger.info("Loading embedding model '%s' …", MODEL_NAME)
    embed_svc = EmbeddingService(model_name=MODEL_NAME, cache_max_size=50, vector_size=386)
    logger.info("Embedding model ready (dim=%d).", embed_svc.embedding_dim)

    # ── Filter already-present articles ───────────────────────────────────────
    records     = df.to_dict(orient="records")
    new_articles: list[dict] = []
    for raw in records:
        art = _row_to_article(raw)
        if not art["id"]:
            continue
        if news_store.article_exists(art["id"]) and qdrant_svc.point_exists(art["id"]):
            continue
        new_articles.append(art)

    logger.info(
        "%d new articles to ingest (skipped %d already present).",
        len(new_articles), len(records) - len(new_articles),
    )

    if not new_articles:
        logger.info("Nothing to ingest. Exiting.")
        return

    # ── Batch ingestion ───────────────────────────────────────────────────────
    ingested       = 0
    errors         = 0
    max_ts_ingested = since_unix

    pbar = tqdm(total=len(new_articles), unit="article", desc="Ingesting")

    for start in range(0, len(new_articles), BATCH_SIZE):
        batch = new_articles[start : start + BATCH_SIZE]
        try:
            titles   = [a["title"]                  for a in batch]
            descs    = [a["introductory_paragraph"]  for a in batch]
            tags_txt = [_tags_as_text(a["tags"])     for a in batch]

            title_vecs, desc_vecs, tags_vecs = embed_svc.encode_articles_batch(
                titles, descs, tags_txt, batch_size=BATCH_SIZE
            )

            qdrant_arts = [
                {
                    "article_id": a["id"],
                    "title":      a["title"],
                    "summary":    strip_html(a["introductory_paragraph"])[:1000],
                    "tags":       parse_tags(a["tags"]),
                    "timestamp":  a.get("published_time_unix"),
                }
                for a in batch
            ]

            qdrant_svc.upsert_articles(
                articles=qdrant_arts,
                vectors=title_vecs,          # fallback (unused when named given)
                batch_size=BATCH_SIZE,
                title_vectors=title_vecs,
                description_vectors=desc_vecs,
                tags_vectors=tags_vecs,
            )

            news_store.bulk_insert(batch)
            ingested += len(batch)

            # Track max timestamp for watermark update
            for a in batch:
                ts = a.get("published_time_unix")
                if ts and ts > max_ts_ingested:
                    max_ts_ingested = ts

            pbar.update(len(batch))

        except Exception as exc:
            logger.error("Batch %d failed: %s — skipping", start // BATCH_SIZE, exc)
            errors += len(batch)

    pbar.close()

    # ── Save watermark ────────────────────────────────────────────────────────
    if ingested > 0 and max_ts_ingested > since_unix:
        _save_watermark(max_ts_ingested)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Ingestion Complete")
    logger.info("=" * 60)
    logger.info("  Ingested  : %d", ingested)
    logger.info("  Errors    : %d", errors)
    logger.info("  Total in PostgreSQL: %d", news_store.count())
    logger.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Incrementally ingest new articles from Athena (Iceberg) → Qdrant + PostgreSQL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many new Athena rows would be ingested, without writing anything.",
    )
    parser.add_argument(
        "--since-days",
        type=int,
        default=None,
        metavar="N",
        help="Override watermark: fetch articles from the last N days.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, since_days=args.since_days)
