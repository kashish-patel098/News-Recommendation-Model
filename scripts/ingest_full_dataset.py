r"""
scripts/ingest_full_dataset.py
-------------------------------
Ingests news_dataset.csv into SQLite + Qdrant in manageable runs.

Run multiple times until all articles are ingested:
  python scripts/ingest_full_dataset.py           # ingest next 1000
  python scripts/ingest_full_dataset.py           # ingest next 1000
  ...                                             # repeat until done

Each run:
  1. Streams the CSV in small chunks (never loads the full file)
  2. Skips articles already in BOTH stores (idempotent)
  3. Stops after --limit NEW articles are ingested (default: 1000)
  4. Prints how many remain so you know when to stop

Options:
  --limit N      Stop after N new articles ingested (default: 1000)
  --batch N      Embedding batch size (default: 16)
  --dry-run      Count new articles without writing anything
  --all          Ignore limit, ingest everything in one run (may be slow)

Examples:
  python scripts/ingest_full_dataset.py
  python scripts/ingest_full_dataset.py --limit 500
  python scripts/ingest_full_dataset.py --all
  python scripts/ingest_full_dataset.py --dry-run
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# -- Project root on sys.path --------------------------------------------------
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(ROOT / ".env", override=False)

from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.utils.text_utils import (
    build_news_embedding_text,
    parse_tags,
    strip_html,
)
from local_store.news_store import NewsStore

# -- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ingest_full_dataset")

# -- Configuration -------------------------------------------------------------
DEFAULT_CSV        = os.getenv("DATASET_PATH",        str(ROOT / "news_dataset.csv"))
DEFAULT_BATCH      = int(os.getenv("INGEST_BATCH_SIZE", "16"))
DEFAULT_LIMIT      = 1000          # new articles per run
COLLECTION_NAME    = os.getenv("COLLECTION_NAME",   "news_embeddings")
QDRANT_HOST        = os.getenv("QDRANT_HOST",        "localhost")
QDRANT_PORT        = int(os.getenv("QDRANT_PORT",    "6333"))
QDRANT_API_KEY     = os.getenv("QDRANT_API_KEY")     or None
MODEL_NAME         = os.getenv("MODEL_NAME",          "BAAI/bge-m3")
DATABASE_URL       = os.getenv("DATABASE_URL",       "postgresql://postgres:postgres@localhost:5432/newsdb")

# All columns from news_dataset.csv
CSV_COLUMNS = [
    "id",
    "published_time",
    "title",
    "introductory_paragraph",
    "descriptive_paragraph",
    "historical_context",
    "economyimpact",
    "impact_matrix",
    "perception_lines",
    "tags",
    "ai_image_prompt",
    "processed_at",
    "published_time_unix",
]


# -- Helpers -------------------------------------------------------------------

def _safe_str(val) -> str:
    """Convert NaN / None to empty string."""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return "" if val is None else str(val)


def _safe_int(val):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _row_to_article(row: pd.Series) -> dict:
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


# -- Main ----------------------------------------------------------------------

def count_total_rows(csv_file: Path) -> int:
    """Fast row count without loading data."""
    count = 0
    with open(csv_file, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            count += 1
    return max(0, count - 1)  # subtract header


def main(
    csv_path: str,
    batch_size: int,
    limit: int,
    dry_run: bool,
    ingest_all: bool,
) -> None:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        logger.error("CSV not found: %s", csv_file)
        sys.exit(1)

    effective_limit = float("inf") if ingest_all else limit

    logger.info("=" * 55)
    logger.info("  News Dataset Ingestion")
    logger.info("=" * 55)
    logger.info("  CSV        : %s", csv_file)
    logger.info("  Batch size : %d", batch_size)
    logger.info("  Run limit  : %s new articles", "ALL" if ingest_all else limit)
    logger.info("  Dry run    : %s", dry_run)
    logger.info("=" * 55)

    # -- Count total rows (quick pass) -----------------------------------------
    logger.info("Counting total rows in CSV...")
    total_rows = count_total_rows(csv_file)
    logger.info("Total articles in CSV: %d", total_rows)

    if dry_run:
        logger.info("[DRY RUN] Would stream CSV and ingest up to %s new articles.", effective_limit)
        return

    # -- Initialise stores -----------------------------------------------------
    logger.info("Connecting to PostgreSQL...")
    news_store = NewsStore(db_url=DATABASE_URL)
    already_in_postgres = news_store.count()
    logger.info("PostgreSQL already has: %d articles", already_in_postgres)


    if QDRANT_HOST in ("", "local", "embedded", "localhost"):
        qdrant_url = "local://qdrant_storage"
    elif QDRANT_HOST.startswith("http://") or QDRANT_HOST.startswith("https://"):
        import re as _re
        clean = _re.sub(r":\d+$", "", QDRANT_HOST.rstrip("/"))
        qdrant_url = f"{clean}:{QDRANT_PORT}"
    else:
        qdrant_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

    logger.info("Connecting to Qdrant at %s...", qdrant_url)
    qdrant_svc = QdrantService(
        url=qdrant_url,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
    )

    qdrant_svc.ensure_collection()

    logger.info("Loading embedding model '%s'...", MODEL_NAME)
    logger.info("(First load downloads ~2GB model weights, please wait...)")
    embed_svc = EmbeddingService(model_name=MODEL_NAME, cache_max_size=50)
    logger.info("Embedding model ready.")

    # -- Stream CSV and process ------------------------------------------------
    ingested_this_run = 0
    skipped_this_run  = 0
    error_this_run    = 0

    # Stream the CSV in chunks — never loads the full file
    csv_iter = pd.read_csv(
        csv_file,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        chunksize=batch_size,     # read exactly batch_size rows at a time
    )

    pbar = tqdm(
        total=limit if not ingest_all else total_rows,
        unit="article",
        desc="Ingesting",
        dynamic_ncols=True,
    )

    stop_requested = False

    for chunk_df in csv_iter:
        if stop_requested:
            break

        # Normalise columns
        chunk_df.columns = [c.strip() for c in chunk_df.columns]

        articles_for_sqlite = []
        articles_for_qdrant  = []
        texts_to_embed        = []

        for _, row in chunk_df.iterrows():
            # Check limit BEFORE processing each article
            if ingested_this_run >= effective_limit:
                stop_requested = True
                break

            article_id = _safe_str(row.get("id"))
            if not article_id:
                skipped_this_run += 1
                continue

            # Idempotency: skip if already in BOTH stores
            in_sqlite = news_store.article_exists(article_id)
            in_qdrant = qdrant_svc.point_exists(article_id)
            if in_sqlite and in_qdrant:
                skipped_this_run += 1
                continue

            try:
                article = _row_to_article(row)
            except Exception as exc:
                logger.warning("Row parse error (id=%s): %s", article_id, exc)
                error_this_run += 1
                continue

            articles_for_sqlite.append(article)

            embed_text = build_news_embedding_text(
                title=article["title"],
                intro=article["introductory_paragraph"],
            )
            texts_to_embed.append(embed_text)

            tags = parse_tags(article["tags"])
            articles_for_qdrant.append({
                "article_id": article_id,
                "title":      article["title"],
                "summary":    strip_html(article["introductory_paragraph"])[:1000],
                "tags":       tags,
                "timestamp":  article.get("published_time_unix"),
            })

        if not articles_for_sqlite:
            continue

        try:
            # 1. Write full data to SQLite
            news_store.bulk_insert(articles_for_sqlite)

            # 2. Embed in batch
            vectors = embed_svc.encode_batch(texts_to_embed, batch_size=len(texts_to_embed))

            # 3. Upsert to Qdrant
            qdrant_svc.upsert_articles(
                articles=articles_for_qdrant,
                vectors=vectors,
                batch_size=len(articles_for_qdrant),
            )

            cnt = len(articles_for_sqlite)
            ingested_this_run += cnt
            pbar.update(cnt)

        except Exception as exc:
            logger.error("Batch failed: %s — skipping batch", exc)
            error_this_run += len(articles_for_sqlite)
            continue

    pbar.close()

    # -- Summary ---------------------------------------------------------------
    total_in_sqlite = news_store.count()
    remaining       = max(0, total_rows - total_in_sqlite)

    logger.info("")
    logger.info("=" * 55)
    logger.info("  Run Complete")
    logger.info("=" * 55)
    logger.info("  Ingested this run  : %d", ingested_this_run)
    logger.info("  Skipped (exist)    : %d", skipped_this_run)
    logger.info("  Errors             : %d", error_this_run)
    logger.info("  Total in SQLite    : %d / %d", total_in_sqlite, total_rows)
    logger.info("  Remaining          : %d", remaining)

    if remaining > 0:
        logger.info("")
        logger.info("  >> Run the script again to ingest the next batch.")
        logger.info("  >> python scripts/ingest_full_dataset.py")
    else:
        logger.info("")
        logger.info("  >> All articles have been ingested!")

    logger.info("=" * 55)


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Ingest news_dataset.csv into SQLite + Qdrant.\n"
            "Run multiple times — each run processes the next N new articles."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="Path to source CSV (default: news_dataset.csv in project root)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Max NEW articles to ingest per run (default: 1000)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help="Embedding batch size — lower = less RAM (default: 16)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="ingest_all",
        help="Ignore --limit and ingest everything in one run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without writing anything",
    )
    args = parser.parse_args()

    main(
        csv_path=args.csv,
        batch_size=args.batch,
        limit=args.limit,
        dry_run=args.dry_run,
        ingest_all=args.ingest_all,
    )
