"""
scripts/ingest_latest_news.py
──────────────────────────────
Ongoing ingestion of NEW articles into both stores.

Use this script whenever you receive a batch of new articles
(e.g. daily feed, API pull, new CSV export from your news provider).

Input formats supported
───────────────────────
  --csv   path/to/new_articles.csv     (same schema as news_dataset.csv)
  --json  path/to/new_articles.json    (list of article objects)

What it does
────────────
  1. Parses the input file
  2. Filters out articles already present in EITHER store (idempotent)
  3. Writes full article data to SQLite   (all columns)
  4. Embeds title + introductory_paragraph with BGE-m3
  5. Upserts lightweight payload + vector to Qdrant
  6. Prints a summary of new articles added

Usage examples
──────────────
  # Activate .venv first
  .\.venv\Scripts\Activate.ps1

  # From a new CSV file
  python scripts/ingest_latest_news.py --csv latest_news.csv

  # From a JSON file
  python scripts/ingest_latest_news.py --json latest_news.json

  # Dry run (just show how many new articles would be added)
  python scripts/ingest_latest_news.py --csv latest_news.csv --dry-run

JSON format
───────────
  Each object must have at minimum: id, title
  All other fields are optional and will default to empty string / None.

  [
    {
      "id": "unique_article_id",
      "published_time": "2026-04-05 10:00:00.000000",
      "published_time_unix": 1775000000000,
      "title": "Breaking: Markets rally on Fed rate cut",
      "introductory_paragraph": "Stock markets surged ...",
      "descriptive_paragraph": "...",
      "historical_context": "...",
      "economyimpact": "...",
      "impact_matrix": "...",
      "perception_lines": "...",
      "tags": ["FINANCE", "US Economy"],
      "ai_image_prompt": "...",
      "processed_at": "2026-04-05T04:30:00.000Z"
    },
    ...
  ]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ingest_latest_news")

# ── Config (same as ingest_full_dataset.py) ───────────────────────────────────
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "news_embeddings")
QDRANT_HOST     = os.getenv("QDRANT_HOST",      "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT",  "6333"))
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")  or None
MODEL_NAME      = os.getenv("MODEL_NAME",        "BAAI/bge-m3")
SQLITE_DB_PATH  = os.getenv("SQLITE_DB_PATH",   str(ROOT / "local_store/news.db"))
BATCH_SIZE      = int(os.getenv("INGEST_BATCH_SIZE", "32"))


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _safe_str(val) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val)


def _safe_int(val) -> int | None:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _to_article_dict(raw: dict) -> dict:
    """Normalise an article from any input format to our internal format."""
    return {
        "id":                     _safe_str(raw.get("id")),
        "published_time":         _safe_str(raw.get("published_time")),
        "published_time_unix":    _safe_int(raw.get("published_time_unix")),
        "title":                  _safe_str(raw.get("title")),
        "introductory_paragraph": _safe_str(raw.get("introductory_paragraph")),
        "descriptive_paragraph":  _safe_str(raw.get("descriptive_paragraph")),
        "historical_context":     _safe_str(raw.get("historical_context")),
        "economyimpact":          _safe_str(raw.get("economyimpact")),
        "impact_matrix":          _safe_str(raw.get("impact_matrix")),
        "perception_lines":       _safe_str(raw.get("perception_lines")),
        "tags":                   raw.get("tags", ""),   # keep raw for SQLite
        "ai_image_prompt":        _safe_str(raw.get("ai_image_prompt")),
        "processed_at":           _safe_str(raw.get("processed_at")),
    }


def load_csv(path: Path) -> list[dict]:
    """Load articles from a CSV file (same schema as news_dataset.csv)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    df.columns = [c.strip() for c in df.columns]
    return df.to_dict(orient="records")


def load_json(path: Path) -> list[dict]:
    """Load articles from a JSON array file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of article objects.")
    return data


# ── Main ──────────────────────────────────────────────────────────────────────

def main(input_path: Path, dry_run: bool) -> None:
    logger.info("=== Latest News Ingestion Starting ===")
    logger.info("Input file : %s", input_path)
    logger.info("Dry run    : %s", dry_run)

    # ── Load input file ───────────────────────────────────────────────────────
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        raw_articles = load_csv(input_path)
    elif suffix == ".json":
        raw_articles = load_json(input_path)
    else:
        logger.error("Unsupported file format: '%s'. Use .csv or .json", suffix)
        sys.exit(1)

    logger.info("Loaded %d articles from input file.", len(raw_articles))

    if dry_run:
        logger.info("[DRY RUN] Would process up to %d articles.", len(raw_articles))
        return

    # ── Initialise stores ─────────────────────────────────────────────────────
    news_store = NewsStore(db_path=SQLITE_DB_PATH)
    qdrant_svc = QdrantService(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
    )
    qdrant_svc.ensure_collection()

    logger.info("Loading embedding model '%s' …", MODEL_NAME)
    embed_svc = EmbeddingService(model_name=MODEL_NAME, cache_max_size=100)
    logger.info("Embedding model ready.")

    # ── Filter already-ingested articles ──────────────────────────────────────
    new_articles: list[dict] = []
    for raw in raw_articles:
        article = _to_article_dict(raw)
        article_id = article["id"]
        if not article_id:
            continue
        if news_store.article_exists(article_id) and qdrant_svc.point_exists(article_id):
            continue
        new_articles.append(article)

    logger.info(
        "%d new articles to ingest (skipped %d already present).",
        len(new_articles),
        len(raw_articles) - len(new_articles),
    )

    if not new_articles:
        logger.info("Nothing to ingest. Exiting.")
        return

    # ── Process in batches ────────────────────────────────────────────────────
    total_sqlite = 0
    total_qdrant = 0

    pbar = tqdm(total=len(new_articles), unit="article", desc="Ingesting latest news")

    for start in range(0, len(new_articles), BATCH_SIZE):
        batch = new_articles[start : start + BATCH_SIZE]

        # ── Write full data to SQLite ─────────────────────────────────────
        count_s = news_store.bulk_insert(batch)
        total_sqlite += count_s

        # ── Build texts for embedding ──────────────────────────────────────
        texts = [
            build_news_embedding_text(
                title=a["title"],
                intro=a["introductory_paragraph"],
            )
            for a in batch
        ]

        # ── Embed in batch ────────────────────────────────────────────────
        import numpy as np
        vectors = embed_svc.encode_batch(texts, batch_size=BATCH_SIZE)

        # ── Build Qdrant lightweight payloads ─────────────────────────────
        qdrant_articles = [
            {
                "article_id": a["id"],
                "title":      a["title"],
                "summary":    strip_html(a["introductory_paragraph"])[:1000],
                "tags":       parse_tags(a["tags"]),
                "timestamp":  a.get("published_time_unix"),
            }
            for a in batch
        ]

        # ── Upsert to Qdrant ──────────────────────────────────────────────
        count_q = qdrant_svc.upsert_articles(
            articles=qdrant_articles,
            vectors=vectors,
            batch_size=BATCH_SIZE,
        )
        total_qdrant += count_q

        pbar.update(len(batch))

    pbar.close()

    logger.info("=== Latest News Ingestion Complete ===")
    logger.info("  SQLite rows written   : %d", total_sqlite)
    logger.info("  Qdrant points upserted: %d", total_qdrant)
    logger.info("  Total in SQLite        : %d", news_store.count())


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Ingest latest news articles into SQLite + Qdrant.\n"
            "Accepts a CSV file (same schema as news_dataset.csv) or a JSON array."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv",  type=Path, help="Path to new articles CSV file")
    group.add_argument("--json", type=Path, help="Path to new articles JSON file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many new articles would be added, without writing anything",
    )
    args = parser.parse_args()

    input_path: Path = args.csv or args.json
    if not input_path.exists():
        logger.error("File not found: %s", input_path)
        sys.exit(1)

    main(input_path=input_path, dry_run=args.dry_run)
