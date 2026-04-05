"""
app/api/ingest_routes.py
------------------------
API endpoints for uploading and ingesting new news articles at runtime.

POST /api/v1/ingest/csv
  - Accepts a multipart CSV file upload
  - Same column schema as news_dataset.csv
  - Runs ingest in the background so the HTTP response returns immediately
  - Returns a job_id; poll GET /api/v1/ingest/status/{job_id} for progress

GET /api/v1/ingest/status/{job_id}
  - Returns status, counts, errors for a running/completed ingest job
"""

import asyncio
import io
import logging
import time
import uuid
from typing import Dict

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, status

from app.utils.text_utils import build_news_embedding_text, parse_tags, strip_html

logger = logging.getLogger(__name__)

ingest_router = APIRouter()

# In-memory job store (good enough for a single-server setup)
_jobs: Dict[str, dict] = {}

CSV_COLUMNS = [
    "id", "published_time", "title", "introductory_paragraph",
    "descriptive_paragraph", "historical_context", "economyimpact",
    "impact_matrix", "perception_lines", "tags", "ai_image_prompt",
    "processed_at", "published_time_unix",
]


# -- Dependency helpers --------------------------------------------------------

def get_embedding_service(request: Request):
    return request.app.state.embedding_service

def get_qdrant_service(request: Request):
    return request.app.state.qdrant_service

def get_news_store(request: Request):
    return request.app.state.news_store


# -- Helpers -------------------------------------------------------------------

def _safe_str(val) -> str:
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


# -- Background ingest task ---------------------------------------------------

async def _run_ingest(
    job_id: str,
    df: pd.DataFrame,
    embed_svc,
    qdrant_svc,
    news_store,
    batch_size: int = 16,
):
    """Runs the CSV → SQLite + Qdrant ingest pipeline asynchronously."""
    job = _jobs[job_id]
    job["status"]   = "running"
    job["total"]    = len(df)
    job["ingested"] = 0
    job["skipped"]  = 0
    job["errors"]   = 0

    try:
        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start : start + batch_size]

            articles_sqlite = []
            articles_qdrant  = []
            texts            = []

            for _, row in chunk.iterrows():
                article_id = _safe_str(row.get("id"))
                if not article_id:
                    job["skipped"] += 1
                    continue

                if news_store.article_exists(article_id) and qdrant_svc.point_exists(article_id):
                    job["skipped"] += 1
                    continue

                article = _row_to_article(row)
                articles_sqlite.append(article)
                texts.append(build_news_embedding_text(
                    title=article["title"],
                    intro=article["introductory_paragraph"],
                ))
                articles_qdrant.append({
                    "article_id": article_id,
                    "title":      article["title"],
                    "summary":    strip_html(article["introductory_paragraph"])[:1000],
                    "tags":       parse_tags(article["tags"]),
                    "timestamp":  article.get("published_time_unix"),
                })

            if not articles_sqlite:
                continue

            # Yield control so the event loop stays responsive
            await asyncio.sleep(0)

            # Run blocking operations in a thread pool
            loop = asyncio.get_event_loop()

            await loop.run_in_executor(None, news_store.bulk_insert, articles_sqlite)

            import numpy as np
            vectors = await loop.run_in_executor(
                None, embed_svc.encode_batch, texts, batch_size
            )

            await loop.run_in_executor(
                None,
                lambda: qdrant_svc.upsert_articles(
                    articles=articles_qdrant,
                    vectors=vectors,
                    batch_size=len(articles_qdrant),
                )
            )

            job["ingested"] += len(articles_sqlite)
            job["message"]   = f"Ingested {job['ingested']} / {job['total']}"

        job["status"]  = "done"
        job["message"] = (
            f"Complete. {job['ingested']} new articles added, "
            f"{job['skipped']} skipped."
        )
        logger.info("Ingest job %s done. %d ingested, %d skipped.", job_id, job["ingested"], job["skipped"])

    except Exception as exc:
        job["status"]  = "error"
        job["message"] = str(exc)
        logger.exception("Ingest job %s failed.", job_id)


# -- Routes -------------------------------------------------------------------

@ingest_router.post(
    "/ingest/csv",
    summary="Upload a latest-news CSV and ingest into both stores",
    description=(
        "Upload a CSV file with the same column schema as news_dataset.csv. "
        "Articles already in both stores are skipped (idempotent). "
        "Ingest runs in the background — use the returned job_id to poll status."
    ),
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_csv(
    file: UploadFile = File(..., description="CSV file (same schema as news_dataset.csv)"),
    embed_svc=Depends(get_embedding_service),
    qdrant_svc=Depends(get_qdrant_service),
    news_store=Depends(get_news_store),
):
    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only .csv files are accepted.",
        )

    # Read file bytes and parse
    raw = await file.read()
    try:
        df = pd.read_csv(
            io.BytesIO(raw),
            dtype=str,
            keep_default_na=False,
            na_values=[""],
        )
        df.columns = [c.strip() for c in df.columns]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse CSV: {exc}",
        )

    if "id" not in df.columns or "title" not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV must contain at least 'id' and 'title' columns.",
        )

    # Keep only known columns
    present = [c for c in CSV_COLUMNS if c in df.columns]
    df = df[present]

    total_rows = len(df)
    if total_rows == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV is empty.",
        )

    # Create job
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "job_id":   job_id,
        "status":   "queued",
        "total":    total_rows,
        "ingested": 0,
        "skipped":  0,
        "errors":   0,
        "message":  f"Queued {total_rows} rows.",
        "filename": file.filename,
        "started_at": time.time(),
    }

    # Fire and forget
    asyncio.create_task(
        _run_ingest(
            job_id=job_id,
            df=df,
            embed_svc=embed_svc,
            qdrant_svc=qdrant_svc,
            news_store=news_store,
        )
    )

    logger.info("Ingest job %s queued: %d rows from '%s'", job_id, total_rows, file.filename)

    return {
        "job_id":    job_id,
        "total":     total_rows,
        "filename":  file.filename,
        "message":   f"Ingesting {total_rows} articles in background.",
        "poll_url":  f"/api/v1/ingest/status/{job_id}",
    }


@ingest_router.get(
    "/ingest/status/{job_id}",
    summary="Poll ingest job status",
)
async def ingest_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return job
