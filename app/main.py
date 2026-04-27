"""
app/main.py
────────────
FastAPI application entrypoint — EC2 / PM2 deployment.

Data stores
───────────
  • Qdrant      — vector search (read-only; populated by separate pipeline)
  • Iceberg     — full article content via AWS Athena (IAM role, no keys)

Startup sequence
────────────────
  1. Load settings from .env
  2. Load BGE-m3 embedding model (386-dim, runs once at startup)
  3. Connect to Qdrant (read-only)
  4. Initialise Athena/Iceberg article service
  5. Load PyTorch NewsRanker weights
"""

import asyncio
import logging
import re
import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

_project_root = str(Path(__file__).parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Load environment variables early so modules can access them at load-time
from dotenv import load_dotenv
load_dotenv(Path(_project_root) / ".env", override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService, VECTOR_SIZE
from app.services.ranking_service import RankingService
from app.services.iceberg_service import IcebergArticleService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Settings ──────────────────────────────────────────────────────────────────

def _load_settings() -> dict:

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")

    if qdrant_host in ("", "local", "embedded"):
        qdrant_url = "local://qdrant_storage"
    elif qdrant_host.startswith("http://") or qdrant_host.startswith("https://"):
        clean = re.sub(r":\d+$", "", qdrant_host.rstrip("/"))
        qdrant_url = f"{clean}:{qdrant_port}"
    else:
        qdrant_url = f"http://{qdrant_host}:{qdrant_port}"

    return {
        "qdrant_url":      qdrant_url,
        "qdrant_api_key":  os.getenv("QDRANT_API_KEY") or None,
        "collection_name": os.getenv("COLLECTION_NAME", "news_embeddings"),
        "model_name":      os.getenv("MODEL_NAME", "BAAI/bge-m3"),
        "vector_size":     int(os.getenv("VECTOR_SIZE", str(VECTOR_SIZE))),
        "ranker_weights":  os.getenv("RANKER_WEIGHTS_PATH", "ranker_weights.pt"),
        "cache_max_size":  int(os.getenv("CACHE_MAX_SIZE", "1000")),
    }


# ── Service loading ───────────────────────────────────────────────────────────

async def _load_services(app: FastAPI, cfg: dict) -> None:
    try:
        # 1. Embedding model
        logger.info("Loading embedding model '%s' (dim=%d) …", cfg["model_name"], cfg["vector_size"])
        loop = asyncio.get_event_loop()
        embedding_svc = await loop.run_in_executor(
            None,
            lambda: EmbeddingService(
                model_name=cfg["model_name"],
                cache_max_size=cfg["cache_max_size"],
                vector_size=cfg["vector_size"],
            ),
        )
        app.state.embedding_service = embedding_svc
        logger.info("Embedding model ready (dim=%d).", embedding_svc.embedding_dim)

        # 2. Qdrant (read-only)
        logger.info("Connecting to Qdrant at %s …", cfg["qdrant_url"])
        qdrant_svc = QdrantService(
            url=cfg["qdrant_url"],
            api_key=cfg["qdrant_api_key"],
            collection_name=cfg["collection_name"],
        )
        qdrant_svc.ensure_collection(vector_size=cfg["vector_size"])
        app.state.qdrant_service = qdrant_svc
        info = qdrant_svc.collection_info()
        logger.info("Qdrant ready — '%s' has %s points.", cfg["collection_name"], info.get("points_count"))

        # 3. Iceberg article service (Athena, IAM role — no keys needed)
        logger.info("Initialising Iceberg/Athena article service …")
        iceberg_svc = IcebergArticleService()
        app.state.iceberg_service = iceberg_svc
        total = iceberg_svc.count()
        logger.info("Iceberg ready — ~%s articles.", total)

        # 4. Neural re-ranker
        logger.info("Loading NewsRanker weights …")
        ranking_svc = RankingService(
            weights_path=cfg["ranker_weights"],
            embedding_dim=embedding_svc.embedding_dim,
        )
        app.state.ranking_service = ranking_svc

        app.state.startup_complete = True
        logger.info("=== All services ready. API is live. ===")

    except Exception:
        logger.exception("FATAL: service initialisation failed")
        app.state.startup_error = True


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = _load_settings()
    logger.info("=== News Recommendation Engine starting up ===")

    app.state.startup_complete  = False
    app.state.startup_error     = False
    app.state.embedding_service = None
    app.state.qdrant_service    = None
    app.state.iceberg_service   = None
    app.state.ranking_service   = None

    asyncio.ensure_future(_load_services(app, cfg))
    yield
    logger.info("=== Shutting down. ===")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="News Recommendation Engine",
    description=(
        "Personalised news recommendations powered by BGE-m3 embeddings, "
        "Qdrant vector search, and a PyTorch neural re-ranker. "
        "Article content served from Apache Iceberg via Athena."
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health():
    if getattr(app.state, "startup_complete", False):
        return {
            "status": "ok",
            "services": {
                "qdrant":   app.state.qdrant_service is not None,
                "iceberg":  app.state.iceberg_service is not None,
                "ranker":   app.state.ranking_service is not None,
            },
        }
    if getattr(app.state, "startup_error", False):
        return JSONResponse(status_code=503, content={"status": "startup_failed"})
    return JSONResponse(status_code=503, content={"status": "loading"})


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "News Recommendation Engine v3", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)