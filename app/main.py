"""
app/main.py
────────────
FastAPI application entrypoint — EC2 / Docker deployment.

Startup sequence (lifespan)
───────────────────────────
  1. Load settings from .env
  2. Initialise PostgreSQL store (full article content)
  3. Load BGE-m3 embedding model with 386-dim projection (runs once at startup)
  4. Connect to EC2 Qdrant and ensure named-vector collection exists
     (title / description / tags — each 386-dim cosine)
  5. Load PyTorch NewsRanker (fine-tunable monthly)
  6. Attach everything to app.state for dependency injection

Health check
────────────
  GET /health — returns 200 once all services are ready, 503 while loading.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

_project_root = str(Path(__file__).parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.api.ingest_routes import ingest_router
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService, VECTOR_SIZE
from app.services.ranking_service import RankingService
from local_store.news_store import NewsStore

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Settings (from .env) ──────────────────────────────────────────────────────

def _load_settings() -> dict:
    import re
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[1] / ".env", override=True)

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
        "database_url":    os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/newsdb"),
        "cache_max_size":  int(os.getenv("CACHE_MAX_SIZE", "1000")),
    }


# ── Service loading ───────────────────────────────────────────────────────────

async def _load_services(app: FastAPI, cfg: dict) -> None:
    """Load all heavy services asynchronously at startup."""
    try:
        # 1. PostgreSQL store
        logger.info("Initialising PostgreSQL store …")
        news_store = NewsStore(db_url=cfg["database_url"])
        app.state.news_store = news_store
        logger.info("PostgreSQL store ready (%d articles).", news_store.count())

        # 2. Embedding model (slow — run in executor to keep event loop free)
        logger.info(
            "Loading embedding model '%s' (vector_size=%d) … (may take a minute)",
            cfg["model_name"], cfg["vector_size"],
        )
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
        logger.info("Embedding model ready (output_dim=%d).", embedding_svc.embedding_dim)

        # 3. Qdrant — named multi-vector collection
        logger.info("Connecting to Qdrant at %s …", cfg["qdrant_url"])
        qdrant_svc = QdrantService(
            url=cfg["qdrant_url"],
            api_key=cfg["qdrant_api_key"],
            collection_name=cfg["collection_name"],
        )
        qdrant_svc.ensure_collection(vector_size=cfg["vector_size"])
        app.state.qdrant_service = qdrant_svc
        info = qdrant_svc.collection_info()
        logger.info(
            "Qdrant ready — collection '%s' has %s points.",
            cfg["collection_name"], info.get("points_count"),
        )

        # 4. Neural re-ranker
        logger.info("Loading neural re-ranker …")
        ranking_svc = RankingService(
            weights_path=cfg["ranker_weights"],
            embedding_dim=embedding_svc.embedding_dim,
        )
        app.state.ranking_service = ranking_svc

        app.state.startup_complete = True
        logger.info("=== All services ready. API is live. ===")

    except Exception:  # noqa: BLE001
        logger.exception(
            "FATAL: service initialisation failed — API stays in 'loading' state"
        )
        app.state.startup_error = True


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bind port immediately, then load heavy services in the background."""
    cfg = _load_settings()
    logger.info("=== News Recommendation Engine starting up ===")

    # Pre-set state flags so health-check doesn't crash with AttributeError
    app.state.startup_complete = False
    app.state.startup_error    = False
    app.state.news_store       = None
    app.state.embedding_service = None
    app.state.qdrant_service   = None
    app.state.ranking_service  = None

    # Fire-and-forget — uvicorn binds to $PORT before this finishes
    asyncio.ensure_future(_load_services(app, cfg))

    yield

    logger.info("=== Shutting down. ===")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="News Recommendation Engine",
    description=(
        "Real-time personalised news recommendations powered by BGE-m3 "
        "embeddings (386-dim named vectors), Qdrant vector search, "
        "and a PyTorch neural re-ranker. Trained monthly on new Athena/Iceberg data."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(router,        prefix="/api/v1")
app.include_router(ingest_router, prefix="/api/v1")

# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health():
    """
    Returns 200 once all services are ready.
    Returns 503 while the app is still initialising.
    Used by docker-compose and ALB health checks.
    """
    if getattr(app.state, "startup_complete", False):
        return {
            "status": "ok",
            "services": {
                "postgres": app.state.news_store is not None,
                "qdrant":   app.state.qdrant_service is not None,
                "ranker":   app.state.ranking_service is not None,
            },
        }
    if getattr(app.state, "startup_error", False):
        return JSONResponse(status_code=503, content={"status": "startup_failed"})
    return JSONResponse(status_code=503, content={"status": "loading"})


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "News Recommendation Engine v2", "docs": "/docs"}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )