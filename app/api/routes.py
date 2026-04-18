"""
app/api/routes.py
------------------
API route definitions.

Endpoints
---------
  POST /recommend              – personalised news from clicked article + interests
  POST /recommend/portfolio    – personalised news from financial portfolio JSON
  GET  /health                 – liveness + readiness check
  GET  /article/{id}           – full article fetch from SQLite
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.models.schemas import (
    HealthResponse,
    PortfolioRecommendRequest,
    RecommendRequest,
    RecommendResponse,
)
from app.utils.financial_utils import build_portfolio_query_text, summarise_portfolio
from app.utils.text_utils import build_user_query_text

logger = logging.getLogger(__name__)

router = APIRouter()


# -- Dependency helpers --------------------------------------------------------

def get_embedding_service(request: Request):
    return request.app.state.embedding_service

def get_qdrant_service(request: Request):
    return request.app.state.qdrant_service

def get_ranking_service(request: Request):
    return request.app.state.ranking_service

def get_news_store(request: Request):
    return request.app.state.news_store


# -- Shared pipeline helper ---------------------------------------------------

def _run_recommendation_pipeline(
    user_id: str,
    query_text: str,
    categories: list,
    embedding_svc,
    qdrant_svc,
    ranking_svc,
    news_store,
) -> RecommendResponse:
    """Common embed → search → rank pipeline used by both endpoints."""

    # Step 1: Embed
    try:
        user_embedding = embedding_svc.encode(query_text)
    except Exception as exc:
        logger.exception("Embedding failed for user %s", user_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service error: {exc}",
        ) from exc

    # Step 2: Qdrant vector search
    try:
        candidates = qdrant_svc.search(
            query_vector=user_embedding,
            top_k=50,
            categories=categories if categories else None,
        )
    except Exception as exc:
        logger.exception("Qdrant search failed for user %s", user_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector DB error: {exc}",
        ) from exc

    if not candidates:
        logger.warning("No Qdrant candidates for user %s", user_id)
        return RecommendResponse(user_id=user_id, total=0, recommendations=[])

    # Step 3: Neural re-ranking
    try:
        ranked_news = ranking_svc.rank(
            user_embedding=user_embedding,
            candidates=candidates,
            top_k=50,
        )
    except Exception as exc:
        logger.exception("Re-ranking failed for user %s", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Re-ranking error: {exc}",
        ) from exc

    # Step 4: Attach full news
    try:
        if ranked_news:
            ids = [item.article_id for item in ranked_news]
            full_articles_map = news_store.get_by_ids(ids)
            for item in ranked_news:
                item.full_article = full_articles_map.get(item.article_id)
    except Exception as exc:
        logger.exception("Failed to attach full news for user %s, continuing without...", user_id)

    return RecommendResponse(
        user_id=user_id,
        total=len(ranked_news),
        recommendations=ranked_news,
    )


# -- Routes -------------------------------------------------------------------

@router.post(
    "/recommend",
    response_model=RecommendResponse,
    status_code=status.HTTP_200_OK,
    summary="Recommend news from clicked article + interests",
    description=(
        "Combines clicked article content, user interests, and preferred "
        "categories into a BGE-m3 query embedding. Fetches the top-50 nearest "
        "neighbours from Qdrant, then re-ranks with a PyTorch neural network."
    ),
)
async def recommend(
    body: RecommendRequest,
    embedding_svc=Depends(get_embedding_service),
    qdrant_svc=Depends(get_qdrant_service),
    ranking_svc=Depends(get_ranking_service),
    news_store=Depends(get_news_store),
) -> RecommendResponse:
    t0 = time.perf_counter()

    query_text = build_user_query_text(
        clicked_news=body.clicked_news,
        interests=body.interests,
        categories=body.categories,
    )

    result = _run_recommendation_pipeline(
        user_id=body.user_id,
        query_text=query_text,
        categories=body.categories,
        embedding_svc=embedding_svc,
        qdrant_svc=qdrant_svc,
        ranking_svc=ranking_svc,
        news_store=news_store,
    )

    logger.info(
        "recommend | user=%s | ranked=%d | %.1f ms",
        body.user_id, result.total, (time.perf_counter() - t0) * 1000,
    )
    return result


@router.post(
    "/recommend/portfolio",
    response_model=RecommendResponse,
    status_code=status.HTTP_200_OK,
    summary="Recommend news from financial portfolio (AA data)",
    description=(
        "Accepts an Account Aggregator (AA) financial portfolio JSON containing "
        "EQUITIES, MUTUALFUNDS, SIP, REIT, INVIT, DEPOSIT_V2, and/or "
        "INSURANCE_POLICIES sections. Extracts stock names, fund names, "
        "asset types, and themes, then runs the full BGE-m3 → Qdrant → "
        "PyTorch re-ranking pipeline to return relevant financial news."
    ),
)
async def recommend_from_portfolio(
    body: PortfolioRecommendRequest,
    embedding_svc=Depends(get_embedding_service),
    qdrant_svc=Depends(get_qdrant_service),
    ranking_svc=Depends(get_ranking_service),
    news_store=Depends(get_news_store),
) -> RecommendResponse:
    t0 = time.perf_counter()

    # Build a rich query string from the portfolio JSON
    query_text = build_portfolio_query_text(
        portfolio=body.portfolio,
        extra_interests=body.interests,
    )

    if not query_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Could not extract any content from the portfolio. "
                "Ensure at least one of EQUITIES, MUTUALFUNDS, SIP, REIT, "
                "INVIT, DEPOSIT_V2, or INSURANCE_POLICIES is present."
            ),
        )

    portfolio_summary = summarise_portfolio(body.portfolio)
    logger.info(
        "recommend/portfolio | user=%s | sections=%s | query_len=%d",
        body.user_id, list(portfolio_summary.keys()), len(query_text),
    )

    result = _run_recommendation_pipeline(
        user_id=body.user_id,
        query_text=query_text,
        categories=body.categories,
        embedding_svc=embedding_svc,
        qdrant_svc=qdrant_svc,
        ranking_svc=ranking_svc,
        news_store=news_store,
    )

    logger.info(
        "recommend/portfolio | user=%s | ranked=%d | %.1f ms",
        body.user_id, result.total, (time.perf_counter() - t0) * 1000,
    )
    return result


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health / readiness check",
)
async def health(request: Request) -> HealthResponse:
    # During background warmup, services may not be ready yet
    if not getattr(request.app.state, "startup_complete", False):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={
                "status": "loading",
                "loading": True,
                "qdrant_connected": None,
                "sqlite_articles": None,
                "embedding_model": None,
            },
        )

    embedding_svc = request.app.state.embedding_service
    qdrant_svc = request.app.state.qdrant_service
    news_store = request.app.state.news_store

    return HealthResponse(
        status="ok",
        loading=False,
        qdrant_connected=qdrant_svc.is_healthy(),
        sqlite_articles=news_store.count(),
        embedding_model=embedding_svc.model_name,
    )


@router.get(
    "/article/{article_id}",
    summary="Fetch full article from local SQLite store",
    description=(
        "Returns the complete article record including all paragraphs, "
        "impact matrix, and image prompt. Fetched from SQLite, not Qdrant."
    ),
)
async def get_article(
    article_id: str,
    news_store=Depends(get_news_store),
) -> dict:
    article = news_store.get_by_id(article_id)
    if article is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article '{article_id}' not found in local store.",
        )
    return article


# -- Embed endpoint (used by ec2_vector.js JS service) -------------------------

from pydantic import BaseModel

class EmbedRequest(BaseModel):
    text: str

@router.post(
    "/embed",
    summary="Embed a text string → 386-dim vector",
    description=(
        "Reuses the already-loaded BGE-m3 model to produce a single embedding. "
        "Called by the Node.js ec2_vector.js pipeline so the model is not "
        "loaded twice on the same machine."
    ),
)
async def embed_text(
    body: EmbedRequest,
    embedding_svc=Depends(get_embedding_service),
) -> dict:
    if not getattr(embedding_svc, "model_name", None):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model not yet loaded — retry in a moment.",
        )
    try:
        vector = embedding_svc.encode(body.text)
        return {"vector": vector.tolist(), "dim": len(vector)}
    except Exception as exc:
        logger.exception("Embed endpoint error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
