"""
app/services/qdrant_service.py
───────────────────────────────
Qdrant vector database client.

Collection schema
─────────────────
  Collection : news_embeddings
  Vector     : 1024-dim, cosine distance
  Payload    : {
      "article_id"  : str,          ← same as point id string
      "title"       : str,
      "summary"     : str,          ← introductory_paragraph
      "tags"        : List[str],    ← parsed category list
      "timestamp"   : int,          ← published_time_unix (ms)
  }

Design choices
──────────────
• Payload stores only the lightweight fields.  Full content lives in SQLite.
• Points use integer IDs (Qdrant native) derived from article_id; a
  separate payload field "article_id" (str) preserves the original string ID.
• category filtering is done via Qdrant MatchAny filter on `tags`.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

logger = logging.getLogger(__name__)

# Default — overridden at runtime from EmbeddingService.embedding_dim
_DEFAULT_VECTOR_SIZE = 1024


def _article_id_to_point_id(article_id: str) -> int:
    """
    Convert a string article ID to a stable integer Qdrant point ID.
    The CSV IDs are already numeric strings (e.g. "1774500060554"),
    so we just parse them as int. Falls back to hash for non-numeric IDs.
    """
    try:
        return int(article_id)
    except (ValueError, TypeError):
        import hashlib
        digest = hashlib.md5(str(article_id).encode()).hexdigest()
        return int(digest[:15], 16)


class QdrantService:
    """Manages all Qdrant interactions for the recommendation engine."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        collection_name: str = "news_embeddings",
    ):
        self.collection_name = collection_name
        self._client = self._build_client(host, port, api_key)

    # ── Client Setup ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_client(
        host: str, port: int, api_key: Optional[str]
    ) -> QdrantClient:
        # Explicitly build an HTTP URL and force prefer_grpc=False to ensure REST API is used everywhere
        url = host if host.startswith("http") else f"http://{host}:{port}"
        return QdrantClient(url=url, api_key=api_key, prefer_grpc=False, timeout=30)

    # ── Collection Management ─────────────────────────────────────────────────

    def ensure_collection(self, vector_size: int = _DEFAULT_VECTOR_SIZE) -> None:
        """Create collection if it doesn't already exist."""
        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection_name in existing:
            logger.info(
                "Collection '%s' already exists, skipping creation.",
                self.collection_name,
            )
            return

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Collection '%s' created (dim=%d, cosine).", self.collection_name, vector_size)

    def collection_info(self) -> Dict[str, Any]:
        info = self._client.get_collection(self.collection_name)
        # vectors_count was renamed/moved in newer qdrant-client versions
        vectors_count = (
            getattr(info, "vectors_count", None)
            or getattr(info, "points_count", None)
            or 0
        )
        points_count = getattr(info, "points_count", None) or 0
        return {
            "vectors_count": vectors_count,
            "points_count":  points_count,
            "status":        str(info.status),
        }

    def is_healthy(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_articles(
        self,
        articles: List[Dict[str, Any]],
        vectors: np.ndarray,
        batch_size: int = 64,
    ) -> int:
        """
        Upsert article payloads + their vectors to Qdrant.

        articles : list of dicts with keys:
                   article_id, title, summary, tags (List[str]), timestamp
        vectors  : np.ndarray of shape (N, 1024)
        """
        assert len(articles) == len(vectors), "articles and vectors must match"
        total_upserted = 0

        for i in range(0, len(articles), batch_size):
            batch_articles = articles[i : i + batch_size]
            batch_vectors = vectors[i : i + batch_size]

            points = [
                PointStruct(
                    id=_article_id_to_point_id(a["article_id"]),
                    vector=batch_vectors[j].tolist(),
                    payload={
                        "article_id": a["article_id"],
                        "title":      a["title"],
                        "summary":    a["summary"],
                        "tags":       a["tags"],          # List[str]
                        "timestamp":  a.get("timestamp"),
                    },
                )
                for j, a in enumerate(batch_articles)
            ]

            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            total_upserted += len(points)
            logger.debug("Qdrant upsert batch %d–%d OK", i, i + len(points))

        logger.info("Upserted %d points to Qdrant.", total_upserted)
        return total_upserted

    def upsert_single(
        self,
        article_id: str,
        title: str,
        summary: str,
        tags: List[str],
        timestamp: Optional[int],
        vector: np.ndarray,
    ) -> None:
        """Upsert a single article — used by the latest-news ingest script."""
        point = PointStruct(
            id=_article_id_to_point_id(article_id),
            vector=vector.tolist(),
            payload={
                "article_id": article_id,
                "title":      title,
                "summary":    summary,
                "tags":       tags,
                "timestamp":  timestamp,
            },
        )
        self._client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        logger.info("Upserted article %s to Qdrant.", article_id)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 50,
        categories: Optional[List[str]] = None,
    ) -> List[ScoredPoint]:
        """
        Retrieve the top-k most similar articles from Qdrant.
        Optionally filter by category tags using MatchAny.
        """
        query_filter: Optional[Filter] = None
        if categories:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="tags",
                        match=MatchAny(any=categories),
                    )
                ]
            )

        t0 = time.perf_counter()
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=True,   # needed for re-ranker
        )
        results = response.points
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Qdrant search returned %d results in %.1f ms", len(results), elapsed_ms
        )
        return results

    # ── Point Existence Check ─────────────────────────────────────────────────

    def point_exists(self, article_id: str) -> bool:
        """Check if a point is already indexed."""
        point_id = _article_id_to_point_id(article_id)
        try:
            result = self._client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=False,
                with_vectors=False,
            )
            return len(result) > 0
        except UnexpectedResponse:
            return False
