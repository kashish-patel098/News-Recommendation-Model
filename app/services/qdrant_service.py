"""
app/services/qdrant_service.py
───────────────────────────────
Qdrant vector database client — **named multi-vector** schema.

Collection schema
─────────────────
  Collection : news_embeddings  (or COLLECTION_NAME env var)
  Vectors    : named, each 386-dim cosine —
      "title"       → embedding of article title
      "description" → embedding of introductory_paragraph
      "tags"        → embedding of space-joined tag list
  Payload    : {
      "article_id"  : str,
      "title"       : str,
      "summary"     : str,          ← introductory_paragraph (truncated)
      "tags"        : List[str],
      "timestamp"   : int,          ← published_time_unix (ms)
  }

Design choices
──────────────
• Three separate 386-dim vectors per article for fine-grained similarity.
• Query is dispatched against the named vector that best matches intent
  (default: "title" for simple queries; caller can override).
• Payload stores lightweight fields; full content lives in PostgreSQL.
• Points use integer IDs derived from article_id string.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Literal

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
    NamedVector,
)

logger = logging.getLogger(__name__)

VECTOR_SIZE = 386
VectorField = Literal["title", "description", "tags"]

_NAMED_VECTORS_CONFIG = {
    "title":       VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    "description": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    "tags":        VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
}


def _article_id_to_point_id(article_id: str) -> int:
    """
    Convert a string article ID to a stable integer Qdrant point ID.
    Numeric strings are parsed directly; non-numeric fall back to MD5 hash.
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
        url: str = "local://qdrant_storage",
        api_key: Optional[str] = None,
        collection_name: str = "news_embeddings",
    ):
        self.collection_name = collection_name
        if url.startswith("local://"):
            path = url[len("local://"):]
            self._client = QdrantClient(path=path)
            logger.info("Qdrant running in embedded mode (path=%s)", path)
        else:
            self._client = QdrantClient(
                url=url, api_key=api_key, prefer_grpc=False, timeout=60,
                check_compatibility=False,
            )
            logger.info("Qdrant client initialised (url=%s)", url)

    # ── Collection Management ─────────────────────────────────────────────────

    def ensure_collection(self, vector_size: int = VECTOR_SIZE) -> None:
        """
        Create the multi-vector collection if it does not already exist.
        `vector_size` is kept as a parameter for API compatibility but the
        authoritative size is always VECTOR_SIZE (386).
        """
        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection_name in existing:
            logger.info(
                "Collection '%s' already exists — skipping creation.",
                self.collection_name,
            )
            return

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=_NAMED_VECTORS_CONFIG,
        )
        logger.info(
            "Collection '%s' created with named vectors (title/description/tags, dim=%d, cosine).",
            self.collection_name,
            VECTOR_SIZE,
        )

    def collection_info(self) -> Dict[str, Any]:
        info = self._client.get_collection(self.collection_name)
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
        vectors: np.ndarray,          # legacy single-vector path (N, 386)
        batch_size: int = 64,
        *,
        title_vectors: Optional[np.ndarray] = None,
        description_vectors: Optional[np.ndarray] = None,
        tags_vectors: Optional[np.ndarray] = None,
    ) -> int:
        """
        Upsert article payloads + named vectors to Qdrant.

        Preferred call: pass title_vectors, description_vectors, tags_vectors
        (each shape (N, 386)).  If only `vectors` is given, it is used for all
        three named vectors (backward-compatible convenience).

        articles: list of dicts with keys:
            article_id, title, summary, tags (List[str]), timestamp
        """
        assert len(articles) == len(vectors), "articles and vectors must match"
        n = len(articles)

        # Fall back to `vectors` for any missing named vector
        tv  = title_vectors       if title_vectors       is not None else vectors
        dv  = description_vectors if description_vectors is not None else vectors
        tagv = tags_vectors       if tags_vectors        is not None else vectors

        assert len(tv)   == n, "title_vectors length mismatch"
        assert len(dv)   == n, "description_vectors length mismatch"
        assert len(tagv) == n, "tags_vectors length mismatch"

        total_upserted = 0

        for i in range(0, n, batch_size):
            batch_arts  = articles[i : i + batch_size]
            batch_tv    = tv[i   : i + batch_size]
            batch_dv    = dv[i   : i + batch_size]
            batch_tagv  = tagv[i : i + batch_size]

            points = [
                PointStruct(
                    id=_article_id_to_point_id(a["article_id"]),
                    vector={
                        "title":       batch_tv[j].tolist(),
                        "description": batch_dv[j].tolist(),
                        "tags":        batch_tagv[j].tolist(),
                    },
                    payload={
                        "article_id": a["article_id"],
                        "title":      a["title"],
                        "summary":    a["summary"],
                        "tags":       a["tags"],
                        "timestamp":  a.get("timestamp"),
                    },
                )
                for j, a in enumerate(batch_arts)
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
        vector: np.ndarray,                    # used for all three if named not given
        *,
        title_vector: Optional[np.ndarray] = None,
        description_vector: Optional[np.ndarray] = None,
        tags_vector: Optional[np.ndarray] = None,
    ) -> None:
        """Upsert a single article with named vectors."""
        tv   = title_vector       if title_vector       is not None else vector
        dv   = description_vector if description_vector is not None else vector
        tagv = tags_vector        if tags_vector        is not None else vector

        point = PointStruct(
            id=_article_id_to_point_id(article_id),
            vector={
                "title":       tv.tolist(),
                "description": dv.tolist(),
                "tags":        tagv.tolist(),
            },
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
        vector_name: VectorField = "title",
    ) -> List[ScoredPoint]:
        """
        Retrieve the top-k most similar articles from Qdrant.

        query_vector : 386-dim float32 array
        vector_name  : which named vector to search against
                       ("title" | "description" | "tags")
        categories   : optional tag filter (MatchAny)
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
            using=vector_name,          # ← named vector selector
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=True,          # needed for re-ranker
        )
        results = response.points
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Qdrant search [%s] returned %d results in %.1f ms",
            vector_name, len(results), elapsed_ms,
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
    # ── Latest News ───────────────────────────────────────────────────────────
    
    def get_latest(self, limit: int = 50, with_vectors: bool = False) -> List[Any]:
        """Fetch the most recently indexed news articles."""
        from qdrant_client.models import OrderBy, Direction

        try:
            # Attempt to use the high-performance OrderBy API (Qdrant 1.10+)
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=None, # Match all points
                limit=limit,
                order_by=OrderBy(
                    key="timestamp",
                    direction=Direction.DESC
                ),
                with_payload=True,
                with_vectors=with_vectors
            )
            return response.points
        except Exception as exc:
            logger.warning("Qdrant OrderBy failed (likely old version), falling back to scroll: %s", exc)
            # Fallback: Scroll a larger chunk and sort in memory
            # This is acceptable for "Latest News" which usually only needs ~50 items
            points, _ = self._client.scroll(
                collection_name=self.collection_name,
                limit=min(limit * 2, 100),
                with_payload=True,
                with_vectors=with_vectors
            )
            
            def get_ts(p):
                ts = (p.payload or {}).get("timestamp")
                if ts is None: return 0
                try: return int(ts)
                except: return 0

            sorted_points = sorted(points, key=get_ts, reverse=True)
            return sorted_points[:limit]
