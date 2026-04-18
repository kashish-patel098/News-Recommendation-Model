"""
app/services/iceberg_service.py
────────────────────────────────
Athena-backed article store — replaces PostgreSQL / NewsStore.

All article content lives in the Iceberg table (curated_news_iceberg).
This service delegates all SQL to AthenaClient which handles the correct
json_format() casting for struct/array columns.

Results are cached in-process (LRU) to avoid repeated Athena calls
for the same article IDs within one API session.
"""

import logging
from typing import Any, Dict, List, Optional

from scripts.athena_client import AthenaClient

logger = logging.getLogger(__name__)


class IcebergArticleService:
    """
    Fetch full article content from Iceberg via Athena.
    Delegates all SQL to AthenaClient (which handles complex types correctly).
    """

    def __init__(self, athena: Optional[AthenaClient] = None):
        self._athena = athena or AthenaClient()
        self._cache: Dict[str, Dict[str, Any]] = {}   # simple session cache
        logger.info("IcebergArticleService ready (Athena/Iceberg backend)")

    def get_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Fetch one article by ID. Returns None if not found."""
        if article_id in self._cache:
            return self._cache[article_id]
        row = self._athena.get_article_by_id(article_id)
        if row:
            self._cache[article_id] = row
        return row

    def get_by_ids(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Batch fetch — returns dict keyed by article_id."""
        if not ids:
            return {}
        missing = [i for i in ids if i not in self._cache]
        if missing:
            fetched = self._athena.get_articles_by_ids(missing)
            self._cache.update(fetched)
        return {i: self._cache[i] for i in ids if i in self._cache}

    def count(self) -> int:
        """Approximate total article count in Iceberg (for /health)."""
        return self._athena.count()
