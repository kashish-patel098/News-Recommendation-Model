"""
app/services/iceberg_service.py
────────────────────────────────
Athena-backed article store — replaces PostgreSQL / NewsStore.

All article content lives in the Iceberg table on S3.
This service queries Athena to fetch articles by ID or in bulk.

Results are cached in-process (LRU) to avoid repeated Athena calls
for the same article IDs within one API session.
"""

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from scripts.athena_client import AthenaClient

logger = logging.getLogger(__name__)


class IcebergArticleService:
    """
    Fetch full article content from Iceberg via Athena.
    Replaces the PostgreSQL-backed NewsStore.
    """

    def __init__(self, athena: Optional[AthenaClient] = None):
        self._athena = athena or AthenaClient()
        self._cache: Dict[str, Dict[str, Any]] = {}   # simple session cache
        logger.info("IcebergArticleService ready (Athena/Iceberg backend)")

    # ── Single article ────────────────────────────────────────────────────────

    def get_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Fetch one article by ID. Returns None if not found."""
        if article_id in self._cache:
            return self._cache[article_id]

        df = self._athena.run_query(
            f"""
            SELECT
                CAST(id AS VARCHAR)                      AS id,
                CAST(published_time AS VARCHAR)          AS published_time,
                CAST(published_time_unix AS VARCHAR)     AS published_time_unix,
                CAST(title AS VARCHAR)                   AS title,
                CAST(introductory_paragraph AS VARCHAR)  AS introductory_paragraph,
                CAST(descriptive_paragraph AS VARCHAR)   AS descriptive_paragraph,
                CAST(historical_context AS VARCHAR)      AS historical_context,
                CAST(economyimpact AS VARCHAR)           AS economyimpact,
                CAST(impact_matrix AS VARCHAR)           AS impact_matrix,
                CAST(perception_lines AS VARCHAR)        AS perception_lines,
                CAST(tags AS VARCHAR)                    AS tags,
                CAST(ai_image_prompt AS VARCHAR)         AS ai_image_prompt,
                CAST(processed_at AS VARCHAR)            AS processed_at
            FROM {self._athena.table}
            WHERE CAST(id AS VARCHAR) = '{article_id}'
            LIMIT 1
            """
        )
        if df.empty:
            return None
        row = df.iloc[0].to_dict()
        self._cache[article_id] = row
        return row

    # ── Batch fetch ───────────────────────────────────────────────────────────

    def get_by_ids(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch multiple articles by ID in one Athena query.
        Returns a dict keyed by article_id.
        """
        if not ids:
            return {}

        # Serve from cache first
        missing = [i for i in ids if i not in self._cache]

        if missing:
            id_list = ", ".join(f"'{i}'" for i in missing)
            df = self._athena.run_query(
                f"""
                SELECT
                    CAST(id AS VARCHAR)                      AS id,
                    CAST(published_time AS VARCHAR)          AS published_time,
                    CAST(published_time_unix AS VARCHAR)     AS published_time_unix,
                    CAST(title AS VARCHAR)                   AS title,
                    CAST(introductory_paragraph AS VARCHAR)  AS introductory_paragraph,
                    CAST(descriptive_paragraph AS VARCHAR)   AS descriptive_paragraph,
                    CAST(historical_context AS VARCHAR)      AS historical_context,
                    CAST(economyimpact AS VARCHAR)           AS economyimpact,
                    CAST(impact_matrix AS VARCHAR)           AS impact_matrix,
                    CAST(perception_lines AS VARCHAR)        AS perception_lines,
                    CAST(tags AS VARCHAR)                    AS tags,
                    CAST(ai_image_prompt AS VARCHAR)         AS ai_image_prompt,
                    CAST(processed_at AS VARCHAR)            AS processed_at
                FROM {self._athena.table}
                WHERE CAST(id AS VARCHAR) IN ({id_list})
                """
            )
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                self._cache[row_dict["id"]] = row_dict

        return {i: self._cache[i] for i in ids if i in self._cache}

    # ── Count (used by /health) ────────────────────────────────────────────────

    def count(self) -> int:
        """Approximate total article count in Iceberg."""
        try:
            df = self._athena.run_query(
                f"SELECT COUNT(*) AS cnt FROM {self._athena.table}"
            )
            return int(df["cnt"].iloc[0]) if not df.empty else 0
        except Exception as exc:
            logger.warning("count() failed: %s", exc)
            return -1
