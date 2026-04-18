"""
scripts/athena_client.py
─────────────────────────
Thin wrapper around AWS Athena + Glue/Iceberg for the news recommendation
system.

Iceberg table schema (curated_news_iceberg):
    id                 bigint
    published_time     timestamp
    title              string
    introductory_paragraph  string
    descriptive_paragraph   string
    historical_context      string
    economyimpact      struct<score: int, reason: string>
    impact_matrix      array<struct<type: string, entity: array<struct<...>>>>
    perception_lines   array<struct<tag: string, text: string>>
    tags               array<string>
    ai_image_prompt    string
    processed_at       string
    published_time_unix bigint

Note: struct/array columns cannot be CAST directly to VARCHAR in Athena.
      Use  json_format(cast(col AS JSON))  instead.

Environment variables (set in .env or via EC2 IAM role):
    AWS_REGION              e.g. us-east-1
    ATHENA_DATABASE         Glue database name  (e.g. news-iceberg-db)
    ATHENA_TABLE            Table name          (e.g. curated_news_iceberg)
    ATHENA_OUTPUT_BUCKET    s3://bucket/athena-results/
"""

import logging
import os
import time
from typing import Optional

import boto3
import pandas as pd

logger = logging.getLogger(__name__)

# ── Defaults (override via .env) ──────────────────────────────────────────────
AWS_REGION            = os.getenv("AWS_REGION",           "us-east-1")
ATHENA_DATABASE       = os.getenv("ATHENA_DATABASE",      "news-iceberg-db")
ATHENA_TABLE          = os.getenv("ATHENA_TABLE",         "curated_news_iceberg")
ATHENA_OUTPUT_BUCKET  = os.getenv("ATHENA_OUTPUT_BUCKET", "s3://your-bucket/athena-results/")

# ── Column projection ─────────────────────────────────────────────────────────
# Scalar columns: CAST(col AS VARCHAR)
# Struct / array columns: json_format(cast(col AS JSON))
_SELECT_COLUMNS = """
    CAST(id               AS VARCHAR)               AS id,
    CAST(published_time   AS VARCHAR)               AS published_time,
    CAST(title            AS VARCHAR)               AS title,
    CAST(introductory_paragraph AS VARCHAR)         AS introductory_paragraph,
    CAST(descriptive_paragraph  AS VARCHAR)         AS descriptive_paragraph,
    CAST(historical_context     AS VARCHAR)         AS historical_context,
    json_format(cast(economyimpact   AS JSON))      AS economyimpact,
    json_format(cast(impact_matrix   AS JSON))      AS impact_matrix,
    json_format(cast(perception_lines AS JSON))     AS perception_lines,
    json_format(cast(tags            AS JSON))      AS tags,
    CAST(ai_image_prompt  AS VARCHAR)               AS ai_image_prompt,
    CAST(processed_at     AS VARCHAR)               AS processed_at,
    CAST(published_time_unix AS VARCHAR)            AS published_time_unix
""".strip()


class AthenaClient:
    """
    Execute SQL against an AWS Glue/Iceberg table via Athena and return
    results as a pandas DataFrame.
    """

    def __init__(
        self,
        region: str = AWS_REGION,
        database: str = ATHENA_DATABASE,
        table: str = ATHENA_TABLE,
        output_location: str = ATHENA_OUTPUT_BUCKET,
    ) -> None:
        self.database        = database
        self.table           = table
        self.output_location = output_location
        self._client = boto3.client("athena", region_name=region)
        logger.info(
            "AthenaClient ready (database=%s, table=%s, output=%s)",
            database, table, output_location,
        )

    # ── Core query runner ─────────────────────────────────────────────────────

    def run_query(self, sql: str, timeout_s: int = 300) -> pd.DataFrame:
        """
        Execute an Athena SQL query and return results as a DataFrame.
        Blocks until the query finishes or `timeout_s` seconds elapse.
        """
        logger.info("Submitting Athena query:\n%s", sql)

        resp = self._client.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": self.database},
            ResultConfiguration={"OutputLocation": self.output_location},
        )
        execution_id = resp["QueryExecutionId"]
        logger.info("Athena QueryExecutionId: %s", execution_id)

        # Poll for completion
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status_resp = self._client.get_query_execution(
                QueryExecutionId=execution_id
            )
            state = status_resp["QueryExecution"]["Status"]["State"]

            if state == "SUCCEEDED":
                break
            elif state in ("FAILED", "CANCELLED"):
                reason = status_resp["QueryExecution"]["Status"].get(
                    "StateChangeReason", "unknown"
                )
                raise RuntimeError(
                    f"Athena query {execution_id} {state}: {reason}"
                )
            else:
                logger.debug("Athena state: %s — waiting …", state)
                time.sleep(2)
        else:
            raise TimeoutError(
                f"Athena query {execution_id} did not finish within {timeout_s}s"
            )

        # Fetch paginated results
        paginator = self._client.get_paginator("get_query_results")
        pages     = paginator.paginate(QueryExecutionId=execution_id)

        rows   = []
        header = []
        for page_num, page in enumerate(pages):
            result_rows = page["ResultSet"]["Rows"]
            if page_num == 0 and result_rows:
                header = [c["VarCharValue"] for c in result_rows[0]["Data"]]
                result_rows = result_rows[1:]   # skip column header row
            for row in result_rows:
                rows.append([col.get("VarCharValue", "") for col in row["Data"]])

        df = pd.DataFrame(rows, columns=header) if rows else pd.DataFrame(columns=header)
        logger.info("Athena query returned %d rows.", len(df))
        return df

    # ── Domain queries ────────────────────────────────────────────────────────

    def fetch_new_articles(
        self,
        since_unix_ms: int,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch articles published AFTER `since_unix_ms` (Unix seconds or ms).
        Struct/array columns are serialised to JSON strings automatically.
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        sql = f"""
            SELECT {_SELECT_COLUMNS}
            FROM {self.table}
            WHERE published_time_unix > {since_unix_ms}
            ORDER BY published_time_unix ASC
            {limit_clause}
        """
        return self.run_query(sql)

    def fetch_all_articles(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch ALL articles from the Iceberg table.
        Used for first-time / full retraining.
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        sql = f"""
            SELECT {_SELECT_COLUMNS}
            FROM {self.table}
            ORDER BY published_time_unix ASC
            {limit_clause}
        """
        return self.run_query(sql)

    def get_article_by_id(self, article_id: str) -> Optional[dict]:
        """Fetch a single article by ID. Returns None if not found."""
        sql = f"""
            SELECT {_SELECT_COLUMNS}
            FROM {self.table}
            WHERE CAST(id AS VARCHAR) = '{article_id}'
            LIMIT 1
        """
        df = self.run_query(sql)
        return df.iloc[0].to_dict() if not df.empty else None

    def get_articles_by_ids(self, ids: list) -> dict:
        """
        Fetch multiple articles by ID in one query.
        Returns dict keyed by id string.
        """
        if not ids:
            return {}
        id_list = ", ".join(f"'{i}'" for i in ids)
        sql = f"""
            SELECT {_SELECT_COLUMNS}
            FROM {self.table}
            WHERE CAST(id AS VARCHAR) IN ({id_list})
        """
        df = self.run_query(sql)
        return {row["id"]: row for row in df.to_dict(orient="records")}

    def get_max_published_time_unix(self) -> Optional[int]:
        """Return the maximum published_time_unix in the Iceberg table."""
        sql = f"SELECT CAST(MAX(published_time_unix) AS VARCHAR) AS max_ts FROM {self.table}"
        df  = self.run_query(sql)
        if df.empty or not df["max_ts"].iloc[0]:
            return None
        try:
            return int(float(df["max_ts"].iloc[0]))
        except (ValueError, TypeError):
            return None

    def count(self) -> int:
        """Approximate total article count."""
        sql = f"SELECT COUNT(*) AS cnt FROM {self.table}"
        df  = self.run_query(sql)
        try:
            return int(df["cnt"].iloc[0]) if not df.empty else 0
        except Exception:
            return -1
