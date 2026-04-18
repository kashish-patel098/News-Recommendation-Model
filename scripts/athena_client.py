"""
scripts/athena_client.py
─────────────────────────
Thin wrapper around AWS Athena + Glue/Iceberg for the news recommendation
system.

Environment variables required (set in .env or EC2 IAM role):
    AWS_REGION              e.g. us-east-1
    AWS_ACCESS_KEY_ID       (not needed if using IAM instance role)
    AWS_SECRET_ACCESS_KEY   (not needed if using IAM instance role)
    ATHENA_DATABASE         Glue database name that exposes the Iceberg table
    ATHENA_TABLE            Iceberg table name (e.g. news_articles)
    ATHENA_OUTPUT_BUCKET    s3://your-bucket/athena-results/

Usage
─────
    from scripts.athena_client import AthenaClient

    client = AthenaClient()

    # Fetch articles published after a given Unix-ms timestamp
    df = client.fetch_new_articles(since_unix_ms=1_700_000_000_000)
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd

logger = logging.getLogger(__name__)

# ── Defaults (override via .env) ──────────────────────────────────────────────
AWS_REGION            = os.getenv("AWS_REGION",           "us-east-1")
ATHENA_DATABASE       = os.getenv("ATHENA_DATABASE",      "news_db")
ATHENA_TABLE          = os.getenv("ATHENA_TABLE",         "news_articles")
ATHENA_OUTPUT_BUCKET  = os.getenv("ATHENA_OUTPUT_BUCKET", "s3://your-bucket/athena-results/")


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
        Fetch articles from the Iceberg table that were published AFTER
        `since_unix_ms` (milliseconds epoch).

        Columns expected in the Iceberg table:
            id, published_time, title, introductory_paragraph,
            descriptive_paragraph, historical_context, economyimpact,
            impact_matrix, perception_lines, tags, ai_image_prompt,
            processed_at, published_time_unix

        Returns a DataFrame with those columns (string-typed).
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        sql = f"""
            SELECT
                CAST(id AS VARCHAR)                      AS id,
                CAST(published_time AS VARCHAR)          AS published_time,
                CAST(title AS VARCHAR)                   AS title,
                CAST(introductory_paragraph AS VARCHAR)  AS introductory_paragraph,
                CAST(descriptive_paragraph AS VARCHAR)   AS descriptive_paragraph,
                CAST(historical_context AS VARCHAR)      AS historical_context,
                CAST(economyimpact AS VARCHAR)           AS economyimpact,
                CAST(impact_matrix AS VARCHAR)           AS impact_matrix,
                CAST(perception_lines AS VARCHAR)        AS perception_lines,
                CAST(tags AS VARCHAR)                    AS tags,
                CAST(ai_image_prompt AS VARCHAR)         AS ai_image_prompt,
                CAST(processed_at AS VARCHAR)            AS processed_at,
                CAST(published_time_unix AS VARCHAR)     AS published_time_unix
            FROM {self.table}
            WHERE published_time_unix > {since_unix_ms}
            ORDER BY published_time_unix ASC
            {limit_clause}
        """
        return self.run_query(sql)

    def fetch_all_articles(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch ALL articles (used for initial bulk ingestion if needed).
        Prefer `fetch_new_articles` for incremental runs.
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        sql = f"""
            SELECT
                CAST(id AS VARCHAR)                      AS id,
                CAST(published_time AS VARCHAR)          AS published_time,
                CAST(title AS VARCHAR)                   AS title,
                CAST(introductory_paragraph AS VARCHAR)  AS introductory_paragraph,
                CAST(descriptive_paragraph AS VARCHAR)   AS descriptive_paragraph,
                CAST(historical_context AS VARCHAR)      AS historical_context,
                CAST(economyimpact AS VARCHAR)           AS economyimpact,
                CAST(impact_matrix AS VARCHAR)           AS impact_matrix,
                CAST(perception_lines AS VARCHAR)        AS perception_lines,
                CAST(tags AS VARCHAR)                    AS tags,
                CAST(ai_image_prompt AS VARCHAR)         AS ai_image_prompt,
                CAST(processed_at AS VARCHAR)            AS processed_at,
                CAST(published_time_unix AS VARCHAR)     AS published_time_unix
            FROM {self.table}
            ORDER BY published_time_unix ASC
            {limit_clause}
        """
        return self.run_query(sql)

    def get_max_published_time_unix(self) -> Optional[int]:
        """Return the maximum published_time_unix currently in the Iceberg table."""
        sql = f"SELECT CAST(MAX(published_time_unix) AS VARCHAR) AS max_ts FROM {self.table}"
        df  = self.run_query(sql)
        if df.empty or not df["max_ts"].iloc[0]:
            return None
        try:
            return int(float(df["max_ts"].iloc[0]))
        except (ValueError, TypeError):
            return None
