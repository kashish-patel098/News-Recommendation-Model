"""
app/models/schemas.py
─────────────────────
Pydantic v2 request / response schemas for the recommendation API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    clicked_news: str = Field(
        ...,
        description="Full text (title + body) of the article the user clicked",
        min_length=1,
    )
    interests: str = Field(
        ...,
        description="Free-text description of the user's interests",
        min_length=1,
    )
    categories: List[str] = Field(
        default_factory=list,
        description="Preferred news categories, e.g. ['finance', 'tech']",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "user_42",
                "clicked_news": (
                    "Crude oil prices surge to $112 amid escalating "
                    "Israel-Iran tensions in the Gulf region."
                ),
                "interests": "global energy markets and geopolitics",
                "categories": ["ENERGY_STOCKS", "IN Economy"],
            }
        }
    }


class PortfolioRecommendRequest(BaseModel):
    """
    Recommend news based on a user's financial portfolio.
    Pass the raw AA (Account Aggregator) portfolio JSON as `portfolio`.
    Supported sections: DEPOSIT_V2, EQUITIES, MUTUALFUNDS, SIP,
                        REIT, INVIT, INSURANCE_POLICIES
    """
    user_id: str = Field(..., description="Unique user identifier")
    portfolio: Dict[str, Any] = Field(
        ...,
        description=(
            "Financial portfolio data from Account Aggregator. "
            "Supports EQUITIES, MUTUALFUNDS, SIP, REIT, INVIT, "
            "DEPOSIT_V2, INSURANCE_POLICIES sections."
        ),
    )
    interests: str = Field(
        default="",
        description="Optional extra free-text interests to blend with portfolio context",
    )
    categories: List[str] = Field(
        default_factory=list,
        description="Optional preferred news categories for Qdrant pre-filter",
    )


# ── Response ──────────────────────────────────────────────────────────────────

class NewsItem(BaseModel):
    article_id: str = Field(..., description="Unique article identifier")
    title: str = Field(..., description="Article headline")
    summary: str = Field(..., description="Introductory paragraph of the article")
    category: List[str] = Field(
        default_factory=list, description="Tags / category labels"
    )
    timestamp: Optional[int] = Field(
        None, description="Published time as Unix timestamp (ms)"
    )
    score: float = Field(..., description="Relevance score from neural re-ranker [0, 1]")
    full_article: Optional[Dict[str, Any]] = Field(
        None, description="Complete article data fetched from local store"
    )


class RecommendResponse(BaseModel):
    user_id: str
    total: int = Field(..., description="Number of recommendations returned")
    recommendations: List[NewsItem]

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "user_42",
                "total": 50,
                "recommendations": [
                    {
                        "article_id": "1773906540622",
                        "title": "Crude oil surges 5% amid Iran-Saudi tensions",
                        "summary": "Oil prices spiked to $112 per barrel...",
                        "category": ["ENERGY_STOCKS", "IN Economy"],
                        "timestamp": 1773906540000,
                        "score": 0.9312,
                    }
                ],
            }
        }
    }


# ── Health Check ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    loading: bool = False
    qdrant_connected: Optional[bool] = None
    sqlite_articles: Optional[int] = None
    embedding_model: Optional[str] = None
