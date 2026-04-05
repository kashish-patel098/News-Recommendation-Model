"""
app/utils/text_utils.py
────────────────────────
Text preprocessing helpers shared across services.
"""

import json
import re
import html
from typing import Any, List


# ── Tag Parsing ───────────────────────────────────────────────────────────────

def parse_tags(raw: Any) -> List[str]:
    """
    Parse the `tags` field into a clean list of strings.

    The CSV field looks like either:
      • A JSON array:  '["IN Economy", "ENERGY_STOCKS"]'
      • A bracketed string: '[IN Economy, ENERGY_STOCKS]'
      • Already a Python list (when read by pandas with converters)
    """
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if t]

    if not isinstance(raw, str) or not raw.strip():
        return []

    raw = raw.strip()

    # Try JSON first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(t).strip() for t in parsed if t]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fall back: strip brackets, split by comma
    inner = re.sub(r"^\[|\]$", "", raw)
    return [t.strip() for t in inner.split(",") if t.strip()]


# ── HTML Stripping ────────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")

def strip_html(text: Any) -> str:
    """Remove HTML tags and decode HTML entities."""
    if not isinstance(text, str):
        return ""
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return " ".join(text.split())   # normalise whitespace


# ── Embedding Text Builder ────────────────────────────────────────────────────

def build_news_embedding_text(title: str, intro: str) -> str:
    """
    Combine title and introductory paragraph into the text that will be
    embedded and stored in Qdrant.  We deliberately keep this lightweight —
    the full descriptive paragraphs live in SQLite only.
    """
    title_clean = strip_html(title).strip()
    intro_clean = strip_html(intro).strip()
    return f"{title_clean}. {intro_clean}" if intro_clean else title_clean


def build_user_query_text(
    clicked_news: str,
    interests: str,
    categories: List[str],
) -> str:
    """
    Combine the user's context into a single query string for embedding.
    """
    parts = [
        strip_html(clicked_news).strip(),
        strip_html(interests).strip(),
    ]
    if categories:
        parts.append(" ".join(categories))
    return ". ".join(p for p in parts if p)


# ── Truncation ────────────────────────────────────────────────────────────────

def truncate_tokens(text: str, max_chars: int = 8192) -> str:
    """
    Naively truncate text to avoid exceeding model context limits.
    BGE-m3 supports up to 8192 tokens; ~4 chars/token is a safe heuristic.
    """
    return text[:max_chars] if len(text) > max_chars else text
