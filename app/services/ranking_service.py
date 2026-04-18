"""
app/services/ranking_service.py
────────────────────────────────
Neural network re-ranking pipeline.

Flow
────
  1. Receive user embedding (1-D numpy array, shape [1024])
  2. Receive candidate news from Qdrant (list of ScoredPoint, each has a
     pre-stored dense vector in the `vectors` field)
  3. Stack candidate vectors → batch tensor [N, 1024]
  4. Run NewsRanker in a single forward pass → scores [N, 1]  (batch scoring)
  5. Zip scores with payload, sort descending, return top-k NewsItem list

No gradient computation needed at inference time (torch.no_grad()).
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from qdrant_client.models import ScoredPoint

from app.models.nn_ranker import NewsRanker
from app.models.schemas import NewsItem
from app.utils.text_utils import parse_tags

logger = logging.getLogger(__name__)


class RankingService:
    """Loads and manages the PyTorch re-ranker; scores candidate articles."""

    def __init__(
        self,
        weights_path: str | Path = "ranker_weights.pt",
        embedding_dim: int = 1024,
        device: Optional[str] = None,
    ):
        self.weights_path = Path(weights_path)
        self.embedding_dim = embedding_dim
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self._load_model()
        logger.info(
            "RankingService ready on device=%s (weights=%s)",
            self.device,
            self.weights_path,
        )

    # ── Model Lifecycle ───────────────────────────────────────────────────────

    def _load_model(self) -> NewsRanker:
        model = NewsRanker(embedding_dim=self.embedding_dim).to(self.device)

        if self.weights_path.exists():
            try:
                state = torch.load(
                    str(self.weights_path),
                    map_location=self.device,
                    weights_only=True,
                )
                model.load_state_dict(state)
                logger.info("Loaded ranker weights from %s", self.weights_path)
            except RuntimeError as exc:
                # Weights were trained with a different embedding model/dimension.
                # Fall back to random init — model still works, just not yet trained
                # for this embedding space. Run monthly_train.py --all-data to retrain.
                logger.warning(
                    "Ranker weights at '%s' are incompatible with embedding_dim=%d "
                    "(likely trained with a different model). Starting with random init. "
                    "Run: python scripts/monthly_train.py --all-data\n  Detail: %s",
                    self.weights_path,
                    self.embedding_dim,
                    exc,
                )
        else:
            logger.warning(
                "No ranker weights found at '%s'. Using random init. "
                "Run scripts/monthly_train.py --all-data to train.",
                self.weights_path,
            )

        model.eval()
        return model

    def save_weights(self) -> None:
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(self.weights_path))
        logger.info("Ranker weights saved to %s", self.weights_path)

    # ── Ranking ───────────────────────────────────────────────────────────────

    def rank(
        self,
        user_embedding: np.ndarray,
        candidates: List[ScoredPoint],
        top_k: int = 50,
    ) -> List[NewsItem]:
        """
        Score all candidate articles and return sorted NewsItem list.

        Parameters
        ----------
        user_embedding : np.ndarray  shape (1024,)
        candidates     : list of Qdrant ScoredPoints with vectors attached
        top_k          : max items to return

        Returns
        -------
        List[NewsItem] sorted by score descending
        """
        if not candidates:
            return []

        t0 = time.perf_counter()

        # ── Build news embedding batch ────────────────────────────────────────
        news_vecs = []
        for point in candidates:
            if point.vector is None:
                # Fallback: zero vector (should not happen if with_vectors=True)
                vec = np.zeros(self.embedding_dim, dtype=np.float32)
            elif isinstance(point.vector, list):
                vec = np.array(point.vector, dtype=np.float32)
            else:
                vec = np.array(point.vector, dtype=np.float32)
            news_vecs.append(vec)

        news_tensor = torch.tensor(
            np.stack(news_vecs), dtype=torch.float32
        ).to(self.device)                           # (N, D)

        user_tensor = torch.tensor(
            user_embedding, dtype=torch.float32
        ).unsqueeze(0).to(self.device)              # (1, D)

        # ── Single forward pass (batch scoring) ──────────────────────────────
        with torch.no_grad():
            scores = self.model(user_tensor, news_tensor)   # (N, 1)
            scores = scores.squeeze(-1).cpu().numpy()       # (N,)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Ranked %d candidates in %.1f ms on %s",
            len(candidates),
            elapsed_ms,
            self.device,
        )

        # ── Build NewsItem list ───────────────────────────────────────────────
        ranked: List[NewsItem] = []
        for point, score in zip(candidates, scores):
            p = point.payload or {}
            tags = parse_tags(p.get("tags", []))
            ranked.append(
                NewsItem(
                    article_id=str(p.get("article_id", "")),
                    title=p.get("title", ""),
                    summary=p.get("summary", ""),
                    category=tags,
                    timestamp=p.get("timestamp"),
                    score=float(round(float(score), 6)),
                )
            )

        # Sort by score descending, take top-k
        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked[:top_k]
