"""
app/models/nn_ranker.py
───────────────────────
PyTorch neural network re-ranker.

Architecture
────────────
  Input  : concat(user_embedding [D], news_embedding [D])  → [2D]
  Layer 1: Linear(2D → 512)  + LayerNorm + ReLU + Dropout(0.3)
  Layer 2: Linear(512 → 128) + LayerNorm + ReLU + Dropout(0.2)
  Layer 3: Linear(128 → 32)  + ReLU
  Output : Linear(32 → 1)    + Sigmoid   → score ∈ [0, 1]

Usage
─────
  model = NewsRanker(embedding_dim=1024)
  scores = model(user_emb_batch, news_emb_batch)   # shape: (B, 1)
"""

import torch
import torch.nn as nn


class NewsRanker(nn.Module):
    """
    Feed-forward relevance scorer.

    Parameters
    ----------
    embedding_dim : int
        Dimension of a single embedding (e.g. 1024 for BGE-m3).
        The network receives 2 × embedding_dim concatenated inputs.
    """

    def __init__(self, embedding_dim: int = 1024):
        super().__init__()
        input_dim = embedding_dim * 2  # user + news concatenated

        self.net = nn.Sequential(
            # ── Block 1 ──────────────────────────────────────────────────────
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            # ── Block 2 ──────────────────────────────────────────────────────
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            # ── Block 3 ──────────────────────────────────────────────────────
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),

            # ── Output ───────────────────────────────────────────────────────
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Initialise weights with Xavier uniform for stable warm start
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        user_emb: torch.Tensor,   # (B, D) or (D,)
        news_emb: torch.Tensor,   # (B, D) or (D,)
    ) -> torch.Tensor:            # (B, 1)
        # Allow single-pair inference without explicit batch dim
        if user_emb.dim() == 1:
            user_emb = user_emb.unsqueeze(0)
        if news_emb.dim() == 1:
            news_emb = news_emb.unsqueeze(0)

        # Broadcast user embedding to match news batch size
        if user_emb.shape[0] == 1 and news_emb.shape[0] > 1:
            user_emb = user_emb.expand(news_emb.shape[0], -1)

        combined = torch.cat([user_emb, news_emb], dim=-1)  # (B, 2D)
        return self.net(combined)                            # (B, 1)
