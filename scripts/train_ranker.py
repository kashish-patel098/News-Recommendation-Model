r"""
scripts/train_ranker.py
------------------------
Optional offline training script for the NewsRanker.

Why train?
----------
  The ranker is initialised with Xavier weights (stable, but not personalised).
  This script generates SYNTHETIC positive/negative training pairs from the
  dataset and fine-tunes the model so it starts with meaningful priors.

  Synthetic training logic
  ------------------------
  - Positive pair: user embedding (from article tags/title) paired with
    the SAME article embedding -> label = 1.0
  - Negative pair: paired with a RANDOM article from a different category -> label = 0.0

  In production replace this with real click-through data.

Usage
-----
  .venv\Scripts\Activate.ps1

  # Quick training (default 2 epochs, 2000 pairs, embed_batch=8 for low RAM)
  python scripts/train_ranker.py

  # Custom
  python scripts/train_ranker.py --pairs 5000 --epochs 5 --embed-batch 16
"""

import argparse
import logging
import os
import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from app.models.nn_ranker import NewsRanker
from app.services.embedding_service import EmbeddingService
from app.utils.text_utils import build_news_embedding_text, parse_tags
from local_store.news_store import NewsStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_ranker")

MODEL_NAME     = os.getenv("MODEL_NAME",           "BAAI/bge-m3")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH",       str(ROOT / "local_store/news.db"))
WEIGHTS_PATH   = os.getenv("RANKER_WEIGHTS_PATH",  str(ROOT / "ranker_weights.pt"))
EMBEDDING_DIM  = 1024


# -- Embedding with progress bar -----------------------------------------------

def embed_articles_with_progress(
    articles: list,
    embed_svc: EmbeddingService,
    embed_batch: int,
) -> np.ndarray:
    """
    Embed articles in small batches, showing a tqdm bar per batch.
    embed_batch=8 keeps RAM below ~2GB on CPU.
    """
    texts = [
        build_news_embedding_text(
            title=a.get("title", ""),
            intro=a.get("introductory_paragraph", ""),
        )
        for a in articles
    ]

    total_batches = (len(texts) + embed_batch - 1) // embed_batch
    all_vecs = []

    pbar = tqdm(
        total=len(texts),
        desc="Embedding articles",
        unit="article",
        dynamic_ncols=True,
    )

    for i in range(0, len(texts), embed_batch):
        chunk = texts[i : i + embed_batch]
        vecs  = embed_svc.encode_batch(chunk, batch_size=len(chunk))
        all_vecs.append(vecs)
        pbar.update(len(chunk))
        pbar.set_postfix(
            batch=f"{i // embed_batch + 1}/{total_batches}",
            dim=vecs.shape[1],
        )

    pbar.close()
    return np.vstack(all_vecs).astype(np.float32)


# -- Build synthetic training pairs --------------------------------------------

def build_training_pairs(
    articles: list,
    all_vecs: np.ndarray,
    num_pairs: int,
) -> tuple:
    """
    Build (user_emb, news_emb, label) arrays from pre-computed vectors.
    Returns three arrays of shape (num_pairs, D), (num_pairs, D), (num_pairs,).
    """
    n = len(articles)
    assert len(all_vecs) == n

    # Pre-build tag lookup for efficient negative mining
    tag_to_idxs: dict = {}
    for i, a in enumerate(articles):
        for tag in parse_tags(a.get("tags", "")):
            tag_to_idxs.setdefault(tag, []).append(i)

    user_embs = []
    news_embs = []
    labels    = []
    generated = 0

    pbar = tqdm(total=num_pairs, desc="Building pairs", unit="pair", dynamic_ncols=True)

    while generated < num_pairs:
        pos_idx = random.randint(0, n - 1)

        # --- Positive pair ---
        user_embs.append(all_vecs[pos_idx])
        news_embs.append(all_vecs[pos_idx])
        labels.append(1.0)
        generated += 1
        pbar.update(1)

        if generated >= num_pairs:
            break

        # --- Negative pair (different category) ---
        pos_tags = parse_tags(articles[pos_idx].get("tags", ""))
        neg_idx  = random.randint(0, n - 1)
        if pos_tags:
            for _ in range(15):          # try 15 times to get a different category
                cand = random.randint(0, n - 1)
                if not set(pos_tags) & set(parse_tags(articles[cand].get("tags", ""))):
                    neg_idx = cand
                    break

        user_embs.append(all_vecs[pos_idx])
        news_embs.append(all_vecs[neg_idx])
        labels.append(0.0)
        generated += 1
        pbar.update(1)

    pbar.close()

    return (
        np.stack(user_embs[:num_pairs]),
        np.stack(news_embs[:num_pairs]),
        np.array(labels[:num_pairs], dtype=np.float32),
    )


# -- Training ------------------------------------------------------------------

def train(
    num_pairs: int,
    epochs: int,
    lr: float,
    train_batch: int,
    embed_batch: int,
    max_articles: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    # -- Load articles from SQLite ---------------------------------------------
    news_store = NewsStore(db_path=SQLITE_DB_PATH)
    total = news_store.count()
    if total == 0:
        logger.error("No articles in SQLite. Run scripts/ingest_full_dataset.py first.")
        sys.exit(1)
    logger.info("SQLite has %d articles.", total)

    conn = sqlite3.connect(str(Path(SQLITE_DB_PATH)))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, title, introductory_paragraph, tags FROM news"
    ).fetchall()
    conn.close()

    articles = [dict(r) for r in rows]

    # Limit articles to reduce embedding time on CPU
    if max_articles and len(articles) > max_articles:
        random.shuffle(articles)
        articles = articles[:max_articles]
        logger.info(
            "Using %d articles for training (--max-articles limit).", len(articles)
        )

    # -- Load embedding model --------------------------------------------------
    logger.info("Loading embedding model '%s' ...", MODEL_NAME)
    embed_svc = EmbeddingService(model_name=MODEL_NAME, cache_max_size=50)
    logger.info("Embedding model ready.")

    # -- Embed articles (with visible progress bar) ----------------------------
    logger.info(
        "Embedding %d articles in batches of %d (this is the slow part on CPU)...",
        len(articles),
        embed_batch,
    )
    all_vecs = embed_articles_with_progress(articles, embed_svc, embed_batch)
    logger.info("All %d articles embedded. Shape: %s", len(articles), all_vecs.shape)

    # -- Build training pairs --------------------------------------------------
    logger.info("Building %d training pairs...", num_pairs)
    random.shuffle(articles)
    user_arr, news_arr, label_arr = build_training_pairs(articles, all_vecs, num_pairs)

    # -- DataLoader ------------------------------------------------------------
    dataset = TensorDataset(
        torch.tensor(user_arr,  dtype=torch.float32),
        torch.tensor(news_arr,  dtype=torch.float32),
        torch.tensor(label_arr, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=train_batch, shuffle=True)

    # -- Model -----------------------------------------------------------------
    model = NewsRanker(embedding_dim=EMBEDDING_DIM).to(device)
    weights_path = Path(WEIGHTS_PATH)
    if weights_path.exists():
        state = torch.load(str(weights_path), map_location=device, weights_only=True)
        model.load_state_dict(state)
        logger.info("Loaded existing weights for fine-tuning.")
    else:
        logger.info("No existing weights — training from Xavier init.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # -- Training loop ---------------------------------------------------------
    logger.info("Training: %d epochs | %d pairs | lr=%.0e", epochs, num_pairs, lr)
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", dynamic_ncols=True)
        for user_b, news_b, label_b in pbar:
            user_b  = user_b.to(device)
            news_b  = news_b.to(device)
            label_b = label_b.to(device)

            optimizer.zero_grad()
            preds = model(user_b, news_b)
            loss  = criterion(preds, label_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(label_b)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(dataset)
        logger.info("Epoch %d/%d complete — avg loss: %.4f", epoch, epochs, avg_loss)

    # -- Save ------------------------------------------------------------------
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(weights_path))
    logger.info("Ranker weights saved to %s", weights_path)
    logger.info("=== Training complete ===")


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train the NewsRanker on synthetic pairs.\n"
            "Tip: keep --embed-batch low (8-16) to avoid RAM crashes on CPU."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pairs", type=int, default=2000,
        help="Number of training pairs (default: 2000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=2,
        help="Training epochs (default: 2)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--train-batch", type=int, default=128,
        help="Neural net training batch size (default: 128)",
    )
    parser.add_argument(
        "--embed-batch", type=int, default=8,
        help="Embedding batch size per BGE-m3 forward pass — lower = less RAM (default: 8)",
    )
    parser.add_argument(
        "--max-articles", type=int, default=500,
        help=(
            "Max articles to embed for training (default: 500). "
            "Lower = faster. Increase once ingestion is near complete."
        ),
    )
    args = parser.parse_args()

    train(
        num_pairs=args.pairs,
        epochs=args.epochs,
        lr=args.lr,
        train_batch=args.train_batch,
        embed_batch=args.embed_batch,
        max_articles=args.max_articles,
    )
