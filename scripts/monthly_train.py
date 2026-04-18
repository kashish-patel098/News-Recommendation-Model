"""
scripts/monthly_train.py
─────────────────────────
Monthly incremental training of the NewsRanker.

Key properties
──────────────
  • Loads existing ranker_weights.pt if it exists (fine-tune, not re-train)
  • Pulls only articles ingested SINCE the last training run (new data)
  • Old learning is preserved — weights are updated, not replaced
  • Saves a timestamped backup before overwriting the weights file
  • Designed to be invoked by a monthly cron job on EC2

Cron entry (edit with: crontab -e)
───────────────────────────────────
  # Run at 02:00 on the 1st of every month
  0 2 1 * * /opt/news-rec/venv/bin/python /opt/news-rec/scripts/monthly_train.py \
            >> /var/log/news-rec/monthly_train.log 2>&1

Usage (manual)
──────────────
  source /opt/news-rec/venv/bin/activate
  python scripts/monthly_train.py
  python scripts/monthly_train.py --pairs 5000 --epochs 3
  python scripts/monthly_train.py --all-data      # train on entire dataset
"""

import argparse
import logging
import os
import random
import shutil
import sys
from datetime import datetime, timezone
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
from scripts.athena_client import AthenaClient

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("monthly_train")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME    = os.getenv("MODEL_NAME",          "BAAI/bge-m3")
WEIGHTS_PATH  = Path(os.getenv("RANKER_WEIGHTS_PATH", str(ROOT / "ranker_weights.pt")))
VECTOR_SIZE   = int(os.getenv("VECTOR_SIZE",    "386"))

# Watermark: tracks how many articles were seen at last training run
TRAIN_WATERMARK_FILE = ROOT / "local_store" / "train_watermark.txt"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_train_watermark() -> int:
    """Return the article count at the time of the last training run."""
    if TRAIN_WATERMARK_FILE.exists():
        try:
            val = int(TRAIN_WATERMARK_FILE.read_text().strip())
            logger.info("Train watermark: %d articles trained on previously.", val)
            return val
        except ValueError:
            pass
    return 0


def _save_train_watermark(count: int) -> None:
    TRAIN_WATERMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_WATERMARK_FILE.write_text(str(count))
    logger.info("Train watermark saved: %d", count)


def _backup_weights() -> None:
    """Create a timestamped backup of the current weights before overwriting."""
    if not WEIGHTS_PATH.exists():
        return
    ts      = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup  = WEIGHTS_PATH.with_name(f"ranker_weights_{ts}.pt")
    shutil.copy2(WEIGHTS_PATH, backup)
    logger.info("Weights backed up to %s", backup)


def embed_articles_with_progress(
    articles: list,
    embed_svc: EmbeddingService,
    embed_batch: int,
) -> np.ndarray:
    """Embed articles using title+intro text. Returns (N, VECTOR_SIZE)."""
    texts = [
        build_news_embedding_text(
            title=a.get("title", ""),
            intro=a.get("introductory_paragraph", ""),
        )
        for a in articles
    ]
    total_batches = (len(texts) + embed_batch - 1) // embed_batch
    all_vecs = []
    pbar = tqdm(total=len(texts), desc="Embedding articles", unit="article", dynamic_ncols=True)
    for i in range(0, len(texts), embed_batch):
        chunk = texts[i : i + embed_batch]
        vecs  = embed_svc.encode_batch(chunk, batch_size=len(chunk))
        all_vecs.append(vecs)
        pbar.update(len(chunk))
        pbar.set_postfix(batch=f"{i // embed_batch + 1}/{total_batches}")
    pbar.close()
    return np.vstack(all_vecs).astype(np.float32)


def build_training_pairs(
    articles: list,
    all_vecs: np.ndarray,
    num_pairs: int,
) -> tuple:
    """
    Build (user_emb, news_emb, label) synthetic pairs.
    Positive: same article → label 1.0
    Negative: different-tag article → label 0.0
    """
    n = len(articles)
    assert len(all_vecs) == n

    tag_to_idxs: dict = {}
    for i, a in enumerate(articles):
        for tag in parse_tags(a.get("tags", "")):
            tag_to_idxs.setdefault(tag, []).append(i)

    user_embs, news_embs, labels = [], [], []
    generated = 0
    pbar = tqdm(total=num_pairs, desc="Building pairs", unit="pair", dynamic_ncols=True)

    while generated < num_pairs:
        pos_idx = random.randint(0, n - 1)

        # Positive
        user_embs.append(all_vecs[pos_idx])
        news_embs.append(all_vecs[pos_idx])
        labels.append(1.0)
        generated += 1
        pbar.update(1)
        if generated >= num_pairs:
            break

        # Negative (different category)
        pos_tags = parse_tags(articles[pos_idx].get("tags", ""))
        neg_idx  = random.randint(0, n - 1)
        if pos_tags:
            for _ in range(15):
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


# ── Main training function ────────────────────────────────────────────────────

def train(
    num_pairs:   int   = 2000,
    epochs:      int   = 2,
    lr:          float = 1e-3,
    train_batch: int   = 128,
    embed_batch: int   = 8,
    max_articles: int  = 500,
    use_all_data: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)
    logger.info("  Monthly Incremental NewsRanker Training")
    logger.info("  Started: %s", datetime.now(timezone.utc).isoformat())
    logger.info("  Device : %s", device)
    logger.info("=" * 60)

    # ── Fetch articles from Athena/Iceberg ──────────────────────────────────────
    logger.info("Connecting to Athena/Iceberg …")
    athena = AthenaClient()

    prev_watermark = _load_train_watermark() if not use_all_data else 0

    if use_all_data:
        logger.info("Fetching ALL articles from Iceberg …")
        df = athena.fetch_all_articles()
    else:
        logger.info("Fetching articles newer than watermark ts=%d …", prev_watermark)
        df = athena.fetch_new_articles(since_unix_ms=prev_watermark)

    total_in_db = len(df)
    logger.info("Iceberg returned %d articles for training.", total_in_db)

    if total_in_db == 0:
        logger.info("No new articles in Iceberg since last training run. Exiting.")
        return

    articles = df.to_dict(orient="records")
    logger.info("Fetched %d articles from Iceberg for training.", len(articles))

    # Optionally cap to memory limit
    if max_articles and len(articles) > max_articles and not use_all_data:
        random.shuffle(articles)
        articles = articles[:max_articles]
        logger.info("Capped to %d articles (--max-articles).", len(articles))

    if len(articles) < 2:
        logger.error("Not enough articles to build training pairs. Exiting.")
        sys.exit(1)

    # ── Load embedding model ───────────────────────────────────────────────────
    logger.info("Loading embedding model '%s' …", MODEL_NAME)
    embed_svc = EmbeddingService(model_name=MODEL_NAME, cache_max_size=50, vector_size=VECTOR_SIZE)
    logger.info("Embedding model ready.")

    # ── Embed ──────────────────────────────────────────────────────────────────
    logger.info(
        "Embedding %d articles in batches of %d …",
        len(articles), embed_batch,
    )
    all_vecs = embed_articles_with_progress(articles, embed_svc, embed_batch)
    logger.info("Embedding complete. Shape: %s", all_vecs.shape)

    # ── Build pairs ────────────────────────────────────────────────────────────
    actual_pairs = min(num_pairs, len(articles) * 2)
    logger.info("Building %d training pairs …", actual_pairs)
    random.shuffle(articles)
    user_arr, news_arr, label_arr = build_training_pairs(articles, all_vecs, actual_pairs)

    # ── DataLoader ─────────────────────────────────────────────────────────────
    dataset = TensorDataset(
        torch.tensor(user_arr,  dtype=torch.float32),
        torch.tensor(news_arr,  dtype=torch.float32),
        torch.tensor(label_arr, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=train_batch, shuffle=True)

    # ── Load model (existing weights → incremental fine-tune) ─────────────────
    model = NewsRanker(embedding_dim=VECTOR_SIZE).to(device)
    if WEIGHTS_PATH.exists():
        state = torch.load(str(WEIGHTS_PATH), map_location=device, weights_only=True)
        model.load_state_dict(state)
        logger.info("Loaded existing weights from %s — fine-tuning.", WEIGHTS_PATH)
    else:
        logger.info("No existing weights — training from Xavier init.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ──────────────────────────────────────────────────────────
    logger.info(
        "Training: %d epochs | %d pairs | lr=%.0e | device=%s",
        epochs, actual_pairs, lr, device,
    )
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
        logger.info("Epoch %d/%d — avg loss: %.4f", epoch, epochs, avg_loss)

    # ── Save weights (with backup) ─────────────────────────────────────────────
    _backup_weights()
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(WEIGHTS_PATH))
    logger.info("Weights saved to %s", WEIGHTS_PATH)

    # ── Update watermark ───────────────────────────────────────────────────────
    _save_train_watermark(total_in_db)

    logger.info("=" * 60)
    logger.info("  Training complete: %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Monthly incremental training of the NewsRanker.\n"
            "Trains only on NEW articles (since last run).\n"
            "Old model weights are preserved and fine-tuned."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pairs",       type=int,   default=2000,  help="Training pairs (default: 2000)")
    parser.add_argument("--epochs",      type=int,   default=2,     help="Training epochs (default: 2)")
    parser.add_argument("--lr",          type=float, default=1e-3,  help="Learning rate (default: 1e-3)")
    parser.add_argument("--train-batch", type=int,   default=128,   help="NN training batch size (default: 128)")
    parser.add_argument("--embed-batch", type=int,   default=8,     help="Embedding batch size — lower = less RAM (default: 8)")
    parser.add_argument("--max-articles",type=int,   default=500,   help="Max new articles to embed (default: 500)")
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="Train on entire dataset (not just new articles). Use for first-time training.",
    )
    args = parser.parse_args()

    train(
        num_pairs    = args.pairs,
        epochs       = args.epochs,
        lr           = args.lr,
        train_batch  = args.train_batch,
        embed_batch  = args.embed_batch,
        max_articles = args.max_articles,
        use_all_data = args.all_data,
    )
