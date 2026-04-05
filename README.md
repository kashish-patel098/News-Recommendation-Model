# 📰 News Recommendation Engine

> Real-time personalised news recommendations powered by **BGE-m3** embeddings, **Qdrant** vector search, **PyTorch** neural re-ranking, and a dual **SQLite + Qdrant** storage architecture.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│  Client  ──►  POST /api/v1/recommend                       │
└─────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │  1. Build query text    │  clicked_news + interests + categories
          │     (text_utils)        │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │  2. BGE-m3 Embedding    │  EmbeddingService (LRU cached)
          │     BAAI/bge-m3         │  → 1024-dim float32 vector
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │  3. Qdrant Vector DB    │  top-50 nearest neighbours
          │     (cosine similarity) │  payload: title, summary, tags, timestamp
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │  4. PyTorch Re-Ranker   │  batch forward pass → relevance score
          │     NewsRanker          │  concat → 3 dense layers → sigmoid
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │  5. Sorted Response     │  top-50 NewsItems with score [0,1]
          └─────────────────────────┘

Storage (separate systems)
──────────────────────────
  Qdrant  → vector + {title, summary, tags, timestamp}  (fast search)
  SQLite  → ALL columns from CSV                         (full content)
```

---

## Storage Design

| Store | File | Purpose |
|---|---|---|
| **Qdrant** | `localhost:6333` | Vector similarity search — stores lightweight payload only |
| **SQLite** | `local_store/news.db` | Full article store — every CSV column preserved |

> When you need the full article body (descriptive_paragraph, impact_matrix, etc.), call `GET /api/v1/article/{id}` — it fetches from SQLite, not Qdrant.

---

## Project Structure

```
News Recommendation/
├── app/
│   ├── main.py                       # FastAPI entrypoint
│   ├── api/routes.py                 # POST /recommend, GET /health, GET /article/{id}
│   ├── models/
│   │   ├── nn_ranker.py              # PyTorch NewsRanker
│   │   └── schemas.py                # Pydantic request/response schemas
│   ├── services/
│   │   ├── embedding_service.py      # BGE-m3 + LRU cache
│   │   ├── qdrant_service.py         # Qdrant client
│   │   └── ranking_service.py        # Neural re-ranking pipeline
│   └── utils/text_utils.py           # HTML stripping, tag parsing, text builders
├── local_store/
│   ├── news_store.py                 # SQLite full-article manager
│   └── news.db                       # Created on first run
├── scripts/
│   ├── ingest_full_dataset.py        # ONE-TIME: CSV → SQLite + Qdrant
│   ├── ingest_latest_news.py         # ONGOING: new articles → SQLite + Qdrant
│   └── train_ranker.py               # Optional: pre-train neural ranker
├── news_dataset.csv                  # Source dataset (~18,895 articles)
├── ranker_weights.pt                 # Created after training (auto-saved)
├── requirements.txt
├── .env.example
├── setup_venv.ps1                    # One-click setup
└── README.md
```

---

## Quick Start

### Step 1 — Setup Environment

```powershell
# In the project directory
.\setup_venv.ps1
```

This creates `.venv` and installs all dependencies. Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 2 — Configure

```powershell
# .env is created automatically by setup_venv.ps1
# Edit if needed (Qdrant host, model name, etc.)
notepad .env
```

### Step 3 — Start Qdrant

```powershell
docker run -d -p 6333:6333 -p 6334:6334 `
  -v "${PWD}/qdrant_storage:/qdrant/storage" `
  qdrant/qdrant
```

Qdrant dashboard: http://localhost:6333/dashboard

### Step 4 — Ingest Dataset (one-time, ~30–60 min on CPU)

```powershell
python scripts/ingest_full_dataset.py
```

Options:
```powershell
python scripts/ingest_full_dataset.py --batch 64   # larger batch for faster GPU
python scripts/ingest_full_dataset.py --dry-run    # count rows, no writes
```

Progress is shown with a tqdm bar. Ingestion is **idempotent** — safe to restart after a failure.

### Step 5 — (Optional) Pre-train Neural Ranker

```powershell
python scripts/train_ranker.py
python scripts/train_ranker.py --pairs 20000 --epochs 10
```

### Step 6 — Start API Server

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs

---

## API Reference

### `POST /api/v1/recommend`

```json
{
  "user_id": "user_42",
  "clicked_news": "Crude oil prices surge to $112 amid escalating tensions",
  "interests": "global energy markets and geopolitics",
  "categories": ["ENERGY_STOCKS", "IN Economy"]
}
```

**Response:**
```json
{
  "user_id": "user_42",
  "total": 50,
  "recommendations": [
    {
      "article_id": "1773906540622",
      "title": "Oil prices surge 3% as Iran attacks Middle Eastern energy facilities",
      "summary": "Oil prices increased by 3% following Iranian attacks...",
      "category": ["ENERGY_STOCKS", "IN Economy"],
      "timestamp": 1773905580000,
      "score": 0.9312
    }
  ]
}
```

### `GET /api/v1/health`
Returns service status, Qdrant connectivity, SQLite article count.

### `GET /api/v1/article/{article_id}`
Returns the full article record from SQLite (all columns, including descriptive paragraphs, impact matrix, image prompt).

---

## Adding Latest News

```powershell
# From a CSV file (same columns as news_dataset.csv)
python scripts/ingest_latest_news.py --csv latest_news.csv

# From a JSON file (array of article objects)
python scripts/ingest_latest_news.py --json latest_news.json

# Dry run (see how many new articles would be added)
python scripts/ingest_latest_news.py --csv latest_news.csv --dry-run
```

Only genuinely new articles (not already in both stores) are processed.

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `QDRANT_HOST` | `localhost` | Qdrant host or Cloud URL |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_API_KEY` | _(empty)_ | For Qdrant Cloud only |
| `COLLECTION_NAME` | `news_embeddings` | Qdrant collection name |
| `MODEL_NAME` | `BAAI/bge-m3` | HuggingFace embedding model |
| `RANKER_WEIGHTS_PATH` | `ranker_weights.pt` | Path to save/load ranker |
| `SQLITE_DB_PATH` | `local_store/news.db` | SQLite database file |
| `CACHE_MAX_SIZE` | `1000` | LRU cache size for embeddings |
| `DATASET_PATH` | `news_dataset.csv` | Source CSV for ingestion |
| `INGEST_BATCH_SIZE` | `32` | Embedding batch size during ingestion |

---

## Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI 0.111+ |
| Embeddings | `BAAI/bge-m3` via `FlagEmbedding` (1024-dim) |
| Vector DB | Qdrant (cosine similarity) |
| Full Article Store | SQLite (built-in Python) |
| Neural Re-Ranker | PyTorch (3-block MLP, Sigmoid output) |
| Async | Python async/await + FastAPI lifespan |
| Caching | LRU in-memory (cachetools) |
| Package Management | Python `.venv` |



























# 1. Setup .venv
cd "d:\Personal Work\News Recommendation"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 3. Ingest ALL 18,895 articles (one-time, ~30-60 min on CPU)
python scripts/ingest_full_dataset.py

# 4. (Optional) Pre-train ranker
python scripts/train_ranker.py

# 5. Start API
uvicorn app.main:app --reload

# 6. Add new articles later
python scripts/ingest_latest_news.py --csv latest_news.csv
python scripts/ingest_latest_news.py --json latest_news.json

python scripts/train_ranker.py --max-articles 2000 --pairs 5000 --epochs 5