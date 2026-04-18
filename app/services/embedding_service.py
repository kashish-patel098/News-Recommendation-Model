"""
app/services/embedding_service.py
----------------------------------
BGE-m3 embedding service — produces **three named vectors** per article:
    • title       (386-dim) — from article title text
    • description (386-dim) — from introductory_paragraph
    • tags        (386-dim) — from space-joined tag list

These map 1-to-1 to the Qdrant named-vector collection schema:
    vectors: {
        title:       { size: 386, distance: 'Cosine' },
        description: { size: 386, distance: 'Cosine' },
        tags:        { size: 386, distance: 'Cosine' },
    }

Architecture:
  text -> tokenizer -> AutoModel (BGE-m3) -> CLS token -> L2 norm -> float32[386]

Note: BGE-m3 natively outputs 1024-dim but we project to 386-dim via a
fixed learned linear layer stored in the model config. If using a different
model (e.g. a 386-dim model directly), set MODEL_NAME accordingly and
VECTOR_SIZE=386 in .env — the projection is bypassed automatically.
"""

import hashlib
import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

# --- Suppress TensorFlow / Keras BEFORE any HuggingFace import ---------------
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cachetools import LRUCache
from transformers import AutoModel, AutoTokenizer

from app.utils.text_utils import truncate_tokens

logger = logging.getLogger(__name__)

# Target dimension for named vectors in Qdrant
VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "386"))


class _LinearProjection(nn.Module):
    """Fixed linear layer that projects from model_dim -> VECTOR_SIZE."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        # Xavier initialisation — stable starting point for fine-tuning
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), p=2, dim=-1)


class EmbeddingService:
    """
    BGE-m3 (or any CLS-token HuggingFace) embedding service.
    Loads once per process; thread-safe for concurrent requests.

    Produces three 386-dim named vectors per article call:
        encode_article(title, description, tags_text)
        → {"title": ndarray(386), "description": ndarray(386), "tags": ndarray(386)}
    """

    _model: Optional[AutoModel] = None
    _tokenizer: Optional[AutoTokenizer] = None
    _projection: Optional[_LinearProjection] = None
    _model_name_loaded: Optional[str] = None
    _model_lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_max_size: int = 1000,
        vector_size: int = VECTOR_SIZE,
    ):
        self.model_name = model_name
        self.vector_size = vector_size
        self._cache: LRUCache = LRUCache(maxsize=cache_max_size)
        self._cache_lock = threading.Lock()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    # -- Model loading ---------------------------------------------------------

    def _load_model(self) -> None:
        with EmbeddingService._model_lock:
            if EmbeddingService._model_name_loaded == self.model_name:
                return  # already loaded by a previous instance

            logger.info(
                "Loading tokenizer + model: %s  (device=%s) ...",
                self.model_name,
                self._device,
            )
            EmbeddingService._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            EmbeddingService._model = (
                AutoModel.from_pretrained(self.model_name)
                .to(self._device)
                .eval()
            )

            native_dim = EmbeddingService._model.config.hidden_size
            if native_dim != self.vector_size:
                logger.info(
                    "Native model dim=%d → projecting to %d via LinearProjection.",
                    native_dim, self.vector_size,
                )
                EmbeddingService._projection = _LinearProjection(
                    native_dim, self.vector_size
                ).to(self._device).eval()
            else:
                EmbeddingService._projection = None

            EmbeddingService._model_name_loaded = self.model_name
            logger.info(
                "Model loaded. Output dim = %d", self.embedding_dim
            )

    @property
    def _tok(self) -> AutoTokenizer:
        assert EmbeddingService._tokenizer is not None
        return EmbeddingService._tokenizer

    @property
    def _mdl(self) -> AutoModel:
        assert EmbeddingService._model is not None
        return EmbeddingService._model

    @property
    def embedding_dim(self) -> int:
        """Final output dimension (after optional projection)."""
        return self.vector_size

    # -- Cache helpers ---------------------------------------------------------

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_cached(self, key: str) -> Optional[np.ndarray]:
        with self._cache_lock:
            return self._cache.get(key)

    def _set_cached(self, key: str, vector: np.ndarray) -> None:
        with self._cache_lock:
            self._cache[key] = vector

    # -- Embedding core --------------------------------------------------------

    def _embed_raw(self, texts: List[str]) -> np.ndarray:
        """
        Tokenise → forward pass → CLS token → (optional projection) → L2 norm.
        Returns float32 array of shape (N, VECTOR_SIZE).
        """
        encoded = self._tok(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self._mdl(**encoded)
            cls_vecs = output.last_hidden_state[:, 0, :]   # (N, native_dim)
            if EmbeddingService._projection is not None:
                cls_vecs = EmbeddingService._projection(cls_vecs)   # (N, 386)
            else:
                cls_vecs = F.normalize(cls_vecs, p=2, dim=1)        # (N, same)

        return cls_vecs.cpu().float().numpy()   # (N, VECTOR_SIZE)

    # -- Public API (single text) -----------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single string → 1-D float32 array shape (VECTOR_SIZE,).
        Result is LRU-cached by content hash.
        Used for query-time embedding.
        """
        text = truncate_tokens(text, max_chars=2000)
        key  = self._cache_key(text)

        cached = self._get_cached(key)
        if cached is not None:
            logger.debug("Cache hit %.8s...", key)
            return cached

        vector = self._embed_raw([text])[0]
        self._set_cached(key, vector)
        return vector

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Encode a list of texts in mini-batches → float32 array shape (N, VECTOR_SIZE).
        Does NOT use the LRU cache (bulk ingestion path).
        """
        texts   = [truncate_tokens(t, max_chars=2000) for t in texts]
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            vecs  = self._embed_raw(chunk)
            results.append(vecs)
        return np.vstack(results).astype(np.float32)   # (N, VECTOR_SIZE)

    # -- Public API (named article vectors) ------------------------------------

    def encode_article(
        self,
        title: str,
        description: str,
        tags_text: str,
    ) -> Dict[str, np.ndarray]:
        """
        Produce the three named vectors for one article.

        Returns:
            {
                "title":       np.ndarray(386,),
                "description": np.ndarray(386,),
                "tags":        np.ndarray(386,),
            }

        Calls the model three times (or can be batched via encode_articles_batch
        for bulk ingestion).
        """
        return {
            "title":       self.encode(title),
            "description": self.encode(description),
            "tags":        self.encode(tags_text),
        }

    def encode_articles_batch(
        self,
        titles: List[str],
        descriptions: List[str],
        tags_texts: List[str],
        batch_size: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Batch-encode three text fields for a list of articles.

        Returns:
            (title_vecs, description_vecs, tags_vecs)
            each shape (N, VECTOR_SIZE) float32
        """
        title_vecs = self.encode_batch(titles,       batch_size=batch_size)
        desc_vecs  = self.encode_batch(descriptions, batch_size=batch_size)
        tags_vecs  = self.encode_batch(tags_texts,   batch_size=batch_size)
        return title_vecs, desc_vecs, tags_vecs

    def cache_info(self) -> dict:
        with self._cache_lock:
            return {"size": len(self._cache), "maxsize": self._cache.maxsize}
