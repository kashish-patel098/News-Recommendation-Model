"""
app/services/embedding_service.py
----------------------------------
BGE-m3 embedding service using plain HuggingFace transformers + PyTorch.

Why this approach?
------------------
  - Zero extra dependencies (transformers + torch are already required)
  - Sets TRANSFORMERS_NO_TF=1 before any import to prevent TF/Keras loading
  - Works on any machine regardless of TensorFlow / Keras version
  - Same 1024-dim BGE-m3 CLS embeddings with L2 normalisation

Architecture:
  text -> tokenizer -> AutoModel (BGE-m3) -> CLS token -> L2 norm -> float32[1024]
"""

import hashlib
import logging
import os
import threading
from typing import List, Optional

# --- Suppress TensorFlow / Keras BEFORE any HuggingFace import ---------------
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from cachetools import LRUCache
from transformers import AutoModel, AutoTokenizer

from app.utils.text_utils import truncate_tokens

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Singleton BGE-m3 embedding service.
    Loads the model once per process; thread-safe for concurrent requests.
    """

    _model: Optional[AutoModel] = None
    _tokenizer: Optional[AutoTokenizer] = None
    _model_name_loaded: Optional[str] = None
    _model_lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_max_size: int = 1000,
    ):
        self.model_name = model_name
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
                "Loading BGE-m3 tokenizer + model: %s  (device=%s) ...",
                self.model_name,
                self._device,
            )
            EmbeddingService._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            EmbeddingService._model = (
                AutoModel.from_pretrained(self.model_name)
                .to(self._device)
                .eval()
            )
            EmbeddingService._model_name_loaded = self.model_name
            logger.info("Model loaded. Embedding dim = %d", self.embedding_dim)

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
        """Read vector size directly from the loaded model — works for any HuggingFace model."""
        return EmbeddingService._model.config.hidden_size

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

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        Tokenise, run forward pass, extract CLS token, L2-normalise.
        Returns float32 array of shape (N, 1024).
        """
        encoded = self._tok(
            texts,
            padding=True,
            truncation=True,
            max_length=512,          # safe limit; BGE-m3 supports up to 8192
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self._mdl(**encoded)

        # BGE models use the [CLS] token (position 0) as the sentence vector
        cls_vecs = output.last_hidden_state[:, 0, :]        # (N, 1024)
        cls_vecs = F.normalize(cls_vecs, p=2, dim=1)        # L2 normalise
        return cls_vecs.cpu().float().numpy()               # (N, 1024) float32

    # -- Public API ------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single string → 1-D float32 array shape (1024,).
        Result is LRU-cached by content hash.
        """
        text = truncate_tokens(text, max_chars=2000)
        key  = self._cache_key(text)

        cached = self._get_cached(key)
        if cached is not None:
            logger.debug("Cache hit %.8s...", key)
            return cached

        vector = self._embed([text])[0]   # (1024,)
        self._set_cached(key, vector)
        return vector

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Encode a list of texts in mini-batches → float32 array shape (N, 1024).
        Used during ingestion — does NOT use the LRU cache.
        Lower batch_size = less RAM (use 8 if you still run out).
        """
        texts = [truncate_tokens(t, max_chars=2000) for t in texts]
        results = []
        for i in range(0, len(texts), batch_size):
            chunk  = texts[i : i + batch_size]
            vecs   = self._embed(chunk)
            results.append(vecs)
        return np.vstack(results).astype(np.float32)    # (N, 1024)

    def cache_info(self) -> dict:
        with self._cache_lock:
            return {"size": len(self._cache), "maxsize": self._cache.maxsize}
