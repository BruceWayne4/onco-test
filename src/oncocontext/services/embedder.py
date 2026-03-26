"""PubMedBERT embedding wrapper using sentence-transformers.

Provides single-text and batch embedding using the PubMedBERT model.
768-dimensional vectors, normalized for cosine similarity.

For MVP simplicity, uses sentence-transformers directly (not ONNX).
The model is ~420MB and will be downloaded on first use from HuggingFace.
"""

from __future__ import annotations

import logging

import numpy as np

from oncocontext.config import settings

logger = logging.getLogger(__name__)


class Embedder:
    """PubMedBERT embedding model via sentence-transformers.

    Properties:
        - Model: pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb
        - Output: 768d vectors, normalized for cosine similarity
        - Lazy-loaded on first use to avoid slow server startup
        - ~420MB download on first use from HuggingFace
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize embedder with lazy model loading.

        Args:
            model_name: HuggingFace model name/path.
                Defaults to settings.EMBEDDING_MODEL.
        """
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None

    def _load_model(self) -> None:
        """Lazy load the sentence-transformers model on first use.

        Downloads the model from HuggingFace if not cached locally.
        """
        if self._model is not None:
            return

        logger.info(
            "Loading embedding model '%s' (this may download ~420MB on first use)...",
            self._model_name,
        )

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded successfully: %s", self._model_name)
        except Exception as exc:
            logger.error("Failed to load embedding model '%s': %s", self._model_name, exc)
            raise RuntimeError(
                f"Failed to load embedding model '{self._model_name}': {exc}"
            ) from exc

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Input text to embed.

        Returns:
            768-dimensional list of floats (normalized).
        """
        self._load_model()
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a batch of text strings efficiently.

        Args:
            texts: List of input texts.
            batch_size: Batch size for encoding.

        Returns:
            List of 768-dimensional lists of floats (normalized).
        """
        if not texts:
            return []

        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    async def embed_text_async(self, text: str) -> list[float]:
        """Async wrapper for embed_text (runs in thread to avoid blocking).

        Args:
            text: Input text to embed.

        Returns:
            768-dimensional list of floats (normalized).
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, text)

    async def embed_batch_async(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Async wrapper for embed_batch (runs in thread to avoid blocking).

        Args:
            texts: List of input texts.
            batch_size: Batch size for encoding.

        Returns:
            List of 768-dimensional lists of floats (normalized).
        """
        if not texts:
            return []

        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_batch, texts, batch_size)

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality."""
        return settings.EMBEDDING_DIM

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model is not None
