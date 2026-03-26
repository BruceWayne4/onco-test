"""Cross-encoder reranker — ms-marco-MiniLM-L-6-v2.

Scores (query, chunk) pairs for precision reranking after vector search.
Two-stage retrieval: ChromaDB returns top-50 candidates, cross-encoder reranks to top-10.
"""

from __future__ import annotations

import logging

from oncocontext.config import settings

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

    Properties:
        - Model: cross-encoder/ms-marco-MiniLM-L-6-v2
        - Size: ~90MB (loaded via sentence-transformers)
        - Latency: 80-150ms for batch of 50 (query, chunk) pairs (CPU)
        - Lazy-loaded to avoid startup cost
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize reranker with lazy model loading.

        Args:
            model_name: Cross-encoder model name/path.
                Defaults to settings.RERANKER_MODEL.
        """
        self._model_name = model_name or settings.RERANKER_MODEL
        self._model = None

    def _load_model(self) -> None:
        """Lazy load the cross-encoder model on first use.

        Downloads the model from HuggingFace if not cached locally.
        """
        if self._model is not None:
            return

        logger.info(
            "Loading cross-encoder model '%s' (this may download ~90MB on first use)...",
            self._model_name,
        )

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
            logger.info("Cross-encoder model loaded successfully: %s", self._model_name)
        except Exception as exc:
            logger.error("Failed to load cross-encoder model '%s': %s", self._model_name, exc)
            raise RuntimeError(
                f"Failed to load cross-encoder model '{self._model_name}': {exc}"
            ) from exc

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank chunks using cross-encoder scores.

        Args:
            query: The search query.
            chunks: List of chunk dicts, each must have a 'text' field.
                May also have 'score' and other metadata fields.
            top_k: Number of top results to return.
                Defaults to settings.RERANK_TOP_K (10).

        Returns:
            List of chunk dicts sorted by cross-encoder score (descending),
            with 'rerank_score' field added to each. Truncated to top_k.
        """
        if not chunks:
            return []

        self._load_model()
        top_k = top_k or settings.RERANK_TOP_K

        # Create query-chunk pairs for cross-encoder
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Score all pairs
        scores = self._model.predict(pairs)

        # Add rerank_score to each chunk
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        # Sort by rerank_score descending, take top_k
        reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
        return reranked

    async def rerank_async(
        self,
        query: str,
        chunks: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Async wrapper for rerank (runs in thread to avoid blocking).

        Args:
            query: The search query.
            chunks: List of chunk dicts with at least 'text' field.
            top_k: Number of top results to return.

        Returns:
            List of chunk dicts sorted by cross-encoder score, with 'rerank_score' added.
        """
        if not chunks:
            return []

        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.rerank, query, chunks, top_k)

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model is not None
