"""ChromaDB manager — manages two collections: literature_chunks and lab_data.

Provides add, search, and stats operations for the local vector store.
Uses PersistentClient so data survives restarts.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
import chromadb.api
from chromadb.config import Settings

from oncocontext.config import settings

logger = logging.getLogger(__name__)


class ChromaManager:
    """Manage ChromaDB collections for literature chunks and lab data.

    Collections:
        - literature_chunks: Embedded chunks from published papers (PMC full text)
        - lab_data: Embedded summaries from researcher's lab files

    Both use:
        - Embedding dimensionality: 768 (PubMedBERT)
        - Distance metric: cosine
    """

    def __init__(self, persist_dir: str | None = None) -> None:
        """Initialize ChromaDB with persistent storage (lazy).

        Args:
            persist_dir: Path to ChromaDB persistent storage directory.
                Defaults to settings.CHROMA_DIR.
        """
        self._persist_dir = persist_dir or str(settings.CHROMA_DIR)
        self._client: chromadb.api.ClientAPI | None = None
        self._literature_collection: Any = None
        self._lab_collection: Any = None

    def _get_client(self) -> chromadb.api.ClientAPI:
        """Lazy init of ChromaDB PersistentClient.

        anonymized_telemetry=False disables the PostHog telemetry client that
        crashes with:
            capture() takes 1 positional argument but 3 were given
        on every ChromaDB initialisation.
        """
        if self._client is None:
            logger.info("Initializing ChromaDB PersistentClient at %s", self._persist_dir)
            self._client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_literature_collection(self) -> Any:
        """Get or create the literature_chunks collection."""
        if self._literature_collection is None:
            client = self._get_client()
            self._literature_collection = client.get_or_create_collection(
                name=settings.CHROMADB_LITERATURE_COLLECTION,
                metadata={"hnsw:space": settings.CHROMADB_DISTANCE_METRIC},
            )
            logger.debug("Got literature_chunks collection")
        return self._literature_collection

    def _get_lab_collection(self) -> Any:
        """Get or create the lab_data collection."""
        if self._lab_collection is None:
            client = self._get_client()
            self._lab_collection = client.get_or_create_collection(
                name=settings.CHROMADB_LAB_COLLECTION,
                metadata={"hnsw:space": settings.CHROMADB_DISTANCE_METRIC},
            )
            logger.debug("Got lab_data collection")
        return self._lab_collection

    def _get_collection(self, collection_name: str) -> Any:
        """Get the appropriate collection by name.

        Args:
            collection_name: 'literature' or 'literature_chunks' for literature,
                           'lab' or 'lab_data' for lab data.

        Returns:
            ChromaDB collection object.
        """
        if collection_name in ("literature", "literature_chunks"):
            return self._get_literature_collection()
        elif collection_name in ("lab", "lab_data"):
            return self._get_lab_collection()
        else:
            raise ValueError(f"Unknown collection: {collection_name}")

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        collection: str = "literature",
    ) -> int:
        """Add chunks with pre-computed embeddings to a collection.

        Each chunk dict must have: chunk_id, text, and metadata fields.
        For literature chunks: paper_pmid, pmc_id, section, paragraph_num, etc.
        For lab chunks: file_id, file_name, chunk_type, row_index, etc.

        Args:
            chunks: List of chunk dicts.
            embeddings: List of embedding vectors matching chunks.
            collection: 'literature' or 'lab'.

        Returns:
            Number of chunks added.
        """
        if not chunks or not embeddings:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        coll = self._get_collection(collection)

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]

        if collection in ("lab", "lab_data"):
            metadatas = [
                {
                    "file_id": str(c.get("file_id", "")),
                    "file_name": str(c.get("file_name", "")),
                    "chunk_type": str(c.get("chunk_type", "row_data")),
                    "row_index": int(c.get("row_index", -1)),
                    "experiment_label": str(c.get("experiment_label", "")),
                    "markers": str(c.get("markers", "")),
                }
                for c in chunks
            ]
        else:
            metadatas = [
                {
                    "paper_pmid": str(c.get("paper_pmid", "")),
                    "pmc_id": str(c.get("pmc_id", "")),
                    "section": str(c.get("section", "")),
                    "paragraph_num": int(c.get("paragraph_num", 0)),
                    "chunk_index": int(c.get("chunk_index", 0)),
                    "token_count": int(c.get("token_count", 0)),
                }
                for c in chunks
            ]

        # ChromaDB upsert to handle duplicates gracefully
        coll.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info("Added %d chunks to '%s' collection", len(ids), collection)
        return len(ids)

    def search(
        self,
        query_embedding: list[float],
        collection: str = "literature",
        n_results: int = 50,
        where: dict | None = None,
    ) -> dict:
        """Search a collection by embedding vector.

        Args:
            query_embedding: 768d query vector.
            collection: 'literature' or 'lab'.
            n_results: Number of results to return.
            where: Optional ChromaDB metadata filter dict.

        Returns:
            Dict with 'ids', 'distances', 'documents', 'metadatas'.
            Each is a list of lists (outer list has one element for single query).
        """
        coll = self._get_collection(collection)

        # Ensure n_results doesn't exceed collection count
        count = coll.count()
        if count == 0:
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}

        effective_n = min(n_results, count)

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": effective_n,
        }
        if where:
            kwargs["where"] = where

        try:
            results = coll.query(**kwargs)
            return results
        except Exception as exc:
            logger.error("ChromaDB search failed: %s", exc)
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}

    def get_collection_stats(self, collection: str = "literature") -> dict:
        """Get statistics for a collection.

        Args:
            collection: 'literature' or 'lab'.

        Returns:
            Dict with 'count', 'name'.
        """
        coll = self._get_collection(collection)
        return {"count": coll.count(), "name": collection}

    def has_paper(self, pmid: str) -> bool:
        """Check if a paper's chunks are already indexed.

        Args:
            pmid: PubMed ID to check.

        Returns:
            True if at least one chunk with this PMID exists.
        """
        coll = self._get_literature_collection()
        try:
            results = coll.get(where={"paper_pmid": pmid}, limit=1)
            return len(results["ids"]) > 0
        except Exception as exc:
            logger.warning("Error checking paper %s in ChromaDB: %s", pmid, exc)
            return False

    def delete_paper(self, pmid: str) -> int:
        """Delete all chunks for a given paper.

        Args:
            pmid: PubMed ID whose chunks should be removed.

        Returns:
            Number of chunks deleted (approximate).
        """
        coll = self._get_literature_collection()
        try:
            # Get all IDs for this paper
            results = coll.get(where={"paper_pmid": pmid})
            ids = results["ids"]
            if ids:
                coll.delete(ids=ids)
                logger.info("Deleted %d chunks for paper %s", len(ids), pmid)
                return len(ids)
            return 0
        except Exception as exc:
            logger.error("Error deleting paper %s from ChromaDB: %s", pmid, exc)
            return 0

    def reset(self, collection: str = "literature") -> None:
        """Delete and recreate a collection (for testing/debugging).

        Args:
            collection: 'literature' or 'lab'.
        """
        client = self._get_client()
        name = (
            settings.CHROMADB_LITERATURE_COLLECTION
            if collection in ("literature", "literature_chunks")
            else settings.CHROMADB_LAB_COLLECTION
        )
        try:
            client.delete_collection(name)
            logger.info("Deleted collection '%s'", name)
        except Exception:
            pass  # Collection didn't exist

        # Reset cached reference
        if collection in ("literature", "literature_chunks"):
            self._literature_collection = None
        else:
            self._lab_collection = None

        # Recreate
        self._get_collection(collection)
        logger.info("Recreated collection '%s'", name)
