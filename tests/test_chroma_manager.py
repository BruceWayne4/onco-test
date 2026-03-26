"""Unit tests for ChromaManager.

Tests ChromaDB operations using a temporary directory so tests don't
affect the production database.
"""

import tempfile

import pytest

from oncocontext.storage.chroma_manager import ChromaManager


@pytest.fixture
def chroma_dir(tmp_path):
    """Provide a temporary directory for ChromaDB."""
    return str(tmp_path / "test_chromadb")


@pytest.fixture
def chroma(chroma_dir) -> ChromaManager:
    """Create a ChromaManager with a temp directory."""
    return ChromaManager(persist_dir=chroma_dir)


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Create sample chunk dicts for testing."""
    return [
        {
            "chunk_id": "12345_methods_0_0",
            "text": "[Methods] Flow cytometry was performed using anti-CD8 clone RPA-T8.",
            "paper_pmid": "12345",
            "pmc_id": "PMC123",
            "section": "methods",
            "paragraph_num": 0,
            "chunk_index": 0,
            "token_count": 12,
        },
        {
            "chunk_id": "12345_methods_1_0",
            "text": "[Methods] Cells were stained with anti-PD-1 clone EH12.2H7.",
            "paper_pmid": "12345",
            "pmc_id": "PMC123",
            "section": "methods",
            "paragraph_num": 1,
            "chunk_index": 0,
            "token_count": 10,
        },
        {
            "chunk_id": "12345_results_0_0",
            "text": "[Results] PD-1+TIM-3+ CD8+ T cells constituted 45-60% of total CD8+ population.",
            "paper_pmid": "12345",
            "pmc_id": "PMC123",
            "section": "results",
            "paragraph_num": 0,
            "chunk_index": 0,
            "token_count": 12,
        },
    ]


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Create sample 768-d embeddings (fake, for testing)."""
    import random
    random.seed(42)
    return [
        [random.uniform(-1, 1) for _ in range(768)]
        for _ in range(3)
    ]


class TestChromaManagerBasic:
    """Basic add and search operations."""

    def test_add_chunks(self, chroma: ChromaManager, sample_chunks, sample_embeddings):
        """Adding chunks returns the correct count."""
        count = chroma.add_chunks(sample_chunks, sample_embeddings, collection="literature")
        assert count == 3

    def test_search_returns_results(self, chroma: ChromaManager, sample_chunks, sample_embeddings):
        """Search returns results after adding chunks."""
        chroma.add_chunks(sample_chunks, sample_embeddings, collection="literature")

        # Search with the first embedding (should find itself as best match)
        results = chroma.search(
            query_embedding=sample_embeddings[0],
            collection="literature",
            n_results=3,
        )

        assert len(results["ids"][0]) > 0
        assert len(results["documents"][0]) > 0
        assert len(results["metadatas"][0]) > 0
        assert len(results["distances"][0]) > 0

    def test_search_empty_collection(self, chroma: ChromaManager, sample_embeddings):
        """Search on empty collection returns empty results."""
        results = chroma.search(
            query_embedding=sample_embeddings[0],
            collection="literature",
            n_results=5,
        )
        assert results["ids"] == [[]]
        assert results["documents"] == [[]]


class TestChromaManagerHasPaper:
    """Tests for has_paper check."""

    def test_has_paper_true(self, chroma: ChromaManager, sample_chunks, sample_embeddings):
        """has_paper returns True for an indexed paper."""
        chroma.add_chunks(sample_chunks, sample_embeddings, collection="literature")
        assert chroma.has_paper("12345") is True

    def test_has_paper_false(self, chroma: ChromaManager):
        """has_paper returns False for a non-indexed paper."""
        assert chroma.has_paper("99999") is False

    def test_has_paper_false_empty(self, chroma: ChromaManager):
        """has_paper returns False on empty collection."""
        assert chroma.has_paper("12345") is False


class TestChromaManagerStats:
    """Tests for collection stats."""

    def test_stats_empty(self, chroma: ChromaManager):
        """Stats for empty collection shows count 0."""
        stats = chroma.get_collection_stats("literature")
        assert stats["count"] == 0
        assert stats["name"] == "literature"

    def test_stats_after_add(self, chroma: ChromaManager, sample_chunks, sample_embeddings):
        """Stats reflect the number of chunks added."""
        chroma.add_chunks(sample_chunks, sample_embeddings, collection="literature")
        stats = chroma.get_collection_stats("literature")
        assert stats["count"] == 3

    def test_stats_lab_collection(self, chroma: ChromaManager):
        """Lab collection stats work too."""
        stats = chroma.get_collection_stats("lab")
        assert stats["count"] == 0
        assert stats["name"] == "lab"


class TestChromaManagerSectionFiltering:
    """Tests for section-based metadata filtering."""

    def test_filter_by_section(self, chroma: ChromaManager, sample_chunks, sample_embeddings):
        """Searching with section filter returns only matching chunks."""
        chroma.add_chunks(sample_chunks, sample_embeddings, collection="literature")

        # Search for methods only
        results = chroma.search(
            query_embedding=sample_embeddings[0],
            collection="literature",
            n_results=10,
            where={"section": "methods"},
        )

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        assert len(ids) > 0
        for meta in metadatas:
            assert meta["section"] == "methods"

    def test_filter_by_pmid(self, chroma: ChromaManager, sample_chunks, sample_embeddings):
        """Searching with PMID filter works."""
        chroma.add_chunks(sample_chunks, sample_embeddings, collection="literature")

        results = chroma.search(
            query_embedding=sample_embeddings[0],
            collection="literature",
            n_results=10,
            where={"paper_pmid": "12345"},
        )

        ids = results["ids"][0]
        assert len(ids) == 3  # all chunks belong to paper 12345


class TestChromaManagerMultiplePapers:
    """Tests for multiple papers in the collection."""

    def test_multiple_papers(self, chroma: ChromaManager, sample_embeddings):
        """Chunks from different papers are stored and retrievable."""
        import random
        random.seed(123)

        chunks_paper1 = [
            {
                "chunk_id": "11111_methods_0_0",
                "text": "[Methods] Paper 1 methods text.",
                "paper_pmid": "11111",
                "pmc_id": "PMC111",
                "section": "methods",
                "paragraph_num": 0,
                "chunk_index": 0,
                "token_count": 6,
            }
        ]
        chunks_paper2 = [
            {
                "chunk_id": "22222_results_0_0",
                "text": "[Results] Paper 2 results text.",
                "paper_pmid": "22222",
                "pmc_id": "PMC222",
                "section": "results",
                "paragraph_num": 0,
                "chunk_index": 0,
                "token_count": 6,
            }
        ]

        emb1 = [[random.uniform(-1, 1) for _ in range(768)]]
        emb2 = [[random.uniform(-1, 1) for _ in range(768)]]

        chroma.add_chunks(chunks_paper1, emb1, collection="literature")
        chroma.add_chunks(chunks_paper2, emb2, collection="literature")

        # Both papers should be findable
        assert chroma.has_paper("11111") is True
        assert chroma.has_paper("22222") is True

        stats = chroma.get_collection_stats("literature")
        assert stats["count"] == 2

    def test_delete_paper(self, chroma: ChromaManager, sample_chunks, sample_embeddings):
        """Deleting a paper removes its chunks."""
        chroma.add_chunks(sample_chunks, sample_embeddings, collection="literature")
        assert chroma.has_paper("12345") is True

        deleted = chroma.delete_paper("12345")
        assert deleted == 3
        assert chroma.has_paper("12345") is False


class TestChromaManagerEdgeCases:
    """Edge case tests."""

    def test_add_empty_chunks(self, chroma: ChromaManager):
        """Adding empty list returns 0."""
        count = chroma.add_chunks([], [], collection="literature")
        assert count == 0

    def test_mismatched_chunks_embeddings(self, chroma: ChromaManager, sample_chunks):
        """Mismatched chunk/embedding counts raises ValueError."""
        with pytest.raises(ValueError, match="Mismatch"):
            chroma.add_chunks(sample_chunks, [[0.1] * 768], collection="literature")

    def test_upsert_idempotent(self, chroma: ChromaManager, sample_chunks, sample_embeddings):
        """Adding the same chunks twice doesn't duplicate them."""
        chroma.add_chunks(sample_chunks, sample_embeddings, collection="literature")
        chroma.add_chunks(sample_chunks, sample_embeddings, collection="literature")
        stats = chroma.get_collection_stats("literature")
        assert stats["count"] == 3  # still 3, not 6

    def test_unknown_collection_raises(self, chroma: ChromaManager):
        """Accessing unknown collection raises ValueError."""
        with pytest.raises(ValueError, match="Unknown collection"):
            chroma._get_collection("nonexistent")

    def test_reset_collection(self, chroma: ChromaManager, sample_chunks, sample_embeddings):
        """Resetting a collection removes all data."""
        chroma.add_chunks(sample_chunks, sample_embeddings, collection="literature")
        assert chroma.get_collection_stats("literature")["count"] == 3

        chroma.reset("literature")
        assert chroma.get_collection_stats("literature")["count"] == 0
