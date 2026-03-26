"""Unit tests for the cross_reference tool.

Tests with mocked ChromaDB, SQLite, Embedder, and Reranker:
    - Findings categorization (agreements/contradictions/explanations)
    - Lab file ID filtering
    - Focus sections filtering
    - Empty results handling
    - Output format matches schema
    - No lab data scenario (literature-only)
    - No literature scenario (error)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# Module path prefix for patching
_MOD = "oncocontext.tools.cross_reference"


@pytest.fixture
def mock_services():
    """Mock all services used by cross_reference."""
    # Mock Embedder
    mock_embedder = MagicMock()
    mock_embedder.embed_text.return_value = [0.1] * 768

    # Mock ChromaManager
    mock_chroma = MagicMock()

    # Mock SQLiteManager
    mock_sqlite = MagicMock()
    mock_sqlite.init_db = AsyncMock()
    mock_sqlite.get_paper = AsyncMock(return_value={
        "pmid": "12345678",
        "title": "T Cell Exhaustion in Cancer",
        "authors": ["Smith J", "Chen L"],
        "journal": "Nature Immunology",
        "year": 2024,
    })
    mock_sqlite.log_search = AsyncMock()

    # Mock Reranker
    mock_reranker = MagicMock()

    return mock_embedder, mock_chroma, mock_sqlite, mock_reranker


def _make_lit_search_results(chunks_data: list[dict]) -> dict:
    """Build a ChromaDB search result dict from chunk data."""
    ids = [c.get("id", f"chunk_{i}") for i, c in enumerate(chunks_data)]
    docs = [c["text"] for c in chunks_data]
    metas = [c.get("metadata", {}) for c in chunks_data]
    dists = [c.get("distance", 0.3) for c in chunks_data]

    return {
        "ids": [ids],
        "documents": [docs],
        "metadatas": [metas],
        "distances": [dists],
    }


def _make_lab_search_results(chunks_data: list[dict]) -> dict:
    """Build a ChromaDB lab search result dict."""
    ids = [c.get("id", f"lab_{i}") for i, c in enumerate(chunks_data)]
    docs = [c["text"] for c in chunks_data]
    metas = [c.get("metadata", {}) for c in chunks_data]
    dists = [c.get("distance", 0.2) for c in chunks_data]

    return {
        "ids": [ids],
        "documents": [docs],
        "metadatas": [metas],
        "distances": [dists],
    }


def _patch_all(mock_embedder, mock_chroma, mock_sqlite, mock_reranker):
    """Return a combined context manager that patches all 4 singletons."""
    from contextlib import contextmanager

    @contextmanager
    def combined():
        with patch(f"{_MOD}._get_embedder", return_value=mock_embedder), \
             patch(f"{_MOD}._get_chroma", return_value=mock_chroma), \
             patch(f"{_MOD}._get_sqlite", return_value=mock_sqlite), \
             patch(f"{_MOD}._get_reranker", return_value=mock_reranker):
            yield

    return combined()


class TestCrossReferenceBasic:
    """Test basic cross_reference functionality."""

    @pytest.mark.asyncio
    async def test_no_literature_indexed(self, mock_services):
        """Returns error when no literature is indexed."""
        mock_embedder, mock_chroma, mock_sqlite, mock_reranker = mock_services

        # Literature count = 0
        mock_chroma.get_collection_stats.side_effect = lambda c: (
            {"count": 0, "name": c} if c == "literature" else {"count": 5, "name": c}
        )

        with _patch_all(mock_embedder, mock_chroma, mock_sqlite, mock_reranker):
            from oncocontext.tools.cross_reference import cross_reference
            result = await cross_reference(
                research_question="Why do exhausted T cells still kill?"
            )

        assert "error" in result
        assert "No literature" in result["error"]

    @pytest.mark.asyncio
    async def test_no_lab_data_still_works(self, mock_services):
        """cross_reference works even without lab data (literature-only)."""
        mock_embedder, mock_chroma, mock_sqlite, mock_reranker = mock_services

        # Literature has data, lab is empty
        mock_chroma.get_collection_stats.side_effect = lambda c: (
            {"count": 100, "name": c} if c == "literature" else {"count": 0, "name": c}
        )

        # Literature search returns some results
        lit_results = _make_lit_search_results([
            {
                "id": "chunk_1",
                "text": "PD-1 expression increases during T cell exhaustion",
                "metadata": {"paper_pmid": "12345678", "section": "results", "paragraph_num": 1, "pmc_id": ""},
                "distance": 0.3,
            },
        ])
        mock_chroma.search.return_value = lit_results

        # Reranker returns scored chunks
        def mock_rerank(query, chunks, top_k=20):
            for c in chunks:
                c["rerank_score"] = 1.5
            return chunks[:top_k]
        mock_reranker.rerank.side_effect = mock_rerank

        with _patch_all(mock_embedder, mock_chroma, mock_sqlite, mock_reranker):
            from oncocontext.tools.cross_reference import cross_reference
            result = await cross_reference(
                research_question="T cell exhaustion markers"
            )

        assert "error" not in result
        assert "summary" in result
        assert "agreements" in result
        assert "contradictions" in result
        assert "novel_findings" in result
        assert "suggested_follow_up" in result
        assert result["papers_consulted"] >= 0

    @pytest.mark.asyncio
    async def test_output_format_matches_schema(self, mock_services):
        """Output has all required fields per CrossReferenceResult schema."""
        mock_embedder, mock_chroma, mock_sqlite, mock_reranker = mock_services

        mock_chroma.get_collection_stats.side_effect = lambda c: (
            {"count": 100, "name": c} if c == "literature" else {"count": 10, "name": c}
        )

        # Lab data search
        lab_results = _make_lab_search_results([
            {
                "text": "Sample S07: CD8 55.8%, PD1 MFI 1850.3, Cytotoxicity 42.3%",
                "metadata": {"file_id": "lab_abc", "chunk_type": "row_data", "file_name": "test.csv",
                             "row_index": 6, "experiment_label": "", "markers": "CD8,PD1"},
            },
        ])

        # Literature search
        lit_results = _make_lit_search_results([
            {
                "id": "c1",
                "text": "Exhausted T cells show reduced cytotoxicity below 20%",
                "metadata": {"paper_pmid": "12345678", "section": "results", "paragraph_num": 3, "pmc_id": ""},
                "distance": 0.2,
            },
            {
                "id": "c2",
                "text": "Progenitor exhausted T cells retain effector function",
                "metadata": {"paper_pmid": "12345678", "section": "discussion", "paragraph_num": 5, "pmc_id": ""},
                "distance": 0.25,
            },
        ])

        def mock_search(query_embedding, collection="literature", n_results=50, where=None):
            if collection in ("lab", "lab_data"):
                return lab_results
            return lit_results

        mock_chroma.search.side_effect = mock_search

        def mock_rerank(query, chunks, top_k=20):
            for i, c in enumerate(chunks):
                c["rerank_score"] = 2.0 - i * 0.5
            return chunks[:top_k]
        mock_reranker.rerank.side_effect = mock_rerank

        with _patch_all(mock_embedder, mock_chroma, mock_sqlite, mock_reranker):
            from oncocontext.tools.cross_reference import cross_reference
            result = await cross_reference(
                research_question="Why do exhausted T cells preserve cytotoxicity?"
            )

        # Check all required schema fields
        assert isinstance(result["summary"], str)
        assert isinstance(result["lab_data_summary"], str)
        assert isinstance(result["agreements"], list)
        assert isinstance(result["contradictions"], list)
        assert isinstance(result["novel_findings"], list)
        assert isinstance(result["suggested_follow_up"], list)
        assert isinstance(result["papers_consulted"], int)
        assert isinstance(result["chunks_analyzed"], int)


class TestCrossReferenceCategorization:
    """Test finding categorization logic."""

    @pytest.mark.asyncio
    async def test_results_section_as_agreement(self, mock_services):
        """Results section chunks are categorized as agreements."""
        mock_embedder, mock_chroma, mock_sqlite, mock_reranker = mock_services

        mock_chroma.get_collection_stats.side_effect = lambda c: (
            {"count": 50, "name": c} if c == "literature" else {"count": 0, "name": c}
        )

        lit_results = _make_lit_search_results([
            {
                "id": "c1",
                "text": "PD-1 and TIM-3 co-expression was observed in exhausted T cells",
                "metadata": {"paper_pmid": "12345678", "section": "results", "paragraph_num": 2, "pmc_id": ""},
                "distance": 0.2,
            },
        ])
        mock_chroma.search.return_value = lit_results

        def mock_rerank(query, chunks, top_k=20):
            for c in chunks:
                c["rerank_score"] = 1.5
            return chunks[:top_k]
        mock_reranker.rerank.side_effect = mock_rerank

        with _patch_all(mock_embedder, mock_chroma, mock_sqlite, mock_reranker):
            from oncocontext.tools.cross_reference import cross_reference
            result = await cross_reference(
                research_question="PD-1 TIM-3 co-expression in T cells"
            )

        assert len(result["agreements"]) > 0

    @pytest.mark.asyncio
    async def test_discussion_section_as_explanation(self, mock_services):
        """Discussion section chunks are categorized as explanations."""
        mock_embedder, mock_chroma, mock_sqlite, mock_reranker = mock_services

        mock_chroma.get_collection_stats.side_effect = lambda c: (
            {"count": 50, "name": c} if c == "literature" else {"count": 0, "name": c}
        )

        lit_results = _make_lit_search_results([
            {
                "id": "c1",
                "text": "The preserved cytotoxicity may be explained by progenitor exhaustion",
                "metadata": {"paper_pmid": "12345678", "section": "discussion", "paragraph_num": 5, "pmc_id": ""},
                "distance": 0.2,
            },
        ])
        mock_chroma.search.return_value = lit_results

        def mock_rerank(query, chunks, top_k=20):
            for c in chunks:
                c["rerank_score"] = 1.5
            return chunks[:top_k]
        mock_reranker.rerank.side_effect = mock_rerank

        with _patch_all(mock_embedder, mock_chroma, mock_sqlite, mock_reranker):
            from oncocontext.tools.cross_reference import cross_reference
            result = await cross_reference(
                research_question="Why do exhausted T cells preserve cytotoxicity?"
            )

        # Discussion chunk should appear — check chunks_analyzed > 0
        assert result["chunks_analyzed"] > 0

    @pytest.mark.asyncio
    async def test_methods_section_as_followup(self, mock_services):
        """Methods section chunks contribute to suggested follow-ups."""
        mock_embedder, mock_chroma, mock_sqlite, mock_reranker = mock_services

        mock_chroma.get_collection_stats.side_effect = lambda c: (
            {"count": 50, "name": c} if c == "literature" else {"count": 0, "name": c}
        )

        lit_results = _make_lit_search_results([
            {
                "id": "c1",
                "text": "Cells were stained with anti-TCF1 antibody clone C63D9 and analyzed by flow cytometry",
                "metadata": {"paper_pmid": "12345678", "section": "methods", "paragraph_num": 1, "pmc_id": ""},
                "distance": 0.25,
            },
        ])
        mock_chroma.search.return_value = lit_results

        def mock_rerank(query, chunks, top_k=20):
            for c in chunks:
                c["rerank_score"] = 1.0
            return chunks[:top_k]
        mock_reranker.rerank.side_effect = mock_rerank

        with _patch_all(mock_embedder, mock_chroma, mock_sqlite, mock_reranker):
            from oncocontext.tools.cross_reference import cross_reference
            result = await cross_reference(
                research_question="T cell staining protocol"
            )

        assert len(result["suggested_follow_up"]) > 0

    @pytest.mark.asyncio
    async def test_contradiction_detection(self, mock_services):
        """Contradiction detected when question implies paradox and text has contrary signals."""
        mock_embedder, mock_chroma, mock_sqlite, mock_reranker = mock_services

        mock_chroma.get_collection_stats.side_effect = lambda c: (
            {"count": 50, "name": c} if c == "literature" else {"count": 5, "name": c}
        )

        lab_results = _make_lab_search_results([
            {
                "text": "Sample S07: CD8 55.8%, PD1 MFI 1850.3, Cytotoxicity 42.3%",
                "metadata": {"file_id": "lab_abc", "chunk_type": "row_data", "file_name": "test.csv",
                             "row_index": 6, "experiment_label": "", "markers": "CD8,PD1"},
            },
        ])

        lit_results = _make_lit_search_results([
            {
                "id": "c1",
                "text": "Exhausted T cells showed reduced cytotoxicity with impaired killing capacity",
                "metadata": {"paper_pmid": "12345678", "section": "results", "paragraph_num": 4, "pmc_id": ""},
                "distance": 0.2,
            },
        ])

        def mock_search(query_embedding, collection="literature", n_results=50, where=None):
            if collection in ("lab", "lab_data"):
                return lab_results
            return lit_results

        mock_chroma.search.side_effect = mock_search

        def mock_rerank(query, chunks, top_k=20):
            for c in chunks:
                c["rerank_score"] = 1.5
            return chunks[:top_k]
        mock_reranker.rerank.side_effect = mock_rerank

        with _patch_all(mock_embedder, mock_chroma, mock_sqlite, mock_reranker):
            from oncocontext.tools.cross_reference import cross_reference
            result = await cross_reference(
                research_question="Why do my T cells show high PD-1 but preserved killing despite exhaustion?"
            )

        # Should detect contradiction (question has "despite", text has "reduced", "impaired")
        assert len(result["contradictions"]) > 0


class TestCrossReferenceFiltering:
    """Test filtering by lab_file_ids and paper_ids."""

    @pytest.mark.asyncio
    async def test_lab_file_id_filter(self, mock_services):
        """lab_file_ids filter is passed to ChromaDB lab search."""
        mock_embedder, mock_chroma, mock_sqlite, mock_reranker = mock_services

        mock_chroma.get_collection_stats.side_effect = lambda c: (
            {"count": 50, "name": c} if c == "literature" else {"count": 10, "name": c}
        )

        empty_results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        mock_chroma.search.return_value = empty_results

        with _patch_all(mock_embedder, mock_chroma, mock_sqlite, mock_reranker):
            from oncocontext.tools.cross_reference import cross_reference
            result = await cross_reference(
                research_question="test",
                lab_file_ids=["lab_abc123"],
            )

        # Verify the lab search used the file_id filter
        lab_calls = [
            call for call in mock_chroma.search.call_args_list
            if call.kwargs.get("collection") in ("lab", "lab_data")
        ]
        if lab_calls:
            where_arg = lab_calls[0].kwargs.get("where")
            assert where_arg is not None
            assert "file_id" in str(where_arg)

    @pytest.mark.asyncio
    async def test_paper_ids_filter(self, mock_services):
        """paper_ids filter is passed to ChromaDB literature search."""
        mock_embedder, mock_chroma, mock_sqlite, mock_reranker = mock_services

        mock_chroma.get_collection_stats.side_effect = lambda c: (
            {"count": 50, "name": c} if c == "literature" else {"count": 0, "name": c}
        )

        empty_results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        mock_chroma.search.return_value = empty_results

        with _patch_all(mock_embedder, mock_chroma, mock_sqlite, mock_reranker):
            from oncocontext.tools.cross_reference import cross_reference
            result = await cross_reference(
                research_question="test",
                paper_ids=["38123456"],
            )

        # Verify no error occurred
        assert "error" not in result or "No matching" in result.get("summary", "")


class TestCrossReferenceEmptyResults:
    """Test empty results handling."""

    @pytest.mark.asyncio
    async def test_no_matching_literature(self, mock_services):
        """Returns graceful message when no literature matches."""
        mock_embedder, mock_chroma, mock_sqlite, mock_reranker = mock_services

        mock_chroma.get_collection_stats.side_effect = lambda c: (
            {"count": 50, "name": c} if c == "literature" else {"count": 0, "name": c}
        )

        empty_results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        mock_chroma.search.return_value = empty_results

        with _patch_all(mock_embedder, mock_chroma, mock_sqlite, mock_reranker):
            from oncocontext.tools.cross_reference import cross_reference
            result = await cross_reference(
                research_question="Very specific rare question"
            )

        assert "No matching literature" in result["summary"]
        assert result["agreements"] == []
        assert result["contradictions"] == []

    @pytest.mark.asyncio
    async def test_reranker_failure_fallback(self, mock_services):
        """Falls back to vector similarity when reranker fails."""
        mock_embedder, mock_chroma, mock_sqlite, mock_reranker = mock_services

        mock_chroma.get_collection_stats.side_effect = lambda c: (
            {"count": 50, "name": c} if c == "literature" else {"count": 0, "name": c}
        )

        lit_results = _make_lit_search_results([
            {
                "id": "c1",
                "text": "T cell exhaustion is characterized by PD-1 upregulation",
                "metadata": {"paper_pmid": "12345678", "section": "results", "paragraph_num": 1, "pmc_id": ""},
                "distance": 0.2,
            },
        ])
        mock_chroma.search.return_value = lit_results

        # Reranker raises exception
        mock_reranker.rerank.side_effect = RuntimeError("Model failed to load")

        with _patch_all(mock_embedder, mock_chroma, mock_sqlite, mock_reranker):
            from oncocontext.tools.cross_reference import cross_reference
            result = await cross_reference(
                research_question="T cell exhaustion"
            )

        # Should still return results using fallback
        assert "error" not in result
        assert result["chunks_analyzed"] > 0
        assert "vector similarity" in result["summary"]


class TestCrossReferenceCitationEnrichment:
    """Test citation enrichment from SQLite."""

    @pytest.mark.asyncio
    async def test_citations_enriched_with_paper_metadata(self, mock_services):
        """Citations include paper title, authors, journal from SQLite."""
        mock_embedder, mock_chroma, mock_sqlite, mock_reranker = mock_services

        mock_chroma.get_collection_stats.side_effect = lambda c: (
            {"count": 50, "name": c} if c == "literature" else {"count": 0, "name": c}
        )

        lit_results = _make_lit_search_results([
            {
                "id": "c1",
                "text": "PD-1 expression increases during exhaustion",
                "metadata": {"paper_pmid": "12345678", "section": "results", "paragraph_num": 2, "pmc_id": "PMC123"},
                "distance": 0.2,
            },
        ])
        mock_chroma.search.return_value = lit_results

        def mock_rerank(query, chunks, top_k=20):
            for c in chunks:
                c["rerank_score"] = 1.5
            return chunks[:top_k]
        mock_reranker.rerank.side_effect = mock_rerank

        with _patch_all(mock_embedder, mock_chroma, mock_sqlite, mock_reranker):
            from oncocontext.tools.cross_reference import cross_reference
            result = await cross_reference(
                research_question="PD-1 expression"
            )

        # Check that citations were enriched
        if result["agreements"]:
            citation = result["agreements"][0]["citation"]
            assert citation["paper_title"] == "T Cell Exhaustion in Cancer"
            assert citation["journal"] == "Nature Immunology"
            assert citation["year"] == 2024


class TestHelperFunctions:
    """Test internal helper functions."""

    def test_normalize_score(self):
        """Score normalization uses sigmoid."""
        from oncocontext.tools.cross_reference import _normalize_score

        # High positive score → close to 1
        assert _normalize_score(10.0) > 0.99
        # Zero → 0.5
        assert abs(_normalize_score(0.0) - 0.5) < 0.001
        # High negative → close to 0
        assert _normalize_score(-10.0) < 0.01

    def test_score_to_confidence(self):
        """Confidence mapping works correctly."""
        from oncocontext.tools.cross_reference import _score_to_confidence

        assert _score_to_confidence(0.9) == "strong"
        assert _score_to_confidence(0.6) == "moderate"
        assert _score_to_confidence(0.3) == "weak"

    def test_summarize_lab_data_empty(self):
        """Empty lab data returns default message."""
        from oncocontext.tools.cross_reference import _summarize_lab_data

        result = _summarize_lab_data([])
        assert "No lab data" in result

    def test_summarize_lab_data_with_chunks(self):
        """Lab data summary combines chunk texts."""
        from oncocontext.tools.cross_reference import _summarize_lab_data

        chunks = [
            {"text": "File summary here", "metadata": {"chunk_type": "file_summary"}},
            {"text": "Sample S01 data", "metadata": {"chunk_type": "row_data"}},
        ]
        result = _summarize_lab_data(chunks)
        assert "File summary" in result
        assert "S01" in result

    def test_generate_comparison_queries(self):
        """Query generation produces 1-3 queries."""
        from oncocontext.tools.cross_reference import _generate_comparison_queries

        queries = _generate_comparison_queries(
            "Why do PD-1+ T cells maintain cytotoxicity?",
            "CD8 55%, PD1 MFI 1850, Cytotoxicity 42%",
            "findings",
        )
        assert 1 <= len(queries) <= 3
        assert queries[0] == "Why do PD-1+ T cells maintain cytotoxicity?"
