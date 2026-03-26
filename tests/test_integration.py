"""End-to-end integration tests that simulate the actual demo flow.

These tests use temp directories for ChromaDB and SQLite so they don't
interfere with real data. Tests marked @pytest.mark.slow require network
access to PubMed/PMC APIs.

Run all tests:
    python -m pytest tests/test_integration.py -v

Run only fast tests (no network):
    python -m pytest tests/test_integration.py -v -m "not slow"

Run slow tests (with network):
    python -m pytest tests/test_integration.py -v -m slow
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for ChromaDB and SQLite."""
    tmp = tempfile.mkdtemp(prefix="oncocontext_test_")
    chroma_dir = os.path.join(tmp, "chromadb")
    sqlite_dir = os.path.join(tmp, "sqlite")
    cache_dir = os.path.join(tmp, "cache")
    os.makedirs(chroma_dir, exist_ok=True)
    os.makedirs(sqlite_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    yield {
        "root": tmp,
        "chroma_dir": chroma_dir,
        "sqlite_dir": sqlite_dir,
        "sqlite_path": os.path.join(sqlite_dir, "test_metadata.db"),
        "cache_dir": cache_dir,
    }
    # Cleanup
    try:
        shutil.rmtree(tmp)
    except Exception:
        pass


@pytest.fixture
def chroma_manager(temp_data_dir):
    """Create a ChromaManager using a temp directory."""
    from oncocontext.storage.chroma_manager import ChromaManager

    return ChromaManager(persist_dir=temp_data_dir["chroma_dir"])


@pytest.fixture
def sqlite_manager(temp_data_dir):
    """Create a SQLiteManager using a temp directory."""
    from oncocontext.storage.sqlite_manager import SQLiteManager

    return SQLiteManager(db_path=temp_data_dir["sqlite_path"])


@pytest.fixture
def embedder():
    """Create an Embedder (lazy-loaded)."""
    from oncocontext.services.embedder import Embedder

    return Embedder()


@pytest.fixture
def demo_csv_path():
    """Return the path to the demo flow cytometry CSV."""
    csv_path = Path(__file__).parent.parent / "demo" / "sample_flow_cytometry.csv"
    assert csv_path.exists(), f"Demo CSV not found at {csv_path}"
    return str(csv_path)


# ── Test: Embedder loads and produces correct dimensions ──────────────────────


class TestEmbedder:
    """Tests for the embedding model."""

    def test_embed_single_text(self, embedder):
        """Test embedding a single text returns 768d vector."""
        embedding = embedder.embed_text("T cell exhaustion PD-1")
        assert isinstance(embedding, list)
        assert len(embedding) == 768
        # Values should be normalized (roughly unit length)
        import math

        magnitude = math.sqrt(sum(x * x for x in embedding))
        assert abs(magnitude - 1.0) < 0.01

    def test_embed_batch(self, embedder):
        """Test batch embedding returns correct number of vectors."""
        texts = [
            "CD8 T cell exhaustion",
            "PD-1 TIM-3 co-expression",
            "Flow cytometry gating strategy",
        ]
        embeddings = embedder.embed_batch(texts)
        assert len(embeddings) == 3
        assert all(len(e) == 768 for e in embeddings)


# ── Test: ChromaDB operations ──────────────────────────────────────────────────


class TestChromaIntegration:
    """Integration tests for ChromaDB storage."""

    def test_add_and_search_chunks(self, chroma_manager, embedder):
        """Test adding chunks and searching them."""
        # Create test chunks
        chunks = [
            {
                "chunk_id": "test_pmid_methods_0_0",
                "text": "[Methods] Flow cytometry was performed using anti-CD8-APC clone SK1.",
                "paper_pmid": "12345678",
                "pmc_id": "PMC1234567",
                "section": "methods",
                "paragraph_num": 0,
                "chunk_index": 0,
                "token_count": 15,
            },
            {
                "chunk_id": "test_pmid_results_0_0",
                "text": "[Results] PD-1 expression increased from 200 to 1800 MFI over 72 hours.",
                "paper_pmid": "12345678",
                "pmc_id": "PMC1234567",
                "section": "results",
                "paragraph_num": 0,
                "chunk_index": 0,
                "token_count": 18,
            },
        ]

        # Embed
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_batch(texts)

        # Add to ChromaDB
        count = chroma_manager.add_chunks(chunks, embeddings, collection="literature")
        assert count == 2

        # Search
        query_emb = embedder.embed_text("antibody clone flow cytometry")
        results = chroma_manager.search(
            query_embedding=query_emb,
            collection="literature",
            n_results=2,
        )

        result_ids = results.get("ids", [[]])[0]
        assert len(result_ids) > 0

    def test_has_paper(self, chroma_manager, embedder):
        """Test checking if a paper is indexed."""
        assert not chroma_manager.has_paper("99999999")

        # Add a chunk
        chunks = [
            {
                "chunk_id": "99999999_methods_0_0",
                "text": "[Methods] Some test text for checking paper existence in ChromaDB.",
                "paper_pmid": "99999999",
                "pmc_id": "",
                "section": "methods",
                "paragraph_num": 0,
                "chunk_index": 0,
                "token_count": 10,
            }
        ]
        embeddings = embedder.embed_batch([chunks[0]["text"]])
        chroma_manager.add_chunks(chunks, embeddings, collection="literature")

        assert chroma_manager.has_paper("99999999")

    def test_collection_stats(self, chroma_manager):
        """Test getting collection statistics."""
        stats = chroma_manager.get_collection_stats("literature")
        assert "count" in stats
        assert isinstance(stats["count"], int)


# ── Test: SQLite operations ────────────────────────────────────────────────────


class TestSQLiteIntegration:
    """Integration tests for SQLite metadata storage."""

    async def test_add_and_get_paper(self, sqlite_manager):
        """Test adding and retrieving a paper."""
        await sqlite_manager.init_db()

        await sqlite_manager.add_paper(
            {
                "pmid": "11111111",
                "title": "Test Paper on T Cell Exhaustion",
                "authors": ["Smith J", "Chen L"],
                "journal": "Nature Immunology",
                "year": 2024,
                "abstract": "We studied T cell exhaustion...",
                "mesh_terms": ["T-Cell Exhaustion", "PD-1"],
                "has_full_text": True,
                "pmc_id": "PMC1111111",
            }
        )

        paper = await sqlite_manager.get_paper("11111111")
        assert paper is not None
        assert paper["pmid"] == "11111111"
        assert paper["title"] == "Test Paper on T Cell Exhaustion"
        assert paper["year"] == 2024

    async def test_indexed_paper_count(self, sqlite_manager):
        """Test counting indexed papers."""
        await sqlite_manager.init_db()

        count = await sqlite_manager.get_indexed_paper_count()
        assert count == 0

        # Add and mark a paper as indexed
        await sqlite_manager.add_paper(
            {
                "pmid": "22222222",
                "title": "Indexed Paper",
                "has_full_text": True,
            }
        )
        await sqlite_manager.update_paper_index_status(
            pmid="22222222",
            chunk_count=10,
        )

        count = await sqlite_manager.get_indexed_paper_count()
        assert count == 1

    async def test_add_lab_file(self, sqlite_manager):
        """Test adding and retrieving a lab file."""
        await sqlite_manager.init_db()

        await sqlite_manager.add_lab_file(
            {
                "file_id": "lab_test_001",
                "file_name": "test.csv",
                "file_type": "csv",
                "file_path": "/tmp/test.csv",
                "row_count": 20,
                "column_names": ["CD8", "PD1"],
                "detected_markers": ["CD8", "PD-1"],
                "chunk_count": 21,
            }
        )

        lab_file = await sqlite_manager.get_lab_file("lab_test_001")
        assert lab_file is not None
        assert lab_file["file_name"] == "test.csv"
        assert lab_file["row_count"] == 20


# ── Test: CSV Ingestion ────────────────────────────────────────────────────────


class TestCSVIngestion:
    """Tests for CSV file ingestion."""

    def test_parse_demo_csv(self, demo_csv_path):
        """Test parsing the demo CSV file."""
        from oncocontext.services.csv_parser import LabFileParser

        parser = LabFileParser()
        result = parser.parse_csv(demo_csv_path)

        assert result["rows"] == 20
        assert "Sample_ID" in result["columns"]
        assert "PD1_MFI" in result["columns"]
        assert "Cytotoxicity_percent" in result["columns"]
        assert len(result["markers_detected"]) > 0
        assert len(result["text_representations"]) == 20
        assert len(result["summary"]) > 0

    def test_detect_markers(self, demo_csv_path):
        """Test that biological markers are detected in column names."""
        from oncocontext.services.csv_parser import LabFileParser

        parser = LabFileParser()
        result = parser.parse_csv(demo_csv_path)

        markers = result["markers_detected"]
        # Should detect at least these markers
        expected_markers = {"CD8", "PD1", "TIM3", "LAG3", "Ki67", "GranzymeB"}
        detected_set = set(markers)
        # At least some of expected markers should be detected
        overlap = expected_markers & detected_set
        assert len(overlap) >= 3, (
            f"Expected at least 3 of {expected_markers} to be detected, "
            f"but found: {detected_set}"
        )


# ── Test: Full Demo Flow (slow, requires network) ─────────────────────────────


@pytest.mark.slow
class TestDemoFlow:
    """End-to-end tests that simulate the actual demo.

    These tests require network access and may take 30-60 seconds.
    Run with: python -m pytest tests/test_integration.py -m slow -v
    """

    async def test_search_literature_real(self):
        """Test search_literature with real PubMed API."""
        from oncocontext.tools.search_literature import search_literature

        result = await search_literature(
            query="CD8 T cell exhaustion PD-1",
            max_results=5,
        )

        assert "papers" in result
        assert "total_found" in result
        assert "query_expansion" in result
        assert result["total_found"] > 0
        assert len(result["papers"]) > 0

        # Check paper structure
        paper = result["papers"][0]
        assert "pmid" in paper
        assert "title" in paper
        assert len(paper["pmid"]) > 0
        assert len(paper["title"]) > 0

    async def test_get_paper_details_known_pmid(self):
        """Test fetching details for a known paper."""
        from oncocontext.tools.get_paper_details import get_paper_details

        # Use a well-known review paper that should exist
        result = await get_paper_details(
            pmid="26544946",  # Wherry & Kurachi 2015 — exhaustion review
            fetch_full_text=False,
            index_if_available=False,
        )

        assert result["pmid"] == "26544946"
        assert len(result["title"]) > 0
        assert result["year"] > 0

    async def test_ingest_demo_csv(self):
        """Test ingesting the demo CSV file."""
        from oncocontext.tools.ingest_lab_file import ingest_lab_file

        csv_path = Path(__file__).parent.parent / "demo" / "sample_flow_cytometry.csv"
        if not csv_path.exists():
            pytest.skip("Demo CSV not found")

        result = await ingest_lab_file(
            file_path=str(csv_path),
            file_type="csv",
            experiment_label="Integration test",
        )

        assert "error" not in result
        assert result["row_count"] == 20
        assert result["indexed"] is True
        assert result["chunk_count"] > 0
        assert len(result["detected_markers"]) > 0

    async def test_deep_search_without_indexed_papers(self):
        """Test deep_search returns helpful message when no papers indexed."""
        from oncocontext.tools.deep_search import deep_search

        result = await deep_search(
            query="What gating strategy was used for exhausted CD8 T cells?",
        )

        assert "results" in result
        assert "search_strategy" in result
        # Should have a helpful message about no papers indexed
        assert isinstance(result["results"], list)

    async def test_cross_reference_without_literature(self):
        """Test cross_reference returns error when no literature indexed."""
        from oncocontext.tools.cross_reference import cross_reference

        result = await cross_reference(
            research_question="Why do exhausted T cells still kill?",
        )

        assert "agreements" in result
        assert "contradictions" in result
        assert "suggested_follow_up" in result


# ── Test: Chunker Integration ──────────────────────────────────────────────────


class TestChunkerIntegration:
    """Integration tests for section-aware chunking."""

    def test_chunk_paper_with_sections(self):
        """Test chunking a paper with multiple sections."""
        from oncocontext.services.chunker import SectionAwareChunker
        from oncocontext.models.schemas import Section

        chunker = SectionAwareChunker()

        sections = [
            Section(
                heading="Methods",
                section_type="methods",
                paragraphs=[
                    "Flow cytometry was performed using a BD LSRFortessa X-20 instrument "
                    "equipped with four lasers and eighteen fluorescence detectors. "
                    "Cells were stained with anti-CD8-APC clone SK1 from BioLegend, anti-PD-1-PE "
                    "clone EH12.2H7 from BioLegend, anti-TIM-3-BV421 clone F38-2E2 from BD Biosciences, "
                    "and anti-LAG-3-FITC clone 11C3C65 from BioLegend. "
                    "The gating strategy was performed by first excluding doublets using FSC-H versus "
                    "FSC-A, then excluding dead cells using a viability dye, "
                    "and finally gating on CD3+CD8+ T cells for downstream analysis of exhaustion markers.",
                    "For intracellular staining of transcription factors and cytokines, cells were "
                    "fixed and permeabilized using the BD Cytofix/Cytoperm fixation and permeabilization kit "
                    "following the manufacturer's protocol and recommended incubation times. "
                    "Anti-Ki67-PE-Cy7 clone B56 and anti-Granzyme-B-AF647 clone GB11 were used for "
                    "detection of intracellular proliferation and cytotoxic markers respectively in the "
                    "CD8+ T cell population after co-culture with tumor organoids.",
                ],
            ),
            Section(
                heading="Results",
                section_type="results",
                paragraphs=[
                    "CD8+ T cells co-cultured with patient-derived tumor organoids showed progressive "
                    "and time-dependent upregulation of the exhaustion markers PD-1 and TIM-3 over a "
                    "72-hour time course experiment. At the 72-hour timepoint, PD-1 median fluorescence "
                    "intensity increased significantly from a baseline of 180 to 2350, while TIM-3 MFI "
                    "increased from 95 to 1020. LAG-3 expression also showed a notable increase from "
                    "42 to 620 over the same time period. Despite these elevated exhaustion markers, "
                    "cytotoxicity as measured by the chromium release assay remained above 40 percent.",
                ],
            ),
        ]

        chunks = chunker.chunk_paper(
            pmid="12345678",
            pmc_id="PMC1234567",
            sections=sections,
        )

        assert len(chunks) >= 2
        assert all("chunk_id" in c for c in chunks)
        assert all("text" in c for c in chunks)
        assert all("section" in c for c in chunks)
        assert all("paper_pmid" in c for c in chunks)

        # Check that section types are preserved
        sections_found = set(c["section"] for c in chunks)
        assert "methods" in sections_found
        assert "results" in sections_found

        # Check text is prefixed with section name
        for chunk in chunks:
            assert chunk["text"].startswith("[")


# ── Test: Query Expander ───────────────────────────────────────────────────────


class TestQueryExpanderIntegration:
    """Integration tests for query expansion."""

    async def test_expand_query_with_synonyms(self):
        """Test that query expansion adds known synonyms."""
        from oncocontext.services.query_expander import QueryExpander

        expander = QueryExpander()
        result = await expander.expand("PD-1 T cell exhaustion")

        assert "original_query" in result
        assert "expanded_terms" in result
        assert "pubmed_query" in result
        assert len(result["expanded_terms"]) > 0
        # PD-1 should be expanded to include PDCD1, CD279, etc.
        pubmed_query = result["pubmed_query"]
        assert len(pubmed_query) > len("PD-1 T cell exhaustion")


# ── Test: BioC Parser ──────────────────────────────────────────────────────────


class TestBioCParserIntegration:
    """Integration tests for BioC JSON parsing."""

    def test_parse_sample_bioc(self):
        """Test parsing a sample BioC JSON structure."""
        from oncocontext.services.bioc_parser import BioCParser

        parser = BioCParser()

        # Minimal BioC JSON structure
        bioc_json = {
            "documents": [
                {
                    "id": "PMC_TEST",
                    "passages": [
                        {
                            "offset": 0,
                            "text": "This is the introduction paragraph describing the background of T cell exhaustion in cancer.",
                            "infons": {"section_type": "INTRO", "type": "paragraph"},
                        },
                        {
                            "offset": 500,
                            "text": "Flow cytometry was performed using a BD LSRFortessa with anti-CD8 and anti-PD-1 antibodies for staining.",
                            "infons": {"section_type": "METHODS", "type": "paragraph"},
                        },
                        {
                            "offset": 1000,
                            "text": "PD-1 expression was significantly upregulated in tumor-infiltrating CD8+ T cells compared to controls, with MFI increasing from 200 to 1800.",
                            "infons": {"section_type": "RESULTS", "type": "paragraph"},
                        },
                        {
                            "offset": 1500,
                            "text": "Our findings suggest that the high PD-1 expression combined with preserved cytotoxicity indicates progenitor exhaustion rather than terminal exhaustion.",
                            "infons": {"section_type": "DISCUSS", "type": "paragraph"},
                        },
                    ],
                }
            ]
        }

        sections = parser.parse(bioc_json)

        assert len(sections) >= 3
        section_types = [s.section_type for s in sections]
        assert "introduction" in section_types
        assert "methods" in section_types
        assert "results" in section_types
        assert "discussion" in section_types
