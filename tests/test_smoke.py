"""Smoke tests for OncoContext Phase 0 skeleton.

Verifies:
    - Package imports correctly
    - Config values are correct
    - All Pydantic models can be instantiated
    - MCP server can be created (not run)
    - All stub tools return placeholder responses
"""

import asyncio
import json


def test_package_imports():
    """Package imports and version string exists."""
    import oncocontext

    assert hasattr(oncocontext, "__version__")
    assert oncocontext.__version__ == "0.1.0"


def test_config_values():
    """Config values match the MVP plan specifications."""
    from oncocontext.config import settings

    # API endpoints
    assert settings.PUBMED_BASE_URL == "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    assert "ncbi.nlm.nih.gov" in settings.PMC_BIOC_URL

    # Embedding model
    assert settings.EMBEDDING_MODEL == "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
    assert settings.EMBEDDING_DIM == 768

    # Reranker
    assert settings.RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking
    assert settings.CHUNK_SIZE == 384
    assert settings.CHUNK_OVERLAP == 64

    # Vector search
    assert settings.VECTOR_TOP_K == 50
    assert settings.RERANK_TOP_K == 10

    # Relevance scoring weights (should sum to 1.0)
    total = (
        settings.WEIGHT_SEMANTIC
        + settings.WEIGHT_TITLE
        + settings.WEIGHT_MESH
        + settings.WEIGHT_RECENCY
        + settings.WEIGHT_JOURNAL
        + settings.WEIGHT_DISCOVERY
    )
    assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"

    assert settings.RELEVANCE_THRESHOLD == 0.40

    # Section boosts
    assert settings.BOOST_METHODS == 1.5
    assert settings.BOOST_RESULTS == 1.3
    assert settings.BOOST_DISCUSSION == 1.5

    # Cache TTLs
    assert settings.CACHE_TTL_PUBMED == 86400     # 24h
    assert settings.CACHE_TTL_PMC == 2592000       # 30d

    # Rate limits
    assert settings.PUBMED_MAX_RESULTS == 50
    assert settings.PUBMED_RATE_LIMIT == 3


def test_pydantic_models():
    """All Pydantic models can be instantiated with defaults."""
    from oncocontext.models.schemas import (
        Agreement,
        Chunk,
        Citation,
        ColumnInfo,
        Contradiction,
        CrossReferenceResult,
        DeepSearchChunkResult,
        DeepSearchResult,
        LabFileInfo,
        PaperDetails,
        PaperResult,
        QueryExpansion,
        SearchResult,
        Section,
        SurroundingContext,
    )

    # Instantiate each model with minimal required fields
    citation = Citation(pmid="12345678")
    assert citation.pmid == "12345678"
    assert citation.section == ""

    paper = PaperResult(pmid="12345678")
    assert paper.relevance_score == 0.0
    assert paper.mesh_terms == []

    section = Section(heading="Methods", section_type="methods", paragraphs=["p1", "p2"])
    assert len(section.paragraphs) == 2

    details = PaperDetails(pmid="12345678")
    assert details.sections is None
    assert details.indexed is False

    qe = QueryExpansion()
    assert qe.expanded_terms == []

    sr = SearchResult()
    assert sr.papers == []
    assert sr.total_found == 0

    ctx = SurroundingContext()
    assert ctx.previous_paragraph is None

    chunk = Chunk(chunk_id="c1", paper_pmid="12345678", text="some text")
    assert chunk.score == 0.0

    ds_chunk = DeepSearchChunkResult()
    assert ds_chunk.relevance_score == 0.0

    ds_result = DeepSearchResult()
    assert ds_result.results == []

    col = ColumnInfo(name="CD8_percent", dtype="float64")
    assert col.stats is None

    lab = LabFileInfo(file_id="lab_001")
    assert lab.detected_markers == []

    agreement = Agreement()
    assert agreement.confidence == "moderate"

    contradiction = Contradiction()
    assert contradiction.possible_explanations == []

    cr = CrossReferenceResult()
    assert cr.agreements == []
    assert cr.novel_findings == []


def test_mcp_server_creation():
    """MCP server can be created (not run)."""
    from oncocontext.server import mcp

    assert mcp is not None
    assert mcp.name == "OncoContext"


def test_tool_search_literature_stub():
    """search_literature stub returns placeholder data."""
    from oncocontext.tools.search_literature import search_literature

    result = asyncio.get_event_loop().run_until_complete(
        search_literature(query="T cell exhaustion")
    )
    assert "papers" in result
    assert "total_found" in result
    assert "query_expansion" in result
    assert len(result["papers"]) > 0
    assert result["papers"][0]["pmid"] is not None


def test_tool_get_paper_details_stub():
    """get_paper_details stub returns placeholder data."""
    from oncocontext.tools.get_paper_details import get_paper_details

    result = asyncio.get_event_loop().run_until_complete(
        get_paper_details(pmid="12345678")
    )
    assert result["pmid"] == "12345678"
    assert "sections" in result
    assert "indexed" in result


def test_tool_deep_search_stub():
    """deep_search returns expected structure (empty when no papers indexed)."""
    from oncocontext.tools.deep_search import deep_search

    result = asyncio.get_event_loop().run_until_complete(
        deep_search(query="antibody panel gating strategy")
    )
    assert "results" in result
    assert "total_indexed_papers" in result
    assert "search_strategy" in result
    # No papers indexed → results should be empty with a helpful message
    assert isinstance(result["results"], list)


def test_tool_ingest_lab_file_signature():
    """ingest_lab_file function has correct signature and handles bad path."""
    from oncocontext.tools.ingest_lab_file import ingest_lab_file
    import inspect

    sig = inspect.signature(ingest_lab_file)
    assert "file_path" in sig.parameters
    assert "file_type" in sig.parameters
    assert "experiment_label" in sig.parameters

    # Calling with a non-existent file should return an error dict (not raise)
    result = asyncio.get_event_loop().run_until_complete(
        ingest_lab_file(file_path="nonexistent_file_xyz.csv")
    )
    assert "error" in result
    assert "not found" in result["error"].lower() or "not found" in str(result).lower()


def test_tool_cross_reference_no_literature():
    """cross_reference returns error when no literature is indexed."""
    from oncocontext.tools.cross_reference import cross_reference

    result = asyncio.get_event_loop().run_until_complete(
        cross_reference(research_question="Why do my exhausted T cells still kill?")
    )
    # With no literature indexed, should return structured response (possibly with error)
    assert "summary" in result or "error" in result
    assert "agreements" in result
    assert "contradictions" in result
    assert "novel_findings" in result
    assert "suggested_follow_up" in result


def test_service_classes_importable():
    """All service classes can be imported."""
    from oncocontext.services import (
        PubMedClient,
        PMCClient,
        BioCParser,
        QueryExpander,
        SectionAwareChunker,
        Embedder,
        Reranker,
        LabFileParser,
    )

    # Verify they can be instantiated
    assert PubMedClient() is not None
    assert PMCClient() is not None
    assert BioCParser() is not None
    assert QueryExpander() is not None
    assert SectionAwareChunker() is not None
    assert Embedder() is not None
    assert Reranker() is not None
    assert LabFileParser() is not None


def test_storage_classes_importable():
    """All storage classes can be imported."""
    from oncocontext.storage import (
        ChromaManager,
        SQLiteManager,
        CacheManager,
    )

    assert ChromaManager() is not None
    assert SQLiteManager() is not None
    assert CacheManager() is not None


def test_synonym_dict_exists():
    """Synonym dictionary file exists and is valid JSON."""
    from oncocontext.config import settings

    path = settings.SYNONYM_DICT_PATH
    assert path.exists(), f"synonym_dict.json not found at {path}"

    with open(path) as f:
        data = json.load(f)

    assert isinstance(data, dict)
    assert len(data) >= 50, f"Expected >=50 entries, got {len(data)}"
    assert "PD-1" in data
    assert "PDCD1" in data["PD-1"]


def test_demo_csv_exists():
    """Demo flow cytometry CSV exists and has expected structure."""
    from pathlib import Path

    csv_path = Path(__file__).parent.parent / "demo" / "sample_flow_cytometry.csv"
    assert csv_path.exists(), f"Demo CSV not found at {csv_path}"

    with open(csv_path) as f:
        lines = f.readlines()

    # Header + 20 data rows
    assert len(lines) >= 21, f"Expected >=21 lines, got {len(lines)}"

    header = lines[0].strip().split(",")
    assert "Sample_ID" in header
    assert "PD1_MFI" in header
    assert "Cytotoxicity_percent" in header
