"""Pydantic models for all OncoContext data types.

Covers: papers, chunks, lab files, search results, cross-reference results,
citations, agreements, contradictions, and explanations.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Citations ──────────────────────────────────────────────────────────────────


class Citation(BaseModel):
    """A citation pointing to a specific paragraph within a paper."""

    pmid: str = ""
    pmc_id: str | None = None
    paper_title: str = ""
    authors: list[str] = Field(default_factory=list)
    journal: str = ""
    year: int = 0
    section: str = ""
    subsection: str | None = None
    paragraph_index: int = 0


# ── Paper Models ───────────────────────────────────────────────────────────────


class PaperResult(BaseModel):
    """A paper returned from a literature search."""

    pmid: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    journal: str = ""
    year: int = 0
    abstract: str = ""
    has_full_text: bool = False
    pmc_id: str | None = None
    relevance_score: float = 0.0
    mesh_terms: list[str] = Field(default_factory=list)
    is_indexed: bool = False


class Section(BaseModel):
    """A section of a paper's full text."""

    heading: str = ""
    section_type: str = ""
    paragraphs: list[str] = Field(default_factory=list)


class PaperDetails(BaseModel):
    """Extended paper details including full-text sections and index status."""

    pmid: str
    pmc_id: str | None = None
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    journal: str = ""
    year: int = 0
    abstract: str = ""
    mesh_terms: list[str] = Field(default_factory=list)
    has_full_text: bool = False
    sections: list[Section] | None = None
    indexed: bool = False
    chunk_count: int = 0


# ── Search Result Models ───────────────────────────────────────────────────────


class QueryExpansion(BaseModel):
    """Details about how a query was expanded with synonyms."""

    original_query: str = ""
    expanded_terms: list[str] = Field(default_factory=list)
    pubmed_query: str = ""


class SearchResult(BaseModel):
    """Result from search_literature tool."""

    papers: list[PaperResult] = Field(default_factory=list)
    total_found: int = 0
    query_expansion: QueryExpansion = Field(default_factory=QueryExpansion)


# ── Chunk Models ───────────────────────────────────────────────────────────────


class SurroundingContext(BaseModel):
    """Surrounding paragraphs for context assembly."""

    previous_paragraph: str | None = None
    next_paragraph: str | None = None


class Chunk(BaseModel):
    """A text chunk from a paper, with metadata and relevance score."""

    chunk_id: str = ""
    paper_pmid: str = ""
    pmc_id: str | None = None
    section: str = ""
    subsection: str | None = None
    paragraph_index: int = 0
    text: str = ""
    token_count: int = 0
    score: float = 0.0


class DeepSearchChunkResult(BaseModel):
    """A single result from deep_search, including citation and context."""

    chunk_text: str = ""
    citation: Citation = Field(default_factory=Citation)
    relevance_score: float = 0.0
    surrounding_context: SurroundingContext = Field(default_factory=SurroundingContext)


class DeepSearchResult(BaseModel):
    """Result from deep_search tool."""

    results: list[DeepSearchChunkResult] = Field(default_factory=list)
    total_indexed_papers: int = 0
    total_indexed_chunks: int = 0
    search_strategy: str = ""


# ── Lab File Models ────────────────────────────────────────────────────────────


class ColumnInfo(BaseModel):
    """Statistics and metadata for a single column in a lab file."""

    name: str = ""
    dtype: str = ""
    non_null_count: int = 0
    sample_values: list = Field(default_factory=list)
    stats: dict | None = None  # mean, median, min, max, std for numeric columns


class LabFileInfo(BaseModel):
    """Result from ingest_lab_file tool."""

    file_id: str = ""
    file_name: str = ""
    file_type_detected: str = ""
    row_count: int = 0
    column_count: int = 0
    columns: list[ColumnInfo] = Field(default_factory=list)
    summary: str = ""
    detected_markers: list[str] = Field(default_factory=list)
    indexed: bool = False
    chunk_count: int = 0


# ── Cross-Reference Models ────────────────────────────────────────────────────


class Agreement(BaseModel):
    """An agreement between lab data and literature."""

    lab_finding: str = ""
    literature_support: str = ""
    citation: Citation = Field(default_factory=Citation)
    confidence: str = "moderate"  # "strong" | "moderate" | "weak"


class Contradiction(BaseModel):
    """A contradiction between lab data and literature."""

    lab_finding: str = ""
    literature_contradiction: str = ""
    citation: Citation = Field(default_factory=Citation)
    possible_explanations: list[str] = Field(default_factory=list)


class CrossReferenceResult(BaseModel):
    """Result from cross_reference tool."""

    summary: str = ""
    lab_data_summary: str = ""
    agreements: list[Agreement] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    novel_findings: list[str] = Field(default_factory=list)
    suggested_follow_up: list[str] = Field(default_factory=list)
    papers_consulted: int = 0
    chunks_analyzed: int = 0
