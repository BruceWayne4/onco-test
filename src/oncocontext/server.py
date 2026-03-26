"""MCP server setup, tool registration, and stdio transport for OncoContext.

Registers all 7 tools with the MCP server using @mcp.tool() decorators.
Each tool delegates to its implementation in the tools/ package.

Services (Embedder, ChromaManager, SQLiteManager, SectionAwareChunker)
are lazily initialized within each tool module — no global init required here.
"""

import json
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from oncocontext.tools.search_literature import search_literature
from oncocontext.tools.get_paper_details import get_paper_details
from oncocontext.tools.deep_search import deep_search
from oncocontext.tools.ingest_lab_file import ingest_lab_file
from oncocontext.tools.cross_reference import cross_reference
from oncocontext.tools.crawl_supplementary import crawl_supplementary as _crawl_supplementary_fn
from oncocontext.tools.crawl_and_report import crawl_and_report as _crawl_and_report_fn
from oncocontext.tools.get_next_page import get_next_page as _get_next_page_fn
from oncocontext.services.response_paginator import get_paginator

# ── Logging Configuration ──────────────────────────────────────────────────────

# Resolve data/logs/ relative to the repo root (two levels up from this file)
_repo_root = Path(__file__).parent.parent.parent
_log_dir = _repo_root / "data" / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)

_log_fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_date_fmt_file = "%Y-%m-%d %H:%M:%S"
_date_fmt_console = "%H:%M:%S"

# stderr handler — real-time stream visible in Roo's Output panel
_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setFormatter(logging.Formatter(_log_fmt, datefmt=_date_fmt_console))

# rotating file handler — writes to data/logs/mcp.log (5 MB max, 3 backups)
_file_handler = RotatingFileHandler(
    _log_dir / "mcp.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(_log_fmt, datefmt=_date_fmt_file))

logging.basicConfig(
    level=logging.INFO,
    handlers=[_stderr_handler, _file_handler],
)

logger = logging.getLogger(__name__)

# Create the MCP server instance
mcp = FastMCP(
    "OncoContext",
    instructions=(
        "OncoContext — Oncology literature deep search & lab data cross-reference.\n\n"
        "This server provides 8 tools for oncology researchers:\n"
        "1. search_literature: Find papers on PubMed with smart synonym expansion\n"
        "2. get_paper_details: Fetch paper metadata and index full text for deep search\n"
        "3. deep_search: Search inside Methods, Results, Discussion sections of indexed papers\n"
        "4. ingest_lab_file: Upload and analyze CSV/Excel lab data (stays local, never sent externally)\n"
        "5. cross_reference: Compare your lab data against published literature with citations\n"
        "6. crawl_supplementary: Fetch and parse supplementary files (CSV, XLSX, PDF, etc.) from papers\n"
        "7. crawl_and_report: Full paper crawl + save all content locally + generate comprehensive Markdown report\n"
        "8. get_next_page: Retrieve subsequent pages when a tool response was paginated (see _pagination block)\n\n"
        "Typical workflow: search_literature → get_paper_details (indexes full text) → deep_search.\n"
        "For supplementary data: crawl_supplementary with pmid or pmc_id.\n"
        "For lab comparison: ingest_lab_file → cross_reference.\n"
        "For full offline archiving + report: crawl_and_report with pmcid_or_pmid.\n\n"
        "PAGINATION: Large responses are automatically split into pages. When a response contains\n"
        "a '_pagination' block with has_more=true, call get_next_page(session_id=..., page=N+1)\n"
        "to retrieve the next page. Sessions expire after 30 minutes.\n\n"
        "All citations include PMID + section + paragraph number for verification."
    ),
)


# ── Pagination helper ─────────────────────────────────────────────────────────


def _paginate(output: str, tool_name: str) -> str:
    """Apply response-size pagination to a JSON string if it exceeds the limit."""
    try:
        return get_paginator().paginate_if_needed(output, tool_name=tool_name)
    except Exception as exc:
        logger.error("Pagination middleware error for '%s': %s", tool_name, exc)
        return output  # Return raw output rather than failing the tool call entirely


# ── Tool 1: search_literature ─────────────────────────────────────────────────


@mcp.tool()
async def tool_search_literature(
    query: str,
    max_results: int = 20,
    date_range: str | None = None,
    full_text_only: bool = False,
) -> str:
    """Search PubMed for oncology papers with synonym expansion and relevance scoring.

    Discovers relevant papers with abstracts, MeSH terms, and relevance scores.
    Automatically expands queries with ontology synonyms (PD-1 → PDCD1, CD279).
    Returns papers ranked by a composite score including semantic similarity,
    keyword overlap, MeSH terms, recency, and journal impact.

    Args:
        query: Natural language research question or structured search query.
            Examples: "CD8 T cell exhaustion in tumor organoids"
                     "PD-1 TIM-3 co-expression gating strategy"
        max_results: Maximum number of papers to return (1-100, default 20).
        date_range: Filter by publication date range, e.g. '2020-2025'.
        full_text_only: Only return papers with PMC full text available.
    """
    logger.info("search_literature called: query=%r, max_results=%d", query[:80], max_results)
    start = time.monotonic()
    try:
        result = await search_literature(
            query=query,
            max_results=max_results,
            date_range=date_range,
            full_text_only=full_text_only,
        )
        output = json.dumps(result, indent=2, default=str)
        duration = time.monotonic() - start
        output = _paginate(output, "search_literature")
        logger.info(
            "search_literature completed in %.2fs, output size=%d bytes",
            duration,
            len(output),
        )
        return output
    except Exception:
        logger.exception("search_literature tool failed")
        raise


# ── Tool 2: get_paper_details ─────────────────────────────────────────────────


@mcp.tool()
async def tool_get_paper_details(
    pmid: str,
    fetch_full_text: bool = True,
    sections: list[str] | None = None,
    index_if_available: bool = True,
) -> str:
    """Fetch full metadata for a paper by PMID, optionally fetch full text and index it.

    Retrieves paper details from PubMed. If PMC full text is available and
    fetch_full_text is True, fetches the full text, parses it into sections
    (Methods, Results, Discussion, etc.), chunks it, embeds it with PubMedBERT,
    and indexes it in ChromaDB for subsequent deep_search queries.

    This is how you make papers searchable — call this before deep_search.

    Args:
        pmid: PubMed ID of the paper (e.g., "38123456").
        fetch_full_text: Whether to fetch and index full text from PMC (default True).
        sections: Filter to specific sections (e.g. ['methods', 'results']).
            Valid values: abstract, introduction, methods, results, discussion, conclusion.
        index_if_available: Whether to index the paper into ChromaDB (default True).
    """
    logger.info("get_paper_details called: pmid=%s, fetch_full_text=%s", pmid, fetch_full_text)
    start = time.monotonic()
    try:
        result = await get_paper_details(
            pmid=pmid,
            fetch_full_text=fetch_full_text,
            sections=sections,
            index_if_available=index_if_available,
        )
        output = json.dumps(result, indent=2, default=str)
        duration = time.monotonic() - start
        output = _paginate(output, "get_paper_details")
        logger.info(
            "get_paper_details completed in %.2fs, output size=%d bytes",
            duration,
            len(output),
        )
        return output
    except Exception:
        logger.exception("get_paper_details tool failed")
        raise


# ── Tool 3: deep_search ───────────────────────────────────────────────────────


@mcp.tool()
async def tool_deep_search(
    query: str,
    search_scope: str = "all",
    max_results: int = 10,
    min_relevance: float = 0.3,
    paper_ids: list[str] | None = None,
) -> str:
    """Semantic search across full-text papers indexed in ChromaDB.

    Searches inside Methods, Results, and Discussion sections of indexed papers.
    Returns specific paragraphs with full citations (PMID + section + paragraph).
    Uses PubMedBERT for semantic embedding and cross-encoder reranking for precision.

    IMPORTANT: Papers must be indexed first using get_paper_details with
    fetch_full_text=True. Pre-indexed papers are searched instantly (<500ms).

    Args:
        query: Detailed mechanistic question to search for in paper full text.
            Examples: "What gating strategy was used for exhausted CD8+ T cells?"
                     "What antibody clones were used for the PD-1 TIM-3 panel?"
        search_scope: Restrict to a section type. Options: 'all' (default),
            'methods', 'results', 'discussion', 'introduction'.
        max_results: Number of top chunks to return after reranking (1-25, default 10).
        min_relevance: Minimum relevance score threshold (0.0-1.0, default 0.3).
        paper_ids: Restrict search to specific PMIDs. None searches all indexed papers.
    """
    logger.info("deep_search called: query=%r, scope=%s", query[:80], search_scope)
    start = time.monotonic()
    try:
        result = await deep_search(
            query=query,
            search_scope=search_scope,
            max_results=max_results,
            min_relevance=min_relevance,
            paper_ids=paper_ids,
        )
        output = json.dumps(result, indent=2, default=str)
        duration = time.monotonic() - start
        output = _paginate(output, "deep_search")
        logger.info("deep_search completed in %.2fs, output size=%d bytes", duration, len(output))
        return output
    except Exception:
        logger.exception("deep_search tool failed")
        raise


# ── Tool 4: ingest_lab_file ───────────────────────────────────────────────────


@mcp.tool()
async def tool_ingest_lab_file(
    file_path: str,
    file_type: str = "auto",
    experiment_label: str | None = None,
    metadata: dict | None = None,
) -> str:
    """Ingest a CSV or Excel file of lab data for cross-referencing with literature.

    Parses the file, detects biological markers (CD8, PD-1, TIM-3, LAG-3, etc.),
    computes summary statistics, generates text representations, and indexes
    the data locally in ChromaDB for use with cross_reference.

    PRIVACY: Your data NEVER leaves your machine. No external API calls are made
    with your lab data. Everything is processed and stored locally.

    Args:
        file_path: Absolute or relative path to the CSV or Excel file on disk.
            Example: "demo/sample_flow_cytometry.csv"
        file_type: File type ('csv', 'excel', 'auto'). Auto detects from extension.
        experiment_label: Researcher's label for this experiment.
            Example: "CD8 exhaustion panel 2024-01"
        metadata: Additional context as key-value pairs.
            Example: {"cell_line": "A375", "treatment": "anti-PD1", "timepoint": "72h"}
    """
    logger.info("ingest_lab_file called: file_path=%r", file_path)
    start = time.monotonic()
    try:
        result = await ingest_lab_file(
            file_path=file_path,
            file_type=file_type,
            experiment_label=experiment_label,
            metadata=metadata,
        )
        output = json.dumps(result, indent=2, default=str)
        duration = time.monotonic() - start
        output = _paginate(output, "ingest_lab_file")
        logger.info("ingest_lab_file completed in %.2fs, output size=%d bytes", duration, len(output))
        return output
    except Exception:
        logger.exception("ingest_lab_file tool failed")
        raise


# ── Tool 5: cross_reference ───────────────────────────────────────────────────


@mcp.tool()
async def tool_cross_reference(
    research_question: str,
    lab_file_ids: list[str] | None = None,
    paper_ids: list[str] | None = None,
    comparison_type: str = "findings",
) -> str:
    """Cross-reference your lab data against indexed literature — THE KILLER FEATURE.

    Compares your private lab data against published findings and returns:
    - Agreements: Where your data matches published results (with citations)
    - Contradictions: Where your data differs from published results (with explanations)
    - Novel findings: Observations with no clear match in literature
    - Suggested follow-ups: Actionable next steps based on the analysis

    Every claim includes a paragraph-level citation (PMID + section + paragraph).

    Requires: At least some papers indexed (via get_paper_details) and ideally
    lab data ingested (via ingest_lab_file). Works with literature only if no
    lab data is available.

    Args:
        research_question: The specific comparison question.
            Example: "My T cells show high PD-1 but preserved cytotoxicity —
            why does this contradict the exhaustion literature?"
        lab_file_ids: IDs of ingested lab files to include. None uses all lab files.
        paper_ids: Restrict comparison to specific PMIDs. None uses all indexed papers.
        comparison_type: Compare 'findings' (default), 'methods', or 'both'.
    """
    logger.info("cross_reference called: question=%r", research_question[:80])
    start = time.monotonic()
    try:
        result = await cross_reference(
            research_question=research_question,
            lab_file_ids=lab_file_ids,
            paper_ids=paper_ids,
            comparison_type=comparison_type,
        )
        output = json.dumps(result, indent=2, default=str)
        duration = time.monotonic() - start
        output = _paginate(output, "cross_reference")
        logger.info("cross_reference completed in %.2fs, output size=%d bytes", duration, len(output))
        return output
    except Exception:
        logger.exception("cross_reference tool failed")
        raise


# ── Tool 6: crawl_supplementary ───────────────────────────────────────────────


@mcp.tool()
async def crawl_supplementary(
    pmid: Optional[str] = None,
    pmc_id: Optional[str] = None,
    direct_url: Optional[str] = None,
    file_types: Optional[str] = None,
    max_files: int = 10,
) -> str:
    """Crawl and extract content from supplementary data files attached to a scientific paper.

    Fetches supplementary materials (tables, protocols, datasets) from PMC/journal pages
    and parses common formats: CSV, TSV, XLSX (Excel), XML, PDF, DOCX, JSON, and plain text.

    Use this tool when a paper references supplementary tables or files that contain
    important data not present in the main text (e.g., complete antibody panels,
    gating strategies, full datasets, extended methods).

    Args:
        pmid: PubMed ID of the paper (e.g., "34789550")
        pmc_id: PubMed Central ID (e.g., "PMC8650059") — preferred over pmid when available
        direct_url: Direct URL to a specific supplementary file to parse
        file_types: Comma-separated list of file types to include (e.g., "csv,xlsx,pdf").
                   If not specified, all types are attempted.
        max_files: Maximum number of supplementary files to fetch (default: 10)

    Returns structured content extracted from supplementary files.
    """
    logger.info(
        "crawl_supplementary called: pmid=%s, pmc_id=%s, direct_url=%s",
        pmid, pmc_id, direct_url,
    )
    start = time.monotonic()
    try:
        result = await _crawl_supplementary_fn(
            pmid=pmid,
            pmc_id=pmc_id,
            direct_url=direct_url,
            file_types=file_types,
            max_files=max_files,
        )
        output = json.dumps(result, indent=2, default=str)
        duration = time.monotonic() - start
        output = _paginate(output, "crawl_supplementary")
        logger.info("crawl_supplementary completed in %.2fs, output size=%d bytes", duration, len(output))
        return output
    except Exception:
        logger.exception("crawl_supplementary tool failed")
        raise


# ── Tool 7: crawl_and_report ──────────────────────────────────────────────────


@mcp.tool()
async def crawl_and_report(
    pmcid_or_pmid: str,
    clinical_question: Optional[str] = None,
    crawl_supplementary_files: bool = True,
    max_supp_files: int = 20,
) -> str:
    """Crawl a paper's full text and supplementary data, then generate a comprehensive Markdown report.

    Fetches complete PMC full text (BioC XML format), crawls ALL supplementary
    materials (tables, figures, PDFs, Excel files, CSVs), saves everything to
    data/crawled/<pmcid>/, and assembles a full non-truncated Markdown clinical
    evidence report saved to data/reports/<pmcid>_report.md.

    Use this when you need the COMPLETE paper content archived locally with a
    structured evidence report — not just abstract-level search results.

    Args:
        pmcid_or_pmid: A PMC ID (e.g. "PMC8650059") or PubMed ID (e.g. "34789550").
        clinical_question: Optional clinical question to frame the evidence report.
            Example: "What gating strategies were used for T cell exhaustion markers?"
        crawl_supplementary_files: Whether to fetch and save supplementary files (default True).
        max_supp_files: Maximum number of supplementary files to fetch (default 20).

    Returns structured result with:
        - local_data_path: path to data/crawled/<pmcid>/
        - report_path: path to data/reports/<pmcid>_report.md
        - report_markdown: FULL Markdown content of the report (no truncation)
        - supplementary_index: dict of supplementary files with URLs and local paths
        - references: list of reference objects
        - total_chars_crawled: total character count of all crawled text
    """
    logger.info(
        "crawl_and_report called: pmcid_or_pmid=%s, question=%r",
        pmcid_or_pmid,
        (clinical_question or "")[:80],
    )
    start = time.monotonic()
    try:
        result = await _crawl_and_report_fn(
            pmcid_or_pmid=pmcid_or_pmid,
            clinical_question=clinical_question,
            crawl_supplementary=crawl_supplementary_files,
            max_supp_files=max_supp_files,
        )
        output = json.dumps(result, indent=2, default=str)
        duration = time.monotonic() - start
        output = _paginate(output, "crawl_and_report")
        logger.info("crawl_and_report completed in %.2fs, output size=%d bytes", duration, len(output))
        return output
    except Exception:
        logger.exception("crawl_and_report tool failed")
        raise


# ── Tool 8: get_next_page ─────────────────────────────────────────────────────


@mcp.tool()
async def tool_get_next_page(session_id: str, page: int) -> str:
    """Retrieve the next page of a paginated MCP tool response.

    When an OncoContext tool response exceeds the ~900KB size limit it is
    automatically split into pages. Page 1 is returned directly with a
    ``_pagination`` metadata block containing ``session_id``, ``page``,
    ``total_pages``, and ``has_more``.

    Call this tool to retrieve pages 2, 3, … until ``has_more`` is ``false``.
    Sessions expire 30 minutes after the original tool call.

    Args:
        session_id: The ``session_id`` from the ``_pagination`` block of the
            previous page response.
        page: 1-based page number to retrieve (e.g. 2 for the second page).
    """
    logger.info("get_next_page called: session_id=%s, page=%d", session_id, page)
    start = time.monotonic()
    try:
        output = await _get_next_page_fn(session_id=session_id, page=page)
        duration = time.monotonic() - start
        logger.info("get_next_page completed in %.2fs, output size=%d bytes", duration, len(output))
        return output
    except Exception:
        logger.exception("get_next_page tool failed")
        raise


# ── Entry Point ────────────────────────────────────────────────────────────────


def main() -> None:
    """Run the OncoContext MCP server via stdio transport."""
    logger.info("Starting OncoContext MCP server...")
    logger.info("8 tools registered: search_literature, get_paper_details, "
                "deep_search, ingest_lab_file, cross_reference, crawl_supplementary, "
                "crawl_and_report, get_next_page")
    mcp.run()


if __name__ == "__main__":
    main()
