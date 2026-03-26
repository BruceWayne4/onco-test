"""crawl_and_report MCP tool — full paper crawl + comprehensive Markdown report.

Orchestrates:
    1. FullPaperCrawler  → fetch + persist all paper content locally
    2. ReportWriter      → assemble a complete Markdown report

Returns a result dict containing the full Markdown string, local paths,
supplementary index, references, and total character count.
"""

from __future__ import annotations

import logging
from typing import Optional

from oncocontext.services.full_paper_crawler import FullPaperCrawler
from oncocontext.services.report_writer import ReportWriter

logger = logging.getLogger(__name__)

# ── Module-level lazy singletons ──────────────────────────────────────────────

_crawler: FullPaperCrawler | None = None
_writer: ReportWriter | None = None


def _get_crawler() -> FullPaperCrawler:
    global _crawler
    if _crawler is None:
        _crawler = FullPaperCrawler()
    return _crawler


def _get_writer() -> ReportWriter:
    global _writer
    if _writer is None:
        _writer = ReportWriter()
    return _writer


# ── Main Tool Function ─────────────────────────────────────────────────────────


async def crawl_and_report(
    pmcid_or_pmid: str,
    clinical_question: Optional[str] = None,
    crawl_supplementary: bool = True,
    max_supp_files: int = 20,
) -> dict:
    """Crawl a paper's full text and supplementary data, then generate a comprehensive report.

    Fetches the complete PMC full text (BioC XML), crawls ALL supplementary
    materials, saves everything to data/crawled/<pmcid>/, and assembles a
    full non-truncated Markdown clinical evidence report.

    Args:
        pmcid_or_pmid: A PMC ID (e.g. "PMC8650059") or PubMed ID (e.g. "34789550").
        clinical_question: Optional clinical question to frame the report.
        crawl_supplementary: Whether to fetch and save supplementary files (default True).
        max_supp_files: Maximum number of supplementary files to fetch (default 20).

    Returns:
        Dict with:
            local_data_path      — path to data/crawled/<pmcid>/
            report_path          — path to data/reports/<pmcid>_report.md
            report_markdown      — FULL Markdown content of the report (no truncation)
            supplementary_index  — dict of supplementary files with URLs and local paths
            references           — list of reference objects
            total_chars_crawled  — total character count of all crawled text
            sections_found       — list of section headings extracted from full text
            pmid                 — resolved PubMed ID (may be None if not found)
            pmc_id               — resolved PMC ID (may be None if not in OA subset)
            errors               — list of non-fatal error/warning strings
    """
    if not pmcid_or_pmid or not pmcid_or_pmid.strip():
        return {
            "status": "error",
            "message": "pmcid_or_pmid must be a non-empty string (PMC ID or PubMed ID)",
        }

    logger.info(
        "crawl_and_report: pmcid_or_pmid=%r, clinical_question=%r",
        pmcid_or_pmid,
        (clinical_question or "")[:80],
    )

    # ── Step 1: Full paper crawl ───────────────────────────────────────────────
    crawler = _get_crawler()
    crawl_result = await crawler.crawl(
        pmcid_or_pmid=pmcid_or_pmid,
        crawl_supplementary=crawl_supplementary,
        max_supp_files=max_supp_files,
    )

    # ── Step 2: Write Markdown report ─────────────────────────────────────────
    writer = _get_writer()
    report_path = writer.write_report(
        pmcid_or_pmid=pmcid_or_pmid,
        crawl_result=crawl_result,
        clinical_question=clinical_question,
    )

    # ── Step 3: Load the full report Markdown (no truncation) ─────────────────
    try:
        from pathlib import Path
        report_markdown = Path(report_path).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Could not read back report file %s: %s", report_path, exc)
        report_markdown = ""

    # ── Step 4: Assemble result ────────────────────────────────────────────────
    return {
        "local_data_path": crawl_result.get("local_data_path", ""),
        "report_path": report_path,
        "report_markdown": report_markdown,
        "supplementary_index": crawl_result.get("supplementary_index", {}),
        "references": crawl_result.get("references", []),
        "total_chars_crawled": crawl_result.get("total_chars_crawled", 0),
        "sections_found": crawl_result.get("sections_found", []),
        "pmid": crawl_result.get("pmid"),
        "pmc_id": crawl_result.get("pmc_id"),
        "errors": crawl_result.get("errors", []),
    }
