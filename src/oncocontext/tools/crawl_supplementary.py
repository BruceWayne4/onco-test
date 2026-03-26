"""crawl_supplementary tool — Fetch and parse supplementary files from PMC/PubMed papers.

Discovers supplementary materials attached to a paper and parses common formats:
CSV, TSV, XLSX, XML, PDF, DOCX, JSON, and plain text.
"""

from __future__ import annotations

import logging
from typing import Optional

from oncocontext.services.supplementary_crawler import SupplementaryCrawler

logger = logging.getLogger(__name__)

# ── Module-level lazy singleton ────────────────────────────────────────────────

_crawler: SupplementaryCrawler | None = None


def _get_crawler() -> SupplementaryCrawler:
    global _crawler
    if _crawler is None:
        _crawler = SupplementaryCrawler()
    return _crawler


# ── Main Tool Function ─────────────────────────────────────────────────────────


async def crawl_supplementary(
    pmid: Optional[str] = None,
    pmc_id: Optional[str] = None,
    direct_url: Optional[str] = None,
    file_types: Optional[str] = None,
    max_files: int = 10,
) -> dict:
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
    if not pmid and not pmc_id and not direct_url:
        return {
            "status": "error",
            "message": "Provide at least one of: pmid, pmc_id, or direct_url",
        }

    crawler = _get_crawler()

    filter_types: list[str] | None = None
    if file_types:
        filter_types = [ft.strip().lower() for ft in file_types.split(",")]

    result = await crawler.crawl(
        pmid=pmid,
        pmc_id=pmc_id,
        direct_url=direct_url,
        file_types=filter_types,
        max_files=max_files,
    )
    return result
