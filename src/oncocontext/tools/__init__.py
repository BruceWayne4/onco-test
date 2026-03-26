"""Tool registry — exports all 7 MCP tool functions."""

from oncocontext.tools.search_literature import search_literature
from oncocontext.tools.get_paper_details import get_paper_details
from oncocontext.tools.deep_search import deep_search
from oncocontext.tools.ingest_lab_file import ingest_lab_file
from oncocontext.tools.cross_reference import cross_reference
from oncocontext.tools.crawl_supplementary import crawl_supplementary
from oncocontext.tools.crawl_and_report import crawl_and_report

__all__ = [
    "search_literature",
    "get_paper_details",
    "deep_search",
    "ingest_lab_file",
    "cross_reference",
    "crawl_supplementary",
    "crawl_and_report",
]
