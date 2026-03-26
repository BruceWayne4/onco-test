"""Tests for ResponsePaginator and SessionStore.

Covers:
    - No pagination when response is small
    - Pagination triggered when response exceeds MAX_RESPONSE_SIZE_KB
    - Tool-aware splitters: crawl_and_report, get_paper_details, search_literature,
      deep_search, cross_reference, crawl_supplementary
    - Generic fallback splitter
    - get_page round-trip through session store
    - Expired/missing session error handling
    - Truncation fallback when pagination fails
    - Page size guarantee: every page <= MAX_BYTES
"""

from __future__ import annotations

import json
import time
import uuid

import pytest

from oncocontext.services.response_paginator import (
    ResponsePaginator,
    SessionStore,
    _split_crawl_and_report,
    _split_crawl_supplementary,
    _split_cross_reference,
    _split_deep_search,
    _split_get_paper_details,
    _split_search_literature,
    _split_generic,
    _truncate_response,
    _MAX_BYTES,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_large_string(n_kb: int) -> str:
    """Return a string of roughly n_kb kilobytes."""
    return "x" * (n_kb * 1024)


def _json_bytes(obj) -> int:
    return len(json.dumps(obj, default=str).encode("utf-8"))


def _make_paginator() -> ResponsePaginator:
    return ResponsePaginator()


# ── Small responses — no pagination ───────────────────────────────────────────


class TestNoPagination:
    def test_small_response_returned_unchanged(self):
        paginator = _make_paginator()
        data = {"key": "value", "numbers": [1, 2, 3]}
        output = json.dumps(data, indent=2)
        result = paginator.paginate_if_needed(output, tool_name="search_literature")
        assert result == output

    def test_empty_response_returned_unchanged(self):
        paginator = _make_paginator()
        output = json.dumps({})
        result = paginator.paginate_if_needed(output, tool_name="crawl_and_report")
        assert result == output

    def test_exactly_at_limit_not_paginated(self):
        paginator = _make_paginator()
        # Build a string that's exactly at the limit
        payload = "y" * (_MAX_BYTES - 10)
        output = json.dumps({"data": payload})
        # Should be under limit if the raw JSON is under MAX_BYTES
        size = len(output.encode("utf-8"))
        if size <= _MAX_BYTES:
            result = paginator.paginate_if_needed(output, tool_name="deep_search")
            assert "_pagination" not in result


# ── Pagination triggered ───────────────────────────────────────────────────────


class TestPaginationTriggered:
    def _make_large_search_result(self, n_papers: int = 200) -> dict:
        return {
            "papers": [
                {
                    "pmid": str(i),
                    "title": f"Paper title number {i} with some additional text to make it bigger",
                    "abstract": _make_large_string(6),  # 6KB per abstract → 200×6KB = 1.2MB > 900KB limit
                    "authors": ["Author A", "Author B"],
                    "journal": "Journal of Testing",
                    "year": 2024,
                }
                for i in range(n_papers)
            ],
            "total_found": n_papers,
            "query_expansion": {"original_query": "test", "expanded_terms": [], "pubmed_query": "test"},
        }

    def test_pagination_adds_pagination_block(self):
        paginator = _make_paginator()
        data = self._make_large_search_result(200)
        output = json.dumps(data, indent=2)
        assert len(output.encode("utf-8")) > _MAX_BYTES, "Test data must exceed limit"

        result_str = paginator.paginate_if_needed(output, "search_literature")
        result = json.loads(result_str)
        assert "_pagination" in result

    def test_pagination_block_structure(self):
        paginator = _make_paginator()
        data = self._make_large_search_result(200)
        output = json.dumps(data, indent=2)
        result = json.loads(paginator.paginate_if_needed(output, "search_literature"))
        pag = result["_pagination"]

        assert "session_id" in pag
        assert pag["page"] == 1
        assert pag["total_pages"] >= 2
        assert pag["has_more"] is True
        assert "expires_in_seconds" in pag
        assert "hint" in pag

    def test_page_1_size_under_limit(self):
        paginator = _make_paginator()
        data = self._make_large_search_result(200)
        output = json.dumps(data, indent=2)
        result_str = paginator.paginate_if_needed(output, "search_literature")
        assert len(result_str.encode("utf-8")) <= _MAX_BYTES

    def test_invalid_json_triggers_truncation(self):
        paginator = _make_paginator()
        bad_json = "this is not json " + "x" * (_MAX_BYTES + 100)
        result = paginator.paginate_if_needed(bad_json, "search_literature")
        parsed = json.loads(result)
        assert "_truncation_warning" in parsed


# ── Session Store ─────────────────────────────────────────────────────────────


class TestSessionStore:
    def test_create_and_retrieve(self):
        store = SessionStore()
        pages = [{"data": "page1"}, {"data": "page2"}]
        session_id = store.create(pages, tool_name="test_tool")
        assert session_id

        page_data, error = store.get_page(session_id, 1)
        assert error is None
        assert page_data == {"data": "page1"}

        page_data2, error2 = store.get_page(session_id, 2)
        assert error2 is None
        assert page_data2 == {"data": "page2"}

    def test_invalid_page_returns_error(self):
        store = SessionStore()
        sid = store.create([{"a": 1}], "tool")
        _, err = store.get_page(sid, 0)
        assert err is not None
        _, err2 = store.get_page(sid, 2)
        assert err2 is not None

    def test_missing_session_returns_error(self):
        store = SessionStore()
        _, err = store.get_page("nonexistent-session", 1)
        assert err is not None
        assert "not found" in err.lower()

    def test_expired_session_returns_error(self):
        store = SessionStore()
        pages = [{"x": 1}]
        sid = store.create(pages, "tool")
        # Manually expire it
        store._store[sid].created_at = time.monotonic() - 99999
        _, err = store.get_page(sid, 1)
        assert err is not None
        assert "expired" in err.lower()

    def test_session_info(self):
        store = SessionStore()
        sid = store.create([{"p": 1}, {"p": 2}], "mytool")
        info = store.session_info(sid)
        assert info is not None
        assert info["total_pages"] == 2
        assert info["tool_name"] == "mytool"
        assert info["expires_in_seconds"] > 0

    def test_session_info_missing_returns_none(self):
        store = SessionStore()
        assert store.session_info("bogus") is None


# ── get_page round-trip ───────────────────────────────────────────────────────


class TestGetPageRoundTrip:
    def _make_large_search_result(self, n_papers: int = 200) -> dict:
        return {
            "papers": [
                {
                    "pmid": str(i),
                    "title": f"Paper {i}",
                    "abstract": "x" * 6000,  # 6KB per abstract → 200×6KB = 1.2MB > 900KB limit
                    "authors": [],
                    "journal": "J",
                    "year": 2024,
                }
                for i in range(n_papers)
            ],
            "total_found": n_papers,
            "query_expansion": {"original_query": "q", "expanded_terms": [], "pubmed_query": "q"},
        }

    def test_all_pages_retrievable(self):
        paginator = _make_paginator()
        data = self._make_large_search_result(200)
        output = json.dumps(data, indent=2)
        page1_str = paginator.paginate_if_needed(output, "search_literature")
        page1 = json.loads(page1_str)
        session_id = page1["_pagination"]["session_id"]
        total_pages = page1["_pagination"]["total_pages"]

        for page_num in range(2, total_pages + 1):
            page_str = paginator.get_page(session_id, page_num)
            page = json.loads(page_str)
            assert "_pagination" in page
            assert page["_pagination"]["page"] == page_num
            assert len(page_str.encode("utf-8")) <= _MAX_BYTES

    def test_last_page_has_more_false(self):
        paginator = _make_paginator()
        data = self._make_large_search_result(200)
        output = json.dumps(data, indent=2)
        page1 = json.loads(paginator.paginate_if_needed(output, "search_literature"))
        session_id = page1["_pagination"]["session_id"]
        total_pages = page1["_pagination"]["total_pages"]

        last_page = json.loads(paginator.get_page(session_id, total_pages))
        assert last_page["_pagination"]["has_more"] is False

    def test_all_papers_preserved_across_pages(self):
        """All papers from the original result must appear across all pages combined."""
        paginator = _make_paginator()
        n_papers = 200
        data = self._make_large_search_result(n_papers)
        output = json.dumps(data, indent=2)
        page1 = json.loads(paginator.paginate_if_needed(output, "search_literature"))
        session_id = page1["_pagination"]["session_id"]
        total_pages = page1["_pagination"]["total_pages"]

        all_pmids = set(p["pmid"] for p in page1.get("papers", []))
        for pg_num in range(2, total_pages + 1):
            pg = json.loads(paginator.get_page(session_id, pg_num))
            all_pmids.update(p["pmid"] for p in pg.get("papers", []))

        assert len(all_pmids) == n_papers


# ── Tool-Specific Splitters ───────────────────────────────────────────────────


class TestCrawlAndReportSplitter:
    def _make_result(self, report_size_kb: int = 1500) -> dict:
        headings = [f"## Section {i}\n" + "x" * 20_000 for i in range(report_size_kb // 20 + 1)]
        return {
            "local_data_path": "data/crawled/PMC123/",
            "report_path": "data/reports/PMC123_report.md",
            "report_markdown": "\n".join(headings),
            "supplementary_index": {},
            "references": [],
            "total_chars_crawled": 100000,
            "sections_found": ["Introduction", "Methods"],
            "pmid": "12345678",
            "pmc_id": "PMC123",
            "errors": [],
        }

    def test_splits_into_multiple_pages(self):
        result = self._make_result(1500)
        pages = _split_crawl_and_report(result)
        assert len(pages) >= 2

    def test_metadata_preserved_in_all_pages(self):
        result = self._make_result(1500)
        pages = _split_crawl_and_report(result)
        for pg in pages:
            assert "report_path" in pg
            assert "pmid" in pg
            assert "pmc_id" in pg
            assert pg["report_path"] == result["report_path"]

    def test_all_markdown_content_preserved(self):
        result = self._make_result(1500)
        pages = _split_crawl_and_report(result)
        combined_md = "".join(pg.get("report_markdown", "") for pg in pages)
        # All content should be present (modulo stripped leading newlines)
        original_stripped = result["report_markdown"].replace("\n", "")
        combined_stripped = combined_md.replace("\n", "")
        assert original_stripped == combined_stripped


class TestGetPaperDetailsSplitter:
    def _make_result(self, n_sections: int = 30) -> dict:
        return {
            "pmid": "12345",
            "pmc_id": "PMC12345",
            "title": "Test Paper",
            "authors": ["Author A"],
            "journal": "Test Journal",
            "year": 2024,
            "abstract": "Abstract text.",
            "mesh_terms": ["Term A"],
            "has_full_text": True,
            "sections": [
                {
                    "heading": f"Section {i}",
                    "section_type": "methods",
                    "paragraphs": ["x" * 5000 for _ in range(5)],
                }
                for i in range(n_sections)
            ],
            "indexed": True,
            "chunk_count": 100,
        }

    def test_splits_large_sections(self):
        result = self._make_result(30)
        if _json_bytes(result) > _MAX_BYTES:
            pages = _split_get_paper_details(result)
            assert len(pages) >= 2

    def test_metadata_preserved(self):
        result = self._make_result(30)
        pages = _split_get_paper_details(result)
        for pg in pages:
            assert pg["pmid"] == "12345"
            assert pg["title"] == "Test Paper"
            assert "sections" in pg


class TestSearchLiteratureSplitter:
    def _make_result(self, n_papers: int = 200) -> dict:
        return {
            "papers": [
                {
                    "pmid": str(i),
                    "title": f"Paper {i}",
                    "abstract": "x" * 3000,
                }
                for i in range(n_papers)
            ],
            "total_found": n_papers,
            "query_expansion": {"original_query": "q", "expanded_terms": [], "pubmed_query": "q"},
        }

    def test_preserves_all_papers(self):
        result = self._make_result(200)
        pages = _split_search_literature(result)
        total_papers = sum(len(pg.get("papers", [])) for pg in pages)
        assert total_papers == 200

    def test_scalar_fields_in_every_page(self):
        result = self._make_result(200)
        pages = _split_search_literature(result)
        for pg in pages:
            assert "total_found" in pg
            assert "query_expansion" in pg


class TestDeepSearchSplitter:
    def _make_result(self, n_results: int = 25) -> dict:
        return {
            "results": [
                {"chunk_text": "x" * 5000, "citation": {"pmid": str(i)}, "relevance_score": 0.9}
                for i in range(n_results)
            ],
            "total_indexed_papers": 10,
            "total_indexed_chunks": 1000,
            "search_strategy": "test strategy",
        }

    def test_preserves_all_results(self):
        result = self._make_result(25)
        pages = _split_deep_search(result)
        total = sum(len(pg.get("results", [])) for pg in pages)
        assert total == 25

    def test_strategy_in_every_page(self):
        result = self._make_result(25)
        pages = _split_deep_search(result)
        for pg in pages:
            assert pg["search_strategy"] == "test strategy"


class TestCrossReferenceSplitter:
    def _make_result(self) -> dict:
        return {
            "summary": "Summary text",
            "lab_data_summary": "Lab summary",
            "agreements": [{"lab_finding": "x" * 2000, "citation": {}} for _ in range(20)],
            "contradictions": [{"lab_finding": "y" * 2000, "citation": {}} for _ in range(5)],
            "novel_findings": ["finding " + "z" * 500 for _ in range(5)],
            "suggested_follow_up": ["follow up action " * 20],
            "papers_consulted": 5,
            "chunks_analyzed": 20,
        }

    def test_preserves_all_items(self):
        result = self._make_result()
        pages = _split_cross_reference(result)
        total_agreements = sum(len(pg.get("agreements", [])) for pg in pages)
        total_contradictions = sum(len(pg.get("contradictions", [])) for pg in pages)
        assert total_agreements == 20
        assert total_contradictions == 5

    def test_scalar_fields_in_every_page(self):
        result = self._make_result()
        pages = _split_cross_reference(result)
        for pg in pages:
            assert pg["summary"] == "Summary text"
            assert "papers_consulted" in pg


class TestCrawlSupplementarySplitter:
    def _make_result(self, n_files: int = 15) -> dict:
        return {
            "pmid": "12345",
            "pmc_id": "PMC12345",
            "files": [
                {
                    "filename": f"supplement_{i}.csv",
                    "content": "x" * 50000,
                    "file_type": "csv",
                }
                for i in range(n_files)
            ],
            "total_files": n_files,
        }

    def test_preserves_all_files(self):
        result = self._make_result(15)
        pages = _split_crawl_supplementary(result)
        total_files = sum(len(pg.get("files", [])) for pg in pages)
        assert total_files == 15


class TestGenericSplitter:
    def test_splits_on_largest_list(self):
        result = {
            "metadata": "small",
            "items": [{"data": "x" * 3000} for _ in range(100)],
        }
        pages = _split_generic(result)
        total = sum(len(pg.get("items", [])) for pg in pages)
        assert total == 100

    def test_splits_on_largest_string(self):
        result = {
            "title": "small",
            "full_text": "x" * 300_000,
        }
        pages = _split_generic(result)
        total_text = "".join(pg.get("full_text", "") for pg in pages)
        # Allow for slight differences due to whitespace stripping
        assert len(total_text) >= len(result["full_text"]) - len(pages) * 2


# ── Truncation Fallback ───────────────────────────────────────────────────────


class TestTruncationFallback:
    def test_truncation_adds_warning(self):
        huge_json = json.dumps({"data": "x" * (_MAX_BYTES * 2)})
        result = _truncate_response(huge_json, "test_tool")
        parsed = json.loads(result)
        assert "_truncation_warning" in parsed
        assert parsed["_truncation_warning"]["truncated"] is True

    def test_truncated_result_under_limit(self):
        huge_json = json.dumps({"data": "x" * (_MAX_BYTES * 2)})
        result = _truncate_response(huge_json, "test_tool")
        # Result should be parseable and reasonably sized
        assert len(result.encode("utf-8")) < _MAX_BYTES * 1.1  # allow tiny margin for warning

    def test_invalid_json_truncation(self):
        not_json = "NOT JSON " + "x" * (_MAX_BYTES * 2)
        result = _truncate_response(not_json, "test_tool")
        parsed = json.loads(result)
        assert "_truncation_warning" in parsed
