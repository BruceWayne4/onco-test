"""Unit tests for SupplementaryCrawler and crawl_supplementary tool.

Tests cover:
    1. No-argument error guard
    2. PMID → PMC ID resolution (now via Europe PMC search primary + idconv fallback)
    3. Direct URL CSV parsing
    4. Direct URL XLSX parsing (real minimal bytes via openpyxl)
    5. Direct URL plain-text parsing
    6. Unsupported binary format returns error
    7. Large CSV truncation (600 rows → 500 rows, truncated=True)
    8. Europe PMC API discovery
    9. File-type filter (csv only, pdf skipped)
    10. Single file error does not abort other files
    -- New tests for Fixes 1–9 --
    11. hasSuppl=N skips all discovery (Fix 1)
    12. _normalize_url converts ftp:// to https:// (Fix 4)
    13. PMC OA API discovery returns tgz URL after normalization (Fix 3 + 4)
    14. Tarball extraction parses CSV member (Fix 9)
    15. eFetch URL contains tool=oncocontext (Fix 5)

All tests mock httpx.AsyncClient — no real network calls are made.
"""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_response(
    status_code: int = 200,
    content: bytes = b"",
    content_type: str = "text/plain",
    json_data: dict | list | None = None,
) -> MagicMock:
    """Build a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.headers = {"content-type": content_type}
    if json_data is not None:
        resp.json = MagicMock(return_value=json_data)
    else:
        resp.json = MagicMock(side_effect=ValueError("no json"))

    from httpx import HTTPStatusError, Request, Response as HttpxResponse
    if status_code >= 400:
        mock_request = MagicMock(spec=Request)
        mock_httpx_resp = MagicMock(spec=HttpxResponse)
        mock_httpx_resp.status_code = status_code
        resp.raise_for_status = MagicMock(
            side_effect=HTTPStatusError(
                f"HTTP {status_code}", request=mock_request, response=mock_httpx_resp
            )
        )
    else:
        resp.raise_for_status = MagicMock(return_value=None)
    return resp


def _make_minimal_xlsx() -> bytes:
    """Create a minimal in-memory XLSX file with openpyxl."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Sheet1"
    ws.append(["marker", "clone", "fluorochrome"])
    ws.append(["PD-1", "EH12.1", "PE"])
    ws.append(["TIM-3", "F38-2E2", "APC"])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _make_large_csv(n_rows: int = 600) -> bytes:
    """Return CSV bytes with n_rows data rows."""
    lines = ["col_a,col_b,col_c"]
    for i in range(n_rows):
        lines.append(f"val_{i},num_{i},{i}")
    return "\n".join(lines).encode()


def _make_europe_pmc_search_response(
    pmcid: str | None = "PMC8650059",
    has_suppl: str = "Y",
) -> dict:
    """Build a mock Europe PMC search API response."""
    result: dict = {"pmid": "34789550", "hasSuppl": has_suppl}
    if pmcid:
        result["pmcid"] = pmcid
    return {
        "resultList": {
            "result": [result]
        }
    }


def _make_pmc_oa_xml(ftp_url: str) -> bytes:
    """Build a minimal PMC OA API XML response with one tgz link."""
    return (
        f'<?xml version="1.0"?>'
        f'<OA><records><record>'
        f'<link format="tgz" href="{ftp_url}"/>'
        f'</record></records></OA>'
    ).encode()


def _make_minimal_tarball_with_csv() -> bytes:
    """Create an in-memory .tar.gz containing a single CSV file."""
    csv_content = b"gene,value\nTP53,1.5\nEGFR,2.3\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="table_s1.csv")
        info.size = len(csv_content)
        tar.addfile(info, io.BytesIO(csv_content))
    return buf.getvalue()


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def crawler():
    """Return a SupplementaryCrawler with its HTTP client patched."""
    from oncocontext.services.supplementary_crawler import SupplementaryCrawler

    c = SupplementaryCrawler()
    c._client = AsyncMock()
    return c


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 — no arguments → error dict
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_crawl_requires_at_least_one_param():
    """Calling crawl_supplementary() with no args returns a status=error dict."""
    from oncocontext.tools.crawl_supplementary import crawl_supplementary

    result = await crawl_supplementary()

    assert result["status"] == "error"
    assert "pmid" in result["message"].lower() or "provide" in result["message"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# Test 2 — PMID resolution (Europe PMC primary + idconv fallback)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pmid_resolves_via_europe_pmc_primary(crawler):
    """PMID→PMC resolution calls Europe PMC search first (Fix 1)."""
    europe_search_resp = _make_response(
        200,
        json_data=_make_europe_pmc_search_response(pmcid="PMC8650059", has_suppl="Y"),
    )
    # OA API returns no tgz link
    oa_api_resp = _make_response(
        200,
        content=b'<?xml version="1.0"?><OA><records><record></record></records></OA>',
        content_type="text/xml",
    )
    efetch_resp = _make_response(200, content=b"<pmc-articleset></pmc-articleset>", content_type="text/xml")
    europe_supp_resp = _make_response(200, json_data=[])

    async def side_effect(url, **kwargs):
        if "europepmc" in url and "search" in url:
            return europe_search_resp
        elif "oa.fcgi" in url:
            return oa_api_resp
        elif "efetch" in url:
            return efetch_resp
        elif "supplementaryFiles" in url:
            return europe_supp_resp
        return _make_response(404)

    crawler._client.get = AsyncMock(side_effect=side_effect)

    result = await crawler.crawl(pmid="34789550")

    assert result["pmc_id"] == "PMC8650059"
    called_urls = [call.args[0] for call in crawler._client.get.call_args_list]
    assert any("europepmc" in u and "search" in u for u in called_urls), \
        f"Europe PMC search not called; got: {called_urls}"


@pytest.mark.asyncio
async def test_pmid_resolves_via_idconv_fallback(crawler):
    """If Europe PMC search fails, idconv fallback is used (Fix 1)."""
    # Europe PMC search fails
    europe_search_resp = _make_response(500)
    idconv_resp = _make_response(
        200,
        json_data={"records": [{"pmid": "34789550", "pmcid": "PMC8650059"}]},
    )
    oa_api_resp = _make_response(
        200,
        content=b'<?xml version="1.0"?><OA><records><record></record></records></OA>',
        content_type="text/xml",
    )
    efetch_resp = _make_response(200, content=b"<pmc-articleset></pmc-articleset>", content_type="text/xml")
    europe_supp_resp = _make_response(200, json_data=[])

    async def side_effect(url, **kwargs):
        if "europepmc" in url and "search" in url:
            return europe_search_resp
        elif "idconv" in url:
            return idconv_resp
        elif "oa.fcgi" in url:
            return oa_api_resp
        elif "efetch" in url:
            return efetch_resp
        elif "supplementaryFiles" in url:
            return europe_supp_resp
        return _make_response(404)

    crawler._client.get = AsyncMock(side_effect=side_effect)

    result = await crawler.crawl(pmid="34789550")

    # idconv fallback should have resolved the PMC ID
    assert result["pmc_id"] == "PMC8650059"
    called_urls = [call.args[0] for call in crawler._client.get.call_args_list]
    assert any("idconv" in u for u in called_urls), f"idconv not called as fallback; got: {called_urls}"


# ══════════════════════════════════════════════════════════════════════════════
# Test 3 — direct_url CSV parsing
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_direct_url_csv_parsing(crawler):
    """Mock httpx returns CSV bytes; tool returns correct columns and rows."""
    csv_bytes = b"marker,clone,fluorochrome\nPD-1,EH12.1,PE\nTIM-3,F38-2E2,APC"
    csv_resp = _make_response(200, content=csv_bytes, content_type="text/csv")
    crawler._client.get = AsyncMock(return_value=csv_resp)

    result = await crawler.crawl(direct_url="https://example.com/supp/table_s1.csv")

    assert len(result["supplementary_files"]) == 1
    f = result["supplementary_files"][0]
    assert f["file_type"] == "csv"
    content = f["content"]
    assert content["columns"] == ["marker", "clone", "fluorochrome"]
    assert len(content["rows"]) == 2
    assert content["rows"][0]["marker"] == "PD-1"
    assert content["rows"][1]["clone"] == "F38-2E2"
    assert content["row_count"] == 2
    assert content["truncated"] is False


# ══════════════════════════════════════════════════════════════════════════════
# Test 4 — direct_url XLSX parsing
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_direct_url_xlsx_parsing(crawler):
    """Mock httpx returns minimal XLSX bytes; tool returns sheet dict with rows."""
    xlsx_bytes = _make_minimal_xlsx()
    xlsx_resp = _make_response(
        200,
        content=xlsx_bytes,
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    crawler._client.get = AsyncMock(return_value=xlsx_resp)

    result = await crawler.crawl(direct_url="https://example.com/supp/panel.xlsx")

    assert len(result["supplementary_files"]) == 1
    f = result["supplementary_files"][0]
    assert f["file_type"] == "xlsx"
    content = f["content"]
    assert "sheets" in content
    assert "Sheet1" in content["sheets"]
    sheet = content["sheets"]["Sheet1"]
    assert sheet["columns"] == ["marker", "clone", "fluorochrome"]
    assert sheet["row_count"] == 2
    assert sheet["rows"][0]["marker"] == "PD-1"


# ══════════════════════════════════════════════════════════════════════════════
# Test 5 — direct_url plain text parsing
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_direct_url_txt_parsing(crawler):
    """Mock httpx returns plain text; tool returns text with char_count."""
    text_content = "This is supplementary methods.\n\nAntibodies were purchased from BioLegend."
    txt_resp = _make_response(200, content=text_content.encode(), content_type="text/plain")
    crawler._client.get = AsyncMock(return_value=txt_resp)

    result = await crawler.crawl(direct_url="https://example.com/supp/methods.txt")

    assert len(result["supplementary_files"]) == 1
    f = result["supplementary_files"][0]
    assert f["file_type"] == "txt"
    content = f["content"]
    assert "text" in content
    assert content["text"] == text_content
    assert content["char_count"] == len(text_content)
    assert content["truncated"] is False


# ══════════════════════════════════════════════════════════════════════════════
# Test 6 — unsupported binary format returns error
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_unsupported_format_returns_error(crawler):
    """Mock returns binary content with unknown extension; content has error key."""
    bin_resp = _make_response(
        200,
        content=b"\x00\x01\x02\x03\xff\xfe",
        content_type="application/octet-stream",
    )
    crawler._client.get = AsyncMock(return_value=bin_resp)

    result = await crawler.crawl(direct_url="https://example.com/supp/data.bin")

    assert len(result["supplementary_files"]) == 1
    f = result["supplementary_files"][0]
    content = f["content"]
    assert "error" in content
    assert "unsupported" in content["error"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# Test 7 — large CSV truncation
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_truncation_for_large_csv(crawler):
    """CSV with 600 rows results in truncated=True and only 500 rows returned."""
    large_csv = _make_large_csv(600)
    csv_resp = _make_response(200, content=large_csv, content_type="text/csv")
    crawler._client.get = AsyncMock(return_value=csv_resp)

    result = await crawler.crawl(direct_url="https://example.com/supp/big_table.csv")

    f = result["supplementary_files"][0]
    content = f["content"]
    assert content["truncated"] is True
    assert content["row_count"] == 600
    assert len(content["rows"]) == 500


# ══════════════════════════════════════════════════════════════════════════════
# Test 8 — Europe PMC API discovery
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_europe_pmc_api_discovery(crawler):
    """Europe PMC API returns a list of files; each is fetched and returned."""
    europe_pmc_data = [
        {
            "url": "https://europepmc.org/articles/PMC9999/bin/table_s1.csv",
            "filename": "table_s1.csv",
            "title": "Supplementary Table 1",
        },
        {
            "url": "https://europepmc.org/articles/PMC9999/bin/table_s2.csv",
            "filename": "table_s2.csv",
            "title": "Supplementary Table 2",
        },
    ]

    csv_bytes = b"gene,value\nTP53,1.5\nEGFR,2.3"
    csv_resp = _make_response(200, content=csv_bytes, content_type="text/csv")

    # Europe PMC search: no PMC ID found but hasSuppl=Y
    europe_search_resp = _make_response(
        200,
        json_data=_make_europe_pmc_search_response(pmcid=None, has_suppl="Y"),
    )
    europe_supp_resp = _make_response(200, json_data=europe_pmc_data)

    async def side_effect(url, **kwargs):
        if "europepmc" in url and "search" in url:
            return europe_search_resp
        elif "idconv" in url:
            return _make_response(200, json_data={"records": [{"pmid": "99999999"}]})
        elif "supplementaryFiles" in url:
            return europe_supp_resp
        else:
            return csv_resp

    crawler._client.get = AsyncMock(side_effect=side_effect)

    result = await crawler.crawl(pmid="99999999")

    assert result["total_files_found"] == 2
    assert len(result["supplementary_files"]) == 2
    for f in result["supplementary_files"]:
        assert f["file_type"] == "csv"
        assert "columns" in f["content"]


# ══════════════════════════════════════════════════════════════════════════════
# Test 9 — file_types filter
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_file_type_filter(crawler):
    """When file_types=['csv'], PDF files in discovery are skipped."""
    result = await crawler.crawl(
        direct_url="https://example.com/supp/protocol.pdf",
        file_types=["csv"],
    )

    assert len(result["supplementary_files"]) == 0


@pytest.mark.asyncio
async def test_file_type_filter_includes_csv(crawler):
    """When file_types=['csv'], CSV files ARE included."""
    csv_bytes = b"col1,col2\nA,1\nB,2"
    csv_resp = _make_response(200, content=csv_bytes, content_type="text/csv")
    crawler._client.get = AsyncMock(return_value=csv_resp)

    result = await crawler.crawl(
        direct_url="https://example.com/supp/data.csv",
        file_types=["csv"],
    )

    assert len(result["supplementary_files"]) == 1
    assert result["supplementary_files"][0]["file_type"] == "csv"


# ══════════════════════════════════════════════════════════════════════════════
# Test 10 — single file error does not abort others
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_single_file_error_doesnt_abort_others(crawler):
    """If one file fetch raises an exception, other files are still returned."""
    two_files = [
        {
            "url": "https://example.com/supp/table_s1.csv",
            "filename": "table_s1.csv",
            "title": "Table S1",
        },
        {
            "url": "https://example.com/supp/table_s2.csv",
            "filename": "table_s2.csv",
            "title": "Table S2",
        },
    ]

    good_csv = b"a,b\n1,2\n3,4"
    good_resp = _make_response(200, content=good_csv, content_type="text/csv")
    bad_resp = _make_response(500)

    call_n = 0

    async def side_effect(url, **kwargs):
        nonlocal call_n
        call_n += 1
        if "table_s1" in url:
            return good_resp
        elif "table_s2" in url:
            return bad_resp
        return _make_response(200, content=b"<html></html>", content_type="text/html")

    crawler._client.get = AsyncMock(side_effect=side_effect)

    with patch.object(
        crawler, "_discover_via_pmc_oa_api", new=AsyncMock(return_value=two_files)
    ), patch.object(
        crawler, "_discover_via_efetch", new=AsyncMock(return_value=[])
    ), patch.object(
        crawler, "_discover_via_europe_pmc", new=AsyncMock(return_value=[])
    ):
        result = await crawler.crawl(pmc_id="PMC1234567")

    assert result["total_files_found"] == 2
    assert len(result["supplementary_files"]) == 2

    first = next(f for f in result["supplementary_files"] if "table_s1" in f["url"])
    assert "error" not in first["content"]
    assert first["content"]["columns"] == ["a", "b"]

    second = next(f for f in result["supplementary_files"] if "table_s2" in f["url"])
    assert "error" in second["content"]


# ══════════════════════════════════════════════════════════════════════════════
# Test 11 — hasSuppl=N skips all discovery (Fix 1)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_hasSuppl_N_skips_discovery(crawler):
    """When Europe PMC search returns hasSuppl='N', immediately return with no files (Fix 1)."""
    europe_search_resp = _make_response(
        200,
        json_data=_make_europe_pmc_search_response(pmcid="PMC8650059", has_suppl="N"),
    )
    crawler._client.get = AsyncMock(return_value=europe_search_resp)

    result = await crawler.crawl(pmid="34789550")

    # Should return immediately with no files
    assert result["supplementary_files"] == []
    assert result["total_files_found"] == 0
    assert result["files_parsed"] == 0

    # Should have a structured error entry about hasSuppl=N
    assert len(result["errors"]) >= 1
    hassupl_errors = [e for e in result["errors"] if e.get("stage") == "hasSuppl_check"]
    assert len(hassupl_errors) == 1
    assert hassupl_errors[0]["api"] == "europe_pmc_search"
    assert "hasSuppl=N" in hassupl_errors[0]["detail"]

    # Verify only 1 API call was made (no OA API, eFetch, or supplementaryFiles)
    called_urls = [call.args[0] for call in crawler._client.get.call_args_list]
    assert all("search" in u for u in called_urls if "europepmc" in u), \
        f"Should only call Europe PMC search, got: {called_urls}"
    # No OA API or eFetch calls
    assert not any("oa.fcgi" in u for u in called_urls)
    assert not any("efetch" in u for u in called_urls)


# ══════════════════════════════════════════════════════════════════════════════
# Test 12 — _normalize_url converts ftp:// to https:// (Fix 4)
# ══════════════════════════════════════════════════════════════════════════════


def test_ftp_url_converted_to_https():
    """_normalize_url() must convert ftp://ftp.ncbi.nlm.nih.gov/ → https://ftp.ncbi.nlm.nih.gov/ (Fix 4)."""
    from oncocontext.services.supplementary_crawler import SupplementaryCrawler

    crawler = SupplementaryCrawler.__new__(SupplementaryCrawler)

    ftp_url = "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/12/34/PMC8650059.tar.gz"
    https_url = crawler._normalize_url(ftp_url)

    assert https_url.startswith("https://ftp.ncbi.nlm.nih.gov/")
    assert not https_url.startswith("ftp://")
    assert https_url == "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/12/34/PMC8650059.tar.gz"


def test_normalize_url_leaves_https_unchanged():
    """_normalize_url() must not modify URLs already using https://."""
    from oncocontext.services.supplementary_crawler import SupplementaryCrawler

    crawler = SupplementaryCrawler.__new__(SupplementaryCrawler)

    https_url = "https://example.com/file.csv"
    assert crawler._normalize_url(https_url) == https_url


# ══════════════════════════════════════════════════════════════════════════════
# Test 13 — PMC OA API discovery returns tgz URL after normalization (Fix 3 + 4)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pmc_oa_api_discovery(crawler):
    """Mock OA API XML with ftp:// tgz URL; discovered entry has https:// URL (Fix 3 + 4)."""
    ftp_url = "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/12/34/PMC8650059.tar.gz"
    oa_xml = _make_pmc_oa_xml(ftp_url)

    oa_resp = _make_response(200, content=oa_xml, content_type="text/xml")
    crawler._client.get = AsyncMock(return_value=oa_resp)

    results = await crawler._discover_via_pmc_oa_api("PMC8650059")

    assert len(results) == 1
    entry = results[0]
    assert entry["url"].startswith("https://ftp.ncbi.nlm.nih.gov/")
    assert not entry["url"].startswith("ftp://")
    assert entry["filename"].endswith(".tar.gz") or entry["filename"].endswith(".tgz") \
        or "PMC8650059" in entry["filename"]


# ══════════════════════════════════════════════════════════════════════════════
# Test 14 — Tarball extraction parses CSV member (Fix 9)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_tarball_extraction(crawler):
    """In-memory tar.gz containing a CSV file is extracted and parsed correctly (Fix 9)."""
    tarball_bytes = _make_minimal_tarball_with_csv()

    results = await crawler._parse_tarball(tarball_bytes, source_url="https://example.com/PMC123.tar.gz")

    assert len(results) == 1
    entry = results[0]
    assert entry["filename"] == "table_s1.csv"
    assert entry["file_type"] == "csv"
    assert "error" not in entry["content"]
    assert entry["content"]["columns"] == ["gene", "value"]
    assert entry["content"]["row_count"] == 2
    assert entry["content"]["rows"][0]["gene"] == "TP53"


@pytest.mark.asyncio
async def test_tarball_too_large_returns_error(crawler):
    """Tarball exceeding 50 MB returns a single error entry (Fix 9)."""
    from oncocontext.services.supplementary_crawler import _TARBALL_MAX_BYTES

    # Create fake oversized data (just bytes, doesn't need to be valid tar)
    oversized = b"\x00" * (_TARBALL_MAX_BYTES + 1)

    results = await crawler._parse_tarball(oversized)

    assert len(results) == 1
    assert results[0]["error"] == "tarball too large"
    assert results[0]["size_bytes"] > _TARBALL_MAX_BYTES


# ══════════════════════════════════════════════════════════════════════════════
# Test 15 — eFetch URL contains tool=oncocontext (Fix 5)
# ══════════════════════════════════════════════════════════════════════════════


def test_ncbi_efetch_has_tool_params():
    """The eFetch URL template must contain tool=oncocontext (Fix 5)."""
    from oncocontext.services.supplementary_crawler import _PMC_EFETCH_URL, NCBI_TOOL_PARAMS

    assert "tool=oncocontext" in _PMC_EFETCH_URL
    assert "email=research@example.com" in _PMC_EFETCH_URL
    assert "tool=oncocontext" in NCBI_TOOL_PARAMS
    assert "email=research@example.com" in NCBI_TOOL_PARAMS


# ══════════════════════════════════════════════════════════════════════════════
# Test — errors schema is list[dict] (Fix 8)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_errors_are_list_of_dicts(crawler):
    """Every entry in result['errors'] must be a dict with at least 'stage' and 'detail' keys (Fix 8)."""
    # Simulate resolution failure to force an error entry
    europe_search_resp = _make_response(500)  # forces error
    idconv_resp = _make_response(200, json_data={"records": [{"pmid": "99"}]})  # no pmcid
    europe_supp_search = _make_response(
        200,
        json_data=_make_europe_pmc_search_response(pmcid=None, has_suppl="Y"),
    )
    europe_supp_resp = _make_response(200, json_data=[])

    call_count = [0]

    async def side_effect(url, **kwargs):
        call_count[0] += 1
        if "europepmc" in url and "search" in url:
            # First call for resolution fails; subsequent call (hasSuppl check in _discover_via_europe_pmc) succeeds
            if call_count[0] == 1:
                return europe_search_resp
            return europe_supp_search
        elif "idconv" in url:
            return idconv_resp
        elif "supplementaryFiles" in url:
            return europe_supp_resp
        return _make_response(404)

    crawler._client.get = AsyncMock(side_effect=side_effect)

    result = await crawler.crawl(pmid="99")

    # Validate errors schema
    for error in result["errors"]:
        assert isinstance(error, dict), f"Error entry must be a dict, got: {error!r}"
        assert "stage" in error, f"Error entry missing 'stage': {error!r}"
        assert "detail" in error, f"Error entry missing 'detail': {error!r}"


# ══════════════════════════════════════════════════════════════════════════════
# Additional unit tests for service internals
# ══════════════════════════════════════════════════════════════════════════════


class TestHelpers:
    """Test pure-helper functions in the supplementary_crawler module."""

    def test_filename_from_url(self):
        from oncocontext.services.supplementary_crawler import _filename_from_url

        assert _filename_from_url("https://example.com/bin/table_s1.xlsx") == "table_s1.xlsx"
        assert _filename_from_url("https://example.com/file.csv?v=2") == "file.csv"

    def test_ext_from_filename(self):
        from oncocontext.services.supplementary_crawler import _ext_from_filename

        assert _ext_from_filename("table_s1.CSV") == "csv"
        assert _ext_from_filename("data.xlsx") == "xlsx"
        assert _ext_from_filename("noext") == ""
        assert _ext_from_filename("") == ""

    def test_ext_from_filename_tarball(self):
        """Fix 9: .tar.gz and .tgz must return 'tgz'."""
        from oncocontext.services.supplementary_crawler import _ext_from_filename

        assert _ext_from_filename("PMC8650059.tar.gz") == "tgz"
        assert _ext_from_filename("archive.tgz") == "tgz"
        assert _ext_from_filename("PMC8650059.TAR.GZ") == "tgz"

    def test_ext_from_content_type(self):
        from oncocontext.services.supplementary_crawler import _ext_from_content_type

        assert _ext_from_content_type("text/csv") == "csv"
        assert _ext_from_content_type("application/pdf") == "pdf"
        assert _ext_from_content_type("application/json") == "json"
        assert _ext_from_content_type("application/octet-stream") == ""

    def test_looks_like_supp_file(self):
        from oncocontext.services.supplementary_crawler import _looks_like_supp_file

        assert _looks_like_supp_file("/pmc/articles/PMC12345/bin/table_s1.xlsx")
        assert _looks_like_supp_file("/download/supplementary_data.csv")
        assert _looks_like_supp_file("/data/s1.pdf")
        assert _looks_like_supp_file("/data/archive.tar.gz")
        assert not _looks_like_supp_file("/about")
        assert not _looks_like_supp_file("/login")

    def test_text_content_no_truncation(self):
        from oncocontext.services.supplementary_crawler import _text_content

        text = "Short text."
        result = _text_content(text)
        assert result["text"] == text
        assert result["char_count"] == len(text)
        assert result["truncated"] is False

    def test_text_content_truncation(self):
        from oncocontext.services.supplementary_crawler import _text_content, _TEXT_CHAR_LIMIT

        long_text = "x" * (_TEXT_CHAR_LIMIT + 100)
        result = _text_content(long_text)
        assert result["truncated"] is True
        assert len(result["text"]) == _TEXT_CHAR_LIMIT
        assert result["char_count"] == len(long_text)


class TestParseCSVDirect:
    """Test SupplementaryCrawler._parse_csv directly (no network)."""

    def test_parse_csv_basic(self):
        from oncocontext.services.supplementary_crawler import SupplementaryCrawler

        crawler = SupplementaryCrawler.__new__(SupplementaryCrawler)
        csv_bytes = b"a,b,c\n1,2,3\n4,5,6"
        result = crawler._parse_csv(csv_bytes, "csv")
        assert result["columns"] == ["a", "b", "c"]
        assert result["row_count"] == 2
        assert result["truncated"] is False

    def test_parse_tsv_basic(self):
        from oncocontext.services.supplementary_crawler import SupplementaryCrawler

        crawler = SupplementaryCrawler.__new__(SupplementaryCrawler)
        tsv_bytes = b"a\tb\tc\n1\t2\t3\n4\t5\t6"
        result = crawler._parse_csv(tsv_bytes, "tsv")
        assert result["columns"] == ["a", "b", "c"]
        assert result["row_count"] == 2

    def test_parse_csv_truncation(self):
        from oncocontext.services.supplementary_crawler import SupplementaryCrawler, _TABLE_ROW_LIMIT

        crawler = SupplementaryCrawler.__new__(SupplementaryCrawler)
        large = _make_large_csv(600)
        result = crawler._parse_csv(large, "csv")
        assert result["truncated"] is True
        assert result["row_count"] == 600
        assert len(result["rows"]) == _TABLE_ROW_LIMIT


class TestParseJSONDirect:
    """Test SupplementaryCrawler._parse_json directly."""

    def test_parse_json_list(self):
        from oncocontext.services.supplementary_crawler import SupplementaryCrawler

        crawler = SupplementaryCrawler.__new__(SupplementaryCrawler)
        payload = json.dumps([{"a": 1}, {"b": 2}]).encode()
        result = crawler._parse_json(payload)
        assert result["data"] == [{"a": 1}, {"b": 2}]
        assert result["truncated"] is False

    def test_parse_json_invalid(self):
        from oncocontext.services.supplementary_crawler import SupplementaryCrawler

        crawler = SupplementaryCrawler.__new__(SupplementaryCrawler)
        result = crawler._parse_json(b"{not valid json")
        assert "error" in result

    def test_parse_json_large_list_truncated(self):
        from oncocontext.services.supplementary_crawler import SupplementaryCrawler, _TABLE_ROW_LIMIT

        crawler = SupplementaryCrawler.__new__(SupplementaryCrawler)
        big_list = [{"i": i} for i in range(_TABLE_ROW_LIMIT + 100)]
        payload = json.dumps(big_list).encode()
        result = crawler._parse_json(payload)
        assert result["truncated"] is True
        assert len(result["data"]) == _TABLE_ROW_LIMIT


class TestParseXMLDirect:
    """Test SupplementaryCrawler._parse_xml directly."""

    def test_parse_xml_generic(self):
        from oncocontext.services.supplementary_crawler import SupplementaryCrawler

        crawler = SupplementaryCrawler.__new__(SupplementaryCrawler)
        xml_bytes = b"<root><child key='val'>text</child></root>"
        result = crawler._parse_xml(xml_bytes)
        assert "tree" in result or "sections" in result

    def test_parse_xml_invalid(self):
        from oncocontext.services.supplementary_crawler import SupplementaryCrawler

        crawler = SupplementaryCrawler.__new__(SupplementaryCrawler)
        result = crawler._parse_xml(b"<not valid xml")
        assert "error" in result
