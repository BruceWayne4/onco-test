"""SupplementaryCrawler — fetch and parse supplementary data files from PMC/PubMed papers.

Discovers supplementary files attached to scientific papers and parses common
formats: CSV, TSV, XLSX, XLS, XML, PDF, DOCX, JSON, plain text, and tar.gz tarballs.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import tarfile
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_USER_AGENT = "OncoContext/1.0 (research tool; mailto:research@example.com)"
_TIMEOUT = 30.0

# Fix 5: NCBI tool params required by NCBI E-utilities policy
NCBI_TOOL_PARAMS = "&tool=oncocontext&email=research@example.com"

_IDCONV_URL = (
    "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
    + NCBI_TOOL_PARAMS
)
# Fix 1: Europe PMC search as primary resolver
_EUROPE_PMC_SEARCH_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    "?query=EXT_ID:{pmid}%20AND%20SRC:MED&format=json&resultType=core"
)
# Fix 3: PMC OA API for package/tarball discovery
_PMC_OA_API_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmc_id}"
# Fix 5: eFetch URL with tool params
_PMC_EFETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=pmc&id={pmc_id}&rettype=xml"
    + NCBI_TOOL_PARAMS
)
_EUROPE_PMC_SUPP_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmid}/supplementaryFiles?format=json"
)

_TABLE_ROW_LIMIT = 500
_TEXT_CHAR_LIMIT = 8000
_TARBALL_MAX_BYTES = 50 * 1024 * 1024  # 50 MB

# Fix 6: retry config
_RETRY_ATTEMPTS = 3
_RETRY_EXCEPTIONS = (httpx.RemoteProtocolError, httpx.ConnectError)


class SupplementaryCrawler:
    """Fetches and parses supplementary data files attached to PMC/PubMed papers.

    Supports CSV, TSV, XLSX, XLS, XML, PDF, DOCX, JSON, plain text, and tar.gz.
    Uses httpx.AsyncClient for all network I/O.
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(_TIMEOUT, connect=10.0),
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    async def crawl(
        self,
        pmid: str | None = None,
        pmc_id: str | None = None,
        direct_url: str | None = None,
        file_types: list[str] | None = None,
        max_files: int = 10,
    ) -> dict:
        """Crawl and parse supplementary files for a paper.

        Args:
            pmid: PubMed ID (e.g., "34789550").
            pmc_id: PMC ID (e.g., "PMC8650059").
            direct_url: Direct URL to a specific supplementary file.
            file_types: List of file extension strings to include (e.g. ["csv", "xlsx"]).
                        None means all types are included.
            max_files: Maximum number of files to fetch and parse.

        Returns:
            Dict with discovered and parsed supplementary files.
        """
        # Fix 8: errors is now list[dict]
        errors: list[dict] = []
        supp_files: list[dict] = []

        # ── Case 1: direct URL ────────────────────────────────────────────────
        if direct_url:
            filename = _filename_from_url(direct_url)
            file_type = _ext_from_filename(filename)
            if file_types is None or file_type in file_types:
                entry = await self._fetch_and_parse(direct_url, filename, title=None)
                supp_files.append(entry)
            return {
                "pmid": pmid,
                "pmc_id": pmc_id,
                "supplementary_files": supp_files,
                "total_files_found": len(supp_files),
                "files_parsed": sum(1 for f in supp_files if "error" not in f.get("content", {})),
                "errors": errors,
            }

        # ── Case 2/3: resolve PMID → PMC ID if needed ────────────────────────
        # Fix 2: track pmid throughout (it may be provided or resolved separately)
        resolved_pmc_id = pmc_id
        resolved_pmid = pmid  # Fix 2: always track pmid even when pmc_id given directly

        # Fix 1: Try Europe PMC search first as primary resolver (also gets hasSuppl)
        has_suppl: str | None = None
        if not resolved_pmc_id and resolved_pmid:
            try:
                pmc_from_europe, has_suppl = await self._resolve_via_europe_pmc_search(resolved_pmid)
                if pmc_from_europe:
                    resolved_pmc_id = pmc_from_europe
                    logger.info("Europe PMC search resolved PMID %s → %s (hasSuppl=%s)",
                                resolved_pmid, resolved_pmc_id, has_suppl)
            except Exception as exc:
                errors.append({
                    "stage": "pmid_resolution",
                    "api": "europe_pmc_search",
                    "detail": f"Europe PMC search failed for PMID {resolved_pmid}: {exc}",
                })
                logger.warning("Europe PMC search resolution failed for PMID %s: %s", resolved_pmid, exc)

            # Fix 1: If Europe PMC returned hasSuppl=N, bail out immediately
            if has_suppl == "N":
                errors.append({
                    "stage": "hasSuppl_check",
                    "api": "europe_pmc_search",
                    "detail": f"hasSuppl=N for PMID {resolved_pmid} — no supplementary files exist",
                })
                return {
                    "pmid": resolved_pmid,
                    "pmc_id": None,
                    "supplementary_files": [],
                    "total_files_found": 0,
                    "files_parsed": 0,
                    "errors": errors,
                }

            # Fix 1: Fall back to NCBI idconv if Europe PMC search failed/returned no PMC ID
            if not resolved_pmc_id:
                try:
                    resolved_pmc_id = await self._resolve_via_ncbi_idconv(resolved_pmid)
                    if resolved_pmc_id:
                        logger.info("NCBI idconv fallback resolved PMID %s → %s",
                                    resolved_pmid, resolved_pmc_id)
                except Exception as exc:
                    errors.append({
                        "stage": "pmid_resolution",
                        "api": "ncbi_idconv",
                        "detail": f"NCBI idconv fallback failed for PMID {resolved_pmid}: {exc}",
                    })
                    logger.warning("idconv fallback error for PMID %s: %s", resolved_pmid, exc)

        # ── Discover supplementary file URLs ──────────────────────────────────
        discovered: list[dict] = []  # list of {"url": str, "filename": str, "title": str|None}

        # Fix 3: Try PMC OA API first (replaces HTML scraping), then eFetch
        if resolved_pmc_id:
            try:
                from_oa_api = await self._discover_via_pmc_oa_api(resolved_pmc_id)
                discovered.extend(from_oa_api)
            except Exception as exc:
                errors.append({
                    "stage": "pmc_oa_api",
                    "api": "pmc_oa_api",
                    "detail": f"PMC OA API discovery failed for {resolved_pmc_id}: {exc}",
                })
                logger.warning("PMC OA API discovery failed: %s", exc)

            if not discovered:
                try:
                    from_efetch = await self._discover_via_efetch(resolved_pmc_id)
                    discovered.extend(from_efetch)
                except Exception as exc:
                    errors.append({
                        "stage": "efetch_discovery",
                        "api": "ncbi_efetch",
                        "detail": f"PMC eFetch discovery failed for {resolved_pmc_id}: {exc}",
                    })
                    logger.warning("PMC eFetch discovery failed: %s", exc)

        # Fix 2: Try Europe PMC API as last resort if pmid is available and nothing found yet
        if resolved_pmid and not discovered:
            try:
                from_europe = await self._discover_via_europe_pmc(resolved_pmid)
                discovered.extend(from_europe)
            except Exception as exc:
                errors.append({
                    "stage": "europe_pmc_discovery",
                    "api": "europe_pmc_supplementary",
                    "detail": f"Europe PMC discovery failed for PMID {resolved_pmid}: {exc}",
                })
                logger.warning("Europe PMC discovery failed: %s", exc)

        # ── Filter by file_types ──────────────────────────────────────────────
        if file_types is not None:
            discovered = [
                d for d in discovered
                if _ext_from_filename(d.get("filename", "")) in file_types
                or _ext_from_filename(d.get("filename", "")) in ("tgz",)  # always include tarballs
            ]

        # ── Fetch and parse up to max_files ───────────────────────────────────
        for item in discovered[:max_files]:
            try:
                entry = await self._fetch_and_parse(
                    item["url"], item.get("filename", ""), item.get("title")
                )
                supp_files.append(entry)
            except Exception as exc:
                errors.append({
                    "stage": "file_parse",
                    "filename": item.get("filename", ""),
                    "detail": f"Failed to fetch/parse {item.get('url', '')}: {exc}",
                })
                logger.warning("Failed to fetch/parse %s: %s", item.get("url", ""), exc)
                supp_files.append({
                    "filename": item.get("filename", ""),
                    "url": item.get("url", ""),
                    "file_type": _ext_from_filename(item.get("filename", "")),
                    "title": item.get("title"),
                    "content": {"error": str(exc)},
                })

        return {
            "pmid": resolved_pmid,
            "pmc_id": resolved_pmc_id,
            "supplementary_files": supp_files,
            "total_files_found": len(discovered),
            "files_parsed": sum(
                1 for f in supp_files if "error" not in f.get("content", {})
            ),
            "errors": errors,
        }

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ── URL Normalization ──────────────────────────────────────────────────────

    def _normalize_url(self, url: str) -> str:
        """Fix 4: Replace ftp://ftp.ncbi.nlm.nih.gov/ with https://ftp.ncbi.nlm.nih.gov/."""
        if url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
            return url.replace("ftp://ftp.ncbi.nlm.nih.gov/", "https://ftp.ncbi.nlm.nih.gov/", 1)
        return url

    # ── HTTP Fetch with Retry ──────────────────────────────────────────────────

    async def _fetch_bytes(self, url: str) -> httpx.Response:
        """Fix 6: Fetch URL bytes with exponential backoff retry on connection errors."""
        last_exc: Exception | None = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                response = await self._client.get(url)
                return response
            except _RETRY_EXCEPTIONS as exc:
                last_exc = exc
                if attempt == _RETRY_ATTEMPTS - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    "Fetch attempt %d/%d failed for %s (%s); retrying in %ds",
                    attempt + 1, _RETRY_ATTEMPTS, url, exc, wait
                )
                await asyncio.sleep(wait)
        # Should never reach here, but satisfy type checker
        raise last_exc  # type: ignore[misc]

    # ── Discovery Methods ──────────────────────────────────────────────────────

    async def _resolve_via_europe_pmc_search(self, pmid: str) -> tuple[str | None, str | None]:
        """Fix 1: Resolve PMID→PMC ID via Europe PMC search API; also returns hasSuppl."""
        url = _EUROPE_PMC_SEARCH_URL.format(pmid=pmid)
        resp = await self._fetch_bytes(url)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("resultList", {}).get("result", [])
        if not results:
            errors_detail = f"pmcid absent in response for PMID {pmid}"
            logger.warning("Europe PMC search: %s", errors_detail)
            return None, None
        first = results[0]
        pmc_id = first.get("pmcid") or None
        has_suppl = first.get("hasSuppl") or None
        return pmc_id, has_suppl

    async def _resolve_via_ncbi_idconv(self, pmid: str) -> str | None:
        """Fix 1 fallback: Resolve PMID→PMC ID via NCBI idconv API."""
        url = _IDCONV_URL.format(pmid=pmid)
        try:
            resp = await self._fetch_bytes(url)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("records", [])
            if records:
                pmc = records[0].get("pmcid")
                if pmc:
                    logger.info("NCBI idconv resolved PMID %s → %s", pmid, pmc)
                    return pmc
        except Exception as exc:
            logger.warning("idconv error for PMID %s: %s", pmid, exc)
        return None

    async def _discover_via_pmc_oa_api(self, pmc_id: str) -> list[dict]:
        """Fix 3: Query the PMC Open Access API for the article package (tarball) URL."""
        try:
            import lxml.etree as etree
        except ImportError:
            logger.warning("lxml not available; skipping PMC OA API discovery")
            return []

        url = _PMC_OA_API_URL.format(pmc_id=pmc_id)
        try:
            resp = await self._fetch_bytes(url)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("PMC OA API fetch failed (%d): %s", exc.response.status_code, url)
            return []

        try:
            root = etree.fromstring(resp.content)
        except etree.XMLSyntaxError as exc:
            logger.warning("PMC OA API XML parse error: %s", exc)
            return []

        results: list[dict] = []

        # The OA API returns <OA><records><record ...><link format="tgz" href="ftp://..."/></record></records></OA>
        for link_el in root.iter("link"):
            fmt = link_el.get("format", "")
            href = link_el.get("href", "")
            if not href:
                continue
            # Fix 4: normalize ftp:// → https://
            normalized = self._normalize_url(href)
            if fmt in ("tgz", "tar.gz") or normalized.endswith(".tar.gz") or normalized.endswith(".tgz"):
                filename = _filename_from_url(normalized)
                logger.info("PMC OA API: tarball URL for %s: %s → %s", pmc_id, href, normalized)
                if href != normalized:
                    logger.info("  (converted ftp:// → https://)")
                results.append({
                    "url": normalized,
                    "filename": filename,
                    "title": None,
                })
            elif fmt == "pdf" or normalized.endswith(".pdf"):
                filename = _filename_from_url(normalized)
                results.append({
                    "url": normalized,
                    "filename": filename,
                    "title": None,
                })

        logger.info("PMC OA API discovery found %d files for %s", len(results), pmc_id)
        return results

    async def _discover_via_efetch(self, pmc_id: str) -> list[dict]:
        """Parse PMC eFetch XML to find supplementary-material tags."""
        try:
            import lxml.etree as etree
        except ImportError:
            logger.warning("lxml not available; skipping eFetch XML discovery")
            return []

        # Strip PMC prefix for eFetch
        numeric_id = pmc_id.upper().replace("PMC", "")
        # Fix 5: NCBI_TOOL_PARAMS already embedded in _PMC_EFETCH_URL
        url = _PMC_EFETCH_URL.format(pmc_id=numeric_id)
        try:
            resp = await self._fetch_bytes(url)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("eFetch failed (%d): %s", exc.response.status_code, url)
            return []

        try:
            root = etree.fromstring(resp.content)
        except etree.XMLSyntaxError as exc:
            logger.warning("eFetch XML parse error: %s", exc)
            return []

        results: list[dict] = []
        seen_urls: set[str] = set()

        for sm in root.iter("supplementary-material"):
            href = sm.get("href") or sm.get(
                "{http://www.w3.org/1999/xlink}href", ""
            )
            if not href:
                continue

            # Build full URL if relative
            if href.startswith("http") or href.startswith("ftp"):
                full_url = self._normalize_url(href)  # Fix 4
            else:
                full_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/bin/{href}"

            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            caption_el = sm.find(".//caption/title")
            title = caption_el.text if caption_el is not None else None

            results.append({
                "url": full_url,
                "filename": _filename_from_url(full_url),
                "title": title,
            })

        logger.info("eFetch discovery found %d files for %s", len(results), pmc_id)
        return results

    async def _discover_via_europe_pmc(self, pmid: str) -> list[dict]:
        """Fix 7: Query the Europe PMC supplementary files API; check hasSuppl first."""
        # Fix 7: do a quick search-API call first to check hasSuppl
        try:
            _, has_suppl = await self._resolve_via_europe_pmc_search(pmid)
            if has_suppl is not None and has_suppl != "Y":
                logger.info(
                    "Europe PMC: hasSuppl=%s for PMID %s — skipping supplementaryFiles endpoint",
                    has_suppl, pmid
                )
                return []
        except Exception as exc:
            logger.warning(
                "Europe PMC hasSuppl pre-check failed for PMID %s: %s — proceeding anyway", pmid, exc
            )

        url = _EUROPE_PMC_SUPP_URL.format(pmid=pmid)
        try:
            resp = await self._fetch_bytes(url)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Europe PMC supplementary API failed for PMID %s: %s", pmid, exc)
            return []

        results: list[dict] = []
        supp_list = data if isinstance(data, list) else data.get("supplementaryFiles", [])
        for item in supp_list:
            file_url = item.get("url") or item.get("downloadUrl") or item.get("link")
            if not file_url:
                continue
            file_url = self._normalize_url(file_url)  # Fix 4
            filename = item.get("filename") or item.get("name") or _filename_from_url(file_url)
            title = item.get("title") or item.get("caption")
            results.append({"url": file_url, "filename": filename, "title": title})

        logger.info("Europe PMC discovery found %d files for PMID %s", len(results), pmid)
        return results

    # ── Fetch + Parse ──────────────────────────────────────────────────────────

    async def _fetch_and_parse(
        self, url: str, filename: str, title: str | None
    ) -> dict:
        """Fetch a URL and parse its content based on file extension / content-type."""
        # Fix 4: normalize URL before fetching
        url = self._normalize_url(url)
        try:
            resp = await self._fetch_bytes(url)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return {
                "filename": filename,
                "url": url,
                "file_type": _ext_from_filename(filename),
                "title": title,
                "content": {"error": f"HTTP {exc.response.status_code}"},
            }

        raw_bytes = resp.content
        content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()

        # Determine file type from filename extension, fall back to content-type
        ext = _ext_from_filename(filename)
        if not ext:
            ext = _ext_from_content_type(content_type)
        if not filename:
            filename = _filename_from_url(url)

        # Fix 9: handle tarballs specially (async dispatch)
        if ext in ("tgz",) or filename.endswith(".tar.gz"):
            content_list = await self._parse_tarball(raw_bytes, url)
            return {
                "filename": filename,
                "url": url,
                "file_type": "tgz",
                "title": title,
                "content": {"files": content_list},
            }

        content = self._parse_content(raw_bytes, ext, content_type, url)

        return {
            "filename": filename,
            "url": url,
            "file_type": ext or "unknown",
            "title": title,
            "content": content,
        }

    def _parse_content(
        self, raw_bytes: bytes, ext: str, content_type: str, url: str
    ) -> dict:
        """Dispatch to the appropriate parser based on file extension."""
        try:
            if ext in ("csv", "tsv"):
                return self._parse_csv(raw_bytes, ext)
            elif ext in ("xlsx", "xls"):
                return self._parse_excel(raw_bytes)
            elif ext == "xml":
                return self._parse_xml(raw_bytes)
            elif ext == "pdf":
                return self._parse_pdf(raw_bytes)
            elif ext == "docx":
                return self._parse_docx(raw_bytes)
            elif ext == "json":
                return self._parse_json(raw_bytes)
            elif ext in ("txt", "md", "text", ""):
                # Also handle text content-type
                if ext == "" and "text" in content_type:
                    return self._parse_text(raw_bytes)
                elif ext in ("txt", "md", "text"):
                    return self._parse_text(raw_bytes)
                else:
                    return {"error": "unsupported format", "content_type": content_type}
            else:
                # Last-chance: if it's a text content type, parse as text
                if "text" in content_type:
                    return self._parse_text(raw_bytes)
                return {"error": "unsupported format", "content_type": content_type}
        except Exception as exc:
            logger.warning("Parse error for %s (%s): %s", url, ext, exc)
            return {"error": f"parse error: {exc}"}

    # ── Format Parsers ─────────────────────────────────────────────────────────

    def _parse_csv(self, raw_bytes: bytes, ext: str) -> dict:
        """Parse CSV or TSV bytes using pandas."""
        import pandas as pd

        sep = "\t" if ext == "tsv" else ","
        text = raw_bytes.decode("utf-8", errors="replace")
        if ext == "csv" and "\t" in text.split("\n")[0] and "," not in text.split("\n")[0]:
            sep = "\t"

        df = pd.read_csv(io.StringIO(text), sep=sep)
        row_count = len(df)
        truncated = row_count > _TABLE_ROW_LIMIT
        if truncated:
            df = df.head(_TABLE_ROW_LIMIT)

        return {
            "columns": list(df.columns),
            "rows": df.fillna("").to_dict(orient="records"),
            "row_count": row_count,
            "truncated": truncated,
        }

    def _parse_excel(self, raw_bytes: bytes) -> dict:
        """Parse XLSX/XLS bytes using pandas (all sheets)."""
        import pandas as pd

        sheets_data = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=None)
        sheets_result: dict[str, Any] = {}

        for sheet_name, df in sheets_data.items():
            row_count = len(df)
            truncated = row_count > _TABLE_ROW_LIMIT
            if truncated:
                df = df.head(_TABLE_ROW_LIMIT)
            sheets_result[str(sheet_name)] = {
                "columns": list(df.columns),
                "rows": df.fillna("").to_dict(orient="records"),
                "row_count": row_count,
                "truncated": truncated,
            }

        return {"sheets": sheets_result}

    def _parse_xml(self, raw_bytes: bytes) -> dict:
        """Parse XML bytes with lxml; detect BioC format."""
        try:
            import lxml.etree as etree
        except ImportError:
            return self._parse_text(raw_bytes)

        try:
            root = etree.fromstring(raw_bytes)
        except etree.XMLSyntaxError as exc:
            return {"error": f"XML parse error: {exc}"}

        # Detect BioC format
        if root.tag == "collection" or root.find(".//document") is not None:
            sections = []
            for doc in root.iter("document"):
                doc_id = doc.findtext("id") or ""
                for passage in doc.iter("passage"):
                    section_type = passage.findtext("infon[@key='section_type']") or ""
                    text_el = passage.find("text")
                    text_val = text_el.text if text_el is not None else ""
                    if text_val:
                        sections.append({
                            "document_id": doc_id,
                            "section": section_type,
                            "text": text_val,
                        })
            return {"sections": sections}

        # Generic XML: return tag/text tree up to 3 levels deep
        tree = _xml_to_dict(root, max_depth=3)
        return {"tree": tree}

    def _parse_pdf(self, raw_bytes: bytes) -> dict:
        """Extract text from PDF using pdfminer.six or pypdf."""
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract
            text = pdfminer_extract(io.BytesIO(raw_bytes))
            return _text_content(text)
        except ImportError:
            pass

        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw_bytes))
            parts = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
            text = "\n".join(parts)
            return _text_content(text)
        except ImportError:
            pass

        return {
            "error": "PDF parsing library not available",
            "install": "pip install pdfminer.six",
        }

    def _parse_docx(self, raw_bytes: bytes) -> dict:
        """Extract text from DOCX using python-docx."""
        try:
            from docx import Document
        except ImportError:
            return {
                "error": "DOCX parsing library not available",
                "install": "pip install python-docx",
            }

        try:
            doc = Document(io.BytesIO(raw_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n".join(paragraphs)
            return _text_content(text)
        except Exception as exc:
            return {"error": f"DOCX parse error: {exc}"}

    def _parse_json(self, raw_bytes: bytes) -> dict:
        """Parse JSON bytes."""
        try:
            text = raw_bytes.decode("utf-8", errors="replace")
            data = json.loads(text)
            truncated = False
            if isinstance(data, list) and len(data) > _TABLE_ROW_LIMIT:
                data = data[:_TABLE_ROW_LIMIT]
                truncated = True
            return {"data": data, "truncated": truncated}
        except json.JSONDecodeError as exc:
            return {"error": f"JSON parse error: {exc}"}

    def _parse_text(self, raw_bytes: bytes) -> dict:
        """Parse plain text, splitting into paragraphs."""
        text = raw_bytes.decode("utf-8", errors="replace")
        return _text_content(text)

    async def _parse_tarball(self, data: bytes, source_url: str = "") -> list[dict]:
        """Fix 9: Parse a .tar.gz tarball, extracting and parsing supported data files."""
        size_bytes = len(data)
        if size_bytes > _TARBALL_MAX_BYTES:
            logger.warning("Tarball too large (%d bytes) from %s — skipping", size_bytes, source_url)
            return [{"error": "tarball too large", "size_bytes": size_bytes}]

        _TARBALL_EXTS = {".csv", ".tsv", ".xlsx", ".xls", ".xml", ".txt", ".pdf", ".docx", ".json"}

        results: list[dict] = []
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    name = member.name
                    # Check extension
                    lower = name.lower()
                    matched_ext = next(
                        (ext.lstrip(".") for ext in _TARBALL_EXTS if lower.endswith(ext)),
                        None
                    )
                    if matched_ext is None:
                        continue
                    try:
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        member_bytes = f.read()
                    except Exception as exc:
                        results.append({
                            "filename": name,
                            "file_type": matched_ext,
                            "content": {"error": f"extraction error: {exc}"},
                        })
                        continue

                    content = self._parse_content(member_bytes, matched_ext, "", name)
                    results.append({
                        "filename": name,
                        "file_type": matched_ext,
                        "content": content,
                    })
        except tarfile.TarError as exc:
            logger.warning("Tarball parse error from %s: %s", source_url, exc)
            results.append({"error": f"tarball parse error: {exc}"})

        return results


# ── Helpers ────────────────────────────────────────────────────────────────────


def _filename_from_url(url: str) -> str:
    """Extract filename from a URL path."""
    path = urlparse(url).path
    parts = path.rstrip("/").split("/")
    return parts[-1] if parts else ""


def _ext_from_filename(filename: str) -> str:
    """Get lowercase extension without leading dot from a filename.

    Fix 9: handles .tar.gz and .tgz specially.
    """
    if not filename:
        return ""
    lower = filename.lower()
    if lower.endswith(".tar.gz"):
        return "tgz"
    if lower.endswith(".tgz"):
        return "tgz"
    parts = filename.rsplit(".", 1)
    if len(parts) == 2:
        return parts[1].lower()
    return ""


def _ext_from_content_type(content_type: str) -> str:
    """Map a MIME content-type to a simple extension string."""
    mapping = {
        "text/csv": "csv",
        "text/tab-separated-values": "tsv",
        "application/vnd.ms-excel": "xls",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        "application/xml": "xml",
        "text/xml": "xml",
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/json": "json",
        "text/plain": "txt",
        "text/markdown": "md",
        "application/x-tar": "tgz",
        "application/gzip": "tgz",
    }
    return mapping.get(content_type, "")


def _looks_like_supp_file(href: str) -> bool:
    """Return True if the href looks like a supplementary file link."""
    href_lower = href.lower()
    data_exts = (
        ".csv", ".tsv", ".xlsx", ".xls", ".pdf", ".docx",
        ".xml", ".json", ".txt", ".zip", ".tar.gz", ".tgz",
    )
    has_data_ext = any(href_lower.endswith(ext) for ext in data_exts)
    pmc_supp_patterns = [
        "/pmc/articles/pmc",
        "/bin/",
        "suppl",
        "supplement",
        "supplementary",
        "s1.", "s2.", "s3.", "s4.", "s5.",
        "table_s", "table-s", "fig_s", "fig-s",
    ]
    has_supp_pattern = any(p in href_lower for p in pmc_supp_patterns)
    return has_data_ext or has_supp_pattern


def _text_content(text: str) -> dict:
    """Build a text content dict with truncation."""
    char_count = len(text)
    truncated = char_count > _TEXT_CHAR_LIMIT
    if truncated:
        text = text[:_TEXT_CHAR_LIMIT]
    return {
        "text": text,
        "char_count": char_count,
        "truncated": truncated,
    }


def _xml_to_dict(element: Any, max_depth: int = 3, current_depth: int = 0) -> dict:
    """Recursively convert an lxml Element to a dict up to max_depth levels."""
    result: dict[str, Any] = {"tag": element.tag}
    if element.text and element.text.strip():
        result["text"] = element.text.strip()
    if element.attrib:
        result["attrib"] = dict(element.attrib)
    if current_depth < max_depth:
        children = []
        for child in element:
            children.append(_xml_to_dict(child, max_depth, current_depth + 1))
        if children:
            result["children"] = children
    return result
