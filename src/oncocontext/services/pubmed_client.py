"""PubMed E-utilities API client — esearch, efetch, rate limiting.

Provides async access to PubMed search and fetch with rate limiting,
retry with exponential backoff, and XML response parsing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from lxml import etree

from oncocontext.config import settings
from oncocontext.storage.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class PubMedClient:
    """Async client for PubMed E-utilities API.

    Handles:
        - esearch: Search PubMed and retrieve PMIDs
        - efetch: Batch fetch metadata for PMIDs (up to 200 per request)
        - Rate limiting: 3 req/s without key, 10 req/s with NCBI API key
        - Retry with exponential backoff (1s, 2s, 4s), max 3 retries
        - XML response parsing to extract paper metadata
    """

    def __init__(self, api_key: str | None = None, cache: CacheManager | None = None) -> None:
        """Initialize PubMed client.

        Args:
            api_key: Optional NCBI API key for higher rate limits.
            cache: Optional CacheManager for response caching.
        """
        self.api_key = api_key or settings.NCBI_API_KEY
        self._base_url = settings.PUBMED_BASE_URL
        self._cache = cache
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
        )

        # Rate limiting
        rate_limit = (
            settings.PUBMED_RATE_LIMIT_WITH_KEY if self.api_key
            else settings.PUBMED_RATE_LIMIT
        )
        self._min_interval = 1.0 / rate_limit
        self._last_request_time: float = 0.0
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_FETCHES)

    # ── Public API ────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        max_results: int = 50,
        date_range: str | None = None,
        full_text_only: bool = False,
    ) -> dict:
        """Search PubMed via esearch and return PMIDs + total count.

        Args:
            query: PubMed query string (already expanded).
            max_results: Maximum PMIDs to retrieve.
            date_range: Optional date filter, e.g. '2020-2025'.
            full_text_only: Add pmc[filter] if True.

        Returns:
            Dict with 'pmids' (list[str]) and 'total_count' (int).
        """
        if not query or not query.strip():
            return {"pmids": [], "total_count": 0}

        # Build cache key
        cache_key = None
        if self._cache:
            cache_key = CacheManager.make_key(
                "esearch", query, str(max_results), str(date_range), str(full_text_only)
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for PubMed search: %s", query[:60])
                return cached

        # Build params
        effective_query = query
        if full_text_only:
            effective_query = f"({query}) AND pmc[filter]"

        params: dict[str, Any] = {
            "db": "pubmed",
            "term": effective_query,
            "retmax": min(max_results, settings.PUBMED_MAX_RESULTS),
            "retmode": "json",
            "sort": "relevance",
        }

        if date_range:
            parts = date_range.split("-")
            if len(parts) == 2:
                params["mindate"] = f"{parts[0]}/01/01"
                params["maxdate"] = f"{parts[1]}/12/31"
                params["datetype"] = "pdat"

        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self._base_url}/esearch.fcgi"
        data = await self._request_with_retry(url, params)

        if data is None:
            return {"pmids": [], "total_count": 0}

        try:
            esearch_result = data.get("esearchresult", {})
            pmids = esearch_result.get("idlist", [])
            total_count = int(esearch_result.get("count", 0))
            result = {"pmids": pmids, "total_count": total_count}

            if self._cache and cache_key:
                self._cache.set(cache_key, result, ttl=settings.CACHE_TTL_PUBMED)

            logger.info("PubMed search returned %d PMIDs (total: %d) for: %s",
                        len(pmids), total_count, query[:60])
            return result
        except (KeyError, ValueError, TypeError) as exc:
            logger.error("Failed to parse esearch response: %s", exc)
            return {"pmids": [], "total_count": 0}

    async def fetch_details(self, pmids: list[str]) -> list[dict]:
        """Batch fetch metadata for a list of PMIDs via efetch.

        Args:
            pmids: List of PubMed IDs (up to 200).

        Returns:
            List of dicts with paper metadata (title, authors, journal, etc.).
        """
        if not pmids:
            return []

        # Check cache for individual PMIDs
        results: list[dict] = []
        uncached_pmids: list[str] = []

        if self._cache:
            for pmid in pmids:
                cache_key = CacheManager.make_key("efetch", pmid)
                cached = self._cache.get(cache_key)
                if cached is not None:
                    results.append(cached)
                else:
                    uncached_pmids.append(pmid)
        else:
            uncached_pmids = list(pmids)

        if not uncached_pmids:
            logger.debug("All %d PMIDs found in cache", len(pmids))
            return results

        # Batch fetch uncached PMIDs (max 200 per request)
        batch_size = 200
        for i in range(0, len(uncached_pmids), batch_size):
            batch = uncached_pmids[i:i + batch_size]
            batch_results = await self._fetch_batch(batch)
            results.extend(batch_results)

            # Cache individual results
            if self._cache:
                for paper in batch_results:
                    cache_key = CacheManager.make_key("efetch", paper["pmid"])
                    self._cache.set(cache_key, paper, ttl=settings.CACHE_TTL_PMC)

        logger.info("Fetched details for %d PMIDs (%d from cache, %d from API)",
                     len(results), len(pmids) - len(uncached_pmids), len(uncached_pmids))
        return results

    async def fetch_mesh_terms(self, pmids: list[str]) -> dict[str, list[str]]:
        """Fetch MeSH terms for given PMIDs.

        This is included in fetch_details, but provided separately for convenience.

        Args:
            pmids: List of PubMed IDs.

        Returns:
            Dict mapping PMID → list of MeSH term strings.
        """
        details = await self.fetch_details(pmids)
        return {d["pmid"]: d.get("mesh_terms", []) for d in details}

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ── Internal Methods ──────────────────────────────────────────────────────

    async def _fetch_batch(self, pmids: list[str]) -> list[dict]:
        """Fetch a batch of PMIDs via efetch and parse the XML response."""
        params: dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self._base_url}/efetch.fcgi"
        xml_text = await self._request_with_retry(url, params, parse_json=False)

        if xml_text is None:
            return []

        return self._parse_efetch_xml(xml_text)

    def _parse_efetch_xml(self, xml_text: str) -> list[dict]:
        """Parse PubMed efetch XML into structured dicts."""
        results: list[dict] = []

        try:
            root = etree.fromstring(xml_text.encode("utf-8"))
        except etree.XMLSyntaxError as exc:
            logger.error("Failed to parse efetch XML: %s", exc)
            return []

        for article in root.findall(".//PubmedArticle"):
            try:
                paper = self._parse_article(article)
                if paper:
                    results.append(paper)
            except Exception as exc:
                logger.warning("Failed to parse article: %s", exc)
                continue

        return results

    def _parse_article(self, article: etree._Element) -> dict | None:
        """Parse a single PubmedArticle element."""
        citation = article.find("MedlineCitation")
        if citation is None:
            return None

        # PMID
        pmid_el = citation.find("PMID")
        pmid = pmid_el.text if pmid_el is not None and pmid_el.text else ""
        if not pmid:
            return None

        art = citation.find("Article")
        if art is None:
            return {"pmid": pmid}

        # Title
        title_el = art.find("ArticleTitle")
        title = self._get_text_content(title_el) if title_el is not None else ""

        # Journal
        journal = ""
        journal_el = art.find("Journal/Title")
        if journal_el is not None and journal_el.text:
            journal = journal_el.text

        # Year
        year = 0
        # Try multiple paths for year
        year_paths = [
            "Journal/JournalIssue/PubDate/Year",
            "Journal/JournalIssue/PubDate/MedlineDate",
        ]
        for ypath in year_paths:
            year_el = art.find(ypath)
            if year_el is not None and year_el.text:
                try:
                    year = int(year_el.text[:4])
                    break
                except (ValueError, IndexError):
                    continue

        # Abstract
        abstract_parts: list[str] = []
        abstract_el = art.find("Abstract")
        if abstract_el is not None:
            for text_el in abstract_el.findall("AbstractText"):
                text = self._get_text_content(text_el)
                if text:
                    label = text_el.get("Label")
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Authors
        authors: list[str] = []
        author_list = art.find("AuthorList")
        if author_list is not None:
            for author_el in author_list.findall("Author"):
                last = author_el.find("LastName")
                fore = author_el.find("ForeName")
                if last is not None and last.text:
                    name = last.text
                    if fore is not None and fore.text:
                        name += f" {fore.text}"
                    authors.append(name)
                else:
                    # Collective name
                    collective = author_el.find("CollectiveName")
                    if collective is not None and collective.text:
                        authors.append(collective.text)

        # MeSH terms
        mesh_terms: list[str] = []
        mesh_list = citation.find("MeshHeadingList")
        if mesh_list is not None:
            for heading in mesh_list.findall("MeshHeading"):
                desc = heading.find("DescriptorName")
                if desc is not None and desc.text:
                    mesh_terms.append(desc.text)

        # Article IDs (PMC, DOI)
        pmc_id: str | None = None
        doi: str | None = None
        pubmed_data = article.find("PubmedData")
        if pubmed_data is not None:
            id_list = pubmed_data.find("ArticleIdList")
            if id_list is not None:
                for aid in id_list.findall("ArticleId"):
                    id_type = aid.get("IdType", "")
                    if id_type == "pmc" and aid.text:
                        pmc_id = aid.text if aid.text.startswith("PMC") else f"PMC{aid.text}"
                    elif id_type == "doi" and aid.text:
                        doi = aid.text

        return {
            "pmid": pmid,
            "title": title,
            "authors": authors,
            "journal": journal,
            "year": year,
            "abstract": abstract,
            "mesh_terms": mesh_terms,
            "pmc_id": pmc_id,
            "doi": doi,
            "has_full_text": pmc_id is not None,
        }

    @staticmethod
    def _get_text_content(element: etree._Element) -> str:
        """Extract all text content from an element, including mixed content."""
        return "".join(element.itertext()).strip()

    async def _request_with_retry(
        self,
        url: str,
        params: dict[str, Any],
        parse_json: bool = True,
        max_retries: int = 3,
    ) -> Any:
        """Make an HTTP request with rate limiting and retry logic.

        Args:
            url: Request URL.
            params: Query parameters.
            parse_json: If True, parse response as JSON. Otherwise return text.
            max_retries: Maximum retry attempts.

        Returns:
            Parsed JSON dict, response text, or None on failure.
        """
        for attempt in range(max_retries):
            async with self._semaphore:
                # Rate limiting
                await self._rate_limit()

                try:
                    response = await self._client.get(url, params=params)
                    response.raise_for_status()

                    if parse_json:
                        return response.json()
                    else:
                        return response.text

                except httpx.HTTPStatusError as exc:
                    logger.warning(
                        "HTTP %d for %s (attempt %d/%d): %s",
                        exc.response.status_code, url, attempt + 1, max_retries, exc,
                    )
                    if exc.response.status_code == 429:
                        # Rate limited — back off more aggressively
                        backoff = 2 ** (attempt + 1)
                        logger.info("Rate limited, backing off %ds", backoff)
                        await asyncio.sleep(backoff)
                    elif exc.response.status_code >= 500:
                        backoff = 2 ** attempt
                        await asyncio.sleep(backoff)
                    else:
                        # Client error (4xx other than 429) — don't retry
                        return None

                except httpx.RequestError as exc:
                    logger.warning(
                        "Request error for %s (attempt %d/%d): %s",
                        url, attempt + 1, max_retries, exc,
                    )
                    backoff = 2 ** attempt
                    await asyncio.sleep(backoff)

        logger.error("All %d retries exhausted for %s", max_retries, url)
        return None

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()
