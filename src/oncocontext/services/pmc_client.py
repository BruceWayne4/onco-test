"""PMC BioC API client — fetch full-text JSON by PMC ID.

Provides async access to the PMC BioC RESTful API for retrieving
full-text articles in structured BioC JSON format.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from oncocontext.config import settings
from oncocontext.storage.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class PMCClient:
    """Async client for PMC BioC API.

    Handles:
        - Fetch full-text BioC JSON for a given PMC ID
        - Check availability of full text in PMC Open Access
        - Response caching (30-day TTL)
        - Timeout handling
    """

    def __init__(self, cache: CacheManager | None = None) -> None:
        """Initialize PMC client.

        Args:
            cache: Optional CacheManager for response caching.
        """
        self._base_url = settings.PMC_BIOC_URL
        self._cache = cache
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            follow_redirects=True,
        )
        self._min_interval = 1.0 / settings.PMC_RATE_LIMIT
        self._last_request_time: float = 0.0
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_FETCHES)

    # ── Public API ────────────────────────────────────────────────────────────

    async def fetch_bioc(self, pmc_id: str, fmt: str = "json") -> dict | None:
        """Fetch full-text BioC JSON document for a PMC ID.

        Args:
            pmc_id: PMC identifier (e.g., 'PMC10234567' or '10234567').
            fmt: Response format — only 'json' is used in Phase 1.

        Returns:
            Parsed BioC JSON dict, or None if not available.
        """
        normalized_id = self._normalize_pmc_id(pmc_id)

        # Check cache
        if self._cache:
            cache_key = CacheManager.make_key("pmc_bioc", normalized_id)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for PMC BioC: %s", normalized_id)
                return cached

        url = f"{self._base_url}/{normalized_id}/unicode"
        data = await self._request_with_retry(url)

        if data is None:
            return None

        # The PMC BioC API may return a list of collections, e.g.:
        # [{"bioctype": "BioCCollection", "documents": [...], ...}]
        # Normalise to a single dict to match the declared return type.
        if isinstance(data, list):
            if not data:
                return None
            data = data[0]

        # Cache the result
        if self._cache and data:
            cache_key = CacheManager.make_key("pmc_bioc", normalized_id)
            self._cache.set(cache_key, data, ttl=settings.CACHE_TTL_PMC)

        logger.info("Fetched BioC JSON for %s", normalized_id)
        return data

    async def check_availability(self, pmc_id: str) -> bool:
        """Check whether full text is available for a PMC ID in the Open Access subset.

        Uses a lightweight HEAD request to avoid downloading the full document.

        Args:
            pmc_id: PMC identifier.

        Returns:
            True if full text is available.
        """
        normalized_id = self._normalize_pmc_id(pmc_id)
        url = f"{self._base_url}/{normalized_id}/unicode"

        try:
            async with self._semaphore:
                await self._rate_limit()
                response = await self._client.head(url)
                return response.status_code == 200
        except httpx.RequestError as exc:
            logger.warning("Error checking PMC availability for %s: %s", normalized_id, exc)
            return False

    async def check_availability_batch(self, pmc_ids: list[str]) -> dict[str, bool]:
        """Check which PMC IDs have full text available.

        Args:
            pmc_ids: List of PMC identifiers.

        Returns:
            Dict of pmc_id → bool.
        """
        results: dict[str, bool] = {}
        for pmc_id in pmc_ids:
            results[pmc_id] = await self.check_availability(pmc_id)
        return results

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ── Internal Methods ──────────────────────────────────────────────────────

    @staticmethod
    def _normalize_pmc_id(pmc_id: str) -> str:
        """Ensure PMC ID has the 'PMC' prefix."""
        pmc_id = pmc_id.strip()
        if not pmc_id.upper().startswith("PMC"):
            return f"PMC{pmc_id}"
        return pmc_id

    async def _request_with_retry(
        self,
        url: str,
        max_retries: int = 3,
    ) -> dict | None:
        """Make an HTTP GET request with rate limiting and retry.

        Args:
            url: Full request URL.
            max_retries: Maximum retry attempts.

        Returns:
            Parsed JSON dict, or None on failure.
        """
        for attempt in range(max_retries):
            async with self._semaphore:
                await self._rate_limit()

                try:
                    response = await self._client.get(url)

                    if response.status_code == 404:
                        logger.debug("PMC article not found: %s", url)
                        return None

                    response.raise_for_status()

                    # F6: Check Content-Type BEFORE attempting JSON parse.
                    # The PMC BioC endpoint sometimes returns an HTML error page
                    # (e.g. a 200 OK maintenance page) whose body is not JSON.
                    # Calling response.json() on HTML raises a ValueError and the
                    # old code would retry 3 times, burning rate-limit quota.
                    # If the server sends HTML we log a warning and return None
                    # immediately — there is no point retrying the same URL.
                    content_type = response.headers.get("content-type", "")
                    if "html" in content_type.lower():
                        logger.warning(
                            "PMC BioC returned HTML instead of JSON for %s "
                            "(Content-Type: %s) — article may not be in OA subset",
                            url, content_type,
                        )
                        return None

                    return response.json()

                except httpx.HTTPStatusError as exc:
                    logger.warning(
                        "HTTP %d for %s (attempt %d/%d)",
                        exc.response.status_code, url, attempt + 1, max_retries,
                    )
                    if exc.response.status_code == 429:
                        backoff = 2 ** (attempt + 1)
                        await asyncio.sleep(backoff)
                    elif exc.response.status_code >= 500:
                        backoff = 2 ** attempt
                        await asyncio.sleep(backoff)
                    else:
                        return None

                except httpx.RequestError as exc:
                    # Network-level errors (connection reset, timeout, etc.) are
                    # worth retrying — the server may recover.
                    logger.warning(
                        "Network error for %s (attempt %d/%d): %s",
                        url, attempt + 1, max_retries, exc,
                    )
                    backoff = 2 ** attempt
                    await asyncio.sleep(backoff)

                except ValueError as exc:
                    # JSON parse failure without an HTML Content-Type header.
                    # This means the server sent unexpected non-JSON content.
                    # Retrying is unlikely to help — bail out immediately.
                    logger.warning(
                        "JSON parse error for %s (not retrying): %s", url, exc,
                    )
                    return None

        logger.error("All %d retries exhausted for %s", max_retries, url)
        return None

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()
