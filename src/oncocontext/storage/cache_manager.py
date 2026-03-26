"""3-tier cache manager — L1 in-memory LRU, L2 diskcache, L3 ChromaDB.

Wraps diskcache for persistent caching with configurable TTLs.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Awaitable

import diskcache

from oncocontext.config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Cache wrapping diskcache for persistent caching.

    Tiers:
        - L1: In-memory dict cache (fast, per-process)
        - L2: diskcache on disk (2GB, configurable TTLs)

    TTLs:
        - PubMed search results: 24 hours (86400s)
        - PMC full-text JSON: 30 days (2592000s)
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        """Initialize cache manager.

        Args:
            cache_dir: Path to diskcache directory.
                Defaults to data/cache/.
        """
        self._l1: dict[str, Any] = {}
        dir_path = cache_dir or str(settings.CACHE_DIR)
        try:
            self._cache: diskcache.Cache | None = diskcache.Cache(
                dir_path,
                size_limit=settings.CACHE_SIZE_LIMIT_GB * 1024 * 1024 * 1024,
            )
            logger.info("Initialized diskcache at %s", dir_path)
        except Exception as exc:
            logger.warning("Failed to initialize diskcache at %s: %s — caching disabled", dir_path, exc)
            self._cache = None

    def get(self, key: str) -> Any | None:
        """Get a value from cache (checks L1, then L2).

        Args:
            key: Cache key.

        Returns:
            Cached value, or None if not found.
        """
        # L1 check
        if key in self._l1:
            logger.debug("L1 cache hit for key=%s", key)
            return self._l1[key]

        # L2 check
        if self._cache is not None:
            try:
                value = self._cache.get(key)
                if value is not None:
                    logger.debug("L2 cache hit for key=%s", key)
                    self._l1[key] = value  # promote to L1
                    return value
            except Exception as exc:
                logger.warning("Error reading from diskcache for key=%s: %s", key, exc)

        return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value in cache (writes to both L1 and L2).

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds. None = no expiry.
        """
        # L1
        self._l1[key] = value

        # L2
        if self._cache is not None:
            try:
                if ttl is not None:
                    self._cache.set(key, value, expire=ttl)
                else:
                    self._cache.set(key, value)
                logger.debug("Cached key=%s (ttl=%s)", key, ttl)
            except Exception as exc:
                logger.warning("Error writing to diskcache for key=%s: %s", key, exc)

    async def get_or_fetch(
        self,
        key: str,
        fetcher: Callable[[], Awaitable[Any]],
        ttl: int | None = None,
    ) -> Any:
        """Get from cache, or fetch and cache if missing.

        Args:
            key: Cache key.
            fetcher: Async callable to fetch the value if not cached.
            ttl: Time-to-live in seconds for the cached value.

        Returns:
            The cached or freshly fetched value.
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        value = await fetcher()
        if value is not None:
            self.set(key, value, ttl=ttl)
        return value

    def clear(self) -> None:
        """Clear all cache entries."""
        self._l1.clear()
        if self._cache is not None:
            try:
                self._cache.clear()
                logger.info("Cache cleared")
            except Exception as exc:
                logger.warning("Error clearing diskcache: %s", exc)

    def close(self) -> None:
        """Close the diskcache."""
        if self._cache is not None:
            try:
                self._cache.close()
            except Exception:
                pass

    @staticmethod
    def make_key(*parts: str) -> str:
        """Create a deterministic cache key from parts.

        Args:
            *parts: Strings to hash together.

        Returns:
            A hex digest cache key.
        """
        joined = "|".join(parts)
        return hashlib.sha256(joined.encode()).hexdigest()
