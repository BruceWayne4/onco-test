"""ResponsePaginator — splits oversized MCP tool responses into pages.

Claude Desktop app enforces a ~1MB limit on MCP tool responses. This module:
  1. Checks response sizes before returning them.
  2. Splits large responses into pages using tool-aware strategies.
  3. Stores pages in a thread-safe in-memory SessionStore with TTL eviction.
  4. Returns page 1 with a _pagination metadata block so Claude can call
     get_next_page to retrieve subsequent pages.

Fallback behaviour: if pagination itself fails for any reason, the response is
truncated to MAX_RESPONSE_SIZE_KB with a clear warning block appended — so
Claude always gets *something* useful rather than a hard error.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from oncocontext.config import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_BYTES = settings.MAX_RESPONSE_SIZE_KB * 1024  # e.g. 900 * 1024 = 921_600 bytes
_SESSION_TTL = settings.PAGINATION_SESSION_TTL     # 1800 s = 30 min
_MAX_SESSIONS = settings.PAGINATION_MAX_SESSIONS   # e.g. 50


# ── Session Store ─────────────────────────────────────────────────────────────


@dataclass
class _Session:
    """A single paginated response session."""

    pages: list[dict]
    tool_name: str
    created_at: float = field(default_factory=time.monotonic)

    @property
    def expires_in(self) -> int:
        """Remaining TTL in seconds (clamped to 0)."""
        elapsed = time.monotonic() - self.created_at
        return max(0, int(_SESSION_TTL - elapsed))

    @property
    def is_expired(self) -> bool:
        return self.expires_in == 0


class SessionStore:
    """Thread-safe in-memory store for paginated response sessions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[str, _Session] = {}

    def create(self, pages: list[dict], tool_name: str) -> str:
        """Store pages and return a new session_id."""
        session_id = str(uuid.uuid4())
        with self._lock:
            self._evict()
            # If at capacity, evict oldest session to make room
            if len(self._store) >= _MAX_SESSIONS:
                oldest = min(self._store, key=lambda k: self._store[k].created_at)
                del self._store[oldest]
                logger.warning("Session store at capacity; evicted oldest session %s", oldest)
            self._store[session_id] = _Session(pages=pages, tool_name=tool_name)
        logger.debug(
            "Created pagination session %s: %d pages for tool '%s'",
            session_id, len(pages), tool_name,
        )
        return session_id

    def get_page(self, session_id: str, page: int) -> tuple[dict | None, str | None]:
        """Return (page_dict, error_message).

        page is 1-based.  Returns (None, error_str) on any problem.
        """
        with self._lock:
            session = self._store.get(session_id)
            if session is None:
                return None, (
                    f"Session '{session_id}' not found. Sessions expire after "
                    f"{_SESSION_TTL // 60} minutes. Re-run the original tool to start a new session."
                )
            if session.is_expired:
                del self._store[session_id]
                return None, (
                    f"Session '{session_id}' has expired (TTL={_SESSION_TTL // 60} min). "
                    "Re-run the original tool to start a new session."
                )
            total = len(session.pages)
            if page < 1 or page > total:
                return None, (
                    f"Invalid page {page} for session '{session_id}'. "
                    f"Valid range: 1–{total}."
                )
            return session.pages[page - 1], None

    def _evict(self) -> None:
        """Remove expired sessions (call while holding lock)."""
        expired = [sid for sid, s in self._store.items() if s.is_expired]
        for sid in expired:
            del self._store[sid]
        if expired:
            logger.debug("Evicted %d expired pagination session(s)", len(expired))

    def session_info(self, session_id: str) -> dict | None:
        """Return lightweight session metadata or None if not found/expired."""
        with self._lock:
            session = self._store.get(session_id)
            if session is None or session.is_expired:
                return None
            return {
                "session_id": session_id,
                "tool_name": session.tool_name,
                "total_pages": len(session.pages),
                "expires_in_seconds": session.expires_in,
            }


# ── Splitting Helpers ─────────────────────────────────────────────────────────


def _byte_size(obj: Any) -> int:
    """Return JSON byte size of obj as it would be serialised in the final output (indent=2)."""
    return len(json.dumps(obj, indent=2, default=str).encode("utf-8"))


def _split_string_into_chunks(text: str, max_chars: int) -> list[str]:
    """Split a long string into chunks of roughly max_chars, preferring line breaks."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        # Try to break on a double newline near max_chars
        split_pos = remaining.rfind("\n\n", 0, max_chars)
        if split_pos == -1:
            # Fall back to single newline
            split_pos = remaining.rfind("\n", 0, max_chars)
        if split_pos == -1:
            # Last resort: hard split
            split_pos = max_chars
        chunks.append(remaining[:split_pos])
        remaining = remaining[split_pos:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


# ~1KB for the _pagination block that will be appended to each page
_PAGINATION_BLOCK_OVERHEAD = 1_024


def _split_list_into_chunks(items: list, base_payload: dict, list_key: str) -> list[list]:
    """Split items so each assembled page dict (base + list_key=chunk + _pagination) fits in _MAX_BYTES.

    Uses a greedy algorithm that measures the *actual* assembled page size so there is
    no guesswork about JSON indent=2 overhead.

    Args:
        items: The list items to be distributed across pages.
        base_payload: All non-list fields from the result dict (will be merged into each page).
        list_key: The key name the list will be stored under in each page dict.
    """
    def _page_bytes(chunk: list) -> int:
        page = {**base_payload, list_key: chunk}
        return _byte_size(page) + _PAGINATION_BLOCK_OVERHEAD

    groups: list[list] = []
    current: list = []

    for item in items:
        candidate = current + [item]
        if current and _page_bytes(candidate) > _MAX_BYTES:
            groups.append(current)
            current = [item]
        else:
            current = candidate

    if current:
        groups.append(current)

    # Safety: if any single item is so huge it exceeds the limit alone, keep it as its own page
    # (the recursive splitter in ResponsePaginator._split will catch and truncate it)
    return groups if groups else [[]]

def _make_pagination_block(
    session_id: str,
    page: int,
    total_pages: int,
    expires_in: int,
) -> dict:
    has_more = page < total_pages
    hint = (
        f"Call get_next_page with session_id='{session_id}' and page={page + 1} to get the next page."
        if has_more
        else "This is the last page — no more data to retrieve."
    )
    return {
        "session_id": session_id,
        "page": page,
        "total_pages": total_pages,
        "has_more": has_more,
        "expires_in_seconds": expires_in,
        "hint": hint,
    }


# ── Tool-Aware Splitters ──────────────────────────────────────────────────────


def _split_crawl_and_report(result: dict) -> list[dict]:
    """Split crawl_and_report responses by splitting report_markdown."""
    report_md: str = result.get("report_markdown", "") or ""
    base = {k: v for k, v in result.items() if k != "report_markdown"}
    base_bytes = _byte_size(base)

    # Budget for the markdown text itself per page
    md_budget = max(_MAX_BYTES - base_bytes - 500, 50_000)
    # Try to split on "## " section headings first
    sections = report_md.split("\n## ")
    pages: list[dict] = []
    current_md = ""

    for i, sec in enumerate(sections):
        chunk = ("\n## " + sec) if i > 0 else sec
        if len((current_md + chunk).encode("utf-8")) > md_budget and current_md:
            pages.append({**base, "report_markdown": current_md})
            current_md = chunk
        else:
            current_md += chunk

    if current_md:
        pages.append({**base, "report_markdown": current_md})

    # If any single page still too large, do a character-level split
    final_pages: list[dict] = []
    for pg in pages:
        md = pg.get("report_markdown", "")
        if len(json.dumps(pg, default=str).encode("utf-8")) > _MAX_BYTES:
            sub_chunks = _split_string_into_chunks(md, md_budget // len("x".encode("utf-8")))
            for sc in sub_chunks:
                final_pages.append({**base, "report_markdown": sc})
        else:
            final_pages.append(pg)

    return final_pages if final_pages else [result]


def _split_get_paper_details(result: dict) -> list[dict]:
    """Split get_paper_details by splitting the sections list."""
    sections_list: list = result.get("sections") or []
    base = {k: v for k, v in result.items() if k != "sections"}

    if not sections_list:
        return [result]

    groups = _split_list_into_chunks(sections_list, base, "sections")
    return [{**base, "sections": grp} for grp in groups]


def _split_crawl_supplementary(result: dict) -> list[dict]:
    """Split crawl_supplementary by splitting the files list."""
    files_list: list = result.get("files") or []
    base = {k: v for k, v in result.items() if k != "files"}

    if not files_list:
        return [result]

    groups = _split_list_into_chunks(files_list, base, "files")
    return [{**base, "files": grp} for grp in groups]


def _split_search_literature(result: dict) -> list[dict]:
    """Split search_literature by splitting the papers list."""
    papers_list: list = result.get("papers") or []
    base = {k: v for k, v in result.items() if k != "papers"}

    if not papers_list:
        return [result]

    groups = _split_list_into_chunks(papers_list, base, "papers")
    return [{**base, "papers": grp} for grp in groups]


def _split_deep_search(result: dict) -> list[dict]:
    """Split deep_search by splitting the results list."""
    results_list: list = result.get("results") or []
    base = {k: v for k, v in result.items() if k != "results"}

    if not results_list:
        return [result]

    groups = _split_list_into_chunks(results_list, base, "results")
    return [{**base, "results": grp} for grp in groups]


def _split_cross_reference(result: dict) -> list[dict]:
    """Split cross_reference by distributing agreements, contradictions etc across pages.

    Uses the same actual-size measurement approach as _split_list_into_chunks.
    """
    list_fields = ["agreements", "contradictions", "novel_findings", "suggested_follow_up"]
    scalars = {k: v for k, v in result.items() if k not in list_fields}

    # Build a flat list of (field_name, item) tuples
    all_items: list[tuple[str, Any]] = []
    for fld in list_fields:
        for item in (result.get(fld) or []):
            all_items.append((fld, item))

    if not all_items:
        return [result]

    def _group_to_page(group: list[tuple[str, Any]]) -> dict:
        page_dict: dict[str, Any] = {**scalars}
        for fld in list_fields:
            page_dict[fld] = [item for fname, item in group if fname == fld]
        return page_dict

    def _page_bytes(group: list[tuple[str, Any]]) -> int:
        return _byte_size(_group_to_page(group)) + _PAGINATION_BLOCK_OVERHEAD

    pages: list[dict] = []
    current_group: list[tuple[str, Any]] = []

    for field_name, item in all_items:
        candidate = current_group + [(field_name, item)]
        if current_group and _page_bytes(candidate) > _MAX_BYTES:
            pages.append(_group_to_page(current_group))
            current_group = [(field_name, item)]
        else:
            current_group = candidate

    if current_group:
        pages.append(_group_to_page(current_group))

    return pages if pages else [result]


def _split_generic(result: dict) -> list[dict]:
    """Generic fallback: find the largest string or list field and split it."""
    # Find largest list field
    largest_list_key = None
    largest_list_size = 0
    for k, v in result.items():
        if isinstance(v, list) and _byte_size(v) > largest_list_size:
            largest_list_key = k
            largest_list_size = _byte_size(v)

    if largest_list_key:
        base = {k: v for k, v in result.items() if k != largest_list_key}
        groups = _split_list_into_chunks(result[largest_list_key], base, largest_list_key)
        return [{**base, largest_list_key: grp} for grp in groups]

    # Find largest string field
    largest_str_key = None
    largest_str_size = 0
    for k, v in result.items():
        if isinstance(v, str) and len(v.encode("utf-8")) > largest_str_size:
            largest_str_key = k
            largest_str_size = len(v.encode("utf-8"))

    if largest_str_key:
        text = result[largest_str_key]
        base = {k: v for k, v in result.items() if k != largest_str_key}
        base_bytes = _byte_size(base)
        budget = max(_MAX_BYTES - base_bytes - _PAGINATION_BLOCK_OVERHEAD - 1_024, 10_000)
        chunks = _split_string_into_chunks(text, budget)
        return [{**base, largest_str_key: c} for c in chunks]

    # Nothing obvious to split — return as single page
    return [result]


_TOOL_SPLITTERS = {
    "crawl_and_report": _split_crawl_and_report,
    "get_paper_details": _split_get_paper_details,
    "crawl_supplementary": _split_crawl_supplementary,
    "search_literature": _split_search_literature,
    "deep_search": _split_deep_search,
    "cross_reference": _split_cross_reference,
}


# ── Truncation Fallback ───────────────────────────────────────────────────────


def _truncate_response(output_json: str, tool_name: str) -> str:
    """Last-resort fallback: truncate JSON string and append a warning block."""
    limit = _MAX_BYTES - 512  # leave 512 bytes for the warning suffix
    truncated_bytes = output_json.encode("utf-8")[:limit]
    # Decode with error replacement to avoid UnicodeDecodeError
    truncated = truncated_bytes.decode("utf-8", errors="replace")

    warning = json.dumps({
        "_truncation_warning": {
            "message": (
                f"Response from '{tool_name}' exceeded the {settings.MAX_RESPONSE_SIZE_KB}KB "
                "limit and pagination also failed. The response was truncated. "
                "To retrieve the full data, try re-running the tool with fewer results "
                "(e.g. smaller max_results, fewer sections, or crawl_supplementary_files=False)."
            ),
            "truncated": True,
        }
    }, indent=2)

    # Try to produce valid JSON; if the truncated text is invalid JSON, wrap it
    try:
        parsed = json.loads(truncated)
        if isinstance(parsed, dict):
            parsed["_truncation_warning"] = json.loads(warning)["_truncation_warning"]
            return json.dumps(parsed, indent=2, default=str)
    except (json.JSONDecodeError, ValueError):
        pass

    # Wrap as a plain object with raw text + warning
    return json.dumps(
        {
            "raw_truncated_response": truncated,
            "_truncation_warning": json.loads(warning)["_truncation_warning"],
        },
        indent=2,
        default=str,
    )


# ── Main ResponsePaginator ────────────────────────────────────────────────────


class ResponsePaginator:
    """Splits large MCP tool responses into pages stored in a SessionStore.

    Usage (in server.py):
        paginator = ResponsePaginator()
        output = json.dumps(result, indent=2, default=str)
        output = paginator.paginate_if_needed(output, tool_name="crawl_and_report")
        return output
    """

    def __init__(self) -> None:
        self.store = SessionStore()

    def paginate_if_needed(self, output_json: str, tool_name: str) -> str:
        """Return output_json unchanged if small, or page-1 of a paginated split.

        Falls back to truncation if pagination itself fails.
        """
        size = len(output_json.encode("utf-8"))
        if size <= _MAX_BYTES:
            return output_json  # Fast path: no pagination needed

        logger.info(
            "Response from '%s' is %d bytes (limit=%d). Paginating…",
            tool_name, size, _MAX_BYTES,
        )

        try:
            result_dict = json.loads(output_json)
        except json.JSONDecodeError:
            logger.warning("Cannot parse JSON for pagination — truncating instead")
            return _truncate_response(output_json, tool_name)

        try:
            pages = self._split(result_dict, tool_name)
        except Exception as exc:
            logger.error("Pagination split failed for '%s': %s — truncating", tool_name, exc)
            return _truncate_response(output_json, tool_name)

        if len(pages) == 0:
            return output_json

        if len(pages) == 1:
            # Still too large even after split (edge case with huge single item)
            page_json = json.dumps(pages[0], indent=2, default=str)
            if len(page_json.encode("utf-8")) > _MAX_BYTES:
                logger.warning(
                    "Single-page split still exceeds limit for '%s' — truncating", tool_name
                )
                return _truncate_response(page_json, tool_name)
            return page_json

        # Store pages in session
        try:
            session_id = self.store.create(pages, tool_name=tool_name)
        except Exception as exc:
            logger.error("Session store failed for '%s': %s — truncating", tool_name, exc)
            return _truncate_response(output_json, tool_name)

        # Build and return page 1
        session = self.store.session_info(session_id)
        expires_in = session["expires_in_seconds"] if session else _SESSION_TTL
        page1 = {
            **pages[0],
            "_pagination": _make_pagination_block(
                session_id=session_id,
                page=1,
                total_pages=len(pages),
                expires_in=expires_in,
            ),
        }
        page1_json = json.dumps(page1, indent=2, default=str)

        # Verify page 1 itself fits; if not, truncate it
        if len(page1_json.encode("utf-8")) > _MAX_BYTES:
            logger.warning(
                "Page 1 of '%s' still exceeds limit — truncating page 1", tool_name
            )
            return _truncate_response(page1_json, tool_name)

        logger.info(
            "Paginated '%s' into %d pages. Session=%s, page_1_size=%d bytes",
            tool_name, len(pages), session_id, len(page1_json.encode("utf-8")),
        )
        return page1_json

    def get_page(self, session_id: str, page: int) -> str:
        """Retrieve a specific page from a session.

        Returns JSON string with the page data and updated _pagination block.
        On error, returns a JSON error dict.
        """
        page_data, error = self.store.get_page(session_id, page)
        if error or page_data is None:
            return json.dumps({"error": error or "Unknown error", "session_id": session_id, "requested_page": page}, indent=2)

        # Re-fetch session info for fresh expires_in
        session = self.store.session_info(session_id)
        total_pages = session["total_pages"] if session else page
        expires_in = session["expires_in_seconds"] if session else 0

        result: dict = {
            **page_data,
            "_pagination": _make_pagination_block(
                session_id=session_id,
                page=page,
                total_pages=total_pages,
                expires_in=expires_in,
            ),
        }
        return json.dumps(result, indent=2, default=str)

    def _split(self, result: dict, tool_name: str) -> list[dict]:
        """Dispatch to the correct tool-aware splitter."""
        splitter = _TOOL_SPLITTERS.get(tool_name, _split_generic)
        pages = splitter(result)
        # Validate: ensure each page is below limit, recursively split if not
        final: list[dict] = []
        for pg in pages:
            pg_bytes = _byte_size(pg)
            if pg_bytes > _MAX_BYTES:
                # Recursive fallback: try generic splitter on this oversized sub-page
                sub_pages = _split_generic(pg)
                final.extend(sub_pages)
            else:
                final.append(pg)
        return final


# ── Module-level singleton ────────────────────────────────────────────────────

_paginator: ResponsePaginator | None = None


def get_paginator() -> ResponsePaginator:
    """Return the module-level ResponsePaginator singleton."""
    global _paginator
    if _paginator is None:
        _paginator = ResponsePaginator()
    return _paginator
