"""get_next_page tool — retrieve subsequent pages of a paginated MCP response.

When an OncoContext tool response exceeds the ~900KB page limit it is
automatically split into pages.  Page 1 is returned directly with a
``_pagination`` metadata block.  Call this tool with the ``session_id``
and desired ``page`` number to retrieve any subsequent page.
"""

from __future__ import annotations

import logging

from oncocontext.services.response_paginator import get_paginator

logger = logging.getLogger(__name__)


async def get_next_page(session_id: str, page: int) -> str:
    """Retrieve a specific page from a previously paginated tool response.

    When a tool response exceeds the size limit it is split into pages.
    Page 1 is returned inline with a ``_pagination`` block containing the
    ``session_id``.  Use this tool to fetch pages 2, 3, … until
    ``_pagination.has_more`` is ``false``.

    Args:
        session_id: The ``session_id`` value from the ``_pagination`` block
            of the previous page.
        page: 1-based page number to retrieve.

    Returns:
        JSON string containing the requested page data plus an updated
        ``_pagination`` metadata block.
    """
    logger.info("get_next_page called: session_id=%s, page=%d", session_id, page)
    paginator = get_paginator()
    return paginator.get_page(session_id=session_id, page=page)
