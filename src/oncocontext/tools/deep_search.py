"""deep_search tool — Semantic search across full-text papers indexed in ChromaDB.

Section-aware filtering with section boost factors and cross-encoder reranking
for precision retrieval.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from oncocontext.config import settings
from oncocontext.services.embedder import Embedder
from oncocontext.services.reranker import Reranker
from oncocontext.storage.chroma_manager import ChromaManager
from oncocontext.storage.sqlite_manager import SQLiteManager

logger = logging.getLogger(__name__)

# ── Module-level lazy singletons ──────────────────────────────────────────────

_embedder: Embedder | None = None
_chroma: ChromaManager | None = None
_sqlite: SQLiteManager | None = None
_reranker: Reranker | None = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _get_chroma() -> ChromaManager:
    global _chroma
    if _chroma is None:
        _chroma = ChromaManager()
    return _chroma


def _get_sqlite() -> SQLiteManager:
    global _sqlite
    if _sqlite is None:
        _sqlite = SQLiteManager()
    return _sqlite


def _get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


# ── Query Type Detection ──────────────────────────────────────────────────────

# Keywords that signal query type for section boosting
_METHOD_KEYWORDS = {
    "how", "protocol", "method", "procedure", "strategy", "gating",
    "antibody", "clone", "fluorochrome", "panel", "stain", "staining",
    "conjugate", "assay", "technique", "preparation", "culture",
    "co-culture", "coculture", "sorting", "facs",
}

_RESULTS_KEYWORDS = {
    "found", "showed", "data", "result", "expression", "level",
    "percentage", "percent", "mfi", "frequency", "population",
    "increase", "decrease", "significant", "correlation",
    "measured", "observed", "detected", "quantified",
}

_DISCUSSION_KEYWORDS = {
    "why", "explain", "mechanism", "suggest", "imply", "hypothesis",
    "interpret", "interpretation", "meaning", "significance",
    "consistent", "inconsistent", "contradiction", "paradox",
}


def _detect_query_type(query: str) -> str:
    """Detect query type from keywords for section boosting.

    Returns:
        One of 'methods', 'results', 'discussion', or 'general'.
    """
    query_lower = query.lower()
    words = set(query_lower.split())

    method_score = len(words & _METHOD_KEYWORDS)
    results_score = len(words & _RESULTS_KEYWORDS)
    discussion_score = len(words & _DISCUSSION_KEYWORDS)

    # Also check for phrases
    if any(p in query_lower for p in ["gating strategy", "antibody panel", "how was", "what method"]):
        method_score += 3
    if any(p in query_lower for p in ["what were", "what data", "what results"]):
        results_score += 3
    if any(p in query_lower for p in ["why does", "what explains", "what mechanism"]):
        discussion_score += 3

    max_score = max(method_score, results_score, discussion_score)
    if max_score == 0:
        return "general"

    if method_score == max_score:
        return "methods"
    elif results_score == max_score:
        return "results"
    else:
        return "discussion"


def _apply_section_boost(
    distances: list[float],
    metadatas: list[dict],
    query_type: str,
) -> list[float]:
    """Apply section-specific boost factors to scores.

    Converts ChromaDB distances (lower = better) to similarity scores (higher = better)
    then applies section boost multipliers.

    Args:
        distances: ChromaDB cosine distances (0 = identical, 2 = opposite).
        metadatas: Metadata dicts with 'section' field.
        query_type: One of 'methods', 'results', 'discussion', 'general'.

    Returns:
        List of boosted similarity scores (0-1 range, higher = better).
    """
    boosted_scores = []

    for dist, meta in zip(distances, metadatas):
        # Convert cosine distance to similarity: sim = 1 - dist
        # ChromaDB cosine distance is in [0, 2], but for normalized vectors it's [0, 2]
        sim = max(0.0, 1.0 - dist)

        section = meta.get("section", "").lower()

        # Apply boost based on query type
        if query_type == "methods":
            if section in ("methods", "materials"):
                sim *= settings.BOOST_METHODS
            elif section == "discussion":
                sim *= 0.8
        elif query_type == "results":
            if section == "results":
                sim *= settings.BOOST_RESULTS
            elif section == "discussion":
                sim *= 1.2
            elif section == "methods":
                sim *= 0.8
        elif query_type == "discussion":
            if section == "discussion":
                sim *= settings.BOOST_DISCUSSION
            elif section == "results":
                sim *= 1.2
        # 'general' → no boost

        # Clamp to [0, 1]
        sim = min(1.0, max(0.0, sim))
        boosted_scores.append(sim)

    return boosted_scores


# ── Main Tool Function ────────────────────────────────────────────────────────


async def deep_search(
    query: str,
    search_scope: str = "all",
    max_results: int = 10,
    min_relevance: float = 0.3,
    paper_ids: list[str] | None = None,
) -> dict:
    """Semantic search across full-text papers already indexed in ChromaDB.

    Algorithm:
        1. Embed query with PubMedBERT (768d)
        2. Vector search ChromaDB literature_chunks (top 50, with section filter)
        3. Detect query type for section boosting
        4. Apply section boost multipliers
        5. Cross-encoder reranking (top max_results from 50 candidates)
        6. Filter by min_relevance threshold
        7. Assemble ±1 paragraph context from SQLite
        8. Format citations with PMID + section + paragraph

    Args:
        query: Detailed mechanistic question to search for in paper full text.
        search_scope: Restrict to a section type ('all', 'methods', 'results', etc.).
        max_results: Number of top chunks after reranking (1-25).
        min_relevance: Minimum relevance score threshold (0.0-1.0).
        paper_ids: Restrict search to specific PMIDs.

    Returns:
        Dict with results (chunks + citations), search stats, and strategy description.
    """
    start_time = time.time()

    embedder = _get_embedder()
    chroma = _get_chroma()
    sqlite = _get_sqlite()
    await sqlite.init_db()

    # Check if any papers are indexed
    lit_stats = chroma.get_collection_stats("literature")
    if lit_stats["count"] == 0:
        return {
            "results": [],
            "total_indexed_papers": 0,
            "total_indexed_chunks": 0,
            "search_strategy": (
                "No papers are indexed yet. Use 'search_literature' to find papers, "
                "then 'get_paper_details' with fetch_full_text=True to index them. "
                "Once papers are indexed, deep_search will search their full text."
            ),
        }

    # Step 1: Embed the query
    try:
        query_embedding = embedder.embed_text(query)
    except Exception as exc:
        logger.error("Failed to embed query: %s", exc)
        return {
            "results": [],
            "total_indexed_papers": 0,
            "total_indexed_chunks": lit_stats["count"],
            "search_strategy": f"Error embedding query: {exc}",
        }

    # Step 2: Build where filter
    where_filter: dict[str, Any] | None = None

    if search_scope != "all":
        where_filter = {"section": search_scope}

    if paper_ids:
        paper_filter = {"paper_pmid": {"$in": paper_ids}}
        if where_filter:
            where_filter = {"$and": [where_filter, paper_filter]}
        else:
            where_filter = paper_filter

    # Step 3: Vector search
    try:
        search_results = chroma.search(
            query_embedding=query_embedding,
            collection="literature",
            n_results=settings.VECTOR_TOP_K,
            where=where_filter,
        )
    except Exception as exc:
        logger.error("ChromaDB search failed: %s", exc)
        return {
            "results": [],
            "total_indexed_papers": 0,
            "total_indexed_chunks": lit_stats["count"],
            "search_strategy": f"Vector store query failed: {exc}",
        }

    # Unpack results (ChromaDB returns lists of lists for batch queries)
    result_ids = search_results.get("ids", [[]])[0]
    result_distances = search_results.get("distances", [[]])[0]
    result_documents = search_results.get("documents", [[]])[0]
    result_metadatas = search_results.get("metadatas", [[]])[0]

    if not result_ids:
        indexed_papers = await sqlite.get_indexed_paper_count()
        return {
            "results": [],
            "total_indexed_papers": indexed_papers,
            "total_indexed_chunks": lit_stats["count"],
            "search_strategy": (
                f"Searched {lit_stats['count']} chunks across {indexed_papers} papers "
                f"with scope='{search_scope}'. No results found. "
                "Try broadening your query or changing the search_scope."
            ),
        }

    # Step 4: Detect query type and apply section boost
    query_type = _detect_query_type(query)
    boosted_scores = _apply_section_boost(
        result_distances, result_metadatas, query_type
    )

    # Step 5: Cross-encoder reranking
    reranker_used = False
    try:
        reranker = _get_reranker()

        # Build chunk dicts for the reranker
        rerank_chunks = []
        for i, (chunk_id, doc, meta, boosted) in enumerate(
            zip(result_ids, result_documents, result_metadatas, boosted_scores)
        ):
            rerank_chunks.append({
                "text": doc,
                "chunk_id": chunk_id,
                "metadata": meta,
                "vector_score": boosted,
                "original_index": i,
            })

        # Rerank using cross-encoder
        reranked = reranker.rerank(query, rerank_chunks, top_k=max_results)
        reranker_used = True

        # Build top_results from reranked output
        # Use rerank_score as primary, but normalize to 0-1 range
        top_results = []
        for chunk in reranked:
            # Normalize rerank_score: cross-encoder scores can be any real number
            # Use sigmoid-like normalization to [0, 1]
            raw_score = chunk.get("rerank_score", 0.0)
            # ms-marco-MiniLM-L-6-v2 typically outputs scores in [-10, 10] range
            import math
            normalized_score = 1.0 / (1.0 + math.exp(-raw_score))

            if normalized_score >= min_relevance:
                top_results.append((
                    chunk["chunk_id"],
                    chunk["text"],
                    chunk["metadata"],
                    round(normalized_score, 4),
                ))

    except Exception as exc:
        # Fallback: use vector similarity + section boost (no reranker)
        logger.warning("Reranker unavailable, falling back to vector similarity: %s", exc)

        scored_results = list(zip(
            result_ids, result_documents, result_metadatas, boosted_scores
        ))
        scored_results.sort(key=lambda x: x[3], reverse=True)

        top_results = [
            r for r in scored_results[:max_results]
            if r[3] >= min_relevance
        ]

    # Step 6 & 7: Assemble context and citations
    output_results = []
    papers_seen = set()

    for chunk_id, chunk_text, metadata, score in top_results:
        pmid = metadata.get("paper_pmid", "")
        section = metadata.get("section", "")
        para_num = metadata.get("paragraph_num", 0)

        papers_seen.add(pmid)

        # Get paper metadata from SQLite
        paper = await sqlite.get_paper(pmid)

        # Get surrounding context
        context = await sqlite.get_surrounding_chunks(pmid, section, para_num)

        # Build citation
        citation = {
            "pmid": pmid,
            "pmc_id": metadata.get("pmc_id", ""),
            "paper_title": paper.get("title", "") if paper else "",
            "authors": paper.get("authors", []) if paper else [],
            "journal": paper.get("journal", "") if paper else "",
            "year": paper.get("year", 0) if paper else 0,
            "section": section,
            "subsection": None,
            "paragraph_index": para_num,
        }

        output_results.append({
            "chunk_text": chunk_text,
            "citation": citation,
            "relevance_score": round(score, 4),
            "surrounding_context": {
                "previous_paragraph": context.get("previous_paragraph"),
                "next_paragraph": context.get("next_paragraph"),
            },
        })

    # Compute stats
    indexed_papers = await sqlite.get_indexed_paper_count()
    elapsed_ms = int((time.time() - start_time) * 1000)

    # Build strategy description
    scope_desc = f"scope='{search_scope}'" if search_scope != "all" else "all sections"
    boost_desc = f"query_type='{query_type}'" if query_type != "general" else "no section boost"
    rerank_desc = "cross-encoder reranking" if reranker_used else "vector similarity only (reranker unavailable)"
    strategy = (
        f"Embedded query and searched {lit_stats['count']} chunks across "
        f"{indexed_papers} indexed papers ({scope_desc}). "
        f"Retrieved top {settings.VECTOR_TOP_K} by cosine similarity, "
        f"applied section boost ({boost_desc}), "
        f"then applied {rerank_desc}, "
        f"returned top {len(output_results)} results above {min_relevance} threshold. "
        f"Completed in {elapsed_ms}ms."
    )

    # Log the search
    try:
        await sqlite.log_search(
            tool_name="deep_search",
            query=query,
            result_count=len(output_results),
            latency_ms=elapsed_ms,
            params={"search_scope": search_scope, "max_results": max_results},
        )
    except Exception as exc:
        logger.warning("Failed to log search: %s", exc)

    return {
        "results": output_results,
        "total_indexed_papers": indexed_papers,
        "total_indexed_chunks": lit_stats["count"],
        "search_strategy": strategy,
    }
