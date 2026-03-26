"""search_literature tool — Discover relevant papers across PubMed with synonym expansion.

Returns ranked papers with abstracts and metadata.

Search strategy (zero-result prevention):
  Phase A — Tiered query funnel (3 tiers, most→least precise):
    Tier 1: Full expanded query with MeSH / field tags
    Tier 2: Core concepts only, all tagged [Title/Abstract]
    Tier 3: Single dominant entity [MeSH Terms]
  Phase B — Result-aware relaxation (only if all tiers still empty):
    Steps 0-4: progressively loosen field tags, drop clauses, first clause only
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime

from oncocontext.config import settings
from oncocontext.models.schemas import PaperResult, QueryExpansion, SearchResult
from oncocontext.services.query_expander import QueryExpander, RELAX_STEPS
from oncocontext.services.pubmed_client import PubMedClient
from oncocontext.storage.cache_manager import CacheManager

logger = logging.getLogger(__name__)

# Minimum results to consider a tier "successful"
_MIN_RESULTS_THRESHOLD = 1

# ── Module-level lazy singletons ──────────────────────────────────────────────

_cache: CacheManager | None = None
_expander: QueryExpander | None = None
_pubmed: PubMedClient | None = None


def _get_cache() -> CacheManager:
    global _cache
    if _cache is None:
        _cache = CacheManager()
    return _cache


def _get_expander() -> QueryExpander:
    global _expander
    if _expander is None:
        _expander = QueryExpander()
    return _expander


def _get_pubmed() -> PubMedClient:
    global _pubmed
    if _pubmed is None:
        _pubmed = PubMedClient(cache=_get_cache())
    return _pubmed


# ── Relevance Scoring (Phase 1 — keyword-based) ──────────────────────────────


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase word tokens."""
    return set(re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower()))


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _term_frequency(query_terms: set[str], text: str) -> float:
    """Compute fraction of query terms found in text."""
    if not query_terms or not text:
        return 0.0
    text_lower = text.lower()
    found = sum(1 for term in query_terms if term in text_lower)
    return found / len(query_terms)


def _recency_boost(year: int) -> float:
    """Compute recency boost score. 1.0 for current year, decaying."""
    if year <= 0:
        return 0.0
    current_year = datetime.now().year
    age = current_year - year
    if age <= 0:
        return 1.0
    if age >= 20:
        return 0.1
    # Exponential decay: half-life of ~5 years
    return math.exp(-0.1 * age)


def _score_paper(paper: dict, query_terms: set[str]) -> float:
    """Simple keyword-based relevance scoring for Phase 1.

    Weights:
        - Title keyword overlap (Jaccard): 0.30
        - Abstract term frequency: 0.40
        - MeSH term overlap: 0.15
        - Recency boost: 0.15
    """
    title_tokens = _tokenize(paper.get("title", ""))
    title_score = _jaccard_similarity(query_terms, title_tokens) * 0.30

    abstract_score = _term_frequency(query_terms, paper.get("abstract", "")) * 0.40

    mesh_tokens = set()
    for term in paper.get("mesh_terms", []):
        mesh_tokens.update(_tokenize(term))
    mesh_score = _jaccard_similarity(query_terms, mesh_tokens) * 0.15

    recency_score = _recency_boost(paper.get("year", 0)) * 0.15

    return title_score + abstract_score + mesh_score + recency_score


# ── Tiered / Relaxation Search Helpers ───────────────────────────────────────


async def _search_with_tiers(
    pubmed: PubMedClient,
    expander: QueryExpander,
    query: str,
    max_results: int,
    date_range: str | None,
    full_text_only: bool,
) -> tuple[list[str], int, str, int]:
    """Run tiered query funnel then relaxation loop until results are found.

    Returns:
        (pmids, total_count, winning_pubmed_query, tier_used)
        tier_used: 1-3 for tier hits, 10+ for relaxation steps (10 = step 0, etc.)
    """
    # ── Phase A: 3-tier funnel ────────────────────────────────────────────────
    tiers = expander.build_query_tiers(query)
    for tier_num, tier_query in enumerate(tiers, 1):
        try:
            result = await pubmed.search(
                query=tier_query,
                max_results=max_results,
                date_range=date_range,
                full_text_only=full_text_only,
            )
            pmids = result.get("pmids", [])
            total = result.get("total_count", 0)
            logger.info(
                "Tier %d query → %d results | query: %s",
                tier_num, total, tier_query[:80],
            )
            if len(pmids) >= _MIN_RESULTS_THRESHOLD:
                return pmids, total, tier_query, tier_num
        except Exception as exc:
            logger.warning("Tier %d search failed: %s", tier_num, exc)

    # ── Phase B: result-aware relaxation on Tier-1 query ─────────────────────
    # IMPORTANT: full_text_only (pmc[filter]) is intentionally NOT passed during
    # relaxation.  That filter appends "AND pmc[filter]" to every query, which
    # kills all results when even the single-term fallback returns 0 with it.
    tier1_query = tiers[0]
    for step_idx, relax_fn in enumerate(RELAX_STEPS):
        relaxed_query = relax_fn(tier1_query)
        if not relaxed_query or relaxed_query == tier1_query and step_idx > 0:
            continue
        try:
            result = await pubmed.search(
                query=relaxed_query,
                max_results=max_results,
                date_range=date_range,
                full_text_only=False,  # F4: drop pmc[filter] during relaxation
            )
            pmids = result.get("pmids", [])
            total = result.get("total_count", 0)
            logger.info(
                "Relaxation step %d → %d results | query: %s",
                step_idx, total, relaxed_query[:80],
            )
            if len(pmids) >= _MIN_RESULTS_THRESHOLD:
                return pmids, total, relaxed_query, 10 + step_idx
        except Exception as exc:
            logger.warning("Relaxation step %d search failed: %s", step_idx, exc)

    return [], 0, tiers[0], -1


# ── Main Tool Function ────────────────────────────────────────────────────────


async def search_literature(
    query: str,
    max_results: int = 20,
    date_range: str | None = None,
    full_text_only: bool = False,
) -> dict:
    """Search PubMed for oncology papers with synonym expansion and relevance scoring.

    Algorithm:
        1. Expand query via synonym_dict.json (PD-1 → PDCD1, CD279, etc.)
        2. Build tiered query funnel (3 tiers: precise → relaxed → broad)
        3. Execute tiers in order; stop at first tier returning ≥1 result
        4. If all tiers fail, apply result-aware relaxation steps on Tier-1 query
        5. PubMed esearch → PMIDs → efetch → metadata XML
        6. Compute relevance score (keyword-based)
        7. Cache results (24h TTL)

    Args:
        query: Natural language research question or structured search query.
        max_results: Maximum number of papers to return (1-100).
        date_range: Filter by publication date range, e.g. '2020-2025'.
        full_text_only: Only return papers with PMC full text available.

    Returns:
        Dict with papers, total_found, and query_expansion details.
    """
    expander = _get_expander()
    pubmed = _get_pubmed()

    # Clamp max_results
    max_results = max(1, min(max_results, settings.MAX_PAPERS_PER_SEARCH))

    # Step 1: Expand query (for metadata / expanded_terms tracking)
    try:
        expansion_data = await expander.expand(query)
        query_expansion = QueryExpansion(**expansion_data)
    except Exception as exc:
        logger.error("Query expansion failed: %s — using raw query", exc)
        query_expansion = QueryExpansion(
            original_query=query,
            expanded_terms=[],
            pubmed_query=query,
        )

    # Step 2: Tiered search with fallback
    try:
        pmids, total_count, winning_query, tier_used = await _search_with_tiers(
            pubmed=pubmed,
            expander=expander,
            query=query,
            max_results=max_results,
            date_range=date_range,
            full_text_only=full_text_only,
        )
        # Update expansion's pubmed_query to reflect the winning query
        query_expansion = QueryExpansion(
            original_query=query_expansion.original_query,
            expanded_terms=query_expansion.expanded_terms,
            pubmed_query=winning_query,
        )
        if tier_used > 0:
            label = f"tier-{tier_used}" if tier_used <= 3 else f"relaxation-step-{tier_used - 10}"
            logger.info("Search succeeded via %s (%d PMIDs)", label, len(pmids))
    except Exception as exc:
        logger.error("Tiered PubMed search failed: %s", exc)
        return SearchResult(
            papers=[],
            total_found=0,
            query_expansion=query_expansion,
        ).model_dump()

    if not pmids:
        logger.warning("All search tiers returned zero results for: %s", query[:80])
        return SearchResult(
            papers=[],
            total_found=total_count,
            query_expansion=query_expansion,
        ).model_dump()

    # Step 3: Fetch paper details
    try:
        papers_data = await pubmed.fetch_details(pmids)
    except Exception as exc:
        logger.error("PubMed fetch details failed: %s", exc)
        return SearchResult(
            papers=[],
            total_found=total_count,
            query_expansion=query_expansion,
        ).model_dump()

    # Step 4: Score and rank papers
    query_terms = _tokenize(query)
    # Also add expanded terms to the scoring vocabulary
    for term in query_expansion.expanded_terms:
        query_terms.update(_tokenize(term))

    scored_papers: list[tuple[float, dict]] = []
    for paper in papers_data:
        score = _score_paper(paper, query_terms)
        scored_papers.append((score, paper))

    # Sort by score descending
    scored_papers.sort(key=lambda x: x[0], reverse=True)

    # Step 5: Build result
    paper_results: list[PaperResult] = []
    for score, paper in scored_papers[:max_results]:
        if full_text_only and not paper.get("has_full_text", False):
            continue

        paper_results.append(PaperResult(
            pmid=paper.get("pmid", ""),
            title=paper.get("title", ""),
            authors=paper.get("authors", []),
            journal=paper.get("journal", ""),
            year=paper.get("year", 0),
            abstract=paper.get("abstract", ""),
            has_full_text=paper.get("has_full_text", False),
            pmc_id=paper.get("pmc_id"),
            relevance_score=round(score, 4),
            mesh_terms=paper.get("mesh_terms", []),
            is_indexed=False,  # Indexing comes in Phase 2
        ))

    result = SearchResult(
        papers=paper_results,
        total_found=total_count,
        query_expansion=query_expansion,
    )

    logger.info(
        "search_literature returned %d papers (total found: %d) for: %s",
        len(paper_results), total_count, query[:60],
    )

    return result.model_dump()
