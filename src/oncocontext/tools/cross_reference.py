"""cross_reference tool — THE KILLER FEATURE.

Compare lab data against indexed literature, returning structured
agreements, contradictions, possible explanations, and follow-ups
with paragraph-level citations.

Uses a section-based heuristic for MVP categorization:
    - Results chunks → AGREEMENT or CONTRADICTION
    - Discussion chunks → POSSIBLE EXPLANATION
    - Methods chunks → SUGGESTED FOLLOW-UP
The LLM (Claude) interprets these results more intelligently.
"""

from __future__ import annotations

import logging
import math
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


# ── Helper Functions ──────────────────────────────────────────────────────────


def _normalize_score(raw_score: float) -> float:
    """Normalize cross-encoder score to [0, 1] using sigmoid."""
    return 1.0 / (1.0 + math.exp(-raw_score))


def _score_to_confidence(score: float) -> str:
    """Convert a normalized score to confidence level."""
    if score >= 0.75:
        return "strong"
    elif score >= 0.55:
        return "moderate"
    else:
        return "weak"


def _summarize_lab_data(lab_chunks: list[dict]) -> str:
    """Build a text summary from lab data chunks.

    Args:
        lab_chunks: List of lab data chunk dicts with 'text' and 'metadata'.

    Returns:
        Concatenated summary string.
    """
    if not lab_chunks:
        return "No lab data available."

    summaries = []
    for chunk in lab_chunks:
        chunk_type = chunk.get("metadata", {}).get("chunk_type", "row_data")
        text = chunk.get("text", "")
        if chunk_type == "file_summary":
            summaries.insert(0, text)  # Put file summary first
        else:
            summaries.append(text)

    return " | ".join(summaries[:10])  # Limit to avoid too-long queries


def _extract_key_observations(lab_chunks: list[dict]) -> list[str]:
    """Extract key observations from lab data chunks.

    Args:
        lab_chunks: List of lab data chunk dicts.

    Returns:
        List of observation strings.
    """
    observations = []
    for chunk in lab_chunks:
        text = chunk.get("text", "")
        if text and chunk.get("metadata", {}).get("chunk_type") != "file_summary":
            # Truncate long row data
            if len(text) > 200:
                text = text[:200] + "..."
            observations.append(text)

    return observations[:5]  # Top 5 observations


def _generate_comparison_queries(
    research_question: str,
    lab_summary: str,
    comparison_type: str,
) -> list[str]:
    """Generate focused search queries for literature comparison.

    Args:
        research_question: The user's research question.
        lab_summary: Summary of relevant lab data.
        comparison_type: 'findings', 'methods', or 'both'.

    Returns:
        List of 2-3 focused query strings.
    """
    queries = [research_question]

    # Extract key terms from the lab summary for a focused query
    # Simple keyword extraction: look for marker names and measurements
    import re
    marker_pattern = re.compile(
        r'\b(CD\d+|PD-?1|TIM-?3|LAG-?3|CTLA-?4|Ki-?67|Granzyme\s*B|'
        r'Cytotoxicity|MFI|percent|IFN|TNF|IL-\d+|TCF-?1|TOX)\b',
        re.IGNORECASE
    )
    lab_markers = set(m.group() for m in marker_pattern.finditer(lab_summary))

    if lab_markers:
        marker_query = " ".join(sorted(lab_markers)[:6])
        queries.append(f"{marker_query} T cell function exhaustion")

    if comparison_type in ("methods", "both"):
        queries.append(f"methods protocol {research_question}")

    return queries[:3]


def _categorize_chunk(
    chunk: dict,
    lab_summary: str,
    research_question: str,
) -> dict | None:
    """Categorize a literature chunk relative to lab data.

    MVP heuristic based on section type:
        - results → AGREEMENT or CONTRADICTION
        - discussion → EXPLANATION
        - methods → FOLLOW-UP
        - abstract/introduction → AGREEMENT (background context)

    Args:
        chunk: Literature chunk dict with text, metadata, rerank_score.
        lab_summary: Lab data summary text.
        research_question: The user's research question.

    Returns:
        Dict with category and structured finding, or None if not relevant.
    """
    text = chunk.get("text", "")
    metadata = chunk.get("metadata", {})
    section = metadata.get("section", "").lower()
    raw_score = chunk.get("rerank_score", 0.0)
    confidence_score = _normalize_score(raw_score)
    confidence = _score_to_confidence(confidence_score)

    # Build citation info
    citation = {
        "pmid": metadata.get("paper_pmid", ""),
        "pmc_id": metadata.get("pmc_id", ""),
        "paper_title": "",  # Will be enriched later
        "authors": [],
        "journal": "",
        "year": 0,
        "section": section,
        "subsection": None,
        "paragraph_index": metadata.get("paragraph_num", 0),
    }

    # Section-based categorization
    if section in ("results", "abstract"):
        # Check if the text discusses findings that could agree or contradict
        # For MVP: results chunks that match the query → agreement
        # We use a simple keyword heuristic
        question_lower = research_question.lower()

        # Contradiction signals: words that suggest opposing findings
        contradiction_signals = [
            "reduced", "decreased", "loss", "lose", "lost", "impaired",
            "dysfunctional", "unable", "fail", "failed", "decline",
            "inversely", "negatively", "however", "contrary", "unlike",
        ]

        has_contradiction_signal = any(s in text.lower() for s in contradiction_signals)

        # Check if the question implies an expected contradiction
        question_implies_contradiction = any(
            word in question_lower
            for word in ["why", "explain", "paradox", "contradiction", "discrepancy",
                        "but", "despite", "however", "unexpected"]
        )

        if has_contradiction_signal and question_implies_contradiction:
            return {
                "category": "contradiction",
                "text": text,
                "citation": citation,
                "confidence": confidence,
                "confidence_score": confidence_score,
            }
        else:
            return {
                "category": "agreement",
                "text": text,
                "citation": citation,
                "confidence": confidence,
                "confidence_score": confidence_score,
            }

    elif section == "discussion":
        return {
            "category": "explanation",
            "text": text,
            "citation": citation,
            "confidence": confidence,
            "confidence_score": confidence_score,
        }

    elif section in ("methods", "materials"):
        return {
            "category": "follow_up",
            "text": text,
            "citation": citation,
            "confidence": confidence,
            "confidence_score": confidence_score,
        }

    elif section == "introduction":
        return {
            "category": "agreement",
            "text": text,
            "citation": citation,
            "confidence": confidence,
            "confidence_score": confidence_score,
        }

    else:
        # Unknown section — categorize as agreement (general context)
        return {
            "category": "agreement",
            "text": text,
            "citation": citation,
            "confidence": confidence,
            "confidence_score": confidence_score,
        }


# ── Main Tool Function ────────────────────────────────────────────────────────


async def cross_reference(
    research_question: str,
    lab_file_ids: list[str] | None = None,
    paper_ids: list[str] | None = None,
    comparison_type: str = "findings",
) -> dict:
    """Compare lab data in ChromaDB against indexed literature.

    Algorithm:
        1. Retrieve lab data from ChromaDB lab_data collection
        2. Summarize lab findings focused on research_question
        3. Generate 2-3 focused comparison queries
        4. For each query: embed → search ChromaDB literature_chunks (top 50)
        5. Deduplicate → cross-encoder reranking → top 20
        6. Categorize findings: AGREE / CONTRADICT / EXPLAIN / FOLLOW-UP
        7. Enrich citations with paper metadata from SQLite
        8. Assemble structured response

    Args:
        research_question: The specific comparison question.
        lab_file_ids: IDs of ingested lab files to include. None = all.
        paper_ids: Restrict literature comparison to specific PMIDs. None = all.
        comparison_type: Compare 'findings', 'methods', or 'both'.

    Returns:
        Dict with summary, agreements, contradictions, novel findings, and follow-ups.
    """
    start_time = time.time()

    embedder = _get_embedder()
    chroma = _get_chroma()
    sqlite = _get_sqlite()
    await sqlite.init_db()

    # ── Step 1: Retrieve lab data ──────────────────────────────────────────

    lab_stats = chroma.get_collection_stats("lab")
    lit_stats = chroma.get_collection_stats("literature")

    # Check if literature is indexed
    if lit_stats["count"] == 0:
        return {
            "error": "No literature papers indexed. Use get_paper_details to index papers first.",
            "summary": "",
            "lab_data_summary": "",
            "agreements": [],
            "contradictions": [],
            "novel_findings": [],
            "suggested_follow_up": [],
            "papers_consulted": 0,
            "chunks_analyzed": 0,
        }

    # Find relevant lab data
    lab_chunks: list[dict] = []
    has_lab_data = lab_stats["count"] > 0

    if has_lab_data:
        try:
            lab_query_embedding = embedder.embed_text(research_question)

            lab_where: dict | None = None
            if lab_file_ids:
                if len(lab_file_ids) == 1:
                    lab_where = {"file_id": lab_file_ids[0]}
                else:
                    lab_where = {"file_id": {"$in": lab_file_ids}}

            lab_results = chroma.search(
                query_embedding=lab_query_embedding,
                collection="lab",
                n_results=10,
                where=lab_where,
            )

            # Unpack lab results
            lab_ids = lab_results.get("ids", [[]])[0]
            lab_docs = lab_results.get("documents", [[]])[0]
            lab_metas = lab_results.get("metadatas", [[]])[0]

            for doc, meta in zip(lab_docs, lab_metas):
                lab_chunks.append({"text": doc, "metadata": meta})

        except Exception as exc:
            logger.warning("Failed to retrieve lab data: %s", exc)
    else:
        # No lab data — still useful for literature-only cross-reference
        logger.info("No lab data ingested. Performing literature-only analysis.")

    # ── Step 2: Summarize lab findings ─────────────────────────────────────

    lab_summary = _summarize_lab_data(lab_chunks)
    key_observations = _extract_key_observations(lab_chunks)

    # ── Step 3: Generate comparison queries ────────────────────────────────

    queries = _generate_comparison_queries(
        research_question, lab_summary, comparison_type
    )

    # ── Step 4: Search literature ──────────────────────────────────────────

    all_literature_chunks: dict[str, dict] = {}  # chunk_id → chunk dict
    total_chunks_searched = 0

    for query_text in queries:
        try:
            query_emb = embedder.embed_text(query_text)

            lit_where: dict[str, Any] | None = None
            if paper_ids:
                lit_where = {"paper_pmid": {"$in": paper_ids}}

            # Apply section focus based on comparison_type
            if comparison_type == "methods":
                section_filter = {"section": {"$in": ["methods", "materials"]}}
                if lit_where:
                    lit_where = {"$and": [lit_where, section_filter]}
                else:
                    lit_where = section_filter

            lit_results = chroma.search(
                query_embedding=query_emb,
                collection="literature",
                n_results=settings.VECTOR_TOP_K,
                where=lit_where,
            )

            # Unpack and deduplicate
            lit_ids = lit_results.get("ids", [[]])[0]
            lit_docs = lit_results.get("documents", [[]])[0]
            lit_metas = lit_results.get("metadatas", [[]])[0]
            lit_dists = lit_results.get("distances", [[]])[0]

            for chunk_id, doc, meta, dist in zip(lit_ids, lit_docs, lit_metas, lit_dists):
                if chunk_id not in all_literature_chunks:
                    all_literature_chunks[chunk_id] = {
                        "chunk_id": chunk_id,
                        "text": doc,
                        "metadata": meta,
                        "vector_score": max(0.0, 1.0 - dist),
                    }

            total_chunks_searched += len(lit_ids)

        except Exception as exc:
            logger.warning("Literature search failed for query '%s': %s", query_text, exc)

    if not all_literature_chunks:
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "summary": "No matching literature found for your research question. "
            "Consider indexing more papers or broadening your question.",
            "lab_data_summary": lab_summary,
            "agreements": [],
            "contradictions": [],
            "novel_findings": key_observations if key_observations else [],
            "suggested_follow_up": [
                "Index additional papers using get_paper_details",
                "Try a broader research question",
            ],
            "papers_consulted": 0,
            "chunks_analyzed": 0,
        }

    # ── Step 5: Rerank literature chunks ───────────────────────────────────

    unique_chunks = list(all_literature_chunks.values())
    reranker_used = False
    max_cross_ref = 20

    try:
        reranker = _get_reranker()
        reranked = reranker.rerank(research_question, unique_chunks, top_k=max_cross_ref)
        reranker_used = True
    except Exception as exc:
        logger.warning("Reranker unavailable for cross-reference: %s", exc)
        # Fallback: sort by vector score
        reranked = sorted(unique_chunks, key=lambda c: c.get("vector_score", 0), reverse=True)[:max_cross_ref]
        # Add a synthetic rerank_score equal to vector_score
        for chunk in reranked:
            chunk["rerank_score"] = chunk.get("vector_score", 0.0)

    # ── Step 6: Categorize findings ────────────────────────────────────────

    agreements = []
    contradictions = []
    explanations = []
    follow_ups = []
    papers_seen = set()

    for chunk in reranked:
        categorized = _categorize_chunk(chunk, lab_summary, research_question)
        if categorized is None:
            continue

        pmid = categorized["citation"].get("pmid", "")
        if pmid:
            papers_seen.add(pmid)

        category = categorized["category"]
        text = categorized["text"]
        citation = categorized["citation"]
        confidence = categorized["confidence"]
        confidence_score = categorized["confidence_score"]

        if category == "agreement":
            agreements.append({
                "lab_finding": lab_summary[:200] if lab_summary else research_question,
                "literature_support": text,
                "citation": citation,
                "confidence": confidence,
            })
        elif category == "contradiction":
            contradictions.append({
                "lab_finding": lab_summary[:200] if lab_summary else research_question,
                "literature_contradiction": text,
                "citation": citation,
                "possible_explanations": [],
            })
        elif category == "explanation":
            explanations.append({
                "text": text,
                "citation": citation,
                "confidence": confidence,
                "confidence_score": confidence_score,
            })
        elif category == "follow_up":
            follow_ups.append({
                "text": text,
                "citation": citation,
            })

    # ── Step 7: Enrich citations with paper metadata ──────────────────────

    paper_cache: dict[str, dict] = {}

    async def _enrich_citation(citation: dict) -> dict:
        """Add paper metadata to a citation dict."""
        pmid = citation.get("pmid", "")
        if not pmid:
            return citation

        if pmid not in paper_cache:
            paper = await sqlite.get_paper(pmid)
            paper_cache[pmid] = paper or {}

        paper = paper_cache[pmid]
        if paper:
            citation["paper_title"] = paper.get("title", "")
            citation["authors"] = paper.get("authors", [])
            citation["journal"] = paper.get("journal", "")
            citation["year"] = paper.get("year", 0)

        return citation

    # Enrich all citations
    for a in agreements:
        a["citation"] = await _enrich_citation(a["citation"])
    for c in contradictions:
        c["citation"] = await _enrich_citation(c["citation"])
    for e in explanations:
        e["citation"] = await _enrich_citation(e["citation"])
    for f in follow_ups:
        f["citation"] = await _enrich_citation(f["citation"])

    # ── Step 7b: Add explanations to contradictions ───────────────────────

    # For each contradiction, add relevant explanations
    for contradiction in contradictions:
        for exp in explanations[:3]:
            contradiction["possible_explanations"].append(exp["text"][:300])

    # ── Step 8: Build novel findings ──────────────────────────────────────

    novel_findings = []
    if key_observations and not agreements and not contradictions:
        for obs in key_observations:
            novel_findings.append(f"Lab observation with no clear match in indexed literature: {obs}")
    elif key_observations:
        # Check if any observations aren't covered
        covered_texts = set()
        for a in agreements:
            covered_texts.add(a["lab_finding"][:50])
        for c in contradictions:
            covered_texts.add(c["lab_finding"][:50])

        for obs in key_observations:
            if obs[:50] not in covered_texts:
                novel_findings.append(obs)

    # ── Step 9: Build suggested follow-ups ────────────────────────────────

    suggested_follow_up = []
    for f in follow_ups[:3]:
        text = f["text"]
        # Extract a suggestion from the methods text
        if len(text) > 300:
            text = text[:300] + "..."
        pmid = f["citation"].get("pmid", "")
        suggested_follow_up.append(
            f"Consider methodology from PMID {pmid}: {text}"
        )

    # Add generic follow-ups if we found contradictions
    if contradictions:
        suggested_follow_up.append(
            "Investigate potential explanations for observed contradictions "
            "with published data — consider additional marker staining or "
            "sub-population analysis."
        )

    if not suggested_follow_up:
        suggested_follow_up.append(
            "Index additional papers to strengthen the cross-reference analysis."
        )

    # ── Step 10: Build summary ─────────────────────────────────────────────

    elapsed_ms = int((time.time() - start_time) * 1000)

    n_agreements = len(agreements)
    n_contradictions = len(contradictions)
    n_explanations = len(explanations)
    n_papers = len(papers_seen)

    summary_parts = [
        f"Cross-referenced {'lab data' if has_lab_data else 'research question'} "
        f"against {n_papers} papers ({len(reranked)} chunks analyzed)."
    ]

    if n_agreements:
        summary_parts.append(f"Found {n_agreements} agreement(s) with published findings.")
    if n_contradictions:
        summary_parts.append(f"Found {n_contradictions} potential contradiction(s).")
    if n_explanations:
        summary_parts.append(f"Found {n_explanations} possible explanation(s).")
    if novel_findings:
        summary_parts.append(f"Identified {len(novel_findings)} novel/unmatched observation(s).")

    summary_parts.append(f"Analysis completed in {elapsed_ms}ms.")
    if not reranker_used:
        summary_parts.append("(Cross-encoder reranking unavailable; used vector similarity.)")

    summary = " ".join(summary_parts)

    # Log the search
    try:
        await sqlite.log_search(
            tool_name="cross_reference",
            query=research_question,
            result_count=n_agreements + n_contradictions,
            latency_ms=elapsed_ms,
            params={"comparison_type": comparison_type, "lab_file_ids": lab_file_ids},
        )
    except Exception:
        pass

    return {
        "summary": summary,
        "lab_data_summary": lab_summary,
        "agreements": agreements,
        "contradictions": contradictions,
        "novel_findings": novel_findings,
        "suggested_follow_up": suggested_follow_up,
        "papers_consulted": n_papers,
        "chunks_analyzed": len(reranked),
    }
