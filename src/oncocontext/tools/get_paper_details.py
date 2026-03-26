"""get_paper_details tool — Fetch full metadata and optionally full text + indexing.

Fetches paper details from PubMed, optionally triggers PMC full-text fetch,
parsing, chunking, embedding, and ChromaDB indexing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from oncocontext.models.schemas import PaperDetails, Section
from oncocontext.services.pubmed_client import PubMedClient
from oncocontext.services.pmc_client import PMCClient
from oncocontext.services.bioc_parser import BioCParser
from oncocontext.services.chunker import SectionAwareChunker
from oncocontext.services.embedder import Embedder
from oncocontext.storage.cache_manager import CacheManager
from oncocontext.storage.chroma_manager import ChromaManager
from oncocontext.storage.sqlite_manager import SQLiteManager

logger = logging.getLogger(__name__)

# ── Module-level lazy singletons ──────────────────────────────────────────────

_cache: CacheManager | None = None
_pubmed: PubMedClient | None = None
_pmc: PMCClient | None = None
_parser: BioCParser | None = None
_chunker: SectionAwareChunker | None = None
_embedder: Embedder | None = None
_chroma: ChromaManager | None = None
_sqlite: SQLiteManager | None = None


def _get_cache() -> CacheManager:
    global _cache
    if _cache is None:
        _cache = CacheManager()
    return _cache


def _get_pubmed() -> PubMedClient:
    global _pubmed
    if _pubmed is None:
        _pubmed = PubMedClient(cache=_get_cache())
    return _pubmed


def _get_pmc() -> PMCClient:
    global _pmc
    if _pmc is None:
        _pmc = PMCClient(cache=_get_cache())
    return _pmc


def _get_parser() -> BioCParser:
    global _parser
    if _parser is None:
        _parser = BioCParser()
    return _parser


def _get_chunker() -> SectionAwareChunker:
    global _chunker
    if _chunker is None:
        _chunker = SectionAwareChunker()
    return _chunker


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


# ── Main Tool Function ────────────────────────────────────────────────────────


async def get_paper_details(
    pmid: str,
    fetch_full_text: bool = True,
    sections: list[str] | None = None,
    index_if_available: bool = True,
) -> dict:
    """Fetch full metadata and abstract for a specific PMID, optionally fetch full text.

    Algorithm:
        1. Fetch metadata from PubMed efetch API
        2. If fetch_full_text=True and PMC ID exists: fetch BioC JSON, parse sections
        3. If index_if_available=True and sections are available:
           a. Check if already indexed
           b. If not: chunk → embed → store in ChromaDB + SQLite
        4. Return paper details with index status

    Args:
        pmid: PubMed ID of the paper.
        fetch_full_text: Whether to fetch full text from PMC.
        sections: Filter to specific sections (abstract, introduction, methods, etc.).
        index_if_available: Whether to index the paper into ChromaDB if full text is available.

    Returns:
        Dict with paper metadata, sections (if available), and index status.
    """
    pubmed = _get_pubmed()
    pmc = _get_pmc()
    parser = _get_parser()
    sqlite = _get_sqlite()

    # Ensure SQLite is initialized
    await sqlite.init_db()

    # Step 1: Fetch PubMed metadata
    try:
        papers = await pubmed.fetch_details([pmid])
    except Exception as exc:
        logger.error("Failed to fetch PubMed details for %s: %s", pmid, exc)
        return PaperDetails(pmid=pmid).model_dump()

    if not papers:
        logger.warning("No data returned from PubMed for PMID %s", pmid)
        return PaperDetails(pmid=pmid).model_dump()

    paper = papers[0]

    # Build base details
    details = PaperDetails(
        pmid=paper.get("pmid", pmid),
        pmc_id=paper.get("pmc_id"),
        title=paper.get("title", ""),
        authors=paper.get("authors", []),
        journal=paper.get("journal", ""),
        year=paper.get("year", 0),
        abstract=paper.get("abstract", ""),
        mesh_terms=paper.get("mesh_terms", []),
        has_full_text=paper.get("has_full_text", False),
        sections=None,
        indexed=False,
        chunk_count=0,
    )

    # Store paper metadata in SQLite
    try:
        await sqlite.add_paper({
            "pmid": details.pmid,
            "pmc_id": details.pmc_id,
            "title": details.title,
            "authors": details.authors,
            "journal": details.journal,
            "year": details.year,
            "abstract": details.abstract,
            "mesh_terms": details.mesh_terms,
            "has_full_text": details.has_full_text,
        })
    except Exception as exc:
        logger.warning("Failed to store paper %s in SQLite: %s", pmid, exc)

    # Check if already indexed
    already_indexed = False
    chroma = _get_chroma()
    try:
        already_indexed = chroma.has_paper(pmid)
    except Exception:
        pass

    if already_indexed:
        details.indexed = True
        # Get chunk count from SQLite
        paper_record = await sqlite.get_paper(pmid)
        if paper_record:
            details.chunk_count = paper_record.get("chunk_count", 0)
        logger.debug("Paper %s already indexed (%d chunks)", pmid, details.chunk_count)

    # Step 2: Fetch full text if requested and PMC ID available
    parsed_sections: list[Section] | None = None
    pmc_id = details.pmc_id

    if fetch_full_text and pmc_id:
        try:
            bioc_json = await pmc.fetch_bioc(pmc_id)
            if bioc_json:
                parsed_sections = parser.parse(bioc_json)

                # Filter to requested sections if specified
                if sections and parsed_sections:
                    sections_lower = [s.lower() for s in sections]
                    parsed_sections = [
                        s for s in parsed_sections
                        if s.section_type.lower() in sections_lower
                    ]

                details.sections = parsed_sections
                details.has_full_text = True
                logger.info(
                    "Fetched full text for %s (%s): %d sections",
                    pmid, pmc_id, len(parsed_sections) if parsed_sections else 0,
                )
            else:
                logger.info("Full text not available in PMC for %s (%s)", pmid, pmc_id)
                details.has_full_text = False

        except Exception as exc:
            logger.warning("Failed to fetch/parse PMC full text for %s: %s", pmc_id, exc)

    # Step 3: Index if available and requested
    if (
        index_if_available
        and not already_indexed
        and parsed_sections
        and len(parsed_sections) > 0
    ):
        try:
            chunk_count = await _index_paper(
                pmid=pmid,
                pmc_id=pmc_id,
                sections=parsed_sections,
                paper_meta=paper,
            )
            details.indexed = True
            details.chunk_count = chunk_count
            logger.info("Indexed %d chunks from PMID %s", chunk_count, pmid)
        except Exception as exc:
            logger.error("Failed to index paper %s: %s", pmid, exc)
            # Still return the paper details even if indexing failed

    return details.model_dump()


async def _index_paper(
    pmid: str,
    pmc_id: str | None,
    sections: list[Section],
    paper_meta: dict,
) -> int:
    """Index a paper's sections into ChromaDB and SQLite.

    Args:
        pmid: PubMed ID.
        pmc_id: PMC ID.
        sections: Parsed Section objects.
        paper_meta: Paper metadata dict.

    Returns:
        Number of chunks indexed.
    """
    chunker = _get_chunker()
    embedder = _get_embedder()
    chroma = _get_chroma()
    sqlite = _get_sqlite()

    # Step 1: Chunk the paper
    chunks = chunker.chunk_paper(pmid=pmid, pmc_id=pmc_id, sections=sections)

    if not chunks:
        logger.warning("No chunks generated for paper %s", pmid)
        return 0

    # Step 2: Embed all chunks
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_batch(texts)

    # Step 3: Store in ChromaDB
    chroma.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        collection="literature",
    )

    # Step 4: Store chunk records in SQLite
    sqlite_chunks = [
        {
            "chunk_id": c["chunk_id"],
            "source_type": "paper",
            "source_id": pmid,
            "section": c["section"],
            "paragraph_index": c["paragraph_num"],
            "token_count": c["token_count"],
            "text": c["text"],
            "chromadb_collection": "literature_chunks",
        }
        for c in chunks
    ]
    await sqlite.add_chunks(sqlite_chunks)

    # Step 5: Update paper index status
    await sqlite.update_paper_index_status(
        pmid=pmid,
        chunk_count=len(chunks),
        indexed_at=datetime.now(timezone.utc).isoformat(),
    )

    return len(chunks)
