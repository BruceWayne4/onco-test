"""Pre-index oncology papers for the OncoContext demo.

This script:
1. Searches PubMed for papers on T cell exhaustion, CAR-T, and related topics
2. Filters for papers with PMC full text available
3. Fetches full text via PMC BioC API
4. Chunks, embeds, and stores in local ChromaDB
5. Reports indexing statistics

Usage:
    python demo/pre_index_papers.py [--max-papers 30] [--topic "T cell exhaustion"]
"""

import asyncio
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oncocontext.services.pubmed_client import PubMedClient
from oncocontext.services.pmc_client import PMCClient
from oncocontext.services.bioc_parser import BioCParser
from oncocontext.services.chunker import SectionAwareChunker
from oncocontext.services.embedder import Embedder
from oncocontext.storage.chroma_manager import ChromaManager
from oncocontext.storage.sqlite_manager import SQLiteManager
from oncocontext.storage.cache_manager import CacheManager
from oncocontext.config import settings


# PubMed queries designed to find papers with rich Methods sections
# about T cell exhaustion, flow cytometry, and immunotherapy
DEMO_QUERIES = [
    # Core T cell exhaustion papers
    (
        '("T cell exhaustion" OR "T-cell exhaustion") AND (PD-1 OR PDCD1) '
        'AND (flow cytometry OR FACS) AND "free full text"[filter]'
    ),
    # CAR-T and exhaustion
    (
        '("CAR-T" OR "chimeric antigen receptor") AND (exhaustion OR dysfunction) '
        'AND (solid tumor OR solid tumour) AND "free full text"[filter]'
    ),
    # Progenitor exhausted T cells (key to the demo story)
    (
        '("progenitor exhausted" OR "TCF1" OR "TCF-1") AND ("CD8" OR "CD8+") '
        'AND (PD-1 OR exhaustion) AND "free full text"[filter]'
    ),
    # T cell exhaustion in tumor organoids
    (
        '("tumor organoid" OR "tumour organoid" OR "organoid co-culture") '
        'AND ("T cell" OR "T-cell") AND (exhaustion OR cytotoxicity) '
        'AND "free full text"[filter]'
    ),
    # Flow cytometry gating strategies for exhaustion markers
    (
        '("gating strategy" OR "flow cytometry panel") AND ("exhaustion" OR "checkpoint") '
        'AND (CD8 OR "T cell") AND "free full text"[filter]'
    ),
]


async def pre_index(max_papers: int = 30, queries: list[str] | None = None) -> dict:
    """Pre-index papers for the demo.

    Args:
        max_papers: Maximum number of papers to index.
        queries: Custom PubMed queries. Defaults to DEMO_QUERIES.

    Returns:
        Dict with indexing statistics.
    """
    queries = queries or DEMO_QUERIES

    # Initialize services
    cache = CacheManager(str(settings.CACHE_DIR))
    pubmed = PubMedClient(cache=cache)
    pmc = PMCClient(cache=cache)
    parser = BioCParser()
    chunker = SectionAwareChunker()
    embedder = Embedder()
    chroma = ChromaManager()
    sqlite = SQLiteManager()

    await sqlite.init_db()

    print(f"\n🔬 OncoContext Pre-Indexing Script")
    print(f"{'=' * 50}")
    print(f"Target: {max_papers} papers with PMC full text")
    print(f"Queries: {len(queries)}")
    print(f"ChromaDB: {settings.CHROMA_DIR}")
    print(f"SQLite: {settings.SQLITE_DB_PATH}")
    print()

    start_time = time.time()

    # Step 1: Search PubMed across all queries
    all_pmids: set[str] = set()
    per_query = max_papers // len(queries) + 10  # fetch more than needed per query

    for i, query in enumerate(queries):
        short_query = query[:80] + "..." if len(query) > 80 else query
        print(f"📚 Query {i + 1}/{len(queries)}: {short_query}")
        try:
            result = await pubmed.search(query, max_results=per_query)
            pmids = result.get("pmids", [])
            total = result.get("total_count", 0)
            print(f"   Found {len(pmids)} PMIDs (total in PubMed: {total})")
            all_pmids.update(pmids)
        except Exception as e:
            print(f"   ⚠️  Error: {e}")

    print(f"\n📊 Total unique PMIDs: {len(all_pmids)}")

    if not all_pmids:
        print("\n❌ No PMIDs found. Check your internet connection and try again.")
        await pubmed.close()
        await pmc.close()
        return {"indexed_count": 0, "total_chunks": 0}

    # Step 2: Fetch details and filter for PMC availability
    print(f"\n📥 Fetching paper details...")
    pmid_list = list(all_pmids)[: max_papers * 2]  # fetch more than needed

    try:
        details = await pubmed.fetch_details(pmid_list)
    except Exception as e:
        print(f"   ⚠️  Error fetching details: {e}")
        details = []

    # Filter for papers with PMC IDs
    pmc_papers = [d for d in details if d.get("pmc_id")]
    print(f"   {len(pmc_papers)} papers have PMC full text (out of {len(details)})")

    if not pmc_papers:
        print("\n❌ No papers with PMC full text found.")
        await pubmed.close()
        await pmc.close()
        return {"indexed_count": 0, "total_chunks": 0}

    # Step 3: Index papers
    indexed_count = 0
    total_chunks = 0
    skipped_already = 0
    skipped_error = 0

    for paper in pmc_papers[:max_papers]:
        pmid = paper["pmid"]
        pmc_id = paper["pmc_id"]
        title = paper.get("title", "Unknown")
        short_title = title[:60] + "..." if len(title) > 60 else title

        # Check if already indexed
        if chroma.has_paper(pmid):
            print(f"   ⏭️  Already indexed: {pmid} - {short_title}")
            indexed_count += 1  # count it
            skipped_already += 1
            continue

        print(f"\n📄 Indexing: {pmid} - {short_title}")

        try:
            # Fetch BioC JSON
            bioc_json = await pmc.fetch_bioc(pmc_id)
            if not bioc_json:
                print(f"   ⚠️  No BioC data for {pmc_id}")
                skipped_error += 1
                continue

            # Parse into sections
            sections = parser.parse(bioc_json)
            if not sections:
                print(f"   ⚠️  No parseable sections for {pmc_id}")
                skipped_error += 1
                continue

            section_names = [s.heading for s in sections]
            print(f"   📑 Sections: {', '.join(section_names)}")

            # Chunk
            chunks = chunker.chunk_paper(
                pmid=pmid, pmc_id=pmc_id, sections=sections
            )
            if not chunks:
                print(f"   ⚠️  No chunks generated")
                skipped_error += 1
                continue

            print(f"   ✂️  Generated {len(chunks)} chunks")

            # Embed
            texts = [c["text"] for c in chunks]
            embeddings = embedder.embed_batch(texts)
            print(f"   🧠 Embedded {len(embeddings)} chunks")

            # Store in ChromaDB
            chroma.add_chunks(chunks, embeddings, collection="literature")

            # Store paper metadata in SQLite
            await sqlite.add_paper(
                {
                    "pmid": pmid,
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", []),
                    "journal": paper.get("journal", ""),
                    "year": paper.get("year", 0),
                    "abstract": paper.get("abstract", ""),
                    "pmc_id": pmc_id,
                    "doi": paper.get("doi", ""),
                    "mesh_terms": paper.get("mesh_terms", []),
                    "chunk_count": len(chunks),
                    "has_full_text": True,
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                }
            )

            # Store chunk records in SQLite
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

            # Update paper index status
            await sqlite.update_paper_index_status(
                pmid=pmid,
                chunk_count=len(chunks),
                indexed_at=datetime.now(timezone.utc).isoformat(),
            )

            indexed_count += 1
            total_chunks += len(chunks)
            print(f"   ✅ Indexed! ({indexed_count}/{max_papers})")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            skipped_error += 1
            continue

        if indexed_count >= max_papers:
            break

    # Summary
    elapsed = time.time() - start_time
    stats = chroma.get_collection_stats("literature")
    paper_count = await sqlite.get_indexed_paper_count()

    print(f"\n{'=' * 50}")
    print(f"✅ Pre-indexing complete!")
    print(f"   Papers indexed this run: {indexed_count - skipped_already}")
    print(f"   Papers already indexed: {skipped_already}")
    print(f"   Papers skipped (errors): {skipped_error}")
    print(f"   Total papers in index: {paper_count}")
    print(f"   Total chunks in ChromaDB: {stats['count']}")
    print(f"   New chunks this run: {total_chunks}")
    print(f"   Time elapsed: {elapsed:.1f}s")

    # Cleanup
    await pubmed.close()
    await pmc.close()

    return {
        "indexed_count": indexed_count,
        "total_chunks": total_chunks,
        "skipped_already": skipped_already,
        "skipped_error": skipped_error,
        "total_in_index": stats["count"],
        "total_papers": paper_count,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Pre-index oncology papers for OncoContext demo"
    )
    arg_parser.add_argument(
        "--max-papers",
        type=int,
        default=30,
        help="Maximum papers to index (default: 30)",
    )
    args = arg_parser.parse_args()

    asyncio.run(pre_index(max_papers=args.max_papers))
