"""Verify that the OncoContext index is ready for the demo.

Checks:
1. ChromaDB has papers indexed
2. SQLite has paper metadata
3. Runs a test deep_search query
4. Runs a test search_literature query
5. Reports overall status

Usage:
    python demo/verify_index.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oncocontext.storage.chroma_manager import ChromaManager
from oncocontext.storage.sqlite_manager import SQLiteManager
from oncocontext.services.embedder import Embedder
from oncocontext.config import settings


async def verify_index() -> bool:
    """Verify that the index is ready for the demo.

    Returns:
        True if all checks pass.
    """
    chroma = ChromaManager()
    sqlite = SQLiteManager()
    await sqlite.init_db()

    all_ok = True

    print(f"\n🔍 OncoContext Index Verification")
    print(f"{'=' * 50}")

    # Check 1: ChromaDB literature collection
    print(f"\n1️⃣  ChromaDB Literature Collection")
    lit_stats = chroma.get_collection_stats("literature")
    lit_count = lit_stats["count"]
    if lit_count > 0:
        print(f"   ✅ {lit_count} chunks indexed")
    else:
        print(f"   ❌ No chunks indexed! Run: python demo/pre_index_papers.py")
        all_ok = False

    # Check 2: ChromaDB lab data collection
    print(f"\n2️⃣  ChromaDB Lab Data Collection")
    lab_stats = chroma.get_collection_stats("lab")
    lab_count = lab_stats["count"]
    if lab_count > 0:
        print(f"   ✅ {lab_count} lab data chunks")
    else:
        print(f"   ⚠️  No lab data indexed (run demo/ingest_demo_csv.py before demo)")

    # Check 3: SQLite paper count
    print(f"\n3️⃣  SQLite Paper Metadata")
    paper_count = await sqlite.get_indexed_paper_count()
    if paper_count > 0:
        print(f"   ✅ {paper_count} papers with index status")
    else:
        print(f"   ❌ No indexed papers in SQLite")
        all_ok = False

    # Check 4: Test embedding model loads
    print(f"\n4️⃣  PubMedBERT Embedding Model")
    try:
        embedder = Embedder()
        test_embedding = embedder.embed_text("T cell exhaustion PD-1")
        if len(test_embedding) == 768:
            print(f"   ✅ Model loaded, produces 768d vectors")
        else:
            print(f"   ❌ Unexpected embedding dimension: {len(test_embedding)}")
            all_ok = False
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        all_ok = False

    # Check 5: Test vector search
    print(f"\n5️⃣  Test Deep Search Query")
    if lit_count > 0:
        try:
            test_query = "What gating strategy was used for exhausted CD8 T cells?"
            query_emb = embedder.embed_text(test_query)
            results = chroma.search(
                query_embedding=query_emb,
                collection="literature",
                n_results=3,
            )
            result_ids = results.get("ids", [[]])[0]
            if result_ids:
                print(f"   ✅ Search returned {len(result_ids)} results")
                # Show a preview of the first result
                first_doc = results.get("documents", [[]])[0]
                if first_doc:
                    preview = first_doc[0][:100] + "..." if len(first_doc[0]) > 100 else first_doc[0]
                    print(f"   📝 First result preview: {preview}")
            else:
                print(f"   ⚠️  Search returned no results")
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            all_ok = False
    else:
        print(f"   ⏭️  Skipped (no papers indexed)")

    # Check 6: Verify demo CSV exists
    print(f"\n6️⃣  Demo CSV File")
    csv_path = Path(__file__).resolve().parent / "sample_flow_cytometry.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            lines = f.readlines()
        print(f"   ✅ Found: {csv_path.name} ({len(lines) - 1} rows)")
    else:
        print(f"   ❌ Not found: {csv_path}")
        all_ok = False

    # Check 7: Verify synonym dictionary
    print(f"\n7️⃣  Synonym Dictionary")
    if settings.SYNONYM_DICT_PATH.exists():
        import json
        with open(settings.SYNONYM_DICT_PATH) as f:
            syn_data = json.load(f)
        print(f"   ✅ Found: {len(syn_data)} entries")
    else:
        print(f"   ❌ Not found: {settings.SYNONYM_DICT_PATH}")
        all_ok = False

    # Summary
    print(f"\n{'=' * 50}")
    if all_ok:
        print(f"✅ All checks passed! Index is ready for the demo.")
        print(f"\n📊 Summary:")
        print(f"   Literature chunks: {lit_count}")
        print(f"   Lab data chunks: {lab_count}")
        print(f"   Indexed papers: {paper_count}")
    else:
        print(f"❌ Some checks failed. Fix the issues above before the demo.")

    return all_ok


if __name__ == "__main__":
    ok = asyncio.run(verify_index())
    sys.exit(0 if ok else 1)
