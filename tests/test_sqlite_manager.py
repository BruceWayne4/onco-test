"""Unit tests for SQLiteManager.

Tests async SQLite operations using a temporary database.
"""

import pytest

from oncocontext.storage.sqlite_manager import SQLiteManager


@pytest.fixture
async def sqlite(tmp_path) -> SQLiteManager:
    """Create a SQLiteManager with a temp database."""
    db_path = str(tmp_path / "test_metadata.db")
    mgr = SQLiteManager(db_path=db_path)
    await mgr.init_db()
    return mgr


@pytest.fixture
def sample_paper() -> dict:
    """Create a sample paper dict."""
    return {
        "pmid": "12345678",
        "pmc_id": "PMC1234567",
        "title": "CD8+ T Cell Exhaustion in Tumor Organoid Co-Culture Models",
        "authors": ["Smith J", "Chen L", "Park S"],
        "journal": "Nature Immunology",
        "year": 2024,
        "abstract": "T cell exhaustion is a hallmark of chronic infections and cancer.",
        "mesh_terms": ["T-Cell Exhaustion", "CD8-Positive T-Lymphocytes", "PD-1"],
        "has_full_text": True,
    }


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Create sample chunk dicts for SQLite."""
    return [
        {
            "chunk_id": "12345678_methods_0_0",
            "source_type": "paper",
            "source_id": "12345678",
            "section": "methods",
            "paragraph_index": 0,
            "token_count": 50,
            "text": "[Methods] Flow cytometry was performed using anti-CD8 clone RPA-T8.",
            "chromadb_collection": "literature_chunks",
        },
        {
            "chunk_id": "12345678_methods_1_0",
            "source_type": "paper",
            "source_id": "12345678",
            "section": "methods",
            "paragraph_index": 1,
            "token_count": 45,
            "text": "[Methods] Cells were stained with anti-PD-1 clone EH12.2H7.",
            "chromadb_collection": "literature_chunks",
        },
        {
            "chunk_id": "12345678_results_0_0",
            "source_type": "paper",
            "source_id": "12345678",
            "section": "results",
            "paragraph_index": 0,
            "token_count": 60,
            "text": "[Results] PD-1+TIM-3+ CD8+ T cells constituted 45-60%.",
            "chromadb_collection": "literature_chunks",
        },
    ]


class TestSQLiteInitDb:
    """Tests for database initialization."""

    @pytest.mark.asyncio
    async def test_init_creates_tables(self, tmp_path):
        """init_db creates all required tables."""
        import aiosqlite

        db_path = str(tmp_path / "init_test.db")
        mgr = SQLiteManager(db_path=db_path)
        await mgr.init_db()

        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in await cursor.fetchall()]

        assert "papers" in tables
        assert "chunks" in tables
        assert "lab_files" in tables
        assert "search_log" in tables

    @pytest.mark.asyncio
    async def test_init_idempotent(self, tmp_path):
        """init_db can be called multiple times safely."""
        db_path = str(tmp_path / "idempotent_test.db")
        mgr = SQLiteManager(db_path=db_path)
        await mgr.init_db()
        await mgr.init_db()  # Should not raise

    @pytest.mark.asyncio
    async def test_init_creates_directory(self, tmp_path):
        """init_db creates parent directory if it doesn't exist."""
        db_path = str(tmp_path / "subdir" / "deep" / "test.db")
        mgr = SQLiteManager(db_path=db_path)
        await mgr.init_db()
        # If we got here without error, the directory was created


class TestSQLitePapers:
    """Tests for paper CRUD operations."""

    @pytest.mark.asyncio
    async def test_add_and_get_paper(self, sqlite: SQLiteManager, sample_paper):
        """add_paper then get_paper returns the same data."""
        await sqlite.add_paper(sample_paper)
        paper = await sqlite.get_paper("12345678")

        assert paper is not None
        assert paper["pmid"] == "12345678"
        assert paper["title"] == "CD8+ T Cell Exhaustion in Tumor Organoid Co-Culture Models"
        assert paper["journal"] == "Nature Immunology"
        assert paper["year"] == 2024
        assert paper["pmc_id"] == "PMC1234567"
        assert paper["has_full_text"] == 1  # SQLite stores as int

    @pytest.mark.asyncio
    async def test_get_paper_authors_parsed(self, sqlite: SQLiteManager, sample_paper):
        """Authors are stored as JSON and parsed back to list."""
        await sqlite.add_paper(sample_paper)
        paper = await sqlite.get_paper("12345678")
        assert isinstance(paper["authors"], list)
        assert "Smith J" in paper["authors"]

    @pytest.mark.asyncio
    async def test_get_paper_mesh_terms_parsed(self, sqlite: SQLiteManager, sample_paper):
        """MeSH terms are stored as JSON and parsed back to list."""
        await sqlite.add_paper(sample_paper)
        paper = await sqlite.get_paper("12345678")
        assert isinstance(paper["mesh_terms"], list)
        assert "T-Cell Exhaustion" in paper["mesh_terms"]

    @pytest.mark.asyncio
    async def test_get_nonexistent_paper(self, sqlite: SQLiteManager):
        """get_paper returns None for non-existent PMID."""
        paper = await sqlite.get_paper("99999999")
        assert paper is None

    @pytest.mark.asyncio
    async def test_add_paper_upsert(self, sqlite: SQLiteManager, sample_paper):
        """Adding a paper twice updates instead of duplicating."""
        await sqlite.add_paper(sample_paper)
        sample_paper["title"] = "Updated Title"
        await sqlite.add_paper(sample_paper)

        paper = await sqlite.get_paper("12345678")
        assert paper["title"] == "Updated Title"


class TestSQLitePaperIndexing:
    """Tests for paper indexing status."""

    @pytest.mark.asyncio
    async def test_is_paper_indexed_false_by_default(self, sqlite: SQLiteManager, sample_paper):
        """Newly added paper is not indexed."""
        await sqlite.add_paper(sample_paper)
        assert await sqlite.is_paper_indexed("12345678") is False

    @pytest.mark.asyncio
    async def test_is_paper_indexed_after_update(self, sqlite: SQLiteManager, sample_paper):
        """Paper is indexed after update_paper_index_status."""
        await sqlite.add_paper(sample_paper)
        await sqlite.update_paper_index_status("12345678", chunk_count=15)
        assert await sqlite.is_paper_indexed("12345678") is True

    @pytest.mark.asyncio
    async def test_is_paper_indexed_nonexistent(self, sqlite: SQLiteManager):
        """Non-existent paper is not indexed."""
        assert await sqlite.is_paper_indexed("99999") is False

    @pytest.mark.asyncio
    async def test_get_indexed_paper_count_empty(self, sqlite: SQLiteManager):
        """No papers indexed returns 0."""
        count = await sqlite.get_indexed_paper_count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_indexed_paper_count(self, sqlite: SQLiteManager):
        """Count reflects number of indexed papers."""
        # Add 3 papers, index 2
        for i in range(3):
            await sqlite.add_paper({"pmid": str(i), "title": f"Paper {i}"})

        await sqlite.update_paper_index_status("0", chunk_count=10)
        await sqlite.update_paper_index_status("1", chunk_count=20)

        count = await sqlite.get_indexed_paper_count()
        assert count == 2


class TestSQLiteChunks:
    """Tests for chunk operations."""

    @pytest.mark.asyncio
    async def test_add_chunks(self, sqlite: SQLiteManager, sample_chunks):
        """add_chunks stores chunk records."""
        await sqlite.add_chunks(sample_chunks)

        # Verify by getting chunks for the paper
        chunks = await sqlite.get_chunks_for_paper("12345678")
        assert len(chunks) == 3

    @pytest.mark.asyncio
    async def test_get_chunks_for_paper(self, sqlite: SQLiteManager, sample_chunks):
        """get_chunks_for_paper returns chunks ordered by section and paragraph."""
        await sqlite.add_chunks(sample_chunks)
        chunks = await sqlite.get_chunks_for_paper("12345678")

        assert len(chunks) == 3
        # methods should come before results alphabetically
        assert chunks[0]["section"] == "methods"
        assert chunks[2]["section"] == "results"

    @pytest.mark.asyncio
    async def test_get_chunks_for_nonexistent_paper(self, sqlite: SQLiteManager):
        """get_chunks_for_paper returns empty list for non-existent paper."""
        chunks = await sqlite.get_chunks_for_paper("99999")
        assert chunks == []

    @pytest.mark.asyncio
    async def test_get_surrounding_chunks(self, sqlite: SQLiteManager, sample_chunks):
        """get_surrounding_chunks returns ±1 paragraphs."""
        await sqlite.add_chunks(sample_chunks)

        # Get context around the second methods paragraph (index 1)
        context = await sqlite.get_surrounding_chunks("12345678", "methods", 1)

        # Previous paragraph (index 0) should exist
        assert context["previous_paragraph"] is not None
        assert "RPA-T8" in context["previous_paragraph"]

        # Next paragraph (index 2) doesn't exist for methods
        assert context["next_paragraph"] is None

    @pytest.mark.asyncio
    async def test_get_surrounding_chunks_first_paragraph(self, sqlite: SQLiteManager, sample_chunks):
        """First paragraph has no previous paragraph."""
        await sqlite.add_chunks(sample_chunks)
        context = await sqlite.get_surrounding_chunks("12345678", "methods", 0)

        assert context["previous_paragraph"] is None
        assert context["next_paragraph"] is not None


class TestSQLiteLabFiles:
    """Tests for lab file operations."""

    @pytest.mark.asyncio
    async def test_add_and_get_lab_file(self, sqlite: SQLiteManager):
        """add_lab_file then get_lab_file returns the same data."""
        lab_file = {
            "file_id": "lab_001",
            "file_name": "flow_cytometry.csv",
            "file_type": "csv",
            "file_path": "/data/flow_cytometry.csv",
            "experiment_label": "CD8 exhaustion panel",
            "row_count": 20,
            "column_names": ["Sample_ID", "CD8_percent", "PD1_MFI"],
            "summary": "Flow cytometry data with 20 samples.",
            "detected_markers": ["CD8", "PD-1"],
            "chunk_count": 5,
        }
        await sqlite.add_lab_file(lab_file)
        result = await sqlite.get_lab_file("lab_001")

        assert result is not None
        assert result["file_id"] == "lab_001"
        assert result["file_name"] == "flow_cytometry.csv"
        assert result["row_count"] == 20
        assert isinstance(result["column_names"], list)
        assert "CD8_percent" in result["column_names"]
        assert isinstance(result["detected_markers"], list)
        assert "CD8" in result["detected_markers"]

    @pytest.mark.asyncio
    async def test_get_nonexistent_lab_file(self, sqlite: SQLiteManager):
        """get_lab_file returns None for non-existent file."""
        result = await sqlite.get_lab_file("nonexistent")
        assert result is None


class TestSQLiteSearchLog:
    """Tests for search logging."""

    @pytest.mark.asyncio
    async def test_log_search(self, sqlite: SQLiteManager):
        """log_search writes a record without error."""
        await sqlite.log_search(
            tool_name="deep_search",
            query="CD8 exhaustion gating strategy",
            result_count=10,
            latency_ms=250,
            params={"search_scope": "methods", "max_results": 10},
        )
        # If we get here without error, the log was written

    @pytest.mark.asyncio
    async def test_log_search_no_params(self, sqlite: SQLiteManager):
        """log_search works with no params."""
        await sqlite.log_search(
            tool_name="search_literature",
            query="T cell exhaustion",
            result_count=20,
            latency_ms=800,
        )

    @pytest.mark.asyncio
    async def test_log_multiple_searches(self, sqlite: SQLiteManager):
        """Multiple searches can be logged."""
        for i in range(5):
            await sqlite.log_search(
                tool_name="deep_search",
                query=f"query {i}",
                result_count=i * 3,
                latency_ms=100 + i * 50,
            )
        # Verify by querying directly
        import aiosqlite
        async with aiosqlite.connect(sqlite._db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM search_log")
            row = await cursor.fetchone()
            assert row[0] == 5
