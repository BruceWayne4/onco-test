"""SQLite metadata manager — papers, chunks, lab files, search log.

Provides async CRUD operations using aiosqlite with the schema defined
in the MVP plan (§13).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from oncocontext.config import settings

logger = logging.getLogger(__name__)

# SQL schema from MVP plan §13
SCHEMA_SQL = """
-- Papers metadata
CREATE TABLE IF NOT EXISTS papers (
    pmid TEXT PRIMARY KEY,
    pmc_id TEXT,
    title TEXT NOT NULL,
    authors TEXT,                -- JSON array: ["Smith J", "Chen L"]
    journal TEXT,
    year INTEGER,
    abstract TEXT,
    mesh_terms TEXT,             -- JSON array: ["T-Cell Exhaustion", "PD-1"]
    has_full_text BOOLEAN DEFAULT FALSE,
    relevance_score REAL,
    indexed_at TIMESTAMP,
    chunk_count INTEGER DEFAULT 0,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Lab files registry
CREATE TABLE IF NOT EXISTS lab_files (
    file_id TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    experiment_label TEXT,
    metadata TEXT,               -- JSON
    row_count INTEGER,
    column_names TEXT,           -- JSON array
    summary TEXT,
    detected_markers TEXT,       -- JSON array
    chunk_count INTEGER DEFAULT 0,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chunk-to-source mapping
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,   -- "paper" | "lab_file"
    source_id TEXT NOT NULL,     -- pmid or file_id
    section TEXT,
    subsection TEXT,
    paragraph_index INTEGER,
    token_count INTEGER,
    text TEXT,
    chromadb_collection TEXT,    -- "literature_chunks" | "lab_data"
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Search log
CREATE TABLE IF NOT EXISTS search_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT NOT NULL,
    query TEXT,
    params TEXT,                 -- JSON
    result_count INTEGER,
    latency_ms INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_indexed ON papers(indexed_at);
CREATE INDEX IF NOT EXISTS idx_papers_pmc ON papers(pmc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section);
CREATE INDEX IF NOT EXISTS idx_lab_files_type ON lab_files(file_type);
CREATE INDEX IF NOT EXISTS idx_search_log_tool ON search_log(tool_name);
"""


class SQLiteManager:
    """Async SQLite manager for metadata storage.

    Tables:
        - papers: Paper metadata (PMID, title, authors, etc.)
        - lab_files: Ingested lab file metadata
        - chunks: Chunk-to-source mapping
        - search_log: Tool usage logging
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize SQLite manager.

        Args:
            db_path: Path to SQLite database file.
                Defaults to settings.SQLITE_DB_PATH.
        """
        self._db_path = db_path or str(settings.SQLITE_DB_PATH)
        self._initialized = False

    async def init_db(self) -> None:
        """Create all tables and indexes if they don't exist."""
        if self._initialized:
            return

        # Ensure directory exists
        db_dir = Path(self._db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(SCHEMA_SQL)
            await db.commit()

        self._initialized = True
        logger.info("SQLite database initialized at %s", self._db_path)

    async def _ensure_init(self) -> None:
        """Ensure DB is initialized before operations."""
        if not self._initialized:
            await self.init_db()

    # ── Papers ─────────────────────────────────────────────────────────────────

    async def add_paper(self, paper: dict) -> None:
        """Insert or update a paper record.

        Args:
            paper: Dict with paper metadata. Required: pmid, title.
                Optional: pmc_id, authors, journal, year, abstract,
                mesh_terms, has_full_text, relevance_score,
                indexed_at, chunk_count.
        """
        await self._ensure_init()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO papers (pmid, pmc_id, title, authors, journal, year,
                                    abstract, mesh_terms, has_full_text,
                                    relevance_score, indexed_at, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pmid) DO UPDATE SET
                    pmc_id = COALESCE(excluded.pmc_id, papers.pmc_id),
                    title = COALESCE(excluded.title, papers.title),
                    authors = COALESCE(excluded.authors, papers.authors),
                    journal = COALESCE(excluded.journal, papers.journal),
                    year = COALESCE(excluded.year, papers.year),
                    abstract = COALESCE(excluded.abstract, papers.abstract),
                    mesh_terms = COALESCE(excluded.mesh_terms, papers.mesh_terms),
                    has_full_text = COALESCE(excluded.has_full_text, papers.has_full_text),
                    relevance_score = COALESCE(excluded.relevance_score, papers.relevance_score),
                    indexed_at = COALESCE(excluded.indexed_at, papers.indexed_at),
                    chunk_count = COALESCE(excluded.chunk_count, papers.chunk_count)
                """,
                (
                    paper.get("pmid"),
                    paper.get("pmc_id"),
                    paper.get("title", ""),
                    json.dumps(paper.get("authors", [])),
                    paper.get("journal", ""),
                    paper.get("year", 0),
                    paper.get("abstract", ""),
                    json.dumps(paper.get("mesh_terms", [])),
                    paper.get("has_full_text", False),
                    paper.get("relevance_score"),
                    paper.get("indexed_at"),
                    paper.get("chunk_count", 0),
                ),
            )
            await db.commit()

    async def get_paper(self, pmid: str) -> dict | None:
        """Retrieve a paper by PMID.

        Args:
            pmid: PubMed ID.

        Returns:
            Paper metadata dict, or None if not found.
        """
        await self._ensure_init()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM papers WHERE pmid = ?", (pmid,))
            row = await cursor.fetchone()

            if row is None:
                return None

            return self._row_to_paper_dict(row)

    async def is_paper_indexed(self, pmid: str) -> bool:
        """Check if paper has been indexed (has chunks).

        Args:
            pmid: PubMed ID.

        Returns:
            True if paper has indexed_at set and chunk_count > 0.
        """
        await self._ensure_init()

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT chunk_count FROM papers WHERE pmid = ? AND indexed_at IS NOT NULL",
                (pmid,),
            )
            row = await cursor.fetchone()
            if row is None:
                return False
            return row[0] > 0

    async def get_indexed_paper_count(self) -> int:
        """Get count of indexed papers (those with chunks).

        Returns:
            Number of papers that have been indexed.
        """
        await self._ensure_init()

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM papers WHERE indexed_at IS NOT NULL AND chunk_count > 0"
            )
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def update_paper_index_status(
        self, pmid: str, chunk_count: int, indexed_at: str | None = None
    ) -> None:
        """Update a paper's indexing status.

        Args:
            pmid: PubMed ID.
            chunk_count: Number of chunks indexed.
            indexed_at: ISO timestamp. Defaults to current time.
        """
        await self._ensure_init()

        if indexed_at is None:
            indexed_at = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE papers SET indexed_at = ?, chunk_count = ? WHERE pmid = ?",
                (indexed_at, chunk_count, pmid),
            )
            await db.commit()

    # ── Chunks ─────────────────────────────────────────────────────────────────

    async def add_chunks(self, chunks: list[dict]) -> None:
        """Insert chunk records (for context assembly later).

        Args:
            chunks: List of chunk dicts. Required: chunk_id, source_type,
                source_id, section, text. Optional: subsection,
                paragraph_index, token_count, chromadb_collection.
        """
        await self._ensure_init()

        async with aiosqlite.connect(self._db_path) as db:
            for chunk in chunks:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO chunks
                        (chunk_id, source_type, source_id, section, subsection,
                         paragraph_index, token_count, text, chromadb_collection)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.get("chunk_id"),
                        chunk.get("source_type", "paper"),
                        chunk.get("source_id", chunk.get("paper_pmid", "")),
                        chunk.get("section", ""),
                        chunk.get("subsection"),
                        chunk.get("paragraph_index", chunk.get("paragraph_num", 0)),
                        chunk.get("token_count", 0),
                        chunk.get("text", ""),
                        chunk.get("chromadb_collection", "literature_chunks"),
                    ),
                )
            await db.commit()

    async def get_chunks_for_paper(self, pmid: str) -> list[dict]:
        """Get all chunks for a given paper.

        Args:
            pmid: PubMed ID.

        Returns:
            List of chunk dicts.
        """
        await self._ensure_init()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT * FROM chunks
                   WHERE source_type = 'paper' AND source_id = ?
                   ORDER BY section, paragraph_index""",
                (pmid,),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_surrounding_chunks(
        self, paper_pmid: str, section: str, paragraph_index: int
    ) -> dict:
        """Get ±1 surrounding paragraphs for context assembly.

        Args:
            paper_pmid: PubMed ID.
            section: Section name.
            paragraph_index: Current paragraph index.

        Returns:
            Dict with 'previous_paragraph' and 'next_paragraph' (or None).
        """
        await self._ensure_init()

        result = {"previous_paragraph": None, "next_paragraph": None}

        async with aiosqlite.connect(self._db_path) as db:
            # Previous paragraph
            cursor = await db.execute(
                """SELECT text FROM chunks
                   WHERE source_type = 'paper' AND source_id = ?
                   AND section = ? AND paragraph_index = ?
                   LIMIT 1""",
                (paper_pmid, section, paragraph_index - 1),
            )
            row = await cursor.fetchone()
            if row:
                result["previous_paragraph"] = row[0]

            # Next paragraph
            cursor = await db.execute(
                """SELECT text FROM chunks
                   WHERE source_type = 'paper' AND source_id = ?
                   AND section = ? AND paragraph_index = ?
                   LIMIT 1""",
                (paper_pmid, section, paragraph_index + 1),
            )
            row = await cursor.fetchone()
            if row:
                result["next_paragraph"] = row[0]

        return result

    # ── Lab Files ──────────────────────────────────────────────────────────────

    async def add_lab_file(self, lab_file: dict) -> None:
        """Insert a lab file record.

        Args:
            lab_file: Dict with lab file metadata. Required: file_id,
                file_name, file_type, file_path.
        """
        await self._ensure_init()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO lab_files
                    (file_id, file_name, file_type, file_path,
                     experiment_label, metadata, row_count,
                     column_names, summary, detected_markers, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lab_file.get("file_id"),
                    lab_file.get("file_name", ""),
                    lab_file.get("file_type", "csv"),
                    lab_file.get("file_path", ""),
                    lab_file.get("experiment_label"),
                    json.dumps(lab_file.get("metadata")) if lab_file.get("metadata") else None,
                    lab_file.get("row_count", 0),
                    json.dumps(lab_file.get("column_names", [])),
                    lab_file.get("summary", ""),
                    json.dumps(lab_file.get("detected_markers", [])),
                    lab_file.get("chunk_count", 0),
                ),
            )
            await db.commit()

    async def get_lab_file(self, file_id: str) -> dict | None:
        """Get lab file by ID.

        Args:
            file_id: Lab file identifier.

        Returns:
            Lab file metadata dict, or None if not found.
        """
        await self._ensure_init()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM lab_files WHERE file_id = ?", (file_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None

            result = dict(row)
            # Parse JSON fields
            for field in ("column_names", "detected_markers"):
                if result.get(field) and isinstance(result[field], str):
                    try:
                        result[field] = json.loads(result[field])
                    except json.JSONDecodeError:
                        pass
            if result.get("metadata") and isinstance(result["metadata"], str):
                try:
                    result["metadata"] = json.loads(result["metadata"])
                except json.JSONDecodeError:
                    pass
            return result

    # ── Search Log ─────────────────────────────────────────────────────────────

    async def log_search(
        self,
        tool_name: str,
        query: str,
        result_count: int,
        latency_ms: int,
        params: dict | None = None,
    ) -> None:
        """Log a tool invocation to the search_log table.

        Args:
            tool_name: Name of the MCP tool called.
            query: User's query string.
            result_count: Number of results returned.
            latency_ms: Execution time in milliseconds.
            params: Tool parameters as dict.
        """
        await self._ensure_init()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO search_log (tool_name, query, params, result_count, latency_ms)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    tool_name,
                    query,
                    json.dumps(params) if params else None,
                    result_count,
                    latency_ms,
                ),
            )
            await db.commit()

    # ── Close ──────────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the database connection (no-op for aiosqlite per-call pattern)."""
        pass

    # ── Helpers ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_paper_dict(row) -> dict:
        """Convert an aiosqlite Row to a paper dict with parsed JSON fields."""
        result = dict(row)

        # Parse JSON fields
        for field in ("authors", "mesh_terms"):
            if result.get(field) and isinstance(result[field], str):
                try:
                    result[field] = json.loads(result[field])
                except json.JSONDecodeError:
                    result[field] = []

        return result
