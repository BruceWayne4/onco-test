"""ingest_lab_file tool — Parse CSV/Excel lab data, analyze, embed, and index.

Handles flow cytometry data, marker detection, and summary generation.
Embeds text representations into ChromaDB lab_data collection.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from uuid import uuid4

from oncocontext.config import settings
from oncocontext.services.csv_parser import LabFileParser
from oncocontext.services.embedder import Embedder
from oncocontext.storage.chroma_manager import ChromaManager
from oncocontext.storage.sqlite_manager import SQLiteManager

logger = logging.getLogger(__name__)

# ── Module-level lazy singletons ──────────────────────────────────────────────

_parser: LabFileParser | None = None
_embedder: Embedder | None = None
_chroma: ChromaManager | None = None
_sqlite: SQLiteManager | None = None


def _get_parser() -> LabFileParser:
    global _parser
    if _parser is None:
        _parser = LabFileParser()
    return _parser


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


# ── Supported extensions ──────────────────────────────────────────────────────

_CSV_EXTENSIONS = {".csv", ".tsv"}
_EXCEL_EXTENSIONS = {".xlsx", ".xls"}
_ALL_EXTENSIONS = _CSV_EXTENSIONS | _EXCEL_EXTENSIONS


def _detect_file_type(file_path: str) -> str:
    """Detect file type from extension.

    Args:
        file_path: Path to the file.

    Returns:
        'csv' or 'excel'.

    Raises:
        ValueError: If extension is not supported.
    """
    ext = Path(file_path).suffix.lower()
    if ext in _CSV_EXTENSIONS:
        return "csv"
    elif ext in _EXCEL_EXTENSIONS:
        return "excel"
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {', '.join(sorted(_ALL_EXTENSIONS))}"
        )


async def ingest_lab_file(
    file_path: str,
    file_type: str = "auto",
    experiment_label: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Parse a CSV or Excel file of lab data, analyze columns, and embed into ChromaDB.

    Algorithm:
        1. Validate file exists, readable, supported extension, <50MB
        2. Detect file type from extension if auto
        3. Parse with pandas (read_csv / read_excel)
        4. Analyze columns: dtype, stats, sample values
        5. Detect biological markers in column names (CD3, PD-1, TIM-3, etc.)
        6. Generate structured text summary
        7. Generate per-row text representations
        8. Embed summaries with PubMedBERT
        9. Store in ChromaDB lab_data collection
        10. Store metadata in SQLite lab_files table

    Args:
        file_path: Absolute or relative path to the CSV or Excel file.
        file_type: File type ('csv', 'excel', 'auto'). Auto detects from extension.
        experiment_label: Researcher's label for this experiment.
        metadata: Additional context (cell_line, treatment, timepoint, etc.).

    Returns:
        Dict with file stats, column analysis, detected markers, and index status.
    """
    start_time = time.time()

    parser = _get_parser()
    embedder = _get_embedder()
    chroma = _get_chroma()
    sqlite = _get_sqlite()
    await sqlite.init_db()

    # Step 1: Validate file
    path = Path(file_path)
    if not path.is_absolute():
        # Try relative to project root
        project_root = Path(settings.DATA_DIR).parent
        candidate = project_root / file_path
        if candidate.exists():
            path = candidate
        elif not path.exists():
            return {"error": f"File not found: {file_path}"}

    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    if not path.is_file():
        return {"error": f"Not a file: {file_path}"}

    # Check file size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > settings.MAX_LAB_FILE_SIZE_MB:
        return {
            "error": f"File exceeds {settings.MAX_LAB_FILE_SIZE_MB}MB limit "
            f"({file_size_mb:.1f}MB). Consider splitting into smaller files."
        }

    # Step 2: Detect file type
    if file_type == "auto":
        try:
            detected_type = _detect_file_type(str(path))
        except ValueError as exc:
            return {"error": str(exc)}
    else:
        detected_type = file_type

    # Step 3: Parse file
    try:
        if detected_type == "csv":
            parsed = parser.parse_csv(str(path))
        else:
            parsed = parser.parse_excel(str(path))
    except ValueError as exc:
        return {"error": f"Failed to parse file: {exc}"}
    except Exception as exc:
        return {
            "error": f"Failed to parse file: {exc}",
            "suggestion": "Check file encoding and format",
        }

    # Step 4-6: Already done by parser (column types, markers, summary)
    markers = parsed["markers_detected"]
    text_reps = parsed["text_representations"]
    file_summary = parsed["summary"]

    # Generate file ID
    file_id = f"lab_{uuid4().hex[:8]}"

    # Step 7-8: Embed text representations
    # Include file summary as first chunk
    all_texts = [file_summary] + text_reps

    try:
        embeddings = embedder.embed_batch(all_texts)
    except Exception as exc:
        logger.error("Failed to embed lab data: %s", exc)
        return {
            "error": f"Embedding failed: {exc}",
            "file_name": path.name,
            "markers_detected": markers,
        }

    # Step 9: Build chunks and store in ChromaDB
    file_name = path.name
    markers_str = ",".join(markers)
    label = experiment_label or ""

    chunks = []

    # File summary chunk
    chunks.append({
        "chunk_id": f"{file_id}_summary",
        "text": file_summary,
        "file_id": file_id,
        "file_name": file_name,
        "chunk_type": "file_summary",
        "row_index": -1,
        "experiment_label": label,
        "markers": markers_str,
    })

    # Per-row chunks
    for i, text in enumerate(text_reps):
        chunks.append({
            "chunk_id": f"{file_id}_row_{i}",
            "text": text,
            "file_id": file_id,
            "file_name": file_name,
            "chunk_type": "row_data",
            "row_index": i,
            "experiment_label": label,
            "markers": markers_str,
        })

    try:
        chunk_count = chroma.add_chunks(chunks, embeddings, collection="lab")
    except Exception as exc:
        logger.error("Failed to store lab data in ChromaDB: %s", exc)
        return {
            "error": f"ChromaDB storage failed: {exc}",
            "file_name": file_name,
            "markers_detected": markers,
        }

    # Step 10: Store metadata in SQLite
    try:
        await sqlite.add_lab_file({
            "file_id": file_id,
            "file_name": file_name,
            "file_type": detected_type,
            "file_path": str(path),
            "experiment_label": experiment_label,
            "metadata": metadata,
            "row_count": parsed["rows"],
            "column_names": parsed["columns"],
            "summary": file_summary,
            "detected_markers": markers,
            "chunk_count": chunk_count,
        })

        # Also store chunks in SQLite for context lookup
        sqlite_chunks = []
        for chunk in chunks:
            sqlite_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "source_type": "lab_file",
                "source_id": file_id,
                "section": chunk["chunk_type"],
                "paragraph_index": chunk["row_index"],
                "token_count": len(chunk["text"].split()),
                "text": chunk["text"],
                "chromadb_collection": "lab_data",
            })
        await sqlite.add_chunks(sqlite_chunks)

    except Exception as exc:
        logger.warning("Failed to store lab file metadata in SQLite: %s", exc)
        # Non-fatal — data is still in ChromaDB

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Log the tool invocation
    try:
        await sqlite.log_search(
            tool_name="ingest_lab_file",
            query=file_path,
            result_count=chunk_count,
            latency_ms=elapsed_ms,
            params={"file_type": detected_type, "experiment_label": experiment_label},
        )
    except Exception:
        pass

    return {
        "file_id": file_id,
        "file_name": file_name,
        "file_type_detected": detected_type,
        "row_count": parsed["rows"],
        "column_count": len(parsed["columns"]),
        "columns": parsed["column_info"],
        "summary": file_summary,
        "detected_markers": markers,
        "indexed": True,
        "chunk_count": chunk_count,
    }
