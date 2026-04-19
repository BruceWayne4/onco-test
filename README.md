# OncoContext

OncoContext is a Model Context Protocol (MCP) server for oncology research. It exposes a set of tools to search and index literature (PubMed / PMC), semantically search full-text papers, ingest and locally index private lab files (CSV/Excel), cross-reference lab findings against the literature with paragraph-level citations, crawl supplementary files, and generate comprehensive offline evidence reports.

This repository contains the server and all services under `src/oncocontext/`.

---

## Quick links
- Source root: `src/oncocontext/`
- Tools are registered in: `src/oncocontext/server.py`
- Synonym dictionary: `resources/synonym_dict.json`
- Crawled data path: `data/crawled/`
- Reports path: `data/reports/`

---

## Highlights / Features

- 8 MCP tools (registered via `@mcp.tool()` in `server.py`):
  - `search_literature` — PubMed search with ontology-aware query expansion and result-aware relaxation.
  - `get_paper_details` — Fetch paper metadata and (optionally) PMC full-text, BioC parsing, and indexing into ChromaDB.
  - `deep_search` — Semantic search across indexed full text (ChromaDB) with cross-encoder reranking and section-aware boosts.
  - `ingest_lab_file` — Parse and index local CSV/Excel lab files (embeddings stored locally). No data ever leaves the machine.
  - `cross_reference` — Compare lab data against indexed literature and produce structured agreements/contradictions/novel findings with citations.
  - `crawl_supplementary` — Discover and parse supplementary files (CSV, XLSX, PDF, XML, tar.gz, etc.) from PMC / Europe PMC.
  - `crawl_and_report` — Full paper crawl + local persistence + Markdown evidence report generation; saves outputs under `data/crawled/<pmcid>/` and `data/reports/`.
  - `get_next_page` — Retrieve subsequent pages for large, paginated responses (sessions expire after 30 minutes).

- Embedding & reranking:
  - Embeddings: `pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb` (768-d, L2-normalized).
  - Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (sigmoid-normalized scores).
  - Chunking: Section-aware 384-token chunks with 64-token overlap.

- Storage layer:
  - Vector store: ChromaDB (collections: `literature_chunks`, `lab_data`, ...)
  - Metadata and mappings: SQLite (tables include `papers`, `lab_files`, `chunks`, `search_log`, etc.)
  - Local crawl files: `data/crawled/<pmcid>/` (full_text.txt, metadata.json, references.json, supplementary/*)

- Privacy-first: Lab files are parsed and indexed locally; no lab data is sent externally by default.

---

## Installation

1. Clone the repo:
   git clone https://github.com/BruceWayne4/onco-test.git
   cd onco-test

2. Create a virtual environment and install dependencies:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

3. Optional: configure environment variables (see Configuration section).

---

## Run the MCP server

Start the MCP server (stdio transport via FastMCP):

python -m src.oncocontext.server

The server registers as "OncoContext" and exposes the tools listed above. Use any MCP-compatible client to call tools over stdio (or integrate into an existing MCP bridge).

Example (conceptual) tool call payload:
{
  "tool": "search_literature",
  "params": {
    "query": "PD-1 expression in non-small cell lung cancer",
    "max_results": 10,
    "full_text_only": true
  }
}

(Concrete client examples are intentionally left generic — use your MCP client to call registered tools.)

---

## Tool details (summary)

- search_literature:
  - Query expansion using `resources/synonym_dict.json` (57 entries).
  - 3-tier query funnel with progressive relaxation, PubMed esearch + efetch.
  - Composite relevance scoring (title Jaccard 30%, abstract TF 40%, MeSH overlap 15%, recency 15%).

- get_paper_details:
  - Fetch metadata, optional PMC BioC full-text (cached), BioC parsing to structured sections.
  - Section-aware chunking → embeddings → ChromaDB → SQLite mappings.

- deep_search:
  - Embed query → ChromaDB vector search (top K candidates) → section boost → cross-encoder rerank → filter by min_relevance → return results with surrounding context and citation enrichment.

- ingest_lab_file:
  - Local CSV / TSV / XLSX ingestion, column classification, marker detection (55 known markers), summary stats, row text generation, embeddings, ChromaDB & SQLite storage.
  - File size limit: 50 MB.

- cross_reference:
  - Embeds research question, searches lab and literature collections, reranks top chunks, categorizes chunks into agreements/contradictions/explanations/follow-ups, flags novel findings.

- crawl_supplementary:
  - Multi-source discovery (Europe PMC, idconv, PMC oa, eFetch XML), robust parsing for CSV/XLSX/XML/PDF/DOCX/JSON/tar.gz, retry/backoff logic.

- crawl_and_report:
  - Saves full_text.txt, metadata.json, references.json, supplementary files and parsed outputs, and a Markdown report at `data/reports/<pmcid>_report.md`.

- get_next_page:
  - Use to fetch pages when a response is paginated. Sessions TTL: 30 minutes, max concurrent sessions: 50.

---

## Configuration & environment variables

Configuration values are read from the repository configuration module (see `src/oncocontext/` for config sources). Typical settings you may encounter:
- Paths: storage dirs for ChromaDB, SQLite DB file, `data/crawled/`, `data/reports/`
- Model cache locations for embedding and reranker models
- Timeouts and TTLs: PMC cache TTL, pagination session TTL
- Network/timeouts for remote APIs (Europe PMC / PMC / NCBI)

Check `src/oncocontext/` for concrete environment variable names and defaults.

---

## Data model & storage

- SQLite tables: `papers`, `lab_files`, `chunks`, `literature_chunks` mappings, `search_log`, indexing status fields.
- ChromaDB collections: `literature_chunks`, `lab_data`.
- Local crawl tree: `data/crawled/<pmcid>/` and `data/reports/<pmcid>_report.md`.

---

## Typical workflows

- Index a paper:
  1. Call `get_paper_details(pmid, fetch_full_text=True, index_if_available=True)`.
  2. The server fetches BioC, parses sections, chunks, embeds, and stores chunks.

- Ingest a lab file:
  1. Call `ingest_lab_file(file_path="path/to/file.csv", experiment_label="Exp1")`.
  2. File is parsed locally, embedded, stored, and a `LabFileInfo` summary is returned.

- Compare lab vs literature:
  1. Ensure relevant papers are indexed.
  2. Call `cross_reference(research_question="...", lab_file_ids=[...])`.
  3. Receive structured agreements, contradictions, novel findings, and suggested follow-ups.

- Crawl a paper and produce an evidence report:
  1. Call `crawl_and_report(pmcid_or_pmid="PMCXXXXXX", clinical_question="...")`.
  2. Outputs are saved under `data/crawled/<pmcid>/` and `data/reports/`.

---

## Logging & Pagination

- All tool responses pass through `_paginate()` middleware. Responses over ~900 KB are split; use `get_next_page(session_id, page)` to retrieve subsequent pages. Pagination sessions expire after 30 minutes.
- Search queries and important operations are logged to SQLite tables (e.g., `search_log`).

---

## Dependencies

Major runtime dependencies include (see `requirements.txt` for exact versions):
- pandas, numpy
- ChromaDB (or the repository's vector store dependency)
- transformers / sentence-transformers (for embedding & reranker models)
- pdfminer.six / pypdf, python-docx, lxml
- sqlite3 (standard library) and SQL helper libs
- fastmcp (MCP runtime used by the server)

---

## Development & testing

- See `src/oncocontext/` for service and tool implementations.
- Add unit tests around services (query expansion, BioC parsing, CSV parsing, chunking).
- Run the MCP server locally and exercise tools using an MCP client that connects over stdio.

---

## Privacy & security

- Lab file ingestion is local-only by design — ingested lab data is not sent to external APIs.
- When fetching external content (PMC / Europe PMC), the server only retrieves publicly available resources.

---

## Contributing

Contributions are welcome. Suggested workflow:
- Fork → feature branch → PR with tests and documentation updates.
- Keep model downloads cached and avoid committing large binary artifacts.

---

## License

Specify license (add a LICENSE file). If the repo already contains a license, follow that.

---

For implementation details, review:
- Tools: `src/oncocontext/tools/*.py`
- Services: `src/oncocontext/services/*.py`
- Storage: `src/oncocontext/storage/*.py`
- Models/schemas: `src/oncocontext/models/schemas.py`
