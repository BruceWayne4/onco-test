"""ReportWriter — assemble a comprehensive Markdown clinical evidence report.

Loads locally crawled data from data/crawled/<pmcid>/ and writes a
full, non-truncated Markdown report to data/reports/<pmcid>_report.md.

Report structure:
    # Clinical Evidence Report: <title>
    ## Query
    ## Paper Metadata
    ## Abstract
    ## Full Text Sections  (one ## subheading per section)
    ## Key Data Tables     (extracted from supplementary)
    ## Supplementary Materials Index
    ## References
    ## Local File Index    (all saved files with relative paths)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from oncocontext.config import settings

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_CRAWLED_DIR = settings.DATA_DIR / "crawled"
_REPORTS_DIR = settings.DATA_DIR / "reports"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict | list | None:
    """Load JSON from *path*; return None on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        logger.warning("Could not load JSON from %s: %s", path, exc)
        return None


def _load_text(path: Path) -> str:
    """Load plain text from *path*; return empty string on any error."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Could not load text from %s: %s", path, exc)
        return ""


def _md_escape(text: str) -> str:
    """Minimally escape pipe characters for Markdown table safety."""
    return text.replace("|", "\\|")


def _render_table_from_dict(content: dict) -> str:
    """Render a parsed CSV/Excel content dict as a Markdown table.

    Handles the structure returned by SupplementaryCrawler:
    - CSV/TSV: {"columns": [...], "rows": [...]}
    - Excel:   {"sheets": {"Sheet1": {"columns": [...], "rows": [...]}, ...}}
    """
    lines: list[str] = []

    # Excel multi-sheet
    sheets = content.get("sheets")
    if sheets and isinstance(sheets, dict):
        for sheet_name, sheet_data in sheets.items():
            lines.append(f"**Sheet: {sheet_name}**\n")
            lines.extend(_render_simple_table(sheet_data))
            lines.append("")
        return "\n".join(lines)

    # CSV/TSV single table
    if "columns" in content and "rows" in content:
        lines.extend(_render_simple_table(content))
        return "\n".join(lines)

    # BioC sections list
    if "sections" in content and isinstance(content["sections"], list):
        for sec in content["sections"]:
            section_name = sec.get("section", "")
            text = sec.get("text", "")
            if text:
                lines.append(f"*Section: {section_name}*")
                lines.append("")
                lines.append(text)
                lines.append("")
        return "\n".join(lines)

    # Plain text fallback
    if "text" in content:
        return content["text"]

    # Generic: render as JSON code block
    return f"```json\n{json.dumps(content, indent=2, ensure_ascii=False, default=str)[:4000]}\n```"


def _render_simple_table(table_data: dict) -> list[str]:
    """Convert a {'columns': [...], 'rows': [...]} dict to Markdown table lines."""
    columns = table_data.get("columns", [])
    rows = table_data.get("rows", [])

    if not columns or not rows:
        return ["*(empty table)*"]

    # Clamp column count to avoid unwieldy tables
    max_cols = 20
    display_cols = [str(c) for c in columns[:max_cols]]
    has_extra_cols = len(columns) > max_cols

    header = "| " + " | ".join(_md_escape(c) for c in display_cols) + (" | …" if has_extra_cols else " |")
    separator = "| " + " | ".join(["---"] * len(display_cols)) + (" | ---" if has_extra_cols else " |")

    lines = [header, separator]
    for row in rows:
        if isinstance(row, dict):
            cells = [_md_escape(str(row.get(c, ""))) for c in columns[:max_cols]]
        elif isinstance(row, list):
            cells = [_md_escape(str(v)) for v in row[:max_cols]]
        else:
            cells = [_md_escape(str(row))]
        row_line = "| " + " | ".join(cells) + (" | …" if has_extra_cols else " |")
        lines.append(row_line)

    row_count = table_data.get("row_count", len(rows))
    truncated = table_data.get("truncated", False)
    if truncated:
        lines.append(f"\n*… {row_count} total rows (table truncated at {len(rows)} rows)*")
    else:
        lines.append(f"\n*{row_count} rows*")

    return lines


# ── Main Class ─────────────────────────────────────────────────────────────────


class ReportWriter:
    """Assemble and write a comprehensive clinical evidence Markdown report.

    Loads all data previously saved by FullPaperCrawler from
    data/crawled/<pmcid>/ and produces a single non-truncated Markdown
    report at data/reports/<pmcid>_report.md.
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    def write_report(
        self,
        pmcid_or_pmid: str,
        crawl_result: dict,
        clinical_question: str | None = None,
    ) -> str:
        """Build and save the Markdown report.

        Args:
            pmcid_or_pmid: The identifier used to crawl (PMC ID or PMID).
            crawl_result: The dict returned by FullPaperCrawler.crawl().
            clinical_question: Optional clinical question framing the report.

        Returns:
            The path (str) to the written Markdown report file.
        """
        # ── Resolve which directory to read from ──────────────────────────────
        pmc_id: str | None = crawl_result.get("pmc_id")
        pmid: str | None = crawl_result.get("pmid")
        dir_key = pmc_id or (f"PMID{pmid}" if pmid else pmcid_or_pmid)

        local_data_path_str = crawl_result.get("local_data_path")
        if local_data_path_str:
            out_dir = Path(local_data_path_str)
        else:
            out_dir = _CRAWLED_DIR / dir_key

        # ── Load persisted data ───────────────────────────────────────────────
        metadata: dict = {}
        references: list = []
        supp_index: dict = {}

        metadata_path = Path(crawl_result.get("metadata_path", str(out_dir / "metadata.json")))
        references_path = Path(crawl_result.get("references_path", str(out_dir / "references.json")))
        supp_index_path = Path(crawl_result.get("supplementary_index_path", str(out_dir / "supplementary_index.json")))
        full_text_path = Path(crawl_result.get("full_text_path", str(out_dir / "full_text.txt")))

        meta_loaded = _load_json(metadata_path)
        if isinstance(meta_loaded, dict):
            metadata = meta_loaded

        refs_loaded = _load_json(references_path)
        if isinstance(refs_loaded, list):
            references = refs_loaded

        supp_loaded = _load_json(supp_index_path)
        if isinstance(supp_loaded, dict):
            supp_index = supp_loaded

        full_text = _load_text(full_text_path)

        # ── Build the report ──────────────────────────────────────────────────
        sections: list[str] = []

        title = metadata.get("title") or "(untitled)"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # ─ Title ─────────────────────────────────────────────────────────────
        sections.append(f"# Clinical Evidence Report: {title}\n")
        sections.append(f"*Generated: {now}*\n")
        sections.append("---\n")

        # ─ Query ─────────────────────────────────────────────────────────────
        sections.append("## Query\n")
        if clinical_question:
            sections.append(f"> {clinical_question}\n")
        else:
            sections.append(f"> *(No specific clinical question provided — full paper content below)*\n")
        sections.append("")

        # ─ Paper Metadata ─────────────────────────────────────────────────────
        sections.append("## Paper Metadata\n")
        authors = metadata.get("authors", [])
        authors_str = ", ".join(authors) if authors else "—"
        mesh_terms = metadata.get("mesh_terms", [])
        mesh_str = ", ".join(mesh_terms) if mesh_terms else "—"

        sections.append("| Field | Value |")
        sections.append("| --- | --- |")
        sections.append(f"| **PMID** | {metadata.get('pmid', pmid or '—')} |")
        sections.append(f"| **PMC ID** | {metadata.get('pmc_id', pmc_id or '—')} |")
        sections.append(f"| **Title** | {_md_escape(title)} |")
        sections.append(f"| **Authors** | {_md_escape(authors_str)} |")
        sections.append(f"| **Journal** | {_md_escape(metadata.get('journal', '—'))} |")
        sections.append(f"| **Year** | {metadata.get('year', '—')} |")
        sections.append(f"| **DOI** | {metadata.get('doi') or '—'} |")
        sections.append(f"| **MeSH Terms** | {_md_escape(mesh_str)} |")
        sections.append(f"| **Full Text Available** | {'Yes' if metadata.get('has_full_text') else 'No'} |")
        sections.append(f"| **Sections Found** | {', '.join(metadata.get('sections_found', crawl_result.get('sections_found', [])))} |")
        sections.append("")

        # ─ Abstract ───────────────────────────────────────────────────────────
        sections.append("## Abstract\n")
        abstract = metadata.get("abstract", "")
        if abstract:
            sections.append(abstract)
        else:
            sections.append("*(Abstract not available)*")
        sections.append("")

        # ─ Full Text Sections ─────────────────────────────────────────────────
        sections.append("## Full Text Sections\n")
        if full_text and full_text.strip() and "(no full text available)" not in full_text:
            # Parse the section blocks from the saved file
            current_section_heading: str | None = None
            current_section_lines: list[str] = []

            for line in full_text.splitlines():
                if line.startswith("SECTION: "):
                    # Flush previous section
                    if current_section_heading is not None:
                        sections.append(f"### {current_section_heading}\n")
                        sections.append("\n".join(current_section_lines).strip())
                        sections.append("")
                    current_section_heading = line[len("SECTION: "):].strip().title()
                    current_section_lines = []
                elif line.startswith("=" * 10):
                    # separator line — skip
                    continue
                else:
                    current_section_lines.append(line)

            # Flush last section
            if current_section_heading is not None and current_section_lines:
                sections.append(f"### {current_section_heading}\n")
                sections.append("\n".join(current_section_lines).strip())
                sections.append("")
        else:
            sections.append("*(Full text not available — abstract only)*\n")

        # ─ Key Data Tables ────────────────────────────────────────────────────
        sections.append("## Key Data Tables\n")
        table_sections: list[str] = []

        for fname, supp_info in supp_index.items():
            if supp_info.get("has_error"):
                continue
            local_path_str = supp_info.get("local_path", "")
            if not local_path_str:
                continue

            # Only render tabular formats
            file_type = supp_info.get("file_type", "").lower()
            if file_type not in ("csv", "tsv", "xlsx", "xls", "json", "xml"):
                continue

            # Try to load the .json representation
            # The local_path is relative to data/, so we reconstruct the abs path
            json_candidates = [
                settings.DATA_DIR / local_path_str,
                Path(local_path_str),
                out_dir / "supplementary" / (fname + ".json"),
            ]
            content: dict | None = None
            for candidate in json_candidates:
                loaded = _load_json(candidate)
                if loaded is not None and isinstance(loaded, dict):
                    content = loaded
                    break

            if content is None:
                continue

            table_title = supp_info.get("title") or fname
            table_section_lines = [f"### {table_title}\n"]
            table_section_lines.append(
                f"*Source file: `{fname}` — type: `{file_type}`*\n"
            )
            table_section_lines.append(_render_table_from_dict(content))
            table_section_lines.append("")
            table_sections.append("\n".join(table_section_lines))

        if table_sections:
            sections.extend(table_sections)
        else:
            sections.append("*(No tabular supplementary data found)*\n")

        # ─ Supplementary Materials Index ─────────────────────────────────────
        sections.append("## Supplementary Materials Index\n")
        if supp_index:
            sections.append("| Filename | Type | Title | URL | Local Path | Status |")
            sections.append("| --- | --- | --- | --- | --- | --- |")
            for fname, info in supp_index.items():
                status = "⚠ Error" if info.get("has_error") else "✓ OK"
                url = info.get("original_url", "")
                url_md = f"[link]({url})" if url else "—"
                sections.append(
                    f"| `{_md_escape(fname)}` "
                    f"| {_md_escape(info.get('file_type',''))} "
                    f"| {_md_escape(info.get('title') or '—')} "
                    f"| {url_md} "
                    f"| `{_md_escape(info.get('local_path',''))}` "
                    f"| {status} |"
                )
        else:
            sections.append("*(No supplementary files found)*")
        sections.append("")

        # ─ References ─────────────────────────────────────────────────────────
        sections.append("## References\n")
        if references:
            for i, ref in enumerate(references, 1):
                ref_text = ref.get("text", "")
                ref_pmid = ref.get("pmid")
                ref_pmc = ref.get("pmc_id")
                ref_doi = ref.get("doi")
                ids_parts: list[str] = []
                if ref_pmid:
                    ids_parts.append(f"PMID: [{ref_pmid}](https://pubmed.ncbi.nlm.nih.gov/{ref_pmid}/)")
                if ref_pmc:
                    ids_parts.append(f"PMC: [{ref_pmc}](https://www.ncbi.nlm.nih.gov/pmc/articles/{ref_pmc}/)")
                if ref_doi:
                    ids_parts.append(f"DOI: [{ref_doi}](https://doi.org/{ref_doi})")
                ids_str = " | ".join(ids_parts) if ids_parts else ""
                sections.append(f"{i}. {ref_text}")
                if ids_str:
                    sections.append(f"   *{ids_str}*")
                sections.append("")
        else:
            sections.append("*(References not extracted from full text — full text may not be available)*")
        sections.append("")

        # ─ Local File Index ───────────────────────────────────────────────────
        sections.append("## Local File Index\n")
        sections.append("All files saved locally during this crawl:\n")
        sections.append("| File | Description |")
        sections.append("| --- | --- |")

        file_descriptions = {
            "full_text.txt": "Complete paper text, section by section",
            "metadata.json": "Paper metadata (title, authors, journal, DOI, abstract, MeSH, date)",
            "references.json": "All cited references with PMIDs/PMCIDs",
            "supplementary_index.json": "Index of all supplementary files with URLs and local paths",
        }

        if out_dir.exists():
            for fpath in sorted(out_dir.rglob("*")):
                if fpath.is_file():
                    rel = fpath.relative_to(out_dir)
                    desc = file_descriptions.get(fpath.name, "Supplementary content file")
                    sections.append(f"| `{_md_escape(str(rel))}` | {desc} |")

        sections.append("")

        # ─ Crawl Errors ──────────────────────────────────────────────────────
        errors = crawl_result.get("errors", [])
        if errors:
            sections.append("## Crawl Notices\n")
            for err in errors:
                sections.append(f"- {err}")
            sections.append("")

        # ── Write report ──────────────────────────────────────────────────────
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_key = dir_key.replace("/", "_").replace("\\", "_")
        report_path = _REPORTS_DIR / f"{safe_key}_report.md"

        full_report = "\n".join(sections)
        report_path.write_text(full_report, encoding="utf-8", errors="replace")
        logger.info(
            "ReportWriter: saved report to %s (%d chars)", report_path, len(full_report)
        )
        return str(report_path)
