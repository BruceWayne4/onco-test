"""FullPaperCrawler — fetch complete paper content and persist it locally.

Orchestrates existing services (PMCClient, PubMedClient, BioCParser,
SupplementaryCrawler) and saves all content to data/crawled/<pmcid>/:

    full_text.txt            — complete paper text, section by section
    metadata.json            — title, authors, journal, DOI, abstract, MeSH, date
    references.json          — cited references with PMIDs/PMCIDs and local paths
    supplementary/           — each supplementary file saved with original filename
    supplementary_index.json — index of all supplementary files with URLs/local paths

All writes use UTF-8 with errors='replace'.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from oncocontext.config import settings
from oncocontext.services.bioc_parser import BioCParser
from oncocontext.services.pmc_client import PMCClient
from oncocontext.services.pubmed_client import PubMedClient
from oncocontext.services.supplementary_crawler import SupplementaryCrawler
from oncocontext.storage.cache_manager import CacheManager

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_CRAWLED_DIR = settings.DATA_DIR / "crawled"
_USER_AGENT = "OncoContext/1.0 (research tool; mailto:research@example.com)"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _safe_filename(name: str) -> str:
    """Sanitise a string so it can be used as a filename."""
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip()


def _write_text(path: Path, text: str) -> None:
    """Write *text* to *path* (UTF-8, replacing unmappable characters)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def _write_json(path: Path, data: Any) -> None:
    """Serialise *data* to pretty-printed JSON and write to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
        errors="replace",
    )


def _extract_references_from_bioc(bioc_json: dict) -> list[dict]:
    """Pull reference passages from a BioC JSON document.

    BioCParser skips passages with section_type="REF", so we extract them
    directly from the raw JSON structure.

    Args:
        bioc_json: Raw BioC JSON dict returned by PMCClient.fetch_bioc().

    Returns:
        List of reference dicts, each with keys: ref_id, text, pmid, pmc_id, doi.
    """
    if not bioc_json:
        return []

    bioc_data = bioc_json[0] if isinstance(bioc_json, list) else bioc_json
    documents = bioc_data.get("documents", [])
    if not documents:
        return []

    doc = documents[0]
    passages = doc.get("passages", [])

    references: list[dict] = []

    for passage in passages:
        infons_raw = passage.get("infons", {})
        if isinstance(infons_raw, list):
            infons: dict = {
                item.get("key", ""): item.get("value", "")
                for item in infons_raw
                if isinstance(item, dict)
            }
        elif isinstance(infons_raw, dict):
            infons = infons_raw
        else:
            infons = {}

        section_type = infons.get("section_type", "").upper()
        if section_type not in ("REF", "REFERENCE", "REFERENCES"):
            continue

        text = passage.get("text", "").strip()
        if not text:
            continue

        # Try to extract identifiers from annotations
        pmid: str | None = None
        pmc_id: str | None = None
        doi: str | None = None

        for ann in passage.get("annotations", []):
            ann_infons_raw = ann.get("infons", {})
            if isinstance(ann_infons_raw, list):
                ann_infons = {
                    item.get("key", ""): item.get("value", "")
                    for item in ann_infons_raw
                    if isinstance(item, dict)
                }
            elif isinstance(ann_infons_raw, dict):
                ann_infons = ann_infons_raw
            else:
                ann_infons = {}

            ann_type = ann_infons.get("type", "").upper()
            if ann_type in ("PMID",):
                pmid = ann.get("text") or ann_infons.get("value")
            elif ann_type in ("PMCID", "PMC"):
                pmc_id = ann.get("text") or ann_infons.get("value")
            elif ann_type in ("DOI",):
                doi = ann.get("text") or ann_infons.get("value")

        references.append({
            "ref_id": infons.get("ref-type", "") or f"ref_{len(references)+1}",
            "text": text,
            "pmid": pmid,
            "pmc_id": pmc_id,
            "doi": doi,
            "local_path": None,  # filled in later if paper is also crawled
        })

    return references


# ── Main Class ─────────────────────────────────────────────────────────────────


class FullPaperCrawler:
    """Orchestrate full paper fetching and local persistence.

    Reuses existing services and adds structured file output to
    data/crawled/<pmcid>/ so the content can be read offline without
    hitting any API again.

    Usage::

        crawler = FullPaperCrawler()
        result  = await crawler.crawl("PMC8650059")
        # — or —
        result  = await crawler.crawl("34789550")   # PMID
    """

    def __init__(self) -> None:
        cache = CacheManager()
        self._pubmed = PubMedClient(cache=cache)
        self._pmc = PMCClient(cache=cache)
        self._parser = BioCParser()
        self._supp_crawler = SupplementaryCrawler()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def crawl(
        self,
        pmcid_or_pmid: str,
        crawl_supplementary: bool = True,
        max_supp_files: int = 20,
    ) -> dict:
        """Fetch a paper and save all content locally.

        Args:
            pmcid_or_pmid: Either a PMC ID (e.g. "PMC8650059") or a
                           PubMed ID (e.g. "34789550").
            crawl_supplementary: Whether to also fetch and save supplementary
                                 files. Default True.
            max_supp_files: Maximum number of supplementary files to fetch.

        Returns:
            Dict with keys:
                pmid, pmc_id, local_data_path, full_text_path,
                metadata_path, references_path, supplementary_index_path,
                supplementary_index (dict), references (list),
                sections_found (list[str]), total_chars_crawled (int),
                errors (list[str]).
        """
        errors: list[str] = []

        # ── Step 1: resolve IDs ───────────────────────────────────────────────
        input_upper = pmcid_or_pmid.strip().upper()
        if input_upper.startswith("PMC"):
            pmc_id: str | None = input_upper
            pmid: str | None = None
        else:
            pmid = pmcid_or_pmid.strip()
            pmc_id = None

        logger.info("FullPaperCrawler.crawl: input=%r → pmid=%s, pmc_id=%s",
                    pmcid_or_pmid, pmid, pmc_id)

        # ── Step 2: fetch PubMed metadata ─────────────────────────────────────
        metadata: dict = {}
        if pmid:
            try:
                papers = await self._pubmed.fetch_details([pmid])
                if papers:
                    metadata = papers[0]
                    if not pmc_id and metadata.get("pmc_id"):
                        pmc_id = metadata["pmc_id"]
                    logger.info("PubMed metadata fetched for PMID %s", pmid)
            except Exception as exc:
                msg = f"PubMed fetch failed for PMID {pmid}: {exc}"
                errors.append(msg)
                logger.warning(msg)

        # If only PMC ID was given, we may not have a PMID yet; extract from metadata
        if pmc_id and not pmid and metadata.get("pmid"):
            pmid = metadata["pmid"]

        # ── Step 3: fetch BioC full text ──────────────────────────────────────
        bioc_json: dict | None = None
        sections_text: list[tuple[str, str]] = []  # [(heading, text), ...]
        references: list[dict] = []

        if pmc_id:
            try:
                bioc_json = await self._pmc.fetch_bioc(pmc_id)
                if bioc_json:
                    logger.info("BioC JSON fetched for %s", pmc_id)
                    parsed_sections = self._parser.parse(bioc_json)
                    for sec in parsed_sections:
                        body = "\n\n".join(sec.paragraphs)
                        sections_text.append((sec.heading, body))
                    references = _extract_references_from_bioc(bioc_json)
                    logger.info(
                        "Parsed %d sections, %d references from BioC",
                        len(sections_text), len(references),
                    )
                    # If metadata not yet populated, pull title from sections
                    if not metadata.get("title"):
                        for heading, body in sections_text:
                            if heading.lower() == "title":
                                metadata["title"] = body.strip()
                                break
                else:
                    msg = f"BioC JSON not available for {pmc_id} (not in Open Access subset)"
                    errors.append(msg)
                    logger.info(msg)
            except Exception as exc:
                msg = f"BioC fetch/parse failed for {pmc_id}: {exc}"
                errors.append(msg)
                logger.warning(msg)
        else:
            msg = "No PMC ID available — cannot fetch full text (not in Open Access subset)"
            errors.append(msg)
            logger.info(msg)

        # ── Step 4: determine output directory ────────────────────────────────
        dir_key = pmc_id or (f"PMID{pmid}" if pmid else "UNKNOWN")
        out_dir = _CRAWLED_DIR / _safe_filename(dir_key)
        out_dir.mkdir(parents=True, exist_ok=True)
        supp_dir = out_dir / "supplementary"
        supp_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Output directory: %s", out_dir)

        # ── Step 5: save full_text.txt ────────────────────────────────────────
        total_chars = 0
        full_text_lines: list[str] = []

        if sections_text:
            for heading, body in sections_text:
                full_text_lines.append(f"{'='*60}")
                full_text_lines.append(f"SECTION: {heading.upper()}")
                full_text_lines.append(f"{'='*60}")
                full_text_lines.append(body)
                full_text_lines.append("")
            full_text_content = "\n".join(full_text_lines)
        else:
            # Fallback: save abstract only
            abstract = metadata.get("abstract", "")
            full_text_content = (
                f"SECTION: ABSTRACT\n{'='*60}\n{abstract}\n"
                if abstract
                else "(no full text available)\n"
            )

        total_chars += len(full_text_content)
        full_text_path = out_dir / "full_text.txt"
        _write_text(full_text_path, full_text_content)
        logger.info("Saved full_text.txt (%d chars)", len(full_text_content))

        # ── Step 6: save metadata.json ────────────────────────────────────────
        meta_payload = {
            "pmid": pmid or metadata.get("pmid"),
            "pmc_id": pmc_id,
            "title": metadata.get("title", ""),
            "authors": metadata.get("authors", []),
            "journal": metadata.get("journal", ""),
            "year": metadata.get("year", 0),
            "doi": metadata.get("doi"),
            "abstract": metadata.get("abstract", ""),
            "mesh_terms": metadata.get("mesh_terms", []),
            "has_full_text": bool(sections_text),
            "sections_found": [h for h, _ in sections_text],
        }
        total_chars += len(json.dumps(meta_payload))
        metadata_path = out_dir / "metadata.json"
        _write_json(metadata_path, meta_payload)
        logger.info("Saved metadata.json")

        # ── Step 7: save references.json ──────────────────────────────────────
        refs_chars = len(json.dumps(references))
        total_chars += refs_chars
        references_path = out_dir / "references.json"
        _write_json(references_path, references)
        logger.info("Saved references.json (%d references)", len(references))

        # ── Step 8: crawl & save supplementary files ──────────────────────────
        supp_index: dict[str, dict] = {}

        if crawl_supplementary:
            try:
                supp_result = await self._supp_crawler.crawl(
                    pmid=pmid,
                    pmc_id=pmc_id,
                    max_files=max_supp_files,
                )
                supp_files = supp_result.get("supplementary_files", [])
                logger.info("Supplementary crawler found %d files", len(supp_files))

                for sf in supp_files:
                    fname = _safe_filename(sf.get("filename") or "unnamed")
                    if not fname:
                        fname = "unnamed_supp"
                    local_path = supp_dir / fname

                    # Serialise content to a local file
                    content = sf.get("content", {})
                    if "error" in content:
                        # Save error marker
                        _write_text(
                            local_path.with_suffix(".error.txt"),
                            f"Error fetching {sf.get('url','')}: {content['error']}\n",
                        )
                        local_rel = str(local_path.with_suffix(".error.txt").relative_to(out_dir.parent.parent))
                    else:
                        # Save JSON representation of parsed content
                        json_path = local_path.with_suffix(".json")
                        _write_json(json_path, content)
                        # Also save raw text if available
                        if "text" in content:
                            _write_text(local_path.with_suffix(".txt"), content["text"])
                        local_rel = str(json_path.relative_to(out_dir.parent.parent))
                        total_chars += len(json.dumps(content))

                    supp_index[fname] = {
                        "filename": fname,
                        "original_url": sf.get("url", ""),
                        "file_type": sf.get("file_type", ""),
                        "title": sf.get("title"),
                        "local_path": local_rel,
                        "has_error": "error" in content,
                    }

                # Propagate any discovery errors
                for err in supp_result.get("errors", []):
                    errors.append(f"supplementary: {err.get('detail', str(err))}")

            except Exception as exc:
                msg = f"Supplementary crawl failed: {exc}"
                errors.append(msg)
                logger.warning(msg)

        # ── Step 9: save supplementary_index.json ────────────────────────────
        supp_index_path = out_dir / "supplementary_index.json"
        _write_json(supp_index_path, supp_index)
        logger.info("Saved supplementary_index.json (%d files)", len(supp_index))

        # ── Step 10: build return dict ────────────────────────────────────────
        result = {
            "pmid": pmid or meta_payload.get("pmid"),
            "pmc_id": pmc_id,
            "local_data_path": str(out_dir),
            "full_text_path": str(full_text_path),
            "metadata_path": str(metadata_path),
            "references_path": str(references_path),
            "supplementary_index_path": str(supp_index_path),
            "supplementary_index": supp_index,
            "references": references,
            "sections_found": [h for h, _ in sections_text],
            "total_chars_crawled": total_chars,
            "errors": errors,
        }

        logger.info(
            "FullPaperCrawler done: %d sections, %d supp files, %d chars, %d errors",
            len(sections_text), len(supp_index), total_chars, len(errors),
        )
        return result

    async def close(self) -> None:
        """Close underlying HTTP clients."""
        await self._pmc.close()
        await self._pubmed.close()
        await self._supp_crawler.close()
