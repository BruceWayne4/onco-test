"""BioC JSON parser — converts PMC BioC JSON into structured sections with paragraph boundaries.

Handles non-standard section names and malformed data gracefully.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from oncocontext.models.schemas import Section

logger = logging.getLogger(__name__)


class BioCParser:
    """Parse BioC JSON into structured paper sections.

    Extracts sections (TITLE, ABSTRACT, INTRO, METHODS, RESULTS, DISCUSS, CONCL)
    with paragraph boundaries and subsection headings.
    """

    # Map BioC section_type values to standard section names
    SECTION_MAP: dict[str, str] = {
        "TITLE": "title",
        "ABSTRACT": "abstract",
        "INTRO": "introduction",
        "METHODS": "methods",
        "MATERIALS": "methods",
        "RESULTS": "results",
        "DISCUSS": "discussion",
        "CONCL": "conclusion",
        "CASE": "case_report",
        "SUPPL": "supplementary",
    }

    # Section types to skip
    SKIP_SECTIONS: set[str] = {
        "REF", "REFERENCE", "REFERENCES",
        "FIG", "FIGURE",
        "TABLE", "TABLE_CAPTION",
        "ABBR", "ABBREVIATION", "ABBREVIATIONS",
        "AUTH_CONT", "AUTHOR_CONTRIB", "AUTHOR_CONTRIBUTIONS",
        "COMP_INT", "COMPETING_INTERESTS", "CONFLICT",
        "ACK_FUND", "ACKNOWLEDGMENT", "ACKNOWLEDGMENTS", "ACKNOWLEDGEMENTS",
        "FUNDING",
        "KEYWORDS",
        "APPENDIX",
    }

    # Minimum passage length to include (skip short headers/captions)
    MIN_PASSAGE_LENGTH: int = 20

    # Regex patterns for fuzzy section name matching
    _SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
        ("introduction", re.compile(r"intro(?:duction)?", re.IGNORECASE)),
        ("methods", re.compile(r"method|material|experimental|procedure", re.IGNORECASE)),
        ("results", re.compile(r"result|finding", re.IGNORECASE)),
        ("discussion", re.compile(r"discuss", re.IGNORECASE)),
        ("conclusion", re.compile(r"conclu", re.IGNORECASE)),
        ("abstract", re.compile(r"abstract|summary", re.IGNORECASE)),
        ("title", re.compile(r"^title$", re.IGNORECASE)),
        ("supplementary", re.compile(r"suppl|supplement", re.IGNORECASE)),
    ]

    def parse(self, bioc_json: dict) -> list[Section]:
        """Parse BioC JSON into a list of Section objects.

        Args:
            bioc_json: Parsed BioC JSON dict from PMC API.

        Returns:
            List of Section objects with heading, section_type, and paragraphs.

        Raises:
            ValueError: If JSON is completely unparseable.
        """
        if not bioc_json:
            raise ValueError("Empty BioC JSON")

        # Handle both list and dict formats from PMC BioC API
        # Sometimes returns: [{"source": "PMC", "documents": [...]}]
        # Sometimes returns: {"documents": [...]}
        if isinstance(bioc_json, list):
            if not bioc_json:
                raise ValueError("Empty BioC JSON list")
            # Take the first item in the list
            bioc_data = bioc_json[0]
        else:
            bioc_data = bioc_json
        
        # Now get documents from the normalized structure
        documents = bioc_data.get("documents", [])
        if not documents:
            raise ValueError("No documents found in BioC JSON")

        # Take the first document (usually there's only one)
        doc = documents[0]
        passages = doc.get("passages", [])

        if not passages:
            logger.warning("No passages found in BioC document %s", doc.get("id", "unknown"))
            return []

        # Sort passages by offset to ensure correct order
        passages.sort(key=lambda p: p.get("offset", 0))

        # Group passages by normalized section type
        section_groups: dict[str, list[tuple[int, str]]] = defaultdict(list)

        for passage in passages:
            infons_raw = passage.get("infons", {})
            
            # Handle both dict and list formats from PMC BioC API
            # Standard format: {"type": "paragraph", "section_type": "METHODS"}
            # Non-standard format: [{"key": "type", "value": "paragraph"}, ...]
            if isinstance(infons_raw, list):
                infons = {
                    item.get("key", ""): item.get("value", "")
                    for item in infons_raw
                    if isinstance(item, dict)
                }
            elif isinstance(infons_raw, dict):
                infons = infons_raw
            else:
                infons = {}
            
            text = passage.get("text", "").strip()
            offset = passage.get("offset", 0)

            if not text or len(text) < self.MIN_PASSAGE_LENGTH:
                continue

            section_type = infons.get("section_type", "")
            passage_type = infons.get("type", "")

            # Normalize section type
            normalized = self._normalize_section_type(section_type, passage_type)
            if normalized is None:
                continue  # skip this passage

            section_groups[normalized].append((offset, text))

        # If no sections were recognized, put everything in "body"
        if not section_groups:
            all_texts = [
                (p.get("offset", 0), p.get("text", "").strip())
                for p in passages
                if p.get("text", "").strip() and len(p.get("text", "").strip()) >= self.MIN_PASSAGE_LENGTH
            ]
            if all_texts:
                section_groups["body"] = all_texts

        # Convert to Section objects, ordered by first appearance
        sections: list[Section] = []
        # Define a canonical ordering
        section_order = [
            "title", "abstract", "introduction", "methods", "results",
            "discussion", "conclusion", "supplementary", "case_report", "body",
        ]

        for section_name in section_order:
            if section_name not in section_groups:
                continue

            entries = section_groups[section_name]
            # Sort by offset and extract texts
            entries.sort(key=lambda x: x[0])
            paragraphs = [text for _, text in entries]

            sections.append(Section(
                heading=section_name.replace("_", " ").title(),
                section_type=section_name,
                paragraphs=paragraphs,
            ))

        # Add any sections not in the canonical order
        for section_name, entries in section_groups.items():
            if section_name not in section_order:
                entries.sort(key=lambda x: x[0])
                paragraphs = [text for _, text in entries]
                sections.append(Section(
                    heading=section_name.replace("_", " ").title(),
                    section_type=section_name,
                    paragraphs=paragraphs,
                ))

        return sections

    def _normalize_section_type(
        self, section_type: str, passage_type: str = ""
    ) -> str | None:
        """Map BioC section_type to standard name. Return None to skip.

        Args:
            section_type: The section_type from BioC infons.
            passage_type: The type from BioC infons (e.g., 'title', 'paragraph').

        Returns:
            Normalized section name, or None to skip this passage.
        """
        if not section_type:
            # Fall back to passage_type
            if passage_type in ("title", "title_1"):
                return "title"
            elif passage_type in ("abstract", "abstract_title_1"):
                return "abstract"
            elif passage_type == "paragraph":
                return "body"
            return None

        upper = section_type.upper().strip()

        # Check skip list first
        if upper in self.SKIP_SECTIONS:
            return None

        # Check direct mapping
        if upper in self.SECTION_MAP:
            return self.SECTION_MAP[upper]

        # Check partial matches in skip list
        for skip in self.SKIP_SECTIONS:
            if skip in upper or upper in skip:
                return None

        # Try fuzzy matching with regex patterns
        for name, pattern in self._SECTION_PATTERNS:
            if pattern.search(section_type):
                return name

        # Unknown section type — include it as-is (lowercased)
        logger.debug("Unknown BioC section_type: %s", section_type)
        return section_type.lower().replace(" ", "_")
