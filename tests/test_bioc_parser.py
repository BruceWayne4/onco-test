"""Unit tests for the BioCParser service.

Tests BioC JSON parsing, section normalization, edge cases.
"""

import pytest

from oncocontext.services.bioc_parser import BioCParser


@pytest.fixture
def parser() -> BioCParser:
    """Create a BioCParser instance."""
    return BioCParser()


@pytest.fixture
def sample_bioc_json() -> dict:
    """A realistic BioC JSON structure for testing."""
    return {
        "documents": [{
            "id": "PMC1234567",
            "passages": [
                {
                    "infons": {"section_type": "TITLE", "type": "title"},
                    "text": "CAR-T Cell Exhaustion in Solid Tumors: Mechanisms and Therapeutic Strategies",
                    "offset": 0,
                },
                {
                    "infons": {"section_type": "ABSTRACT", "type": "abstract"},
                    "text": "Chimeric antigen receptor T (CAR-T) cell therapy has shown remarkable efficacy in hematological malignancies. However, T cell exhaustion remains a major barrier in solid tumors. This review discusses the molecular mechanisms underlying CAR-T exhaustion and emerging strategies to overcome it.",
                    "offset": 100,
                },
                {
                    "infons": {"section_type": "INTRO", "type": "paragraph"},
                    "text": "T cell exhaustion was first described in the context of chronic viral infections and has since been recognized as a critical limitation of cancer immunotherapy. Exhausted T cells exhibit progressive loss of effector functions, sustained expression of inhibitory receptors, and altered transcriptional programs.",
                    "offset": 500,
                },
                {
                    "infons": {"section_type": "METHODS", "type": "paragraph"},
                    "text": "We performed a comprehensive literature review using PubMed, searching for articles published between 2015 and 2024. Search terms included CAR-T, exhaustion, solid tumors, PD-1, TIM-3, LAG-3, and TOX. We analyzed 250 papers meeting our inclusion criteria.",
                    "offset": 1000,
                },
                {
                    "infons": {"section_type": "METHODS", "type": "paragraph"},
                    "text": "Flow cytometry analysis was performed using anti-CD8 (clone RPA-T8), anti-PD-1 (clone EH12.2H7), and anti-TIM-3 (clone F38-2E2) antibodies. Cells were acquired on a BD LSRFortessa and analyzed using FlowJo v10.",
                    "offset": 1500,
                },
                {
                    "infons": {"section_type": "RESULTS", "type": "paragraph"},
                    "text": "Analysis of tumor-infiltrating lymphocytes revealed that PD-1+TIM-3+ CD8+ T cells constituted 45-60% of the total CD8+ population in solid tumors. These cells showed significantly reduced IFN-gamma and granzyme B production compared to PD-1- counterparts.",
                    "offset": 2000,
                },
                {
                    "infons": {"section_type": "RESULTS", "type": "paragraph"},
                    "text": "TOX expression was found to be the master regulator of the exhaustion program, with knockout studies demonstrating prevention of terminal exhaustion in multiple tumor models.",
                    "offset": 2500,
                },
                {
                    "infons": {"section_type": "DISCUSS", "type": "paragraph"},
                    "text": "Our findings highlight the critical role of TOX-mediated transcriptional programs in driving CAR-T cell exhaustion. The identification of progenitor exhausted T cells as a therapeutic target opens new avenues for combination immunotherapy strategies.",
                    "offset": 3000,
                },
                {
                    "infons": {"section_type": "CONCL", "type": "paragraph"},
                    "text": "CAR-T cell exhaustion in solid tumors is a multifaceted process driven by sustained antigen exposure and the immunosuppressive TME. Targeting TOX and epigenetic reprogramming represent promising strategies.",
                    "offset": 3500,
                },
                {
                    "infons": {"section_type": "REF", "type": "reference"},
                    "text": "1. Wherry EJ. T cell exhaustion. Nat Immunol. 2011;12(6):492-499.",
                    "offset": 4000,
                },
                {
                    "infons": {"section_type": "FIG", "type": "fig_caption"},
                    "text": "Fig 1.",
                    "offset": 4500,
                },
            ],
        }],
    }


class TestBioCParserStandard:
    """Standard BioC JSON parsing tests."""

    def test_parses_all_standard_sections(self, parser: BioCParser, sample_bioc_json: dict):
        """Parser extracts all standard sections from BioC JSON."""
        sections = parser.parse(sample_bioc_json)
        section_types = [s.section_type for s in sections]

        assert "title" in section_types
        assert "abstract" in section_types
        assert "introduction" in section_types
        assert "methods" in section_types
        assert "results" in section_types
        assert "discussion" in section_types
        assert "conclusion" in section_types

    def test_methods_sections_merged(self, parser: BioCParser, sample_bioc_json: dict):
        """Multiple METHODS passages are merged into a single section."""
        sections = parser.parse(sample_bioc_json)
        methods = [s for s in sections if s.section_type == "methods"]
        assert len(methods) == 1
        assert len(methods[0].paragraphs) == 2  # two METHODS passages

    def test_results_sections_merged(self, parser: BioCParser, sample_bioc_json: dict):
        """Multiple RESULTS passages are merged into a single section."""
        sections = parser.parse(sample_bioc_json)
        results = [s for s in sections if s.section_type == "results"]
        assert len(results) == 1
        assert len(results[0].paragraphs) == 2

    def test_references_skipped(self, parser: BioCParser, sample_bioc_json: dict):
        """REF sections are filtered out."""
        sections = parser.parse(sample_bioc_json)
        section_types = [s.section_type for s in sections]
        assert "references" not in section_types

    def test_figures_skipped(self, parser: BioCParser, sample_bioc_json: dict):
        """FIG sections and short captions are filtered out."""
        sections = parser.parse(sample_bioc_json)
        for s in sections:
            for p in s.paragraphs:
                assert p != "Fig 1."

    def test_section_ordering(self, parser: BioCParser, sample_bioc_json: dict):
        """Sections are returned in canonical order."""
        sections = parser.parse(sample_bioc_json)
        section_types = [s.section_type for s in sections]
        expected_order = ["title", "abstract", "introduction", "methods", "results", "discussion", "conclusion"]
        assert section_types == expected_order

    def test_paragraphs_ordered_by_offset(self, parser: BioCParser, sample_bioc_json: dict):
        """Paragraphs within a section are ordered by offset."""
        sections = parser.parse(sample_bioc_json)
        methods = [s for s in sections if s.section_type == "methods"][0]
        # First paragraph should be the literature review one (offset 1000)
        assert "comprehensive literature review" in methods.paragraphs[0]
        # Second should be the flow cytometry one (offset 1500)
        assert "Flow cytometry" in methods.paragraphs[1]


class TestBioCParserSectionNormalization:
    """Tests for section type normalization."""

    def test_normalize_title(self, parser: BioCParser):
        """TITLE normalizes to 'title'."""
        assert parser._normalize_section_type("TITLE") == "title"

    def test_normalize_abstract(self, parser: BioCParser):
        """ABSTRACT normalizes to 'abstract'."""
        assert parser._normalize_section_type("ABSTRACT") == "abstract"

    def test_normalize_materials(self, parser: BioCParser):
        """MATERIALS normalizes to 'methods'."""
        assert parser._normalize_section_type("MATERIALS") == "methods"

    def test_normalize_discuss(self, parser: BioCParser):
        """DISCUSS normalizes to 'discussion'."""
        assert parser._normalize_section_type("DISCUSS") == "discussion"

    def test_normalize_ref_skipped(self, parser: BioCParser):
        """REF is skipped (returns None)."""
        assert parser._normalize_section_type("REF") is None

    def test_normalize_fig_skipped(self, parser: BioCParser):
        """FIG is skipped (returns None)."""
        assert parser._normalize_section_type("FIG") is None

    def test_normalize_unknown_fuzzy_match(self, parser: BioCParser):
        """Fuzzy matching handles non-standard section names."""
        assert parser._normalize_section_type("MATERIAL AND METHODS") == "methods"
        assert parser._normalize_section_type("Introduction") == "introduction"
        assert parser._normalize_section_type("Results and Discussion") == "results"


class TestBioCParserEdgeCases:
    """Edge case tests."""

    def test_empty_json_raises(self, parser: BioCParser):
        """Empty dict raises ValueError."""
        with pytest.raises(ValueError, match="Empty"):
            parser.parse({})

    def test_no_documents_raises(self, parser: BioCParser):
        """JSON with no documents raises ValueError."""
        with pytest.raises(ValueError, match="No documents"):
            parser.parse({"documents": []})

    def test_no_passages_returns_empty(self, parser: BioCParser):
        """Document with no passages returns empty list."""
        bioc = {"documents": [{"id": "test", "passages": []}]}
        sections = parser.parse(bioc)
        assert sections == []

    def test_short_passages_filtered(self, parser: BioCParser):
        """Passages shorter than MIN_PASSAGE_LENGTH are skipped."""
        bioc = {
            "documents": [{
                "id": "test",
                "passages": [
                    {
                        "infons": {"section_type": "TITLE", "type": "title"},
                        "text": "Short",  # Too short
                        "offset": 0,
                    },
                    {
                        "infons": {"section_type": "ABSTRACT", "type": "abstract"},
                        "text": "This is a sufficiently long abstract passage for testing the parser.",
                        "offset": 100,
                    },
                ],
            }],
        }
        sections = parser.parse(bioc)
        section_types = [s.section_type for s in sections]
        assert "title" not in section_types  # too short
        assert "abstract" in section_types

    def test_no_section_type_fallback_to_body(self, parser: BioCParser):
        """Passages without section_type fall back to body section."""
        bioc = {
            "documents": [{
                "id": "test",
                "passages": [
                    {
                        "infons": {"type": "paragraph"},
                        "text": "This is a paragraph with no section_type specified in the infons.",
                        "offset": 0,
                    },
                ],
            }],
        }
        sections = parser.parse(bioc)
        assert len(sections) >= 1
        assert sections[0].section_type == "body"

    def test_heading_capitalization(self, parser: BioCParser, sample_bioc_json: dict):
        """Section headings are title-cased."""
        sections = parser.parse(sample_bioc_json)
        for s in sections:
            # Headings should be title-cased
            assert s.heading[0].isupper()
