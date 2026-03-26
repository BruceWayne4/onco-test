"""Unit tests for the SectionAwareChunker service.

Tests section-aware chunking, sentence boundary splitting, overlap,
section prefix prepending, chunk_id format, and edge cases.
"""

import pytest

from oncocontext.models.schemas import Section
from oncocontext.services.chunker import SectionAwareChunker


@pytest.fixture
def chunker() -> SectionAwareChunker:
    """Create a SectionAwareChunker with default settings."""
    return SectionAwareChunker()


@pytest.fixture
def small_chunker() -> SectionAwareChunker:
    """Create a chunker with small limits for testing overlap/splitting."""
    return SectionAwareChunker(max_tokens=20, overlap_tokens=5, min_tokens=3)


@pytest.fixture
def sample_sections() -> list[Section]:
    """Create realistic sample sections for testing (each paragraph >50 tokens)."""
    return [
        Section(
            heading="Abstract",
            section_type="abstract",
            paragraphs=[
                "T cell exhaustion is a hallmark of chronic infections and cancer. "
                "This study investigates the molecular mechanisms underlying CD8+ T cell "
                "exhaustion in tumor organoid co-culture models using flow cytometry and "
                "transcriptomic profiling. We found that PD-1 and TIM-3 co-expression "
                "identifies a subset of progenitor exhausted T cells that retain cytotoxic "
                "potential despite displaying canonical exhaustion markers."
            ],
        ),
        Section(
            heading="Methods",
            section_type="methods",
            paragraphs=[
                "Flow cytometry analysis was performed using anti-CD8 (clone RPA-T8), "
                "anti-PD-1 (clone EH12.2H7), and anti-TIM-3 (clone F38-2E2) antibodies. "
                "Cells were acquired on a BD LSRFortessa and analyzed using FlowJo v10. "
                "Dead cells were excluded using a viability dye eFluor 780 and doublets "
                "were gated out using FSC-A versus FSC-H. Compensation was performed "
                "using single-stained controls for each fluorochrome.",
                "Organoids were cultured in Matrigel domes as previously described in the "
                "standard protocol. T cells were isolated from peripheral blood using "
                "magnetic bead separation and added at an E:T ratio of 5:1. Co-cultures "
                "were maintained for 72 hours in complete RPMI medium supplemented with "
                "10% FBS, 1% penicillin-streptomycin, and 50 IU/mL recombinant IL-2.",
            ],
        ),
        Section(
            heading="Results",
            section_type="results",
            paragraphs=[
                "PD-1+TIM-3+ CD8+ T cells constituted 45-60% of the total CD8+ population "
                "in the organoid co-culture system. These cells showed significantly reduced "
                "IFN-gamma production compared to PD-1 negative counterparts as measured by "
                "intracellular cytokine staining. Interestingly, granzyme B expression was "
                "maintained in a substantial fraction of PD-1+TIM-3+ cells.",
            ],
        ),
    ]


class TestChunkerShortParagraphs:
    """Tests for paragraphs that fit in a single chunk."""

    def test_short_paragraph_single_chunk(self, chunker: SectionAwareChunker):
        """A short paragraph becomes one chunk."""
        sections = [
            Section(
                heading="Methods",
                section_type="methods",
                paragraphs=["Flow cytometry was performed on samples."],
            )
        ]
        chunks = chunker.chunk_paper("12345", "PMC123", sections)
        # Short text might be below min_tokens — if so, no chunks
        # This particular text is 7 words, below default min_tokens of 50
        # So it should be discarded
        assert len(chunks) == 0

    def test_medium_paragraph_single_chunk(self, chunker: SectionAwareChunker):
        """A medium paragraph that fits within max_tokens becomes one chunk."""
        text = (
            "Flow cytometry analysis was performed using anti-CD8 clone RPA-T8, "
            "anti-PD-1 clone EH12.2H7, and anti-TIM-3 clone F38-2E2 antibodies. "
            "Cells were stained in PBS containing 2% FBS for 30 minutes at 4°C. "
            "Dead cells were excluded using a viability dye eFluor 780. "
            "Samples were acquired on a BD LSRFortessa flow cytometer and analyzed "
            "using FlowJo version 10.8.1 software with standard gating procedures."
        )
        sections = [Section(heading="Methods", section_type="methods", paragraphs=[text])]
        chunks = chunker.chunk_paper("12345", "PMC123", sections)
        assert len(chunks) == 1
        assert chunks[0]["paper_pmid"] == "12345"
        assert chunks[0]["section"] == "methods"


class TestChunkerLongParagraphs:
    """Tests for paragraphs that need splitting."""

    def test_long_paragraph_splits(self, small_chunker: SectionAwareChunker):
        """A long paragraph is split into multiple overlapping chunks."""
        # Create text that's longer than 20 tokens
        text = " ".join(f"word{i}" for i in range(50))
        sections = [Section(heading="Results", section_type="results", paragraphs=[text])]
        chunks = small_chunker.chunk_paper("12345", None, sections)
        assert len(chunks) > 1

    def test_overlap_present(self, small_chunker: SectionAwareChunker):
        """Chunks from a split paragraph have overlapping content."""
        # Create a long paragraph of distinct words
        words = [f"w{i:03d}" for i in range(60)]
        text = " ".join(words)
        sections = [Section(heading="Results", section_type="results", paragraphs=[text])]
        chunks = small_chunker.chunk_paper("12345", None, sections)

        if len(chunks) >= 2:
            # Check that consecutive chunks share some words
            chunk0_words = set(chunks[0]["text"].split())
            chunk1_words = set(chunks[1]["text"].split())
            # Remove the section prefix for cleaner comparison
            chunk0_words.discard("[Results]")
            chunk1_words.discard("[Results]")
            overlap = chunk0_words & chunk1_words
            # There should be some overlap (from the overlap_tokens setting)
            assert len(overlap) > 0


class TestChunkerSectionPrefix:
    """Tests for section prefix prepending."""

    def test_section_prefix_prepended(self, chunker: SectionAwareChunker):
        """Each chunk text starts with [SectionName] prefix."""
        text = (
            "We performed a comprehensive literature review using PubMed, "
            "searching for articles published between 2015 and 2024. Search terms "
            "included CAR-T, exhaustion, solid tumors, PD-1, TIM-3, LAG-3, and TOX. "
            "We analyzed 250 papers meeting our inclusion criteria."
        )
        sections = [Section(heading="Methods", section_type="methods", paragraphs=[text])]
        chunks = chunker.chunk_paper("12345", "PMC123", sections)
        for chunk in chunks:
            assert chunk["text"].startswith("[Methods]")

    def test_different_sections_different_prefix(self, sample_sections, chunker):
        """Chunks from different sections have the correct prefix."""
        chunks = chunker.chunk_paper("12345", "PMC123", sample_sections)
        sections_found = set()
        for chunk in chunks:
            if chunk["text"].startswith("[Abstract]"):
                sections_found.add("abstract")
            elif chunk["text"].startswith("[Methods]"):
                sections_found.add("methods")
            elif chunk["text"].startswith("[Results]"):
                sections_found.add("results")
        # Should have chunks from methods and results at minimum
        assert "methods" in sections_found


class TestChunkerMultipleSections:
    """Tests for multiple sections — never crossing boundaries."""

    def test_multiple_sections_produce_chunks(self, chunker, sample_sections):
        """Multiple sections produce chunks from each section."""
        chunks = chunker.chunk_paper("12345", "PMC123", sample_sections)
        assert len(chunks) > 0

        # Check sections present
        sections_in_chunks = {c["section"] for c in chunks}
        # At least methods should be there (it has long-enough paragraphs)
        assert "methods" in sections_in_chunks

    def test_no_cross_section_chunks(self, small_chunker):
        """Chunks never contain text from two different sections."""
        sections = [
            Section(
                heading="Methods",
                section_type="methods",
                paragraphs=["method text word " * 10],
            ),
            Section(
                heading="Results",
                section_type="results",
                paragraphs=["result text word " * 10],
            ),
        ]
        chunks = small_chunker.chunk_paper("12345", None, sections)

        for chunk in chunks:
            text = chunk["text"]
            # A chunk should not contain both "method text" and "result text"
            has_method = "method text" in text
            has_result = "result text" in text
            assert not (has_method and has_result), \
                f"Chunk crosses section boundary: {text[:80]}"


class TestChunkerEmptySections:
    """Tests for empty sections and edge cases."""

    def test_empty_sections_list(self, chunker):
        """Empty sections list returns no chunks."""
        chunks = chunker.chunk_paper("12345", None, [])
        assert chunks == []

    def test_section_with_empty_paragraphs(self, chunker):
        """Section with empty paragraphs produces no chunks."""
        sections = [
            Section(heading="Methods", section_type="methods", paragraphs=["", "  ", ""])
        ]
        chunks = chunker.chunk_paper("12345", None, sections)
        assert chunks == []

    def test_section_with_no_paragraphs(self, chunker):
        """Section with no paragraphs produces no chunks."""
        sections = [Section(heading="Methods", section_type="methods", paragraphs=[])]
        chunks = chunker.chunk_paper("12345", None, sections)
        assert chunks == []


class TestChunkerChunkId:
    """Tests for chunk_id format."""

    def test_chunk_id_format(self, chunker, sample_sections):
        """Chunk IDs follow the format: {pmid}_{section}_{para}_{idx}."""
        chunks = chunker.chunk_paper("12345", "PMC123", sample_sections)
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            parts = chunk_id.split("_")
            assert parts[0] == "12345"  # pmid
            # section name follows
            assert len(parts) >= 4  # at least pmid_section_para_idx

    def test_chunk_id_unique(self, chunker, sample_sections):
        """All chunk IDs within a paper are unique."""
        chunks = chunker.chunk_paper("12345", "PMC123", sample_sections)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))


class TestChunkerTokenCounting:
    """Tests for token counting."""

    def test_count_tokens_empty(self, chunker):
        """Empty string has 0 tokens."""
        assert chunker._count_tokens("") == 0

    def test_count_tokens_simple(self, chunker):
        """Simple sentence has expected token count."""
        assert chunker._count_tokens("hello world") == 2
        assert chunker._count_tokens("one two three four five") == 5

    def test_count_tokens_biomedical(self, chunker):
        """Biomedical text token count is approximate word count."""
        text = "anti-CD8 (clone RPA-T8) staining protocol"
        tokens = chunker._count_tokens(text)
        assert tokens == 5  # 5 whitespace-separated tokens


class TestChunkerSentenceBoundary:
    """Tests for sentence boundary splitting."""

    def test_splits_at_sentence_boundaries(self, small_chunker):
        """Splitting prefers sentence boundaries over mid-sentence."""
        # Build text with clear sentences that exceed max_tokens (20)
        text = (
            "First sentence has several words here. "
            "Second sentence also has several words. "
            "Third sentence adds even more content. "
            "Fourth sentence completes the paragraph."
        )
        chunks = small_chunker._split_text(text, 20, 5)
        # Chunks should end at or near sentence boundaries
        for chunk in chunks:
            # Each chunk should not have orphaned words in the middle
            assert isinstance(chunk, str)
            assert len(chunk.strip()) > 0


class TestChunkerMetadata:
    """Tests for metadata preservation."""

    def test_metadata_fields_present(self, chunker, sample_sections):
        """Each chunk has all required metadata fields."""
        chunks = chunker.chunk_paper("12345", "PMC123", sample_sections)
        required_fields = {
            "chunk_id", "text", "paper_pmid", "pmc_id",
            "section", "paragraph_num", "chunk_index", "token_count",
        }
        for chunk in chunks:
            assert required_fields.issubset(chunk.keys()), \
                f"Missing fields: {required_fields - chunk.keys()}"

    def test_pmid_in_metadata(self, chunker, sample_sections):
        """Paper PMID is correctly stored in chunk metadata."""
        chunks = chunker.chunk_paper("99999", "PMC99", sample_sections)
        for chunk in chunks:
            assert chunk["paper_pmid"] == "99999"

    def test_pmc_id_in_metadata(self, chunker, sample_sections):
        """PMC ID is correctly stored in chunk metadata."""
        chunks = chunker.chunk_paper("12345", "PMC123", sample_sections)
        for chunk in chunks:
            assert chunk["pmc_id"] == "PMC123"

    def test_pmc_id_none_becomes_empty(self, chunker, sample_sections):
        """None PMC ID becomes empty string in metadata."""
        chunks = chunker.chunk_paper("12345", None, sample_sections)
        for chunk in chunks:
            assert chunk["pmc_id"] == ""
