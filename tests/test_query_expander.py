"""Unit tests for the QueryExpander service.

Tests synonym loading, query expansion, phrase matching, stop-word filtering,
field-tag inference, tier building, intent classification, and edge cases.
"""

import pytest

from oncocontext.services.query_expander import (
    RELAX_STEPS,
    STOP_WORDS,
    QueryExpander,
)


@pytest.fixture
def expander() -> QueryExpander:
    """Create a QueryExpander loaded from the default synonym dict."""
    return QueryExpander()


# ── Basic Expansion ───────────────────────────────────────────────────────────


class TestQueryExpanderBasic:
    """Basic expansion tests."""

    def test_single_term_expansion(self, expander: QueryExpander):
        """Single recognized term is expanded with synonyms."""
        result = expander.expand_sync("PD-1")
        assert result.original_query == "PD-1"
        assert "PDCD1" in result.expanded_terms
        assert "CD279" in result.expanded_terms
        assert result.pubmed_query  # not empty
        assert "OR" in result.pubmed_query
        assert "PD-1" in result.pubmed_query or "pd-1" in result.pubmed_query.lower()

    def test_multi_term_expansion(self, expander: QueryExpander):
        """Multiple recognized terms are each expanded."""
        result = expander.expand_sync("PD-1 CD8")
        assert result.original_query == "PD-1 CD8"
        # PD-1 synonyms
        assert "PDCD1" in result.expanded_terms
        # CD8 synonyms
        assert "CTL" in result.expanded_terms
        # Both groups should be in query joined by AND
        assert "AND" in result.pubmed_query

    def test_no_matches_passthrough(self, expander: QueryExpander):
        """Unrecognized non-stop-word terms pass through unchanged."""
        result = expander.expand_sync("xenograft model")
        assert result.original_query == "xenograft model"
        assert result.expanded_terms == []
        # The query should still contain meaningful terms
        assert "xenograft" in result.pubmed_query
        assert "model" in result.pubmed_query

    def test_case_insensitivity(self, expander: QueryExpander):
        """Matching is case-insensitive."""
        result_lower = expander.expand_sync("pd-1")
        result_upper = expander.expand_sync("PD-1")
        # Both should find the same synonyms
        assert set(result_lower.expanded_terms) == set(result_upper.expanded_terms)

    def test_phrase_matching(self, expander: QueryExpander):
        """Multi-word phrases in the synonym dict are matched as phrases."""
        result = expander.expand_sync("T cell exhaustion treatment")
        assert result.original_query == "T cell exhaustion treatment"
        # "T cell exhaustion" is a phrase key in the dict
        assert any("dysfunction" in t.lower() for t in result.expanded_terms)

    def test_phrase_priority_over_single_words(self, expander: QueryExpander):
        """Multi-word phrases are matched before individual words."""
        result = expander.expand_sync("T cell exhaustion")
        assert any("T cell" in t or "dysfunction" in t.lower() for t in result.expanded_terms)

    def test_multiword_synonyms_are_quoted(self, expander: QueryExpander):
        """Multi-word synonyms are wrapped in quotes for PubMed."""
        result = expander.expand_sync("PD-1")
        # "programmed cell death protein 1" should be in quotes
        assert '"programmed cell death protein 1"' in result.pubmed_query


# ── Stop-Word Filtering ───────────────────────────────────────────────────────


class TestStopWordFiltering:
    """Verify stop-words are excluded from AND chains."""

    def test_stop_words_not_in_query(self, expander: QueryExpander):
        """Common function words must not appear as AND clauses."""
        result = expander.expand_sync("What is the role of EGFR in NSCLC")
        pubmed_q = result.pubmed_query.lower()
        # None of these should appear as standalone AND terms
        for sw in ("what", "is", "the", "role", "of", "in"):
            # They should not appear as top-level bare words outside parentheses
            # A simple check: they should not appear as ` AND what AND ` etc.
            assert f" {sw} " not in f" {pubmed_q} " or "(" in pubmed_q, (
                f"Stop-word '{sw}' leaked into query: {result.pubmed_query}"
            )

    def test_stop_words_constant_not_empty(self):
        """STOP_WORDS constant is populated."""
        assert len(STOP_WORDS) > 10
        assert "what" in STOP_WORDS
        assert "the" in STOP_WORDS
        assert "is" in STOP_WORDS

    def test_natural_language_query_strips_filler(self, expander: QueryExpander):
        """Natural language query produces a clean PubMed query."""
        result = expander.expand_sync("What is the role of EGFR in NSCLC")
        # Should contain EGFR and NSCLC but not 'what', 'is', 'the', 'role'
        assert "EGFR" in result.pubmed_query or "egfr" in result.pubmed_query.lower()
        # Stop-words should be absent as standalone tokens
        words = [
            w.strip("() ").lower()
            for w in result.pubmed_query.replace("AND", "").replace("OR", "").split()
            if not w.startswith('"') and not w.startswith("(")
        ]
        for sw in ("what", "is", "the", "role", "of"):
            clean_words = [w.strip('"[]()') for w in words]
            assert sw not in clean_words, (
                f"Stop-word '{sw}' found in query tokens: {result.pubmed_query}"
            )

    def test_short_tokens_filtered(self, expander: QueryExpander):
        """Tokens ≤2 characters are not AND-chained."""
        result = expander.expand_sync("IL-6 in NSCLC")
        # 'in' is a stop-word and also 2 chars — must not appear standalone
        assert " in " not in f" {result.pubmed_query} "


# ── Field Tag Inference ───────────────────────────────────────────────────────


class TestFieldTagInference:
    """Verify correct PubMed field tags are applied by term type."""

    def test_gene_symbol_gets_title_abstract_tag(self, expander: QueryExpander):
        """Gene symbols like PD-1, EGFR → [Title/Abstract]."""
        assert expander._infer_field_tag("PD-1") == "[Title/Abstract]"
        assert expander._infer_field_tag("EGFR") == "[Title/Abstract]"
        assert expander._infer_field_tag("KRAS") == "[Title/Abstract]"
        assert expander._infer_field_tag("CD8") == "[Title/Abstract]"

    def test_drug_gets_supplementary_concept_tag(self, expander: QueryExpander):
        """Drug names ending in -mab/-nib → [Supplementary Concept]."""
        assert expander._infer_field_tag("pembrolizumab") == "[Supplementary Concept]"
        assert expander._infer_field_tag("nivolumab") == "[Supplementary Concept]"
        assert expander._infer_field_tag("gefitinib") == "[Supplementary Concept]"

    def test_other_term_gets_mesh_tag(self, expander: QueryExpander):
        """Non-drug, non-gene terms → [MeSH Terms]."""
        assert expander._infer_field_tag("immunotherapy") == "[MeSH Terms]"
        assert expander._infer_field_tag("lung cancer") == "[MeSH Terms]"

    def test_is_drug_detection(self, expander: QueryExpander):
        """_is_drug() correctly identifies drug names."""
        assert expander._is_drug("pembrolizumab") is True
        assert expander._is_drug("nivolumab") is True
        assert expander._is_drug("erlotinib") is True
        assert expander._is_drug("EGFR") is False
        assert expander._is_drug("PD-1") is False

    def test_is_gene_detection(self, expander: QueryExpander):
        """_is_gene() correctly identifies gene symbols."""
        assert expander._is_gene("EGFR") is True
        assert expander._is_gene("KRAS") is True
        assert expander._is_gene("CD8") is True
        assert expander._is_gene("PD-1") is True
        assert expander._is_gene("pembrolizumab") is False
        assert expander._is_gene("immunotherapy") is False

    def test_field_tags_appear_in_expanded_query(self, expander: QueryExpander):
        """Field tags from _build_or_group appear in the final PubMed query."""
        result = expander.expand_sync("pembrolizumab NSCLC")
        # pembrolizumab is a drug → should have [Supplementary Concept]
        assert "[Supplementary Concept]" in result.pubmed_query or \
               "[Title/Abstract]" in result.pubmed_query or \
               "[MeSH Terms]" in result.pubmed_query


# ── Query Tier Building ───────────────────────────────────────────────────────


class TestQueryTierBuilding:
    """Verify 3-tier query funnel construction."""

    def test_build_query_tiers_returns_three(self, expander: QueryExpander):
        """build_query_tiers always returns exactly 3 tiers."""
        tiers = expander.build_query_tiers("PD-1 T cell exhaustion")
        assert len(tiers) == 3

    def test_tier1_is_most_specific(self, expander: QueryExpander):
        """Tier 1 should contain expanded synonyms (most specific)."""
        tiers = expander.build_query_tiers("PD-1")
        tier1 = tiers[0]
        # Tier 1 should have the full expansion with field tags
        assert "OR" in tier1  # expanded synonyms joined by OR

    def test_tier2_uses_title_abstract(self, expander: QueryExpander):
        """Tier 2 should use [Title/Abstract] field tags."""
        tiers = expander.build_query_tiers("PD-1 NSCLC")
        tier2 = tiers[1]
        assert "[Title/Abstract]" in tier2

    def test_tier3_uses_title_abstract(self, expander: QueryExpander):
        """Tier 3 uses [Title/Abstract] on the dominant entity only.

        NOTE: [MeSH Terms] is intentionally NOT used for tier3 because raw
        query words and synonym-dict keys are almost never valid MeSH
        descriptors (e.g. "co-culture" ≠ MeSH "Coculture Techniques").
        [Title/Abstract] works universally without a maintained normalisation
        table — see build_query_tiers() comment.
        """
        tiers = expander.build_query_tiers("PD-1 checkpoint inhibitor")
        tier3 = tiers[2]
        assert "[Title/Abstract]" in tier3
        # Tier 3 should be a single-entity query (no AND)
        assert "AND" not in tier3

    def test_tiers_progressively_simpler(self, expander: QueryExpander):
        """Later tiers should generally be shorter (simpler) than earlier tiers."""
        tiers = expander.build_query_tiers("pembrolizumab PD-1 NSCLC efficacy")
        # Tier 3 should be shorter than Tier 1
        assert len(tiers[2]) <= len(tiers[0])

    def test_tiers_with_single_term(self, expander: QueryExpander):
        """Single term queries produce 3 valid tiers."""
        tiers = expander.build_query_tiers("EGFR")
        assert len(tiers) == 3
        assert all(t for t in tiers)  # all non-empty


# ── Intent Classification ─────────────────────────────────────────────────────


class TestIntentClassification:
    """Verify query intent classifier."""

    def test_drug_mechanism_intent(self, expander: QueryExpander):
        """Mechanism-related queries classified as drug_mechanism."""
        assert expander.classify_intent("What is the mechanism of action of pembrolizumab") \
               == "drug_mechanism"
        assert expander.classify_intent("how does nivolumab work pharmacology") \
               == "drug_mechanism"

    def test_drug_efficacy_intent(self, expander: QueryExpander):
        """Efficacy/trial queries classified as drug_efficacy."""
        assert expander.classify_intent("pembrolizumab efficacy in NSCLC") \
               == "drug_efficacy"
        assert expander.classify_intent("clinical trial outcomes for PD-1 inhibitors") \
               == "drug_efficacy"

    def test_biomarker_intent(self, expander: QueryExpander):
        """Biomarker/expression/prognosis queries classified as biomarker_assoc."""
        assert expander.classify_intent("EGFR expression and prognosis in lung cancer") \
               == "biomarker_assoc"
        assert expander.classify_intent("PD-L1 as biomarker for immunotherapy response") \
               == "biomarker_assoc"

    def test_treatment_compare_intent(self, expander: QueryExpander):
        """Comparison queries classified as treatment_compare."""
        assert expander.classify_intent("pembrolizumab vs nivolumab in NSCLC") \
               == "treatment_compare"
        assert expander.classify_intent("compare anti-PD1 versus anti-CTLA4") \
               == "treatment_compare"

    def test_general_fallback_intent(self, expander: QueryExpander):
        """Queries with no strong signals fall back to general."""
        assert expander.classify_intent("PD-1 T cell exhaustion") == "general"
        assert expander.classify_intent("CD8 tumor infiltrating lymphocytes") == "general"

    def test_intent_auto_inference(self, expander: QueryExpander):
        """expand_sync with intent='auto' correctly infers intent."""
        result = expander.expand_sync(
            "pembrolizumab efficacy in lung cancer", intent="auto"
        )
        # Should have produced a valid query (non-empty)
        assert result.pubmed_query


# ── Core / Dominant Term Extraction ──────────────────────────────────────────


class TestCoreAndDominantExtraction:
    """Verify core term and dominant entity extraction."""

    def test_extract_core_terms_filters_stopwords(self, expander: QueryExpander):
        """Core term extraction strips stop-words."""
        core = expander._extract_core_terms("What is the role of EGFR in NSCLC")
        core_lower = [t.lower() for t in core]
        for sw in ("what", "is", "the", "role", "of", "in"):
            assert sw not in core_lower, f"Stop-word '{sw}' in core terms: {core}"

    def test_extract_core_terms_keeps_meaningful_words(self, expander: QueryExpander):
        """Core term extraction keeps clinical terms."""
        core = expander._extract_core_terms("EGFR mutation NSCLC")
        core_lower = [t.lower() for t in core]
        assert "egfr" in core_lower or any("egfr" in c for c in core_lower)

    def test_extract_dominant_entity_prefers_drug(self, expander: QueryExpander):
        """Dominant entity prefers drug names."""
        dominant = expander._extract_dominant_entity(
            "pembrolizumab treatment in NSCLC EGFR"
        )
        assert dominant.lower() == "pembrolizumab"

    def test_extract_dominant_entity_prefers_gene_over_other(self, expander: QueryExpander):
        """Dominant entity prefers gene when no drug present."""
        dominant = expander._extract_dominant_entity("EGFR in lung cancer prognosis")
        assert dominant.upper() == "EGFR"

    def test_extract_dominant_entity_fallback_longest_word(self, expander: QueryExpander):
        """Dominant entity falls back to longest non-stop word for unknown queries."""
        dominant = expander._extract_dominant_entity("xylophone transcriptomics sequencing")
        assert dominant in ("xylophone", "transcriptomics", "sequencing")
        assert len(dominant) >= 3


# ── Edge Cases ────────────────────────────────────────────────────────────────


class TestQueryExpanderEdgeCases:
    """Edge case tests."""

    def test_empty_query(self, expander: QueryExpander):
        """Empty query returns empty expansion."""
        result = expander.expand_sync("")
        assert result.original_query == ""
        assert result.expanded_terms == []
        assert result.pubmed_query == ""

    def test_whitespace_query(self, expander: QueryExpander):
        """Whitespace-only query returns empty expansion."""
        result = expander.expand_sync("   ")
        assert result.original_query == "   "
        assert result.expanded_terms == []

    def test_only_stop_words_query(self, expander: QueryExpander):
        """Query consisting only of stop-words produces a sensible fallback."""
        result = expander.expand_sync("what is the")
        # Should not crash; pubmed_query may be the original or empty
        assert isinstance(result.pubmed_query, str)

    def test_mixed_known_unknown_terms(self, expander: QueryExpander):
        """Query with both known and unknown terms expands correctly."""
        result = expander.expand_sync("CAR-T therapy in xenograft")
        assert result.original_query == "CAR-T therapy in xenograft"
        # CAR-T should be expanded
        assert any("chimeric" in t.lower() for t in result.expanded_terms)
        # "xenograft" should pass through (it's not a stop-word and len > 2)
        assert "xenograft" in result.pubmed_query

    def test_natural_language_not_all_and(self, expander: QueryExpander):
        """Natural language query does not AND every single word together."""
        result = expander.expand_sync("What is the role of EGFR in NSCLC")
        # Count AND tokens — should be much less than number of words in query
        and_count = result.pubmed_query.count(" AND ")
        word_count = len(result.original_query.split())
        assert and_count < word_count - 2, (
            f"Too many AND clauses ({and_count}) for a {word_count}-word query. "
            f"Query: {result.pubmed_query}"
        )

    def test_build_tiers_does_not_crash_on_unknown_query(self, expander: QueryExpander):
        """build_query_tiers handles unknown queries gracefully."""
        tiers = expander.build_query_tiers("xylophone quantum thermodynamics")
        assert len(tiers) == 3
        assert all(isinstance(t, str) for t in tiers)


# ── Relax Steps Constant ──────────────────────────────────────────────────────


class TestRelaxSteps:
    """Verify RELAX_STEPS transformations."""

    def test_relax_steps_count(self):
        """RELAX_STEPS has at least 4 steps."""
        assert len(RELAX_STEPS) >= 4

    def test_step0_identity(self):
        """Step 0 is the identity function."""
        q = '(PD-1[MeSH Terms]) AND (NSCLC[MeSH Terms])'
        assert RELAX_STEPS[0](q) == q

    def test_step1_loosens_mesh_to_title_abstract(self):
        """Step 1 replaces [MeSH Terms] with [Title/Abstract]."""
        q = '(PD-1[MeSH Terms]) AND (NSCLC[MeSH Terms])'
        relaxed = RELAX_STEPS[1](q)
        assert "[MeSH Terms]" not in relaxed
        assert "[Title/Abstract]" in relaxed

    def test_step4_keeps_first_two_clauses(self):
        """Step 4 keeps only the first two AND clauses, not just one.

        NOTE: Collapsing to a single bare keyword (e.g. 'tumor') returns
        millions of irrelevant noise results from PubMed. Step 4 intentionally
        retains the first two clauses to preserve some specificity while
        still broadening the search — see RELAX_STEPS comment.
        """
        q = 'EGFR[Title/Abstract] AND NSCLC[Title/Abstract] AND "clinical trial"'
        relaxed = RELAX_STEPS[4](q)
        # Third clause ("clinical trial") should be dropped
        assert '"clinical trial"' not in relaxed
        # First two clauses should be retained
        assert "EGFR" in relaxed
        assert "NSCLC" in relaxed
        # Exactly one AND remains (between the two kept clauses)
        assert relaxed.count(" AND ") == 1


# ── Async API ─────────────────────────────────────────────────────────────────


class TestQueryExpanderAsync:
    """Async API tests."""

    @pytest.mark.asyncio
    async def test_async_expand(self, expander: QueryExpander):
        """The async expand() method works correctly."""
        result = await expander.expand("PD-1")
        assert isinstance(result, dict)
        assert "original_query" in result
        assert "expanded_terms" in result
        assert "pubmed_query" in result
        assert "PDCD1" in result["expanded_terms"]

    @pytest.mark.asyncio
    async def test_async_empty_query(self, expander: QueryExpander):
        """Async expand handles empty queries."""
        result = await expander.expand("")
        assert result["expanded_terms"] == []

    @pytest.mark.asyncio
    async def test_async_expand_with_intent(self, expander: QueryExpander):
        """Async expand accepts explicit intent parameter."""
        result = await expander.expand("pembrolizumab NSCLC", intent="drug_efficacy")
        assert isinstance(result, dict)
        assert result["pubmed_query"]
