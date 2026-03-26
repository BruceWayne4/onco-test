"""Query expansion using synonym_dict.json.

Expands user queries with oncology-specific synonyms for improved recall.
Implements tiered query building, stop-word filtering, field-tag inference,
intent classification, and result-aware relaxation helpers.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from oncocontext.config import settings
from oncocontext.models.schemas import QueryExpansion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop-words — never AND these into a PubMed query
# ---------------------------------------------------------------------------

STOP_WORDS: frozenset[str] = frozenset({
    "what", "is", "the", "of", "for", "in", "a", "an", "and", "or",
    "with", "does", "how", "are", "do", "can", "used", "treat",
    "treatment", "role", "effect", "effects", "on", "at", "by",
    "between", "among", "about", "its", "their", "this", "that",
    "which", "who", "when", "where", "why", "be", "been", "was",
    "were", "has", "have", "had", "not", "no", "vs", "versus",
    "use", "using", "study", "studies", "patients", "clinical",
})

# ---------------------------------------------------------------------------
# Intent templates — slots: {drug}, {cancer}, {gene}, {drug_a}, {drug_b}, {terms}
# ---------------------------------------------------------------------------

QUERY_TEMPLATES: dict[str, str] = {
    "drug_mechanism": (
        '({drug}) AND ("mechanism of action"[Title/Abstract] OR "pharmacology"[MeSH Terms])'
    ),
    "drug_efficacy": (
        '({drug}) AND ({cancer}) AND '
        '("clinical trial"[Publication Type] OR "efficacy"[Title/Abstract])'
    ),
    "biomarker_assoc": (
        '({gene}) AND ({cancer}) AND '
        '("biomarker"[Title/Abstract] OR "prognosis"[MeSH Terms])'
    ),
    "treatment_compare": (
        '({drug_a}) AND ({drug_b}) AND ({cancer}) AND "comparative study"[Publication Type]'
    ),
    "general": "{terms}",
}

# Regex-based intent signals
_INTENT_SIGNALS: list[tuple[str, re.Pattern[str]]] = [
    ("drug_mechanism", re.compile(
        r'\b(mechanism|how does|pharmacology|mode of action)\b', re.I
    )),
    ("drug_efficacy", re.compile(
        r'\b(efficacy|response rate|clinical trial|outcome|survival)\b', re.I
    )),
    ("biomarker_assoc", re.compile(
        r'\b(biomarker|expression|prognosis|prognostic|predict)\b', re.I
    )),
    ("treatment_compare", re.compile(
        r'\b(compare|versus|vs\.?|combination|combined with)\b', re.I
    )),
]

# Drug name suffix pattern
_DRUG_SUFFIX_RE = re.compile(
    r'(mab|nib|zumab|tinib|ciclib|lisib|rafenib|tuzumab|ximab|'
    r'lumab|simab|denib|gefitinib|erlotinib|imatinib)$',
    re.I,
)

# Gene/biomarker pattern — all-caps alphanumeric, 2–12 chars, may include hyphens
_GENE_RE = re.compile(r'^[A-Z][A-Z0-9\-]{1,11}$')

# RELAX_STEPS — ordered transformations applied to a PubMed query string
# Each step returns a (possibly simpler) query string
RELAX_STEPS: list = [
    lambda q: q,                                                          # step 0: original
    lambda q: re.sub(r'\[MeSH Terms\]', '[Title/Abstract]', q),          # step 1: loosen field
    lambda q: re.sub(r'\[Supplementary Concept\]', '[Title/Abstract]', q),  # step 2: drug field
    lambda q: re.sub(r' AND "[^"]+"', '', q, count=1),                   # step 3: drop 1 clause
    # step 4: keep first TWO AND-clauses — never collapse to a single bare keyword.
    # A single keyword (e.g. "tumor") returns millions of irrelevant noise results.
    lambda q: " AND ".join(q.split(" AND ")[:2]) if len(q.split(" AND ")) >= 2 else q,
]

# Maximum number of AND-chained clause groups in Tier-1 / Tier-2 queries.
# Chaining 8+ AND conditions for niche multi-concept queries reliably returns 0.
# The most-specific synonym groups are kept; generic trailing words are dropped.
_MAX_TIER1_CLAUSES = 4
_MAX_TIER2_CLAUSES = 3


class QueryExpander:
    """Expand queries using the oncology synonym dictionary.

    Loads synonym_dict.json (~50+ entries) and expands recognized terms
    in the user's query with their synonyms for PubMed search.

    Key improvements over the naive AND-everything approach:
    - Stop-word filtering: common English function words are not AND-chained
    - Field-tag inference: drug names → [Supplementary Concept],
      gene symbols → [Title/Abstract], other → [MeSH Terms]
    - Tiered query building: Tier 1 (full+expanded), Tier 2 (core[Title/Abstract]),
      Tier 3 (dominant entity[MeSH Terms])
    - Intent classification: routes to appropriate PubMed query template
    """

    def __init__(self, synonym_dict_path: str | None = None) -> None:
        """Initialize the query expander.

        Args:
            synonym_dict_path: Path to synonym_dict.json.
                Defaults to resources/synonym_dict.json.
        """
        self._synonyms: dict[str, list[str]] = {}
        self._lower_to_original: dict[str, str] = {}
        path = Path(synonym_dict_path) if synonym_dict_path else settings.SYNONYM_DICT_PATH
        self._load_synonyms(path)

    def _load_synonyms(self, path: Path) -> None:
        """Load synonym dictionary from JSON file."""
        try:
            with open(path, encoding="utf-8") as f:
                raw: dict[str, list[str]] = json.load(f)
            self._synonyms = raw
            # Build case-insensitive lookup: lowercase key → original key
            self._lower_to_original = {k.lower(): k for k in raw}
            logger.info("Loaded %d synonym entries from %s", len(raw), path)
        except FileNotFoundError:
            logger.warning("Synonym dictionary not found at %s — no expansions available", path)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse synonym dictionary at %s: %s", path, exc)

    # ── Public API ────────────────────────────────────────────────────────────

    async def expand(self, query: str, intent: str = "auto") -> dict:
        """Expand a natural language query with synonyms.

        Args:
            query: Original user query.
            intent: Query intent hint. One of 'auto', 'drug_mechanism',
                'drug_efficacy', 'biomarker_assoc', 'treatment_compare',
                'general'. When 'auto', intent is inferred from the query text.

        Returns:
            Dict with:
                - original_query (str)
                - expanded_terms (list[str]): Synonyms added
                - pubmed_query (str): Tier-1 PubMed query string
                - query_tiers (list[str]): [tier1, tier2, tier3] queries
        """
        if not query or not query.strip():
            return QueryExpansion(
                original_query=query or "",
                expanded_terms=[],
                pubmed_query=query or "",
            ).model_dump()

        expansion = self._expand_query(query.strip(), intent=intent)
        return expansion.model_dump()

    def expand_sync(self, query: str, intent: str = "auto") -> QueryExpansion:
        """Synchronous version of expand — returns a QueryExpansion model."""
        if not query or not query.strip():
            return QueryExpansion(
                original_query=query or "",
                expanded_terms=[],
                pubmed_query=query or "",
            )
        return self._expand_query(query.strip(), intent=intent)

    def build_query_tiers(self, query: str, intent: str = "auto") -> list[str]:
        """Build a funnel of 3 progressively relaxed PubMed queries.

        Tier 1 — Full expanded query with MeSH/field tags (most precise)
        Tier 2 — Core concepts only, all tagged [Title/Abstract]
        Tier 3 — Single dominant entity tagged [MeSH Terms] (broadest)

        Args:
            query: Original user query string.
            intent: Query intent hint (see expand()).

        Returns:
            List of three PubMed query strings, from most to least precise.
        """
        expansion = self._expand_query(query.strip(), intent=intent)
        tiers: list[str] = [expansion.pubmed_query]

        # Tier 2: core concepts (original terms, no synonyms) → [Title/Abstract]
        # Cap at _MAX_TIER2_CLAUSES to prevent zero-result over-specification.
        core_terms = self._extract_core_terms(query)
        if core_terms:
            tier2 = " AND ".join(
                f"{self._quote_if_multiword(t)}[Title/Abstract]"
                for t in core_terms[:_MAX_TIER2_CLAUSES]
            )
        else:
            tier2 = expansion.pubmed_query
        tiers.append(tier2)

        # Tier 3: single dominant entity → [Title/Abstract]
        # IMPORTANT: Do NOT use [MeSH Terms] here.  Raw query words and
        # synonym-dict keys are almost never valid MeSH descriptors
        # (e.g. "co-culture" is not a MeSH term — "Coculture Techniques" is).
        # [Title/Abstract] works universally for any term the author may have
        # written without requiring a maintained normalisation table.
        dominant = self._extract_dominant_entity(query)
        tier3 = f"{self._quote_if_multiword(dominant)}[Title/Abstract]"
        tiers.append(tier3)

        return tiers

    def classify_intent(self, query: str) -> str:
        """Classify query intent into a known template key.

        Args:
            query: Original user query string.

        Returns:
            Intent key: 'drug_mechanism', 'drug_efficacy', 'biomarker_assoc',
            'treatment_compare', or 'general'.
        """
        for intent_key, pattern in _INTENT_SIGNALS:
            if pattern.search(query):
                return intent_key
        return "general"

    # ── Internal Logic ────────────────────────────────────────────────────────

    def _expand_query(self, query: str, intent: str = "auto") -> QueryExpansion:
        """Core expansion logic.

        Algorithm:
        1. Classify intent (auto or provided)
        2. Try to match multi-word phrases first (longest match first)
        3. Then match remaining single-word tokens
        4. Filter stop-words from unmatched gap text
        5. Apply field tags based on term type
        6. Build PubMed query with OR groups for expanded terms, AND between groups
        7. Optionally apply intent-based template
        """
        all_expanded_terms: list[str] = []
        expansions: dict[str, list[str]] = {}

        # Resolve intent
        effective_intent = self.classify_intent(query) if intent == "auto" else intent

        # Sort synonym keys by length descending so multi-word phrases match first
        sorted_keys = sorted(self._synonyms.keys(), key=len, reverse=True)
        sorted_keys_lower = [k.lower() for k in sorted_keys]

        query_lower = query.lower()
        matched_spans: list[tuple[int, int]] = []

        # Phase 1: find all phrase matches (multi-word first, longest first)
        for key, key_lower in zip(sorted_keys, sorted_keys_lower):
            # Use word boundary matching to avoid partial matches
            pattern = re.compile(r'\b' + re.escape(key_lower) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(query_lower):
                start, end = match.start(), match.end()
                # Check no overlap with already matched spans
                if not any(s <= start < e or s < end <= e for s, e in matched_spans):
                    matched_spans.append((start, end))
                    original_term = query[start:end]
                    synonyms = self._synonyms[key]
                    expansions[original_term] = synonyms
                    all_expanded_terms.extend(synonyms)

        # Sort matched spans by position
        matched_spans.sort(key=lambda x: x[0])

        # Phase 2: build PubMed query parts
        query_groups: list[str] = []
        prev_end = 0

        for start, end in matched_spans:
            # Add any unmatched text between spans — filter stop-words
            gap = query[prev_end:start].strip()
            if gap:
                for word in gap.split():
                    clean = word.strip('.,;:!?()"\'')
                    if clean and clean.lower() not in STOP_WORDS and len(clean) > 2:
                        query_groups.append(self._quote_if_multiword(clean))

            # Add the expanded group with appropriate field tag
            original_term = query[start:end]
            synonyms = expansions[original_term]
            group = self._build_or_group(original_term, synonyms)
            query_groups.append(group)
            prev_end = end

        # Add any trailing unmatched text — filter stop-words
        trailing = query[prev_end:].strip()
        if trailing:
            for word in trailing.split():
                clean = word.strip('.,;:!?()"\'')
                if clean and clean.lower() not in STOP_WORDS and len(clean) > 2:
                    query_groups.append(self._quote_if_multiword(clean))

        # If nothing was matched at all, use the original query
        if not query_groups:
            pubmed_query = query
        else:
            # Cap at _MAX_TIER1_CLAUSES to prevent zero-result over-specification.
            # For niche multi-concept queries, chaining 8+ AND conditions returns
            # 0 results. Keep only the most-specific groups (synonym-dict matches
            # appear first because matched_spans is processed in position order,
            # so the lead concepts are retained).
            pubmed_query = " AND ".join(query_groups[:_MAX_TIER1_CLAUSES])

        # Apply intent-based template if not general
        if effective_intent != "general" and effective_intent in QUERY_TEMPLATES:
            pubmed_query = self._apply_intent_template(
                effective_intent, pubmed_query, expansions, query
            )

        return QueryExpansion(
            original_query=query,
            expanded_terms=all_expanded_terms,
            pubmed_query=pubmed_query,
        )

    def _build_or_group(self, original: str, synonyms: list[str]) -> str:
        """Build a PubMed OR group with field tags: (term[tag] OR syn1[tag] ...).

        Field tag is inferred from the original term type:
        - Drug names  → [Supplementary Concept]
        - Gene/marker → [Title/Abstract]
        - Other       → [MeSH Terms]
        """
        field = self._infer_field_tag(original)
        parts = [f"{self._quote_if_multiword(original)}{field}"]
        for syn in synonyms:
            parts.append(f"{self._quote_if_multiword(syn)}{field}")
        return "(" + " OR ".join(parts) + ")"

    def _infer_field_tag(self, term: str) -> str:
        """Return PubMed field tag based on term characteristics.

        Strategy:
        - Drug names (ends in -mab, -nib, etc.) → [Supplementary Concept]
        - Gene/biomarker symbols (ALL-CAPS alphanumeric) → [Title/Abstract]
        - Well-established cancer/biology terms → [MeSH Terms]
        """
        if self._is_drug(term):
            return "[Supplementary Concept]"
        if self._is_gene(term):
            return "[Title/Abstract]"
        return "[MeSH Terms]"

    @staticmethod
    def _is_drug(term: str) -> bool:
        """True if the term looks like a pharmaceutical drug name."""
        return bool(_DRUG_SUFFIX_RE.search(term))

    @staticmethod
    def _is_gene(term: str) -> bool:
        """True if the term looks like a gene symbol or biomarker abbreviation.

        Matches patterns like: PD-1, EGFR, KRAS, TIM-3, CD8, HER2
        """
        # Remove hyphens/numbers for pure letter check
        stripped = re.sub(r'[-0-9+]', '', term).upper()
        # Must be mostly uppercase letters (gene-like)
        return bool(_GENE_RE.match(term.upper())) and stripped.isalpha() and len(stripped) >= 2

    def _extract_core_terms(self, query: str) -> list[str]:
        """Extract core clinical/biological terms from query, stripping stop-words.

        Used for Tier-2 query building — keeps only terms that are likely
        to be indexed in PubMed Title/Abstract fields.

        Returns list of terms in original case, filtering stop-words and
        short tokens.
        """
        core: list[str] = []
        query_lower = query.lower()
        matched_spans: list[tuple[int, int]] = []

        # Prefer matched phrase spans (they are meaningful concepts)
        sorted_keys = sorted(self._synonyms.keys(), key=len, reverse=True)
        for key in sorted_keys:
            pattern = re.compile(r'\b' + re.escape(key.lower()) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(query_lower):
                start, end = match.start(), match.end()
                if not any(s <= start < e or s < end <= e for s, e in matched_spans):
                    matched_spans.append((start, end))
                    core.append(query[start:end])

        matched_spans.sort(key=lambda x: x[0])

        # Add unmatched non-stop words
        prev_end = 0
        for start, end in matched_spans:
            gap = query[prev_end:start].strip()
            for word in gap.split():
                clean = word.strip('.,;:!?()"\'')
                if clean and clean.lower() not in STOP_WORDS and len(clean) > 2:
                    core.append(clean)
            prev_end = end
        trailing = query[prev_end:].strip()
        for word in trailing.split():
            clean = word.strip('.,;:!?()"\'')
            if clean and clean.lower() not in STOP_WORDS and len(clean) > 2:
                core.append(clean)

        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for t in core:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                result.append(t)
        return result

    def _extract_dominant_entity(self, query: str) -> str:
        """Extract the single most important entity from the query.

        Priority order:
        1. First matched drug name (ends in -mab/-nib)
        2. First matched gene symbol
        3. First matched synonym-dict term
        4. Longest non-stop word

        Used for Tier-3 (broadest) fallback query.
        """
        query_lower = query.lower()

        # Check synonym dict hits — prefer drug names, then genes, then others
        sorted_keys = sorted(self._synonyms.keys(), key=len, reverse=True)
        drug_match: str | None = None
        gene_match: str | None = None
        other_match: str | None = None

        for key in sorted_keys:
            pattern = re.compile(r'\b' + re.escape(key.lower()) + r'\b', re.IGNORECASE)
            m = pattern.search(query_lower)
            if m:
                original_term = query[m.start():m.end()]
                if self._is_drug(original_term) and drug_match is None:
                    drug_match = original_term
                elif self._is_gene(original_term) and gene_match is None:
                    gene_match = original_term
                elif other_match is None:
                    other_match = original_term

        if drug_match:
            return drug_match
        if gene_match:
            return gene_match

        # Scan raw query tokens for gene-like patterns not in synonym dict
        for word in query.split():
            clean = word.strip('.,;:!?()"\'')
            if (
                clean
                and clean.lower() not in STOP_WORDS
                and self._is_gene(clean)
                and not self._is_drug(clean)
            ):
                return clean

        if other_match:
            return other_match

        # Fallback: longest non-stop word in query
        words = [
            w.strip('.,;:!?()"\'') for w in query.split()
            if w.strip('.,;:!?()"\'').lower() not in STOP_WORDS
            and len(w.strip('.,;:!?()"\'')) > 2
        ]
        if words:
            return max(words, key=len)
        return query.split()[0] if query.split() else query

    def _apply_intent_template(
        self,
        intent: str,
        base_query: str,
        expansions: dict[str, list[str]],
        original_query: str,
    ) -> str:
        """Apply an intent-based template to the query.

        Slots drug/cancer/gene entities from the expansions dict into the
        template. Falls back to the base_query if slots can't be filled.
        """
        template = QUERY_TEMPLATES.get(intent, "{terms}")

        # Identify matched entities by type
        drugs: list[str] = []
        genes: list[str] = []
        others: list[str] = []
        for term in expansions:
            if self._is_drug(term):
                drugs.append(term)
            elif self._is_gene(term):
                genes.append(term)
            else:
                others.append(term)

        def _or_group(terms: list[str]) -> str:
            if not terms:
                return ""
            return "(" + " OR ".join(self._quote_if_multiword(t) for t in terms) + ")"

        try:
            if intent == "drug_mechanism" and drugs:
                return template.format(drug=_or_group(drugs))
            if intent == "drug_efficacy" and drugs and others:
                return template.format(drug=_or_group(drugs), cancer=_or_group(others))
            if intent == "biomarker_assoc" and genes and others:
                return template.format(gene=_or_group(genes), cancer=_or_group(others))
            if intent == "treatment_compare" and len(drugs) >= 2 and others:
                return template.format(
                    drug_a=_or_group(drugs[:1]),
                    drug_b=_or_group(drugs[1:2]),
                    cancer=_or_group(others),
                )
        except (KeyError, IndexError):
            pass

        # Could not fill slots — fall back to base_query
        return base_query

    @staticmethod
    def _quote_if_multiword(term: str) -> str:
        """Wrap multi-word terms in quotes for PubMed."""
        if " " in term or "-" in term:
            # If already quoted, don't double-quote
            if term.startswith('"') and term.endswith('"'):
                return term
            return f'"{term}"'
        return term
