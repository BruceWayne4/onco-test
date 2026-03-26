"""Section-aware chunker — 384-token chunks with 64-token overlap.

Respects paragraph and section boundaries to produce semantically coherent chunks.
Each chunk is prefixed with [SectionName] for embedding context.
"""

from __future__ import annotations

import logging
import re

from oncocontext.config import settings
from oncocontext.models.schemas import Section

logger = logging.getLogger(__name__)

# Sentence-ending pattern: split at '. ', '? ', '! ' etc.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class SectionAwareChunker:
    """Chunk paper sections into 384-token segments with 64-token overlap.

    Properties:
        - Section-aware: never crosses section boundaries
        - Paragraph-aware: prefers paragraph boundaries for splits
        - Metadata preserved: each chunk tagged with section, paragraph_num, chunk_index
        - Min chunk size: 50 tokens (shorter chunks discarded)
        - Prepends [SectionName] prefix to each chunk text for embedding context
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
        min_tokens: int | None = None,
    ) -> None:
        """Initialize chunker.

        Args:
            max_tokens: Maximum tokens per chunk. Defaults to settings.CHUNK_SIZE (384).
            overlap_tokens: Overlap between consecutive chunks. Defaults to settings.CHUNK_OVERLAP (64).
            min_tokens: Minimum chunk size (discard shorter). Defaults to settings.CHUNK_MIN_TOKENS (50).
        """
        self.max_tokens = max_tokens if max_tokens is not None else settings.CHUNK_SIZE
        self.overlap_tokens = overlap_tokens if overlap_tokens is not None else settings.CHUNK_OVERLAP
        self.min_tokens = min_tokens if min_tokens is not None else settings.CHUNK_MIN_TOKENS

    def chunk_paper(
        self,
        pmid: str,
        pmc_id: str | None,
        sections: list[Section],
    ) -> list[dict]:
        """Chunk a parsed paper into embedding-ready segments.

        For each section, for each paragraph:
        - If paragraph fits within max_tokens, it becomes one chunk
        - If paragraph exceeds max_tokens, split into overlapping chunks at sentence boundaries
        - Each chunk is prefixed with [SectionName] for embedding context

        Args:
            pmid: PubMed ID of the paper.
            pmc_id: PMC ID if available.
            sections: List of parsed Section objects.

        Returns:
            List of chunk dicts with: chunk_id, text, paper_pmid, pmc_id,
            section, paragraph_num, chunk_index, token_count.
        """
        all_chunks: list[dict] = []

        for section in sections:
            section_name = section.section_type or section.heading.lower()
            section_label = section.heading or section_name.replace("_", " ").title()

            for para_idx, paragraph in enumerate(section.paragraphs):
                if not paragraph or not paragraph.strip():
                    continue

                # Split the paragraph into chunks
                text_chunks = self._split_text(
                    paragraph.strip(),
                    self.max_tokens,
                    self.overlap_tokens,
                )

                for chunk_idx, chunk_text in enumerate(text_chunks):
                    # Prepend section context
                    prefixed_text = f"[{section_label}] {chunk_text}"
                    token_count = self._count_tokens(prefixed_text)

                    # Skip chunks below minimum size
                    if token_count < self.min_tokens:
                        continue

                    chunk_id = f"{pmid}_{section_name}_{para_idx}_{chunk_idx}"

                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "text": prefixed_text,
                        "paper_pmid": pmid,
                        "pmc_id": pmc_id or "",
                        "section": section_name,
                        "paragraph_num": para_idx,
                        "chunk_index": chunk_idx,
                        "token_count": token_count,
                    })

        logger.info(
            "Chunked paper %s: %d sections → %d chunks",
            pmid, len(sections), len(all_chunks),
        )
        return all_chunks

    def _split_text(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        """Split text into overlapping chunks at sentence boundaries.

        If the text fits within max_tokens, returns it as a single chunk.
        Otherwise, splits into overlapping chunks preferring sentence boundaries.

        Args:
            text: Input text to split.
            max_tokens: Maximum tokens per chunk.
            overlap: Number of overlap tokens between consecutive chunks.

        Returns:
            List of text chunks.
        """
        token_count = self._count_tokens(text)

        # If text fits in one chunk, return as-is
        if token_count <= max_tokens:
            return [text]

        # Split into sentences
        sentences = _SENTENCE_SPLIT_RE.split(text)
        if not sentences:
            return [text]

        chunks: list[str] = []
        current_sentences: list[str] = []
        current_token_count = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = self._count_tokens(sentence)

            # If a single sentence exceeds max_tokens, force-split by words
            if sentence_tokens > max_tokens:
                # Flush current buffer first
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences = []
                    current_token_count = 0

                # Force-split the long sentence by word boundaries
                word_chunks = self._force_split_by_words(sentence, max_tokens, overlap)
                chunks.extend(word_chunks)
                continue

            # Would adding this sentence exceed the limit?
            if current_token_count + sentence_tokens > max_tokens and current_sentences:
                # Emit current chunk
                chunks.append(" ".join(current_sentences))

                # Calculate overlap: take sentences from the end of current chunk
                overlap_sentences: list[str] = []
                overlap_count = 0
                for s in reversed(current_sentences):
                    s_tokens = self._count_tokens(s)
                    if overlap_count + s_tokens > overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_count += s_tokens

                current_sentences = list(overlap_sentences)
                current_token_count = overlap_count

            current_sentences.append(sentence)
            current_token_count += sentence_tokens

        # Flush remaining
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks if chunks else [text]

    def _force_split_by_words(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        """Force-split text by word boundaries when a single sentence exceeds max_tokens.

        Args:
            text: Long text to split.
            max_tokens: Maximum tokens per chunk.
            overlap: Overlap tokens between chunks.

        Returns:
            List of text chunks.
        """
        words = text.split()
        chunks: list[str] = []
        start = 0

        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

            if end >= len(words):
                break

            # Move start forward by (max_tokens - overlap)
            step = max(1, max_tokens - overlap)
            start += step

        return chunks

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Count tokens in text using simple whitespace tokenization.

        This is a fast approximation. For biomedical text, word-based
        tokenization is close enough to subword tokenization for chunking.

        Args:
            text: Input text.

        Returns:
            Approximate token count.
        """
        if not text:
            return 0
        return len(text.split())
