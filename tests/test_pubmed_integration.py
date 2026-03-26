"""Integration tests for PubMed and PMC API clients.

These tests make REAL API calls to NCBI services.
Mark with @pytest.mark.slow so they can be skipped in CI with: pytest -m "not slow"
"""

import pytest

from oncocontext.services.pubmed_client import PubMedClient
from oncocontext.services.pmc_client import PMCClient
from oncocontext.services.bioc_parser import BioCParser


# Mark all tests in this module as slow (real network calls)
pytestmark = pytest.mark.slow


@pytest.fixture
async def pubmed_client():
    """Create a PubMedClient for testing."""
    client = PubMedClient()
    yield client
    await client.close()


@pytest.fixture
async def pmc_client():
    """Create a PMCClient for testing."""
    client = PMCClient()
    yield client
    await client.close()


class TestPubMedSearch:
    """Integration tests for PubMed esearch."""

    @pytest.mark.asyncio
    async def test_search_returns_pmids(self, pubmed_client: PubMedClient):
        """Search for 'CAR-T cell exhaustion' returns PMIDs."""
        result = await pubmed_client.search("CAR-T cell exhaustion", max_results=5)
        assert "pmids" in result
        assert "total_count" in result
        assert len(result["pmids"]) > 0
        assert result["total_count"] > 0
        # PMIDs should be numeric strings
        for pmid in result["pmids"]:
            assert pmid.isdigit()

    @pytest.mark.asyncio
    async def test_search_with_date_range(self, pubmed_client: PubMedClient):
        """Search with date range narrows results."""
        result = await pubmed_client.search(
            "PD-1 checkpoint inhibitor",
            max_results=5,
            date_range="2023-2025",
        )
        assert "pmids" in result
        assert len(result["pmids"]) > 0

    @pytest.mark.asyncio
    async def test_search_empty_query(self, pubmed_client: PubMedClient):
        """Empty query returns empty results."""
        result = await pubmed_client.search("")
        assert result["pmids"] == []
        assert result["total_count"] == 0

    @pytest.mark.asyncio
    async def test_search_nonsense_query(self, pubmed_client: PubMedClient):
        """Nonsensical query returns few/no results without error."""
        result = await pubmed_client.search("xyzzyspoon123456789nonsense")
        assert "pmids" in result
        # May return 0 results, which is fine


class TestPubMedFetchDetails:
    """Integration tests for PubMed efetch."""

    @pytest.mark.asyncio
    async def test_fetch_known_pmid(self, pubmed_client: PubMedClient):
        """Fetch details for a known PMID returns expected fields."""
        # PMID 35922516 = a well-known T cell exhaustion paper
        papers = await pubmed_client.fetch_details(["35922516"])
        assert len(papers) == 1
        paper = papers[0]

        assert paper["pmid"] == "35922516"
        assert paper["title"]  # non-empty title
        assert len(paper["authors"]) > 0
        assert paper["journal"]  # non-empty journal
        assert paper["year"] > 2000

    @pytest.mark.asyncio
    async def test_fetch_multiple_pmids(self, pubmed_client: PubMedClient):
        """Fetch details for multiple PMIDs returns all of them."""
        pmids = ["35922516", "33811159"]
        papers = await pubmed_client.fetch_details(pmids)
        assert len(papers) == 2
        fetched_pmids = {p["pmid"] for p in papers}
        assert "35922516" in fetched_pmids
        assert "33811159" in fetched_pmids

    @pytest.mark.asyncio
    async def test_fetch_invalid_pmid(self, pubmed_client: PubMedClient):
        """Invalid PMID is handled gracefully (returns empty or skips)."""
        papers = await pubmed_client.fetch_details(["99999999999"])
        # Should not raise, may return empty list
        assert isinstance(papers, list)

    @pytest.mark.asyncio
    async def test_fetch_empty_list(self, pubmed_client: PubMedClient):
        """Empty PMID list returns empty results."""
        papers = await pubmed_client.fetch_details([])
        assert papers == []


class TestPMCClient:
    """Integration tests for PMC BioC API."""

    @pytest.mark.asyncio
    async def test_fetch_bioc_known_article(self, pmc_client: PMCClient):
        """Fetch BioC JSON for a known open-access PMC article."""
        # PMC7842210 = a known open-access article
        bioc = await pmc_client.fetch_bioc("PMC7842210")
        if bioc is not None:
            # If available, it should have the expected structure
            assert "documents" in bioc
            assert len(bioc["documents"]) > 0
            assert "passages" in bioc["documents"][0]

    @pytest.mark.asyncio
    async def test_fetch_bioc_nonexistent(self, pmc_client: PMCClient):
        """Non-existent PMC ID returns None."""
        bioc = await pmc_client.fetch_bioc("PMC999999999")
        assert bioc is None

    @pytest.mark.asyncio
    async def test_check_availability(self, pmc_client: PMCClient):
        """Availability check returns a boolean."""
        result = await pmc_client.check_availability("PMC7842210")
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_normalize_pmc_id_without_prefix(self, pmc_client: PMCClient):
        """PMC ID without prefix is normalized correctly."""
        normalized = PMCClient._normalize_pmc_id("7842210")
        assert normalized == "PMC7842210"

    @pytest.mark.asyncio
    async def test_normalize_pmc_id_with_prefix(self, pmc_client: PMCClient):
        """PMC ID with prefix is kept as-is."""
        normalized = PMCClient._normalize_pmc_id("PMC7842210")
        assert normalized == "PMC7842210"


class TestEndToEndPipeline:
    """End-to-end tests combining search → fetch → parse."""

    @pytest.mark.asyncio
    async def test_search_and_fetch(self, pubmed_client: PubMedClient):
        """Search → fetch details pipeline works."""
        search_result = await pubmed_client.search("CAR-T exhaustion", max_results=3)
        pmids = search_result["pmids"]
        assert len(pmids) > 0

        papers = await pubmed_client.fetch_details(pmids)
        assert len(papers) > 0

        for paper in papers:
            assert "pmid" in paper
            assert "title" in paper
            assert "authors" in paper

    @pytest.mark.asyncio
    async def test_fetch_and_parse_fulltext(self, pubmed_client: PubMedClient, pmc_client: PMCClient):
        """Fetch paper with PMC ID → get BioC → parse sections."""
        # Search for a paper likely to have PMC full text
        search_result = await pubmed_client.search(
            "CAR-T cell therapy review",
            max_results=10,
        )
        papers = await pubmed_client.fetch_details(search_result["pmids"])

        # Find one with a PMC ID
        paper_with_pmc = None
        for paper in papers:
            if paper.get("pmc_id"):
                paper_with_pmc = paper
                break

        if paper_with_pmc is None:
            pytest.skip("No paper with PMC ID found in search results")

        # Fetch BioC
        bioc = await pmc_client.fetch_bioc(paper_with_pmc["pmc_id"])
        if bioc is None:
            pytest.skip("BioC not available for this PMC article")

        # Parse sections
        parser = BioCParser()
        sections = parser.parse(bioc)
        assert len(sections) > 0

        # Should have at least some content
        total_paragraphs = sum(len(s.paragraphs) for s in sections)
        assert total_paragraphs > 0
