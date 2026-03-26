"""All settings, paths, constants, and environment variable overrides for OncoContext.

Uses Pydantic BaseSettings with ONCO_ prefix for environment variable overrides.
"""

from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field


# Resolve project root (two levels up from this file: src/oncocontext/config.py -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """OncoContext configuration with environment variable overrides (prefix: ONCO_)."""

    model_config = {"env_prefix": "ONCO_"}

    # ── API Endpoints ──────────────────────────────────────────────────────────
    PUBMED_BASE_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    PMC_BIOC_URL: str = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json"
    PMC_BIOC_XML_URL: str = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml"
    NCBI_API_KEY: str | None = None

    # ── Embedding Model ────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
    EMBEDDING_DIM: int = 768
    EMBEDDING_MAX_LENGTH: int = 512

    # ── Reranker ───────────────────────────────────────────────────────────────
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Chunking Parameters ────────────────────────────────────────────────────
    CHUNK_SIZE: int = 384          # max tokens per chunk
    CHUNK_OVERLAP: int = 64        # overlap between consecutive chunks
    CHUNK_MIN_TOKENS: int = 50     # minimum chunk size (discard shorter)

    # ── Vector Search ──────────────────────────────────────────────────────────
    VECTOR_TOP_K: int = 50         # initial retrieval candidates
    RERANK_TOP_K: int = 10         # results after reranking

    # ── Relevance Scoring Weights ──────────────────────────────────────────────
    RELEVANCE_THRESHOLD: float = 0.40
    WEIGHT_SEMANTIC: float = 0.35
    WEIGHT_TITLE: float = 0.25
    WEIGHT_MESH: float = 0.15
    WEIGHT_RECENCY: float = 0.10
    WEIGHT_JOURNAL: float = 0.10
    WEIGHT_DISCOVERY: float = 0.05

    # ── Section Boost Factors ──────────────────────────────────────────────────
    BOOST_METHODS: float = 1.5
    BOOST_RESULTS: float = 1.3
    BOOST_DISCUSSION: float = 1.5

    # ── Cache TTLs (seconds) ───────────────────────────────────────────────────
    CACHE_TTL_PUBMED: int = 86400       # 24 hours
    CACHE_TTL_PMC: int = 2592000        # 30 days
    CACHE_SIZE_LIMIT_GB: int = 2        # maximum L2 cache size
    CACHE_L1_MAX_SIZE_MB: int = 100     # in-memory LRU cap

    # ── PubMed Rate Limits ─────────────────────────────────────────────────────
    PUBMED_MAX_RESULTS: int = 50
    PUBMED_RATE_LIMIT: int = 3          # requests/sec without API key
    PUBMED_RATE_LIMIT_WITH_KEY: int = 10
    PMC_RATE_LIMIT: int = 5
    MAX_CONCURRENT_FETCHES: int = 5

    # ── File Limits ────────────────────────────────────────────────────────────
    MAX_LAB_FILE_SIZE_MB: int = 50
    MAX_PAPERS_PER_SEARCH: int = 100

    # ── Response Size Limits (pagination) ──────────────────────────────────────
    MAX_RESPONSE_SIZE_KB: int = 900      # ~900KB to stay safely under Claude Desktop's 1MB limit
    PAGINATION_SESSION_TTL: int = 1800   # 30 minutes TTL for paginated sessions
    PAGINATION_MAX_SESSIONS: int = 50    # cap on concurrent in-memory sessions

    # ── Data Directories ───────────────────────────────────────────────────────
    DATA_DIR: Path = Field(default_factory=lambda: _PROJECT_ROOT / "data")
    CHROMA_DIR: Path = Field(default_factory=lambda: _PROJECT_ROOT / "data" / "chromadb")
    CACHE_DIR: Path = Field(default_factory=lambda: _PROJECT_ROOT / "data" / "cache")
    SQLITE_DIR: Path = Field(default_factory=lambda: _PROJECT_ROOT / "data" / "sqlite")
    MODEL_DIR: Path = Field(default_factory=lambda: _PROJECT_ROOT / "data" / "models")

    # ── ChromaDB ───────────────────────────────────────────────────────────────
    CHROMADB_LITERATURE_COLLECTION: str = "literature_chunks"
    CHROMADB_LAB_COLLECTION: str = "lab_data"
    CHROMADB_DISTANCE_METRIC: str = "cosine"

    # ── SQLite ─────────────────────────────────────────────────────────────────
    @property
    def SQLITE_DB_PATH(self) -> Path:
        return self.SQLITE_DIR / "metadata.db"

    # ── Resources ──────────────────────────────────────────────────────────────
    @property
    def SYNONYM_DICT_PATH(self) -> Path:
        return _PROJECT_ROOT / "resources" / "synonym_dict.json"


# Singleton settings instance
settings = Settings()
