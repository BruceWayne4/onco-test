"""Unit tests for the Reranker (cross-encoder reranking service).

Tests:
    - Reranking changes order based on query relevance
    - top_k limiting works
    - Empty input returns empty list
    - rerank_score field is added to chunks
    - Uses mock to avoid loading the actual cross-encoder model
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from oncocontext.services.reranker import Reranker


@pytest.fixture
def mock_reranker():
    """Create a Reranker with a mocked cross-encoder model."""
    reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Mock the model's predict method
    mock_model = MagicMock()
    reranker._model = mock_model
    return reranker, mock_model


class TestRerankerInit:
    """Test Reranker initialization."""

    def test_init_default_model(self):
        """Reranker initializes with default model name from config."""
        from oncocontext.config import settings
        reranker = Reranker()
        assert reranker._model_name == settings.RERANKER_MODEL
        assert reranker._model is None

    def test_init_custom_model(self):
        """Reranker accepts a custom model name."""
        reranker = Reranker(model_name="custom-model")
        assert reranker._model_name == "custom-model"

    def test_is_loaded_false_initially(self):
        """Model is not loaded at init."""
        reranker = Reranker()
        assert reranker.is_loaded is False

    def test_is_loaded_true_after_mock(self, mock_reranker):
        """Model reports loaded after being set."""
        reranker, _ = mock_reranker
        assert reranker.is_loaded is True


class TestRerankerRerank:
    """Test the rerank method."""

    def test_rerank_empty_input(self, mock_reranker):
        """Empty chunks list returns empty list."""
        reranker, mock_model = mock_reranker
        result = reranker.rerank("test query", [])
        assert result == []
        mock_model.predict.assert_not_called()

    def test_rerank_adds_score_field(self, mock_reranker):
        """Each chunk gets a 'rerank_score' field."""
        reranker, mock_model = mock_reranker
        mock_model.predict.return_value = np.array([0.9, 0.1, 0.5])

        chunks = [
            {"text": "chunk A", "id": 1},
            {"text": "chunk B", "id": 2},
            {"text": "chunk C", "id": 3},
        ]

        result = reranker.rerank("test query", chunks, top_k=3)
        for chunk in result:
            assert "rerank_score" in chunk
            assert isinstance(chunk["rerank_score"], float)

    def test_rerank_changes_order(self, mock_reranker):
        """Reranking reorders chunks by cross-encoder score."""
        reranker, mock_model = mock_reranker
        # Score: chunk B gets highest, chunk A lowest
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])

        chunks = [
            {"text": "chunk A", "id": 1},
            {"text": "chunk B", "id": 2},
            {"text": "chunk C", "id": 3},
        ]

        result = reranker.rerank("test query", chunks, top_k=3)

        # Should be sorted: B (0.9), C (0.5), A (0.1)
        assert result[0]["id"] == 2
        assert result[1]["id"] == 3
        assert result[2]["id"] == 1

    def test_rerank_top_k_limiting(self, mock_reranker):
        """top_k limits the number of returned results."""
        reranker, mock_model = mock_reranker
        mock_model.predict.return_value = np.array([0.9, 0.7, 0.5, 0.3, 0.1])

        chunks = [
            {"text": f"chunk {i}", "id": i} for i in range(5)
        ]

        result = reranker.rerank("test query", chunks, top_k=2)
        assert len(result) == 2

    def test_rerank_default_top_k(self, mock_reranker):
        """Default top_k comes from settings.RERANK_TOP_K."""
        reranker, mock_model = mock_reranker
        from oncocontext.config import settings

        # Create more chunks than RERANK_TOP_K
        n = settings.RERANK_TOP_K + 5
        mock_model.predict.return_value = np.array([float(i) for i in range(n)])

        chunks = [{"text": f"chunk {i}", "id": i} for i in range(n)]

        result = reranker.rerank("test query", chunks)
        assert len(result) == settings.RERANK_TOP_K

    def test_rerank_scores_descending(self, mock_reranker):
        """Results are sorted by rerank_score in descending order."""
        reranker, mock_model = mock_reranker
        mock_model.predict.return_value = np.array([0.3, 0.8, 0.1, 0.6, 0.5])

        chunks = [{"text": f"chunk {i}", "id": i} for i in range(5)]

        result = reranker.rerank("test query", chunks, top_k=5)
        scores = [c["rerank_score"] for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_creates_correct_pairs(self, mock_reranker):
        """Verify the model receives correct (query, text) pairs."""
        reranker, mock_model = mock_reranker
        mock_model.predict.return_value = np.array([0.5, 0.3])

        chunks = [
            {"text": "about immunology", "id": 1},
            {"text": "about cancer", "id": 2},
        ]

        reranker.rerank("T cell exhaustion", chunks, top_k=2)

        # Check the pairs passed to predict
        call_args = mock_model.predict.call_args[0][0]
        assert call_args == [
            ("T cell exhaustion", "about immunology"),
            ("T cell exhaustion", "about cancer"),
        ]

    def test_rerank_single_chunk(self, mock_reranker):
        """Single chunk input works correctly."""
        reranker, mock_model = mock_reranker
        mock_model.predict.return_value = np.array([0.7])

        chunks = [{"text": "only chunk", "id": 1}]

        result = reranker.rerank("test", chunks, top_k=5)
        assert len(result) == 1
        assert result[0]["rerank_score"] == 0.7

    def test_rerank_preserves_chunk_metadata(self, mock_reranker):
        """Original chunk fields are preserved after reranking."""
        reranker, mock_model = mock_reranker
        mock_model.predict.return_value = np.array([0.5])

        chunks = [{
            "text": "some text",
            "id": 1,
            "pmid": "12345",
            "section": "methods",
            "extra_field": "preserved",
        }]

        result = reranker.rerank("test", chunks, top_k=1)
        assert result[0]["pmid"] == "12345"
        assert result[0]["section"] == "methods"
        assert result[0]["extra_field"] == "preserved"

    def test_rerank_with_negative_scores(self, mock_reranker):
        """Cross-encoder can produce negative scores; reranking still works."""
        reranker, mock_model = mock_reranker
        mock_model.predict.return_value = np.array([-1.0, 2.0, -5.0, 0.0])

        chunks = [{"text": f"chunk {i}", "id": i} for i in range(4)]

        result = reranker.rerank("test", chunks, top_k=4)
        assert result[0]["rerank_score"] == 2.0
        assert result[-1]["rerank_score"] == -5.0


class TestRerankerLazyLoad:
    """Test lazy model loading behavior."""

    @patch("oncocontext.services.reranker.CrossEncoder", create=True)
    def test_lazy_load_on_first_rerank(self, mock_ce_class):
        """Model is loaded lazily on first rerank call."""
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = np.array([0.5])
        mock_ce_class.return_value = mock_model_instance

        with patch("oncocontext.services.reranker.CrossEncoder", mock_ce_class):
            reranker = Reranker()
            assert reranker._model is None

            # Import and patch correctly
            import oncocontext.services.reranker as reranker_mod
            original_load = reranker._load_model

            def patched_load():
                reranker._model = mock_model_instance

            reranker._load_model = patched_load
            reranker.rerank("test", [{"text": "chunk"}], top_k=1)
            assert reranker._model is not None
