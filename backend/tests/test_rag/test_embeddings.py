"""Tests for embedding model."""

import pytest
from unittest.mock import MagicMock, patch


class TestEmbeddingModel:
    """Test cases for EmbeddingModel."""

    def test_embedding_model_init(self):
        """Test EmbeddingModel initialization."""
        with patch("app.rag.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 1024
            mock_st.return_value = mock_model

            from app.rag.embeddings import EmbeddingModel

            model = EmbeddingModel(model_name="test-model")

            assert model.model_name == "test-model"
            assert model.dimension == 1024

    def test_embed_query(self):
        """Test embedding a single query."""
        with patch("app.rag.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            import numpy as np
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
            mock_model.get_sentence_embedding_dimension.return_value = 3
            mock_st.return_value = mock_model

            from app.rag.embeddings import EmbeddingModel

            model = EmbeddingModel()
            result = model.embed_query("test query")

            assert isinstance(result, list)
            assert len(result) == 3
            mock_model.encode.assert_called_once()

    def test_embed_documents(self):
        """Test embedding multiple documents."""
        with patch("app.rag.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            import numpy as np
            mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
            mock_model.get_sentence_embedding_dimension.return_value = 2
            mock_st.return_value = mock_model

            from app.rag.embeddings import EmbeddingModel

            model = EmbeddingModel()
            result = model.embed_documents(["doc1", "doc2"])

            assert isinstance(result, list)
            assert len(result) == 2
            assert len(result[0]) == 2

    def test_embed_empty_documents(self):
        """Test embedding empty document list."""
        with patch("app.rag.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 1024
            mock_st.return_value = mock_model

            from app.rag.embeddings import EmbeddingModel

            model = EmbeddingModel()
            result = model.embed_documents([])

            assert result == []


class TestGetEmbeddingModel:
    """Test cases for get_embedding_model function."""

    def test_get_embedding_model_cached(self):
        """Test that get_embedding_model returns cached instance."""
        from app.rag.embeddings import get_embedding_model, EmbeddingModel

        # Clear cache first
        get_embedding_model.cache_clear()

        with patch("app.rag.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 1024
            mock_st.return_value = mock_model

            model1 = get_embedding_model()
            model2 = get_embedding_model()

            # Should be the same instance (cached)
            assert model1 is model2

    def test_get_embedding_model_with_custom_params(self):
        """Test get_embedding_model with custom parameters."""
        from app.rag.embeddings import get_embedding_model

        get_embedding_model.cache_clear()

        with patch("app.rag.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 1024
            mock_st.return_value = mock_model

            model = get_embedding_model(model_name="custom-model")

            assert model.model_name == "custom-model"