"""Tests for vector store."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestVectorStore:
    """Test cases for VectorStore."""

    @patch("app.rag.vectorstore.get_embedding_model")
    @patch("app.rag.vectorstore.chromadb.PersistentClient")
    def test_vectorstore_init(self, mock_client, mock_get_embedding):
        """Test VectorStore initialization."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_chroma_client

        mock_embedding = MagicMock()
        mock_embedding.dimension = 1024
        mock_get_embedding.return_value = mock_embedding

        from app.rag.vectorstore import VectorStore

        vs = VectorStore(collection_name="test_collection")

        assert vs.collection_name == "test_collection"
        assert vs.count == 0

    @patch("app.rag.vectorstore.get_embedding_model")
    @patch("app.rag.vectorstore.chromadb.PersistentClient")
    def test_add_documents(self, mock_client, mock_get_embedding):
        """Test adding documents to vector store."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_chroma_client

        mock_embedding = MagicMock()
        mock_embedding.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_get_embedding.return_value = mock_embedding

        from app.rag.vectorstore import VectorStore
        from app.rag.document_processor import DocumentChunk

        vs = VectorStore()
        chunks = [
            DocumentChunk(content="test 1", metadata={"file_id": "f1"}),
            DocumentChunk(content="test 2", metadata={"file_id": "f2"}),
        ]

        result = vs.add_documents(chunks)

        assert result == 2
        mock_collection.add.assert_called_once()

    @patch("app.rag.vectorstore.get_embedding_model")
    @patch("app.rag.vectorstore.chromadb.PersistentClient")
    def test_similarity_search(self, mock_client, mock_get_embedding):
        """Test similarity search."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"file_id": "f1"}, {"file_id": "f2"}]],
            "distances": [[0.1, 0.2]],
        }

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_chroma_client

        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_get_embedding.return_value = mock_embedding

        from app.rag.vectorstore import VectorStore

        vs = VectorStore()
        results = vs.similarity_search("test query", k=2)

        assert len(results) == 2
        assert results[0]["content"] == "doc1"
        assert results[0]["distance"] == 0.1

    @patch("app.rag.vectorstore.get_embedding_model")
    @patch("app.rag.vectorstore.chromadb.PersistentClient")
    def test_keyword_search_with_scores(self, mock_client, mock_get_embedding):
        """Test keyword search ranks exact matches."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": [
                "营收增长 120 毛利率提升",
                "这是一段无关内容",
            ],
            "metadatas": [
                {"file_id": "f1", "file_name": "report.pdf"},
                {"file_id": "f2", "file_name": "other.pdf"},
            ],
        }

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_chroma_client

        mock_embedding = MagicMock()
        mock_get_embedding.return_value = mock_embedding

        from app.rag.vectorstore import VectorStore

        vs = VectorStore()
        results = vs.keyword_search_with_scores("营收 毛利率", k=5)

        assert len(results) == 1
        assert results[0][0]["id"] == "id1"
        assert results[0][0]["content"] == "营收增长 120 毛利率提升"
        assert results[0][1] > 0

    @patch("app.rag.vectorstore.get_embedding_model")
    @patch("app.rag.vectorstore.chromadb.PersistentClient")
    def test_keyword_search_expands_chinese_title_terms(self, mock_client, mock_get_embedding):
        """Test keyword search can match company/title terms without spaces."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": [
                "连接器业务持续拓展，低空经济打开新增量。",
                "其他公司也布局低空经济。",
            ],
            "metadatas": [
                {"file_id": "f1", "file_name": "【创益通】连接器小巨人，布局低空经济.pdf"},
                {"file_id": "f2", "file_name": "【其他公司】低空经济行业报告.pdf"},
            ],
        }

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_chroma_client
        mock_get_embedding.return_value = MagicMock()

        from app.rag.vectorstore import VectorStore

        vs = VectorStore()
        results = vs.keyword_search_with_scores("创益通连接器业务低空经济打开想象空间", k=2)

        assert len(results) == 2
        assert results[0][0]["id"] == "id1"
        assert results[0][1] > results[1][1]

    @patch("app.rag.vectorstore.get_embedding_model")
    @patch("app.rag.vectorstore.chromadb.PersistentClient")
    def test_keyword_search_passes_filter(self, mock_client, mock_get_embedding):
        """Test keyword search passes metadata filters to Chroma."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_chroma_client
        mock_get_embedding.return_value = MagicMock()

        from app.rag.vectorstore import VectorStore

        vs = VectorStore()
        vs.keyword_search_with_scores("营收", k=5, filter_={"file_id": "f1"})

        mock_collection.get.assert_called_once_with(
            where={"file_id": "f1"},
            include=["documents", "metadatas"],
        )

    @patch("app.rag.vectorstore.get_embedding_model")
    @patch("app.rag.vectorstore.chromadb.PersistentClient")
    def test_delete_by_file_id(self, mock_client, mock_get_embedding):
        """Test deleting documents by file ID."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": ["id1", "id2"]}

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_chroma_client

        mock_embedding = MagicMock()
        mock_get_embedding.return_value = mock_embedding

        from app.rag.vectorstore import VectorStore

        vs = VectorStore()
        result = vs.delete_by_file_id("test_file_id")

        assert result == 2
        mock_collection.delete.assert_called_once()

    @patch("app.rag.vectorstore.get_embedding_model")
    @patch("app.rag.vectorstore.chromadb.PersistentClient")
    def test_list_files(self, mock_client, mock_get_embedding):
        """Test listing files in collection."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {
            "metadatas": [
                {"file_id": "f1", "file_name": "test1.pdf"},
                {"file_id": "f1", "file_name": "test1.pdf"},
                {"file_id": "f2", "file_name": "test2.pdf"},
            ]
        }

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_chroma_client

        mock_embedding = MagicMock()
        mock_get_embedding.return_value = mock_embedding

        from app.rag.vectorstore import VectorStore

        vs = VectorStore()
        files = vs.list_files()

        assert len(files) == 2  # Two unique file_ids
        assert files[0]["file_id"] == "f1"
        assert files[1]["file_id"] == "f2"


class TestGetVectorStore:
    """Test cases for get_vector_store function."""

    @patch("app.rag.vectorstore.get_embedding_model")
    @patch("app.rag.vectorstore.chromadb.PersistentClient")
    def test_get_vector_store_singleton(self, mock_client, mock_get_embedding):
        """Test that get_vector_store returns singleton."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_chroma_client

        mock_embedding = MagicMock()
        mock_get_embedding.return_value = mock_embedding

        from app.rag.vectorstore import get_vector_store, VectorStore

        # Reset singleton
        import app.rag.vectorstore as vs_module
        vs_module._vector_store = None

        vs1 = get_vector_store()
        vs2 = get_vector_store()

        # Should be the same instance
        assert vs1 is vs2
