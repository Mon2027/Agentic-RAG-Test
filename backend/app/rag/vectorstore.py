"""Vector store wrapper using ChromaDB."""

import logging
import math
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core import get_settings
from app.rag.document_processor import DocumentChunk
from app.rag.embeddings import EmbeddingModel, get_embedding_model

logger = logging.getLogger(__name__)

KEYWORD_STOPWORDS = {
    "的",
    "了",
    "和",
    "与",
    "及",
    "或",
    "在",
    "中",
    "对",
    "为",
    "是",
    "有",
    "请",
    "分析",
    "情况",
    "变化",
    "如何",
    "多少",
}

KEYWORD_HINT_TERMS = {
    "低空经济",
    "通航",
    "通用航空",
    "营收",
    "收入",
    "净利润",
    "利润",
    "同比",
    "增长",
    "扭亏为盈",
    "业务",
    "布局",
    "战略",
    "主业",
    "交付",
    "费用",
    "数字化",
    "智慧交通",
    "商业航天",
    "空管",
    "连接器",
    "机器人",
    "轴承",
    "无人物流",
    "军用型号",
    "应用场景",
    "新增长极",
    "增长引擎",
}


class VectorStore:
    """ChromaDB-based vector store for document embeddings.

    This class provides a high-level interface for:
    - Storing document embeddings
    - Similarity search
    - Document management (add, delete, update)
    """

    def __init__(
        self,
        collection_name: str = "reports",
        persist_directory: Path | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        """Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory to persist the database.
                Defaults to settings.vector_store_path.
            embedding_model: Embedding model to use.
                Defaults to the cached model from get_embedding_model().
        """
        settings = get_settings()

        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.vector_store_path

        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing vector store at: {self.persist_directory}")

        # Initialize embedding model
        self._embedding_model = embedding_model or get_embedding_model()

        # Initialize ChromaDB client
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"Collection '{collection_name}' initialized with "
            f"{self._collection.count()} documents"
        )

    @property
    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self._collection.count()

    def add_documents(
        self,
        chunks: list[DocumentChunk],
        ids: list[str] | None = None,
    ) -> int:
        """Add document chunks to the vector store.

        Args:
            chunks: List of DocumentChunk objects to add.
            ids: Optional list of unique IDs for each chunk.
                If not provided, IDs will be generated.

        Returns:
            Number of documents added.
        """
        if not chunks:
            return 0

        logger.info(f"Adding {len(chunks)} documents to vector store")

        # Extract texts and generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self._embedding_model.embed_documents(texts)

        # Generate IDs if not provided
        if ids is None:
            import uuid

            ids = [str(uuid.uuid4()) for _ in chunks]

        # Prepare metadata
        metadatas = [chunk.metadata for chunk in chunks]

        # Add to collection
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(chunks)} documents to collection")

        return len(chunks)

    def add_document_for_file(
        self,
        chunks: list[DocumentChunk],
        file_id: str,
    ) -> int:
        """Add document chunks with file-specific IDs.

        Args:
            chunks: List of DocumentChunk objects to add.
            file_id: The file ID to use as prefix for chunk IDs.

        Returns:
            Number of documents added.
        """
        # Generate IDs with file_id prefix
        ids = [f"{file_id}_{i}" for i in range(len(chunks))]

        return self.add_documents(chunks, ids)

    def replace_document_for_file(
        self,
        chunks: list[DocumentChunk],
        file_id: str,
    ) -> int:
        """Replace one file's chunks and restore its old index on failure.

        New embeddings and a complete snapshot of the previous records are
        prepared before deletion. If deletion or insertion fails, the method
        removes any partially written replacement records and restores the old
        snapshot with its original embeddings.

        Args:
            chunks: Replacement chunks for the document.
            file_id: File identifier shared by all replacement chunks.

        Returns:
            Number of replacement chunks written.

        Raises:
            ValueError: If the replacement contains no chunks.
            RuntimeError: If replacement fails and rollback also fails.
            Exception: The original Chroma or embedding exception after a
                successful rollback.
        """
        if not chunks:
            raise ValueError("Cannot replace index with empty document")

        new_ids = [f"{file_id}_{i}" for i in range(len(chunks))]
        new_documents = [chunk.content for chunk in chunks]
        new_metadatas = [chunk.metadata for chunk in chunks]

        # Embedding failures must happen before the old index is touched.
        new_embeddings = self._embedding_model.embed_documents(new_documents)

        old_records = self._collection.get(
            where={"file_id": file_id},
            include=["documents", "metadatas", "embeddings"],
        )
        old_ids = list(old_records.get("ids") or [])
        old_documents = list(old_records.get("documents") or [])
        old_metadatas = list(old_records.get("metadatas") or [])
        raw_old_embeddings = old_records.get("embeddings")
        old_embeddings = (
            raw_old_embeddings.tolist()
            if hasattr(raw_old_embeddings, "tolist")
            else raw_old_embeddings
        )

        if old_ids:
            if len(old_documents) != len(old_ids):
                raise RuntimeError("Cannot snapshot all old document contents")
            if len(old_metadatas) != len(old_ids):
                raise RuntimeError("Cannot snapshot all old document metadata")
            if old_embeddings is None or len(old_embeddings) != len(old_ids):
                raise RuntimeError("Cannot snapshot all old document embeddings")

        try:
            if old_ids:
                self._collection.delete(ids=old_ids)

            self._collection.add(
                ids=new_ids,
                documents=new_documents,
                embeddings=new_embeddings,
                metadatas=new_metadatas,
            )
        except Exception:
            logger.exception(
                "Failed to replace index for file_id %s; restoring old snapshot",
                file_id,
            )
            rollback_failures: list[Exception] = []

            try:
                self._collection.delete(ids=new_ids)
            except Exception as cleanup_error:
                rollback_failures.append(cleanup_error)
                logger.exception(
                    "Failed to clean partial replacement for file_id %s",
                    file_id,
                )

            if old_ids:
                try:
                    self._collection.upsert(
                        ids=old_ids,
                        documents=old_documents,
                        embeddings=old_embeddings,
                        metadatas=old_metadatas,
                    )
                except Exception as restore_error:
                    rollback_failures.append(restore_error)
                    logger.exception(
                        "Failed to restore old index for file_id %s",
                        file_id,
                    )

            if rollback_failures:
                raise RuntimeError(
                    f"Failed to replace index for {file_id}; rollback also failed"
                ) from rollback_failures[-1]
            raise

        logger.info(
            "Replaced index for file_id %s with %d chunks",
            file_id,
            len(chunks),
        )
        return len(chunks)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents.

        Args:
            query: The search query.
            k: Number of results to return.
            filter_: Optional metadata filter.

        Returns:
            List of search results with content, metadata, and distance.
        """
        logger.debug(f"Searching for: {query[:50]}... (k={k})")

        # Generate query embedding
        query_embedding = self._embedding_model.embed_query(query)

        # Search
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        search_results: list[dict[str, Any]] = []

        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                result = {
                    "id": results["ids"][0][i] if results.get("ids") else "",
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                }
                search_results.append(result)

        logger.debug(f"Found {len(search_results)} results")

        return search_results

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter_: dict[str, Any] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Search for similar documents with similarity scores.

        Args:
            query: The search query.
            k: Number of results to return.
            filter_: Optional metadata filter.

        Returns:
            List of (document, score) tuples. Score is cosine similarity (1 - distance).
        """
        results = self.similarity_search(query, k, filter_)

        # Convert distance to similarity score (assuming cosine distance)
        return [(r, 1 - r["distance"]) for r in results]

    def keyword_search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter_: dict[str, Any] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Search documents with lightweight keyword scoring.

        Args:
            query: The search query.
            k: Number of results to return.
            filter_: Optional metadata filter.

        Returns:
            List of (document, score) tuples. Score is normalized to 0-1.
        """
        query_terms = self._keyword_terms(query)
        if not query_terms:
            return []

        results = self._collection.get(
            where=filter_,
            include=["documents", "metadatas"],
        )

        scored: list[tuple[dict[str, Any], float]] = []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        ids = results.get("ids") or []

        for i, content in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            score = self._keyword_score(query_terms, content, metadata)
            if score <= 0:
                continue

            scored.append((
                {
                    "id": ids[i] if i < len(ids) else "",
                    "content": content,
                    "metadata": metadata,
                },
                score,
            ))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def delete_by_file_id(self, file_id: str) -> int:
        """Delete all documents associated with a file.

        Args:
            file_id: The file ID to delete.

        Returns:
            Number of documents deleted.
        """
        logger.info(f"Deleting documents for file_id: {file_id}")

        # Get all documents with this file_id
        results = self._collection.get(
            where={"file_id": file_id},
            include=["metadatas"],
        )

        if not results["ids"]:
            logger.info(f"No documents found for file_id: {file_id}")
            return 0

        # Delete by IDs
        self._collection.delete(ids=results["ids"])

        deleted_count = len(results["ids"])
        logger.info(f"Deleted {deleted_count} documents for file_id: {file_id}")

        return deleted_count

    def delete_all(self) -> None:
        """Delete all documents in the collection."""
        logger.warning("Deleting all documents in collection")

        # Get all IDs
        results = self._collection.get()

        if results["ids"]:
            self._collection.delete(ids=results["ids"])

        logger.info("All documents deleted")

    def get_file_chunks(self, file_id: str) -> list[dict[str, Any]]:
        """Get all chunks for a specific file.

        Args:
            file_id: The file ID to retrieve.

        Returns:
            List of document chunks with metadata.
        """
        results = self._collection.get(
            where={"file_id": file_id},
            include=["documents", "metadatas"],
        )

        chunks: list[dict[str, Any]] = []

        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                chunks.append({
                    "id": doc_id,
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })

        return chunks

    def _keyword_terms(self, query: str) -> list[str]:
        """Extract terms for keyword search."""
        normalized = query.lower()
        raw_terms = re.split(r"[\s,，。；;：:、/\\()\[\]{}<>《》\"'“”！？?+\-=]+", normalized)
        terms: list[str] = []

        for term in raw_terms:
            term = term.strip()
            if len(term) < 2 or term in KEYWORD_STOPWORDS:
                continue
            terms.append(term)
            terms.extend(self._expand_chinese_query_terms(term))

        return list(dict.fromkeys(terms))

    def _keyword_score(
        self,
        query_terms: list[str],
        content: str,
        metadata: dict[str, Any],
    ) -> float:
        """Compute a normalized keyword relevance score."""
        normalized_content = content.lower()
        file_name = str(metadata.get("file_name", "")).lower()
        section_title = str(metadata.get("section_title", "")).lower()
        if not normalized_content and not file_name and not section_title:
            return 0.0

        total_weight = sum(self._term_weight(term) for term in query_terms)
        matched_weight = 0.0
        field_score = 0.0

        for term in query_terms:
            weight = self._term_weight(term)
            content_count = normalized_content.count(term)
            file_count = file_name.count(term)
            section_count = section_title.count(term)
            if not any([content_count, file_count, section_count]):
                continue

            matched_weight += weight
            if file_count:
                field_score += weight * 2.4
            if section_count:
                field_score += weight * 1.4
            if content_count:
                field_score += weight * min(1.2, 0.45 + math.log(content_count + 1) * 0.35)

        if matched_weight == 0 or total_weight == 0:
            return 0.0

        coverage = matched_weight / total_weight
        field_strength = min(1.0, field_score / max(1.0, total_weight * 2.2))
        return round(min(1.0, (coverage * 0.45) + (field_strength * 0.55)), 6)

    def _expand_chinese_query_terms(self, term: str) -> list[str]:
        """Add Chinese title/company fragments for keyword retrieval."""
        if not re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9%]+", term):
            return []

        expanded: list[str] = []
        for hint in KEYWORD_HINT_TERMS:
            if hint in term:
                expanded.append(hint)

        chinese_prefix = re.match(r"^[\u4e00-\u9fff]{2,6}", term)
        if chinese_prefix:
            expanded.append(chinese_prefix.group(0))

        if len(term) <= 10:
            return expanded

        for size in (4, 3, 2):
            for start in range(0, len(term) - size + 1):
                piece = term[start:start + size]
                if re.fullmatch(r"[\u4e00-\u9fff]+", piece) and piece not in KEYWORD_STOPWORDS:
                    expanded.append(piece)

        return expanded[:40]

    def _term_weight(self, term: str) -> float:
        """Weight longer and domain-specific keyword terms more heavily."""
        if term in KEYWORD_HINT_TERMS:
            return 2.0
        if re.search(r"\d", term):
            return 1.6
        if len(term) >= 5:
            return 1.5
        if len(term) >= 3:
            return 1.1
        return 0.7

    def list_files(self) -> list[dict[str, Any]]:
        """List all unique files in the collection.

        Returns:
            List of file information with metadata.
        """
        # Get all documents
        results = self._collection.get(include=["metadatas"])

        if not results["metadatas"]:
            return []

        # Group by file_id
        files: dict[str, dict[str, Any]] = {}

        for metadata in results["metadatas"]:
            file_id = metadata.get("file_id")
            if file_id and file_id not in files:
                files[file_id] = {
                    "file_id": file_id,
                    "file_name": metadata.get("file_name", "Unknown"),
                    "source": metadata.get("source", ""),
                    "chunk_count": 0,
                }
            if file_id:
                files[file_id]["chunk_count"] += 1

        return list(files.values())


# Singleton instance
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Get or create the singleton vector store instance.

    Returns:
        VectorStore instance.
    """
    global _vector_store

    if _vector_store is None:
        _vector_store = VectorStore()

    return _vector_store
