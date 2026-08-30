"""Embedding model wrapper for RAG system."""

import logging
import os
from functools import lru_cache

from langchain_core.embeddings import Embeddings

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer

from app.core import get_settings

logger = logging.getLogger(__name__)


class EmbeddingModel(Embeddings):
    """Wrapper for sentence-transformers embedding model.

    This class provides a LangChain-compatible interface for the
    sentence-transformers embedding model, optimized for Chinese
    and English research documents.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize the embedding model.

        Args:
            model_name: Name or path of the sentence-transformers model.
                Defaults to the model specified in settings.
            device: Device to run the model on ('cpu', 'cuda', 'mps').
                Defaults to auto-detection.
            normalize_embeddings: Whether to normalize embeddings to unit length.
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.normalize_embeddings = normalize_embeddings

        logger.info(f"Loading embedding model: {self.model_name}")

        self._model = SentenceTransformer(
            self.model_name,
            device=device,
            trust_remote_code=True,
        )

        # Get embedding dimension
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self._dimension}")

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        logger.debug(f"Embedding {len(texts)} documents")

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query.

        Args:
            text: The query text to embed.

        Returns:
            Embedding vector.
        """
        logger.debug(f"Embedding query: {text[:50]}...")

        embedding = self._model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        return embedding.tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously embed a list of documents.

        Note: sentence-transformers doesn't have native async support,
        so this runs the sync version in a thread pool.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronously embed a single query.

        Args:
            text: The query text to embed.

        Returns:
            Embedding vector.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)


@lru_cache(maxsize=1)
def get_embedding_model(
    model_name: str | None = None,
    device: str | None = None,
) -> EmbeddingModel:
    """Get or create a cached embedding model instance.

    Args:
        model_name: Name or path of the model.
        device: Device to run the model on.

    Returns:
        Cached EmbeddingModel instance.
    """
    return EmbeddingModel(model_name=model_name, device=device)
