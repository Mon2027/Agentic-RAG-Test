"""RAG Retrieval Evaluation Module.

This module provides evaluation metrics for RAG retrieval quality:
- Recall@K: Proportion of relevant documents retrieved
- Precision@K: Proportion of retrieved documents that are relevant
- MRR (Mean Reciprocal Rank): Position of first relevant document
- NDCG (Normalized Discounted Cumulative Gain): Ranking quality
- Term Coverage@K: Whether expected answer terms appear in the first K chunks
- Evidence Group Recall@K: Whether fact-bearing alternative evidence is retrieved
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.rag.retriever import RetrievedContext, get_retriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvidenceLocation:
    """One acceptable file and page set for an evidence group."""

    file_id: str
    pages: frozenset[int]


@dataclass(frozen=True)
class EvidenceGroup:
    """A fact group that may be supported by alternative evidence locations."""

    group_id: str
    alternatives: tuple[EvidenceLocation, ...]
    terms: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-compatible representation."""
        return {
            "id": self.group_id,
            "terms": sorted(self.terms),
            "alternatives": [
                {
                    "file_id": alternative.file_id,
                    "pages": sorted(alternative.pages),
                }
                for alternative in self.alternatives
            ],
        }


@dataclass
class EvaluationSample:
    """A single evaluation sample.

    Attributes:
        query: The search query
        relevant_doc_ids: Set of document IDs that are relevant (ground truth)
        sample_id: Optional stable dataset sample ID
        relevant_chunks: Optional set of specific chunk IDs that are relevant
        expected_terms: Optional terms that should appear in retrieved evidence
        relevant_evidence_pages: Optional pages that contain answer evidence by file ID
        evidence_groups: Optional fact groups with alternative acceptable locations
    """

    query: str
    relevant_doc_ids: set[str]
    sample_id: str | None = None
    relevant_chunks: set[str] | None = None
    expected_terms: set[str] | None = None
    relevant_evidence_pages: dict[str, set[int]] | None = None
    evidence_groups: tuple[EvidenceGroup, ...] | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "EvaluationSample":
        """Create an evaluation sample from JSON-compatible data."""
        relevant_evidence_pages: dict[str, set[int]] = {}
        for evidence in item.get("evidence", []):
            file_id = str(evidence.get("file_id") or "")
            pages: set[int] = set()
            for page in evidence.get("pages", []):
                try:
                    pages.add(int(page))
                except (TypeError, ValueError):
                    continue
            if file_id and pages:
                relevant_evidence_pages.setdefault(file_id, set()).update(pages)

        evidence_groups: list[EvidenceGroup] = []
        for index, group in enumerate(item.get("evidence_groups", []), start=1):
            alternatives: list[EvidenceLocation] = []
            for alternative in group.get("alternatives", []):
                file_id = str(alternative.get("file_id") or "")
                pages: set[int] = set()
                for page in alternative.get("pages", []):
                    try:
                        pages.add(int(page))
                    except (TypeError, ValueError):
                        continue
                if file_id and pages:
                    alternatives.append(EvidenceLocation(file_id, frozenset(pages)))
            if alternatives:
                evidence_groups.append(EvidenceGroup(
                    group_id=str(group.get("id") or f"group-{index}"),
                    alternatives=tuple(alternatives),
                    terms=frozenset(
                        term for term in group.get("terms", [])
                        if isinstance(term, str) and term.strip()
                    ),
                ))

        return cls(
            query=item["query"],
            relevant_doc_ids=set(item["relevant_doc_ids"]),
            sample_id=item.get("id"),
            relevant_chunks=set(item.get("relevant_chunks", [])) or None,
            expected_terms=set(item.get("expected_terms", [])) or None,
            relevant_evidence_pages=relevant_evidence_pages or None,
            evidence_groups=tuple(evidence_groups) or None,
            metadata=item.get("metadata"),
        )


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics for a single query."""

    recall_at_k: dict[int, float]  # K -> recall value
    precision_at_k: dict[int, float]  # K -> precision value
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain
    term_coverage: float = 0.0  # Compatibility: all retrieved chunks
    term_coverage_at_k: dict[int, float] | None = None
    evidence_page_recall_at_k: dict[int, float] | None = None
    evidence_group_recall_at_k: dict[int, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "term_coverage": self.term_coverage,
            "term_coverage_at_k": self.term_coverage_at_k,
            "evidence_page_recall_at_k": self.evidence_page_recall_at_k,
            "evidence_group_recall_at_k": self.evidence_group_recall_at_k,
        }


@dataclass
class EvaluationResult:
    """Aggregated evaluation results."""

    num_queries: int
    num_evidence_group_queries: int
    avg_recall_at_k: dict[int, float]
    avg_precision_at_k: dict[int, float]
    avg_mrr: float
    avg_ndcg: float
    avg_term_coverage: float
    avg_term_coverage_at_k: dict[int, float]
    avg_evidence_page_recall_at_k: dict[int, float]
    avg_evidence_group_recall_at_k: dict[int, float]
    per_query_results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_queries": self.num_queries,
            "num_evidence_group_queries": self.num_evidence_group_queries,
            "avg_recall_at_k": self.avg_recall_at_k,
            "avg_precision_at_k": self.avg_precision_at_k,
            "avg_mrr": self.avg_mrr,
            "avg_ndcg": self.avg_ndcg,
            "avg_term_coverage": self.avg_term_coverage,
            "avg_term_coverage_at_k": self.avg_term_coverage_at_k,
            "avg_evidence_page_recall_at_k": self.avg_evidence_page_recall_at_k,
            "avg_evidence_group_recall_at_k": self.avg_evidence_group_recall_at_k,
            "per_query_results": self.per_query_results,
        }

    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            "=" * 60,
            "RAG Retrieval Evaluation Results",
            "=" * 60,
            f"Number of queries: {self.num_queries}",
            f"Queries with evidence groups: {self.num_evidence_group_queries}",
            "",
            "Recall@K:",
        ]
        for k, v in sorted(self.avg_recall_at_k.items()):
            lines.append(f"  Recall@{k}: {v:.4f}")

        lines.append("\nPrecision@K:")
        for k, v in sorted(self.avg_precision_at_k.items()):
            lines.append(f"  Precision@{k}: {v:.4f}")

        lines.extend([
            f"\nMRR (Mean Reciprocal Rank): {self.avg_mrr:.4f}",
            f"NDCG: {self.avg_ndcg:.4f}",
            f"Expected term coverage (all retrieved chunks): {self.avg_term_coverage:.4f}",
        ])

        if self.avg_term_coverage_at_k:
            lines.append("\nExpected Term Coverage@K:")
            for k, v in sorted(self.avg_term_coverage_at_k.items()):
                lines.append(f"  Term Coverage@{k}: {v:.4f}")

        if self.avg_evidence_page_recall_at_k:
            lines.append("\nEvidence Page Recall@K:")
            for k, v in sorted(self.avg_evidence_page_recall_at_k.items()):
                lines.append(f"  Evidence Page Recall@{k}: {v:.4f}")

        if self.avg_evidence_group_recall_at_k:
            lines.append("\nEvidence Group Recall@K:")
            for k, v in sorted(self.avg_evidence_group_recall_at_k.items()):
                lines.append(f"  Evidence Group Recall@{k}: {v:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class QueryEvaluation:
    """Evaluation output for one query."""

    metrics: RetrievalMetrics
    retrieved_documents: list[str]
    retrieved_sources: list[dict[str, Any]]
    hit_rank: int | None
    evidence_hit_rank: int | None
    evidence_group_hit_ranks: dict[str, int | None]


class RAGEvaluator:
    """Evaluator for RAG retrieval quality."""

    def __init__(
        self,
        k_values: list[int] | None = None,
        retriever: Any | None = None,
        topic: str | None = None,
    ):
        """Initialize the evaluator.

        Args:
            k_values: List of K values for Recall@K and Precision@K.
                Defaults to [1, 3, 5, 10].
            retriever: Optional retriever object for tests or experiments.
            topic: Optional corpus topic passed to the retriever.
        """
        self.k_values = k_values or [1, 3, 5, 10]
        self.retriever = retriever or get_retriever()
        self.topic = topic.strip() if topic and topic.strip() else None

    def calculate_recall_at_k(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int,
    ) -> float:
        """Calculate Recall@K.

        Recall@K = |relevant ∩ retrieved@K| / |relevant|

        Args:
            retrieved_ids: List of retrieved document IDs (in rank order)
            relevant_ids: Set of relevant document IDs (ground truth)
            k: Number of top results to consider

        Returns:
            Recall@K value between 0 and 1
        """
        if not relevant_ids:
            return 0.0

        top_k = set(retrieved_ids[:k])
        hits = len(top_k & relevant_ids)

        return hits / len(relevant_ids)

    def calculate_precision_at_k(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int,
    ) -> float:
        """Calculate Precision@K.

        Precision@K = |relevant ∩ retrieved@K| / K

        Args:
            retrieved_ids: List of retrieved document IDs (in rank order)
            relevant_ids: Set of relevant document IDs (ground truth)
            k: Number of top results to consider

        Returns:
            Precision@K value between 0 and 1
        """
        if k == 0:
            return 0.0

        top_k = set(retrieved_ids[:k])
        hits = len(top_k & relevant_ids)

        return hits / k

    def calculate_mrr(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
    ) -> float:
        """Calculate Mean Reciprocal Rank.

        MRR = 1 / rank of first relevant document

        Args:
            retrieved_ids: List of retrieved document IDs (in rank order)
            relevant_ids: Set of relevant document IDs (ground truth)

        Returns:
            MRR value between 0 and 1
        """
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank

        return 0.0

    def calculate_ndcg(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int = 10,
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain.

        NDCG = DCG / IDCG

        Where:
        DCG = Σ (relevance_i / log2(i + 1)) for i in top-k
        IDCG = ideal DCG (all relevant docs at top)

        Args:
            retrieved_ids: List of retrieved document IDs (in rank order)
            relevant_ids: Set of relevant document IDs (ground truth)
            k: Number of top results to consider

        Returns:
            NDCG value between 0 and 1
        """
        import math

        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], start=1):
            if doc_id in relevant_ids:
                dcg += 1.0 / math.log2(i + 1)

        # Calculate IDCG (ideal case: all relevant docs at top)
        idcg = 0.0
        for i in range(1, min(len(relevant_ids), k) + 1):
            idcg += 1.0 / math.log2(i + 1)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def evaluate_single_query(
        self,
        sample: EvaluationSample,
        top_k: int = 20,
    ) -> RetrievalMetrics:
        """Evaluate a single query.

        Args:
            sample: Evaluation sample with query and ground truth
            top_k: Number of documents to retrieve

        Returns:
            RetrievalMetrics for this query
        """
        return self._evaluate_query(sample, top_k).metrics

    def _evaluate_query(
        self,
        sample: EvaluationSample,
        top_k: int = 20,
    ) -> QueryEvaluation:
        """Evaluate one query and keep retrieved evidence for diagnostics."""
        retrieve_kwargs: dict[str, Any] = {"top_k": top_k}
        if self.topic:
            retrieve_kwargs["topic"] = self.topic
        context = self.retriever.retrieve(sample.query, **retrieve_kwargs)
        retrieved_ids = self._unique_doc_ids(context)

        recall_at_k = {}
        precision_at_k = {}

        for k in self.k_values:
            recall_at_k[k] = self.calculate_recall_at_k(
                retrieved_ids, sample.relevant_doc_ids, k
            )
            precision_at_k[k] = self.calculate_precision_at_k(
                retrieved_ids, sample.relevant_doc_ids, k
            )

        mrr = self.calculate_mrr(retrieved_ids, sample.relevant_doc_ids)
        ndcg = self.calculate_ndcg(retrieved_ids, sample.relevant_doc_ids)
        term_coverage = self.calculate_term_coverage(context, sample.expected_terms)
        term_coverage_at_k = {
            k: self.calculate_term_coverage(context, sample.expected_terms, top_k=k)
            for k in self.k_values
        }
        evidence_page_recall_at_k = None
        if sample.relevant_evidence_pages:
            evidence_page_recall_at_k = {
                k: self.calculate_evidence_page_recall_at_k(
                    context,
                    sample.relevant_evidence_pages,
                    k,
                )
                for k in self.k_values
            }
        evidence_group_recall_at_k = None
        if sample.evidence_groups:
            evidence_group_recall_at_k = {
                k: self.calculate_evidence_group_recall_at_k(
                    context,
                    sample.evidence_groups,
                    k,
                )
                for k in self.k_values
            }

        metrics = RetrievalMetrics(
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            mrr=mrr,
            ndcg=ndcg,
            term_coverage=term_coverage,
            term_coverage_at_k=term_coverage_at_k,
            evidence_page_recall_at_k=evidence_page_recall_at_k,
            evidence_group_recall_at_k=evidence_group_recall_at_k,
        )
        hit_rank = self._first_hit_rank(retrieved_ids, sample.relevant_doc_ids)
        evidence_hit_rank = self._first_evidence_hit_rank(
            context,
            sample.relevant_evidence_pages,
        )
        evidence_group_hit_ranks = {
            group.group_id: self._first_evidence_group_hit_rank(context, group)
            for group in (sample.evidence_groups or ())
        }

        return QueryEvaluation(
            metrics=metrics,
            retrieved_documents=retrieved_ids,
            retrieved_sources=self._retrieved_sources(context),
            hit_rank=hit_rank,
            evidence_hit_rank=evidence_hit_rank,
            evidence_group_hit_ranks=evidence_group_hit_ranks,
        )

    def evaluate(
        self,
        samples: list[EvaluationSample],
        top_k: int = 20,
    ) -> EvaluationResult:
        """Evaluate multiple queries.

        Args:
            samples: List of evaluation samples
            top_k: Number of documents to retrieve per query

        Returns:
            EvaluationResult with aggregated metrics
        """
        all_recall_at_k = {k: [] for k in self.k_values}
        all_precision_at_k = {k: [] for k in self.k_values}
        all_mrr = []
        all_ndcg = []
        all_term_coverage = []
        all_term_coverage_at_k = {k: [] for k in self.k_values}
        all_evidence_page_recall_at_k = {k: [] for k in self.k_values}
        all_evidence_group_recall_at_k = {k: [] for k in self.k_values}
        per_query_results = []

        for sample in samples:
            query_eval = self._evaluate_query(sample, top_k)
            metrics = query_eval.metrics

            for k in self.k_values:
                all_recall_at_k[k].append(metrics.recall_at_k[k])
                all_precision_at_k[k].append(metrics.precision_at_k[k])

            all_mrr.append(metrics.mrr)
            all_ndcg.append(metrics.ndcg)
            all_term_coverage.append(metrics.term_coverage)
            if metrics.term_coverage_at_k:
                for k in self.k_values:
                    all_term_coverage_at_k[k].append(metrics.term_coverage_at_k[k])
            if metrics.evidence_page_recall_at_k:
                for k in self.k_values:
                    all_evidence_page_recall_at_k[k].append(
                        metrics.evidence_page_recall_at_k[k]
                    )
            if metrics.evidence_group_recall_at_k:
                for k in self.k_values:
                    all_evidence_group_recall_at_k[k].append(
                        metrics.evidence_group_recall_at_k[k]
                    )

            per_query_results.append({
                "id": sample.sample_id,
                "query": sample.query,
                "relevant_doc_ids": list(sample.relevant_doc_ids),
                "relevant_evidence_pages": {
                    file_id: sorted(pages)
                    for file_id, pages in (sample.relevant_evidence_pages or {}).items()
                },
                "evidence_groups": [
                    group.to_dict() for group in (sample.evidence_groups or ())
                ],
                "expected_terms": sorted(sample.expected_terms or []),
                "hit_rank": query_eval.hit_rank,
                "evidence_hit_rank": query_eval.evidence_hit_rank,
                "evidence_group_hit_ranks": query_eval.evidence_group_hit_ranks,
                "retrieved_documents": query_eval.retrieved_documents,
                "retrieved_sources": query_eval.retrieved_sources,
                "metrics": metrics.to_dict(),
                "metadata": sample.metadata or {},
            })

        # Calculate averages
        avg_recall_at_k = {
            k: sum(v) / len(v) if v else 0.0
            for k, v in all_recall_at_k.items()
        }
        avg_precision_at_k = {
            k: sum(v) / len(v) if v else 0.0
            for k, v in all_precision_at_k.items()
        }
        avg_mrr = sum(all_mrr) / len(all_mrr) if all_mrr else 0.0
        avg_ndcg = sum(all_ndcg) / len(all_ndcg) if all_ndcg else 0.0
        avg_term_coverage = (
            sum(all_term_coverage) / len(all_term_coverage)
            if all_term_coverage
            else 0.0
        )
        avg_term_coverage_at_k = {
            k: sum(values) / len(values)
            for k, values in all_term_coverage_at_k.items()
            if values
        }
        avg_evidence_page_recall_at_k = {
            k: sum(values) / len(values)
            for k, values in all_evidence_page_recall_at_k.items()
            if values
        }
        avg_evidence_group_recall_at_k = {
            k: sum(values) / len(values)
            for k, values in all_evidence_group_recall_at_k.items()
            if values
        }

        return EvaluationResult(
            num_queries=len(samples),
            num_evidence_group_queries=sum(
                1 for sample in samples if sample.evidence_groups
            ),
            avg_recall_at_k=avg_recall_at_k,
            avg_precision_at_k=avg_precision_at_k,
            avg_mrr=avg_mrr,
            avg_ndcg=avg_ndcg,
            avg_term_coverage=avg_term_coverage,
            avg_term_coverage_at_k=avg_term_coverage_at_k,
            avg_evidence_page_recall_at_k=avg_evidence_page_recall_at_k,
            avg_evidence_group_recall_at_k=avg_evidence_group_recall_at_k,
            per_query_results=per_query_results,
        )

    def evaluate_from_file(
        self,
        filepath: Path | str,
        top_k: int = 20,
    ) -> EvaluationResult:
        """Evaluate from a JSON file containing evaluation samples.

        Expected format:
        [
            {
                "query": "查询文本",
                "relevant_doc_ids": ["doc_id_1", "doc_id_2"],
                "relevant_chunks": ["chunk_id_1"],  // optional
                "metadata": {}  // optional
            }
        ]

        Args:
            filepath: Path to the JSON file
            top_k: Number of documents to retrieve per query

        Returns:
            EvaluationResult with aggregated metrics
        """
        filepath = Path(filepath)

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        samples = [EvaluationSample.from_dict(item) for item in data]

        return self.evaluate(samples, top_k)

    def calculate_term_coverage(
        self,
        context: RetrievedContext,
        expected_terms: set[str] | None,
        top_k: int | None = None,
    ) -> float:
        """Calculate expected-term coverage in all or the first K chunks."""
        if not expected_terms:
            return 0.0

        evidence_parts: list[str] = []
        results = context.results if top_k is None else context.results[:top_k]
        for result in results:
            evidence_parts.extend([
                result.content,
                str(result.metadata.get("file_name", "")),
                str(result.metadata.get("section_title", "")),
            ])

        evidence_text = self._normalize_evidence_text("\n".join(evidence_parts))
        matched = sum(
            1 for term in expected_terms
            if self._normalize_evidence_text(term) in evidence_text
        )
        return matched / len(expected_terms)

    def calculate_evidence_group_recall_at_k(
        self,
        context: RetrievedContext,
        evidence_groups: tuple[EvidenceGroup, ...],
        k: int,
    ) -> float:
        """Calculate fact-group recall using alternative pages and anchor terms."""
        if not evidence_groups:
            return 0.0

        hits = sum(
            1
            for group in evidence_groups
            if self._evidence_group_is_hit(context.results[:k], group)
        )
        return hits / len(evidence_groups)

    def _evidence_group_is_hit(
        self,
        results: list[Any],
        group: EvidenceGroup,
    ) -> bool:
        """Return whether matched locations contain every anchor term for a group."""
        evidence_parts: list[str] = []
        for result in results:
            file_id = str(result.metadata.get("file_id") or "")
            result_pages = self._metadata_pages(result.metadata)
            if not any(
                file_id == alternative.file_id
                and bool(result_pages & alternative.pages)
                for alternative in group.alternatives
            ):
                continue
            evidence_parts.extend([
                result.content,
                str(result.metadata.get("section_title", "")),
            ])

        if not evidence_parts:
            return False
        if not group.terms:
            return True

        evidence_text = self._normalize_evidence_text("\n".join(evidence_parts))
        return all(
            self._normalize_evidence_text(term) in evidence_text
            for term in group.terms
        )

    def _first_evidence_group_hit_rank(
        self,
        context: RetrievedContext,
        group: EvidenceGroup,
    ) -> int | None:
        """Return the first prefix rank that fully covers an evidence group."""
        for rank in range(1, len(context.results) + 1):
            if self._evidence_group_is_hit(context.results[:rank], group):
                return rank
        return None

    def calculate_evidence_page_recall_at_k(
        self,
        context: RetrievedContext,
        relevant_evidence_pages: dict[str, set[int]],
        k: int,
    ) -> float:
        """Calculate recall of annotated evidence pages within the first K chunks."""
        expected = {
            (file_id, page)
            for file_id, pages in relevant_evidence_pages.items()
            for page in pages
        }
        if not expected:
            return 0.0

        retrieved: set[tuple[str, int]] = set()
        for result in context.results[:k]:
            file_id = str(result.metadata.get("file_id") or "")
            for page in self._metadata_pages(result.metadata):
                retrieved.add((file_id, page))

        return len(expected & retrieved) / len(expected)

    def _first_evidence_hit_rank(
        self,
        context: RetrievedContext,
        relevant_evidence_pages: dict[str, set[int]] | None,
    ) -> int | None:
        """Return the first chunk rank that overlaps an annotated evidence page."""
        if not relevant_evidence_pages:
            return None
        expected = {
            (file_id, page)
            for file_id, pages in relevant_evidence_pages.items()
            for page in pages
        }
        for rank, result in enumerate(context.results, start=1):
            file_id = str(result.metadata.get("file_id") or "")
            if any(
                (file_id, page) in expected
                for page in self._metadata_pages(result.metadata)
            ):
                return rank
        return None

    def _metadata_pages(self, metadata: dict[str, Any]) -> set[int]:
        """Expand page metadata into a set of page numbers."""
        start_value = metadata.get("page_start") or metadata.get("page_number")
        end_value = metadata.get("page_end") or start_value
        try:
            start = int(start_value)
            end = int(end_value)
        except (TypeError, ValueError):
            return set()
        if end < start:
            start, end = end, start
        return set(range(start, end + 1))

    def _normalize_evidence_text(self, text: str) -> str:
        """Normalize PDF whitespace and punctuation before evidence matching."""
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", text.lower()).replace("_", "")

    def _unique_doc_ids(self, context: RetrievedContext) -> list[str]:
        """Return first-seen file IDs from chunk-level retrieval results."""
        doc_ids: list[str] = []
        seen: set[str] = set()
        for result in context.results:
            file_id = result.metadata.get("file_id", "")
            if not file_id or file_id in seen:
                continue
            seen.add(file_id)
            doc_ids.append(file_id)
        return doc_ids

    def _retrieved_sources(self, context: RetrievedContext) -> list[dict[str, Any]]:
        """Build compact source diagnostics for evaluation output."""
        sources: list[dict[str, Any]] = []
        for rank, result in enumerate(context.results, start=1):
            metadata = result.metadata
            sources.append({
                "rank": rank,
                "file_id": metadata.get("file_id", ""),
                "file_name": metadata.get("file_name", ""),
                "page_start": metadata.get("page_start") or metadata.get("page_number"),
                "page_end": metadata.get("page_end") or metadata.get("page_number"),
                "content_type": metadata.get("content_type", "text"),
                "score": round(result.score, 6),
                "search_type": metadata.get("search_type"),
                "rerank_score": metadata.get("rerank_score"),
                "snippet": result.content[:180].replace("\n", " "),
            })
        return sources

    def _first_hit_rank(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
    ) -> int | None:
        """Return 1-based rank of the first relevant document."""
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return rank
        return None


def create_sample_evaluation_dataset(
    output_path: Path | str,
    samples: list[dict[str, Any]] | None = None,
) -> None:
    """Create a sample evaluation dataset file.

    Args:
        output_path: Path to save the JSON file
        samples: Optional list of sample queries. If not provided,
                 creates a template with example format.
    """
    if samples is None:
        samples = [
            {
                "query": "中信海直2025年业绩如何？",
                "relevant_doc_ids": ["YOUR_DOC_ID_1"],
                "metadata": {"category": "业绩分析"},
            },
            {
                "query": "低空经济发展趋势",
                "relevant_doc_ids": ["YOUR_DOC_ID_1", "YOUR_DOC_ID_2"],
                "metadata": {"category": "行业分析"},
            },
        ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    logger.info(f"Sample evaluation dataset saved to {output_path}")


def create_evaluation_dataset_from_reports(
    reports_dir: Path | str,
    output_path: Path | str,
    min_file_size: int = 1024,
) -> list[dict[str, Any]]:
    """Create a filename-grounded evaluation dataset from report PDFs.

    This is intended as a bootstrap dataset. It uses report filenames as weak
    labels, so it is best paired with a hand-curated dataset for exact facts.
    """
    reports_dir = Path(reports_dir)
    samples: list[dict[str, Any]] = []

    for pdf_path in sorted(reports_dir.glob("*.pdf")):
        if pdf_path.stat().st_size < min_file_size:
            continue

        file_id, title = _split_report_filename(pdf_path)
        if not file_id or not title:
            continue

        company = _extract_company(title)
        topic = _strip_company(title, company)
        expected_terms = _expected_terms_from_title(title, company)
        query = f"{company}{topic}" if company else topic

        samples.append({
            "query": query,
            "relevant_doc_ids": [file_id],
            "expected_terms": expected_terms,
            "metadata": {
                "category": _infer_category(title),
                "difficulty": "easy",
                "company": company,
                "file_name": pdf_path.name,
                "source": "reports_dir_filename",
            },
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    logger.info(f"Created {len(samples)} report evaluation samples at {output_path}")
    return samples


def _split_report_filename(pdf_path: Path) -> tuple[str, str]:
    """Split '<file_id>_<title>.pdf' report filenames."""
    stem = pdf_path.stem
    if "_" not in stem:
        return "", stem
    file_id, title = stem.split("_", 1)
    return file_id, title


def _extract_company(title: str) -> str:
    """Extract company name wrapped by Chinese brackets."""
    match = re.search(r"【([^】]+)】", title)
    return match.group(1) if match else ""


def _strip_company(title: str, company: str) -> str:
    """Remove company marker from report title."""
    if company:
        title = title.replace(f"【{company}】", "", 1)
    return title.strip(" ：:，,")


def _expected_terms_from_title(title: str, company: str) -> list[str]:
    """Pick stable expected terms from a report title."""
    candidates = [
        company,
        "低空经济",
        "营收",
        "净利润",
        "利润",
        "同比",
        "增长",
        "通航",
        "无人物流",
        "连接器",
        "智慧交通",
        "商业航天",
        "空管",
        "机器人",
        "轴承",
        "风险",
        "估值",
    ]
    terms = [term for term in candidates if term and term in title]
    return list(dict.fromkeys(terms)) or ([company] if company else [])


def _infer_category(title: str) -> str:
    """Infer a coarse evaluation category from title text."""
    if any(term in title for term in ["营收", "净利润", "利润", "业绩", "三季报", "半年报", "中报"]):
        return "业绩分析"
    if any(term in title for term in ["布局", "低空经济", "新增长"]):
        return "业务布局"
    if any(term in title for term in ["首次覆盖", "公司信息更新"]):
        return "公司研究"
    return "综合查询"


# Convenience function
def evaluate_rag(
    samples: list[EvaluationSample] | Path | str,
    k_values: list[int] | None = None,
    top_k: int = 20,
) -> EvaluationResult:
    """Evaluate RAG retrieval quality.

    Args:
        samples: List of EvaluationSample objects or path to JSON file
        k_values: List of K values for Recall@K and Precision@K
        top_k: Number of documents to retrieve per query

    Returns:
        EvaluationResult with metrics
    """
    evaluator = RAGEvaluator(k_values)

    if isinstance(samples, (str, Path)):
        return evaluator.evaluate_from_file(samples, top_k)

    return evaluator.evaluate(samples, top_k)
