"""Retriever for RAG system."""

import logging
import re
from dataclasses import dataclass
from typing import Any

from app.core import get_settings
from app.rag.vectorstore import KEYWORD_HINT_TERMS, VectorStore, get_vector_store

logger = logging.getLogger(__name__)

QUERY_STOPWORDS = {
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

TABLE_QUERY_TERMS = {
    "表格",
    "数据",
    "指标",
    "财务",
    "营收",
    "收入",
    "利润",
    "净利润",
    "毛利率",
    "费用率",
    "同比",
    "环比",
    "预测",
    "估值",
    "pe",
    "pb",
    "eps",
    "roe",
    "roa",
}

QUERY_REWRITE_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("盈利", "赚钱", "利润", "业绩"),
        (
            "盈利能力 毛利率 净利率 净利润",
            "盈利预测 EPS 归母净利润",
            "财务表现 收入 利润",
        ),
    ),
    (
        ("收入", "营收", "销售"),
        (
            "营业收入 收入增长 同比 环比",
            "收入结构 主营业务 产品收入",
        ),
    ),
    (
        ("风险", "不确定", "问题"),
        (
            "风险提示 行业风险 公司风险",
            "经营风险 财务风险 政策风险",
        ),
    ),
    (
        ("估值", "贵", "便宜", "pe", "pb", "市盈率", "市净率"),
        (
            "估值 PE PB 市盈率 市净率",
            "目标价 投资评级 估值水平",
        ),
    ),
    (
        ("预测", "未来", "展望", "空间"),
        (
            "盈利预测 收入预测 净利润预测 EPS",
            "未来展望 增长驱动 业绩预测",
        ),
    ),
    (
        ("毛利", "毛利率", "费用率", "净利率"),
        (
            "毛利率 费用率 净利率 盈利能力",
            "成本结构 毛利变化 费用控制",
        ),
    ),
    (
        ("现金流", "经营现金", "自由现金"),
        (
            "经营现金流 自由现金流 现金流量",
            "现金流质量 收现比 资本开支",
        ),
    ),
    (
        ("竞争", "行业", "格局", "份额"),
        (
            "竞争格局 市场份额 行业地位",
            "行业趋势 竞争优势 市占率",
        ),
    ),
)

TABLE_QUERY_REWRITE_VARIANT = "财务指标 营收 净利润 毛利率 EPS ROE 估值"

EXPLANATION_QUERY_TERMS = {
    "为什么",
    "为何",
    "原因",
    "因素",
    "承压",
    "阶段",
    "进展",
    "进度",
}

EXPLANATION_CONTENT_TERMS = {
    "经营分析",
    "原因",
    "因素",
    "主要系",
    "由于",
    "导致",
    "承压",
    "其一",
    "其二",
    "其三",
    "客户导入",
    "验证阶段",
    "poc",
    "小批量",
    "尚未",
    "未起量",
    "账期",
    "损失",
}

EXPLANATION_QUERY_REWRITES = (
    "经营分析 利润承压 原因 因素",
    "项目阶段 客户导入 验证 小批量 量产进展",
)

PROGRESS_QUERY_TERMS = {
    "合作",
    "送样",
    "小批量",
    "量产",
    "落地",
}

PROGRESS_CONTENT_TERMS = {
    "合作",
    "送样",
    "客户导入",
    "客户验证",
    "验证阶段",
    "小批量",
    "量产",
    "落地",
    "收购",
    "并购",
    "关节模组",
    "精密传动",
    "运动控制",
    "运控",
    "业务协同",
}

PROGRESS_QUERY_REWRITE = "合作 客户验证 送样 导入 小批量 量产 项目进展"
BUSINESS_PROGRESS_QUERY_REWRITE = "并购 收购 产品方案 核心零部件 业务协同 交付进展"
ROBOT_PROGRESS_QUERY_REWRITE = "机器人 关节模组 执行器 精密传动 运动控制 并购 收购"


def _coerce_page(value: Any) -> int | None:
    """Convert page metadata to int when possible."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _page_range_from_metadata(metadata: dict[str, Any]) -> tuple[int | None, int | None]:
    """Read page range metadata while keeping compatibility with page_number."""
    page_start = _coerce_page(metadata.get("page_start") or metadata.get("page_number"))
    page_end = _coerce_page(metadata.get("page_end") or page_start)
    return page_start, page_end


def _format_page_range(metadata: dict[str, Any]) -> str:
    """Format page metadata for human-readable citations."""
    page_start, page_end = _page_range_from_metadata(metadata)
    if not page_start:
        return ""
    if page_end and page_end != page_start:
        return f"第{page_start}-{page_end}页"
    return f"第{page_start}页"


@dataclass
class RetrievalResult:
    """A single retrieval result."""

    content: str
    metadata: dict[str, Any]
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
        }


@dataclass
class RetrievedContext:
    """Aggregated context from retrieval."""

    query: str
    results: list[RetrievalResult]
    sources: list[dict[str, Any]]

    def format_context(self, include_metadata: bool = True) -> str:
        """Format results as context string for LLM.

        Args:
            include_metadata: Whether to include metadata in the output.

        Returns:
            Formatted context string.
        """
        if not self.results:
            return "未找到相关信息。"

        context_parts: list[str] = []

        for i, result in enumerate(self.results, 1):
            if include_metadata:
                content_type = result.metadata.get("content_type")
                source_info = f"[来源: {result.metadata.get('file_name', '未知')}"
                if page_range := _format_page_range(result.metadata):
                    source_info += f", {page_range}"
                if section := result.metadata.get("section_title"):
                    source_info += f", 章节: {section}"
                if content_type == "table":
                    source_info += ", 类型: 表格"
                source_info += f", 相似度: {result.score:.2f}]"
                label = "表格片段" if content_type == "table" else "文档片段"
                context_parts.append(f"--- {label} {i} {source_info} ---\n{result.content}")
            else:
                context_parts.append(f"--- 文档片段 {i} ---\n{result.content}")

        return "\n\n".join(context_parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "sources": self.sources,
        }


class Retriever:
    """High-level retriever for document search.

    This class provides:
    - Semantic similarity search
    - Filtering by file, metadata
    - Result reranking and deduplication
    - Context formatting for LLM consumption
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        default_top_k: int = 5,
        score_threshold: float = 0.0,
        candidate_multiplier: int = 4,
        hybrid_search_enabled: bool = True,
        query_rewrite_enabled: bool = True,
        query_rewrite_max_variants: int = 3,
    ) -> None:
        """Initialize the retriever.

        Args:
            vector_store: VectorStore instance to use.
                Defaults to the singleton from get_vector_store().
            default_top_k: Default number of results to retrieve.
            score_threshold: Minimum similarity score threshold.
            candidate_multiplier: Number of candidates to recall before reranking.
            hybrid_search_enabled: Whether to combine vector and keyword retrieval.
            query_rewrite_enabled: Whether to expand user queries with report-style terms.
            query_rewrite_max_variants: Maximum query variants including the original query.
        """
        self._vector_store = vector_store or get_vector_store()
        self.default_top_k = default_top_k
        self.score_threshold = score_threshold
        self.candidate_multiplier = max(1, candidate_multiplier)
        self.hybrid_search_enabled = hybrid_search_enabled
        self.query_rewrite_enabled = query_rewrite_enabled
        self.query_rewrite_max_variants = max(1, query_rewrite_max_variants)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        file_ids: list[str] | None = None,
        score_threshold: float | None = None,
        topic: str | None = None,
    ) -> RetrievedContext:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query.
            top_k: Number of results to return.
                Defaults to default_top_k.
            file_ids: Optional list of file IDs to search within.
            score_threshold: Minimum similarity score.
                Defaults to self.score_threshold.
            topic: Optional corpus topic stored in document metadata.

        Returns:
            RetrievedContext with results and sources.
        """
        top_k = top_k or self.default_top_k
        score_threshold = score_threshold if score_threshold is not None else self.score_threshold

        topic = topic.strip() if topic and topic.strip() else None
        logger.info(
            "Retrieving for query: %s... (top_k=%s, topic=%s)",
            query[:50],
            top_k,
            topic or "all",
        )

        # Apply metadata filters before vector and keyword candidate recall.
        filter_conditions: list[dict[str, Any]] = []
        if file_ids and len(file_ids) == 1:
            filter_conditions.append({"file_id": file_ids[0]})
        if topic:
            filter_conditions.append({"topic": topic})

        filter_: dict[str, Any] | None = None
        if len(filter_conditions) == 1:
            filter_ = filter_conditions[0]
        elif len(filter_conditions) > 1:
            filter_ = {"$and": filter_conditions}

        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        if file_ids and len(file_ids) > 1:
            candidate_k *= 2

        query_variants = self._query_variants(query)
        vector_results: list[tuple[dict[str, Any], float]] = []
        keyword_results: list[tuple[dict[str, Any], float]] = []

        for variant_index, query_variant in enumerate(query_variants):
            variant_k = candidate_k if variant_index == 0 else max(top_k, top_k * 2)
            vector_results.extend(self._with_query_variant(
                results=self._vector_store.similarity_search_with_scores(
                    query=query_variant,
                    k=variant_k,
                    filter_=filter_,
                ),
                query_variant=query_variant,
            ))
            if self.hybrid_search_enabled:
                keyword_results.extend(self._with_query_variant(
                    results=self._keyword_search(
                        query=query_variant,
                        k=variant_k,
                        filter_=filter_,
                    ),
                    query_variant=query_variant,
                ))

        results_with_scores = self._merge_hybrid_results(
            vector_results=vector_results,
            keyword_results=keyword_results,
        )

        # Filter by file_ids if multiple provided
        if file_ids and len(file_ids) > 1:
            results_with_scores = [
                (r, s) for r, s in results_with_scores
                if r["metadata"].get("file_id") in file_ids
            ]

        # Filter by score threshold
        results_with_scores = [
            (r, s) for r, s in results_with_scores
            if s >= score_threshold
        ]

        results_with_scores = self._rerank_results(
            query=" ".join(query_variants),
            original_query=query,
            results_with_scores=results_with_scores,
            top_k=top_k,
        )

        # Convert to RetrievalResult
        retrieval_results: list[RetrievalResult] = [
            RetrievalResult(
                content=r["content"],
                metadata=r["metadata"],
                score=s,
            )
            for r, s in results_with_scores
        ]

        # Build unique source list at the file/page level so citations stay precise.
        sources: list[dict[str, Any]] = []
        seen_sources: set[tuple[str, Any, Any]] = set()

        for result in retrieval_results:
            file_id = result.metadata.get("file_id") or result.metadata.get("source") or "unknown"
            page_start, page_end = _page_range_from_metadata(result.metadata)
            source_key = (str(file_id), page_start, page_end)
            if source_key in seen_sources:
                continue

            sources.append({
                "file_id": file_id,
                "file_name": result.metadata.get("file_name", "Unknown"),
                "page_number": page_start,
                "page_start": page_start,
                "page_end": page_end,
                "section_title": result.metadata.get("section_title"),
                "content_type": result.metadata.get("content_type", "text"),
                "score": result.score,
            })
            seen_sources.add(source_key)

        logger.info(f"Retrieved {len(retrieval_results)} results from {len(sources)} sources")

        return RetrievedContext(
            query=query,
            results=retrieval_results,
            sources=sources,
        )

    def _keyword_search(
        self,
        query: str,
        k: int,
        filter_: dict[str, Any] | None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Run keyword retrieval when the vector store supports it."""
        keyword_search = getattr(self._vector_store, "keyword_search_with_scores", None)
        if not callable(keyword_search):
            return []

        try:
            results = keyword_search(query=query, k=k, filter_=filter_)
        except Exception as e:
            logger.warning(f"Keyword search failed, falling back to vector-only: {e}")
            return []

        return results if isinstance(results, list) else []

    def _with_query_variant(
        self,
        results: list[tuple[dict[str, Any], float]],
        query_variant: str,
    ) -> list[tuple[dict[str, Any], float]]:
        """Tag retrieval hits with the query variant that found them."""
        tagged_results: list[tuple[dict[str, Any], float]] = []

        for result, score in results:
            metadata = dict(result.get("metadata") or {})
            metadata["query_variant"] = query_variant
            tagged_results.append((
                {
                    **result,
                    "metadata": metadata,
                },
                score,
            ))

        return tagged_results

    def _merge_hybrid_results(
        self,
        vector_results: list[tuple[dict[str, Any], float]],
        keyword_results: list[tuple[dict[str, Any], float]],
    ) -> list[tuple[dict[str, Any], float]]:
        """Fuse vector and keyword retrieval results with RRF-style ranking."""
        merged: dict[str, dict[str, Any]] = {}

        self._accumulate_ranked_results(
            merged=merged,
            results=vector_results,
            source="vector",
        )
        self._accumulate_ranked_results(
            merged=merged,
            results=keyword_results,
            source="keyword",
        )

        fused: list[tuple[dict[str, Any], float, float]] = []
        for item in merged.values():
            metadata = dict(item["result"].get("metadata") or {})
            vector_score = item.get("vector_score")
            keyword_score = item.get("keyword_score")
            base_score = max(
                score for score in [vector_score, keyword_score] if score is not None
            )
            hybrid_rank_score = base_score + item["rrf_score"]
            search_types = sorted(item["search_types"])
            query_variants = sorted(item["query_variants"])
            metadata["search_type"] = "+".join(search_types)
            metadata["hybrid_score"] = round(hybrid_rank_score, 6)
            if query_variants:
                metadata["query_variants"] = " | ".join(query_variants)
                metadata["matched_query_count"] = len(query_variants)
            if vector_score is not None:
                metadata["vector_score"] = round(vector_score, 6)
            if keyword_score is not None:
                metadata["keyword_score"] = round(keyword_score, 6)

            fused.append((
                {
                    "id": item["result"].get("id", ""),
                    "content": item["result"].get("content", ""),
                    "metadata": metadata,
                },
                base_score,
                hybrid_rank_score,
            ))

        fused.sort(key=lambda item: item[2], reverse=True)
        return [(result, score) for result, score, _ in fused]

    def _accumulate_ranked_results(
        self,
        merged: dict[str, dict[str, Any]],
        results: list[tuple[dict[str, Any], float]],
        source: str,
    ) -> None:
        """Accumulate ranked results for hybrid fusion."""
        for rank, (result, score) in enumerate(results, start=1):
            key = self._result_key(result)
            if key not in merged:
                merged[key] = {
                    "result": result,
                    "rrf_score": 0.0,
                    "search_types": set(),
                    "query_variants": set(),
                    "vector_score": None,
                    "keyword_score": None,
                }

            query_variant = (result.get("metadata") or {}).get("query_variant")
            if query_variant:
                merged[key]["query_variants"].add(query_variant)
            merged[key]["rrf_score"] += 1 / (60 + rank)
            merged[key]["search_types"].add(source)
            merged[key][f"{source}_score"] = max(
                merged[key].get(f"{source}_score") or 0,
                score,
            )

    def _result_key(self, result: dict[str, Any]) -> str:
        """Build a stable key for deduplicating vector and keyword results."""
        if result.get("id"):
            return f"id:{result['id']}"

        metadata = result.get("metadata") or {}
        file_id = metadata.get("file_id") or metadata.get("source") or "unknown"
        page_start, page_end = _page_range_from_metadata(metadata)
        content = result.get("content", "")
        return f"{file_id}:{page_start}:{page_end}:{hash(content[:200])}"

    def _rerank_results(
        self,
        query: str,
        original_query: str,
        results_with_scores: list[tuple[dict[str, Any], float]],
        top_k: int,
    ) -> list[tuple[dict[str, Any], float]]:
        """Rerank vector candidates with lightweight lexical and metadata signals."""
        query_terms = self._query_terms(query)
        target_companies = self._target_companies(original_query, results_with_scores)
        progress_query = self._is_progress_query(original_query)
        explanation_query = (
            self._is_explanation_query(original_query)
            and not progress_query
        )
        table_query = (
            self._is_table_or_metric_query(query_terms)
            and not explanation_query
            and not progress_query
        )
        reranked: list[tuple[dict[str, Any], float, float]] = []

        for result, score in results_with_scores:
            metadata = dict(result.get("metadata") or {})
            content = result.get("content", "")
            rerank_score, reasons = self._rerank_score(
                query_terms=query_terms,
                table_query=table_query,
                explanation_query=explanation_query,
                progress_query=progress_query,
                content=content,
                metadata=metadata,
                vector_score=score,
                target_companies=target_companies,
            )
            metadata["retrieval_score"] = round(score, 6)
            metadata["rerank_score"] = round(rerank_score, 6)
            if reasons:
                metadata["rerank_reasons"] = ",".join(reasons)

            reranked.append((
                {
                    "content": content,
                    "metadata": metadata,
                },
                score,
                rerank_score,
            ))

        reranked.sort(key=lambda item: item[2], reverse=True)
        selected = self._select_diverse_results(
            reranked=reranked,
            top_k=top_k,
            target_companies=target_companies,
        )
        return [(result, score) for result, score, _ in selected]

    def _rerank_score(
        self,
        query_terms: list[str],
        table_query: bool,
        explanation_query: bool,
        progress_query: bool,
        content: str,
        metadata: dict[str, Any],
        vector_score: float,
        target_companies: set[str],
    ) -> tuple[float, list[str]]:
        """Compute rerank score and short reason labels."""
        normalized_content = content.lower()
        section_title = str(metadata.get("section_title") or "").lower()
        file_name = str(metadata.get("file_name") or "").lower()
        company_name = self._company_from_file_name(file_name)
        content_type = metadata.get("content_type", "text")

        score = vector_score
        reasons: list[str] = []

        coverage = self._term_coverage(query_terms, normalized_content)
        if coverage > 0:
            score += min(0.12, coverage * 0.12)
            reasons.append("keyword")

        section_coverage = self._term_coverage(query_terms, section_title)
        if section_coverage > 0:
            score += min(0.08, section_coverage * 0.08)
            reasons.append("section")

        file_coverage = self._term_coverage(query_terms, file_name)
        if file_coverage > 0:
            score += min(0.16, file_coverage * 0.16)
            reasons.append("filename")

        if company_name and company_name in target_companies:
            score += 0.45
            reasons.append("company")
        elif target_companies and company_name:
            score -= 0.08
            reasons.append("other_company")

        title_term_hits = self._title_term_hits(query_terms, file_name)
        if title_term_hits:
            score += min(0.12, title_term_hits * 0.035)
            reasons.append("title")

        if progress_query:
            progress_hits = sum(
                1 for term in PROGRESS_CONTENT_TERMS
                if term in normalized_content
            )
            if progress_hits:
                score += min(0.32, progress_hits * 0.045)
                reasons.append("progress")
        elif explanation_query and content_type == "table":
            score -= 0.08
            reasons.append("explanation_table_penalty")
        elif explanation_query:
            explanation_hits = sum(
                1 for term in EXPLANATION_CONTENT_TERMS
                if term in normalized_content
            )
            if explanation_hits:
                score += min(0.24, explanation_hits * 0.04)
                reasons.append("explanation")
        elif content_type == "table" and table_query:
            score += 0.08
            reasons.append("table")
        elif content_type == "table":
            score += 0.02
            reasons.append("structured")

        return score, reasons

    def _target_companies(
        self,
        query: str,
        results_with_scores: list[tuple[dict[str, Any], float]],
    ) -> set[str]:
        """Find report companies explicitly named in the user query."""
        normalized_query = query.lower()
        companies: set[str] = set()

        for result, _ in results_with_scores:
            metadata = result.get("metadata") or {}
            file_name = str(metadata.get("file_name") or "").lower()
            company_name = self._company_from_file_name(file_name)
            if company_name and company_name in normalized_query:
                companies.add(company_name)

        return companies

    def _select_diverse_results(
        self,
        reranked: list[tuple[dict[str, Any], float, float]],
        top_k: int,
        target_companies: set[str],
    ) -> list[tuple[dict[str, Any], float, float]]:
        """Limit non-target report dominance for explicit-company questions."""
        if not target_companies:
            return reranked[:top_k]

        non_target_limit = max(1, min(2, top_k // 2))
        non_target_counts: dict[str, int] = {}
        selected: list[tuple[dict[str, Any], float, float]] = []
        selected_indexes: set[int] = set()
        seeded_companies: set[str] = set()
        deferred: list[tuple[dict[str, Any], float, float]] = []

        # Preserve the best candidate for every explicitly named company first.
        for index, item in enumerate(reranked):
            result = item[0]
            metadata = result.get("metadata") or {}
            file_name = str(metadata.get("file_name") or "").lower()
            company_name = self._company_from_file_name(file_name)
            if company_name not in target_companies or company_name in seeded_companies:
                continue

            selected.append(item)
            selected_indexes.add(index)
            seeded_companies.add(company_name)
            if len(selected) >= top_k:
                return selected[:top_k]

        for index, item in enumerate(reranked):
            if index in selected_indexes:
                continue

            result = item[0]
            metadata = result.get("metadata") or {}
            file_name = str(metadata.get("file_name") or "").lower()
            company_name = self._company_from_file_name(file_name)
            is_target_report = company_name in target_companies
            report_key = str(
                metadata.get("file_id")
                or metadata.get("source")
                or metadata.get("file_name")
                or ""
            )

            if is_target_report or not report_key:
                selected.append(item)
            elif non_target_counts.get(report_key, 0) < non_target_limit:
                selected.append(item)
                non_target_counts[report_key] = non_target_counts.get(report_key, 0) + 1
            else:
                deferred.append(item)

            if len(selected) >= top_k:
                break

        if len(selected) < top_k:
            selected.extend(deferred[:top_k - len(selected)])

        return selected[:top_k]

    def _query_terms(self, query: str) -> list[str]:
        """Extract lightweight query terms for reranking."""
        normalized = query.lower()
        raw_terms = re.split(r"[\s,，。；;：:、/\\()\[\]{}<>《》\"'“”！？?+\-=]+", normalized)
        terms: list[str] = []

        for term in raw_terms:
            term = term.strip()
            if len(term) < 2 or term in QUERY_STOPWORDS:
                continue
            terms.append(term)
            terms.extend(self._expand_chinese_query_terms(term))

        for table_term in TABLE_QUERY_TERMS:
            if table_term in normalized and table_term not in terms:
                terms.append(table_term)

        return list(dict.fromkeys(terms))

    def _query_variants(self, query: str) -> list[str]:
        """Rewrite a user query into a few retrieval-oriented variants."""
        original_query = query.strip()
        if not original_query or not self.query_rewrite_enabled:
            return [query]

        normalized = original_query.lower()
        variants = [original_query]

        if self._is_progress_query(original_query):
            variants.append(PROGRESS_QUERY_REWRITE)
            if "具身智能" in normalized or "机器人" in normalized:
                variants.append(ROBOT_PROGRESS_QUERY_REWRITE)
            else:
                variants.append(BUSINESS_PROGRESS_QUERY_REWRITE)
        elif self._is_explanation_query(original_query):
            variants.extend(EXPLANATION_QUERY_REWRITES)
        else:
            for triggers, rewrites in QUERY_REWRITE_RULES:
                if any(trigger in normalized for trigger in triggers):
                    variants.extend(rewrites)

            if self._is_table_or_metric_query(self._query_terms(original_query)):
                variants.append(TABLE_QUERY_REWRITE_VARIANT)

        deduped_variants: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            variant = variant.strip()
            if not variant or variant in seen:
                continue
            seen.add(variant)
            deduped_variants.append(variant)
            if len(deduped_variants) >= self.query_rewrite_max_variants:
                break

        return deduped_variants or [query]

    def _term_coverage(self, query_terms: list[str], text: str) -> float:
        """Return fraction of query terms found in text."""
        if not query_terms or not text:
            return 0.0
        matched = sum(1 for term in query_terms if term in text)
        return matched / len(query_terms)

    def _expand_chinese_query_terms(self, term: str) -> list[str]:
        """Add Chinese company/title fragments for reranking."""
        if not re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9%]+", term):
            return []

        expanded: list[str] = []
        for hint in KEYWORD_HINT_TERMS | TABLE_QUERY_TERMS:
            if hint in term:
                expanded.append(hint)

        chinese_prefix = re.match(r"^[\u4e00-\u9fff]{2,6}", term)
        if chinese_prefix:
            expanded.append(chinese_prefix.group(0))

        if len(term) > 10:
            for size in (4, 3, 2):
                for start in range(0, len(term) - size + 1):
                    piece = term[start:start + size]
                    if re.fullmatch(r"[\u4e00-\u9fff]+", piece) and piece not in QUERY_STOPWORDS:
                        expanded.append(piece)

        return expanded[:40]

    def _company_from_file_name(self, file_name: str) -> str:
        """Extract the company name from report filenames like 【公司】标题.pdf."""
        match = re.search(r"【([^】]+)】", file_name)
        return match.group(1).lower() if match else ""

    def _title_term_hits(self, query_terms: list[str], file_name: str) -> int:
        """Count strong query terms that appear in the report title."""
        return sum(
            1 for term in query_terms
            if len(term) >= 3 and term in file_name
        )

    def _is_table_or_metric_query(self, query_terms: list[str]) -> bool:
        """Return whether query is likely asking for metrics or tabular data."""
        return any(term in TABLE_QUERY_TERMS for term in query_terms)

    def _is_explanation_query(self, query: str) -> bool:
        """Return whether a query asks for causes or stage evidence."""
        normalized = query.lower()
        return any(term in normalized for term in EXPLANATION_QUERY_TERMS)

    def _is_progress_query(self, query: str) -> bool:
        """Return whether a query asks for cooperation or delivery progress."""
        normalized = query.lower()
        return any(term in normalized for term in PROGRESS_QUERY_TERMS)

    def retrieve_for_file(
        self,
        query: str,
        file_id: str,
        top_k: int | None = None,
        topic: str | None = None,
    ) -> RetrievedContext:
        """Retrieve documents from a specific file.

        Args:
            query: The search query.
            file_id: The file ID to search within.
            top_k: Number of results to return.
            topic: Optional corpus topic stored in document metadata.

        Returns:
            RetrievedContext with results.
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            file_ids=[file_id],
            topic=topic,
        )

    def check_relevance(
        self,
        query: str,
        top_k: int = 3,
        min_relevant_score: float = 0.5,
        topic: str | None = None,
    ) -> tuple[bool, float]:
        """Check if relevant documents exist for a query.

        This is useful for the main agent to decide whether to use
        RAG retrieval or fallback to web search.

        Args:
            query: The search query.
            top_k: Number of results to check.
            min_relevant_score: Minimum score to consider relevant.
            topic: Optional corpus topic stored in document metadata.

        Returns:
            Tuple of (is_relevant, best_score).
        """
        context = self.retrieve(query=query, top_k=top_k, topic=topic)

        if not context.results:
            return False, 0.0

        best_score = max(r.score for r in context.results)
        is_relevant = best_score >= min_relevant_score

        logger.info(
            f"Relevance check for '{query[:30]}...': "
            f"{'relevant' if is_relevant else 'not relevant'} (score={best_score:.3f})"
        )

        return is_relevant, best_score

    def get_document_summary(self, file_id: str) -> dict[str, Any]:
        """Get summary information about a document.

        Args:
            file_id: The file ID to summarize.

        Returns:
            Dictionary with document summary info.
        """
        chunks = self._vector_store.get_file_chunks(file_id)

        if not chunks:
            return {
                "file_id": file_id,
                "found": False,
                "message": "文档未找到或尚未处理",
            }

        # Aggregate metadata
        total_chars = sum(len(c["content"]) for c in chunks)
        pages = set()

        for chunk in chunks:
            page_start, page_end = _page_range_from_metadata(chunk["metadata"])
            if page_start and page_end:
                pages.update(range(page_start, page_end + 1))
            elif page_start:
                pages.add(page_start)

        file_name = chunks[0]["metadata"].get("file_name", "Unknown") if chunks else "Unknown"

        return {
            "file_id": file_id,
            "found": True,
            "file_name": file_name,
            "chunk_count": len(chunks),
            "total_characters": total_chars,
            "pages": sorted(pages),
            "page_count": len(pages),
        }


# Singleton instance
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Get or create the singleton retriever instance.

    Returns:
        Retriever instance.
    """
    global _retriever

    if _retriever is None:
        settings = get_settings()
        _retriever = Retriever(
            default_top_k=settings.retrieval_top_k,
            score_threshold=settings.retrieval_score_threshold,
            candidate_multiplier=getattr(settings, "retrieval_candidate_multiplier", 4),
            hybrid_search_enabled=getattr(settings, "hybrid_search_enabled", True),
            query_rewrite_enabled=getattr(settings, "query_rewrite_enabled", True),
            query_rewrite_max_variants=getattr(settings, "query_rewrite_max_variants", 3),
        )

    return _retriever
