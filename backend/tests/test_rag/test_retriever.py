"""RAG 检索器的单元测试与高价值召回场景测试。

本文件从低到高覆盖四层能力：

1. ``RetrievalResult`` 与 ``RetrievedContext`` 的数据结构和上下文格式化；
2. 向量召回、关键词召回、查询改写、结果融合去重和重排；
3. file_id/主题过滤、分数阈值、页码与表格来源信息保留；
4. 指定公司、多公司比较以及“战略 + 合作进展”等多事实 Top-K 召回。

绝大多数测试用 ``MagicMock`` 代替真实向量库，直接控制候选结果和原始分数。
这样可以精确验证检索算法如何处理输入，而不会受到嵌入模型或 Chroma 状态影响。
测试通常遵循：准备候选集（Arrange）→ 执行 retrieve（Act）→ 检查排序、过滤、
metadata 和向量库调用参数（Assert）。
"""

from unittest.mock import MagicMock, patch


class TestRetrievalResult:
    """单条检索结果数据结构的序列化测试。"""

    def test_retrieval_result_to_dict(self):
        """to_dict 应完整保留正文、元数据和相关性分数。"""
        # 延迟导入可减少测试模块收集时的副作用，也让每个用例清楚展示被测对象。
        from app.rag.retriever import RetrievalResult

        result = RetrievalResult(
            content="test content",
            metadata={"file_id": "f1", "file_name": "test.pdf"},
            score=0.85,
        )

        result_dict = result.to_dict()

        assert result_dict["content"] == "test content"
        assert result_dict["metadata"]["file_id"] == "f1"
        assert result_dict["score"] == 0.85


class TestRetrievedContext:
    """多条检索结果组成的上下文及其提示词格式测试。"""

    def test_format_context_with_results(self):
        """普通结果应被格式化为带序号、文件名和页码的文档片段。"""
        from app.rag.retriever import RetrievalResult, RetrievedContext

        results = [
            RetrievalResult(
                content="Content 1",
                metadata={"file_name": "test.pdf", "page_number": 1},
                score=0.9,
            ),
            RetrievalResult(
                content="Content 2",
                metadata={"file_name": "test2.pdf"},
                score=0.8,
            ),
        ]

        context = RetrievedContext(
            query="test query",
            results=results,
            sources=[{"file_id": "f1", "file_name": "test.pdf"}],
        )

        # format_context 生成的是最终提供给大模型的“检索证据文本”。
        formatted = context.format_context()

        assert "--- 文档片段 1" in formatted
        assert "Content 1" in formatted
        assert "test.pdf" in formatted
        assert "第1页" in formatted

    def test_format_context_with_page_range_and_section(self):
        """跨页分块应显示页码范围，并保留所属章节标题。"""
        from app.rag.retriever import RetrievalResult, RetrievedContext

        results = [
            RetrievalResult(
                content="跨页内容",
                metadata={
                    "file_name": "test.pdf",
                    "page_start": 3,
                    "page_end": 4,
                    "section_title": "一、核心观点",
                },
                score=0.9,
            ),
        ]

        context = RetrievedContext(
            query="test query",
            results=results,
            sources=[],
        )

        formatted = context.format_context()

        assert "第3-4页" in formatted
        assert "章节: 一、核心观点" in formatted

    def test_format_context_with_table_result(self):
        """表格分块应明确标注为表格，避免大模型把它误当普通叙述。"""
        from app.rag.retriever import RetrievalResult, RetrievedContext

        results = [
            RetrievalResult(
                content="| 指标 | 2024 |\n| --- | --- |\n| 营收 | 120 |",
                metadata={
                    "file_name": "test.pdf",
                    "page_number": 5,
                    "content_type": "table",
                },
                score=0.92,
            ),
        ]

        context = RetrievedContext(
            query="营收",
            results=results,
            sources=[],
        )

        formatted = context.format_context()

        assert "--- 表格片段 1" in formatted
        assert "类型: 表格" in formatted
        assert "| 营收 | 120 |" in formatted

    def test_format_context_empty(self):
        """没有召回结果时应返回明确的“未找到”提示，而不是空字符串。"""
        from app.rag.retriever import RetrievedContext

        context = RetrievedContext(
            query="test query",
            results=[],
            sources=[],
        )

        formatted = context.format_context()

        assert "未找到相关信息" in formatted

    def test_format_context_without_metadata(self):
        """include_metadata=False 时只输出正文，不泄露分数等附加信息。"""
        from app.rag.retriever import RetrievalResult, RetrievedContext

        results = [
            RetrievalResult(
                content="Test content",
                metadata={"file_id": "f1"},
                score=0.9,
            ),
        ]

        context = RetrievedContext(
            query="test query",
            results=results,
            sources=[],
        )

        formatted = context.format_context(include_metadata=False)

        assert "--- 文档片段 1 ---" in formatted
        assert "Test content" in formatted
        assert "相似度" not in formatted

    def test_retrieved_context_to_dict(self):
        """上下文序列化后应保留查询、结果列表和来源列表。"""
        from app.rag.retriever import RetrievalResult, RetrievedContext

        results = [
            RetrievalResult(
                content="content",
                metadata={"file_id": "f1"},
                score=0.9,
            ),
        ]

        context = RetrievedContext(
            query="test query",
            results=results,
            sources=[{"file_id": "f1"}],
        )

        result_dict = context.to_dict()

        assert result_dict["query"] == "test query"
        assert len(result_dict["results"]) == 1
        assert len(result_dict["sources"]) == 1


class TestRetriever:
    """Retriever 召回、融合、重排、过滤和辅助方法测试。"""

    @patch("app.rag.retriever.get_vector_store")
    def test_retriever_init(self, mock_get_vs):
        """初始化参数应正确保存，混合检索和查询改写默认开启。"""
        # patch get_vector_store，避免构造 Retriever 时连接真实 Chroma。
        mock_vs = MagicMock()
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever(
            default_top_k=10,
            score_threshold=0.5,
            candidate_multiplier=3,
        )

        assert retriever.default_top_k == 10
        assert retriever.score_threshold == 0.5
        assert retriever.candidate_multiplier == 3
        assert retriever.hybrid_search_enabled is True
        assert retriever.query_rewrite_enabled is True
        assert retriever.query_rewrite_max_variants == 3

    @patch("app.rag.retriever.get_vector_store")
    def test_retrieve(self, mock_get_vs):
        """基础检索应把向量库记录转换为结果对象，并补充重排元数据。"""
        mock_vs = MagicMock()
        # 向量库接口返回 (记录字典, 相似度分数) 的有序列表。
        mock_vs.similarity_search_with_scores.return_value = [
            ({"content": "result 1", "metadata": {"file_id": "f1", "file_name": "test.pdf"}}, 0.9),
            ({"content": "result 2", "metadata": {"file_id": "f2", "file_name": "test2.pdf"}}, 0.8),
        ]
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever()
        context = retriever.retrieve("test query", top_k=5)

        assert context.query == "test query"
        assert len(context.results) == 2
        assert context.results[0].content == "result 1"
        assert context.results[0].score == 0.9
        # retrieval_score 保存原始召回分，rerank_score 保存经过规则重排后的最终分。
        assert context.results[0].metadata["retrieval_score"] == 0.9
        assert "rerank_score" in context.results[0].metadata

    @patch("app.rag.retriever.get_vector_store")
    def test_retrieve_recalls_more_candidates_before_rerank(self, mock_get_vs):
        """重排前应召回 top_k 的多倍候选，给重排算法足够选择空间。"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_scores.return_value = []
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        # candidate_multiplier=4 且 top_k=5，所以向量库应先返回 20 个候选。
        retriever = Retriever(candidate_multiplier=4)
        retriever.retrieve("test query", top_k=5)

        call_args = mock_vs.similarity_search_with_scores.call_args
        assert call_args.kwargs["k"] == 20

    @patch("app.rag.retriever.get_vector_store")
    def test_rerank_boosts_metric_table_results(self, mock_get_vs):
        """指标类问题应提升表格分块，使其超过原始分更高的泛化正文。"""
        mock_vs = MagicMock()
        # 正文原始分 0.83 高于表格 0.80，用来证明最终逆转确实来自重排规则。
        mock_vs.similarity_search_with_scores.return_value = [
            (
                {
                    "content": "这是一段泛泛的业务描述。",
                    "metadata": {"file_id": "f1", "file_name": "test.pdf", "content_type": "text"},
                },
                0.83,
            ),
            (
                {
                    "content": "| 指标 | 2024 |\n| --- | --- |\n| 营收 | 120 |",
                    "metadata": {
                        "file_id": "f1",
                        "file_name": "test.pdf",
                        "page_number": 5,
                        "content_type": "table",
                    },
                },
                0.8,
            ),
        ]
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever(candidate_multiplier=4)
        context = retriever.retrieve("营收是多少", top_k=1)

        assert len(context.results) == 1
        assert context.results[0].metadata["content_type"] == "table"
        # rerank_reasons 不只验证排序结果，还说明“为什么”发生了加分。
        assert "table" in context.results[0].metadata["rerank_reasons"]

    def test_explanation_query_uses_narrative_rewrites(self):
        """原因/阶段问题必须使用叙述性改写，不能退化为纯财务指标查询。"""
        from app.rag.retriever import Retriever, TABLE_QUERY_REWRITE_VARIANT

        retriever = Retriever(vector_store=MagicMock(), query_rewrite_max_variants=3)
        query = "公司利润为什么承压，项目处于什么阶段？"

        # 这里直接测试私有方法，是为了隔离验证查询意图分类与改写规则。
        variants = retriever._query_variants(query)

        assert variants == [
            query,
            "经营分析 利润承压 原因 因素",
            "项目阶段 客户导入 验证 小批量 量产进展",
        ]
        assert TABLE_QUERY_REWRITE_VARIANT not in variants

    def test_ra_rag_004_progress_query_uses_cooperation_rewrites(self):
        """RA-RAG-004：合作进展问题应使用进展改写，不能套用原因分析模板。"""
        from app.rag.retriever import (
            EXPLANATION_QUERY_REWRITES,
            PROGRESS_QUERY_REWRITE,
            ROBOT_PROGRESS_QUERY_REWRITE,
            Retriever,
        )

        retriever = Retriever(vector_store=MagicMock(), query_rewrite_max_variants=3)
        query = "奥普特如何从机器视觉延伸到具身智能？目前有哪些合作或送样进展？"

        variants = retriever._query_variants(query)

        assert variants == [
            query,
            PROGRESS_QUERY_REWRITE,
            ROBOT_PROGRESS_QUERY_REWRITE,
        ]
        assert EXPLANATION_QUERY_REWRITES[0] not in variants

    def test_ra_rag_004_keeps_strategy_and_progress_facts_in_top_five(self):
        """Top-5 必须同时召回技术路线和合作/送样进展，不能只命中其中一类事实。"""
        mock_vs = MagicMock()
        # strategy 原始分最高；progress 原始分仅 0.70，前面还有多个泛化干扰片段。
        # 若没有进展意图重排，progress 很容易被截断在 Top-5 之外。
        candidates = [
            (
                {
                    "id": "strategy",
                    "content": "奥普特提供全系列3D视觉传感器，并拥有自研工业AI算法。",
                    "metadata": {
                        "file_id": "opt-report",
                        "file_name": "【奥普特】视觉龙头迈向具身智能.pdf",
                        "page_number": 25,
                    },
                },
                0.95,
            )
        ]
        candidates.extend([
            (
                {
                    "id": f"generic-{index}",
                    "content": "奥普特机器视觉产品覆盖多个工业场景，具身智能空间广阔。",
                    "metadata": {
                        "file_id": "opt-report",
                        "file_name": "【奥普特】视觉龙头迈向具身智能.pdf",
                        "page_number": page,
                    },
                },
                score,
            )
            for index, (page, score) in enumerate(
                ((20, 0.93), (4, 0.91), (23, 0.89), (8, 0.87), (13, 0.85)),
                start=1,
            )
        ])
        candidates.append((
            {
                "id": "progress",
                "content": (
                    "公司收购东莞泰莱补充精密传动能力，与越疆开展手眼脑一体化合作，"
                    "机器人关节模组已进入送样阶段。"
                ),
                "metadata": {
                    "file_id": "opt-report",
                    "file_name": "【奥普特】视觉龙头迈向具身智能.pdf",
                    "page_number": 1,
                },
            },
            0.70,
        ))
        mock_vs.similarity_search_with_scores.return_value = candidates

        from app.rag.retriever import Retriever

        retriever = Retriever(
            vector_store=mock_vs,
            hybrid_search_enabled=False,
            query_rewrite_enabled=True,
        )
        context = retriever.retrieve(
            "奥普特如何从机器视觉延伸到具身智能？目前有哪些合作或送样进展？",
            top_k=5,
        )

        # 多事实问题不能只断言某个文档命中，而要检查所有答案锚点都在 Top-5 证据中。
        top_five_text = "\n".join(result.content for result in context.results)
        top_five_pages = {result.metadata["page_number"] for result in context.results}
        expected_terms = (
            "3D视觉传感器",
            "工业AI",
            "东莞泰莱",
            "越疆",
            "手眼脑一体化",
            "关节模组",
            "送样",
        )
        progress_result = next(
            result for result in context.results
            if result.metadata["page_number"] == 1
        )

        assert {1, 25}.issubset(top_five_pages)
        assert all(term in top_five_text for term in expected_terms)
        assert "progress" in progress_result.metadata["rerank_reasons"]

    def test_explanation_query_promotes_narrative_and_penalizes_table(self):
        """原因类叙述证据应超过原始分更高的财务表格。"""
        mock_vs = MagicMock()
        # 表格分 0.90、原因正文分 0.78，构造一个需要靠意图重排才能纠正的逆序。
        mock_vs.similarity_search_with_scores.return_value = [
            (
                {
                    "id": "finance-table",
                    "content": "营业收入 归母净利润 同比 环比",
                    "metadata": {
                        "file_id": "target",
                        "file_name": "【测试公司】一季报点评.pdf",
                        "content_type": "table",
                    },
                },
                0.90,
            ),
            (
                {
                    "id": "cause-text",
                    "content": (
                        "经营分析：利润端受多重因素阶段性承压。其一是汇兑损失；"
                        "其二是项目尚处客户导入与 POC 阶段，小批量出货尚未起量。"
                    ),
                    "metadata": {
                        "file_id": "target",
                        "file_name": "【测试公司】一季报点评.pdf",
                        "content_type": "text",
                    },
                },
                0.78,
            ),
        ]

        from app.rag.retriever import Retriever

        retriever = Retriever(
            vector_store=mock_vs,
            hybrid_search_enabled=False,
            query_rewrite_enabled=True,
        )
        context = retriever.retrieve(
            "测试公司利润为什么承压，项目处于什么阶段？",
            top_k=1,
        )

        assert context.results[0].content.startswith("经营分析")
        assert "explanation" in context.results[0].metadata["rerank_reasons"]

    @patch("app.rag.retriever.get_vector_store")
    def test_query_rewrite_expands_profit_query(self, mock_get_vs):
        """口语化盈利问题应改写为研报常用的毛利率、净利润、EPS 等术语。"""
        mock_vs = MagicMock()
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever(query_rewrite_max_variants=3)
        variants = retriever._query_variants("这家公司盈利怎么样")

        assert variants[0] == "这家公司盈利怎么样"
        assert "盈利能力 毛利率 净利率 净利润" in variants
        assert "盈利预测 EPS 归母净利润" in variants

    @patch("app.rag.retriever.get_vector_store")
    def test_query_rewrite_runs_multiple_retrieval_queries(self, mock_get_vs):
        """原始问题和每个改写变体都必须真正参与向量召回。"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_scores.return_value = []
        mock_vs.keyword_search_with_scores.return_value = []
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever(query_rewrite_max_variants=3)
        retriever.retrieve("盈利怎么样", top_k=2)

        # 从 mock 的调用历史提取 query 参数，验证三个变体都执行过检索且顺序正确。
        called_queries = [
            call.kwargs["query"]
            for call in mock_vs.similarity_search_with_scores.call_args_list
        ]
        assert called_queries == [
            "盈利怎么样",
            "盈利能力 毛利率 净利率 净利润",
            "盈利预测 EPS 归母净利润",
        ]

    @patch("app.rag.retriever.get_vector_store")
    def test_query_rewrite_keeps_variant_metadata_on_hits(self, mock_get_vs):
        """合并后的命中应记录由哪些查询变体召回，便于诊断召回来源。"""
        mock_vs = MagicMock()
        # 三次变体检索都返回同一 id，最终应去重为一条并累计命中来源。
        result = {
            "id": "same1",
            "content": "公司净利润和毛利率均有改善。",
            "metadata": {"file_id": "f1", "file_name": "test.pdf"},
        }
        mock_vs.similarity_search_with_scores.return_value = [(result, 0.8)]
        mock_vs.keyword_search_with_scores.return_value = []
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever(query_rewrite_max_variants=3)
        context = retriever.retrieve("盈利怎么样", top_k=1)

        metadata = context.results[0].metadata
        assert metadata["matched_query_count"] == 3
        assert "盈利怎么样" in metadata["query_variants"]
        assert "盈利能力 毛利率 净利率 净利润" in metadata["query_variants"]

    @patch("app.rag.retriever.get_vector_store")
    def test_hybrid_search_includes_keyword_only_results(self, mock_get_vs):
        """混合检索必须保留仅关键词通道命中的精确数字/指标证据。"""
        mock_vs = MagicMock()
        # 向量通道召回语义近似文本；关键词通道命中包含 2025E EPS 的精确事实。
        mock_vs.similarity_search_with_scores.return_value = [
            (
                {
                    "id": "vector1",
                    "content": "语义相近但没有精确指标。",
                    "metadata": {"file_id": "f1", "file_name": "test.pdf"},
                },
                0.55,
            ),
        ]
        mock_vs.keyword_search_with_scores.return_value = [
            (
                {
                    "id": "keyword1",
                    "content": "2025E EPS 为 2.10 元。",
                    "metadata": {"file_id": "f1", "file_name": "test.pdf", "page_number": 8},
                },
                0.95,
            ),
        ]
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever(candidate_multiplier=4)
        context = retriever.retrieve("2025E EPS", top_k=1)

        assert context.results[0].content == "2025E EPS 为 2.10 元。"
        assert context.results[0].metadata["search_type"] == "keyword"
        assert context.results[0].metadata["keyword_score"] == 0.95

    @patch("app.rag.retriever.get_vector_store")
    def test_hybrid_search_deduplicates_vector_and_keyword_results(self, mock_get_vs):
        """同一分块同时被向量和关键词命中时应融合为一条结果，而非重复返回。"""
        mock_vs = MagicMock()
        # 两个通道返回相同 id，融合后应同时保留 vector_score 与 keyword_score。
        result = {
            "id": "same1",
            "content": "营收增长 20%。",
            "metadata": {"file_id": "f1", "file_name": "test.pdf", "page_number": 3},
        }
        mock_vs.similarity_search_with_scores.return_value = [(result, 0.82)]
        mock_vs.keyword_search_with_scores.return_value = [(result, 0.9)]
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever()
        context = retriever.retrieve("营收增长", top_k=5)

        assert len(context.results) == 1
        assert context.results[0].metadata["search_type"] == "keyword+vector"
        assert context.results[0].metadata["vector_score"] == 0.82
        assert context.results[0].metadata["keyword_score"] == 0.9

    @patch("app.rag.retriever.get_vector_store")
    def test_rerank_boosts_company_and_filename_matches(self, mock_get_vs):
        """查询中的公司名和标题词命中文件名时，应提升对应报告的排序。"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_scores.return_value = [
            (
                {
                    "id": "generic",
                    "content": "低空经济行业空间广阔。",
                    "metadata": {"file_id": "f2", "file_name": "【其他公司】低空经济行业报告.pdf"},
                },
                0.78,
            ),
            (
                {
                    "id": "company",
                    "content": "连接器业务持续拓展，低空经济打开新增量。",
                    "metadata": {"file_id": "f1", "file_name": "【创益通】连接器小巨人，布局低空经济.pdf"},
                },
                0.72,
            ),
        ]
        mock_vs.keyword_search_with_scores.return_value = []
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever(query_rewrite_enabled=False)
        context = retriever.retrieve("创益通连接器业务低空经济打开想象空间", top_k=1)

        assert context.results[0].metadata["file_id"] == "f1"
        assert "company" in context.results[0].metadata["rerank_reasons"]

    def test_ra_rag_003_keeps_named_company_financial_evidence_in_top_five(self):
        """RA-RAG-003：竞品高分片段不能把指定公司的财务事实挤出 Top-5。"""
        mock_vs = MagicMock()
        # 五个杭叉片段原始分全部高于凌云光目标片段，模拟同语料库竞品“淹没”问题。
        candidates = [
            (
                {
                    "id": f"hangcha-{index}",
                    "content": "营业收入与归母净利润同比增长，经营表现稳健。",
                    "metadata": {
                        "file_id": "hangcha-report",
                        "file_name": "【杭叉集团】叉车行业深度报告.pdf",
                        "page_number": index,
                    },
                },
                score,
            )
            for index, score in enumerate((1.00, 0.98, 0.96, 0.94, 0.92), start=1)
        ]
        candidates.extend([
            (
                {
                    "id": "lingyunguang-target",
                    "content": (
                        "凌云光2025年营收29.12亿元，同比增长30.35%；"
                        "归母净利润1.61亿元，同比增长50.70%。"
                    ),
                    "metadata": {
                        "file_id": "lingyunguang-report",
                        "file_name": "【凌云光】机器视觉与具身智能业务点评.pdf",
                        "page_number": 1,
                    },
                },
                0.69,
            ),
            (
                {
                    "id": "other-1",
                    "content": "机器人行业景气度持续提升。",
                    "metadata": {
                        "file_id": "other-report-1",
                        "file_name": "【开特股份】机器人业务报告.pdf",
                    },
                },
                0.75,
            ),
            (
                {
                    "id": "other-2",
                    "content": "具身智能产业链进入加速期。",
                    "metadata": {
                        "file_id": "other-report-2",
                        "file_name": "【绿的谐波】机器人行业报告.pdf",
                    },
                },
                0.74,
            ),
        ])
        mock_vs.similarity_search_with_scores.return_value = candidates

        from app.rag.retriever import Retriever

        retriever = Retriever(
            vector_store=mock_vs,
            candidate_multiplier=2,
            hybrid_search_enabled=False,
            query_rewrite_enabled=False,
        )
        context = retriever.retrieve(
            "凌云光2025年营收和归母净利润是多少，分别同比增长多少？",
            top_k=5,
        )

        top_five_text = "\n".join(result.content for result in context.results)
        top_five_file_ids = [result.metadata["file_id"] for result in context.results]
        # 精确公司匹配应把目标证据提升到首位，并限制单一竞品占满结果列表。
        assert top_five_file_ids[0] == "lingyunguang-report"
        assert top_five_file_ids.count("hangcha-report") <= 2
        assert all(term in top_five_text for term in ("29.12", "30.35%", "1.61亿元", "50.70%"))
        assert "company" in context.results[0].metadata["rerank_reasons"]

    def test_ra_rag_003_supports_queries_naming_two_companies(self):
        """比较两家公司时，两者都应获得精确公司匹配的重排加分。"""
        mock_vs = MagicMock()
        # 无关公司的原始分最高；两个被点名公司的原始分较低但都应进入 Top-2。
        mock_vs.similarity_search_with_scores.return_value = [
            (
                {
                    "id": "other",
                    "content": "行业公司收入持续增长。",
                    "metadata": {
                        "file_id": "other-report",
                        "file_name": "【杭叉集团】行业报告.pdf",
                    },
                },
                0.95,
            ),
            (
                {
                    "id": "lingyunguang",
                    "content": "凌云光机器视觉业务收入情况。",
                    "metadata": {
                        "file_id": "lingyunguang-report",
                        "file_name": "【凌云光】公司报告.pdf",
                    },
                },
                0.62,
            ),
            (
                {
                    "id": "opt",
                    "content": "奥普特机器视觉业务收入情况。",
                    "metadata": {
                        "file_id": "opt-report",
                        "file_name": "【奥普特】公司报告.pdf",
                    },
                },
                0.61,
            ),
        ]

        from app.rag.retriever import Retriever

        retriever = Retriever(
            vector_store=mock_vs,
            hybrid_search_enabled=False,
            query_rewrite_enabled=False,
        )
        context = retriever.retrieve("比较凌云光与奥普特的机器视觉业务收入", top_k=2)

        assert {result.metadata["file_id"] for result in context.results} == {
            "lingyunguang-report",
            "opt-report",
        }
        assert all("company" in result.metadata["rerank_reasons"] for result in context.results)

    def test_ra_rag_003_does_not_diversify_generic_queries(self):
        """未指定公司的通用问题不应误触发公司多样化，仍按普通分数排序。"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_scores.return_value = [
            (
                {
                    "id": f"generic-{index}",
                    "content": "具身智能行业发展趋势。",
                    "metadata": {
                        "file_id": "same-report",
                        "file_name": "【测试公司】专题报告.pdf",
                    },
                },
                score,
            )
            for index, score in enumerate((0.90, 0.89, 0.88), start=1)
        ]

        from app.rag.retriever import Retriever

        retriever = Retriever(
            vector_store=mock_vs,
            hybrid_search_enabled=False,
            query_rewrite_enabled=False,
        )
        context = retriever.retrieve("具身智能行业发展趋势", top_k=3)

        assert [result.score for result in context.results] == [0.90, 0.89, 0.88]
        assert all(
            "other_company" not in result.metadata.get("rerank_reasons", "")
            for result in context.results
        )

    @patch("app.rag.retriever.get_vector_store")
    def test_retrieve_with_file_filter(self, mock_get_vs):
        """指定单个 file_id 时，应把过滤条件原样传给向量召回。"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_scores.return_value = [
            ({"content": "result 1", "metadata": {"file_id": "target_file", "file_name": "test.pdf"}}, 0.9),
        ]
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever()
        retriever.retrieve("test query", file_ids=["target_file"])

        mock_vs.similarity_search_with_scores.assert_called_once()
        call_args = mock_vs.similarity_search_with_scores.call_args
        assert call_args.kwargs["filter_"] == {"file_id": "target_file"}

    def test_topic_filter_reaches_vector_and_keyword_recall(self):
        """同一个主题过滤条件必须同时约束向量和关键词两个召回通道。"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_scores.return_value = []
        mock_vs.keyword_search_with_scores.return_value = []

        from app.rag.retriever import Retriever

        retriever = Retriever(vector_store=mock_vs, query_rewrite_enabled=False)
        retriever.retrieve("凌云光机器人业务", topic="embodied_intelligence")

        # 若只过滤一个通道，另一个通道仍可能把跨主题污染结果合并进来。
        expected_filter = {"topic": "embodied_intelligence"}
        assert mock_vs.similarity_search_with_scores.call_args.kwargs["filter_"] == expected_filter
        assert mock_vs.keyword_search_with_scores.call_args.kwargs["filter_"] == expected_filter

    def test_topic_and_single_file_filters_are_combined(self):
        """同时指定报告与主题时，应使用 $and 组合两个过滤条件。"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_scores.return_value = []

        from app.rag.retriever import Retriever

        retriever = Retriever(
            vector_store=mock_vs,
            hybrid_search_enabled=False,
            query_rewrite_enabled=False,
        )
        retriever.retrieve(
            "财务表现",
            file_ids=["report-001"],
            topic="embodied_intelligence",
        )

        assert mock_vs.similarity_search_with_scores.call_args.kwargs["filter_"] == {
            "$and": [
                {"file_id": "report-001"},
                {"topic": "embodied_intelligence"},
            ]
        }

    def test_ra_rag_001_topic_filter_excludes_cross_topic_competitor(self):
        """RA-RAG-001：具身智能检索必须排除低空经济主题的高分竞品片段。"""
        mock_vs = MagicMock()
        # 低空经济候选分数更高，只有把 topic 下推到向量库召回阶段才能彻底排除。
        all_candidates = [
            (
                {
                    "id": "low-altitude-competitor",
                    "content": "宗申动力营业收入和归母净利润",
                    "metadata": {
                        "file_id": "low-altitude-report",
                        "file_name": "宗申动力低空经济研报.pdf",
                        "topic": "low_altitude",
                    },
                },
                0.98,
            ),
            (
                {
                    "id": "embodied-target",
                    "content": "凌云光机器视觉与机器人业务财务表现",
                    "metadata": {
                        "file_id": "embodied-report",
                        "file_name": "凌云光具身智能研报.pdf",
                        "topic": "embodied_intelligence",
                    },
                },
                0.86,
            ),
        ]

        def similarity_search(*, query, k, filter_):
            """模拟向量库根据 filter_ 在召回阶段筛选语料。"""
            del query, k
            if filter_ == {"topic": "embodied_intelligence"}:
                return [all_candidates[1]]
            return all_candidates

        mock_vs.similarity_search_with_scores.side_effect = similarity_search

        from app.rag.retriever import Retriever

        retriever = Retriever(
            vector_store=mock_vs,
            candidate_multiplier=1,
            hybrid_search_enabled=False,
            query_rewrite_enabled=False,
        )
        # baseline 不加主题限制，用于证明干扰候选确实会出现；filtered 才是安全结果。
        baseline = retriever.retrieve("凌云光财务表现", top_k=2)
        filtered = retriever.retrieve(
            "凌云光财务表现",
            top_k=2,
            topic="embodied_intelligence",
        )

        assert any(result.metadata["topic"] == "low_altitude" for result in baseline.results)
        assert [result.metadata["file_id"] for result in filtered.results] == ["embodied-report"]
        assert {result.metadata["topic"] for result in filtered.results} == {"embodied_intelligence"}

    @patch("app.rag.retriever.get_vector_store")
    def test_retrieve_with_score_threshold(self, mock_get_vs):
        """低于相关性阈值的结果应在最终上下文中被过滤。"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_scores.return_value = [
            ({"content": "high score", "metadata": {"file_id": "f1"}}, 0.9),
            ({"content": "low score", "metadata": {"file_id": "f2"}}, 0.2),
        ]
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever()
        context = retriever.retrieve("test query", score_threshold=0.5)

        # 阈值在最终结果阶段生效，低分记录不应进入 RetrievedContext。
        assert len(context.results) == 1
        assert context.results[0].content == "high score"

    @patch("app.rag.retriever.get_vector_store")
    def test_retrieve_keeps_page_level_sources(self, mock_get_vs):
        """同一文件的不同命中页应保留为不同来源，不能只按 file_id 去重。"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_scores.return_value = [
            (
                {
                    "content": "page 1 result",
                    "metadata": {"file_id": "f1", "file_name": "test.pdf", "page_number": 1},
                },
                0.9,
            ),
            (
                {
                    "content": "page 2 result",
                    "metadata": {"file_id": "f1", "file_name": "test.pdf", "page_number": 2},
                },
                0.8,
            ),
        ]
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever()
        context = retriever.retrieve("test query")

        # 来源引用的最小粒度是“文件 + 页面”，这对答案引用和证据页评估很重要。
        assert len(context.sources) == 2
        assert {source["page_number"] for source in context.sources} == {1, 2}

    @patch("app.rag.retriever.get_vector_store")
    def test_retrieve_keeps_page_range_sources(self, mock_get_vs):
        """跨页分块的来源应保留起止页码和章节标题。"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_scores.return_value = [
            (
                {
                    "content": "page range result",
                    "metadata": {
                        "file_id": "f1",
                        "file_name": "test.pdf",
                        "page_number": 3,
                        "page_start": 3,
                        "page_end": 4,
                        "section_title": "一、核心观点",
                    },
                },
                0.9,
            ),
        ]
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever()
        context = retriever.retrieve("test query")

        assert len(context.sources) == 1
        assert context.sources[0]["page_number"] == 3
        assert context.sources[0]["page_start"] == 3
        assert context.sources[0]["page_end"] == 4
        assert context.sources[0]["section_title"] == "一、核心观点"

    @patch("app.rag.retriever.get_vector_store")
    def test_retrieve_keeps_table_source_type(self, mock_get_vs):
        """表格命中的来源元数据必须保留 content_type=table。"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_scores.return_value = [
            (
                {
                    "content": "table result",
                    "metadata": {
                        "file_id": "f1",
                        "file_name": "test.pdf",
                        "page_number": 5,
                        "content_type": "table",
                    },
                },
                0.9,
            ),
        ]
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever()
        context = retriever.retrieve("营收")

        assert context.sources[0]["content_type"] == "table"

    @patch("app.rag.retriever.get_vector_store")
    def test_retrieve_for_file(self, mock_get_vs):
        """retrieve_for_file 应作为限定单文件检索的便捷入口。"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_scores.return_value = [
            ({"content": "result", "metadata": {"file_id": "file123"}}, 0.9),
        ]
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever()
        context = retriever.retrieve_for_file("test query", file_id="file123")

        assert len(context.results) == 1

    @patch("app.rag.retriever.get_vector_store")
    def test_check_relevance_relevant(self, mock_get_vs):
        """最佳分数达到阈值时，相关性判断应返回 True 和该分数。"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_scores.return_value = [
            ({"content": "result", "metadata": {"file_id": "f1"}}, 0.8),
        ]
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever()
        # check_relevance 是检索结果之上的便捷判断，供上层决定是否采用 RAG 证据。
        is_relevant, best_score = retriever.check_relevance("test query", min_relevant_score=0.5)

        assert is_relevant is True
        assert best_score == 0.8

    @patch("app.rag.retriever.get_vector_store")
    def test_check_relevance_not_relevant(self, mock_get_vs):
        """最佳分数低于阈值时，相关性判断应返回 False 但保留实际分数。"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_scores.return_value = [
            ({"content": "result", "metadata": {"file_id": "f1"}}, 0.3),
        ]
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever()
        is_relevant, best_score = retriever.check_relevance("test query", min_relevant_score=0.5)

        assert is_relevant is False
        assert best_score == 0.3

    @patch("app.rag.retriever.get_vector_store")
    def test_check_relevance_empty(self, mock_get_vs):
        """完全没有召回结果时，应返回不相关和 0 分。"""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_scores.return_value = []
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever()
        is_relevant, best_score = retriever.check_relevance("test query")

        assert is_relevant is False
        assert best_score == 0.0

    @patch("app.rag.retriever.get_vector_store")
    def test_get_document_summary(self, mock_get_vs):
        """文档摘要应统计文件名、分块数量和去重后的页码。"""
        mock_vs = MagicMock()
        mock_vs.get_file_chunks.return_value = [
            {"id": "c1", "content": "Content 1", "metadata": {"file_name": "test.pdf", "page_number": 1}},
            {"id": "c2", "content": "Content 2", "metadata": {"file_name": "test.pdf", "page_number": 2}},
        ]
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever()
        summary = retriever.get_document_summary("file123")

        assert summary["found"] is True
        assert summary["file_name"] == "test.pdf"
        assert summary["chunk_count"] == 2
        assert summary["page_count"] == 2
        assert 1 in summary["pages"]
        assert 2 in summary["pages"]

    @patch("app.rag.retriever.get_vector_store")
    def test_get_document_summary_counts_page_ranges(self, mock_get_vs):
        """跨页分块应展开为连续页码后再计算文档页数。"""
        mock_vs = MagicMock()
        mock_vs.get_file_chunks.return_value = [
            {
                "id": "c1",
                "content": "Content 1",
                "metadata": {"file_name": "test.pdf", "page_start": 2, "page_end": 4},
            },
        ]
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever()
        summary = retriever.get_document_summary("file123")

        assert summary["page_count"] == 3
        assert summary["pages"] == [2, 3, 4]

    @patch("app.rag.retriever.get_vector_store")
    def test_get_document_summary_not_found(self, mock_get_vs):
        """没有任何分块时，文档摘要应返回 found=False 和友好提示。"""
        mock_vs = MagicMock()
        mock_vs.get_file_chunks.return_value = []
        mock_get_vs.return_value = mock_vs

        from app.rag.retriever import Retriever

        retriever = Retriever()
        summary = retriever.get_document_summary("nonexistent")

        assert summary["found"] is False
        assert "未找到" in summary["message"]


class TestGetRetriever:
    """全局 Retriever 工厂及单例配置测试。"""

    @patch("app.rag.retriever.get_settings")
    @patch("app.rag.retriever.get_vector_store")
    def test_get_retriever_singleton(self, mock_get_vs, mock_get_settings):
        """工厂应只创建一个 Retriever，并从应用设置读取全部检索参数。"""
        mock_vs = MagicMock()
        mock_get_vs.return_value = mock_vs
        mock_settings = MagicMock()
        mock_settings.retrieval_top_k = 7
        mock_settings.retrieval_score_threshold = 0.25
        mock_settings.retrieval_candidate_multiplier = 3
        mock_settings.hybrid_search_enabled = False
        mock_settings.query_rewrite_enabled = False
        mock_settings.query_rewrite_max_variants = 2
        mock_get_settings.return_value = mock_settings

        # 单例属于模块级全局状态；测试前必须重置，否则会继承其他用例创建的实例。
        import app.rag.retriever as retriever_module
        from app.rag.retriever import get_retriever

        retriever_module._retriever = None

        r1 = get_retriever()
        r2 = get_retriever()

        # 连续两次调用必须返回同一对象，并且首次构造时读取了测试配置。
        assert r1 is r2
        assert r1.default_top_k == 7
        assert r1.score_threshold == 0.25
        assert r1.candidate_multiplier == 3
        assert r1.hybrid_search_enabled is False
        assert r1.query_rewrite_enabled is False
        assert r1.query_rewrite_max_variants == 2
