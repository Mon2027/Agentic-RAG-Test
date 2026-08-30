"""RAG 检索质量评估工具的测试。

评估数据由“查询 + 标准答案（ground truth）”组成，检索结果按排名与标准答案比较。
本项目使用的主要指标包括：

* Recall@K：前 K 个结果找回了多少比例的相关文档，强调“有没有漏召回”；
* Precision@K：前 K 个结果中相关文档所占比例，强调“噪声有多少”；
* MRR：第一个相关文档排名的倒数，例如第 2 名首次命中则为 1/2；
* NDCG：对相关结果所在排名进行对数折损，再与理想排序比较；
* term coverage：期望关键词/事实锚点在证据文本中的覆盖比例；
* evidence page recall：是否命中人工标注的准确证据页；
* evidence group recall：允许同一事实存在多个等价页面，但页面与锚点词必须同时满足。

测试使用确定性的 Retriever stub 固定返回顺序，避免嵌入模型变化导致指标波动。
"""

from app.evaluation.rag_evaluator import (
    EvaluationSample,
    RAGEvaluator,
    create_evaluation_dataset_from_reports,
)
from app.rag.retriever import RetrievalResult, RetrievedContext


class FakeRetriever:
    """返回固定排序结果的检索器 stub，供基础指标测试使用。"""

    def retrieve(self, query: str, top_k: int = 20):
        # doc1 连续出现两次，用于验证评估时必须按文档去重后再计算排名指标。
        return RetrievedContext(
            query=query,
            results=[
                RetrievalResult(
                    content="中信海直 通航 主业稳健增长",
                    metadata={
                        "file_id": "doc1",
                        "file_name": "中信海直.pdf",
                        "page_number": 2,
                        "search_type": "keyword+vector",
                        "rerank_score": 0.92,
                    },
                    score=0.9,
                ),
                RetrievalResult(
                    content="重复命中文档的另一个片段",
                    metadata={"file_id": "doc1", "file_name": "中信海直.pdf", "page_number": 3},
                    score=0.85,
                ),
                RetrievalResult(
                    content="其他公司低空经济布局",
                    metadata={"file_id": "doc2", "file_name": "其他.pdf", "page_number": 1},
                    score=0.8,
                ),
            ],
            sources=[],
        )


class TopicRecordingRetriever(FakeRetriever):
    """记录调用参数的 spy，用于验证评估器是否传递生产主题过滤条件。"""

    def __init__(self):
        # 保存全部调用，既能检查参数，也能检查调用次数与顺序。
        self.calls = []

    def retrieve(self, query: str, top_k: int = 20, topic: str | None = None):
        self.calls.append({"query": query, "top_k": top_k, "topic": topic})
        return super().retrieve(query, top_k)


class EvidencePageRetriever:
    """目标报告先命中，但真正答案页到第 3 个分块才出现的测试替身。"""

    def retrieve(self, query: str, top_k: int = 20):
        return RetrievedContext(
            query=query,
            results=[
                RetrievalResult(
                    content="目标报告的财务表格",
                    metadata={"file_id": "doc1", "page_number": 2, "content_type": "table"},
                    score=0.9,
                ),
                RetrievalResult(
                    content="其他报告正文",
                    metadata={"file_id": "doc2", "page_number": 1},
                    score=0.8,
                ),
                RetrievalResult(
                    content="目标报告的原因说明",
                    metadata={"file_id": "doc1", "page_number": 1},
                    score=0.7,
                ),
            ][:top_k],
            sources=[],
        )


class CoverageDepthRetriever:
    """第二个必要事实在第 6 个分块才出现，用于暴露 Top-5 覆盖缺口。"""

    def retrieve(self, query: str, top_k: int = 20):
        # 第 1 名包含“第一事实”，第 2～5 名是噪声，第 6 名才包含“第二事实”。
        results = [
            RetrievalResult(
                content="第一事实",
                metadata={"file_id": "doc1", "page_number": 1},
                score=0.9,
            ),
            *[
                RetrievalResult(
                    content=f"无关片段{index}",
                    metadata={"file_id": f"other{index}", "page_number": 1},
                    score=0.9 - index / 100,
                )
                for index in range(2, 6)
            ],
            RetrievalResult(
                content="第二事实",
                metadata={"file_id": "doc1", "page_number": 2},
                score=0.8,
            ),
        ]
        return RetrievedContext(query=query, results=results[:top_k], sources=[])


class AlternativeEvidenceRetriever:
    """在等价证据页命中完整事实组的检索器替身。"""

    def retrieve(self, query: str, top_k: int = 20):
        return RetrievedContext(
            query=query,
            results=[
                RetrievalResult(
                    content="新增谐波减速器和执行器模组产能",
                    metadata={"file_id": "doc1", "page_number": 4},
                    score=0.9,
                ),
            ][:top_k],
            sources=[],
        )


class SamePageEvidenceRetriever:
    """正确页面先出现，但真正含事实锚点的分块排在后面的测试替身。"""

    def retrieve(self, query: str, top_k: int = 20):
        return RetrievedContext(
            query=query,
            results=[
                RetrievalResult(
                    content="目标报告的财务表格",
                    metadata={"file_id": "doc1", "page_number": 1},
                    score=0.9,
                ),
                RetrievalResult(
                    content="利润承压源于汇兑损失，项目处于POC阶段",
                    metadata={"file_id": "doc1", "page_number": 1},
                    score=0.8,
                ),
            ][:top_k],
            sources=[],
        )


def test_evaluator_uses_unique_document_ranking():
    """同一文档的多个分块只能算一次文档命中，不能挤占文档排名。"""
    # K 取 1、2，便于观察 doc2 在去重文档榜中的排名从未命中变为命中。
    evaluator = RAGEvaluator(k_values=[1, 2], retriever=FakeRetriever())
    sample = EvaluationSample(
        query="中信海直通航主业",
        relevant_doc_ids={"doc2"},
        expected_terms={"低空经济"},
    )

    # expected_terms 用于额外计算关键词覆盖率，不影响文档相关性 ground truth。
    result = evaluator.evaluate([sample], top_k=3)

    query_result = result.per_query_results[0]
    # 原始分块顺序是 doc1、doc1、doc2；文档去重后应变成 doc1、doc2。
    assert query_result["retrieved_documents"] == ["doc1", "doc2"]
    assert query_result["hit_rank"] == 2
    # doc2 排名第 2：Recall@1=0，Recall@2=1，MRR 理论值为 1/2；NDCG 也会
    # 根据第 2 名命中进行位置折损。这里重点断言 Recall 与可诊断排名。
    assert query_result["metrics"]["recall_at_k"][1] == 0.0
    assert query_result["metrics"]["recall_at_k"][2] == 1.0
    assert query_result["metrics"]["term_coverage"] == 1.0
    assert query_result["retrieved_sources"][0]["file_name"] == "中信海直.pdf"
    assert result.avg_evidence_page_recall_at_k == {}


def test_evaluator_passes_topic_to_official_retrieval_entry():
    """带主题的评估必须调用正式 retrieve 入口并原样下传 topic。"""
    retriever = TopicRecordingRetriever()
    evaluator = RAGEvaluator(
        k_values=[1],
        retriever=retriever,
        topic="embodied_intelligence",
    )
    sample = EvaluationSample(
        query="凌云光财务表现",
        relevant_doc_ids={"doc1"},
    )

    evaluator.evaluate([sample], top_k=5)

    # 精确比较调用列表，同时验证只调用一次、top_k 和主题值都正确。
    assert retriever.calls == [{
        "query": "凌云光财务表现",
        "top_k": 5,
        "topic": "embodied_intelligence",
    }]


def test_evidence_page_recall_exposes_report_level_false_positive():
    """RA-RAG-002：命中正确报告但页码错误，不能算作证据命中。"""
    evaluator = RAGEvaluator(k_values=[1, 2, 3], retriever=EvidencePageRetriever())
    sample = EvaluationSample(
        query="利润为什么承压？",
        relevant_doc_ids={"doc1"},
        relevant_evidence_pages={"doc1": {1}},
    )

    result = evaluator.evaluate([sample], top_k=3).per_query_results[0]

    # 第 1 个分块已来自 doc1，所以文档级指标认为首位命中。
    assert result["hit_rank"] == 1
    assert result["metrics"]["recall_at_k"][1] == 1.0
    # 但人工标注的真正证据在 doc1 第 1 页，到第 3 个分块才首次出现。
    assert result["evidence_hit_rank"] == 3
    assert result["metrics"]["evidence_page_recall_at_k"] == {
        1: 0.0,
        2: 0.0,
        3: 1.0,
    }


def test_term_coverage_at_k_exposes_top_five_gap():
    """RA-EVAL-001：全量覆盖率不能掩盖 Top-5 中事实不完整的问题。"""
    evaluator = RAGEvaluator(k_values=[5, 10], retriever=CoverageDepthRetriever())
    sample = EvaluationSample(
        query="两个事实是否都已召回？",
        relevant_doc_ids={"doc1"},
        expected_terms={"第一事实", "第二事实"},
    )

    result = evaluator.evaluate([sample], top_k=10)
    metrics = result.per_query_results[0]["metrics"]

    # 全部 Top-10 包含两个事实，所以总体覆盖率是 1；Top-5 只含一个，覆盖率为 0.5。
    assert metrics["term_coverage"] == 1.0
    assert metrics["term_coverage_at_k"] == {5: 0.5, 10: 1.0}
    assert result.avg_term_coverage_at_k == {5: 0.5, 10: 1.0}


def test_evidence_group_accepts_equivalent_fact_bearing_page():
    """RA-EVAL-001：包含完整事实的等价页面应被证据组指标接受。"""
    evaluator = RAGEvaluator(k_values=[1], retriever=AlternativeEvidenceRetriever())
    sample = EvaluationSample.from_dict({
        "id": "EI-PROG-002",
        "query": "新增了哪些产能？",
        "relevant_doc_ids": ["doc1"],
        "expected_terms": ["谐波减速器", "执行器模组"],
        # 传统证据页只标第 1 页；evidence_groups 声明第 1 或第 4 页都可承载该事实。
        "evidence": [{"file_id": "doc1", "pages": [1]}],
        "evidence_groups": [{
            "id": "capacity_layout",
            "terms": ["谐波减速器", "执行器模组"],
            "alternatives": [{"file_id": "doc1", "pages": [1, 4]}],
        }],
    })

    evaluation = evaluator.evaluate([sample], top_k=1)
    result = evaluation.per_query_results[0]

    assert evaluation.num_evidence_group_queries == 1
    # 返回第 4 页：严格页码指标为 0，但等价证据组命中且两个锚点词都存在，所以为 1。
    assert result["metrics"]["evidence_page_recall_at_k"] == {1: 0.0}
    assert result["metrics"]["evidence_group_recall_at_k"] == {1: 1.0}
    assert result["evidence_group_hit_ranks"] == {"capacity_layout": 1}


def test_evidence_group_rejects_irrelevant_chunk_on_correct_page():
    """RA-EVAL-001：仅页码相同但正文无锚点词，不能算事实证据命中。"""
    evaluator = RAGEvaluator(k_values=[1, 2], retriever=SamePageEvidenceRetriever())
    sample = EvaluationSample.from_dict({
        "query": "利润为什么承压？",
        "relevant_doc_ids": ["doc1"],
        "evidence": [{"file_id": "doc1", "pages": [1]}],
        "evidence_groups": [{
            "id": "profit_pressure_causes",
            "terms": ["汇兑损失", "POC阶段"],
            "alternatives": [{"file_id": "doc1", "pages": [1]}],
        }],
    })

    result = evaluator.evaluate([sample], top_k=2).per_query_results[0]

    # 第 1 条已在正确页面，因此页码召回为 1；但它是无关表格，不含两个事实锚点。
    assert result["metrics"]["evidence_page_recall_at_k"] == {1: 1.0, 2: 1.0}
    # 到第 2 条原因正文出现后，证据组才真正命中。
    assert result["metrics"]["evidence_group_recall_at_k"] == {1: 0.0, 2: 1.0}
    assert result["evidence_group_hit_ranks"] == {"profit_pressure_causes": 2}


def test_term_coverage_normalizes_pdf_whitespace_and_line_breaks():
    """关键词覆盖匹配应容忍 PDF 抽取产生的空格和中文断行。"""
    evaluator = RAGEvaluator(retriever=FakeRetriever())
    context = RetrievedContext(
        query="项目阶段",
        results=[
            RetrievalResult(
                # PDF 常把“POC阶段”插入空格，也可能把“信用”拆到两行。
                content="项目处于客户导入与 POC 阶段，信\n用减值损失有所增加。",
                metadata={"file_id": "doc1", "page_number": 1},
                score=0.9,
            ),
        ],
        sources=[],
    )

    coverage = evaluator.calculate_term_coverage(
        context,
        {"POC阶段", "信用减值损失"},
    )

    assert coverage == 1.0


def test_evaluation_sample_from_dict_reads_expected_terms():
    """JSON 样本解析应保留关键词、证据页、事实组和元数据，并忽略坏页码。"""
    sample = EvaluationSample.from_dict({
        "id": "EI-CAUSE-001",
        "query": "盈利怎么样",
        "relevant_doc_ids": ["doc1"],
        "expected_terms": ["净利润", "毛利率"],
        # 页码既可能是整数也可能是数字字符串；无法转为整数的 bad 应被跳过。
        "evidence": [{"file_id": "doc1", "pages": [1, "2", "bad"]}],
        "evidence_groups": [{
            "id": "profitability",
            "terms": ["净利润", "毛利率"],
            "alternatives": [{"file_id": "doc1", "pages": [1, "2", "bad"]}],
        }],
        "metadata": {"category": "业绩分析"},
    })

    assert sample.sample_id == "EI-CAUSE-001"
    assert sample.query == "盈利怎么样"
    assert sample.relevant_doc_ids == {"doc1"}
    assert sample.expected_terms == {"净利润", "毛利率"}
    # 集合结构自动去重，frozenset 用于让证据组成为不可变评估配置。
    assert sample.relevant_evidence_pages == {"doc1": {1, 2}}
    assert sample.evidence_groups is not None
    assert sample.evidence_groups[0].group_id == "profitability"
    assert sample.evidence_groups[0].terms == frozenset({"净利润", "毛利率"})
    assert sample.evidence_groups[0].alternatives[0].pages == frozenset({1, 2})
    assert sample.metadata == {"category": "业绩分析"}


def test_create_evaluation_dataset_from_reports_skips_tiny_placeholders(tmp_path):
    """按报告文件名生成评估集时，应跳过过小的占位 PDF。"""
    # tmp_path 保证测试生成的 PDF 与 JSON 不污染真实 reports 目录。
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    tiny_pdf = reports_dir / "00000000-0000-0000-0000-000000000000_test.pdf"
    real_pdf = reports_dir / "doc123_【创益通】连接器小巨人，布局低空经济打开想象空间.pdf"
    # 默认最小文件大小为 1024 字节：tiny 被过滤，2048 字节的 real_pdf 被采用。
    tiny_pdf.write_bytes(b"tiny")
    real_pdf.write_bytes(b"x" * 2048)

    output_path = tmp_path / "eval.json"
    samples = create_evaluation_dataset_from_reports(reports_dir, output_path)

    assert len(samples) == 1
    # 生成器从“file_id_【公司】标题.pdf”中提取查询、相关文档和标题关键词弱标签。
    assert samples[0]["query"] == "创益通连接器小巨人，布局低空经济打开想象空间"
    assert samples[0]["relevant_doc_ids"] == ["doc123"]
    assert "低空经济" in samples[0]["expected_terms"]
    assert output_path.exists()
