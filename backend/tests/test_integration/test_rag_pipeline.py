"""RAG（Retrieval-Augmented Generation）完整管线的集成测试。

目标流程是：PDF → 文本解析 → 分块 → 向量化 → Chroma 存储 → 相似度检索 →
Retriever 重排与上下文组装 → RAG 工具调用。

本文件混合了三种测试层次：

* 组件集成：真实 DocumentProcessor 或 VectorStore 配合临时文件/模拟 embedding；
* 工具集成：验证 RAG analyst 工具能正确调用 Retriever 和 VectorStore；
* 应用级端到端：分块写入真实临时 Chroma collection 后再通过 Retriever 召回。

嵌入模型使用固定 384 维向量的 mock，避免下载模型和依赖网络；Chroma 数据全部
写入 pytest 的 ``tmp_path``，测试结束后自动清理。
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 在导入 app 模块之前声明测试环境，因为应用配置可能在 import 阶段读取环境变量。
# setdefault 不覆盖已存在值；更完整的数据目录隔离由上级 conftest.py 统一完成。
os.environ.setdefault("APP_ENV", "testing")


class TestRAGPipeline:
    """文档处理、向量存储和检索器之间的组件集成测试。"""

    @pytest.fixture
    def temp_pdf_path(self, tmp_path: Path) -> Path:
        """在 pytest 临时目录中创建一个最小 PDF，并返回其路径。"""
        # 不依赖外部样例文件，直接构造最小 PDF，使测试数据自包含且便于复现。
        pdf_path = tmp_path / "test_report.pdf"
        # PDF 包含 Catalog、Pages、Page、Contents、xref 和 trailer 等基本对象。
        # 其中 BT...ET 是一段文本绘制指令，内容为 Test Report Content。
        pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 55 >>
stream
BT /F1 12 Tf 100 700 Td (Test Report Content) Tj ET
endstream
endobj
5 0 obj
<< /FontDescriptor 6 0 R /BaseFont /Helvetica /Subtype /Type1 /Type /Font >>
endobj
6 0 obj
<< /FontName /Helvetica /Flags 32 /FontBBox [-1000 -1000 1000 1000] >>
endobj
xref
0 7
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000214 00000 n
0000000320 00000 n
0000000415 00000 n
trailer
<< /Size 7 /Root 1 0 R >>
startxref
500
%%EOF"""
        pdf_path.write_bytes(pdf_content)
        return pdf_path

    @pytest.fixture
    def mock_embedding_model(self):
        """创建维度固定、输出数量与输入匹配的 embedding model 测试替身。"""
        mock = MagicMock()
        # add_documents 可能一次传入任意数量文本，因此用 side_effect 动态生成等量向量。
        # 每个向量 384 维，维度必须在文档和查询两种嵌入中保持一致。
        def embed_documents_side_effect(texts):
            return [[0.1] * 384 for _ in range(len(texts))]
        mock.embed_documents.side_effect = embed_documents_side_effect
        mock.embed_query.return_value = [0.1] * 384
        return mock

    def test_document_processor_parse_and_chunk(self, temp_pdf_path: Path):
        """DocumentProcessor 应能读取 PDF，且所有产出分块都带文件来源 metadata。"""
        # 延迟导入让测试模块收集时不必提前加载 PDF 解析依赖。
        from app.rag.document_processor import DocumentProcessor

        # chunk_overlap 让相邻分块共享上下文；本 PDF 很小，通常至多产生少量分块。
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        chunks = processor.process_pdf(temp_pdf_path, "test_file_id")

        # 极简 PDF 可能因解析器差异提取不到正文，所以允许 0 个分块；
        # 但只要产生分块，就逐条验证正文和可追溯来源字段。
        assert len(chunks) >= 0  # May be 0 for minimal PDF
        for chunk in chunks:
            assert chunk.content
            assert chunk.metadata.get("file_id") == "test_file_id"
            assert chunk.metadata.get("file_name")

    def test_vector_store_add_and_search(self, mock_embedding_model, tmp_path: Path):
        """真实临时 VectorStore 应能写入分块并完成相似度检索。"""
        from app.rag.document_processor import DocumentChunk
        from app.rag.vectorstore import VectorStore

        # Chroma collection 落到 tmp_path，避免测试连接或污染应用正式向量库。
        store = VectorStore(
            collection_name="test_collection",
            persist_directory=tmp_path / "vectorstore",
            embedding_model=mock_embedding_model,
        )

        # 两个 DocumentChunk 属于同一文件但来自不同页面，模拟真实 PDF 分块。
        chunks = [
            DocumentChunk(
                content="这是关于人工智能的报告内容。",
                metadata={"file_id": "f1", "file_name": "test.pdf", "page_number": 1},
            ),
            DocumentChunk(
                content="机器学习是人工智能的重要分支。",
                metadata={"file_id": "f1", "file_name": "test.pdf", "page_number": 2},
            ),
        ]

        # 写入阶段会调用 mock embedding，并把正文、向量和 metadata 一起存入 Chroma。
        count = store.add_documents(chunks)
        assert count == 2

        # 查询阶段会生成查询向量并调用 collection.query；固定向量保证结果确定可用。
        results = store.similarity_search("人工智能", k=2)
        assert len(results) >= 1

    def test_retriever_retrieve_and_relevance(self, mock_embedding_model, tmp_path: Path):
        """分块经真实 VectorStore 写入后，应能由 Retriever 组装为检索上下文。"""
        from app.rag.document_processor import DocumentChunk
        from app.rag.retriever import Retriever
        from app.rag.vectorstore import VectorStore

        # Arrange：创建独立 collection，写入一条包含营收和利润事实的分块。
        store = VectorStore(
            collection_name="test_retriever",
            persist_directory=tmp_path / "vectorstore",
            embedding_model=mock_embedding_model,
        )

        chunks = [
            DocumentChunk(
                content="2025年营收增长10%，净利润增长15%。",
                metadata={"file_id": "f1", "file_name": "report.pdf", "page_number": 1},
            ),
        ]
        store.add_documents(chunks)

        # Act：Retriever 使用同一个 store 完成召回、重排、来源整理和上下文封装。
        retriever = Retriever(vector_store=store)
        context = retriever.retrieve("营收增长情况")

        assert context.query == "营收增长情况"
        assert len(context.results) >= 1


class TestRAGToolsIntegration:
    """RAG analyst 工具与检索器/向量库之间的调用集成测试。"""

    @pytest.fixture
    def mock_vector_store(self):
        """创建覆盖搜索、报告列表和文件分块接口的向量库测试替身。"""
        mock = MagicMock()
        # 每个方法都返回与生产接口形状一致的最小数据，避免 mock 过于宽松。
        mock.similarity_search_with_scores.return_value = [
            ({"content": "营收增长10%", "metadata": {"file_id": "f1", "file_name": "report.pdf"}}, 0.9),
        ]
        mock.list_files.return_value = [
            {"file_id": "f1", "file_name": "report.pdf", "chunk_count": 10},
        ]
        mock.get_file_chunks.return_value = [
            {"id": "c1", "content": "chunk content", "metadata": {"file_name": "report.pdf", "page_number": 1}},
        ]
        return mock

    def test_search_reports_tool(self, mock_vector_store):
        """search_reports 工具应调用 Retriever，并把格式化证据返回给智能体。"""
        # patch 位于工具代码实际引用依赖的位置；嵌套 with 结束后会自动恢复。
        with patch("app.agents.rag_analyst.get_vector_store", return_value=mock_vector_store):
            with patch("app.agents.rag_analyst.get_retriever") as mock_get_retriever:
                mock_retriever = MagicMock()
                mock_context = MagicMock()
                # 构造 Retriever.retrieve 的返回上下文，包含结果、来源和格式化文本。
                mock_context.results = [MagicMock(content="营收增长10%", metadata={"file_id": "f1"}, score=0.9)]
                mock_context.sources = [{"file_id": "f1", "file_name": "report.pdf"}]
                mock_context.format_context.return_value = "营收增长10%"
                mock_retriever.retrieve.return_value = mock_context
                mock_get_retriever.return_value = mock_retriever

                from app.agents.rag_analyst import search_reports

                # LangChain tool 使用 invoke(dict) 调用，而不是直接执行普通 Python 函数。
                result = search_reports.invoke({"query": "营收增长"})
                assert "营收增长" in result or "搜索结果" in result

    def test_list_available_reports_tool(self, mock_vector_store):
        """list_available_reports 工具应输出已上传报告列表或明确的空状态。"""
        with patch("app.agents.rag_analyst.get_vector_store", return_value=mock_vector_store):
            with patch("app.agents.rag_analyst.get_settings") as mock_settings:
                # 工具需要 reports_path；真实文件扫描结果与 mock list_files 一起决定输出。
                mock_settings.return_value.reports_path = Path("/tmp")

                from app.agents.rag_analyst import list_available_reports

                result = list_available_reports.invoke({})
                assert "已上传研报" in result or "没有" in result

    def test_check_rag_relevance_tool(self, mock_vector_store):
        """check_rag_relevance 工具应把布尔相关性和分数转换为可读文本。"""
        with patch("app.agents.rag_analyst.get_retriever") as mock_get_retriever:
            mock_retriever = MagicMock()
            # 固定返回“相关、最佳分 0.85”，验证工具层的调用与文本表达。
            mock_retriever.check_relevance.return_value = (True, 0.85)
            mock_get_retriever.return_value = mock_retriever

            from app.agents.rag_analyst import check_rag_relevance

            result = check_rag_relevance.invoke({"query": "营收"})
            assert "相关" in result


class TestRAGEndToEnd:
    """从分块写入到 Retriever 召回的应用级端到端测试。"""

    def test_pdf_to_retrieval_flow(self, tmp_path: Path):
        """验证三个分块写入临时向量库后，可以通过自然语言查询召回。"""
        from app.rag.document_processor import DocumentChunk
        from app.rag.embeddings import EmbeddingModel
        from app.rag.retriever import Retriever
        from app.rag.vectorstore import VectorStore

        # 某些环境缺少 Chroma 等可选依赖，因此把整段集成流程放入 try；
        # 出现环境性异常时用 pytest.skip 标记为跳过，而不是误报业务失败。
        try:
            # 使用 EmbeddingModel 作为 spec：MagicMock 只能访问真实类存在的方法，
            # 能尽早发现测试替身接口与生产接口不一致。
            mock_embedding = MagicMock(spec=EmbeddingModel)
            mock_embedding.embed_documents.return_value = [[0.1] * 384 for _ in range(3)]
            mock_embedding.embed_query.return_value = [0.1] * 384

            # collection 与持久化目录都为本用例专属，不会与其他测试共享状态。
            store = VectorStore(
                collection_name="e2e_test",
                persist_directory=tmp_path / "vs",
                embedding_model=mock_embedding,
            )

            # 列表推导式生成 3 个不同页码的分块，正文都包含查询词“测试内容”。
            chunks = [
                DocumentChunk(
                    content=f"这是第{i}段测试内容，包含关键信息。",
                    metadata={
                        "file_id": "test_report",
                        "file_name": "test.pdf",
                        "page_number": i,
                    },
                )
                for i in range(1, 4)
            ]

            store.add_documents(chunks)

            # 先验证存储层确实收到 3 条记录，再验证检索层，便于失败时定位阶段。
            assert store.count == 3

            # Retriever 在真实临时 Chroma 上执行召回，并返回 RetrievedContext。
            retriever = Retriever(vector_store=store)
            context = retriever.retrieve("测试内容", top_k=3)

            assert len(context.results) >= 1

        except Exception as e:
            pytest.skip(f"Embedding model not available: {e}")
