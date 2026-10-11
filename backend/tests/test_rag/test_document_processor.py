"""文档处理器的单元测试。

按“参数初始化 → 文本与页面清洗 → 分块及元数据 → 表格处理”的顺序阅读。
这些用例直接构造文本、页面字典和表格数据，不依赖真实 PDF 或外部服务，
因此可以聚焦观察 DocumentProcessor 的输入、输出与边界行为。
"""


from app.rag.document_processor import DocumentChunk, DocumentProcessor, ParsedDocument


class TestDocumentProcessor:
    """验证 DocumentProcessor 的清洗、切分及元数据保留逻辑。"""

    def test_init_default_params(self):
        """不传参数时，使用默认的分块大小和重叠长度。"""
        processor = DocumentProcessor()
        # 分块大小和重叠长度会影响后续文本切分及相邻块的上下文连续性。
        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 200

    def test_init_custom_params(self):
        """显式传入的切分参数应覆盖默认值。"""
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 100

    def test_clean_text(self):
        """清理行首尾空白后，正文内容仍应保留。"""
        processor = DocumentProcessor()

        # 模拟 PDF 提取文本时常见的多余空格和空行。
        dirty_text = "  Hello   World  \n\n  Test  "
        clean = processor._clean_text(dirty_text)

        # 先检查有效词没有丢失，再检查清洗结果两端没有空格。
        assert "Hello" in clean
        assert "World" in clean
        assert "Test" in clean
        assert not clean.startswith(" ")
        assert not clean.endswith(" ")

    def test_clean_text_removes_page_number_noise(self):
        """不同格式的独立页码行应被移除，正文不受影响。"""
        processor = DocumentProcessor()

        # 同时覆盖横线页码、中文页码、分数形式和英文 Page 形式。
        dirty_text = "\n".join([
            "公司深度报告",
            "- 12 -",
            "第 12 页",
            "12 / 80",
            "Page 12 of 80",
            "收入增长主要来自海外业务扩张。",
        ])

        clean = processor._clean_text(dirty_text)

        # 负向断言证明噪声已删除；正向断言防止清洗过度。
        assert "- 12 -" not in clean
        assert "第 12 页" not in clean
        assert "12 / 80" not in clean
        assert "Page 12 of 80" not in clean
        assert "收入增长主要来自海外业务扩张。" in clean

    def test_clean_text_removes_disclaimer_lines(self):
        """切分前过滤常见免责声明，同时保留分析正文。"""
        processor = DocumentProcessor()

        dirty_text = "\n".join([
            "请务必阅读正文之后的免责声明",
            "本报告仅供内部参考",
            "盈利预测上调，维持买入评级。",
        ])

        clean = processor._clean_text(dirty_text)

        assert "免责声明" not in clean
        assert "本报告仅供内部参考" not in clean
        assert "盈利预测上调，维持买入评级。" in clean

    def test_clean_pages_removes_repeated_headers_and_footers(self):
        """跨页重复的页眉、页脚应当删除，逐页特有的正文应当保留。"""
        processor = DocumentProcessor()

        # 三页共享标题与免责声明，但每页的核心内容不同。
        pages = [
            {
                "page_number": 1,
                "content": "某某证券研究报告\n行业深度\n核心观点：需求持续改善。\n请务必阅读正文之后的免责声明",
            },
            {
                "page_number": 2,
                "content": "某某证券研究报告\n行业深度\n财务表现：毛利率提升。\n请务必阅读正文之后的免责声明",
            },
            {
                "page_number": 3,
                "content": "某某证券研究报告\n行业深度\n风险因素：海外需求波动。\n请务必阅读正文之后的免责声明",
            },
        ]

        cleaned_pages = processor._clean_pages(pages)
        combined = "\n".join(page["content"] for page in cleaned_pages)

        # 将清洗后的页面拼起来，分别检查重复噪声和独有正文。
        assert "某某证券研究报告" not in combined
        assert "行业深度" not in combined
        assert "免责声明" not in combined
        assert "核心观点：需求持续改善。" in combined
        assert "财务表现：毛利率提升。" in combined
        assert "风险因素：海外需求波动。" in combined

    def test_chunk_document_preserves_page_metadata_after_cleaning(self):
        """正文里的页码噪声被删除后，分块仍要保留真实页码元数据。"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)

        # 页面文本中的“- 12 -”是噪声；page_number=12 才是结构化页码。
        pages = processor._clean_pages([
            {
                "page_number": 12,
                "content": "公司研究\n- 12 -\n收入增长主要来自海外业务扩张。",
            }
        ])

        document = ParsedDocument(
            file_id="file123",
            file_name="report.pdf",
            content="\n\n".join(page["content"] for page in pages),
            pages=pages,
            metadata={
                "file_id": "file123",
                "file_name": "report.pdf",
                "source": "report.pdf",
            },
        )

        chunks = processor.chunk_document(document)

        # 同时检查“删除正文噪声”和“保留定位信息”，避免把两者混淆。
        assert chunks
        assert "- 12 -" not in chunks[0].content
        assert chunks[0].metadata["page_number"] == 12

    def test_chunk_document_can_span_adjacent_pages(self):
        """相邻短页面可以合并成一个分块，元数据应标出跨页范围。"""
        processor = DocumentProcessor(chunk_size=300, chunk_overlap=20)
        # 两页内容总长度小于 chunk_size，可用于检验跨页合并。
        pages = [
            {
                "page_number": 1,
                "content": "一、核心观点\n需求持续改善，订单保持增长。",
                "char_count": 22,
            },
            {
                "page_number": 2,
                "content": "盈利能力提升，费用率稳定。",
                "char_count": 13,
            },
        ]
        document = ParsedDocument(
            file_id="file123",
            file_name="report.pdf",
            content="\n\n".join(page["content"] for page in pages),
            pages=pages,
            metadata={
                "file_id": "file123",
                "file_name": "report.pdf",
                "source": "report.pdf",
            },
        )

        chunks = processor.chunk_document(document)

        # page_number 指首个页面；page_start/end 和 page_numbers 表示完整范围。
        assert len(chunks) == 1
        assert chunks[0].metadata["page_number"] == 1
        assert chunks[0].metadata["page_start"] == 1
        assert chunks[0].metadata["page_end"] == 2
        assert chunks[0].metadata["page_numbers"] == "1,2"
        # 跨页分块还应携带可检索的章节标题。
        assert chunks[0].metadata["section_title"] == "一、核心观点"

    def test_chunk_document_adds_section_title_metadata(self):
        """分块应识别章节标题，并把标题写入元数据。"""
        processor = DocumentProcessor(chunk_size=120, chunk_overlap=10)
        pages = [
            {
                "page_number": 3,
                "content": "2.1 财务分析\n收入同比增长，毛利率持续改善。",
                "char_count": 23,
            }
        ]
        document = ParsedDocument(
            file_id="file123",
            file_name="report.pdf",
            content=pages[0]["content"],
            pages=pages,
            metadata={
                "file_id": "file123",
                "file_name": "report.pdf",
                "source": "report.pdf",
            },
        )

        chunks = processor.chunk_document(document)

        # 单页分块的起止页相同，标题来自该页正文中的章节行。
        assert chunks
        assert chunks[0].metadata["page_start"] == 3
        assert chunks[0].metadata["page_end"] == 3
        assert chunks[0].metadata["section_title"] == "2.1 财务分析"

    def test_table_to_markdown_cleans_empty_rows_and_columns(self):
        """表格清洗应去掉全空行列，再转换为可检索的 Markdown 文本。"""
        processor = DocumentProcessor()

        # 最后一列和最后一行都是空值，用于验证清理逻辑。
        table = [
            ["指标", "2023", "2024", None],
            ["营收", "100", "120", None],
            ["毛利率", "30%", "35%", None],
            [None, None, None, None],
        ]

        cleaned = processor._clean_table(table)
        markdown = processor._table_to_markdown(cleaned)

        # 先检查二维数据结构，再检查 Markdown 表头、分隔行和正文。
        assert cleaned == [
            ["指标", "2023", "2024"],
            ["营收", "100", "120"],
            ["毛利率", "30%", "35%"],
        ]
        assert "| 指标 | 2023 | 2024 |" in markdown
        assert "| --- | --- | --- |" in markdown
        assert "| 营收 | 100 | 120 |" in markdown

    def test_chunk_document_adds_table_chunks(self):
        """页面中的表格应生成独立分块，并保留表格位置和结构信息。"""
        processor = DocumentProcessor(chunk_size=300, chunk_overlap=20)
        # 模拟解析器已从第 5 页提取出 Markdown 表格及行列数。
        pages = [
            {
                "page_number": 5,
                "content": "2.1 财务分析\n收入增长稳健。",
                "char_count": 15,
                "tables": [
                    {
                        "table_index": 1,
                        "markdown": "| 指标 | 2023 | 2024 |\n| --- | --- | --- |\n| 营收 | 100 | 120 |",
                        "row_count": 2,
                        "column_count": 3,
                    }
                ],
            },
        ]
        document = ParsedDocument(
            file_id="file123",
            file_name="report.pdf",
            content=pages[0]["content"],
            pages=pages,
            metadata={
                "file_id": "file123",
                "file_name": "report.pdf",
                "source": "report.pdf",
            },
        )

        chunks = processor.chunk_document(document)
        # 文本分块与表格分块会一起返回，用 content_type 只筛出表格。
        table_chunks = [
            chunk for chunk in chunks
            if chunk.metadata.get("content_type") == "table"
        ]

        assert table_chunks
        # 除表格正文外，页码、表格序号、行列数和章节标题也要可追溯。
        assert "表格内容" in table_chunks[0].content
        assert "| 指标 | 2023 | 2024 |" in table_chunks[0].content
        assert table_chunks[0].metadata["page_number"] == 5
        assert table_chunks[0].metadata["table_index"] == 1
        assert table_chunks[0].metadata["row_count"] == 2
        assert table_chunks[0].metadata["column_count"] == 3
        assert table_chunks[0].metadata["section_title"] == "2.1 财务分析"

    def test_extract_page_tables_filters_weak_tables(self):
        """表格提取应过滤结构不足的表格，只保留可索引的有效表格。"""
        processor = DocumentProcessor()

        # 假页面替代 pdfplumber 页面，避免依赖实际 PDF 文件。
        class FakePage:
            def extract_tables(self):
                return [
                    # 两行两列，有足够的结构化信息。
                    [["指标", "2023"], ["营收", "100"]],
                    # 仅一行，不满足有效表格条件。
                    [["只有一行", "无效"]],
                ]

        tables = processor._extract_page_tables(FakePage())

        # 只留下第一张表，并验证编号、规模和转换后的文本。
        assert len(tables) == 1
        assert tables[0]["table_index"] == 1
        assert tables[0]["row_count"] == 2
        assert tables[0]["column_count"] == 2
        assert "| 指标 | 2023 |" in tables[0]["markdown"]

    def test_chunk_text_simple(self):
        """较长文本应被切为非空的 DocumentChunk 对象。"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

        # 重复文本长度超过 chunk_size，确保切分器有内容可处理。
        text = "这是一个测试文本。" * 20
        chunks = processor._chunk_text(text)

        # all() 对空列表也会返回 True，因此先确认确实生成了分块。
        assert len(chunks) > 0
        # 生成器逐个检查 c；只有每个元素都是 DocumentChunk，all() 才返回 True。
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        # 逐个检查 content 的真值；空字符串为 False，可发现空内容分块。
        assert all(c.content for c in chunks)

    def test_chunk_text_with_metadata(self):
        """额外元数据应复制到每个分块，且每块应有自身序号。"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

        text = "这是测试内容。" * 20
        metadata = {"file_id": "test123", "source": "test.pdf"}

        chunks = processor._chunk_text(text, extra_metadata=metadata)

        assert len(chunks) > 0
        # file_id/source 用于回溯原文档；chunk_index 用于标识分块顺序。
        for chunk in chunks:
            assert chunk.metadata["file_id"] == "test123"
            assert chunk.metadata["source"] == "test.pdf"
            assert "chunk_index" in chunk.metadata

    def test_chunk_empty_text(self):
        """空字符串和仅含空白字符的文本都不应产生分块。"""
        processor = DocumentProcessor()

        chunks = processor._chunk_text("")
        assert chunks == []

        chunks = processor._chunk_text("   \n\n  ")
        assert chunks == []

    def test_chinese_separators(self):
        """中文句号分隔的文本应能进入正常切分流程。"""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)

        text = "第一段内容。第二段内容。第三段内容。第四段内容。"
        chunks = processor._chunk_text(text)

        # 当前用例只验证能生成分块，没有断言实际切分位置。
        assert len(chunks) > 0


class TestDocumentChunk:
    """验证 DocumentChunk 数据类的输出格式。"""

    def test_to_dict(self):
        """to_dict 应原样保留正文和元数据，供后续向量存储使用。"""
        chunk = DocumentChunk(
            content="Test content",
            metadata={"file_id": "test", "page": 1}
        )

        result = chunk.to_dict()

        # 分别检查顶层正文与 metadata 中的来源字段。
        assert result["content"] == "Test content"
        assert result["metadata"]["file_id"] == "test"
        assert result["metadata"]["page"] == 1
