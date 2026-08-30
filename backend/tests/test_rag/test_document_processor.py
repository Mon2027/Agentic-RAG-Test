"""Tests for document processor."""

import pytest
from pathlib import Path

from app.rag.document_processor import DocumentProcessor, DocumentChunk, ParsedDocument


class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        processor = DocumentProcessor()
        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 200

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 100

    def test_clean_text(self):
        """Test text cleaning functionality."""
        processor = DocumentProcessor()

        dirty_text = "  Hello   World  \n\n  Test  "
        clean = processor._clean_text(dirty_text)

        # _clean_text strips leading/trailing whitespace from each line
        assert "Hello" in clean
        assert "World" in clean
        assert "Test" in clean
        # Check that leading/trailing spaces are removed from lines
        assert not clean.startswith(" ")
        assert not clean.endswith(" ")

    def test_clean_text_removes_page_number_noise(self):
        """Test page number lines are removed from extracted content."""
        processor = DocumentProcessor()

        dirty_text = "\n".join([
            "公司深度报告",
            "- 12 -",
            "第 12 页",
            "12 / 80",
            "Page 12 of 80",
            "收入增长主要来自海外业务扩张。",
        ])

        clean = processor._clean_text(dirty_text)

        assert "- 12 -" not in clean
        assert "第 12 页" not in clean
        assert "12 / 80" not in clean
        assert "Page 12 of 80" not in clean
        assert "收入增长主要来自海外业务扩张。" in clean

    def test_clean_text_removes_disclaimer_lines(self):
        """Test common report disclaimers are removed before chunking."""
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
        """Test repeated top/bottom lines are treated as header/footer noise."""
        processor = DocumentProcessor()

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

        assert "某某证券研究报告" not in combined
        assert "行业深度" not in combined
        assert "免责声明" not in combined
        assert "核心观点：需求持续改善。" in combined
        assert "财务表现：毛利率提升。" in combined
        assert "风险因素：海外需求波动。" in combined

    def test_chunk_document_preserves_page_metadata_after_cleaning(self):
        """Test content page number noise removal does not remove chunk metadata."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)

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

        assert chunks
        assert "- 12 -" not in chunks[0].content
        assert chunks[0].metadata["page_number"] == 12

    def test_chunk_document_can_span_adjacent_pages(self):
        """Test short adjacent pages can form a cross-page chunk with page ranges."""
        processor = DocumentProcessor(chunk_size=300, chunk_overlap=20)
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

        assert len(chunks) == 1
        assert chunks[0].metadata["page_number"] == 1
        assert chunks[0].metadata["page_start"] == 1
        assert chunks[0].metadata["page_end"] == 2
        assert chunks[0].metadata["page_numbers"] == "1,2"
        assert chunks[0].metadata["section_title"] == "一、核心观点"

    def test_chunk_document_adds_section_title_metadata(self):
        """Test section headings are added to chunk metadata."""
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

        assert chunks
        assert chunks[0].metadata["page_start"] == 3
        assert chunks[0].metadata["page_end"] == 3
        assert chunks[0].metadata["section_title"] == "2.1 财务分析"

    def test_table_to_markdown_cleans_empty_rows_and_columns(self):
        """Test raw PDF tables are normalized and converted to Markdown."""
        processor = DocumentProcessor()

        table = [
            ["指标", "2023", "2024", None],
            ["营收", "100", "120", None],
            ["毛利率", "30%", "35%", None],
            [None, None, None, None],
        ]

        cleaned = processor._clean_table(table)
        markdown = processor._table_to_markdown(cleaned)

        assert cleaned == [
            ["指标", "2023", "2024"],
            ["营收", "100", "120"],
            ["毛利率", "30%", "35%"],
        ]
        assert "| 指标 | 2023 | 2024 |" in markdown
        assert "| --- | --- | --- |" in markdown
        assert "| 营收 | 100 | 120 |" in markdown

    def test_chunk_document_adds_table_chunks(self):
        """Test parsed page tables are stored as standalone table chunks."""
        processor = DocumentProcessor(chunk_size=300, chunk_overlap=20)
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
        table_chunks = [
            chunk for chunk in chunks
            if chunk.metadata.get("content_type") == "table"
        ]

        assert table_chunks
        assert "表格内容" in table_chunks[0].content
        assert "| 指标 | 2023 | 2024 |" in table_chunks[0].content
        assert table_chunks[0].metadata["page_number"] == 5
        assert table_chunks[0].metadata["table_index"] == 1
        assert table_chunks[0].metadata["row_count"] == 2
        assert table_chunks[0].metadata["column_count"] == 3
        assert table_chunks[0].metadata["section_title"] == "2.1 财务分析"

    def test_extract_page_tables_filters_weak_tables(self):
        """Test table extraction keeps only useful structured tables."""
        processor = DocumentProcessor()

        class FakePage:
            def extract_tables(self):
                return [
                    [["指标", "2023"], ["营收", "100"]],
                    [["只有一行", "无效"]],
                ]

        tables = processor._extract_page_tables(FakePage())

        assert len(tables) == 1
        assert tables[0]["table_index"] == 1
        assert tables[0]["row_count"] == 2
        assert tables[0]["column_count"] == 2
        assert "| 指标 | 2023 |" in tables[0]["markdown"]

    def test_chunk_text_simple(self):
        """Test basic text chunking."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

        text = "这是一个测试文本。" * 20
        chunks = processor._chunk_text(text)

        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.content for c in chunks)

    def test_chunk_text_with_metadata(self):
        """Test chunking with metadata preservation."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

        text = "这是测试内容。" * 20
        metadata = {"file_id": "test123", "source": "test.pdf"}

        chunks = processor._chunk_text(text, extra_metadata=metadata)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata["file_id"] == "test123"
            assert chunk.metadata["source"] == "test.pdf"
            assert "chunk_index" in chunk.metadata

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        processor = DocumentProcessor()

        chunks = processor._chunk_text("")
        assert chunks == []

        chunks = processor._chunk_text("   \n\n  ")
        assert chunks == []

    def test_chinese_separators(self):
        """Test Chinese text splitting with proper separators."""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)

        text = "第一段内容。第二段内容。第三段内容。第四段内容。"
        chunks = processor._chunk_text(text)

        # Should split at Chinese periods
        assert len(chunks) > 0


class TestDocumentChunk:
    """Test cases for DocumentChunk dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        chunk = DocumentChunk(
            content="Test content",
            metadata={"file_id": "test", "page": 1}
        )

        result = chunk.to_dict()

        assert result["content"] == "Test content"
        assert result["metadata"]["file_id"] == "test"
        assert result["metadata"]["page"] == 1
