"""Document processor for PDF parsing and chunking."""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of document content with metadata."""

    content: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for vector storage."""
        return {
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass
class ParsedDocument:
    """A parsed document with its content and metadata."""

    file_id: str
    file_name: str
    content: str
    pages: list[dict[str, Any]]
    metadata: dict[str, Any]


class DocumentProcessor:
    """Process PDF documents for RAG system.

    This class handles:
    1. PDF parsing and text extraction
    2. Text chunking with configurable parameters
    3. Metadata preservation (page numbers, source file, etc.)
    """

    PAGE_NUMBER_PATTERNS = [
        re.compile(r"^\s*-+\s*\d{1,4}\s*-+\s*$"),
        re.compile(r"^\s*第\s*\d{1,4}\s*页\s*$"),
        re.compile(r"^\s*\d{1,4}\s*/\s*\d{1,4}\s*$"),
        re.compile(r"^\s*page\s+\d{1,4}(?:\s+of\s+\d{1,4})?\s*$", re.IGNORECASE),
    ]

    DISCLAIMER_PATTERNS = [
        re.compile(r"请务必阅读.*(?:免责|声明|披露)"),
        re.compile(r"(?:重要声明|免责声明|法律声明)"),
        re.compile(r"本报告仅供.*参考"),
        re.compile(r"不构成.*投资建议"),
        re.compile(r"市场有风险.*投资需谨慎"),
        re.compile(r"过往业绩不代表.*未来表现"),
        re.compile(r"未经.*(?:许可|授权).*(?:不得|禁止)"),
    ]

    SECTION_HEADING_PATTERNS = [
        re.compile(r"^(?:第[一二三四五六七八九十\d]+[章节]|[一二三四五六七八九十]+[、.．])\s*.{1,45}$"),
        re.compile(r"^(?:\d+(?:\.\d+)*[、.．\s]+|[（(][一二三四五六七八九十\d]+[）)])\s*.{1,45}$"),
    ]

    SECTION_KEYWORDS = {
        "核心观点",
        "投资要点",
        "盈利预测",
        "风险提示",
        "财务分析",
        "行业分析",
        "估值分析",
        "公司概况",
        "投资建议",
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        """Initialize the document processor.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
            separators: Custom separators for text splitting.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators optimized for Chinese and English documents
        default_separators = [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            "。",    # Chinese period
            "．",    # Full-width period
            ".",     # English period
            "！",    # Chinese exclamation
            "？",    # Chinese question mark
            "；",    # Chinese semicolon
            "：",    # Chinese colon
            " ",     # Space
            "",      # Character-level split as fallback
        ]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or default_separators,
            length_function=len,
            is_separator_regex=False,
        )

    def parse_pdf(self, file_path: Path, file_id: str) -> ParsedDocument:
        """Parse a PDF file and extract text content.

        Args:
            file_path: Path to the PDF file.
            file_id: Unique identifier for the document.

        Returns:
            ParsedDocument containing the extracted content and metadata.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ValueError: If the PDF cannot be parsed.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.info(f"Parsing PDF: {file_path}")

        raw_pages: list[dict[str, Any]] = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    tables = self._extract_page_tables(page)

                    raw_pages.append({
                        "page_number": page_num,
                        "content": text,
                        "tables": tables,
                    })
                    logger.debug(
                        f"Page {page_num}: extracted {len(text)} characters "
                        f"and {len(tables)} tables"
                    )

        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            raise ValueError(f"Failed to parse PDF: {e}") from e

        pages = self._clean_pages(raw_pages)
        full_content = [page["content"] for page in pages]
        total_content = "\n\n".join(full_content)  # Join pages with double newlines for better chunking

        document = ParsedDocument(
            file_id=file_id,
            file_name=file_path.name,
            content=total_content,
            pages=pages,
            metadata={
                "file_id": file_id,
                "file_name": file_path.name,
                "total_pages": len(pages),
                "total_chars": len(total_content),
                "source": str(file_path),
            },
        )

        logger.info(
            f"Parsed {file_path.name}: {len(pages)} pages, {len(total_content)} chars"
        )

        return document

    def chunk_document(
        self,
        document: ParsedDocument,
        preserve_page_metadata: bool = True,
    ) -> list[DocumentChunk]:
        """Chunk a parsed document into smaller pieces for embedding.

        Args:
            document: The parsed document to chunk.
            preserve_page_metadata: Whether to add page information to chunk metadata.

        Returns:
            List of DocumentChunk objects.
        """
        logger.info(f"Chunking document: {document.file_name}")

        if preserve_page_metadata:
            chunks = self._chunk_pages(document)
            chunks.extend(self._chunk_tables(document))
            self._renumber_chunks(chunks)
        else:
            # Chunk the entire document as one
            chunks = self._chunk_text(
                text=document.content,
                extra_metadata={
                    "file_id": document.file_id,
                    "file_name": document.file_name,
                    "source": document.metadata["source"],
                },
            )

        logger.info(f"Created {len(chunks)} chunks from {document.file_name}")

        return chunks

    def _chunk_pages(self, document: ParsedDocument) -> list[DocumentChunk]:
        """Chunk pages as one stream while preserving page ranges."""
        combined_text, page_spans = self._combine_pages_with_spans(document.pages)
        if not combined_text.strip():
            return []

        split_texts = self.text_splitter.split_text(combined_text)
        chunks: list[DocumentChunk] = []
        search_offset = 0

        for i, chunk_text in enumerate(split_texts):
            chunk_start = self._find_chunk_start(combined_text, chunk_text, search_offset)
            chunk_end = chunk_start + len(chunk_text)
            page_metadata = self._page_metadata_for_range(
                page_spans=page_spans,
                start=chunk_start,
                end=chunk_end,
            )

            metadata = {
                "chunk_index": i,
                "chunk_size": len(chunk_text),
                "file_id": document.file_id,
                "file_name": document.file_name,
                "source": document.metadata["source"],
                "content_type": "text",
                **page_metadata,
            }

            section_title = self._infer_section_title(
                text=combined_text,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
            )
            if section_title:
                metadata["section_title"] = section_title

            chunks.append(DocumentChunk(content=chunk_text, metadata=metadata))
            search_offset = max(chunk_start + 1, chunk_end - self.chunk_overlap)

        return chunks

    def _chunk_tables(self, document: ParsedDocument) -> list[DocumentChunk]:
        """Convert parsed tables into standalone chunks for retrieval."""
        chunks: list[DocumentChunk] = []

        for page in document.pages:
            page_number = page["page_number"]
            tables = page.get("tables") or []
            for table in tables:
                markdown = table.get("markdown", "")
                if not markdown.strip():
                    continue

                content = (
                    f"表格内容\n"
                    f"文件: {document.file_name}\n"
                    f"页码: 第{page_number}页\n"
                    f"表格: {table['table_index']}\n\n"
                    f"{markdown}"
                )
                metadata = {
                    "chunk_index": 0,
                    "chunk_size": len(content),
                    "file_id": document.file_id,
                    "file_name": document.file_name,
                    "source": document.metadata["source"],
                    "content_type": "table",
                    "page_number": page_number,
                    "page_start": page_number,
                    "page_end": page_number,
                    "page_numbers": str(page_number),
                    "table_index": table["table_index"],
                    "row_count": table["row_count"],
                    "column_count": table["column_count"],
                }

                section_title = self._infer_section_title(
                    text=page.get("content", ""),
                    chunk_start=len(page.get("content", "")),
                    chunk_end=len(page.get("content", "")),
                )
                if section_title:
                    metadata["section_title"] = section_title

                chunks.append(DocumentChunk(content=content, metadata=metadata))

        return chunks

    def _renumber_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Assign stable global chunk indexes after text/table chunks are merged."""
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index

    def _chunk_text(
        self,
        text: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Split text into chunks using the text splitter.

        Args:
            text: The text to chunk.
            extra_metadata: Additional metadata to include in each chunk.

        Returns:
            List of DocumentChunk objects.
        """
        if not text.strip():
            return []

        split_texts = self.text_splitter.split_text(text)

        chunks: list[DocumentChunk] = []
        for i, chunk_text in enumerate(split_texts):
            metadata = {
                "chunk_index": i,
                "chunk_size": len(chunk_text),
                **(extra_metadata or {}),
            }
            chunks.append(DocumentChunk(content=chunk_text, metadata=metadata))

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean extracted text.

        Args:
            text: The raw extracted text.

        Returns:
            Cleaned text.
        """
        lines = text.split("\n")
        cleaned_lines: list[str] = []

        for line in lines:
            line = self._normalize_line(line)
            if (
                line
                and not self._is_page_number_line(line)
                and not self._is_disclaimer_line(line)
            ):
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _combine_pages_with_spans(
        self,
        pages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, int]]]:
        """Combine page content and track character spans for page attribution."""
        parts: list[str] = []
        page_spans: list[dict[str, int]] = []
        cursor = 0

        for page in pages:
            content = page.get("content", "")
            if not content.strip():
                continue

            if parts:
                parts.append("\n\n")
                cursor += 2

            start = cursor
            parts.append(content)
            cursor += len(content)
            page_spans.append({
                "page_number": int(page["page_number"]),
                "start": start,
                "end": cursor,
            })

        return "".join(parts), page_spans

    def _find_chunk_start(self, text: str, chunk_text: str, search_offset: int) -> int:
        """Find chunk offset in the combined document text."""
        start = text.find(chunk_text, search_offset)
        if start >= 0:
            return start

        start = text.find(chunk_text)
        if start >= 0:
            return start

        logger.debug("Falling back to search offset for chunk attribution")
        return search_offset

    def _page_metadata_for_range(
        self,
        page_spans: list[dict[str, int]],
        start: int,
        end: int,
    ) -> dict[str, Any]:
        """Build page metadata for a chunk character range."""
        pages = [
            span["page_number"]
            for span in page_spans
            if span["start"] < end and span["end"] > start
        ]

        if not pages and page_spans:
            nearest = min(page_spans, key=lambda span: abs(span["start"] - start))
            pages = [nearest["page_number"]]

        if not pages:
            return {}

        page_start = min(pages)
        page_end = max(pages)
        return {
            "page_number": page_start,
            "page_start": page_start,
            "page_end": page_end,
            "page_numbers": ",".join(str(page) for page in sorted(set(pages))),
        }

    def _infer_section_title(
        self,
        text: str,
        chunk_start: int,
        chunk_end: int,
    ) -> str | None:
        """Infer the nearest section heading around a chunk."""
        context_start = max(0, chunk_start - 1200)
        context = text[context_start:chunk_end]

        for line in reversed(context.split("\n")):
            line = self._normalize_line(line)
            if self._is_section_heading(line):
                return line[:60]

        return None

    def _clean_pages(self, raw_pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Clean page text and remove repeated headers/footers across pages."""
        basic_pages: list[dict[str, Any]] = []
        page_lines: list[list[str]] = []

        for page in raw_pages:
            cleaned_text = self._clean_text(page.get("content", ""))
            lines = cleaned_text.split("\n") if cleaned_text else []
            basic_pages.append({
                "page_number": page["page_number"],
                "content": cleaned_text,
                "char_count": len(cleaned_text),
                "tables": page.get("tables") or [],
            })
            page_lines.append(lines)

        repeated_margin_lines = self._detect_repeated_margin_lines(page_lines)
        if not repeated_margin_lines:
            return [page for page in basic_pages if page["content"].strip()]

        cleaned_pages: list[dict[str, Any]] = []
        for page, lines in zip(basic_pages, page_lines):
            filtered_lines = [
                line for line in lines
                if self._normalize_for_repetition(line) not in repeated_margin_lines
            ]
            content = "\n".join(filtered_lines)
            if content.strip():
                cleaned_pages.append({
                    **page,
                    "content": content,
                    "char_count": len(content),
                })

        return cleaned_pages

    def _extract_page_tables(self, page: Any) -> list[dict[str, Any]]:
        """Extract tables from a pdfplumber page and convert them to Markdown."""
        try:
            raw_tables = page.extract_tables() or []
        except Exception as e:
            logger.warning(f"Failed to extract tables from page: {e}")
            return []

        tables: list[dict[str, Any]] = []
        for index, raw_table in enumerate(raw_tables, start=1):
            cleaned_table = self._clean_table(raw_table)
            if not self._is_useful_table(cleaned_table):
                continue

            markdown = self._table_to_markdown(cleaned_table)
            tables.append({
                "table_index": index,
                "markdown": markdown,
                "row_count": len(cleaned_table),
                "column_count": max((len(row) for row in cleaned_table), default=0),
            })

        return tables

    def _clean_table(self, table: list[list[Any]]) -> list[list[str]]:
        """Normalize table cells and remove empty rows/columns."""
        normalized_rows: list[list[str]] = []
        max_columns = max((len(row) for row in table if row), default=0)

        for row in table:
            normalized_row = [
                self._normalize_table_cell(cell)
                for cell in (row or [])
            ]
            normalized_row.extend([""] * (max_columns - len(normalized_row)))
            if any(cell for cell in normalized_row):
                normalized_rows.append(normalized_row)

        if not normalized_rows:
            return []

        non_empty_columns = [
            col_index
            for col_index in range(max_columns)
            if any(row[col_index] for row in normalized_rows)
        ]

        return [
            [row[col_index] for col_index in non_empty_columns]
            for row in normalized_rows
        ]

    def _normalize_table_cell(self, cell: Any) -> str:
        """Normalize a single table cell."""
        if cell is None:
            return ""
        text = self._clean_text(str(cell))
        text = text.replace("\n", " ")
        text = text.replace("|", "\\|")
        return self._normalize_line(text)

    def _is_useful_table(self, table: list[list[str]]) -> bool:
        """Return whether a table has enough structure to be indexed."""
        if len(table) < 2:
            return False
        column_count = max((len(row) for row in table), default=0)
        non_empty_cells = sum(1 for row in table for cell in row if cell)
        return column_count >= 2 and non_empty_cells >= 4

    def _table_to_markdown(self, table: list[list[str]]) -> str:
        """Convert a cleaned table to Markdown."""
        if not table:
            return ""

        column_count = max(len(row) for row in table)
        padded_rows = [
            [*row, *([""] * (column_count - len(row)))]
            for row in table
        ]

        header = [
            cell or f"列{index + 1}"
            for index, cell in enumerate(padded_rows[0])
        ]
        separator = ["---"] * column_count
        body = padded_rows[1:]

        rows = [header, separator, *body]
        return "\n".join(
            "| " + " | ".join(row) + " |"
            for row in rows
        )

    def _detect_repeated_margin_lines(self, page_lines: list[list[str]]) -> set[str]:
        """Find likely headers/footers by counting repeated top/bottom lines."""
        if len(page_lines) < 2:
            return set()

        candidates: list[str] = []
        for lines in page_lines:
            margin_lines = [*lines[:3], *lines[-3:]]
            seen_on_page: set[str] = set()
            for line in margin_lines:
                normalized = self._normalize_for_repetition(line)
                if self._can_be_repeated_noise(normalized):
                    seen_on_page.add(normalized)
            candidates.extend(seen_on_page)

        min_count = max(2, round(len(page_lines) * 0.5))
        return {
            line
            for line, count in Counter(candidates).items()
            if count >= min_count
        }

    def _normalize_line(self, line: str) -> str:
        """Normalize whitespace inside a single extracted line."""
        return re.sub(r"\s+", " ", line).strip()

    def _normalize_for_repetition(self, line: str) -> str:
        """Normalize dynamic values so repeated headers with page numbers still match."""
        line = self._normalize_line(line).lower()
        line = re.sub(r"\d+", "#", line)
        return line

    def _can_be_repeated_noise(self, normalized_line: str) -> bool:
        """Return whether a line is plausible repeated header/footer noise."""
        if len(normalized_line) < 4:
            return False
        if len(normalized_line) > 80:
            return False
        return True

    def _is_page_number_line(self, line: str) -> bool:
        """Return whether a line only contains page numbering."""
        return any(pattern.match(line) for pattern in self.PAGE_NUMBER_PATTERNS)

    def _is_disclaimer_line(self, line: str) -> bool:
        """Return whether a line is a common low-value report disclaimer."""
        return any(pattern.search(line) for pattern in self.DISCLAIMER_PATTERNS)

    def _is_section_heading(self, line: str) -> bool:
        """Return whether a line looks like a report section heading."""
        if not 2 <= len(line) <= 60:
            return False
        if self._is_page_number_line(line) or self._is_disclaimer_line(line):
            return False
        if line in self.SECTION_KEYWORDS:
            return True
        return any(pattern.match(line) for pattern in self.SECTION_HEADING_PATTERNS)

    def process_pdf(
        self,
        file_path: Path,
        file_id: str,
        preserve_page_metadata: bool = True,
    ) -> list[DocumentChunk]:
        """Process a PDF file: parse and chunk in one step.

        This is a convenience method that combines parse_pdf and chunk_document.

        Args:
            file_path: Path to the PDF file.
            file_id: Unique identifier for the document.
            preserve_page_metadata: Whether to preserve page information in chunks.

        Returns:
            List of DocumentChunk objects ready for embedding.
        """
        document = self.parse_pdf(file_path, file_id)
        return self.chunk_document(document, preserve_page_metadata)
