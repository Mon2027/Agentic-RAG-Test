"""RAG Analyst Sub-Agent tools.

This module provides tools for the RAG Analyst sub-agent to search
and retrieve information from uploaded research reports.
"""

import logging
from typing import Annotated

from langchain_core.tools import tool

from app.core import get_settings
from app.rag import get_retriever, get_vector_store

logger = logging.getLogger(__name__)


def _format_source_page(source: dict) -> str:
    """Format page or page range for source summary."""
    page_start = source.get("page_start") or source.get("page_number")
    page_end = source.get("page_end") or page_start
    content_type = "，表格" if source.get("content_type") == "table" else ""
    if not page_start:
        return f"（{content_type.lstrip('，')}）" if content_type else ""
    if page_end and page_end != page_start:
        return f"第{page_start}-{page_end}页{content_type}"
    return f"第{page_start}页{content_type}"


def _format_source_citation(source: dict) -> str:
    """Format one source as a citation the LLM can copy directly."""
    page_text = _format_source_page(source)
    section_text = f"，章节：{source['section_title']}" if source.get("section_title") else ""
    if page_text:
        return f"- 研报《{source['file_name']}》{page_text}{section_text}"
    return f"- 研报《{source['file_name']}》{section_text or '，页码未知'}"


@tool
def search_reports(
    query: Annotated[str, "The search query to find relevant content in reports"],
    top_k: Annotated[int, "Number of top results to return"] = 5,
    file_id: Annotated[str | None, "Optional file ID to search within a specific report"] = None,
    topic: Annotated[
        str | None,
        "Optional corpus topic: embodied_intelligence for 具身智能 or low_altitude for 低空经济",
    ] = None,
) -> str:
    """Search through uploaded research reports for relevant information.

    This tool uses semantic search to find the most relevant content
    from all uploaded PDF reports in the vector database.

    Returns the most relevant text chunks along with their sources.
    Use this when the user asks about content in the research reports.
    """
    try:
        retriever = get_retriever()

        # Perform search
        if file_id:
            context = retriever.retrieve_for_file(
                query=query,
                file_id=file_id,
                top_k=top_k,
                topic=topic,
            )
        else:
            context = retriever.retrieve(
                query=query,
                top_k=top_k,
                topic=topic,
            )

        if not context.results:
            return f"""搜索结果: 未找到与 "{query}" 相关的内容。

可能的原因:
1. 研报尚未上传或处理完成
2. 查询内容不在已上传的研报中
3. 需要调整搜索关键词

建议: 请尝试更具体的公司名、指标或报告名称；如用户明确需要实时/外部信息，再由主控 Agent 决定是否使用外部搜索。"""

        # Format results
        formatted_context = context.format_context(include_metadata=True)

        # Build source summary
        sources_str = "\n".join(_format_source_citation(s) for s in context.sources)

        return f"""搜索结果 (找到 {len(context.results)} 个相关片段):

{formatted_context}

---
信息来源:
{sources_str}

证据使用约束:
- 日期、年份只能支持同一句或同一项目中明确关联的事实。
- 不得把收购、并表、报告发布日期、财务年度或相邻事件的年份转用于送样、交付、量产等其他事件。
- 如果片段只说明“正在送样/交付/量产”而未披露开始年份，回答必须明确“研报未披露该事件开始年份”。"""

    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"搜索出错: {str(e)}"


@tool
def get_report_summary(
    file_id: Annotated[str, "The unique identifier of the report file"],
) -> str:
    """Get a summary of a specific research report.

    Returns information about the report including its chunk count,
    page coverage, and availability for querying.
    """
    try:
        retriever = get_retriever()
        summary = retriever.get_document_summary(file_id)

        if not summary.get("found"):
            return f"未找到 ID 为 {file_id} 的研报。请确认研报已上传并处理完成。"

        return f"""研报摘要: {summary['file_name']}

文件 ID: {summary['file_id']}
文档片段数: {summary['chunk_count']}
总字符数: {summary['total_characters']:,}
页数覆盖: {summary['page_count']} 页 ({min(summary['pages']) if summary['pages'] else 0} - {max(summary['pages']) if summary['pages'] else 0})

此研报已处理完成，可以使用 search_reports 工具检索其中的内容。"""

    except Exception as e:
        logger.error(f"Get summary error: {e}")
        return f"获取研报摘要出错: {str(e)}"


@tool
def list_available_reports() -> str:
    """List all uploaded research reports available for querying.

    Returns a list of all reports with their IDs, names, and processing status.
    Use this to see what reports are available before searching.
    """
    try:
        settings = get_settings()
        vector_store = get_vector_store()

        # Get files from vector store
        indexed_files = vector_store.list_files()

        # Get files from filesystem
        fs_reports = list(settings.reports_path.glob("*.pdf"))

        if not indexed_files and not fs_reports:
            return """当前没有已上传的研报。

上传研报:
1. 使用 /api/upload/report 接口上传 PDF 文件
2. 系统将自动解析、分块并向量化
3. 处理完成后即可检索查询"""

        # Build report list
        report_lines: list[str] = []
        indexed_ids = {f["file_id"] for f in indexed_files}

        # Add indexed reports
        for f in indexed_files:
            status = "✅ 已索引" if f["chunk_count"] > 0 else "⏳ 处理中"
            report_lines.append(
                f"- ID: {f['file_id']}\n"
                f"  名称: {f['file_name']}\n"
                f"  状态: {status}\n"
                f"  片段数: {f['chunk_count']}"
            )

        # Add pending reports (in filesystem but not indexed)
        for report_path in fs_reports:
            parts = report_path.name.split("_", 1)
            file_id = parts[0]
            if file_id not in indexed_ids:
                original_name = parts[1] if len(parts) > 1 else report_path.name
                report_lines.append(
                    f"- ID: {file_id}\n"
                    f"  名称: {original_name}\n"
                    f"  状态: ⏳ 待处理"
                )

        return f"""已上传研报列表 (共 {len(report_lines)} 个):

{chr(10).join(report_lines)}

使用文件 ID 配合 search_reports 或 get_report_summary 工具进行查询。"""

    except Exception as e:
        logger.error(f"List reports error: {e}")
        return f"获取研报列表出错: {str(e)}"


@tool
def check_rag_relevance(
    query: Annotated[str, "The query to check relevance for"],
    topic: Annotated[
        str | None,
        "Optional corpus topic: embodied_intelligence for 具身智能 or low_altitude for 低空经济",
    ] = None,
) -> str:
    """Check if the RAG system has relevant information for a query.

    Use this to determine whether to search the reports or fallback to web search.
    Returns relevance score and recommendation.
    """
    try:
        settings = get_settings()
        retriever = get_retriever()
        is_relevant, best_score = retriever.check_relevance(
            query,
            min_relevant_score=settings.rag_relevance_threshold,
            topic=topic,
        )

        if is_relevant:
            return f"""RAG 相关性检查结果: 相关

查询: "{query}"
最佳匹配分数: {best_score:.3f}
建议: 建议使用 search_reports 工具检索研报内容。"""
        else:
            return f"""RAG 相关性检查结果: 不相关

查询: "{query}"
最佳匹配分数: {best_score:.3f}
建议: 本地研报中可能没有足够证据；请先尝试更具体的关键词或报告名称，只有用户明确需要实时/外部信息时再补充外部搜索。"""

    except Exception as e:
        logger.error(f"Relevance check error: {e}")
        return f"相关性检查出错: {str(e)}\n建议: 请尝试直接使用 search_reports 检索，或让用户提供更明确的报告/公司/指标。"


def create_rag_analyst_tools():
    """Create all RAG analyst tools."""
    return [
        search_reports,
        get_report_summary,
        list_available_reports,
        check_rag_relevance,
    ]
