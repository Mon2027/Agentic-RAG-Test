"""Web search tool using Tavily API.

This module provides a web search tool for the main agent to use
as a fallback when RAG retrieval doesn't find relevant information.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Annotated, Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core import get_settings

logger = logging.getLogger(__name__)


class TavilySearchResult(BaseModel):
    """A single search result from Tavily."""

    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    content: str = Field(description="Content/snippet from the search result")
    score: float = Field(description="Relevance score")
    published_date: str | None = Field(
        default=None,
        description="Publication date returned by the search provider, if available",
    )


class TavilySearchResponse(BaseModel):
    """Response from Tavily search API."""

    query: str = Field(description="The search query")
    results: list[TavilySearchResult] = Field(description="List of search results")
    answer: str | None = Field(default=None, description="AI-generated answer if available")


def _tavily_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Execute a search using Tavily API.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.

    Returns:
        Raw API response as dictionary.
    """
    import os

    import httpx

    settings = get_settings()
    api_key = os.environ.get("TAVILY_API_KEY") or settings.tavily_api_key

    if not api_key:
        raise ValueError("TAVILY_API_KEY not configured. Please set it in environment or settings.")

    url = "https://api.tavily.com/search"

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",  # Use advanced for better results
        "include_answer": True,  # Include AI-generated answer
        "include_raw_content": False,  # Don't include raw HTML
        "include_images": False,
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


_ABSOLUTE_DATE = re.compile(
    r"(?<!\d)(?:19|20)\d{2}(?:[-/.]\d{1,2}[-/.]\d{1,2}|年\d{1,2}月\d{1,2}日)"
)
_TARGET_DATE = re.compile(
    r"(?:截至|截止|在)?\s*((?:19|20)\d{2})[年/-](\d{1,2})[月/-](\d{1,2})日?"
)
_FINANCE_MARKERS = ("股价", "股票", "收盘", "开盘", "最高价", "最低价", "市值")
_NEWS_MARKERS = ("新闻", "消息", "动态", "今天", "今日", "最新")


def _target_date(query: str) -> str | None:
    """Return an explicit query cut-off date in ISO form when present."""
    match = _TARGET_DATE.search(query)
    if match is None:
        return None
    year, month, day = (int(value) for value in match.groups())
    try:
        return datetime(year, month, day).date().isoformat()
    except ValueError:
        return None


def _prepare_search_query(query: str) -> str:
    """Add evidence-oriented terms without changing the requested cut-off date."""
    additions: list[str] = []
    if any(marker in query for marker in _FINANCE_MARKERS):
        additions.append("历史行情 收盘价 交易日期（使用截止日或之前最近交易日）")
    if any(marker in query for marker in _NEWS_MARKERS):
        additions.append("发布日期 事件发生日期")
    return " ".join([query.strip(), *additions]).strip()


def _has_historical_price_table(response: dict[str, Any]) -> bool:
    """Return whether Tavily supplied a dated close-price table row."""
    row_pattern = re.compile(
        r"\|\s*(?:19|20)\d{2}年\d{1,2}月\d{1,2}日\s*\|\s*\d+(?:\.\d+)?\s*\|"
    )
    return any(
        "日期" in str(result.get("content") or "")
        and "收盘" in str(result.get("content") or "")
        and row_pattern.search(str(result.get("content") or "")) is not None
        for result in response.get("results", [])
    )


def _finance_retry_query(query: str) -> str:
    """Build a narrow second query for a dated historical quote table."""
    target = _target_date(query)
    date_term = ""
    if target:
        target_day = datetime.strptime(target, "%Y-%m-%d").date()
        while target_day.weekday() >= 5:
            target_day -= timedelta(days=1)
        date_term = f" {target_day.year}年{target_day.month}月{target_day.day}日"
    return (
        f"{query.strip()}{date_term} 股票历史数据表 日期 收盘 开盘 高 低 交易量 "
        "historical data"
    )


def _merge_search_responses(
    primary: dict[str, Any],
    retry: dict[str, Any],
) -> dict[str, Any]:
    """Prefer retry evidence while de-duplicating URLs from both searches."""
    results = []
    seen_urls: set[str] = set()
    for result in [*retry.get("results", []), *primary.get("results", [])]:
        url = str(result.get("url") or "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        results.append(result)
    return {
        **primary,
        "results": results,
        "answer": retry.get("answer") or primary.get("answer"),
    }


def _search_with_finance_retry(query: str, max_results: int) -> dict[str, Any]:
    """Search once, then retry narrowly when a dated price table is absent."""
    effective_query = _prepare_search_query(query)
    response = _tavily_search(effective_query, max_results)
    if (
        any(marker in query for marker in _FINANCE_MARKERS)
        and _target_date(query)
        and not _has_historical_price_table(response)
    ):
        retry = _tavily_search(_finance_retry_query(query), max_results)
        response = _merge_search_responses(response, retry)
    return response


def _result_date_candidates(result: dict[str, Any]) -> list[str]:
    """Extract absolute dates from one result as candidates, not verified metadata."""
    text = "\n".join(
        str(result.get(field) or "") for field in ("title", "content")
    )
    return list(dict.fromkeys(_ABSOLUTE_DATE.findall(text)))[:5]


def _format_web_evidence(
    *,
    query: str,
    response: dict[str, Any],
    max_results: int | None = None,
) -> str:
    """Format Web output as traceable evidence instead of an answer-only blob."""
    results = list(response.get("results", []))
    if max_results is not None:
        results = results[:max_results]
    answer = response.get("answer")

    if not results and not answer:
        return f'网络搜索结果: 未找到与 "{query}" 相关的信息。'

    target = _target_date(query)
    output_parts = [
        "【Web 证据】",
        f"原始查询：{query}",
        f"查询执行日期：{datetime.now().astimezone().date().isoformat()}",
        f"用户截止日：{target or '未指定'}",
    ]

    if results:
        output_parts.append(f"【来源记录】（共 {len(results)} 条）")
        for index, result in enumerate(results, 1):
            published_date = (
                result.get("published_date")
                or result.get("published_at")
                or result.get("date")
                or "搜索服务未提供"
            )
            candidates = _result_date_candidates(result)
            output_parts.extend([
                f"[来源 {index}]",
                f"标题：{result.get('title', '无标题')}",
                f"URL：{result.get('url', '')}",
                f"发布日期：{published_date}",
                f"内容日期候选：{', '.join(candidates) if candidates else '未发现'}",
                f"内容：{result.get('content', '无内容')}",
                f"相关度：{float(result.get('score') or 0):.2f}",
            ])

    if answer:
        output_parts.extend([
            "【搜索服务 AI 摘要（未核验）】",
            str(answer),
        ])

    output_parts.extend([
        "---",
        "证据使用规则：AI摘要不能作为精确日期、新闻时效或行情数字的唯一依据；"
        "必须引用上面的直接URL。发布日期与事件发生日要分开说明，缺失时明确写未核实。",
        "若问题指定截止日，不得采用截止日之后的信息，也不得照抄网页标题中的“今天”"
        "或“倒计时N天”；只能根据绝对日期重新表述。",
        "行情问题必须使用截止日或之前最近交易日的收盘数据，明确交易日期；"
        "不得把盘中价、当前快照或前一交易日数据冒充目标日收盘价。",
    ])
    return "\n".join(output_parts)


@tool
def web_search(
    query: Annotated[str, "The search query to look up on the web"],
    max_results: Annotated[int, "Maximum number of results to return"] = 5,
) -> str:
    """Search the web for information using Tavily search engine.

    Use this tool when:
    1. The RAG system doesn't have relevant information
    2. The user asks about current events or real-time information
    3. You need to verify or supplement information from reports

    Returns search results with titles, URLs, and content snippets.
    """
    try:
        logger.info(f"Performing web search: {query}")

        response = _search_with_finance_retry(query, max_results)

        if not response.get("results") and not response.get("answer"):
            return f"""网络搜索结果: 未找到与 "{query}" 相关的信息。

建议:
1. 尝试调整搜索关键词
2. 确认查询内容是否正确"""

        return _format_web_evidence(query=query, response=response)

    except ValueError as e:
        logger.error(f"Web search configuration error: {e}")
        return f"网络搜索配置错误: {str(e)}\n请联系管理员配置 TAVILY_API_KEY。"

    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"网络搜索出错: {str(e)}\n请稍后重试或尝试其他查询方式。"


@tool
def web_search_quick(
    query: Annotated[str, "The search query"],
) -> str:
    """Quick Web search that still retains URLs and date metadata.

    Use this for simple fact-checking or quick lookups.
    """
    try:
        response = _search_with_finance_retry(query, max_results=3)
        return _format_web_evidence(query=query, response=response, max_results=3)

    except Exception as e:
        logger.error(f"Quick search error: {e}")
        return f"搜索出错: {str(e)}"


def create_web_search_tool() -> list:
    """Create web search tools for the agent.

    Returns:
        List of web search tools.
    """
    return [web_search, web_search_quick]
