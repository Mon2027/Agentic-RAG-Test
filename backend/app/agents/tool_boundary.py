"""Business-tool boundary for DeepAgents-based application agents.

DeepAgents keeps its filesystem middleware even when the corresponding tools
are hidden from the model.  That middleware also injects a filesystem prompt,
which can cause models to invent calls to hidden tools.  This module removes
those stale instructions, filters the model-facing tool schemas, and rejects
disabled calls before they can reach a backend.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import date, timedelta
from typing import Any
from uuid import uuid4

from deepagents.middleware.filesystem import (
    EXECUTION_SYSTEM_PROMPT,
    FILESYSTEM_SYSTEM_PROMPT,
)
from langchain.agents.middleware.todo import WRITE_TODOS_SYSTEM_PROMPT
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

BUSINESS_TOOL_BOUNDARY_PROMPT = """## 业务工具边界

只允许调用“当前可用工具”中列出的工具；不得根据历史提示、示例或常识构造未提供的工具调用。
主 Agent 应通过 task 将研报问题交给 rag-analyst、将 CSV/Excel 问题交给 data-analyst；业务子 Agent 只能使用当前列出的业务工具。
如果当前没有适用工具，应直接说明限制或回答问题，不要尝试访问文件系统或执行命令。"""

MAX_IDENTICAL_TOOL_FAILURES = 2
MAIN_AGENT_TOOL_CALL_BUDGET = 8
DATA_AGENT_TOOL_CALL_BUDGET = 8
RAG_AGENT_TOOL_CALL_BUDGET = 12
MAX_RAG_SEARCH_CALLS = 4
MAX_RAG_RETRIEVAL_CALLS = 5
RAG_RETRIEVAL_TOOLS = frozenset({
    "check_rag_relevance",
    "get_report_summary",
    "list_available_reports",
    "search_reports",
})
WEB_TOOLS = frozenset({"web_search", "web_search_quick"})

LOCAL_RAG_WEB_BOUNDARY_PROMPT = """## 本地研报来源边界

当前用户没有要求联网或实时信息，且问题属于公司业绩、财务指标或本地研报分析范围。
不得调用 Web 工具；必须优先通过 task 委派给 rag-analyst，并仅根据本地研报结果回答。"""

RAG_SEARCH_LIMIT_PROMPT = """## RAG 检索已收口

当前问题已经达到有限检索上限。不得继续调用任何研报检索工具；请立即使用已有工具结果回答用户。
如果已有结果没有用户要求的精确事实或数字，必须明确说明“已上传研报未提供该信息”，不得推断、编造或改用联网搜索。"""

TOOL_CIRCUIT_BREAKER_MESSAGE = (
    "工具调用已由安全熔断器停止：{reason}。"
    "请检查工具参数或缩小任务范围后重试；本次不会继续调用模型或工具。"
)

_SOURCE_ONLY_ANSWER = re.compile(
    r"^(?:📊|📌|🌐)?\s*来源\s*[:：].*$",
    flags=re.DOTALL,
)
_CHART_URL = re.compile(r"/static/charts/chart_[\w-]+\.png")
_RAG_PAGE_CITATION = re.compile(
    r"📌\s*来源\s*[:：].*第\d+(?:-\d+)?页",
    flags=re.DOTALL,
)
_YEAR_TEXT = r"(?:19|20)\d{2}年(?:\d{1,2}月)?"
_SAMPLING_TIME_PATTERNS = (
    re.compile(
        rf"(?P<prefix>送样(?:开始)?(?:时间|年份)\*{{0,2}}\s*[:：]\s*)"
        rf"(?P<claim>[^。\r\n]*?(?P<year>{_YEAR_TEXT})[^。\r\n]*)"
    ),
    re.compile(
        rf"(?P<prefix>送样过程(?:发生|开始|进行|已开始)?(?:在|于)?)"
        rf"(?P<claim>\s*(?P<year>{_YEAR_TEXT})[^。\r\n]*)"
    ),
    re.compile(
        rf"(?P<prefix>)(?P<claim>(?P<year>{_YEAR_TEXT})(?:期间)?"
        rf"(?:开始|进行|已开始|已进行)送样)"
    ),
)
_REPORT_CHUNK = re.compile(
    r"--- 文档片段 \d+ \[来源: (?P<file_name>.*?), "
    r"第(?P<pages>\d+(?:-\d+)?)页(?:,.*?)?\] ---\s*"
    r"(?P<body>.*?)"
    r"(?=\n--- 文档片段 \d+ \[来源:|\n---\s*\n信息来源:|\Z)",
    flags=re.DOTALL,
)
_SPECIFIC_YEAR_MONTH = re.compile(r"(?:19|20)\d{2}年\d{1,2}月")
_TEXTUAL_TOOL_CALL = re.compile(
    r"<tool_call>\s*(?P<name>[A-Za-z_][\w-]*)\s*(?P<body>.*?)</tool_call>\s*$",
    flags=re.DOTALL,
)
_TEXTUAL_TOOL_ARGUMENT = re.compile(
    r"<arg_key>\s*(?P<key>.*?)\s*</arg_key>\s*"
    r"<arg_value>\s*(?P<value>.*?)\s*</arg_value>",
    flags=re.DOTALL,
)
_WEB_SOURCE_BLOCK = re.compile(
    r"\[来源 \d+\]\s*\n"
    r"标题：(?P<title>[^\n]*)\n"
    r"URL：(?P<url>https?://[^\s]+)\n"
    r"发布日期：(?P<published>[^\n]*)\n"
    r"内容日期候选：(?P<dates>[^\n]*)",
)
_EXPLICIT_DAY = re.compile(
    r"((?:19|20)\d{2})[年/-](\d{1,2})[月/-](\d{1,2})日?"
)
_HISTORICAL_PRICE_ROW = re.compile(
    r"\|\s*(?P<year>(?:19|20)\d{2})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日\s*"
    r"\|\s*(?P<close>\d+(?:\.\d+)?)\s*\|"
)


def _tool_name(tool: Any) -> str | None:
    """Return a tool name from LangChain tools or provider-style schemas."""
    if isinstance(tool, dict):
        name = tool.get("name")
        if isinstance(name, str):
            return name

        function = tool.get("function")
        if isinstance(function, dict):
            function_name = function.get("name")
            if isinstance(function_name, str):
                return function_name
        return None

    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


def _canonical_tool_arguments(arguments: Any) -> str:
    """Return a deterministic representation for one tool-call argument set."""
    try:
        return json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        return repr(arguments)


def _tool_history(
    messages: Sequence[AnyMessage],
) -> tuple[int, Counter[tuple[str, str, str]]]:
    """Count tool calls and identical error signatures in one Agent state."""
    calls_by_id: dict[str, tuple[str, str]] = {}
    total_calls = 0
    failed_signatures: Counter[tuple[str, str, str]] = Counter()

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                total_calls += 1
                tool_call_id = tool_call.get("id")
                tool_call_name = tool_call.get("name")
                if isinstance(tool_call_id, str) and isinstance(tool_call_name, str):
                    calls_by_id[tool_call_id] = (
                        tool_call_name,
                        _canonical_tool_arguments(tool_call.get("args", {})),
                    )
            continue

        if not isinstance(message, ToolMessage) or message.status != "error":
            continue

        call_signature = calls_by_id.get(message.tool_call_id)
        if call_signature is None:
            continue
        normalized_error = " ".join(message.text.split())
        failed_signatures[(*call_signature, normalized_error)] += 1

    return total_calls, failed_signatures


def _tool_call_counts(messages: Sequence[AnyMessage]) -> Counter[str]:
    """Count model-requested tools by name in one Agent state."""
    counts: Counter[str] = Counter()
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.tool_calls:
            tool_name = tool_call.get("name")
            if isinstance(tool_name, str):
                counts[tool_name] += 1
    return counts


def _rag_retrieval_limit_reached(messages: Sequence[AnyMessage]) -> bool:
    """Return whether the RAG Agent must stop retrieving and synthesize."""
    counts = _tool_call_counts(messages)
    retrieval_calls = sum(counts[tool_name] for tool_name in RAG_RETRIEVAL_TOOLS)
    return (
        counts["search_reports"] >= MAX_RAG_SEARCH_CALLS
        or retrieval_calls >= MAX_RAG_RETRIEVAL_CALLS
    )


def _state_messages(state: Any) -> Sequence[AnyMessage]:
    """Return Agent messages from a ToolCallRequest state shape."""
    if isinstance(state, Sequence) and not isinstance(state, (str, bytes)):
        return state
    if isinstance(state, dict):
        messages = state.get("messages")
    else:
        messages = getattr(state, "messages", None)
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        return messages
    return ()


def _latest_human_query(messages: Sequence[AnyMessage]) -> str:
    """Return the most recent user-authored query in one Agent state."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.text.strip()
    return ""


def _requires_local_rag_before_web(messages: Sequence[AnyMessage]) -> bool:
    """Return whether Web must be hidden for a local-report-first question."""
    query = _latest_human_query(messages).lower()
    if not query:
        return False

    explicit_web_prohibitions = (
        "不要联网",
        "不得联网",
        "禁止联网",
        "只使用已上传研报",
        "仅使用已上传研报",
        "只根据已上传研报",
        "仅根据已上传研报",
        "只使用本地研报",
        "仅使用本地研报",
    )
    if any(marker in query for marker in explicit_web_prohibitions):
        return True

    explicit_web_requests = (
        "请联网",
        "联网查询",
        "联网搜索",
        "联网核实",
        "互联网",
        "网上",
        "实时",
        "今天",
        "今日",
        "天气",
        "最新股价",
        "当前股价",
        "最新消息",
        "最新新闻",
    )
    if any(marker in query for marker in explicit_web_requests):
        return False

    local_report_markers = (
        "已上传研报",
        "上传的研报",
        "本地研报",
        "研报中",
        "报告中",
        "文档中",
    )
    if any(marker in query for marker in local_report_markers):
        return True

    direct_explanation_markers = (
        "什么是",
        "解释",
        "区别",
        "含义",
        "什么意思",
        "怎么理解",
    )
    if any(marker in query for marker in direct_explanation_markers):
        return False

    report_analysis_markers = (
        "营业收入",
        "营收",
        "归母净利润",
        "扣非净利润",
        "净利润",
        "同比",
        "环比",
        "毛利率",
        "净利率",
        "业绩",
        "估值",
        "盈利预测",
        "目标价",
        "市盈率",
        "市净率",
        "业务布局",
        "风险因素",
        "财务指标",
    )
    return any(marker in query for marker in report_analysis_markers)


def _infer_rag_topic(arguments: Any) -> str | None:
    """Infer a strong corpus topic from one RAG query when the model omitted it."""
    if not isinstance(arguments, dict):
        return None
    explicit_topic = arguments.get("topic")
    if isinstance(explicit_topic, str) and explicit_topic.strip():
        return explicit_topic

    query = str(arguments.get("query") or "").lower()
    embodied_markers = (
        "具身智能",
        "机器人",
        "robot",
        "robotics",
        "人形",
        "灵巧手",
        "执行器",
        "关节",
        "丝杠",
        "减速器",
        "avant",
        "凌云光",
    )
    if any(marker in query for marker in embodied_markers):
        return "embodied_intelligence"

    low_altitude_markers = (
        "低空经济",
        "低空飞行",
        "无人机",
        "evtol",
        "飞行汽车",
        "通航",
    )
    if any(marker in query for marker in low_altitude_markers):
        return "low_altitude"
    return None


def _with_inferred_rag_topic(request: ToolCallRequest) -> ToolCallRequest:
    """Add an inferred topic to RAG tool arguments without overriding explicit input."""
    if request.tool_call["name"] not in {"check_rag_relevance", "search_reports"}:
        return request
    arguments = request.tool_call.get("args")
    inferred_topic = _infer_rag_topic(arguments)
    if inferred_topic is None or not isinstance(arguments, dict):
        return request
    if arguments.get("topic") == inferred_topic:
        return request

    return request.override(tool_call={
        **request.tool_call,
        "args": {**arguments, "topic": inferred_topic},
    })


def _with_web_cutoff_date(request: ToolCallRequest) -> ToolCallRequest:
    """Carry an explicit user cutoff date into a model-authored Web query."""
    if request.tool_call["name"] not in WEB_TOOLS:
        return request
    arguments = request.tool_call.get("args")
    if not isinstance(arguments, dict):
        return request
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return request

    human_query = _latest_human_query(_state_messages(request.state))
    match = _EXPLICIT_DAY.search(human_query)
    if match is None:
        return request
    year, month, day = (int(value) for value in match.groups())
    cutoff_text = f"{year}年{month}月{day}日"
    if cutoff_text in query or f"{year:04d}-{month:02d}-{day:02d}" in query:
        return request

    updated_tool_call = {
        **request.tool_call,
        "args": {**arguments, "query": f"{query.strip()} 截至{cutoff_text}"},
    }
    override = getattr(request, "override", None)
    if callable(override):
        return override(tool_call=updated_tool_call)
    request_copy = copy.copy(request)
    request_copy.tool_call = updated_tool_call
    return request_copy


def _decode_textual_tool_argument(value: str) -> Any:
    """Decode JSON-compatible textual arguments while preserving plain strings."""
    normalized = value.strip()
    try:
        return json.loads(normalized)
    except (json.JSONDecodeError, TypeError):
        return normalized


def _parse_textual_tool_call(
    text: str,
    visible_tool_names: set[str | None],
) -> tuple[str, dict[str, Any], str] | None:
    """Parse one complete provider-style textual call to a declared tool."""
    matches = list(_TEXTUAL_TOOL_CALL.finditer(text))
    if len(matches) != 1:
        return None
    match = matches[0]
    tool_name = match.group("name")
    if tool_name not in visible_tool_names:
        return None

    body = match.group("body")
    argument_matches = list(_TEXTUAL_TOOL_ARGUMENT.finditer(body))
    if argument_matches and _TEXTUAL_TOOL_ARGUMENT.sub("", body).strip():
        return None
    if not argument_matches and body.strip():
        return None

    arguments: dict[str, Any] = {}
    for argument_match in argument_matches:
        key = argument_match.group("key").strip()
        if not key or key in arguments:
            return None
        arguments[key] = _decode_textual_tool_argument(
            argument_match.group("value")
        )

    prefix = text[:match.start()].strip()
    return tool_name, arguments, prefix


def _normalize_textual_tool_call(
    request: ModelRequest,
    response: Any,
) -> Any:
    """Convert a compatible model's textual tool markup into a real tool call."""
    if not isinstance(response, ModelResponse):
        return response
    visible_tool_names = {_tool_name(tool) for tool in request.tools}
    result_messages = list(response.result)

    for index in range(len(result_messages) - 1, -1, -1):
        message = result_messages[index]
        if not isinstance(message, AIMessage):
            continue
        if message.tool_calls:
            return response

        parsed = _parse_textual_tool_call(message.text, visible_tool_names)
        if parsed is None:
            return response
        tool_name, arguments, prefix = parsed
        result_messages[index] = message.model_copy(update={
            "content": prefix,
            "tool_calls": [{
                "name": tool_name,
                "args": arguments,
                "id": f"textual_tool_call_{uuid4().hex}",
                "type": "tool_call",
            }],
        })
        return ModelResponse(
            result=result_messages,
            structured_response=response.structured_response,
        )

    return response


def _restore_subagent_result_if_source_only(
    request: ModelRequest,
    response: Any,
) -> Any:
    """Preserve authoritative task content dropped or distorted by the main Agent.

    Some compatible models correctly receive the full ``task`` ToolMessage but
    answer with only a source label. This narrow fallback applies only to the
    main Agent (identified by the visible ``task`` tool), only after a successful
    task result. Empty/source-only answers are restored. For a single chart
    task, the data Agent's complete result is authoritative so model-authored
    restatements cannot lose its URL or alter labels and values.
    """
    visible_tool_names = {_tool_name(tool) for tool in request.tools}
    if "task" not in visible_tool_names or not isinstance(response, ModelResponse):
        return response

    task_results = [
        message.text.strip()
        for message in request.messages
        if isinstance(message, ToolMessage)
        and message.name == "task"
        and message.status != "error"
        and message.text.strip()
    ]
    task_result = next(
        (
            result
            for result in reversed(task_results)
        ),
        None,
    )
    if task_result is None:
        return response

    result_messages = list(response.result)
    for index in range(len(result_messages) - 1, -1, -1):
        message = result_messages[index]
        if not isinstance(message, AIMessage):
            continue
        if message.tool_calls:
            return response

        answer_text = message.text.strip()
        chart_urls = list(dict.fromkeys(
            url
            for result in task_results
            for url in _CHART_URL.findall(result)
        ))
        if len(task_results) == 1 and chart_urls:
            combined_content = task_result
            source_line = next(
                (
                    line.strip()
                    for line in reversed(answer_text.splitlines())
                    if _SOURCE_ONLY_ANSWER.fullmatch(line.strip()) is not None
                ),
                None,
            )
            if source_line and source_line not in combined_content:
                combined_content = f"{combined_content}\n\n{source_line}"
            result_messages[index] = message.model_copy(
                update={"content": combined_content}
            )
            return ModelResponse(
                result=result_messages,
                structured_response=response.structured_response,
            )

        has_web_result = any(
            isinstance(tool_message, ToolMessage)
            and tool_message.name in WEB_TOOLS
            and tool_message.status != "error"
            and tool_message.text.strip()
            for tool_message in request.messages
        )
        if (
            len(task_results) == 1
            and _RAG_PAGE_CITATION.search(task_result)
            and not has_web_result
        ):
            result_messages[index] = message.model_copy(
                update={"content": task_result}
            )
            return ModelResponse(
                result=result_messages,
                structured_response=response.structured_response,
            )

        if answer_text and _SOURCE_ONLY_ANSWER.fullmatch(answer_text) is None:
            missing_urls = [url for url in chart_urls if url not in answer_text]
            if not missing_urls:
                return response
            combined_content = (
                f"{answer_text}\n\n图表：\n" + "\n".join(missing_urls)
            )
        else:
            combined_content = task_result
            if answer_text:
                combined_content = f"{combined_content}\n\n{answer_text}"
        result_messages[index] = message.model_copy(
            update={"content": combined_content}
        )
        return ModelResponse(
            result=result_messages,
            structured_response=response.structured_response,
        )

    return response


def _ensure_web_source_traceability(
    request: ModelRequest,
    response: Any,
) -> Any:
    """Keep direct Web URLs and available dates in the final main-agent answer."""
    if not isinstance(response, ModelResponse):
        return response
    visible_tool_names = {_tool_name(tool) for tool in request.tools}
    if not WEB_TOOLS & visible_tool_names:
        return response

    sources: list[tuple[str, str, str, str]] = []
    seen_urls: set[str] = set()
    for message in request.messages:
        if (
            not isinstance(message, ToolMessage)
            or message.name not in WEB_TOOLS
            or message.status == "error"
        ):
            continue
        for match in _WEB_SOURCE_BLOCK.finditer(message.text):
            url = match.group("url").rstrip(".,;，。；")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            sources.append((
                match.group("title").strip() or "未命名页面",
                url,
                match.group("published").strip() or "搜索服务未提供",
                match.group("dates").strip() or "未发现",
            ))

    if not sources:
        return response

    result_messages = list(response.result)
    changed = False
    for index, message in enumerate(result_messages):
        if not isinstance(message, AIMessage) or message.tool_calls:
            continue
        answer = message.text.strip()
        missing_sources = [source for source in sources if source[1] not in answer][:3]
        has_direct_url = bool(re.search(r"https?://", answer))
        has_dated_web_source = "🌐 来源" in answer and has_direct_url
        if has_dated_web_source:
            continue
        if not missing_sources and has_direct_url:
            continue

        source_lines = []
        for title, url, published, dates in missing_sources or sources[:3]:
            date_note = f"发布日期：{published}"
            if dates != "未发现":
                date_note += f"；内容日期：{dates}"
            elif published == "搜索服务未提供":
                date_note = "日期未核实"
            source_lines.append(f"- {title}（{date_note}） {url}")
        if not source_lines:
            continue
        content = f"{answer}\n\n🌐 来源：\n" + "\n".join(source_lines)
        result_messages[index] = message.model_copy(update={"content": content})
        changed = True

    if not changed:
        return response
    return ModelResponse(
        result=result_messages,
        structured_response=response.structured_response,
    )


def _explicit_query_date(messages: Sequence[AnyMessage]) -> date | None:
    """Return the latest user's explicit calendar date when it is valid."""
    match = _EXPLICIT_DAY.search(_latest_human_query(messages))
    if match is None:
        return None
    try:
        return date(*(int(value) for value in match.groups()))
    except ValueError:
        return None


def _guard_same_day_web_news(
    request: ModelRequest,
    response: Any,
) -> Any:
    """Do not let undated or older Web results masquerade as today's news."""
    if not isinstance(response, ModelResponse):
        return response
    query = _latest_human_query(request.messages)
    if not any(marker in query for marker in ("今天", "今日")):
        return response
    if not any(marker in query for marker in ("新闻", "消息", "动态", "最新")):
        return response
    target = _explicit_query_date(request.messages)
    if target is None:
        return response

    target_variants = {
        target.isoformat(),
        f"{target.year}年{target.month}月{target.day}日",
        f"{target.year}年{target.month:02d}月{target.day:02d}日",
    }
    published_dates = []
    for message in request.messages:
        if (
            not isinstance(message, ToolMessage)
            or message.name not in WEB_TOOLS
            or message.status == "error"
        ):
            continue
        published_dates.extend(
            match.group("published").strip()
            for match in _WEB_SOURCE_BLOCK.finditer(message.text)
        )
    if any(
        variant in published
        for published in published_dates
        for variant in target_variants
    ):
        return response

    replacement = (
        f"截至{target.year}年{target.month}月{target.day}日，当前 Web 搜索结果"
        "没有提供可核验为该日发布的具身智能行业消息，因此无法可靠列出"
        "“今天”的新动态。检索到的页面发布日期缺失或早于该日，只能作为"
        "历史背景，不能冒充当日新闻。若要继续核验，应查找带明确发布日期"
        "的官方公告或新闻原文。"
    )
    result_messages = list(response.result)
    changed = False
    for index, message in enumerate(result_messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            result_messages[index] = message.model_copy(update={"content": replacement})
            changed = True
    if not changed:
        return response
    return ModelResponse(
        result=result_messages,
        structured_response=response.structured_response,
    )


def _guard_weekend_stock_date(
    request: ModelRequest,
    response: Any,
) -> Any:
    """Correct a weekend cutoff that the model mislabeled as a trading day."""
    if not isinstance(response, ModelResponse):
        return response
    query = _latest_human_query(request.messages)
    if not any(marker in query for marker in ("股价", "股票", "收盘价")):
        return response
    target = _explicit_query_date(request.messages)
    if target is None or target.weekday() < 5:
        return response

    previous_weekday = target - timedelta(days=target.weekday() - 4)
    target_cn = f"{target.year}年{target.month}月{target.day}日"
    previous_cn = (
        f"{previous_weekday.year}年{previous_weekday.month}月{previous_weekday.day}日"
    )
    trading_date_pattern = re.compile(
        rf"(?P<prefix>\*{{0,2}}交易日期\*{{0,2}}\s*[:：]\s*)"
        rf"(?:{re.escape(target_cn)}|{target.isoformat()})"
    )

    result_messages = list(response.result)
    changed = False
    for index, message in enumerate(result_messages):
        if not isinstance(message, AIMessage) or message.tool_calls:
            continue
        content, replacements = trading_date_pattern.subn(
            rf"\g<prefix>{previous_cn}（{target_cn}为非交易日）",
            message.text,
        )
        if previous_cn not in content:
            content += (
                f"\n\n交易日期说明：{target_cn}为非交易日；应采用其之前最近"
                f"工作日{previous_cn}的收盘数据，并由日期化行情页面核实。"
            )
            replacements += 1
        if replacements:
            result_messages[index] = message.model_copy(update={"content": content})
            changed = True
    if not changed:
        return response
    return ModelResponse(
        result=result_messages,
        structured_response=response.structured_response,
    )


def _limit_unrequested_stock_metrics(
    request: ModelRequest,
    response: Any,
) -> Any:
    """Remove extra market metrics when the user asked only for the latest price."""
    if not isinstance(response, ModelResponse):
        return response
    query = _latest_human_query(request.messages)
    if not any(marker in query for marker in ("股价", "收盘价")):
        return response

    optional_metrics = {
        "开盘价": ("开盘",),
        "最高价": ("最高价", "当日最高"),
        "最低价": ("最低价", "当日最低"),
        "涨跌幅": ("涨跌幅", "涨幅", "跌幅"),
        "成交量": ("成交量",),
        "市值": ("市值",),
        "52周": ("52周", "52 周"),
        "目标价": ("目标价",),
        "分析师": ("分析师", "评级为", "看涨潜力"),
    }
    requested_groups = {
        group
        for group, markers in optional_metrics.items()
        if any(marker in query for marker in markers)
    }
    forbidden_markers = tuple(
        marker
        for group, markers in optional_metrics.items()
        if group not in requested_groups
        for marker in markers
    )
    if not forbidden_markers:
        return response

    result_messages = list(response.result)
    changed = False
    for index, message in enumerate(result_messages):
        if not isinstance(message, AIMessage) or message.tool_calls:
            continue
        lines = message.text.splitlines()
        filtered_lines = [
            line for line in lines
            if not any(marker in line for marker in forbidden_markers)
        ]
        if filtered_lines == lines:
            continue
        while filtered_lines and not filtered_lines[-1].strip():
            filtered_lines.pop()
        result_messages[index] = message.model_copy(
            update={"content": "\n".join(filtered_lines)}
        )
        changed = True
    if not changed:
        return response
    return ModelResponse(
        result=result_messages,
        structured_response=response.structured_response,
    )


def _dated_historical_closes(
    messages: Sequence[AnyMessage],
    cutoff: date,
) -> list[tuple[date, str, str, str]]:
    """Extract dated close values only from pipe-formatted historical tables."""
    rows: list[tuple[date, str, str, str]] = []
    for message in messages:
        if (
            not isinstance(message, ToolMessage)
            or message.name not in WEB_TOOLS
            or message.status == "error"
        ):
            continue
        source_matches = list(_WEB_SOURCE_BLOCK.finditer(message.text))
        for source_index, source_match in enumerate(source_matches):
            segment_end = (
                source_matches[source_index + 1].start()
                if source_index + 1 < len(source_matches)
                else len(message.text)
            )
            segment = message.text[source_match.start():segment_end]
            for row_match in _HISTORICAL_PRICE_ROW.finditer(segment):
                try:
                    row_date = date(
                        int(row_match.group("year")),
                        int(row_match.group("month")),
                        int(row_match.group("day")),
                    )
                except ValueError:
                    continue
                if row_date <= cutoff:
                    rows.append((
                        row_date,
                        row_match.group("close"),
                        source_match.group("title").strip() or "股票历史行情",
                        source_match.group("url").rstrip(".,;，。；"),
                    ))
    return rows


def _ground_stock_close_from_history(
    request: ModelRequest,
    response: Any,
) -> Any:
    """Ground a price-only answer in the latest dated historical table row."""
    if not isinstance(response, ModelResponse):
        return response
    query = _latest_human_query(request.messages)
    if not any(marker in query for marker in ("股价", "收盘价")):
        return response
    extra_requests = (
        "开盘", "最高价", "最低价", "涨跌幅", "涨幅", "跌幅",
        "成交量", "市值", "52周", "52 周", "目标价", "分析师",
    )
    if any(marker in query for marker in extra_requests):
        return response
    cutoff = _explicit_query_date(request.messages)
    if cutoff is None:
        return response

    rows = _dated_historical_closes(request.messages, cutoff)
    latest_date = max((row[0] for row in rows), default=None)
    latest_rows = [row for row in rows if row[0] == latest_date]
    unique_closes = {row[1] for row in latest_rows}
    if latest_date is not None and len(unique_closes) == 1:
        row_date, close, title, url = latest_rows[0]
        cutoff_cn = f"{cutoff.year}年{cutoff.month}月{cutoff.day}日"
        row_date_cn = f"{row_date.year}年{row_date.month}月{row_date.day}日"
        date_note = (
            f"{cutoff_cn}没有可用交易数据，采用该日之前最近交易日；"
            if row_date < cutoff
            else ""
        )
        stock_section = (
            f"截至{cutoff_cn}，{date_note}可核验的最近交易日为"
            f"{row_date_cn}，收盘价为 **{close}元**。\n\n"
            f"🌐 来源：{title}（{row_date_cn}历史行情） {url}"
        )
    else:
        cutoff_cn = f"{cutoff.year}年{cutoff.month}月{cutoff.day}日"
        stock_section = (
            f"截至{cutoff_cn}，当前 Web 结果未提供可唯一核验的日期化历史"
            "行情表，或同一交易日的收盘数据存在冲突，因此不输出未经核实"
            "的股价数字。"
        )

    result_messages = list(response.result)
    changed = False
    has_task_result = any(
        isinstance(message, ToolMessage)
        and message.name == "task"
        and message.status != "error"
        for message in request.messages
    )
    for index, message in enumerate(result_messages):
        if not isinstance(message, AIMessage) or message.tool_calls:
            continue
        content = message.text
        if has_task_result:
            lines = content.splitlines()
            heading_index = next(
                (
                    line_index
                    for line_index, line in enumerate(lines)
                    if line.lstrip().startswith("#") and "股价" in line
                ),
                None,
            )
            if heading_index is not None:
                rag_prefix = "\n".join(lines[:heading_index]).rstrip()
                rag_citations = [
                    line.strip() for line in lines if "📌 来源" in line
                ]
                parts = [rag_prefix, lines[heading_index].strip(), stock_section]
                parts.extend(
                    citation for citation in rag_citations if citation not in rag_prefix
                )
                content = "\n\n".join(part for part in parts if part)
            else:
                content = f"{content.rstrip()}\n\n## 最新股价\n\n{stock_section}"
        else:
            content = stock_section
        result_messages[index] = message.model_copy(update={"content": content})
        changed = True
    if not changed:
        return response
    return ModelResponse(
        result=result_messages,
        structured_response=response.structured_response,
    )


def _sampling_year_is_directly_supported(evidence: str, year: str) -> bool:
    """Return whether evidence directly assigns ``year`` to sampling."""
    confounding_events = ("收购", "并购", "并表", "营收", "收入", "发布", "合作", "成立")
    segments = re.split(r"[。！？；;，,\r\n]+", evidence)
    for segment in segments:
        if year not in segment or "送样" not in segment:
            continue
        if any(event in segment for event in confounding_events):
            continue
        if re.search(rf"送样(?:开始)?(?:时间|年份)[^。；\n]{{0,12}}{re.escape(year)}", segment):
            return True
        if re.search(rf"{re.escape(year)}[^。；\n]{{0,12}}送样", segment):
            return True
    return False


def _guard_unsupported_sampling_years(
    request: ModelRequest,
    response: Any,
) -> Any:
    """Remove sampling years that are not directly supported by tool evidence."""
    if not isinstance(response, ModelResponse):
        return response

    evidence = "\n".join(
        message.text
        for message in request.messages
        if isinstance(message, ToolMessage)
        and message.name in {*RAG_RETRIEVAL_TOOLS, "task"}
        and message.status != "error"
        and message.text.strip()
    )
    if not evidence:
        return response

    result_messages = list(response.result)
    changed = False
    replacement = "研报未披露送样开始年份；截至报告披露时，产品正在送样过程中"

    for index, message in enumerate(result_messages):
        if not isinstance(message, AIMessage) or message.tool_calls:
            continue
        content = message.text

        for pattern in _SAMPLING_TIME_PATTERNS:
            def replace_if_unsupported(match: re.Match[str]) -> str:
                nonlocal changed
                year = match.group("year")
                if _sampling_year_is_directly_supported(evidence, year):
                    return match.group(0)
                changed = True
                prefix = match.group("prefix")
                if not prefix or prefix.startswith("送样过程"):
                    return replacement
                return f"{prefix}{replacement}"

            content = pattern.sub(replace_if_unsupported, content)

        sampling_years = dict.fromkeys(re.findall(_YEAR_TEXT, evidence))
        evidence_has_sampling_year = any(
            _sampling_year_is_directly_supported(evidence, year)
            for year in sampling_years
        )
        sampling_year_limit_present = (
            "未披露" in content and "年份" in content
        )
        if (
            "送样" in evidence
            and "送样" in content
            and not evidence_has_sampling_year
            and not sampling_year_limit_present
        ):
            sampling_line = next(
                (
                    line
                    for line in content.splitlines()
                    if "送样" in line and any(
                        state in line for state in ("过程中", "正在", "送样阶段")
                    )
                ),
                None,
            )
            if sampling_line:
                content = content.replace(
                    sampling_line,
                    f"{sampling_line}\n- 研报未披露送样开始年份",
                    1,
                )
                changed = True

        if content != message.text:
            result_messages[index] = message.model_copy(update={"content": content})

    if not changed:
        return response
    return ModelResponse(
        result=result_messages,
        structured_response=response.structured_response,
    )


def _clean_report_name(file_name: str) -> str:
    """Remove storage identifiers and the PDF suffix from a report name."""
    cleaned = re.sub(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_",
        "",
        file_name.strip(),
    )
    return cleaned[:-4] if cleaned.lower().endswith(".pdf") else cleaned


def _fact_overlap_score(fact_line: str, evidence_body: str) -> int:
    """Score event-word overlap so a matching date alone cannot select a page."""
    normalized = _SPECIFIC_YEAR_MONTH.sub("", fact_line)
    normalized = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "", normalized)
    if len(normalized) < 2:
        return 0
    bigrams = {
        normalized[index:index + 2]
        for index in range(len(normalized) - 1)
    }
    return sum(1 for bigram in bigrams if bigram in evidence_body)


def _ensure_rag_specific_date_citations(
    request: ModelRequest,
    response: Any,
) -> Any:
    """Append the evidence page for each cited YYYY年M月 fact when missing."""
    if not isinstance(response, ModelResponse):
        return response

    evidence = "\n".join(
        message.text
        for message in request.messages
        if isinstance(message, ToolMessage)
        and message.name == "search_reports"
        and message.status != "error"
        and message.text.strip()
    )
    chunks = list(_REPORT_CHUNK.finditer(evidence))
    if not chunks:
        return response

    result_messages = list(response.result)
    changed = False
    for index, message in enumerate(result_messages):
        if not isinstance(message, AIMessage) or message.tool_calls:
            continue
        answer_text = message.text
        missing_citations = []
        for date_text in dict.fromkeys(_SPECIFIC_YEAR_MONTH.findall(answer_text)):
            fact_line = next(
                (line for line in answer_text.splitlines() if date_text in line),
                "",
            )
            candidates = [
                (_fact_overlap_score(fact_line, chunk.group("body")), chunk)
                for chunk in chunks
                if date_text in chunk.group("body")
            ]
            if not candidates:
                continue
            overlap_score, supporting_chunk = max(
                candidates,
                key=lambda candidate: candidate[0],
            )
            if overlap_score < 2:
                continue
            pages = supporting_chunk.group("pages")
            if f"第{pages}页" in answer_text:
                continue
            report_name = _clean_report_name(supporting_chunk.group("file_name"))
            citation = f"- 研报《{report_name}》第{pages}页"
            if citation not in missing_citations:
                missing_citations.append(citation)

        if not missing_citations:
            continue
        changed = True
        heading = "" if "📌 来源" in answer_text else "\n\n📌 来源："
        content = f"{answer_text}{heading}\n" + "\n".join(missing_citations)
        result_messages[index] = message.model_copy(update={"content": content})

    if not changed:
        return response
    return ModelResponse(
        result=result_messages,
        structured_response=response.structured_response,
    )


def _remove_injected_tool_prompts(text: str) -> str:
    """Remove tool instructions injected by hidden DeepAgents middleware."""
    cleaned = text
    for injected_prompt in (
        FILESYSTEM_SYSTEM_PROMPT,
        EXECUTION_SYSTEM_PROMPT,
        WRITE_TODOS_SYSTEM_PROMPT,
    ):
        cleaned = cleaned.replace(f"\n\n{injected_prompt}", "")
        cleaned = cleaned.replace(injected_prompt, "")
    return cleaned


def sanitize_system_message(system_message: SystemMessage | None) -> SystemMessage | None:
    """Return a system message without hidden-tool prompt sections."""
    if system_message is None:
        return None

    cleaned_blocks: list[Any] = []
    for block in system_message.content_blocks:
        if isinstance(block, str):
            cleaned_text = _remove_injected_tool_prompts(block)
            if cleaned_text.strip():
                cleaned_blocks.append({"type": "text", "text": cleaned_text})
            continue

        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                cleaned_text = _remove_injected_tool_prompts(text)
                if cleaned_text.strip():
                    cleaned_blocks.append({**block, "text": cleaned_text})
                continue

        cleaned_blocks.append(block)

    return SystemMessage(content_blocks=cleaned_blocks)


def _append_boundary_prompt(
    system_message: SystemMessage | None,
    visible_tool_names: list[str],
) -> SystemMessage:
    """Append one explicit, request-specific tool allow-list."""
    tool_list = "、".join(visible_tool_names) if visible_tool_names else "无"
    boundary_text = f"{BUSINESS_TOOL_BOUNDARY_PROMPT}\n当前可用工具：{tool_list}"

    blocks = list(system_message.content_blocks) if system_message else []
    if blocks:
        boundary_text = f"\n\n{boundary_text}"
    blocks.append({"type": "text", "text": boundary_text})
    return SystemMessage(content_blocks=blocks)


def _append_rag_search_limit_prompt(
    system_message: SystemMessage | None,
) -> SystemMessage:
    """Tell the RAG Agent to synthesize after its finite search budget."""
    blocks = list(system_message.content_blocks) if system_message else []
    prompt = f"\n\n{RAG_SEARCH_LIMIT_PROMPT}" if blocks else RAG_SEARCH_LIMIT_PROMPT
    blocks.append({"type": "text", "text": prompt})
    return SystemMessage(content_blocks=blocks)


def _append_local_rag_web_boundary_prompt(
    system_message: SystemMessage | None,
) -> SystemMessage:
    """Tell the main Agent why Web is unavailable for this user turn."""
    blocks = list(system_message.content_blocks) if system_message else []
    prompt = (
        f"\n\n{LOCAL_RAG_WEB_BOUNDARY_PROMPT}"
        if blocks
        else LOCAL_RAG_WEB_BOUNDARY_PROMPT
    )
    blocks.append({"type": "text", "text": prompt})
    return SystemMessage(content_blocks=blocks)


class BusinessToolBoundaryMiddleware(AgentMiddleware):
    """Enforce the application's business-only tool boundary."""

    def __init__(self, disabled_tools: Iterable[str]) -> None:
        super().__init__()
        self.disabled_tools = frozenset(disabled_tools)

    def _tool_call_budget(self, tools: Sequence[Any]) -> int:
        visible_names = {_tool_name(tool) for tool in tools}
        if "analyze_data" in visible_names:
            return DATA_AGENT_TOOL_CALL_BUDGET
        if "search_reports" in visible_names:
            return RAG_AGENT_TOOL_CALL_BUDGET
        return MAIN_AGENT_TOOL_CALL_BUDGET

    def _circuit_breaker_message(self, request: ModelRequest) -> AIMessage | None:
        total_calls, failed_signatures = _tool_history(request.messages)
        repeated_failure_count = max(failed_signatures.values(), default=0)
        if repeated_failure_count >= MAX_IDENTICAL_TOOL_FAILURES:
            return AIMessage(content=TOOL_CIRCUIT_BREAKER_MESSAGE.format(
                reason=(
                    f"同一工具、参数和错误已重复 {repeated_failure_count} 次"
                )
            ))

        tool_call_budget = self._tool_call_budget(request.tools)
        if total_calls >= tool_call_budget:
            return AIMessage(content=TOOL_CIRCUIT_BREAKER_MESSAGE.format(
                reason=f"已达到当前 Agent 的 {tool_call_budget} 次工具调用预算"
            ))
        return None

    def _prepare_model_request(self, request: ModelRequest) -> ModelRequest:
        visible_tools = [
            tool for tool in request.tools if _tool_name(tool) not in self.disabled_tools
        ]
        visible_tool_names_before_limit = {_tool_name(tool) for tool in visible_tools}
        local_rag_web_boundary = (
            "task" in visible_tool_names_before_limit
            and bool(WEB_TOOLS & visible_tool_names_before_limit)
            and _requires_local_rag_before_web(request.messages)
        )
        if local_rag_web_boundary:
            visible_tools = [
                tool for tool in visible_tools if _tool_name(tool) not in WEB_TOOLS
            ]
            visible_tool_names_before_limit -= WEB_TOOLS
        rag_search_limit_reached = (
            "search_reports" in visible_tool_names_before_limit
            and _rag_retrieval_limit_reached(request.messages)
        )
        if rag_search_limit_reached:
            visible_tools = [
                tool
                for tool in visible_tools
                if _tool_name(tool) not in RAG_RETRIEVAL_TOOLS
            ]
        visible_tool_names = sorted(
            name for tool in visible_tools if (name := _tool_name(tool)) is not None
        )
        system_message = sanitize_system_message(request.system_message)
        if local_rag_web_boundary:
            system_message = _append_local_rag_web_boundary_prompt(system_message)
        if rag_search_limit_reached:
            system_message = _append_rag_search_limit_prompt(system_message)
        system_message = _append_boundary_prompt(system_message, visible_tool_names)
        return request.override(tools=visible_tools, system_message=system_message)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> Any:
        """Sanitize the synchronous model request before it reaches the model."""
        if circuit_breaker_message := self._circuit_breaker_message(request):
            return circuit_breaker_message
        prepared_request = self._prepare_model_request(request)
        response = handler(prepared_request)
        response = _normalize_textual_tool_call(prepared_request, response)
        response = _guard_unsupported_sampling_years(prepared_request, response)
        response = _ensure_rag_specific_date_citations(prepared_request, response)
        response = _restore_subagent_result_if_source_only(prepared_request, response)
        response = _guard_same_day_web_news(prepared_request, response)
        response = _ground_stock_close_from_history(prepared_request, response)
        response = _guard_weekend_stock_date(prepared_request, response)
        response = _limit_unrequested_stock_metrics(prepared_request, response)
        return _ensure_web_source_traceability(prepared_request, response)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[Any]],
    ) -> Any:
        """Sanitize the asynchronous model request before it reaches the model."""
        if circuit_breaker_message := self._circuit_breaker_message(request):
            return circuit_breaker_message
        prepared_request = self._prepare_model_request(request)
        response = await handler(prepared_request)
        response = _normalize_textual_tool_call(prepared_request, response)
        response = _guard_unsupported_sampling_years(prepared_request, response)
        response = _ensure_rag_specific_date_citations(prepared_request, response)
        response = _restore_subagent_result_if_source_only(prepared_request, response)
        response = _guard_same_day_web_news(prepared_request, response)
        response = _ground_stock_close_from_history(prepared_request, response)
        response = _guard_weekend_stock_date(prepared_request, response)
        response = _limit_unrequested_stock_metrics(prepared_request, response)
        return _ensure_web_source_traceability(prepared_request, response)

    def _blocked_tool_message(self, request: ToolCallRequest) -> ToolMessage:
        tool_name = request.tool_call["name"]
        return ToolMessage(
            content=(
                f"工具 {tool_name!r} 已被应用业务边界禁用，未执行任何操作。"
                "请仅使用当前模型请求中列出的业务工具重新完成任务。"
            ),
            tool_call_id=request.tool_call["id"],
            name=tool_name,
            status="error",
        )

    def _rag_search_limit_message(self, request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(
            content=(
                "研报检索工具已达到当前问题的有限检索上限，"
                "本次调用未执行。请立即基于已有结果回答；若没有精确证据，"
                "明确说明已上传研报未提供该信息。"
            ),
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
        )

    def _rag_search_limit_reached(self, request: ToolCallRequest) -> bool:
        return (
            request.tool_call["name"] in RAG_RETRIEVAL_TOOLS
            and _rag_retrieval_limit_reached(_state_messages(request.state))
        )

    def _local_rag_web_boundary_reached(self, request: ToolCallRequest) -> bool:
        return (
            request.tool_call["name"] in WEB_TOOLS
            and _requires_local_rag_before_web(_state_messages(request.state))
        )

    def _blocked_web_message(self, request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(
            content=(
                "当前问题属于本地研报优先范围，且用户没有要求联网或实时信息，"
                "本次 Web 调用未执行。请通过 task 委派给 rag-analyst，并根据"
                "本地研报结果回答。"
            ),
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """Reject disabled synchronous calls before backend execution."""
        if request.tool_call["name"] in self.disabled_tools:
            return self._blocked_tool_message(request)
        if self._local_rag_web_boundary_reached(request):
            return self._blocked_web_message(request)
        if self._rag_search_limit_reached(request):
            return self._rag_search_limit_message(request)
        request = _with_inferred_rag_topic(request)
        return handler(_with_web_cutoff_date(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        """Reject disabled asynchronous calls before backend execution."""
        if request.tool_call["name"] in self.disabled_tools:
            return self._blocked_tool_message(request)
        if self._local_rag_web_boundary_reached(request):
            return self._blocked_web_message(request)
        if self._rag_search_limit_reached(request):
            return self._rag_search_limit_message(request)
        request = _with_inferred_rag_topic(request)
        return await handler(_with_web_cutoff_date(request))
