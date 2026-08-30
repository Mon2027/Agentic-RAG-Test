"""DeepAgents 业务工具边界中间件的安全与答案修复测试。

``BusinessToolBoundaryMiddleware`` 位于模型与工具执行之间，提供两道防线：

1. 模型调用边界：删除文件系统/执行类提示词和工具 schema，按问题类型动态隐藏
   Web/RAG 工具，检查循环熔断与总工具预算，并在模型回答后修复明显丢失的证据；
2. 工具运行时边界：即使模型幻觉出隐藏工具调用，也在真正执行前再次拒绝，并可
   为 RAG 查询推断主题、为 Web 查询补充截止日期。

测试还覆盖来源 URL、页码引用、送样年份、周末交易日、历史行情冲突等答案护栏。
大量用例直接构造 LangChain ``ModelRequest``、``ModelResponse``、``ToolMessage``
和 ``ToolCallRequest``，因此很适合学习 Agent 中间件如何观察和改写消息状态。
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from deepagents.middleware.filesystem import (
    EXECUTION_SYSTEM_PROMPT,
    FILESYSTEM_SYSTEM_PROMPT,
)
from langchain.agents.middleware.todo import WRITE_TODOS_SYSTEM_PROMPT
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from app.agents.tool_boundary import (
    BUSINESS_TOOL_BOUNDARY_PROMPT,
    LOCAL_RAG_WEB_BOUNDARY_PROMPT,
    MAX_RAG_RETRIEVAL_CALLS,
    MAX_RAG_SEARCH_CALLS,
    RAG_SEARCH_LIMIT_PROMPT,
    BusinessToolBoundaryMiddleware,
)

DISABLED_TOOLS = frozenset({"write_todos", "ls", "read_file", "execute"})


def _make_model_request(
    *,
    messages=None,
    tools=None,
) -> ModelRequest:
    """构造同时含业务提示和待清理注入提示的标准模型请求。"""
    # 故意混入 Todo、文件系统和执行提示，用于验证中间件只删除被禁能力说明，
    # 同时保留最前面的业务 Agent 原始提示。
    system_message = SystemMessage(
        content_blocks=[
            {"type": "text", "text": "保留业务 Agent 的原始提示。"},
            {"type": "text", "text": f"\n\n{WRITE_TODOS_SYSTEM_PROMPT}"},
            {
                "type": "text",
                "text": f"\n\n{FILESYSTEM_SYSTEM_PROMPT}\n\n{EXECUTION_SYSTEM_PROMPT}",
            },
        ]
    )
    return ModelRequest(
        model=MagicMock(),
        messages=messages or [],
        system_message=system_message,
        # 默认工具同时包含允许和禁用项，并混合对象/dict 两种工具 schema 形态。
        tools=tools or [
            SimpleNamespace(name="task"),
            SimpleNamespace(name="read_file"),
            {"type": "function", "function": {"name": "web_search"}},
            {"name": "execute"},
        ],
    )


def _visible_tool_names(tools):
    """统一提取对象式与 OpenAI function-dict 式工具定义中的名称。"""
    return [
        getattr(tool, "name", tool.get("function", {}).get("name"))
        if isinstance(tool, dict)
        else tool.name
        for tool in tools
    ]


def test_model_boundary_removes_hidden_prompts_and_tool_schemas():
    """模型只能看到业务工具，系统提示中不能残留已禁用能力说明。"""
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)
    handler = MagicMock(return_value="model-response")

    result = middleware.wrap_model_call(_make_model_request(), handler)

    assert result == "model-response"
    # handler 收到的是中间件清洗后的新请求，可从调用参数检查真正的模型视图。
    prepared_request = handler.call_args.args[0]
    prompt = prepared_request.system_message.text
    assert "保留业务 Agent 的原始提示。" in prompt
    assert FILESYSTEM_SYSTEM_PROMPT not in prompt
    assert EXECUTION_SYSTEM_PROMPT not in prompt
    assert WRITE_TODOS_SYSTEM_PROMPT not in prompt
    assert BUSINESS_TOOL_BOUNDARY_PROMPT in prompt
    assert "当前可用工具：task、web_search" in prompt
    assert _visible_tool_names(prepared_request.tools) == ["task", "web_search"]


def test_model_boundary_hides_web_for_financial_rag_question():
    """公司财务问题必须先走本地 RAG，因此模型本轮不能看到 Web 工具。"""
    request = _make_model_request(
        messages=[HumanMessage(
            content="凌云光2025年营收和归母净利润是多少，分别同比增长多少？"
        )],
        tools=[
            SimpleNamespace(name="task"),
            SimpleNamespace(name="web_search"),
            SimpleNamespace(name="web_search_quick"),
        ],
    )
    handler = MagicMock(return_value="model-response")
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    result = middleware.wrap_model_call(request, handler)

    assert result == "model-response"
    prepared = handler.call_args.args[0]
    assert _visible_tool_names(prepared.tools) == ["task"]
    assert LOCAL_RAG_WEB_BOUNDARY_PROMPT in prepared.system_message.text
    assert "当前可用工具：task" in prepared.system_message.text


@pytest.mark.parametrize("query", [
    "核实中信海直截至2026年8月8日的最新股价，不要使用历史价格。",
    (
        "先根据已上传研报概括奥普特的合作进展，再联网核实奥普特的"
        "最新股价，并明确区分两个来源。"
    ),
])
def test_model_boundary_keeps_web_for_explicit_realtime_request(query):
    """明确实时查询或 RAG+Web 组合请求应继续保留 Web 工具。"""
    request = _make_model_request(
        messages=[HumanMessage(content=query)],
        tools=[
            SimpleNamespace(name="task"),
            SimpleNamespace(name="web_search"),
            SimpleNamespace(name="web_search_quick"),
        ],
    )
    handler = MagicMock(return_value="model-response")
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    result = middleware.wrap_model_call(request, handler)

    assert result == "model-response"
    prepared = handler.call_args.args[0]
    assert _visible_tool_names(prepared.tools) == [
        "task",
        "web_search",
        "web_search_quick",
    ]
    assert LOCAL_RAG_WEB_BOUNDARY_PROMPT not in prepared.system_message.text


def test_model_boundary_keeps_web_available_for_direct_definition():
    """仅解释财务术语不等于查询公司研报，不能误触发本地 RAG 限制。"""
    request = _make_model_request(
        messages=[HumanMessage(content="用一句话解释什么是同比增长。")],
        tools=[
            SimpleNamespace(name="task"),
            SimpleNamespace(name="web_search"),
        ],
    )
    handler = MagicMock(return_value="model-response")
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    middleware.wrap_model_call(request, handler)

    assert _visible_tool_names(handler.call_args.args[0].tools) == [
        "task",
        "web_search",
    ]


@pytest.mark.asyncio
async def test_async_model_boundary_uses_same_sanitized_request():
    """异步模型调用路径必须应用与同步路径相同的提示词和工具清洗。"""
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)
    handler = AsyncMock(return_value="model-response")

    result = await middleware.awrap_model_call(_make_model_request(), handler)

    assert result == "model-response"
    prepared_request = handler.await_args.args[0]
    assert FILESYSTEM_SYSTEM_PROMPT not in prepared_request.system_message.text
    assert [tool.name for tool in prepared_request.tools if hasattr(tool, "name")] == [
        "task"
    ]


# ------------------------- 最终答案完整性修复 -------------------------
# 以下用例验证中间件只修复“来源标签替代正文、图表 URL 丢失”等明显缺陷，
# 对已经具备实质内容的主 Agent 综合答案保持克制，不重新生成业务结论。
@pytest.mark.asyncio
async def test_async_boundary_restores_task_body_when_answer_is_source_only():
    """主 Agent 若只输出来源标签，应恢复成功数据子 Agent 的实质正文。"""
    # ToolMessage 是 task 子 Agent 已完成的权威结果，AIMessage 则模拟主 Agent 丢正文。
    task_body = (
        "salary 描述性统计：样本数 8，均值 24,625，中位数 22,500，"
        "标准差 9,164.18，最小值 15,000，最大值 40,000。"
    )
    request = _make_model_request(
        messages=[ToolMessage(
            content=task_body,
            tool_call_id="task-call-1",
            name="task",
            status="success",
        )],
        tools=[SimpleNamespace(name="task")],
    )
    original_message = AIMessage(
        content="📊 来源：数据分析",
        id="answer-1",
        response_metadata={"model_name": "glm-4.5-air"},
    )
    handler = AsyncMock(return_value=ModelResponse(result=[original_message]))
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = await middleware.awrap_model_call(request, handler)

    assert isinstance(response, ModelResponse)
    # 修复应只替换正文，保留消息 id 和模型 response_metadata 等协议字段。
    repaired = response.result[0]
    assert isinstance(repaired, AIMessage)
    assert repaired.text == f"{task_body}\n\n📊 来源：数据分析"
    assert repaired.id == "answer-1"
    assert repaired.response_metadata["model_name"] == "glm-4.5-air"


def test_boundary_keeps_substantive_main_agent_answer_unchanged():
    """主 Agent 已给出实质性综合回答时，中间件不能越权覆盖模型内容。"""
    task_body = "子代理给出的详细数据分析。"
    request = _make_model_request(
        messages=[ToolMessage(
            content=task_body,
            tool_call_id="task-call-2",
            name="task",
            status="success",
        )],
        tools=[SimpleNamespace(name="task")],
    )
    original_message = AIMessage(
        content="salary 均值为 24,625，整体呈轻微右偏。\n\n📊 来源：数据分析"
    )
    expected = ModelResponse(result=[original_message])
    handler = MagicMock(return_value=expected)
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, handler)

    # 对象身份不变证明没有创建修复副本或重新生成消息。
    assert response is expected
    assert response.result[0] is original_message


def test_boundary_restores_single_chart_task_dropped_by_main_agent():
    """单图任务应采用权威 task 正文，保留可访问的图表 URL。"""
    chart_url = "/static/charts/chart_abc12345.png"
    request = _make_model_request(
        messages=[ToolMessage(
            content=f"Chart created successfully. Path: {chart_url}",
            tool_call_id="task-chart-1",
            name="task",
            status="success",
        )],
        tools=[SimpleNamespace(name="task")],
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(
        content="柱状图已经生成。\n\n📊 来源：数据分析"
    )]))
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, handler)

    assert response.result[0].text == (
        f"Chart created successfully. Path: {chart_url}\n\n📊 来源：数据分析"
    )


def test_boundary_does_not_duplicate_existing_chart_url():
    """答案已有图表 URL 时，权威正文恢复也不能把同一 URL 重复追加。"""
    chart_url = "/static/charts/chart_abc12345.png"
    request = _make_model_request(
        messages=[ToolMessage(
            content=f"Chart created successfully. Path: {chart_url}",
            tool_call_id="task-chart-2",
            name="task",
            status="success",
        )],
        tools=[SimpleNamespace(name="task")],
    )
    original = AIMessage(content=f"柱状图：{chart_url}\n\n📊 来源：数据分析")
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(
        request,
        MagicMock(return_value=ModelResponse(result=[original])),
    )

    assert response.result[0].text.count(chart_url) == 1
    assert response.result[0].text.startswith("Chart created successfully")
    assert response.result[0].text.endswith("📊 来源：数据分析")


def test_boundary_appends_chart_url_without_replacing_multi_task_synthesis():
    """多任务回答保留主 Agent 综合正文，只补回遗漏的图表 URL。"""
    chart_url = "/static/charts/chart_abc12345.png"
    request = _make_model_request(
        messages=[
            ToolMessage(
                content=f"数据分析完成。图表：{chart_url}",
                tool_call_id="task-chart-3",
                name="task",
                status="success",
            ),
            ToolMessage(
                content="研报分析结论。",
                tool_call_id="task-rag-1",
                name="task",
                status="success",
            ),
        ],
        tools=[SimpleNamespace(name="task")],
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(
        content="数据与研报两部分均已完成。"
    )]))
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, handler)

    assert response.result[0].text == (
        f"数据与研报两部分均已完成。\n\n图表：\n{chart_url}"
    )


# ------------------------- RAG 事实与引用护栏 -------------------------
# 年份和页码必须与对应事实直接绑定；“附近出现同一年/月”不构成证据支持。
def test_boundary_removes_sampling_year_borrowed_from_acquisition_evidence():
    """收购年份不能被错误迁移为产品开始送样的年份。"""
    # 证据只把 2025 年绑定到收购；“已在送样”没有给出开始年份。
    request = _make_model_request(
        messages=[ToolMessage(
            content=(
                "2025年4月，奥普特收购东莞泰莱51%股权。\n"
                "东莞泰莱机器人关节模组产品已在送样过程中。"
            ),
            tool_call_id="search-1",
            name="search_reports",
            status="success",
        )],
        tools=[SimpleNamespace(name="search_reports")],
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(
        content="**送样时间**：2025年期间进行送样。"
    )]))
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, handler)

    answer = response.result[0].text
    assert "2025年期间进行送样" not in answer
    assert "研报未披露送样开始年份" in answer
    assert "产品正在送样过程中" in answer


def test_boundary_repairs_historical_sampling_process_wording():
    """历史缺陷措辞应被改写为语法通顺且准确的未知年份限制。"""
    request = _make_model_request(
        messages=[ToolMessage(
            content=(
                "2025年4月，奥普特收购东莞泰莱51%股权。\n"
                "东莞泰莱机器人关节模组产品已在送样过程中。"
            ),
            tool_call_id="search-historical-1",
            name="search_reports",
            status="success",
        )],
        tools=[SimpleNamespace(name="search_reports")],
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(
        content="根据研报信息，送样过程发生在2025年期间。"
    )]))
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, handler)

    assert response.result[0].text == (
        "根据研报信息，研报未披露送样开始年份；"
        "截至报告披露时，产品正在送样过程中。"
    )


def test_boundary_adds_missing_sampling_year_limitation():
    """回答提到送样却没有年份时，应明确补充“研报未披露开始年份”。"""
    request = _make_model_request(
        messages=[ToolMessage(
            content=(
                "2025年4月，奥普特收购东莞泰莱51%股权。\n"
                "东莞泰莱机器人关节模组产品已在送样过程中。"
            ),
            tool_call_id="search-undated-1",
            name="search_reports",
            status="success",
        )],
        tools=[SimpleNamespace(name="search_reports")],
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content=(
        "## 送样进展\n- 东莞泰莱机器人关节模组产品已在送样过程中。"
    ))]))
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, handler)

    assert response.result[0].text == (
        "## 送样进展\n- 东莞泰莱机器人关节模组产品已在送样过程中。\n"
        "- 研报未披露送样开始年份"
    )


def test_boundary_keeps_directly_supported_sampling_year():
    """证据直接把年月与送样动作绑定时，受支持年份必须原样保留。"""
    request = _make_model_request(
        messages=[ToolMessage(
            content="机器人关节模组于2025年6月开始向客户送样。",
            tool_call_id="search-2",
            name="search_reports",
            status="success",
        )],
        tools=[SimpleNamespace(name="search_reports")],
    )
    original = AIMessage(content="**送样时间**：2025年6月开始送样。")
    expected = ModelResponse(result=[original])
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, MagicMock(return_value=expected))

    assert response is expected
    assert response.result[0] is original


def test_boundary_keeps_acquisition_year_when_not_claimed_as_sampling_time():
    """只要没有冒充送样时间，相邻证据中受支持的收购日期仍应保留。"""
    request = _make_model_request(
        messages=[ToolMessage(
            content=(
                "2025年4月，奥普特收购东莞泰莱51%股权。\n"
                "机器人关节模组产品已在送样过程中。"
            ),
            tool_call_id="search-3",
            name="search_reports",
            status="success",
        )],
        tools=[SimpleNamespace(name="search_reports")],
    )
    original = AIMessage(content=(
        "奥普特于2025年4月收购东莞泰莱；其机器人关节模组正在送样，"
        "研报未披露送样开始年份。"
    ))
    expected = ModelResponse(result=[original])
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, MagicMock(return_value=expected))

    assert response is expected


def test_boundary_restores_single_rag_task_with_page_citations():
    """单一 RAG 任务的完整结果若含页码，主 Agent 丢引用时应恢复权威正文。"""
    task_body = (
        "东莞泰莱机器人关节模组正在送样，研报未披露送样开始年份。\n\n"
        "📌 来源：研报《奥普特深度报告》第1页、第24页"
    )
    request = _make_model_request(
        messages=[ToolMessage(
            content=task_body,
            tool_call_id="task-rag-page-1",
            name="task",
            status="success",
        )],
        tools=[SimpleNamespace(name="task")],
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(
        content="机器人关节模组正在送样。\n\n📌 来源：研报《奥普特深度报告》"
    )]))
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, handler)

    assert response.result[0].text == task_body


def test_boundary_appends_page_that_directly_supports_specific_date():
    """具体年月事实应追加包含该年月的研报分块页码引用。"""
    evidence = """搜索结果 (找到 2 个相关片段):

--- 文档片段 1 [来源: a5d7a533-f0be-4bdb-82e3-ffb2badf7253_【奥普特】深度报告.pdf, 第23页, 章节: 合作, 相似度: 0.80] ---
2026年5月，奥普特与越疆科技达成战略合作。

--- 文档片段 2 [来源: a5d7a533-f0be-4bdb-82e3-ffb2badf7253_【奥普特】深度报告.pdf, 第21-22页, 章节: 并购, 相似度: 0.75] ---
2025年4月，公司收购东莞泰莱51%股权。

---
信息来源:
- 研报《【奥普特】深度报告.pdf》第23页
- 研报《【奥普特】深度报告.pdf》第21-22页"""
    request = _make_model_request(
        messages=[ToolMessage(
            content=evidence,
            tool_call_id="search-date-page-1",
            name="search_reports",
            status="success",
        )],
        tools=[SimpleNamespace(name="search_reports")],
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content=(
        "2026年5月与越疆合作；2025年4月收购东莞泰莱。\n\n"
        "📌 来源：研报《【奥普特】深度报告》第23页"
    ))]))
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, handler)

    answer = response.result[0].text
    assert answer.count("第23页") == 1
    assert answer.count("第21-22页") == 1
    assert "研报《【奥普特】深度报告》第21-22页" in answer


def test_boundary_does_not_duplicate_specific_date_page_citation():
    """答案已经标注支持页码时，不应重复追加同一引用。"""
    evidence = """搜索结果 (找到 1 个相关片段):

--- 文档片段 1 [来源: report_【奥普特】深度报告.pdf, 第21-22页, 相似度: 0.75] ---
2025年4月，公司收购东莞泰莱51%股权。

---
信息来源:
- 研报《【奥普特】深度报告.pdf》第21-22页"""
    request = _make_model_request(
        messages=[ToolMessage(
            content=evidence,
            tool_call_id="search-date-page-2",
            name="search_reports",
            status="success",
        )],
        tools=[SimpleNamespace(name="search_reports")],
    )
    original = AIMessage(content=(
        "2025年4月收购东莞泰莱。\n\n"
        "📌 来源：研报《奥普特深度报告》第21-22页"
    ))
    expected = ModelResponse(result=[original])
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, MagicMock(return_value=expected))

    assert response is expected


def test_boundary_does_not_bind_same_date_from_unrelated_page():
    """无关事实页即使出现同一日期，也不能被错误绑定为当前事实引用。"""
    evidence = """搜索结果 (找到 2 个相关片段):

--- 文档片段 1 [来源: report_奥普特深度报告.pdf, 第2页, 相似度: 0.60] ---
图2：公司股权结构（截至2026年7月23日）。

--- 文档片段 2 [来源: report_奥普特深度报告.pdf, 第23页, 相似度: 0.80] ---
2026年5月双方达成合作，6月发布方案，7月在上海AMTS展会展示落地案例。

---
信息来源:
- 研报《奥普特深度报告.pdf》第2页
- 研报《奥普特深度报告.pdf》第23页"""
    request = _make_model_request(
        messages=[ToolMessage(
            content=evidence,
            tool_call_id="search-date-page-3",
            name="search_reports",
            status="success",
        )],
        tools=[SimpleNamespace(name="search_reports")],
    )
    original = AIMessage(content=(
        "2026年7月在上海AMTS展会展示落地案例。\n\n"
        "📌 来源：研报《奥普特深度报告》第23页"
    ))
    expected = ModelResponse(result=[original])
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, MagicMock(return_value=expected))

    assert response is expected
    assert "第2页" not in response.result[0].text


def test_boundary_normalizes_declared_textual_tool_call():
    """模型输出的 provider 风格文本工具标记应转换成可见工具的结构化调用。"""
    request = _make_model_request(
        tools=[SimpleNamespace(name="search_reports")],
    )
    textual_message = AIMessage(content="""
<tool_call>search_reports
<arg_key>query</arg_key>
<arg_value>奥普特 越疆 战略合作</arg_value>
<arg_key>topic</arg_key>
<arg_value>embodied_intelligence</arg_value>
<arg_key>top_k</arg_key>
<arg_value>5</arg_value>
</tool_call>""")
    handler = MagicMock(return_value=ModelResponse(result=[textual_message]))
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, handler)

    normalized = response.result[0]
    assert normalized.text == ""
    assert len(normalized.tool_calls) == 1
    assert normalized.tool_calls[0]["name"] == "search_reports"
    assert normalized.tool_calls[0]["args"] == {
        "query": "奥普特 越疆 战略合作",
        "topic": "embodied_intelligence",
        "top_k": 5,
    }
    assert normalized.tool_calls[0]["id"].startswith("textual_tool_call_")


def test_boundary_preserves_prefix_when_normalizing_textual_tool_call():
    """规范化文本工具调用时，应保留调用前简短的说明性前缀。"""
    request = _make_model_request(
        tools=[SimpleNamespace(name="search_reports")],
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="""
让我继续检索更精确的证据：
<tool_call>search_reports
<arg_key>query</arg_key>
<arg_value>奥普特 关节模组 机器人应用</arg_value>
</tool_call>""")]))
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, handler)

    assert response.result[0].text == "让我继续检索更精确的证据："
    assert response.result[0].tool_calls[0]["name"] == "search_reports"


def test_boundary_normalizes_declared_textual_tool_call_without_arguments():
    """完整标记中的零参数已声明工具也应恢复为结构化调用。"""
    request = _make_model_request(
        tools=[SimpleNamespace(name="list_available_reports")],
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(
        content="<tool_call>list_available_reports</tool_call>"
    )]))
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, handler)

    assert response.result[0].tool_calls[0]["name"] == "list_available_reports"
    assert response.result[0].tool_calls[0]["args"] == {}


@pytest.mark.parametrize("content", [
    "<tool_call>execute<arg_key>command</arg_key><arg_value>dir</arg_value></tool_call>",
    "<tool_call>search_reports<arg_key>query</arg_key>缺少参数值</tool_call>",
])
def test_boundary_does_not_normalize_unknown_or_malformed_textual_call(content):
    """未知工具或残缺标记必须保持普通文本，不能被误执行。"""
    request = _make_model_request(
        tools=[SimpleNamespace(name="search_reports")],
    )
    original = AIMessage(content=content)
    expected = ModelResponse(result=[original])
    handler = MagicMock(return_value=expected)
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, handler)

    assert response is expected
    assert response.result[0] is original
    assert response.result[0].tool_calls == []


def test_boundary_leaves_native_structured_tool_call_unchanged():
    """模型原生结构化工具调用应绕过文本兼容转换并保持不变。"""
    request = _make_model_request(
        tools=[SimpleNamespace(name="search_reports")],
    )
    original = AIMessage(content="", tool_calls=[{
        "name": "search_reports",
        "args": {"query": "原生结构化调用"},
        "id": "native-call",
        "type": "tool_call",
    }])
    expected = ModelResponse(result=[original])
    handler = MagicMock(return_value=expected)
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, handler)

    assert response is expected
    assert response.result[0] is original


def test_boundary_does_not_restore_task_body_during_another_tool_call():
    """模型正在发起后续工具调用时，不能过早把旧 task 正文当最终答案恢复。"""
    request = _make_model_request(
        messages=[ToolMessage(
            content="已有的阶段性分析",
            tool_call_id="task-call-3",
            name="task",
            status="success",
        )],
        tools=[SimpleNamespace(name="task")],
    )
    tool_call_message = AIMessage(
        content="",
        tool_calls=[{
            "name": "task",
            "args": {
                "subagent_type": "data-analyst",
                "description": "补充分析",
            },
            "id": "task-call-4",
            "type": "tool_call",
        }],
    )
    expected = ModelResponse(result=[tool_call_message])
    handler = MagicMock(return_value=expected)
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, handler)

    assert response is expected
    assert response.result[0].tool_calls[0]["id"] == "task-call-4"


# ------------------------- 运行时工具执行防线 -------------------------
# 即使模型绕过 schema 可见性限制生成了结构化调用，wrap_tool_call 仍会在后端执行前
# 检查禁用工具、RAG/Web 路由策略和检索预算。
def test_runtime_boundary_rejects_disabled_tool_without_calling_backend():
    """模型幻觉出的禁用工具调用应返回错误，且不得触发后端副作用。"""
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)
    request = SimpleNamespace(
        tool_call={"name": "read_file", "args": {"file_path": "/sample.csv"}, "id": "call-1"}
    )
    handler = MagicMock()

    result = middleware.wrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.tool_call_id == "call-1"
    assert "未执行任何操作" in result.text
    handler.assert_not_called()


def test_runtime_boundary_rejects_web_for_financial_rag_question():
    """本地研报优先的财务问题即使幻觉出 Web 调用，运行时也必须拒绝。"""
    request = SimpleNamespace(
        tool_call={
            "name": "web_search",
            "args": {"query": "凌云光2025年营收和归母净利润"},
            "id": "blocked-web",
        },
        state={"messages": [HumanMessage(
            content="凌云光2025年营收和归母净利润是多少，分别同比增长多少？"
        )]},
    )
    handler = MagicMock()
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    result = middleware.wrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "本次 Web 调用未执行" in result.text
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_runtime_boundary_allows_web_for_explicit_latest_price_request():
    """用户明确查询最新股价时，Web 工具仍可进入异步后端执行。"""
    request = SimpleNamespace(
        tool_call={
            "name": "web_search_quick",
            "args": {"query": "中信海直 最新股价"},
            "id": "allowed-web",
        },
        state={"messages": [HumanMessage(
            content="请联网核实中信海直截至2026年8月8日的最新股价。"
        )]},
    )
    expected = ToolMessage(content="14.20元", tool_call_id="allowed-web")
    handler = AsyncMock(return_value=expected)
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    result = await middleware.awrap_tool_call(request, handler)

    assert result is expected
    forwarded = handler.await_args.args[0]
    assert forwarded.tool_call["args"]["query"].endswith("截至2026年8月8日")


# ------------------------- Web 来源与时效性护栏 -------------------------
# Web 结果必须可追溯到直接 URL；涉及“今天/最新股价”时，还要核验发布日期、
# 截止日是否为交易日以及结构化历史行情是否与摘要冲突。
def test_model_boundary_appends_traceable_web_sources_to_final_answer():
    """最终回答不能只写泛化 Web 来源标签，必须追加可追溯 URL。"""
    web_result = """【Web 证据】
[来源 1]
标题：中信海直历史行情
URL：https://example.com/000099/history
发布日期：2026-08-08
内容日期候选：2026-08-07
内容：2026-08-07收盘价14.20元
相关度：0.96"""
    request = _make_model_request(
        messages=[
            HumanMessage(content="核实中信海直截至2026年8月8日的最新股价。"),
            ToolMessage(
                content=web_result,
                tool_call_id="web-1",
                name="web_search",
            ),
        ],
        tools=[SimpleNamespace(name="task"), SimpleNamespace(name="web_search")],
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(
        content="截至8月8日，最新股价为14.20元。\n\n🌐 来源：联网搜索"
    )]))

    response = BusinessToolBoundaryMiddleware(DISABLED_TOOLS).wrap_model_call(
        request, handler
    )

    answer = response.result[0].text
    assert "https://example.com/000099/history" in answer
    assert "发布日期：2026-08-08" in answer
    assert "内容日期：2026-08-07" in answer


def test_multi_answer_is_not_replaced_by_rag_task_when_web_result_exists():
    """RAG+Web 综合回答必须保留 Web 部分，并补充直接来源 URL。"""
    task_body = "合作进展成立。\n\n📌 来源：研报《奥普特》第23页"
    web_result = """[来源 1]
标题：奥普特历史行情
URL：https://example.com/688686/history
发布日期：搜索服务未提供
内容日期候选：2026-08-07
内容：| 日期 | 收盘 | 开盘 | 高 | 低 | 交易量 | 涨跌幅 |
| 2026年08月07日 | 120.00 | 117.50 | 120.68 | 114.80 | 1.75M | +3.47% |
相关度：0.90"""
    request = _make_model_request(
        messages=[
            HumanMessage(content="先查研报，再核实截至2026年8月8日的股价。"),
            ToolMessage(content=task_body, tool_call_id="task-1", name="task"),
            ToolMessage(content=web_result, tool_call_id="web-1", name="web_search"),
        ],
        tools=[SimpleNamespace(name="task"), SimpleNamespace(name="web_search")],
    )
    synthesis = "合作进展成立；最近交易日收盘价120.00元。\n\n📌 来源：研报《奥普特》第23页"
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content=synthesis)]))

    response = BusinessToolBoundaryMiddleware(DISABLED_TOOLS).wrap_model_call(
        request, handler
    )

    answer = response.result[0].text
    assert synthesis in answer
    assert "https://example.com/688686/history" in answer
    assert answer != task_body


def test_same_day_news_without_verified_publication_date_is_downgraded():
    """旧页面或无发布日期页面不能被宣称为截止日当天新闻。"""
    web_result = """[来源 1]
标题：具身智能早间速递 2026年8月7日
URL：https://example.com/news/20260807
发布日期：搜索服务未提供
内容日期候选：2026年8月7日
内容：8月6日发布政策
相关度：0.90"""
    request = _make_model_request(
        messages=[
            HumanMessage(content="截至2026年8月8日，具身智能行业今天有什么最新动态？"),
            ToolMessage(content=web_result, tool_call_id="web-1", name="web_search"),
        ],
        tools=[SimpleNamespace(name="task"), SimpleNamespace(name="web_search")],
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(
        content="今天的新动态包括8月6日发布的政策。"
    )]))

    response = BusinessToolBoundaryMiddleware(DISABLED_TOOLS).wrap_model_call(
        request, handler
    )

    answer = response.result[0].text
    assert "无法可靠列出“今天”的新动态" in answer
    assert "历史背景" in answer
    assert "今天的新动态包括" not in answer
    assert "https://example.com/news/20260807" in answer


def test_same_day_news_with_verified_publication_date_is_retained():
    """来源明确发布于截止日当天时，才可以支持“今日新闻”表述。"""
    web_result = """[来源 1]
标题：官方公告
URL：https://example.com/news/today
发布日期：2026-08-08
内容日期候选：2026-08-08
内容：今日发布
相关度：0.95"""
    request = _make_model_request(
        messages=[
            HumanMessage(content="截至2026年8月8日，今天有什么最新消息？"),
            ToolMessage(content=web_result, tool_call_id="web-1", name="web_search"),
        ],
        tools=[SimpleNamespace(name="web_search")],
    )
    original = "今天发布了官方公告。"
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content=original)]))

    response = BusinessToolBoundaryMiddleware(DISABLED_TOOLS).wrap_model_call(
        request, handler
    )

    assert original in response.result[0].text
    assert "无法可靠列出" not in response.result[0].text


def test_weekend_stock_cutoff_uses_previous_weekday_as_trading_date():
    """周六截止日不能被标成交易日，应回退到有行情的最近工作日。"""
    web_result = """[来源 1]
标题：历史行情
URL：https://example.com/history
发布日期：搜索服务未提供
内容日期候选：2026-08-07
内容：| 日期 | 收盘 | 开盘 | 高 | 低 | 交易量 | 涨跌幅 |
| 2026年08月07日 | 14.20 | 14.36 | 14.36 | 14.01 | 9.17M | -1.11% |
相关度：0.95"""
    request = _make_model_request(
        messages=[
            HumanMessage(content="核实中信海直截至2026年8月8日的最新股价。"),
            ToolMessage(content=web_result, tool_call_id="web-1", name="web_search"),
        ],
        tools=[SimpleNamespace(name="web_search")],
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(
        content="最新股价14.20元。\n- **交易日期**：2026年8月8日"
    )]))

    response = BusinessToolBoundaryMiddleware(DISABLED_TOOLS).wrap_model_call(
        request, handler
    )

    answer = response.result[0].text
    assert "最近交易日为2026年8月7日" in answer
    assert "2026年8月8日没有可用交易数据" in answer
    assert "14.20元" in answer
    assert "https://example.com/history" in answer


def test_latest_price_answer_drops_unrequested_and_conflicting_market_metrics():
    """只问股价时，应删除未请求且无可靠证据的市值、区间和分析师数据。"""
    web_result = """[来源 1]
标题：奥普特历史行情
URL：https://example.com/history
发布日期：搜索服务未提供
内容日期候选：2026-08-07
内容：| 日期 | 收盘 | 开盘 | 高 | 低 | 交易量 | 涨跌幅 |
| 2026年08月07日 | 120.00 | 117.50 | 120.68 | 114.80 | 1.75M | +3.47% |
相关度：0.95"""
    request = _make_model_request(
        messages=[
            HumanMessage(content="核实奥普特截至2026年8月8日的最新股价。"),
            ToolMessage(content=web_result, tool_call_id="web-1", name="web_search"),
        ],
        tools=[SimpleNamespace(name="web_search")],
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="""
最新收盘价：120.00元
交易日期：2026年8月7日（最近交易日）
- 当日表现：开盘117.50元，最高价120.68元，涨跌幅+3.47%，成交量175万股
- 市值：141.40亿元
- 52周价格区间：85.89-187.00元
分析师平均目标价128.51元，评级为买入。
""")]))

    response = BusinessToolBoundaryMiddleware(DISABLED_TOOLS).wrap_model_call(
        request, handler
    )

    answer = response.result[0].text
    assert "收盘价为 **120.00元**" in answer
    assert "最近交易日为2026年8月7日" in answer
    assert "141.40" not in answer
    assert "52周" not in answer
    assert "分析师" not in answer
    assert "https://example.com/history" in answer


def test_historical_table_overrides_conflicting_search_summary_price():
    """带日期的历史行情表与搜索摘要冲突时，应以结构化表格价格为准。"""
    web_result = """[来源 1]
标题：奥普特历史行情
URL：https://example.com/history
发布日期：搜索服务未提供
内容日期候选：2026年08月07日
内容：| 日期 | 收盘 | 开盘 | 高 | 低 | 交易量 | 涨跌幅 |
| 2026年08月07日 | 120.00 | 117.50 | 120.68 | 114.80 | 1.75M | +3.47% |
【搜索服务 AI 摘要（未核验）】2026年8月7日收盘价118.14元
相关度：0.95"""
    request = _make_model_request(
        messages=[
            HumanMessage(content="核实奥普特截至2026年8月8日的最新股价。"),
            ToolMessage(content=web_result, tool_call_id="web-1", name="web_search"),
        ],
        tools=[SimpleNamespace(name="web_search")],
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(
        content="截至8月8日，奥普特最新股价是118.14元。"
    )]))

    response = BusinessToolBoundaryMiddleware(DISABLED_TOOLS).wrap_model_call(
        request, handler
    )

    answer = response.result[0].text
    assert "120.00元" in answer
    assert "118.14" not in answer
    assert "2026年8月7日" in answer
    assert "https://example.com/history" in answer


@pytest.mark.asyncio
async def test_runtime_boundary_allows_business_tool():
    """允许的业务工具应原样进入正常异步处理器。"""
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)
    request = SimpleNamespace(
        tool_call={"name": "task", "args": {}, "id": "call-2"}
    )
    expected = ToolMessage(content="ok", tool_call_id="call-2")
    handler = AsyncMock(return_value=expected)

    result = await middleware.awrap_tool_call(request, handler)

    assert result is expected
    handler.assert_awaited_once_with(request)


def _tool_result_pair(
    *,
    call_id: str,
    arguments: dict,
    tool_name: str = "analyze_data",
    status: str = "error",
    content: str = "validation failed",
):
    """构造一次结构化工具调用及其对应 ToolMessage 结果。"""
    return [
        AIMessage(
            content="",
            tool_calls=[{
                "name": tool_name,
                "args": arguments,
                "id": call_id,
                "type": "tool_call",
            }],
        ),
        ToolMessage(
            content=content,
            tool_call_id=call_id,
            name=tool_name,
            status=status,
        ),
    ]


# ------------------------- RAG 检索预算与主题归一化 -------------------------
# RAG 子 Agent 的搜索次数和总检索次数分别受限；工具执行前还会在缺省时推断语料 topic。
def _completed_rag_searches(count: int = MAX_RAG_SEARCH_CALLS):
    """生成指定次数已完成的 RAG 搜索历史，供预算边界测试复用。"""
    messages = []
    for index in range(count):
        messages.extend(_tool_result_pair(
            call_id=f"search-{index}",
            tool_name="search_reports",
            arguments={
                "query": f"Avant Robotics 销量 查询 {index}",
                "topic": "embodied_intelligence",
            },
            status="success",
            content="没有找到精确销量",
        ))
    return messages


def test_model_boundary_hides_rag_tools_after_four_searches():
    """完成四次 search_reports 后应隐藏 RAG 工具，强制 Agent 开始综合。"""
    request = _make_model_request(
        messages=_completed_rag_searches(),
        tools=[
            SimpleNamespace(name="list_available_reports"),
            SimpleNamespace(name="check_rag_relevance"),
            SimpleNamespace(name="search_reports"),
            SimpleNamespace(name="get_report_summary"),
        ],
    )
    expected = ModelResponse(result=[AIMessage(
        content="已上传研报未提供该实际销量。"
    )])
    handler = MagicMock(return_value=expected)
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, handler)

    assert response is expected
    prepared = handler.call_args.args[0]
    assert prepared.tools == []
    assert RAG_SEARCH_LIMIT_PROMPT in prepared.system_message.text
    assert "当前可用工具：无" in prepared.system_message.text


def test_runtime_boundary_rejects_rag_tool_after_search_limit():
    """模型幻觉出的第五次检索调用不能到达 RAG 后端。"""
    request = SimpleNamespace(
        tool_call={
            "name": "search_reports",
            "args": {"query": "继续搜索"},
            "id": "search-over-limit",
        },
        state={"messages": _completed_rag_searches()},
    )
    handler = MagicMock()
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    result = middleware.wrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "有限检索上限" in result.text
    handler.assert_not_called()


def test_model_boundary_counts_all_rag_retrieval_tools_toward_limit():
    """预检索工具也占用 RAG 总预算，从而减少后续可用搜索次数。"""
    messages = [
        *_tool_result_pair(
            call_id="relevance-1",
            tool_name="check_rag_relevance",
            arguments={"query": "Avant Robotics"},
            status="success",
            content="相关",
        ),
        *_tool_result_pair(
            call_id="reports-1",
            tool_name="list_available_reports",
            arguments={},
            status="success",
            content="研报列表",
        ),
        *_completed_rag_searches(MAX_RAG_RETRIEVAL_CALLS - 2),
    ]
    request = _make_model_request(
        messages=messages,
        tools=[SimpleNamespace(name="search_reports")],
    )
    expected = ModelResponse(result=[AIMessage(
        content="已上传研报未提供该实际销量。"
    )])
    handler = MagicMock(return_value=expected)
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    response = middleware.wrap_model_call(request, handler)

    assert response is expected
    prepared = handler.call_args.args[0]
    assert prepared.tools == []
    assert RAG_SEARCH_LIMIT_PROMPT in prepared.system_message.text


def test_runtime_boundary_infers_embodied_topic_for_robotics_query():
    """机器人强特征查询在执行前应自动补充具身智能语料主题。"""
    request = ToolCallRequest(
        tool_call={
            "name": "search_reports",
            "args": {"query": "道通科技 Avant Robotics Gen1 实际销量"},
            "id": "search-topic",
            "type": "tool_call",
        },
        tool=MagicMock(),
        state={"messages": []},
        runtime=None,
    )
    expected = ToolMessage(content="ok", tool_call_id="search-topic")
    handler = MagicMock(return_value=expected)
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    result = middleware.wrap_tool_call(request, handler)

    assert result is expected
    normalized_request = handler.call_args.args[0]
    assert normalized_request.tool_call["args"]["topic"] == "embodied_intelligence"
    assert "topic" not in request.tool_call["args"]


def test_runtime_boundary_infers_embodied_topic_for_lingyunguang_financial_query():
    """已知具身智能公司即使查询财务数据，也应自动限定到对应语料主题。"""
    request = ToolCallRequest(
        tool_call={
            "name": "search_reports",
            "args": {
                "query": "凌云光2025年营收归母净利润预测同比增长",
                "file_id": "lingyunguang-report-id",
            },
            "id": "search-lingyunguang-topic",
            "type": "tool_call",
        },
        tool=MagicMock(),
        state={"messages": []},
        runtime=None,
    )
    expected = ToolMessage(content="ok", tool_call_id="search-lingyunguang-topic")
    handler = MagicMock(return_value=expected)
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    result = middleware.wrap_tool_call(request, handler)

    assert result is expected
    normalized_request = handler.call_args.args[0]
    assert normalized_request.tool_call["args"]["topic"] == "embodied_intelligence"
    assert normalized_request.tool_call["args"]["file_id"] == "lingyunguang-report-id"


def test_runtime_boundary_preserves_explicit_rag_topic():
    """调用方显式指定的 RAG topic 必须优先于关键词自动推断。"""
    request = ToolCallRequest(
        tool_call={
            "name": "search_reports",
            "args": {"query": "机器人与低空经济交叉研究", "topic": "low_altitude"},
            "id": "search-explicit-topic",
            "type": "tool_call",
        },
        tool=MagicMock(),
        state={"messages": []},
        runtime=None,
    )
    expected = ToolMessage(content="ok", tool_call_id="search-explicit-topic")
    handler = MagicMock(return_value=expected)
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)

    result = middleware.wrap_tool_call(request, handler)

    assert result is expected
    assert handler.call_args.args[0] is request
    assert request.tool_call["args"]["topic"] == "low_altitude"


# ------------------------- 循环熔断与 Agent 总预算 -------------------------
# 熔断签名由“工具名 + 规范化参数 + 错误结果”组成；只有完全相同的失败才累计。
def test_model_boundary_stops_before_third_identical_failed_call():
    """同一工具、参数和错误连续失败两次后，应在第三次前本地熔断。"""
    arguments = {
        "file_id": "sample.csv",
        "analysis_type": "describe",
        "columns": '["salary"]',
    }
    messages = [
        *_tool_result_pair(call_id="call-1", arguments=arguments),
        *_tool_result_pair(call_id="call-2", arguments=arguments),
    ]
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)
    handler = MagicMock()

    result = middleware.wrap_model_call(
        _make_model_request(messages=messages),
        handler,
    )

    assert isinstance(result, AIMessage)
    assert "同一工具、参数和错误已重复 2 次" in result.text
    assert "不会继续调用模型或工具" in result.text
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_async_model_boundary_stops_identical_failed_call_loop():
    """异步 Agent 路径必须使用相同的重复失败熔断器。"""
    arguments = {"file_id": "sample.csv", "columns": '["salary"]'}
    messages = [
        *_tool_result_pair(call_id="call-1", arguments=arguments),
        *_tool_result_pair(call_id="call-2", arguments=arguments),
    ]
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)
    handler = AsyncMock()

    result = await middleware.awrap_model_call(
        _make_model_request(messages=messages),
        handler,
    )

    assert isinstance(result, AIMessage)
    assert "安全熔断器" in result.text
    handler.assert_not_awaited()


def test_model_boundary_does_not_mix_different_failures():
    """参数不同的失败具有不同签名，不能合并计数并误触发熔断。"""
    messages = [
        *_tool_result_pair(
            call_id="call-1",
            arguments={"file_id": "sample.csv", "columns": '["salary"]'},
        ),
        *_tool_result_pair(
            call_id="call-2",
            arguments={"file_id": "sample.csv", "columns": '["age"]'},
        ),
    ]
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)
    handler = MagicMock(return_value="model-response")

    result = middleware.wrap_model_call(
        _make_model_request(messages=messages),
        handler,
    )

    assert result == "model-response"
    handler.assert_called_once()


def test_model_boundary_does_not_treat_success_as_failed_loop():
    """重复成功调用只受总预算限制，不能被当成失败循环。"""
    arguments = {"file_id": "sample.csv", "columns": ["salary"]}
    messages = [
        *_tool_result_pair(
            call_id="call-1", arguments=arguments, status="success", content="ok"
        ),
        *_tool_result_pair(
            call_id="call-2", arguments=arguments, status="success", content="ok"
        ),
    ]
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)
    handler = MagicMock(return_value="model-response")

    result = middleware.wrap_model_call(
        _make_model_request(messages=messages),
        handler,
    )

    assert result == "model-response"
    handler.assert_called_once()


def test_model_boundary_stops_at_agent_tool_call_budget():
    """主 Agent 用满 8 次工具调用预算后，不能再请求新工具。"""
    messages = []
    for index in range(8):
        messages.extend(_tool_result_pair(
            call_id=f"call-{index}",
            arguments={"index": index},
            status="success",
            content="ok",
        ))
    middleware = BusinessToolBoundaryMiddleware(DISABLED_TOOLS)
    handler = MagicMock()

    result = middleware.wrap_model_call(
        _make_model_request(messages=messages),
        handler,
    )

    assert isinstance(result, AIMessage)
    assert "8 次工具调用预算" in result.text
    handler.assert_not_called()
