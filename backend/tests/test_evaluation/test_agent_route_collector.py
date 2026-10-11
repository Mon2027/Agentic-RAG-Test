"""Agent 完整工具调用轨迹收集器的测试。

LangChain/DeepAgents 通过 ``astream_events(version="v2")`` 异步产出链、模型和工具
事件。评测前必须把原始事件整理成稳定的 ``AgentRouteTrace``，包括：

* 收集器观察到的工具开始/结束事件顺序；
* task 工具选择的 subagent_type 及工具参数；
* start/end 事件之间的参数关联与可 JSON 序列化输出；
* 主链最终输出或流式异常；
* 超出每个评测用例工具预算时立即关闭异步流。

这里用 ``FakeStreamingAgent`` 精确控制事件序列，不需要真实模型或业务工具。
每个用例都遵循“准备事件 → 收集异步事件流 → 检查轨迹”的流程。
事件中的工具名称仅表示模拟发生的调用，不会在测试中真正执行这些工具。

阅读时区分三个字段：``run_id`` 标识一次运行并关联开始/结束；``parent_ids``
表示父级运行链；``subagent_type`` 是 task 输入参数中选择的子 Agent 类型。
收集结果中，``events`` 保存工具生命周期事件，``final_output`` 单独保存主链输出，
``stream_error`` 保存收集过程中出现的异常或工具预算错误。
"""

# dataclass 自动生成初始化方法，让 FakeOutput(content="检索完成") 成为简单数据对象。
from dataclasses import dataclass
from typing import Any

import pytest

from app.evaluation.agent_route_collector import collect_agent_route_events


class FakeStreamingAgent:
    """按给定顺序产出事件，并记录调用参数与关闭状态的确定性 Agent stub。"""

    def __init__(
        self,
        events: list[dict[str, Any]],
        error: Exception | None = None,
    ) -> None:
        # events 是正常产出序列；error 若存在，会在所有事件产出后抛出。
        self.events = events
        self.error = error
        # calls 记录收集器传给 astream_events 的参数；closed 用于观察流的生命周期。
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def astream_events(
        self,
        agent_input: dict[str, Any],
        *,
        version: str,
        config: dict[str, Any],
    ):
        """模拟 LangChain v2 异步事件流，并确保 finally 可观察流是否关闭。"""
        # async def 中包含 yield，所以此方法是异步生成器；收集器通过 async for
        # 逐条取得事件。它不会像普通协程那样一次返回整个事件列表。
        self.calls.append({
            "agent_input": agent_input,
            "version": version,
            "config": config,
        })
        try:
            for event in self.events:
                # yield 把一个事件交给收集器；只有收集器继续取值时才向下执行。
                yield event
            if self.error is not None:
                raise self.error
        finally:
            # 正常耗尽、异常或收集器主动 aclose，都会进入 finally。
            self.closed = True


@dataclass
class FakeOutput:
    """非字典输出，用于验证收集器能把 dataclass 转成 JSON 安全结构。"""

    # 收集器会将该对象转换为 {"content": "..."}，方便后续保存为 JSON。
    content: str


# pytest 的异步插件根据 asyncio 标记，在事件循环中运行这些 async 测试函数。
@pytest.mark.asyncio
async def test_collects_subagent_arguments_and_complete_tool_order():
    """应收集 task→search_reports 完整层级、参数、输出和最终回答。"""
    # 原始事件模拟：主 Agent 启动 task，RAG 子 Agent 启动/结束 search_reports，
    # task 随后结束，最后主链给出最终回答。
    events = [
        {
            # start 提供调用输入；后面的 end 使用相同 run_id 提供输出。
            "event": "on_tool_start",
            "name": "task",
            "run_id": "task-1",
            "parent_ids": ["root"],
            "tags": ["route"],
            "metadata": {"lc_agent_name": "main-agent"},
            "data": {
                "input": {
                    "subagent_type": "rag-analyst",
                    "description": "检索具身智能产业链",
                }
            },
        },
        {
            "event": "on_tool_start",
            "name": "search_reports",
            "run_id": "search-1",
            "parent_ids": ["root", "task-1"],
            "metadata": {"lc_agent_name": "rag-analyst"},
            "data": {
                "input": {
                    "query": "具身智能产业链",
                    "top_k": 5,
                    "topic": "embodied_intelligence",
                }
            },
        },
        {
            "event": "on_tool_end",
            "name": "search_reports",
            "run_id": "search-1",
            "parent_ids": ["root", "task-1"],
            "metadata": {"lc_agent_name": "rag-analyst"},
            "data": {"output": FakeOutput(content="检索完成")},
        },
        {
            "event": "on_tool_end",
            "name": "task",
            "run_id": "task-1",
            "parent_ids": ["root"],
            "metadata": {"lc_agent_name": "main-agent"},
            "data": {"output": "子Agent完成"},
        },
        {
            "event": "on_chain_end",
            "name": "main-agent",
            "run_id": "root",
            # 空父链表示根链结束，output 因此应成为 trace.final_output。
            "parent_ids": [],
            "data": {"output": {"messages": ["最终回答"]}},
        },
    ]
    agent = FakeStreamingAgent(events)

    # collector 会在调用方 config 基础上补默认 recursion_limit=50。
    # recursion_limit 是图运行的步数限制，与下方 max_tool_calls 的工具计数限制不同。
    trace = await collect_agent_route_events(
        agent,
        {"messages": ["用户问题"]},
        config={"configurable": {"test_case_id": "AGENT-RAG-001"}},
    )

    # 整个列表相等：既检查只启动一次事件流，也检查输入、事件协议和配置传递。
    assert agent.calls == [{
        "agent_input": {"messages": ["用户问题"]},
        "version": "v2",
        "config": {
            "recursion_limit": 50,
            "configurable": {"test_case_id": "AGENT-RAG-001"},
        },
    }]
    # tool_call_order 只统计 start，避免同一工具的 end 被误算成第二次调用。
    assert trace.tool_call_order == ["task", "search_reports"]
    assert trace.task_subagent_sequence == ["rag-analyst"]
    # 原始序列有5条事件，工具轨迹只有4条；主链结束事件单独用于保存最终输出。
    assert [event.event for event in trace.events] == [
        "on_tool_start",
        "on_tool_start",
        "on_tool_end",
        "on_tool_end",
    ]
    # sequence 是收集器从1开始分配的工具事件序号，不是时间戳。
    assert [event.sequence for event in trace.events] == [1, 2, 3, 4]

    # 通过 run_id 关联后，end 事件应继承对应 start 的参数与子 Agent 类型。
    # 列表解包为四个变量，方便分别检查嵌套工具的开始与结束记录。
    task_start, search_start, search_end, task_end = trace.events
    assert task_start.arguments == {
        "subagent_type": "rag-analyst",
        "description": "检索具身智能产业链",
    }
    assert task_start.subagent_type == "rag-analyst"
    assert search_start.agent_name == "rag-analyst"
    assert search_start.arguments == {
        "query": "具身智能产业链",
        "top_k": 5,
        "topic": "embodied_intelligence",
    }
    assert search_end.arguments == search_start.arguments
    # FakeOutput 被转成字典；这里检查结构转换结果，没有实际调用 json.dumps。
    assert search_end.output == {"content": "检索完成"}
    assert task_end.arguments == task_start.arguments
    assert task_end.subagent_type == "rag-analyst"
    assert trace.final_output == {"messages": ["最终回答"]}
    assert trace.stream_error is None


@pytest.mark.asyncio
async def test_keeps_repeated_and_interleaved_tool_event_order():
    """工具并发交错或重复时，应严格保留原始 start/end 到达顺序。"""
    # task 内启动 list_data_files，同时主链又启动 web_search；结束顺序与启动顺序相反。
    # 这些事件模拟并发交错，测试本身按列表顺序产出，不会启动真实并发任务。
    events = [
        {
            "event": "on_tool_start",
            "name": "task",
            "run_id": "data-task",
            "data": {"input": {"subagent_type": "data-analyst", "description": "分析数据"}},
        },
        {
            "event": "on_tool_start",
            "name": "list_data_files",
            "run_id": "list-1",
            "parent_ids": ["data-task"],
            "data": {"input": {}},
        },
        {
            "event": "on_tool_start",
            "name": "web_search",
            "run_id": "web-1",
            "data": {"input": {"query": "今日新闻", "max_results": 3}},
        },
        {
            "event": "on_tool_end",
            "name": "web_search",
            "run_id": "web-1",
            "data": {"output": "结果"},
        },
        {
            "event": "on_tool_end",
            "name": "list_data_files",
            "run_id": "list-1",
            "parent_ids": ["data-task"],
            "data": {"output": "文件列表"},
        },
        {
            "event": "on_tool_end",
            "name": "task",
            "run_id": "data-task",
            "data": {"output": "分析完成"},
        },
    ]

    trace = await collect_agent_route_events(
        FakeStreamingAgent(events),
        {"messages": ["组合问题"]},
    )

    # 调用顺序按 start 事件，而完整事件序列还必须保留交错的 end。
    assert trace.tool_call_order == ["task", "list_data_files", "web_search"]
    # 列表推导式把事件简化为“工具名、阶段”元组，直接检查到达顺序。
    assert [(event.tool, event.phase) for event in trace.events] == [
        ("task", "start"),
        ("list_data_files", "start"),
        ("web_search", "start"),
        ("web_search", "end"),
        ("list_data_files", "end"),
        ("task", "end"),
    ]
    # 索引3是第4条事件，即 web_search 的结束事件；它应继承开始时的查询参数。
    assert trace.events[3].arguments == {"query": "今日新闻", "max_results": 3}
    # -1 取最后一条事件，即 task 结束记录，子 Agent 类型也应得到恢复。
    assert trace.events[-1].subagent_type == "data-analyst"


@pytest.mark.asyncio
async def test_returns_partial_trace_when_stream_fails():
    """事件流中途失败时，应返回已收集的部分轨迹和结构化错误。"""
    agent = FakeStreamingAgent(
        events=[{
            "event": "on_tool_start",
            "name": "web_search",
            "run_id": "web-1",
            "data": {"input": {"query": "实时股价"}},
        }],
        # 替身会先产出上面的工具开始事件，再主动抛出超时异常。
        # 这是人为触发的异常，不会等待真实模型请求超时。
        error=TimeoutError("model timed out"),
    )

    trace = await collect_agent_route_events(
        agent,
        {"messages": ["查询实时股价"]},
        config={"recursion_limit": 25},
    )

    assert trace.tool_call_order == ["web_search"]
    assert trace.events[0].arguments == {"query": "实时股价"}
    # 超时不是让 collector 自身抛错，而是成为可评分、可持久化的 trace.stream_error。
    assert trace.stream_error == {
        "type": "TimeoutError",
        "message": "model timed out",
    }
    # 显式配置应覆盖默认的50；异常发生后，流也应完成关闭。
    assert agent.calls[0]["config"]["recursion_limit"] == 25
    assert agent.closed is True


@pytest.mark.asyncio
async def test_allows_stream_to_complete_at_exact_tool_call_budget():
    """工具调用数恰好等于预算时应允许完成并保留最终输出。"""
    # 两个 start 加一个根链 end；预算统计工具启动次数，不统计链结束事件。
    events = [
        {
            "event": "on_tool_start",
            "name": "tool_1",
            "run_id": "tool-1",
            "data": {"input": {"index": 1}},
        },
        {
            "event": "on_tool_start",
            "name": "tool_2",
            "run_id": "tool-2",
            "data": {"input": {"index": 2}},
        },
        {
            "event": "on_chain_end",
            "name": "main-agent",
            "run_id": "root",
            "parent_ids": [],
            "data": {"output": {"messages": ["预算内完成"]}},
        },
    ]
    agent = FakeStreamingAgent(events)

    trace = await collect_agent_route_events(
        agent,
        {"messages": ["测试预算边界"]},
        max_tool_calls=2,
    )

    # 边界条件是“超过”才关闭，等于 max_tool_calls 不应误判。
    assert trace.tool_call_order == ["tool_1", "tool_2"]
    assert trace.final_output == {"messages": ["预算内完成"]}
    assert trace.stream_error is None
    assert agent.closed is True


@pytest.mark.asyncio
async def test_closes_stream_immediately_after_tool_call_budget_is_exceeded():
    """观察到第一个超预算工具 start 后，应立即关闭流并记录预算错误。"""
    # range(1, 5) 生成四个工具开始事件，末尾再放一个不应被读取的最终输出。
    events = [
        {
            "event": "on_tool_start",
            "name": f"tool_{index}",
            "run_id": f"tool-{index}",
            "data": {"input": {"index": index}},
        }
        for index in range(1, 5)
    ]
    events.append({
        "event": "on_chain_end",
        "name": "main-agent",
        "run_id": "root",
        "parent_ids": [],
        "data": {"output": {"messages": ["不应到达"]}},
    })
    agent = FakeStreamingAgent(events)

    trace = await collect_agent_route_events(
        agent,
        {"messages": ["测试预算"]},
        max_tool_calls=2,
    )

    # 第 3 次 start 是发现越界所必需的证据；第 4 次和 chain_end 不应再被消费。
    # 收集器记录的是已观察事件，所以会保留第3条；这不表示它在执行前拒绝了第3次调用。
    assert trace.tool_call_order == ["tool_1", "tool_2", "tool_3"]
    assert trace.final_output is None
    assert trace.stream_error == {
        "type": "ToolCallBudgetExceeded",
        "message": (
            "Observed 3 tool calls; maximum allowed is 2. "
            "The Agent event stream was closed immediately."
        ),
    }
    # break 停止收集后，收集器主动 aclose()，触发替身生成器的 finally。
    assert agent.closed is True


@pytest.mark.asyncio
async def test_rejects_negative_tool_call_budget_before_starting_stream():
    """负数预算属于调用方配置错误，应在启动 Agent 流之前直接拒绝。"""
    agent = FakeStreamingAgent([])

    # 配置非法时应直接抛错；pytest.raises 检查异常类型，match 匹配错误文本。
    # 与流运行异常不同，这类参数错误不应被包装进 trace.stream_error。
    with pytest.raises(ValueError, match="max_tool_calls must be non-negative"):
        await collect_agent_route_events(
            agent,
            {"messages": ["非法预算"]},
            max_tool_calls=-1,
        )

    # 零调用证明参数校验发生在 astream_events 之前，没有模型成本或副作用。
    assert agent.calls == []
