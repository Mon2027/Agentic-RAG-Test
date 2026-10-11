"""Agent 路由评测数据集执行与自动评分的离线测试。

评测流程分成三个步骤：

1. collector 把 Agent 的异步事件流保存为 ``AgentRouteTrace``；
2. evaluator 将轨迹与数据集 expected 规则比较，检查子 Agent、工具集合/顺序、
   参数约束、Web 策略、最大工具次数和是否应该直接回答。
3. 批量执行器按计划运行用例，汇总每次结果，并计算重复运行的路由稳定性。

本文件验证数据集加载、单用例评分、边界用例重复运行的稳定性、硬工具预算，以及
批量执行遇到智谱 429/1113 额度耗尽时的排除与提前终止语义。所有 Agent 都是本地
事件流替身，不访问真实模型。

学习时区分“期望规则”和“实际轨迹”：AgentRouteCase.expected 保存应该怎样调用，
AgentRouteTrace 保存实际观察到的调用，score_agent_route 用普通 Python 规则比较二者。
completion_criteria 是供人工复核的业务要求；自动通过不代表模型回答内容已被判定正确。
同步用例直接构造轨迹检查评分，异步用例则让替身产出事件，检查完整的批量流程。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.evaluation.agent_route_collector import AgentRouteEvent, AgentRouteTrace
from app.evaluation.agent_route_evaluator import (
    AgentRouteCase,
    AgentRouteDataset,
    detect_zhipu_quota_exhaustion,
    evaluate_agent_route_dataset,
    load_agent_route_dataset,
    planned_invocation_count,
    score_agent_route,
)

# 从当前文件向上：parents[0] 是 test_evaluation，parents[1] 是 tests，
# parents[2] 是 backend；因此定位数据集不依赖运行 pytest 时的工作目录。
BACKEND_DIR = Path(__file__).resolve().parents[2]
# 该 JSON 文件只被读取，用于验证已有数据集的规则和运行计划。
DATASET_PATH = (
    BACKEND_DIR / "data" / "evaluation" / "agent_routing_eval_dataset_v1.json"
)


def _start_event(
    sequence: int,
    tool: str,
    arguments: dict[str, Any],
    *,
    subagent_type: str | None = None,
) -> AgentRouteEvent:
    """快速构造一个带稳定 run_id 的工具开始事件。"""
    # 评分器主要根据 start 事件统计调用；这里直接构造标准事件，省去事件流收集步骤。
    # sequence 表示事件顺序，subagent_type 只有 task 委派场景才需要设置。
    return AgentRouteEvent(
        sequence=sequence,
        event="on_tool_start",
        phase="start",
        tool=tool,
        run_id=f"run-{sequence}",
        parent_ids=(),
        agent_name=None,
        subagent_type=subagent_type,
        arguments=arguments,
    )


def _trace(*events: AgentRouteEvent) -> AgentRouteTrace:
    """把给定事件包装成包含正常最终输出的轨迹。"""
    # *events 接收任意数量的位置参数，将它们组成元组；再转成轨迹需要的列表。
    # 补一个非空最终输出，避免只关注路由的用例因为缺少最终响应而意外失败。
    return AgentRouteTrace(
        events=list(events),
        final_output={"messages": ["完成"]},
    )


def _check_map(result) -> dict[str, bool]:
    """把检查项列表转换成 name→passed 字典，便于按规则名断言。"""
    # 字典推导式将 RouteCheck 对象列表变成 {"required_tools": True, ...}。
    return {check.name: check.passed for check in result.checks}


def test_loads_v1_dataset_and_plans_twenty_cases_with_repeats():
    """应加载 20 个唯一用例，并按 split 与边界重复策略计算计划调用数。"""
    dataset = load_agent_route_dataset(DATASET_PATH)

    # 数据集有20个 case，其中5个指定边界用例各运行3次，其他各运行1次：
    # 总次数为 20 + 5 × (3 - 1) = 30；计划计数本身不会调用 Agent。
    assert dataset.dataset_id == "agent_routing_eval_v1"
    assert len(dataset.cases) == 20
    # 集合会去重，因此该断言也检查20个 case_id 没有重复。
    assert len({case.case_id for case in dataset.cases}) == 20
    assert planned_invocation_count(dataset) == 30
    assert planned_invocation_count(dataset, splits={"dev"}) == 10
    assert planned_invocation_count(dataset, splits={"test"}) == 20
    assert planned_invocation_count(
        dataset,
        repeat_boundary_cases=False,
    ) == 20


def test_scores_correct_rag_route_and_topic_parameter():
    """正确选择 rag-analyst、search_reports 和具身主题时应全部通过。"""
    dataset = load_agent_route_dataset(DATASET_PATH)
    # next 取出指定 ID 的用例，读取它的期望规则；若不存在，会抛出 StopIteration。
    case = next(case for case in dataset.cases if case.case_id == "AGENT-RAG-001")
    trace = _trace(
        _start_event(
            1,
            "task",
            {"subagent_type": "rag-analyst", "description": "检索奥普特"},
            subagent_type="rag-analyst",
        ),
        _start_event(
            2,
            "search_reports",
            {"query": "奥普特具身智能", "topic": "embodied_intelligence"},
        ),
    )

    # score_agent_route 是纯离线函数，只读取期望规则与已收集轨迹。
    result = score_agent_route(case, trace, dataset.global_constraints)

    assert result.passed is True
    # route_signature 仅记录子 Agent 与工具启动序列，供重复稳定性比较，不包含答案正文。
    assert result.route_signature == {
        "subagents": ["rag-analyst"],
        "tools": ["task", "search_reports"],
    }
    # values() 取每项检查的布尔结果；all 要求全部为真，才能整体通过。
    assert all(_check_map(result).values())


def test_reports_general_purpose_web_fallback_wrong_order_and_wrong_topic():
    """错误子 Agent、Web 回退、工具顺序和 topic 应分别产生失败检查项。"""
    # 一条故意错误的轨迹同时制造多个违规点，用于验证评分诊断不是只给总分。
    # 具体错误：先检索后委派、topic 错误、委派 general-purpose、擅自联网。
    dataset = load_agent_route_dataset(DATASET_PATH)
    case = next(case for case in dataset.cases if case.case_id == "AGENT-RAG-001")
    trace = _trace(
        _start_event(
            1,
            "search_reports",
            {"query": "奥普特", "topic": "low_altitude"},
        ),
        _start_event(
            2,
            "task",
            {"subagent_type": "general-purpose", "description": "代为处理"},
            subagent_type="general-purpose",
        ),
        _start_event(3, "web_search", {"query": "奥普特"}),
    )

    result = score_agent_route(case, trace, dataset.global_constraints)
    checks = _check_map(result)

    # 每个命名检查都应独立失败，便于评测报告定位具体路由问题。
    # required 表示必须出现，allowed 表示允许使用，forbidden 表示禁止使用。
    assert result.passed is False
    assert checks["required_subagents"] is False
    assert checks["allowed_subagents"] is False
    assert checks["forbidden_subagents"] is False
    assert checks["required_tool_order"] is False
    assert checks["forbidden_tools"] is False
    assert checks["web_policy"] is False
    assert checks["parameter_constraints"] is False


def test_accepts_alternative_web_tool_group_and_contains_constraint():
    """Web 工具组允许候选工具二选一，查询参数只需包含规定关键词。"""
    dataset = load_agent_route_dataset(DATASET_PATH)
    case = next(case for case in dataset.cases if case.case_id == "AGENT-WEB-001")
    trace = _trace(
        _start_event(
            1,
            "web_search_quick",
            {"query": "英伟达最新股价 2026-08-08"},
        )
    )

    # required_tool_groups 中的 [web_search, web_search_quick] 只要求至少使用其一。
    # contains 只要求 query 包含“英伟达”，所以允许额外带上日期等检索词。
    result = score_agent_route(case, trace, dataset.global_constraints)

    assert result.passed is True
    assert _check_map(result)["required_tool_groups"] is True
    assert _check_map(result)["parameter_constraints"] is True


class DirectAnswerAgent:
    """只产出最终回答事件的 Agent，用于测试批量顺序和重复运行。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def astream_events(
        self,
        agent_input: dict[str, Any],
        *,
        version: str,
        config: dict[str, Any],
    ):
        """记录每次评测配置，然后产出一个正常主链结束事件。"""
        # async def 中有 yield，返回异步生成器；收集器使用 async for 读取事件。
        self.calls.append({
            "agent_input": agent_input,
            "version": version,
            "config": config,
        })
        yield {
            "event": "on_chain_end",
            "name": "main-agent",
            "run_id": f"root-{len(self.calls)}",
            "parent_ids": [],
            "data": {"output": {"messages": ["直接回答"]}},
        }


class UnexpectedToolAgent(DirectAnswerAgent):
    """本应直接回答却错误启动一次 Web 工具的 Agent。"""

    async def astream_events(
        self,
        agent_input: dict[str, Any],
        *,
        version: str,
        config: dict[str, Any],
    ):
        """先产生违规工具 start，再产生一个理论上不应被收集到的最终输出。"""
        self.calls.append({
            "agent_input": agent_input,
            "version": version,
            "config": config,
        })
        yield {
            "event": "on_tool_start",
            "name": "web_search",
            "run_id": "web-1",
            "parent_ids": ["root"],
            "data": {"input": {"query": "不应执行"}},
        }
        # 零预算会在上一条 start 被观察到后关闭生成器，因此不会读取这一条。
        yield {
            "event": "on_chain_end",
            "name": "main-agent",
            "run_id": "root",
            "parent_ids": [],
            "data": {"output": {"messages": ["不应到达"]}},
        }


class RateLimitError(RuntimeError):
    """类名模拟供应商限流异常，避免引入真实 SDK 异常依赖。"""


class FirstCallFailsAgent(DirectAnswerAgent):
    """第一次调用抛指定供应商错误，后续调用恢复正常回答。"""

    def __init__(self, message: str) -> None:
        # 初始化父类的 calls 列表，再保存本用例希望模拟的异常文本。
        super().__init__()
        self.message = message

    async def astream_events(
        self,
        agent_input: dict[str, Any],
        *,
        version: str,
        config: dict[str, Any],
    ):
        """用调用次数控制“首次失败、第二次成功”的确定性行为。"""
        self.calls.append({
            "agent_input": agent_input,
            "version": version,
            "config": config,
        })
        # 同一个替身被批量复用：第一次抛异常，若执行器继续运行，第二次便正常产出。
        if len(self.calls) == 1:
            raise RateLimitError(self.message)
        yield {
            "event": "on_chain_end",
            "name": "main-agent",
            "run_id": f"root-{len(self.calls)}",
            "parent_ids": [],
            "data": {"output": {"messages": ["直接回答"]}},
        }


def _direct_case(case_id: str) -> AgentRouteCase:
    """构造禁止任何工具调用的最小直接回答评测用例。"""
    return AgentRouteCase.from_dict({
        "id": case_id,
        "question": "直接回答",
        "category": "direct",
        "difficulty": "basic",
        "metadata": {"split": "dev"},
        "expected": {
            "route": "direct",
            "required_subagents": [],
            "allowed_subagents": [],
            "forbidden_subagents": [],
            "required_tools": [],
            "required_tool_groups": [],
            "forbidden_tools": [],
            # 即使 forbidden_tools 为空，零预算和直接回答策略也会限制所有工具调用。
            "max_tool_calls": 0,
            "allow_web": False,
            "should_direct_answer": True,
            "parameter_constraints": [],
            # completion_criteria 留给人工复核，不是自动解析答案正文的规则。
            "completion_criteria": ["直接回答"],
        },
    })


def _two_case_dataset() -> AgentRouteDataset:
    """构造两个直接回答用例，用于验证批量失败后是否继续执行。"""
    return AgentRouteDataset(
        dataset_id="fail-fast-test",
        version="1",
        execution_policy={
            "runs_per_case": 1,
            "boundary_repeat_runs": 1,
            "repeat_case_ids": [],
            "target_route_accuracy": 0.9,
        },
        data_fixture={},
        global_constraints={},
        cases=(_direct_case("DIRECT-1"), _direct_case("DIRECT-2")),
    )


# asyncio 标记让 pytest 在事件循环内运行 async 测试函数。
@pytest.mark.asyncio
async def test_batch_runner_repeats_designated_case_and_reports_stability():
    """边界用例应运行三次，并正确计算 case/run 准确率和重复稳定性。"""
    case = AgentRouteCase.from_dict({
        "id": "DIRECT-REPEAT",
        "question": "一加一等于几？",
        "category": "direct",
        "difficulty": "boundary",
        "metadata": {"split": "test"},
        "expected": {
            "route": "direct",
            "required_subagents": [],
            "allowed_subagents": [],
            "forbidden_subagents": [],
            "required_tools": [],
            "required_tool_groups": [],
            "forbidden_tools": [],
            "max_tool_calls": 0,
            "allow_web": False,
            "should_direct_answer": True,
            "parameter_constraints": [],
            "completion_criteria": ["回答正确"],
        },
    })
    dataset = AgentRouteDataset(
        dataset_id="repeat-test",
        version="1",
        execution_policy={
            "runs_per_case": 1,
            "boundary_repeat_runs": 3,
            # 重复运行由 repeat_case_ids 指定；difficulty 标签本身不触发三次运行。
            "repeat_case_ids": ["DIRECT-REPEAT"],
            "target_route_accuracy": 0.9,
        },
        data_fixture={},
        global_constraints={
            "forbidden_subagents": ["general-purpose"],
            "forbidden_tools": ["write_file"],
        },
        cases=(case,),
    )
    agent = DirectAnswerAgent()

    evaluation = await evaluate_agent_route_dataset(agent, dataset)
    metrics = evaluation.metrics()

    # 每次调用的 evaluation_run_index 必须递增，便于日志和结果对应具体重复轮次。
    assert len(agent.calls) == 3
    assert [
        call["config"]["configurable"]["evaluation_run_index"]
        for call in agent.calls
    ] == [1, 2, 3]
    # 三次结果都应通过：运行准确率为3/3，用例准确率为1/1。
    # 用例通过要求该用例的所有有效重复运行都通过。
    assert all(result.passed for result in evaluation.results)
    assert metrics["case_route_accuracy"] == 1.0
    assert metrics["run_route_accuracy"] == 1.0
    # 三次都没有工具或子 Agent 调用，路由签名相同，因此重复稳定性为1。
    assert metrics["repeat_route_stability"] == 1.0
    # 全批次完成且用例准确率达到配置的0.9目标，target_met 才应为 True。
    assert metrics["target_met"] is True


@pytest.mark.asyncio
async def test_batch_runner_enforces_zero_tool_budget_for_direct_case():
    """直接回答用例的零工具预算应在首次工具 start 时立即关闭事件流。"""
    case = AgentRouteCase.from_dict({
        "id": "DIRECT-HARD-BUDGET",
        "question": "直接回答",
        "category": "direct",
        "difficulty": "basic",
        "metadata": {"split": "dev"},
        "expected": {
            "route": "direct",
            "required_subagents": [],
            "allowed_subagents": [],
            "forbidden_subagents": [],
            "required_tools": [],
            "required_tool_groups": [],
            "forbidden_tools": ["web_search"],
            "max_tool_calls": 0,
            "allow_web": False,
            "should_direct_answer": True,
            "parameter_constraints": [],
            "completion_criteria": ["不得调用工具"],
        },
    })
    dataset = AgentRouteDataset(
        dataset_id="hard-budget-test",
        version="1",
        execution_policy={
            "runs_per_case": 1,
            "boundary_repeat_runs": 1,
            "repeat_case_ids": [],
            "target_route_accuracy": 0.9,
        },
        data_fixture={},
        global_constraints={
            "forbidden_subagents": ["general-purpose"],
            "forbidden_tools": [],
        },
        cases=(case,),
    )
    agent = UnexpectedToolAgent()

    evaluation = await evaluate_agent_route_dataset(agent, dataset)

    assert len(evaluation.results) == 1
    result = evaluation.results[0]
    assert result.passed is False
    assert result.trace.tool_call_order == ["web_search"]
    # collector 在 web_search start 后已关闭流，所以后面的 chain_end 不会进入轨迹。
    # 轨迹仍记录已观察到的违规调用；这不是工具执行前的业务拦截测试。
    assert result.trace.final_output is None
    assert result.trace.stream_error == {
        "type": "ToolCallBudgetExceeded",
        "message": (
            "Observed 1 tool calls; maximum allowed is 0. "
            "The Agent event stream was closed immediately."
        ),
    }
    checks = _check_map(result)
    assert checks["stream_completed"] is False
    assert checks["max_tool_calls"] is False


@pytest.mark.asyncio
async def test_batch_stops_after_zhipu_429_1113_and_excludes_quota_failure():
    """智谱 429/1113 额度耗尽应排除本次评分，并提前终止剩余批次。"""
    agent = FirstCallFailsAgent(
        "Error code: 429 - {'error': {'type': 'rate_limit_error', "
        "'code': '1113', 'message': '余额不足或无可用资源包'}}"
    )

    evaluation = await evaluate_agent_route_dataset(agent, _two_case_dataset())

    assert len(agent.calls) == 1
    assert len(evaluation.results) == 1
    result = evaluation.results[0]
    # 基础设施额度问题不是 Agent 路由错误，因此 passed/score 都为 None 而非 False/0。
    assert result.passed is None
    assert result.automatic_score is None
    assert result.scoring_exclusion == {
        "provider": "zhipu",
        "http_status": 429,
        "provider_code": "1113",
        "type": "RateLimitError",
        "message": (
            "Error code: 429 - {'error': {'type': 'rate_limit_error', "
            "'code': '1113', 'message': '余额不足或无可用资源包'}}"
        ),
    }
    assert evaluation.termination is not None
    assert evaluation.termination.reason == "zhipu_quota_exhausted"
    assert evaluation.termination.triggering_case_id == "DIRECT-1"

    # 指标必须明确区分 planned、observed、excluded 与 skipped，且目标是否达成未知。
    # 计划2次、实际1次、排除1次、跳过1次；total_runs 仅统计有效评分，因此为0。
    metrics = evaluation.metrics()
    assert metrics["evaluation_complete"] is False
    assert metrics["planned_runs"] == 2
    assert metrics["observed_runs"] == 1
    assert metrics["excluded_runs"] == 1
    assert metrics["skipped_runs"] == 1
    assert metrics["total_runs"] == 0
    assert metrics["target_met"] is None
    assert evaluation.to_dict()["batch_status"]["complete"] is False
    assert "(INCOMPLETE)" in evaluation.summary()


@pytest.mark.asyncio
async def test_batch_does_not_stop_for_other_429_codes():
    """其他 429 错误不等于额度耗尽：首用例失败后仍应继续后续用例。"""
    agent = FirstCallFailsAgent(
        "Error code: 429 - {'error': {'type': 'rate_limit_error', "
        "'code': '1302', 'message': 'requests too frequent'}}"
    )

    evaluation = await evaluate_agent_route_dataset(agent, _two_case_dataset())

    assert len(agent.calls) == 2
    assert len(evaluation.results) == 2
    # 1302 不满足额度耗尽识别规则，首次失败仍计入评分，第二个用例继续执行。
    assert evaluation.results[0].passed is False
    assert evaluation.results[0].scoring_exclusion is None
    assert evaluation.results[1].passed is True
    assert evaluation.termination is None
    # complete 表示计划全部执行完，不表示所有用例都通过。
    assert evaluation.metrics()["evaluation_complete"] is True


def test_detects_zhipu_quota_exhaustion_from_tool_error_event():
    """额度识别器应能从嵌套的 on_tool_error 结构中提取 429/1113。"""
    trace = AgentRouteTrace(events=[AgentRouteEvent(
        sequence=1,
        event="on_tool_error",
        phase="error",
        tool="task",
        run_id="task-1",
        parent_ids=(),
        agent_name=None,
        subagent_type="data-analyst",
        arguments={"subagent_type": "data-analyst"},
        error={
            "status_code": 429,
            "error": {
                "type": "rate_limit_error",
                "code": "1113",
                "message": "余额不足或无可用资源包",
            },
        },
    )])

    # 识别器还要检查工具错误载荷，不能仅检查 trace.stream_error；
    # 此处的 HTTP 429、供应商码1113与限流/余额文本共同满足识别条件。
    detected = detect_zhipu_quota_exhaustion(trace)

    assert detected is not None
    assert detected["http_status"] == 429
    assert detected["provider_code"] == "1113"


def test_rejects_unknown_case_selection_before_any_agent_call():
    """选择不存在的 case_id 时应在执行 Agent 前直接报错。"""
    dataset = load_agent_route_dataset(DATASET_PATH)

    # pytest.raises 检查异常类型及错误文本；本用例只调用计划函数，没有启动 Agent。
    with pytest.raises(ValueError, match="Unknown case IDs"):
        planned_invocation_count(dataset, case_ids={"AGENT-UNKNOWN"})
