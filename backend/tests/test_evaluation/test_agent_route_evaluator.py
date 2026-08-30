"""Agent 路由评测数据集执行与自动评分的离线测试。

评测流程分成两部分：

1. collector 把 Agent 的异步事件流保存为 ``AgentRouteTrace``；
2. evaluator 将轨迹与数据集 expected 规则比较，检查子 Agent、工具集合/顺序、
   参数约束、Web 策略、最大工具次数和是否应该直接回答。

本文件验证数据集加载、单用例评分、边界用例重复运行的稳定性、硬工具预算，以及
批量执行遇到智谱 429/1113 额度耗尽时的排除与提前终止语义。所有 Agent 都是本地
事件流替身，不访问真实模型。
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


BACKEND_DIR = Path(__file__).resolve().parents[2]
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
    return AgentRouteTrace(
        events=list(events),
        final_output={"messages": ["完成"]},
    )


def _check_map(result) -> dict[str, bool]:
    """把检查项列表转换成 name→passed 字典，便于按规则名断言。"""
    return {check.name: check.passed for check in result.checks}


def test_loads_v1_dataset_and_plans_twenty_cases_with_repeats():
    """应加载 20 个唯一用例，并按 split 与边界重复策略计算计划调用数。"""
    dataset = load_agent_route_dataset(DATASET_PATH)

    # 数据集有 20 个 case，但边界 case 重复后总计划运行数为 30。
    assert dataset.dataset_id == "agent_routing_eval_v1"
    assert len(dataset.cases) == 20
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
    assert result.route_signature == {
        "subagents": ["rag-analyst"],
        "tools": ["task", "search_reports"],
    }
    assert all(_check_map(result).values())


def test_reports_general_purpose_web_fallback_wrong_order_and_wrong_topic():
    """错误子 Agent、Web 回退、工具顺序和 topic 应分别产生失败检查项。"""
    # 一条故意错误的轨迹同时制造多个违规点，用于验证评分诊断不是只给总分。
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
            "max_tool_calls": 0,
            "allow_web": False,
            "should_direct_answer": True,
            "parameter_constraints": [],
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
    assert all(result.passed for result in evaluation.results)
    assert metrics["case_route_accuracy"] == 1.0
    assert metrics["run_route_accuracy"] == 1.0
    assert metrics["repeat_route_stability"] == 1.0
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
    assert evaluation.results[0].passed is False
    assert evaluation.results[0].scoring_exclusion is None
    assert evaluation.results[1].passed is True
    assert evaluation.termination is None
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

    detected = detect_zhipu_quota_exhaustion(trace)

    assert detected is not None
    assert detected["http_status"] == 429
    assert detected["provider_code"] == "1113"


def test_rejects_unknown_case_selection_before_any_agent_call():
    """选择不存在的 case_id 时应在执行 Agent 前直接报错。"""
    dataset = load_agent_route_dataset(DATASET_PATH)

    with pytest.raises(ValueError, match="Unknown case IDs"):
        planned_invocation_count(dataset, case_ids={"AGENT-UNKNOWN"})
