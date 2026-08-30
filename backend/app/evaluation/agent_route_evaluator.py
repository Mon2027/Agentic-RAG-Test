"""Evaluate Agent routing traces against a structured routing dataset.

The evaluator is deliberately independent from the real LLM.  Production
traces, deterministic fake traces, and previously saved traces can therefore
all be scored with the same rules.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.agent_route_collector import (
    AgentRouteEvent,
    AgentRouteTrace,
    collect_agent_route_events,
)


@dataclass(frozen=True)
class AgentRouteCase:
    """One routing question and its machine-checkable expectations."""

    case_id: str
    question: str
    category: str
    difficulty: str
    split: str
    metadata: dict[str, Any]
    expected: dict[str, Any]

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "AgentRouteCase":
        metadata = dict(item.get("metadata") or {})
        expected = dict(item.get("expected") or {})
        required_values = {
            "id": item.get("id"),
            "question": item.get("question"),
            "category": item.get("category"),
            "difficulty": item.get("difficulty"),
            "metadata.split": metadata.get("split"),
            "expected.route": expected.get("route"),
        }
        missing = [name for name, value in required_values.items() if not value]
        if missing:
            raise ValueError(
                "Agent route case is missing required values: "
                + ", ".join(missing)
            )
        return cls(
            case_id=str(item["id"]),
            question=str(item["question"]),
            category=str(item["category"]),
            difficulty=str(item["difficulty"]),
            split=str(metadata["split"]),
            metadata=metadata,
            expected=expected,
        )


@dataclass(frozen=True)
class AgentRouteDataset:
    """Validated routing dataset plus execution and global safety policies."""

    dataset_id: str
    version: str
    execution_policy: dict[str, Any]
    data_fixture: dict[str, Any]
    global_constraints: dict[str, Any]
    cases: tuple[AgentRouteCase, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentRouteDataset":
        cases = tuple(AgentRouteCase.from_dict(item) for item in data.get("cases", []))
        if not cases:
            raise ValueError("Agent route dataset contains no cases")

        case_ids = [case.case_id for case in cases]
        duplicate_ids = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
        if duplicate_ids:
            raise ValueError(
                "Agent route dataset contains duplicate IDs: "
                + ", ".join(duplicate_ids)
            )

        execution_policy = dict(data.get("execution_policy") or {})
        repeat_ids = set(execution_policy.get("repeat_case_ids") or [])
        unknown_repeat_ids = sorted(repeat_ids.difference(case_ids))
        if unknown_repeat_ids:
            raise ValueError(
                "Execution policy references unknown repeat IDs: "
                + ", ".join(unknown_repeat_ids)
            )

        return cls(
            dataset_id=str(data.get("dataset_id") or "agent-routing-evaluation"),
            version=str(data.get("version") or "unknown"),
            execution_policy=execution_policy,
            data_fixture=dict(data.get("data_fixture") or {}),
            global_constraints=dict(data.get("global_constraints") or {}),
            cases=cases,
        )


@dataclass(frozen=True)
class RouteCheck:
    """One auditable automatic scoring decision."""

    name: str
    passed: bool
    expected: Any
    actual: Any
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
            "detail": self.detail,
        }


@dataclass
class AgentRouteCaseResult:
    """Scored result for one invocation of one dataset case."""

    case_id: str
    run_index: int
    split: str
    category: str
    difficulty: str
    expected_route: str
    checks: list[RouteCheck]
    trace: AgentRouteTrace
    completion_criteria: list[str] = field(default_factory=list)
    scoring_exclusion: dict[str, Any] | None = None

    @property
    def passed(self) -> bool | None:
        if self.scoring_exclusion is not None:
            return None
        return all(check.passed for check in self.checks)

    @property
    def automatic_score(self) -> float | None:
        if self.scoring_exclusion is not None:
            return None
        if not self.checks:
            return 0.0
        return sum(check.passed for check in self.checks) / len(self.checks)

    @property
    def route_signature(self) -> dict[str, list[str]]:
        """Return the exact observable route for repeat-stability checks."""
        return {
            "subagents": self.trace.task_subagent_sequence,
            "tools": self.trace.tool_call_order,
        }

    def to_dict(self) -> dict[str, Any]:
        automatic_score = self.automatic_score
        return {
            "case_id": self.case_id,
            "run_index": self.run_index,
            "split": self.split,
            "category": self.category,
            "difficulty": self.difficulty,
            "expected_route": self.expected_route,
            "passed": self.passed,
            "automatic_score": (
                round(automatic_score, 6) if automatic_score is not None else None
            ),
            "scoring": {
                "included": self.scoring_exclusion is None,
                "exclusion_reason": self.scoring_exclusion,
            },
            "route_signature": self.route_signature,
            "checks": [check.to_dict() for check in self.checks],
            "manual_review": {
                "required": bool(self.completion_criteria),
                "completion_criteria": self.completion_criteria,
            },
            "trace": self.trace.to_dict(),
        }


@dataclass(frozen=True)
class EvaluationTermination:
    """Why a real evaluation batch stopped before all planned invocations."""

    reason: str
    triggering_case_id: str
    triggering_run_index: int
    error: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "triggering_case_id": self.triggering_case_id,
            "triggering_run_index": self.triggering_run_index,
            "error": self.error,
        }


@dataclass
class AgentRouteEvaluationResult:
    """All invocation results and aggregate routing metrics."""

    dataset_id: str
    dataset_version: str
    target_route_accuracy: float
    results: list[AgentRouteCaseResult]
    planned_invocations: int | None = None
    termination: EvaluationTermination | None = None
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def metrics(self) -> dict[str, Any]:
        scored_results = [
            result for result in self.results if result.scoring_exclusion is None
        ]
        observed_runs = len(self.results)
        excluded_runs = observed_runs - len(scored_results)
        planned_runs = (
            self.planned_invocations
            if self.planned_invocations is not None
            else observed_runs
        )
        skipped_runs = max(0, planned_runs - observed_runs)
        evaluation_complete = self.termination is None and skipped_runs == 0

        total_runs = len(scored_results)
        passed_runs = sum(result.passed is True for result in scored_results)
        grouped_by_case: dict[str, list[AgentRouteCaseResult]] = defaultdict(list)
        for result in scored_results:
            grouped_by_case[result.case_id].append(result)

        passed_cases = sum(
            all(result.passed for result in case_results)
            for case_results in grouped_by_case.values()
        )
        repeated_cases = {
            case_id: case_results
            for case_id, case_results in grouped_by_case.items()
            if len(case_results) > 1
        }
        stable_repeated_cases = sum(
            len({
                json.dumps(result.route_signature, ensure_ascii=False, sort_keys=True)
                for result in case_results
            }) == 1
            for case_results in repeated_cases.values()
        )

        split_metrics = _group_pass_metrics(scored_results, "split")
        category_metrics = _group_pass_metrics(scored_results, "category")

        checks: dict[str, list[bool]] = defaultdict(list)
        for result in scored_results:
            for check in result.checks:
                checks[check.name].append(check.passed)
        check_pass_rates = {
            name: round(sum(values) / len(values), 6)
            for name, values in sorted(checks.items())
        }

        run_accuracy = passed_runs / total_runs if total_runs else 0.0
        total_cases = len(grouped_by_case)
        case_accuracy = passed_cases / total_cases if total_cases else 0.0
        target_met = (
            case_accuracy >= self.target_route_accuracy
            if evaluation_complete
            else None
        )
        stability_rate = (
            stable_repeated_cases / len(repeated_cases)
            if repeated_cases
            else None
        )
        return {
            "evaluation_complete": evaluation_complete,
            "planned_runs": planned_runs,
            "observed_runs": observed_runs,
            "excluded_runs": excluded_runs,
            "skipped_runs": skipped_runs,
            "total_runs": total_runs,
            "passed_runs": passed_runs,
            "run_route_accuracy": round(run_accuracy, 6),
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "case_route_accuracy": round(case_accuracy, 6),
            "target_route_accuracy": self.target_route_accuracy,
            "target_met": target_met,
            "repeated_cases": len(repeated_cases),
            "stable_repeated_cases": stable_repeated_cases,
            "repeat_route_stability": (
                round(stability_rate, 6) if stability_rate is not None else None
            ),
            "by_split": split_metrics,
            "by_category": category_metrics,
            "check_pass_rates": check_pass_rates,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "generated_at": self.generated_at,
            "metrics": self.metrics(),
            "batch_status": {
                "complete": self.termination is None,
                "termination": (
                    self.termination.to_dict() if self.termination is not None else None
                ),
            },
            "results": [result.to_dict() for result in self.results],
        }

    def summary(self) -> str:
        metrics = self.metrics()
        stability = metrics["repeat_route_stability"]
        stability_text = "未执行重复样例" if stability is None else f"{stability:.1%}"
        target_met = metrics["target_met"]
        target_status = (
            "INCOMPLETE"
            if target_met is None
            else ("PASS" if target_met else "FAIL")
        )
        lines = [
            f"Dataset: {self.dataset_id} v{self.dataset_version}",
            (
                "Execution: "
                f"{metrics['observed_runs']}/{metrics['planned_runs']} observed, "
                f"{metrics['excluded_runs']} excluded, "
                f"{metrics['skipped_runs']} skipped"
            ),
            (
                "Route accuracy (cases): "
                f"{metrics['passed_cases']}/{metrics['total_cases']} "
                f"({metrics['case_route_accuracy']:.1%})"
            ),
            (
                "Route accuracy (runs): "
                f"{metrics['passed_runs']}/{metrics['total_runs']} "
                f"({metrics['run_route_accuracy']:.1%})"
            ),
            f"Repeat route stability: {stability_text}",
            (
                f"Target: {self.target_route_accuracy:.1%} "
                f"({target_status})"
            ),
        ]
        if self.termination is not None:
            lines.append(
                "Stopped early: "
                f"{self.termination.reason} at "
                f"{self.termination.triggering_case_id} "
                f"run {self.termination.triggering_run_index}"
            )
        return "\n".join(lines)


def load_agent_route_dataset(path: Path | str) -> AgentRouteDataset:
    """Load and structurally validate a routing dataset JSON file."""
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)
    if not isinstance(raw_data, dict):
        raise ValueError("Agent route dataset root must be a JSON object")
    return AgentRouteDataset.from_dict(raw_data)


def detect_zhipu_quota_exhaustion(trace: AgentRouteTrace) -> dict[str, Any] | None:
    """Return a normalized fatal-error record for Zhipu 429/code 1113.

    Only error payloads are inspected. Requiring the HTTP status, provider code,
    and a rate-limit/balance marker avoids stopping a batch for unrelated text
    that happens to contain one of the numbers.
    """
    error_payloads: list[Any] = []
    if trace.stream_error is not None:
        error_payloads.append(trace.stream_error)
    error_payloads.extend(
        event.error
        for event in trace.events
        if event.phase == "error" and event.error is not None
    )
    if not error_payloads:
        return None

    serialized = json.dumps(error_payloads, ensure_ascii=False, sort_keys=True)
    normalized = serialized.lower()
    has_http_429 = re.search(r"(?<!\d)429(?!\d)", normalized) is not None
    has_provider_code_1113 = re.search(r"(?<!\d)1113(?!\d)", normalized) is not None
    has_quota_marker = any(marker in normalized for marker in (
        "ratelimiterror",
        "rate_limit_error",
        "余额不足",
        "资源包",
    ))
    if not (has_http_429 and has_provider_code_1113 and has_quota_marker):
        return None

    stream_error = trace.stream_error or {}
    return {
        "provider": "zhipu",
        "http_status": 429,
        "provider_code": "1113",
        "type": str(stream_error.get("type") or "ProviderQuotaExhausted"),
        "message": str(stream_error.get("message") or serialized),
    }


def planned_invocation_count(
    dataset: AgentRouteDataset,
    *,
    splits: set[str] | None = None,
    case_ids: set[str] | None = None,
    repeat_boundary_cases: bool = True,
) -> int:
    """Return how many real Agent calls the selected plan would make."""
    selected = _select_cases(dataset, splits=splits, case_ids=case_ids)
    return sum(
        _run_count(dataset, case, repeat_boundary_cases)
        for case in selected
    )


def score_agent_route(
    case: AgentRouteCase,
    trace: AgentRouteTrace,
    global_constraints: dict[str, Any] | None = None,
    *,
    run_index: int = 1,
    scoring_exclusion: dict[str, Any] | None = None,
) -> AgentRouteCaseResult:
    """Score a collected trace without invoking a model or external service."""
    constraints = global_constraints or {}
    expected = case.expected
    checks: list[RouteCheck] = []
    tool_order = trace.tool_call_order
    subagent_order = trace.task_subagent_sequence
    start_events = [event for event in trace.events if event.phase == "start"]

    def add_check(
        name: str,
        passed: bool,
        expected_value: Any,
        actual_value: Any,
        detail: str,
    ) -> None:
        checks.append(RouteCheck(
            name=name,
            passed=bool(passed),
            expected=expected_value,
            actual=actual_value,
            detail=detail,
        ))

    add_check(
        "stream_completed",
        trace.stream_error is None,
        None,
        trace.stream_error,
        "Agent event stream must complete without an exception.",
    )
    tool_errors = [event.to_dict() for event in trace.events if event.phase == "error"]
    add_check(
        "tool_calls_completed",
        not tool_errors,
        [],
        tool_errors,
        "No tool call may finish with an on_tool_error event.",
    )
    add_check(
        "final_output_present",
        trace.final_output is not None,
        "non-null final output",
        trace.final_output,
        "The root Agent invocation must emit a final output.",
    )

    required_subagents = _string_list(expected.get("required_subagents"))
    missing_subagents = [
        name for name in required_subagents if name not in subagent_order
    ]
    add_check(
        "required_subagents",
        not missing_subagents,
        required_subagents,
        subagent_order,
        f"Missing required subagents: {missing_subagents}",
    )

    allowed_subagents = set(_string_list(expected.get("allowed_subagents")))
    unexpected_subagents = [
        name for name in subagent_order if name not in allowed_subagents
    ]
    add_check(
        "allowed_subagents",
        not unexpected_subagents,
        sorted(allowed_subagents),
        subagent_order,
        f"Unexpected subagents: {unexpected_subagents}",
    )

    forbidden_subagents = set(_string_list(expected.get("forbidden_subagents")))
    forbidden_subagents.update(_string_list(constraints.get("forbidden_subagents")))
    used_forbidden_subagents = [
        name for name in subagent_order if name in forbidden_subagents
    ]
    add_check(
        "forbidden_subagents",
        not used_forbidden_subagents,
        sorted(forbidden_subagents),
        used_forbidden_subagents,
        "Forbidden subagents must never be delegated to.",
    )

    required_tools = _string_list(expected.get("required_tools"))
    missing_tools = [name for name in required_tools if name not in tool_order]
    add_check(
        "required_tools",
        not missing_tools,
        required_tools,
        tool_order,
        f"Missing required tools: {missing_tools}",
    )
    add_check(
        "required_tool_order",
        _is_ordered_subsequence(required_tools, tool_order),
        required_tools,
        tool_order,
        "Required tools must appear in dataset order, while extra calls are allowed.",
    )

    required_groups = [
        _string_list(group)
        for group in expected.get("required_tool_groups") or []
    ]
    missing_groups = [
        group for group in required_groups
        if not any(tool_name in tool_order for tool_name in group)
    ]
    add_check(
        "required_tool_groups",
        not missing_groups,
        required_groups,
        tool_order,
        f"No alternative tool was used for groups: {missing_groups}",
    )

    forbidden_tools = set(_string_list(expected.get("forbidden_tools")))
    forbidden_tools.update(_string_list(constraints.get("forbidden_tools")))
    used_forbidden_tools = [
        tool_name for tool_name in tool_order if tool_name in forbidden_tools
    ]
    add_check(
        "forbidden_tools",
        not used_forbidden_tools,
        sorted(forbidden_tools),
        used_forbidden_tools,
        "Case-level and global forbidden tools must not be called.",
    )

    max_tool_calls = int(expected.get("max_tool_calls", 0))
    add_check(
        "max_tool_calls",
        len(tool_order) <= max_tool_calls,
        max_tool_calls,
        len(tool_order),
        "Tool-call count must stay within the case budget.",
    )

    web_tools = {"web_search", "web_search_quick"}
    used_web_tools = [name for name in tool_order if name in web_tools]
    allow_web = bool(expected.get("allow_web"))
    add_check(
        "web_policy",
        allow_web or not used_web_tools,
        {"allow_web": allow_web},
        used_web_tools,
        "Web tools are prohibited unless this case explicitly allows them.",
    )

    should_direct_answer = bool(expected.get("should_direct_answer"))
    is_direct_answer = not tool_order
    add_check(
        "direct_answer_policy",
        is_direct_answer == should_direct_answer,
        should_direct_answer,
        is_direct_answer,
        "Direct-answer cases use no tools; routed cases must use at least one tool.",
    )

    parameter_results = []
    for constraint in expected.get("parameter_constraints") or []:
        result = _evaluate_parameter_constraint(constraint, start_events)
        parameter_results.append(result)
    add_check(
        "parameter_constraints",
        all(result["passed"] for result in parameter_results),
        expected.get("parameter_constraints") or [],
        parameter_results,
        "Each declared tool-argument constraint must be satisfied by a matching call.",
    )

    return AgentRouteCaseResult(
        case_id=case.case_id,
        run_index=run_index,
        split=case.split,
        category=case.category,
        difficulty=case.difficulty,
        expected_route=str(expected["route"]),
        checks=checks,
        trace=trace,
        completion_criteria=_string_list(expected.get("completion_criteria")),
        scoring_exclusion=scoring_exclusion,
    )


async def evaluate_agent_route_dataset(
    agent: Any,
    dataset: AgentRouteDataset,
    *,
    splits: set[str] | None = None,
    case_ids: set[str] | None = None,
    repeat_boundary_cases: bool = True,
    recursion_limit: int = 50,
) -> AgentRouteEvaluationResult:
    """Invoke an Agent for selected cases and score every collected trace."""
    selected = _select_cases(dataset, splits=splits, case_ids=case_ids)
    planned_runs = [
        (case, run_index)
        for case in selected
        for run_index in range(
            1,
            _run_count(dataset, case, repeat_boundary_cases) + 1,
        )
    ]
    results: list[AgentRouteCaseResult] = []
    termination: EvaluationTermination | None = None
    for case, run_index in planned_runs:
        max_tool_calls = int(case.expected.get("max_tool_calls", 0))
        trace = await collect_agent_route_events(
            agent,
            {
                "messages": [{
                    "role": "user",
                    "content": case.question,
                }]
            },
            config={
                "recursion_limit": recursion_limit,
                "configurable": {
                    "evaluation_dataset_id": dataset.dataset_id,
                    "evaluation_case_id": case.case_id,
                    "evaluation_run_index": run_index,
                },
            },
            max_tool_calls=max_tool_calls,
        )
        quota_error = detect_zhipu_quota_exhaustion(trace)
        results.append(score_agent_route(
            case,
            trace,
            dataset.global_constraints,
            run_index=run_index,
            scoring_exclusion=quota_error,
        ))
        if quota_error is not None:
            termination = EvaluationTermination(
                reason="zhipu_quota_exhausted",
                triggering_case_id=case.case_id,
                triggering_run_index=run_index,
                error=quota_error,
            )
            break

    return AgentRouteEvaluationResult(
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        target_route_accuracy=float(
            dataset.execution_policy.get("target_route_accuracy", 0.9)
        ),
        results=results,
        planned_invocations=len(planned_runs),
        termination=termination,
    )


def _select_cases(
    dataset: AgentRouteDataset,
    *,
    splits: set[str] | None,
    case_ids: set[str] | None,
) -> list[AgentRouteCase]:
    known_ids = {case.case_id for case in dataset.cases}
    if case_ids:
        unknown_ids = sorted(case_ids.difference(known_ids))
        if unknown_ids:
            raise ValueError("Unknown case IDs: " + ", ".join(unknown_ids))
    selected = [
        case for case in dataset.cases
        if (not splits or case.split in splits)
        and (not case_ids or case.case_id in case_ids)
    ]
    if not selected:
        raise ValueError("No Agent routing cases matched the requested selection")
    return selected


def _run_count(
    dataset: AgentRouteDataset,
    case: AgentRouteCase,
    repeat_boundary_cases: bool,
) -> int:
    base_runs = max(1, int(dataset.execution_policy.get("runs_per_case", 1)))
    repeat_ids = set(dataset.execution_policy.get("repeat_case_ids") or [])
    if repeat_boundary_cases and case.case_id in repeat_ids:
        return max(
            base_runs,
            int(dataset.execution_policy.get("boundary_repeat_runs", base_runs)),
        )
    return base_runs


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _is_ordered_subsequence(required: list[str], observed: list[str]) -> bool:
    if not required:
        return True
    required_index = 0
    for item in observed:
        if item == required[required_index]:
            required_index += 1
            if required_index == len(required):
                return True
    return False


def _evaluate_parameter_constraint(
    constraint: dict[str, Any],
    start_events: list[AgentRouteEvent],
) -> dict[str, Any]:
    tool_pattern = str(constraint.get("tool") or "")
    tool_names = set(tool_pattern.split("|"))
    argument_name = str(constraint.get("argument") or "")
    operator = str(constraint.get("operator") or "")
    expected_value = constraint.get("value")
    matching_events = [event for event in start_events if event.tool in tool_names]
    actual_values = [
        _event_argument(event, argument_name) for event in matching_events
    ]
    passed = any(
        _value_matches(actual, operator, expected_value)
        for actual in actual_values
    )
    return {
        "tool": tool_pattern,
        "argument": argument_name,
        "operator": operator,
        "expected": expected_value,
        "actual_values": actual_values,
        "passed": passed,
    }


def _event_argument(event: AgentRouteEvent, argument_name: str) -> Any:
    if argument_name == "subagent_type" and event.subagent_type is not None:
        return event.subagent_type
    value: Any = event.arguments
    for part in argument_name.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _value_matches(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "equals":
        return actual == expected
    if operator in {"contains", "includes"}:
        if isinstance(actual, str):
            return str(expected) in actual
        if isinstance(actual, (list, tuple, set, frozenset, dict)):
            return expected in actual
        return False
    if operator == "not_empty":
        if actual is None:
            return False
        if isinstance(actual, (str, list, tuple, set, frozenset, dict)):
            return len(actual) > 0
        return bool(actual)
    raise ValueError(f"Unsupported parameter constraint operator: {operator}")


def _group_pass_metrics(
    results: list[AgentRouteCaseResult],
    attribute: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[AgentRouteCaseResult]] = defaultdict(list)
    for result in results:
        grouped[str(getattr(result, attribute))].append(result)
    metrics: dict[str, dict[str, Any]] = {}
    for name, group in sorted(grouped.items()):
        passed = sum(result.passed for result in group)
        metrics[name] = {
            "runs": len(group),
            "passed": passed,
            "accuracy": round(passed / len(group), 6),
        }
    return metrics
