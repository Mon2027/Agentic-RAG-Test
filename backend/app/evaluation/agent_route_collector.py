"""Collect auditable tool-routing traces from a DeepAgents event stream."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


TOOL_EVENT_PHASES = {
    "on_tool_start": "start",
    "on_tool_end": "end",
    "on_tool_error": "error",
}


def _json_safe(value: Any) -> Any:
    """Convert LangChain event values to stable JSON-compatible structures."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped_value = model_dump()
        except Exception:
            dumped_value = value
        if dumped_value is not value and isinstance(
            dumped_value,
            (Mapping, list, tuple, set, frozenset, bool, int, float, str),
        ):
            return _json_safe(dumped_value)

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


@dataclass(frozen=True)
class AgentRouteEvent:
    """One tool lifecycle event in the order emitted by the agent."""

    sequence: int
    event: str
    phase: str
    tool: str
    run_id: str | None
    parent_ids: tuple[str, ...]
    agent_name: str | None
    subagent_type: str | None
    arguments: Any = None
    output: Any = None
    error: Any = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible event record."""
        return {
            "sequence": self.sequence,
            "event": self.event,
            "phase": self.phase,
            "tool": self.tool,
            "run_id": self.run_id,
            "parent_ids": list(self.parent_ids),
            "agent_name": self.agent_name,
            "subagent_type": self.subagent_type,
            "arguments": self.arguments,
            "output": self.output,
            "error": self.error,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


@dataclass
class AgentRouteTrace:
    """Complete routing trace for one agent invocation."""

    events: list[AgentRouteEvent] = field(default_factory=list)
    final_output: Any = None
    stream_error: dict[str, str] | None = None

    @property
    def tool_call_order(self) -> list[str]:
        """Return tool names in start-call order, including repeated calls."""
        return [event.tool for event in self.events if event.phase == "start"]

    @property
    def task_subagent_sequence(self) -> list[str]:
        """Return each requested ``task.subagent_type`` in call order."""
        return [
            event.subagent_type
            for event in self.events
            if event.phase == "start"
            and event.tool == "task"
            and event.subagent_type is not None
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible trace and its routing summaries."""
        return {
            "tool_call_order": self.tool_call_order,
            "task_subagent_sequence": self.task_subagent_sequence,
            "events": [event.to_dict() for event in self.events],
            "final_output": self.final_output,
            "stream_error": self.stream_error,
        }


async def collect_agent_route_events(
    agent: Any,
    agent_input: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    max_tool_calls: int | None = None,
) -> AgentRouteTrace:
    """Collect tool events from one ``agent.astream_events`` invocation.

    The collector intentionally reads the raw v2 event stream instead of the
    public SSE API because the API currently omits the input of the ``task``
    tool. Start, end, and error events are kept separately so their list order
    remains the exact order observed from the stream, including parallel calls.

    Args:
        agent: A compiled agent exposing ``astream_events``.
        agent_input: Invocation input, normally ``{"messages": [...]}``.
        config: Optional LangGraph configuration. ``recursion_limit`` defaults
            to the production value of 50.
        max_tool_calls: Optional hard limit for observed ``on_tool_start``
            events. The event that exceeds the limit is retained, then the
            stream is explicitly closed before any further execution is read.

    Returns:
        A partial or complete trace. Stream exceptions are recorded in
        ``stream_error`` together with all events received before the failure.
    """
    if max_tool_calls is not None and max_tool_calls < 0:
        raise ValueError("max_tool_calls must be non-negative or None")

    trace = AgentRouteTrace()
    active_calls: dict[str, tuple[Any, str | None]] = {}
    effective_config: dict[str, Any] = {"recursion_limit": 50}
    if config:
        effective_config.update(config)

    event_stream = agent.astream_events(
        agent_input,
        version="v2",
        config=effective_config,
    )
    try:
        async for raw_event in event_stream:
            event_name = str(raw_event.get("event") or "")
            raw_parent_ids = raw_event.get("parent_ids") or []
            parent_ids = tuple(str(parent_id) for parent_id in raw_parent_ids)
            raw_metadata = _json_safe(raw_event.get("metadata") or {})
            metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
            data = raw_event.get("data")
            if not isinstance(data, Mapping):
                data = {}

            if event_name == "on_chain_end" and not parent_ids:
                trace.final_output = _json_safe(data.get("output"))
                continue

            phase = TOOL_EVENT_PHASES.get(event_name)
            if phase is None:
                continue

            tool_name = str(raw_event.get("name") or "")
            raw_run_id = raw_event.get("run_id")
            run_id = str(raw_run_id) if raw_run_id is not None else None
            arguments = _json_safe(data.get("input"))
            subagent_type: str | None = None

            if phase == "start" and run_id is not None:
                if tool_name == "task" and isinstance(arguments, dict):
                    raw_subagent_type = arguments.get("subagent_type")
                    if raw_subagent_type is not None:
                        subagent_type = str(raw_subagent_type)
                active_calls[run_id] = (arguments, subagent_type)
            elif run_id is not None and run_id in active_calls:
                started_arguments, started_subagent_type = active_calls[run_id]
                if arguments is None:
                    arguments = started_arguments
                subagent_type = started_subagent_type
                active_calls.pop(run_id, None)

            raw_agent_name = metadata.get("lc_agent_name")
            agent_name = str(raw_agent_name) if raw_agent_name is not None else None
            raw_tags = raw_event.get("tags") or []
            tags = tuple(str(tag) for tag in raw_tags)

            trace.events.append(AgentRouteEvent(
                sequence=len(trace.events) + 1,
                event=event_name,
                phase=phase,
                tool=tool_name,
                run_id=run_id,
                parent_ids=parent_ids,
                agent_name=agent_name,
                subagent_type=subagent_type,
                arguments=arguments,
                output=_json_safe(data.get("output")),
                error=_json_safe(data.get("error")),
                tags=tags,
                metadata=metadata,
            ))

            if (
                phase == "start"
                and max_tool_calls is not None
                and len(trace.tool_call_order) > max_tool_calls
            ):
                trace.stream_error = {
                    "type": "ToolCallBudgetExceeded",
                    "message": (
                        f"Observed {len(trace.tool_call_order)} tool calls; "
                        f"maximum allowed is {max_tool_calls}. "
                        "The Agent event stream was closed immediately."
                    ),
                }
                break
    except Exception as exc:  # evaluation must retain a partial failed trace
        trace.stream_error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        close_stream = getattr(event_stream, "aclose", None)
        if callable(close_stream):
            try:
                await close_stream()
            except Exception as exc:
                if trace.stream_error is None:
                    trace.stream_error = {
                        "type": type(exc).__name__,
                        "message": f"Failed to close Agent event stream: {exc}",
                    }

    return trace
