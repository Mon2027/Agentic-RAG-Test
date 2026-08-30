"""Command-line runner for the Agent routing evaluation dataset.

The command is plan-only by default.  ``--execute`` is intentionally required
before the runner creates the real Agent or sends any LLM/web requests.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from app.evaluation.agent_route_evaluator import (
    AgentRouteDataset,
    evaluate_agent_route_dataset,
    load_agent_route_dataset,
    planned_invocation_count,
)

DEFAULT_DATASET = Path("data/evaluation/agent_routing_eval_dataset_v1.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute the Agent routing evaluation",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Routing dataset path (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Result JSON path; a timestamped path is used when omitted",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=["dev", "test"],
        dest="splits",
        help="Only run this split; may be specified more than once",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Only run this case ID; may be specified more than once",
    )
    parser.add_argument(
        "--no-boundary-repeats",
        action="store_true",
        help="Run repeat-designated boundary cases only once",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=50,
        help="LangGraph recursion limit for each invocation (default: 50)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually create the Agent and call configured LLM/web services",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help=(
            "Directory for generated chart artifacts; by default charts are "
            "saved beside the result under artifacts/<result-stem>"
        ),
    )
    return parser


def evaluation_plan(
    dataset: AgentRouteDataset,
    *,
    splits: set[str] | None,
    case_ids: set[str] | None,
    repeat_boundary_cases: bool,
) -> dict[str, object]:
    """Build the plan displayed before any external service is called."""
    selected_cases = [
        case for case in dataset.cases
        if (not splits or case.split in splits)
        and (not case_ids or case.case_id in case_ids)
    ]
    invocations = planned_invocation_count(
        dataset,
        splits=splits,
        case_ids=case_ids,
        repeat_boundary_cases=repeat_boundary_cases,
    )
    return {
        "mode": "plan-only",
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.version,
        "selected_case_count": len(selected_cases),
        "planned_agent_invocations": invocations,
        "selected_splits": sorted({case.split for case in selected_cases}),
        "case_ids": [case.case_id for case in selected_cases],
        "boundary_repeats_enabled": repeat_boundary_cases,
        "external_calls_started_at_plan_output": False,
    }


@contextmanager
def isolated_upload_fixture(
    dataset: AgentRouteDataset,
    *,
    backend_dir: Path,
) -> Iterator[Path]:
    """Stage the CSV fixture and point data tools at a temporary upload area."""
    fixture_source = (backend_dir / str(dataset.data_fixture["source_path"])).resolve()
    if not fixture_source.is_file():
        raise FileNotFoundError(f"Data fixture does not exist: {fixture_source}")

    staged_name = str(dataset.data_fixture["staged_file_name"])
    previous_uploads_path = os.environ.get("UPLOADS_PATH")
    with tempfile.TemporaryDirectory(prefix="agent-route-eval-") as temp_dir:
        uploads_path = Path(temp_dir).resolve() / "uploads"
        uploads_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fixture_source, uploads_path / staged_name)
        os.environ["UPLOADS_PATH"] = str(uploads_path)

        # Settings may already have been read by another imported module.
        from app.core.config import get_settings

        get_settings.cache_clear()
        try:
            yield uploads_path
        finally:
            if previous_uploads_path is None:
                os.environ.pop("UPLOADS_PATH", None)
            else:
                os.environ["UPLOADS_PATH"] = previous_uploads_path
            get_settings.cache_clear()


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data/evaluation/results") / f"agent_routing_eval_{timestamp}.json"


def _write_new_json(path: Path, payload: dict[str, object]) -> None:
    """Write a new result file without overwriting an earlier evaluation."""
    output_path = path.resolve()
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing evaluation result: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _persist_chart_artifacts(
    uploads_path: Path,
    artifacts_dir: Path,
) -> list[dict[str, object]]:
    """Copy generated charts out of the temporary upload fixture."""
    charts_dir = uploads_path / "charts"
    if not charts_dir.is_dir():
        return []

    chart_paths = sorted(path for path in charts_dir.iterdir() if path.is_file())
    if not chart_paths:
        return []

    destination_dir = artifacts_dir.resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    persisted = []
    for source in chart_paths:
        destination = destination_dir / source.name
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing evaluation artifact: {destination}"
            )
        shutil.copy2(source, destination)
        persisted.append({
            "source_url": f"/static/charts/{source.name}",
            "path": str(destination),
            "sha256": _sha256(destination),
            "size_bytes": destination.stat().st_size,
        })
    return persisted


async def _execute(args: argparse.Namespace, dataset: AgentRouteDataset) -> int:
    backend_dir = Path.cwd().resolve()
    repeat_boundary_cases = not args.no_boundary_repeats
    splits = set(args.splits) if args.splits else None
    case_ids = set(args.case_ids) if args.case_ids else None

    with isolated_upload_fixture(dataset, backend_dir=backend_dir) as uploads_path:
        # Import only after the isolated UPLOADS_PATH is active.  This is the
        # first point at which model configuration and real tools are created.
        from app.agents.main_agent import get_main_agent

        agent = get_main_agent()
        result = await evaluate_agent_route_dataset(
            agent,
            dataset,
            splits=splits,
            case_ids=case_ids,
            repeat_boundary_cases=repeat_boundary_cases,
            recursion_limit=args.recursion_limit,
        )

        output_path = args.output or _default_output_path()
        artifacts_dir = args.artifacts_dir or (
            output_path.parent / "artifacts" / output_path.stem
        )
        persisted_artifacts = _persist_chart_artifacts(
            uploads_path,
            artifacts_dir,
        )
        payload = result.to_dict()
        payload["execution"] = {
            "isolated_uploads": True,
            "staged_fixture_name": dataset.data_fixture.get("staged_file_name"),
            "temporary_uploads_path_removed_after_run": True,
            "persisted_artifacts": persisted_artifacts,
        }
        _write_new_json(output_path, payload)
        print(result.summary())
        print(f"Result: {output_path.resolve()}")
        if persisted_artifacts:
            print(f"Persisted artifacts: {artifacts_dir.resolve()}")
        print(f"Temporary uploads: {uploads_path} (removed after this command)")
    return 2 if result.termination is not None else 0


def main() -> int:
    args = build_parser().parse_args()
    dataset = load_agent_route_dataset(args.dataset)
    splits = set(args.splits) if args.splits else None
    case_ids = set(args.case_ids) if args.case_ids else None
    repeat_boundary_cases = not args.no_boundary_repeats

    plan = evaluation_plan(
        dataset,
        splits=splits,
        case_ids=case_ids,
        repeat_boundary_cases=repeat_boundary_cases,
    )
    plan["mode"] = "execute" if args.execute else "plan-only"
    plan["will_call_real_services"] = bool(args.execute)
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if not args.execute:
        print("Plan only: no Agent, LLM, RAG, data, or web tool was invoked.")
        print("Add --execute only after the real API run is approved.")
        return 0
    return asyncio.run(_execute(args, dataset))


if __name__ == "__main__":
    raise SystemExit(main())
