"""Statically validate a retrieval-and-answer evaluation dataset."""

from __future__ import annotations

import argparse
import json
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.shared_system_client import SharedSystemClient


REQUIRED_SAMPLE_FIELDS = {
    "id",
    "query",
    "relevant_doc_ids",
    "expected_terms",
    "expected_answer",
    "evidence",
    "metadata",
}


def normalize_text(value: str) -> str:
    """Normalize case, spacing, and common full-width punctuation for matching."""
    value = value.casefold()
    value = re.sub(r"\s+", "", value)
    return value.translate(str.maketrans({"＋": "+", "％": "%", "．": "."}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/evaluation/corpus_manifest.json"),
    )
    parser.add_argument("--vectorstore-dir", type=Path, default=Path("data/vectorstore"))
    parser.add_argument("--collection", default="reports")
    parser.add_argument("--max-relevant-docs", type=int, default=5)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> dict[str, Any]:
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise ValueError("Dataset root must be a JSON array")

    report_by_id = {report["file_id"]: report for report in manifest["reports"]}
    client = chromadb.PersistentClient(path=str(args.vectorstore_dir.resolve()))
    collection = client.get_collection(args.collection)
    chroma = collection.get(include=["documents", "metadatas"])

    text_by_file: dict[str, list[str]] = defaultdict(list)
    text_by_file_page: dict[tuple[str, int], list[str]] = defaultdict(list)
    pages_by_file: dict[str, set[int]] = defaultdict(set)
    for document, metadata in zip(
        chroma.get("documents") or [],
        chroma.get("metadatas") or [],
        strict=True,
    ):
        file_id = str(metadata.get("file_id") or "")
        if not file_id:
            continue
        text_by_file[file_id].append(document)
        chunk_pages: set[int] = set()
        for key in ("page_start", "page_end"):
            page = metadata.get(key)
            if isinstance(page, int):
                chunk_pages.add(page)
            elif isinstance(page, float) and page.is_integer():
                chunk_pages.add(int(page))
        for page in chunk_pages:
            pages_by_file[file_id].add(page)
            text_by_file_page[(file_id, page)].append(document)

    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    category_counts: Counter[str] = Counter()
    answerable_count = 0

    for index, sample in enumerate(dataset, start=1):
        label = str(sample.get("id") or f"sample#{index}")
        missing = sorted(REQUIRED_SAMPLE_FIELDS - sample.keys())
        if missing:
            errors.append(f"{label}: missing fields {missing}")
            continue

        if label in seen_ids:
            errors.append(f"{label}: duplicate id")
        seen_ids.add(label)

        query = str(sample["query"]).strip()
        normalized_query = normalize_text(query)
        if not query:
            errors.append(f"{label}: empty query")
        elif normalized_query in seen_queries:
            errors.append(f"{label}: duplicate normalized query")
        seen_queries.add(normalized_query)

        metadata = sample["metadata"]
        if not isinstance(metadata, dict):
            errors.append(f"{label}: metadata must be an object")
            continue
        category = str(metadata.get("category") or "")
        category_counts[category] += 1
        answerable = metadata.get("answerable")
        if not isinstance(answerable, bool):
            errors.append(f"{label}: metadata.answerable must be boolean")
            continue

        relevant_ids = sample["relevant_doc_ids"]
        if not isinstance(relevant_ids, list):
            errors.append(f"{label}: relevant_doc_ids must be an array")
            continue
        if len(relevant_ids) != len(set(relevant_ids)):
            errors.append(f"{label}: duplicate relevant_doc_ids")
        if len(relevant_ids) > args.max_relevant_docs:
            errors.append(
                f"{label}: {len(relevant_ids)} relevant documents exceed "
                f"max {args.max_relevant_docs}"
            )
        if answerable and not relevant_ids:
            errors.append(f"{label}: answerable sample has no relevant documents")
        if not answerable and relevant_ids:
            warnings.append(f"{label}: no-answer sample has relevant documents")
        if answerable:
            answerable_count += 1

        for file_id in relevant_ids:
            try:
                uuid.UUID(file_id)
            except (ValueError, TypeError, AttributeError):
                errors.append(f"{label}: invalid UUID {file_id!r}")
                continue
            report = report_by_id.get(file_id)
            if report is None:
                errors.append(f"{label}: file_id absent from corpus manifest: {file_id}")
                continue
            expected_topic = metadata.get("topic")
            if expected_topic and report.get("topic") != expected_topic:
                errors.append(
                    f"{label}: file_id {file_id} has topic {report.get('topic')!r}, "
                    f"expected {expected_topic!r}"
                )
            if file_id not in text_by_file:
                errors.append(f"{label}: file_id absent from Chroma: {file_id}")

        evidence_ids: set[str] = set()
        for evidence in sample["evidence"]:
            file_id = evidence.get("file_id")
            evidence_ids.add(file_id)
            if file_id not in relevant_ids:
                errors.append(f"{label}: evidence file_id is not relevant: {file_id}")
            known_pages = pages_by_file.get(file_id, set())
            for page in evidence.get("pages", []):
                if not isinstance(page, int) or page < 1:
                    errors.append(f"{label}: invalid evidence page {page!r}")
                elif page not in known_pages:
                    errors.append(
                        f"{label}: page {page} absent from Chroma metadata for {file_id}"
                    )
        if answerable and set(relevant_ids) != evidence_ids:
            errors.append(f"{label}: every relevant document must have evidence")

        evidence_groups = sample.get("evidence_groups", [])
        if not isinstance(evidence_groups, list):
            errors.append(f"{label}: evidence_groups must be an array")
            evidence_groups = []
        seen_group_ids: set[str] = set()
        for group_index, group in enumerate(evidence_groups, start=1):
            if not isinstance(group, dict):
                errors.append(f"{label}: evidence group #{group_index} must be an object")
                continue
            group_id = str(group.get("id") or "").strip()
            if not group_id:
                errors.append(f"{label}: evidence group #{group_index} has no id")
            elif group_id in seen_group_ids:
                errors.append(f"{label}: duplicate evidence group id {group_id!r}")
            seen_group_ids.add(group_id)

            terms = group.get("terms", [])
            if not isinstance(terms, list) or not terms:
                errors.append(f"{label}/{group_id}: terms must be a non-empty array")
                terms = []
            elif any(not isinstance(term, str) or not term.strip() for term in terms):
                errors.append(f"{label}/{group_id}: terms must be non-empty strings")

            alternatives = group.get("alternatives", [])
            if not isinstance(alternatives, list) or not alternatives:
                errors.append(
                    f"{label}/{group_id}: alternatives must be a non-empty array"
                )
                continue

            group_locations: set[tuple[str, int]] = set()
            for alternative_index, alternative in enumerate(alternatives, start=1):
                if not isinstance(alternative, dict):
                    errors.append(
                        f"{label}/{group_id}: alternative #{alternative_index} "
                        "must be an object"
                    )
                    continue
                file_id = str(alternative.get("file_id") or "")
                if file_id not in relevant_ids:
                    errors.append(
                        f"{label}/{group_id}: alternative file_id is not relevant: "
                        f"{file_id}"
                    )
                pages = alternative.get("pages", [])
                if not isinstance(pages, list) or not pages:
                    errors.append(
                        f"{label}/{group_id}: alternative pages must be non-empty"
                    )
                    continue
                known_pages = pages_by_file.get(file_id, set())
                for page in pages:
                    if not isinstance(page, int) or page < 1:
                        errors.append(
                            f"{label}/{group_id}: invalid alternative page {page!r}"
                        )
                    elif page not in known_pages:
                        errors.append(
                            f"{label}/{group_id}: page {page} absent from Chroma "
                            f"metadata for {file_id}"
                        )
                    else:
                        group_locations.add((file_id, page))

            group_text = normalize_text(
                "\n".join(
                    text
                    for location in group_locations
                    for text in text_by_file_page.get(location, [])
                )
            )
            for term in terms:
                if isinstance(term, str) and normalize_text(term) not in group_text:
                    errors.append(
                        f"{label}/{group_id}: group term not found in alternative "
                        f"documents: {term!r}"
                    )

        corpus_text = normalize_text(
            "\n".join(
                text
                for file_id in relevant_ids
                for text in text_by_file.get(file_id, [])
            )
        )
        for term in sample["expected_terms"]:
            if normalize_text(str(term)) not in corpus_text:
                errors.append(f"{label}: expected term not found in relevant text: {term!r}")

        if answerable and not str(sample["expected_answer"]).strip():
            errors.append(f"{label}: answerable sample has empty expected_answer")
        if not answerable and sample["expected_terms"]:
            warnings.append(f"{label}: no-answer sample has expected_terms")

    SharedSystemClient.clear_system_cache()
    return {
        "dataset": str(args.dataset.resolve()),
        "samples": len(dataset),
        "answerable": answerable_count,
        "no_answer": len(dataset) - answerable_count,
        "categories": dict(sorted(category_counts.items())),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def main() -> None:
    args = parse_args()
    result = validate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
