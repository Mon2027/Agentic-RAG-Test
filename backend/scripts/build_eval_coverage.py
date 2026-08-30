"""Build a compact evidence coverage map from topic-tagged Chroma chunks."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.shared_system_client import SharedSystemClient


SIGNALS: dict[str, tuple[str, ...]] = {
    "financial": (
        "营业收入",
        "营业总收入",
        "营收",
        "归母净利润",
        "净利润",
        "毛利率",
        "同比",
        "2025",
        "2026",
    ),
    "product": (
        "具身智能",
        "机器人",
        "机器视觉",
        "谐波减速器",
        "传感器",
        "控制器",
        "执行器",
        "关节",
        "灵巧手",
    ),
    "progress": (
        "量产",
        "送样",
        "订单",
        "客户",
        "合作",
        "签约",
        "研发",
        "落地",
    ),
    "risk": (
        "风险提示",
        "不及预期",
        "市场竞争",
        "研发风险",
        "客户集中",
        "政策风险",
    ),
}


def company_from_name(file_name: str) -> str:
    match = re.search(r"【([^】]+)】", file_name)
    return match.group(1) if match else ""


def page_value(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def signal_score(text: str, terms: tuple[str, ...]) -> tuple[int, list[str]]:
    hits = [term for term in terms if term.lower() in text.lower()]
    return len(hits), hits


def compact_snippet(text: str, limit: int = 700) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:limit]


def build_coverage(
    vectorstore_dir: Path,
    output_path: Path,
    topic: str,
) -> dict[str, Any]:
    client = chromadb.PersistentClient(path=str(vectorstore_dir))
    collection = client.get_collection("reports")
    result = collection.get(
        where={"topic": topic},
        include=["documents", "metadatas"],
    )
    documents = list(result.get("documents") or [])
    metadatas = list(result.get("metadatas") or [])
    if len(documents) != len(metadatas):
        raise RuntimeError("Chroma returned inconsistent documents and metadata")

    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for document, metadata in zip(documents, metadatas, strict=True):
        file_id = str(metadata.get("file_id") or "")
        if not file_id:
            raise ValueError("Topic chunk is missing file_id")
        grouped[file_id].append((document, metadata))

    reports: list[dict[str, Any]] = []
    for file_id, chunks in grouped.items():
        first_metadata = chunks[0][1]
        original_name = str(
            first_metadata.get("original_file_name")
            or first_metadata.get("file_name")
            or ""
        )
        pages = sorted(
            {
                page
                for _, metadata in chunks
                for page in (
                    page_value(metadata, "page_start"),
                    page_value(metadata, "page_end"),
                )
                if page is not None
            }
        )

        evidence: list[dict[str, Any]] = []
        used_chunk_indexes: set[int] = set()
        for category, terms in SIGNALS.items():
            ranked: list[tuple[int, int, str, dict[str, Any], list[str]]] = []
            for index, (document, metadata) in enumerate(chunks):
                score, hits = signal_score(document, terms)
                if score == 0:
                    continue
                page_start = page_value(metadata, "page_start") or 10_000
                ranked.append((score, -page_start, document, metadata, hits))
            if not ranked:
                continue
            score, _, document, metadata, hits = max(ranked, key=lambda item: (item[0], item[1]))
            chunk_index = int(metadata.get("chunk_index", -1))
            used_chunk_indexes.add(chunk_index)
            evidence.append(
                {
                    "category": category,
                    "signal_score": score,
                    "matched_terms": hits,
                    "chunk_index": chunk_index,
                    "content_type": metadata.get("content_type", "text"),
                    "page_start": page_value(metadata, "page_start"),
                    "page_end": page_value(metadata, "page_end"),
                    "section_title": metadata.get("section_title"),
                    "snippet": compact_snippet(document),
                }
            )

        if not evidence and chunks:
            document, metadata = chunks[0]
            evidence.append(
                {
                    "category": "fallback",
                    "signal_score": 0,
                    "matched_terms": [],
                    "chunk_index": metadata.get("chunk_index"),
                    "content_type": metadata.get("content_type", "text"),
                    "page_start": page_value(metadata, "page_start"),
                    "page_end": page_value(metadata, "page_end"),
                    "section_title": metadata.get("section_title"),
                    "snippet": compact_snippet(document),
                }
            )

        reports.append(
            {
                "file_id": file_id,
                "company": company_from_name(original_name),
                "original_file_name": original_name,
                "stored_file_name": first_metadata.get("file_name"),
                "chunk_count": len(chunks),
                "text_chunk_count": sum(
                    metadata.get("content_type", "text") == "text"
                    for _, metadata in chunks
                ),
                "table_chunk_count": sum(
                    metadata.get("content_type") == "table"
                    for _, metadata in chunks
                ),
                "page_min": pages[0] if pages else None,
                "page_max": pages[-1] if pages else None,
                "evidence_candidates": evidence,
            }
        )

    reports.sort(key=lambda item: (item["company"], item["original_file_name"]))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "topic": topic,
        "report_count": len(reports),
        "chunk_count": sum(item["chunk_count"] for item in reports),
        "reports": reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    SharedSystemClient.clear_system_cache()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectorstore-dir", type=Path, default=Path("data/vectorstore"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/embodied_intelligence_coverage.json"),
    )
    parser.add_argument("--topic", default="embodied_intelligence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_coverage(
        vectorstore_dir=args.vectorstore_dir.resolve(),
        output_path=args.output.resolve(),
        topic=args.topic,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "topic": payload["topic"],
                "report_count": payload["report_count"],
                "chunk_count": payload["chunk_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
