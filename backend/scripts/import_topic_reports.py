"""Import topic-organized PDF reports into the shared Chroma collection.

The importer is intentionally resumable:

* Existing PDFs are matched by SHA-256 and keep their current file IDs.
* Existing vector records receive topic metadata without being re-embedded.
* New PDFs are copied into ``data/reports`` and indexed once.
* A corpus manifest is atomically refreshed after every successful report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chromadb.api.shared_system_client import SharedSystemClient

from app.core import get_settings
from app.rag.document_processor import DocumentProcessor
from app.rag.vectorstore import VectorStore

LOGGER = logging.getLogger("topic_report_import")
COLLECTION_NAME = "reports"
FILE_ID_PREFIX = re.compile(
    r"^(?P<file_id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})_"
)
TOPICS = {
    "低空经济": "low_altitude",
    "具身智能": "embodied_intelligence",
}


@dataclass(frozen=True)
class SourceReport:
    topic: str
    topic_label: str
    source_path: Path
    sha256: str
    size: int


@dataclass
class ManifestReport:
    file_id: str
    topic: str
    topic_label: str
    original_file_name: str
    stored_file_name: str
    sha256: str
    size: int
    chunk_count: int
    status: str
    reused_existing_file: bool


def hash_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_pdf(path: Path) -> None:
    """Reject empty files and files without the PDF signature."""
    if path.stat().st_size == 0:
        raise ValueError(f"Empty PDF: {path}")
    with path.open("rb") as stream:
        if stream.read(4) != b"%PDF":
            raise ValueError(f"Invalid PDF signature: {path}")


def scan_sources(source_root: Path) -> list[SourceReport]:
    """Scan every configured topic directory and reject duplicate content."""
    reports: list[SourceReport] = []
    hashes: dict[str, Path] = {}

    for topic_label, topic in TOPICS.items():
        topic_dir = source_root / topic_label
        if not topic_dir.is_dir():
            raise FileNotFoundError(f"Topic directory does not exist: {topic_dir}")

        topic_pdfs = sorted(topic_dir.glob("*.pdf"), key=lambda path: path.name)
        if not topic_pdfs:
            raise ValueError(f"No PDF files found in: {topic_dir}")

        for path in topic_pdfs:
            validate_pdf(path)
            sha256 = hash_file(path)
            if sha256 in hashes:
                raise ValueError(
                    f"Duplicate PDF content: {path} and {hashes[sha256]}"
                )
            hashes[sha256] = path
            reports.append(
                SourceReport(
                    topic=topic,
                    topic_label=topic_label,
                    source_path=path,
                    sha256=sha256,
                    size=path.stat().st_size,
                )
            )

    return reports


def file_id_from_stored_name(path: Path) -> str:
    """Extract and validate the UUID prefix used by the application."""
    match = FILE_ID_PREFIX.match(path.name)
    if not match:
        raise ValueError(f"Stored report has no UUID prefix: {path.name}")
    return str(uuid.UUID(match.group("file_id")))


def scan_existing_reports(reports_dir: Path) -> dict[str, tuple[str, Path]]:
    """Map content hashes to the existing application file ID and path."""
    existing: dict[str, tuple[str, Path]] = {}
    for path in sorted(reports_dir.glob("*.pdf"), key=lambda item: item.name):
        validate_pdf(path)
        sha256 = hash_file(path)
        if sha256 in existing:
            raise ValueError(
                f"Duplicate content already stored: {path} and {existing[sha256][1]}"
            )
        existing[sha256] = (file_id_from_stored_name(path), path)
    return existing


def metadata_for(report: SourceReport, corpus_version: str) -> dict[str, Any]:
    """Build stable metadata shared by every chunk from one source PDF."""
    return {
        "topic": report.topic,
        "topic_label": report.topic_label,
        "source_sha256": report.sha256,
        "original_file_name": report.source_path.name,
        "corpus_version": corpus_version,
    }


def get_file_records(collection: Any, file_id: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Read all Chroma IDs and metadata associated with one file."""
    records = collection.get(
        where={"file_id": file_id},
        include=["metadatas"],
    )
    ids = list(records.get("ids") or [])
    metadatas = list(records.get("metadatas") or [])
    if len(ids) != len(metadatas):
        raise RuntimeError(f"Incomplete Chroma metadata for file_id={file_id}")
    return ids, metadatas


def update_existing_metadata(
    collection: Any,
    file_id: str,
    extra_metadata: dict[str, Any],
) -> int:
    """Merge topic metadata into existing chunks without recomputing embeddings."""
    ids, metadatas = get_file_records(collection, file_id)
    if not ids:
        return 0
    merged = [{**metadata, **extra_metadata} for metadata in metadatas]
    collection.update(ids=ids, metadatas=merged)
    return len(ids)


def write_manifest(
    manifest_path: Path,
    source_root: Path,
    corpus_version: str,
    reports: list[ManifestReport],
) -> None:
    """Atomically write the current import state."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    topic_summary: dict[str, dict[str, int]] = {}
    for topic_label, topic in TOPICS.items():
        topic_reports = [item for item in reports if item.topic == topic]
        topic_summary[topic] = {
            "report_count": len(topic_reports),
            "chunk_count": sum(item.chunk_count for item in topic_reports),
        }

    payload = {
        "corpus_version": corpus_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "collection_name": COLLECTION_NAME,
        "topic_labels": {value: key for key, value in TOPICS.items()},
        "summary": {
            "report_count": len(reports),
            "chunk_count": sum(item.chunk_count for item in reports),
            "topics": topic_summary,
        },
        "reports": [asdict(item) for item in reports],
    }
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary_path, manifest_path)


def build_plan(
    sources: list[SourceReport],
    existing: dict[str, tuple[str, Path]],
) -> list[dict[str, Any]]:
    """Create a serializable import plan without changing project data."""
    plan: list[dict[str, Any]] = []
    for report in sources:
        existing_item = existing.get(report.sha256)
        plan.append(
            {
                "topic": report.topic,
                "topic_label": report.topic_label,
                "source_file": report.source_path.name,
                "sha256": report.sha256,
                "action": "reuse_and_tag" if existing_item else "copy_parse_embed",
                "file_id": existing_item[0] if existing_item else None,
            }
        )
    return plan


def import_reports(
    source_root: Path,
    reports_dir: Path,
    vectorstore_dir: Path,
    manifest_path: Path,
    corpus_version: str,
    dry_run: bool,
) -> None:
    """Import all configured topic reports into the shared collection."""
    sources = scan_sources(source_root)
    reports_dir.mkdir(parents=True, exist_ok=True)
    existing = scan_existing_reports(reports_dir)
    plan = build_plan(sources, existing)

    plan_summary = {
        "source_reports": len(sources),
        "reuse_and_tag": sum(item["action"] == "reuse_and_tag" for item in plan),
        "copy_parse_embed": sum(item["action"] == "copy_parse_embed" for item in plan),
        "topics": {
            topic: sum(item["topic"] == topic for item in plan)
            for topic in TOPICS.values()
        },
    }
    print(json.dumps({"summary": plan_summary, "plan": plan}, ensure_ascii=False, indent=2))
    if dry_run:
        return

    settings = get_settings()
    processor = DocumentProcessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    vector_store = VectorStore(
        collection_name=COLLECTION_NAME,
        persist_directory=vectorstore_dir,
    )
    # VectorStore owns the collection. Direct access is used only because Chroma
    # can update metadata without recalculating existing embeddings.
    collection = vector_store._collection
    manifest_reports: list[ManifestReport] = []

    try:
        for position, report in enumerate(sources, start=1):
            extra_metadata = metadata_for(report, corpus_version)
            existing_item = existing.get(report.sha256)
            reused = existing_item is not None
            copied_new_file = False

            if existing_item:
                file_id, stored_path = existing_item
            else:
                file_id = str(uuid.uuid4())
                stored_path = reports_dir / f"{file_id}_{report.source_path.name}"
                if stored_path.exists():
                    raise FileExistsError(f"Refusing to overwrite: {stored_path}")
                shutil.copy2(report.source_path, stored_path)
                copied_new_file = True
                existing[report.sha256] = (file_id, stored_path)

            LOGGER.info(
                "[%s/%s] %s: %s",
                position,
                len(sources),
                report.topic_label,
                report.source_path.name,
            )

            try:
                chunk_count = update_existing_metadata(
                    collection=collection,
                    file_id=file_id,
                    extra_metadata=extra_metadata,
                )
                if chunk_count:
                    status = "reused_and_tagged"
                else:
                    chunks = processor.process_pdf(stored_path, file_id)
                    if not chunks:
                        raise ValueError(f"PDF produced zero chunks: {stored_path}")
                    for chunk in chunks:
                        chunk.metadata.update(extra_metadata)
                    vector_store.add_document_for_file(chunks, file_id)
                    chunk_count = len(chunks)
                    status = "indexed"

                item = ManifestReport(
                    file_id=file_id,
                    topic=report.topic,
                    topic_label=report.topic_label,
                    original_file_name=report.source_path.name,
                    stored_file_name=stored_path.name,
                    sha256=report.sha256,
                    size=report.size,
                    chunk_count=chunk_count,
                    status=status,
                    reused_existing_file=reused,
                )
                manifest_reports.append(item)
                write_manifest(
                    manifest_path=manifest_path,
                    source_root=source_root,
                    corpus_version=corpus_version,
                    reports=manifest_reports,
                )
                LOGGER.info(
                    "Completed %s with %s chunks (%s)",
                    report.source_path.name,
                    chunk_count,
                    status,
                )
            except Exception:
                if copied_new_file:
                    vector_store.delete_by_file_id(file_id)
                    stored_path.unlink(missing_ok=True)
                    existing.pop(report.sha256, None)
                raise
    finally:
        SharedSystemClient.clear_system_cache()

    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(manifest_path),
                "report_count": len(manifest_reports),
                "chunk_count": sum(item.chunk_count for item in manifest_reports),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("data/reports"))
    parser.add_argument("--vectorstore-dir", type=Path, default=Path("data/vectorstore"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/evaluation/corpus_manifest.json"),
    )
    parser.add_argument("--corpus-version", default="2026-08-07")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    import_reports(
        source_root=args.source_root.resolve(),
        reports_dir=args.reports_dir.resolve(),
        vectorstore_dir=args.vectorstore_dir.resolve(),
        manifest_path=args.manifest.resolve(),
        corpus_version=args.corpus_version,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
