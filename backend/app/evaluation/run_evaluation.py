"""CLI script for running RAG evaluation."""

import argparse
import json
import logging
from pathlib import Path

from app.evaluation.rag_evaluator import (
    EvaluationSample,
    RAGEvaluator,
    create_evaluation_dataset_from_reports,
    create_sample_evaluation_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_evaluation(
    dataset_path: Path,
    output_path: Path | None = None,
    k_values: list[int] | None = None,
    top_k: int = 20,
    topic: str | None = None,
) -> None:
    """Run evaluation on a dataset.

    Args:
        dataset_path: Path to evaluation dataset JSON file
        output_path: Optional path to save results
        k_values: List of K values for metrics
        top_k: Number of documents to retrieve per query
        topic: Optional corpus topic metadata filter
    """
    logger.info(f"Loading evaluation dataset from {dataset_path}")

    evaluator = RAGEvaluator(k_values, topic=topic)
    result = evaluator.evaluate_from_file(dataset_path, top_k)

    # Print summary
    print(result.summary())

    # Save results if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to {output_path}")


def quick_eval(queries: list[str], doc_ids: list[str]) -> None:
    """Quick evaluation with inline queries.

    Args:
        queries: List of test queries
        doc_ids: List of relevant document IDs for all queries
    """
    samples = [
        EvaluationSample(
            query=query,
            relevant_doc_ids=set(doc_ids),
        )
        for query in queries
    ]

    evaluator = RAGEvaluator()
    result = evaluator.evaluate(samples)

    print(result.summary())


def main():
    parser = argparse.ArgumentParser(description="RAG Retrieval Evaluation")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument(
        "dataset",
        type=Path,
        help="Path to evaluation dataset JSON file",
    )
    eval_parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Path to save evaluation results",
    )
    eval_parser.add_argument(
        "-k", "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="K values for Recall@K and Precision@K",
    )
    eval_parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of documents to retrieve per query",
    )
    eval_parser.add_argument(
        "--topic",
        help="Optional corpus topic metadata filter",
    )

    # Create sample dataset command
    create_parser = subparsers.add_parser(
        "create-sample",
        help="Create sample evaluation dataset"
    )
    create_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("evaluation_dataset.json"),
        help="Output path for sample dataset",
    )

    # Create dataset from report PDFs command
    report_parser = subparsers.add_parser(
        "create-from-reports",
        help="Create an evaluation dataset from local report PDF filenames",
    )
    report_parser.add_argument(
        "reports_dir",
        type=Path,
        help="Directory containing report PDF files",
    )
    report_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("data/evaluation/report_eval_dataset.json"),
        help="Output path for generated dataset",
    )
    report_parser.add_argument(
        "--min-file-size",
        type=int,
        default=1024,
        help="Skip tiny placeholder PDFs below this size in bytes",
    )

    # Quick test command
    quick_parser = subparsers.add_parser("quick", help="Quick evaluation test")
    quick_parser.add_argument(
        "-q", "--query",
        action="append",
        dest="queries",
        required=True,
        help="Test query (can be specified multiple times)",
    )
    quick_parser.add_argument(
        "-d", "--doc-id",
        action="append",
        dest="doc_ids",
        required=True,
        help="Relevant document ID (can be specified multiple times)",
    )

    args = parser.parse_args()

    if args.command == "eval":
        run_evaluation(
            dataset_path=args.dataset,
            output_path=args.output,
            k_values=args.k_values,
            top_k=args.top_k,
            topic=args.topic,
        )
    elif args.command == "create-sample":
        create_sample_evaluation_dataset(args.output)
        print(f"Sample dataset created at {args.output}")
    elif args.command == "create-from-reports":
        samples = create_evaluation_dataset_from_reports(
            reports_dir=args.reports_dir,
            output_path=args.output,
            min_file_size=args.min_file_size,
        )
        print(f"Created {len(samples)} report evaluation samples at {args.output}")
    elif args.command == "quick":
        quick_eval(args.queries, args.doc_ids)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
