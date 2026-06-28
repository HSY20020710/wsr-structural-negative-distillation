from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.teacher_metrics import evaluate_teacher  # noqa: E402
from src.student.data import read_jsonl  # noqa: E402


METHODS = [
    ("prompt_only", "Qwen3-1.7B prompt-only"),
    ("qwen3_supervised", "Qwen3 supervised"),
    ("self_training", "Self-training"),
    ("standard_kd", "Standard KD"),
    ("contrastive_kd", "Contrastive KD"),
    ("kd_wsr_post", "KD + WSR post-processing"),
    ("ours", "Ours"),
    ("entity", "Entity stage"),
    ("relation", "Relation stage"),
]

METRICS = [
    ("Parse success", "parse_success_rate"),
    ("Strict Entity F1", "entity_span_type_f1"),
    ("Relaxed Entity F1", "relaxed_entity_span_type_f1"),
    ("Relation F1 pre-Gate", "relation_f1_before_gate"),
    ("Relation F1", "relation_f1"),
    ("Cond. Relation F1", "conditional_relation_f1"),
    ("Triple F1", "triple_f1"),
    ("CVR-All", "cvr_all"),
    ("Gate retention", "gate_relation_retention"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all six student methods on the same fixed Test set."
    )
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        default=ROOT / "outputs/student_predictions",
    )
    parser.add_argument(
        "--test", type=Path, default=ROOT / "data/student/test.jsonl"
    )
    parser.add_argument(
        "--ontology", type=Path, default=ROOT / "configs/wsr_ontology.yaml"
    )
    parser.add_argument(
        "--reports_dir",
        type=Path,
        default=ROOT / "reports/student_experiments",
    )
    parser.add_argument(
        "--method",
        choices=[method for method, _ in METHODS],
        help="Evaluate one completed method instead of requiring all methods.",
    )
    args = parser.parse_args()

    gold = read_jsonl(args.test)
    gold_ids = [str(record["case_id"]) for record in gold]
    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    reports = []
    inputs = []
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    selected_methods = (
        [item for item in METHODS if item[0] == args.method]
        if args.method
        else METHODS
    )
    for method, display_name in selected_methods:
        path = args.predictions_dir / f"{method}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        predictions = read_jsonl(path)
        prediction_ids = [str(record["case_id"]) for record in predictions]
        if prediction_ids != gold_ids:
            raise ValueError(
                f"{method}: prediction order/membership differs from Test "
                f"({len(prediction_ids)} vs {len(gold_ids)})"
            )
        report = evaluate_teacher(predictions, gold, ontology)
        report.update(
            {
                "method": method,
                "display_name": display_name,
                "predictions_path": str(path.resolve()),
                "predictions_sha256": sha256(path),
                "test_sha256": sha256(args.test),
                "ontology_sha256": sha256(args.ontology),
            }
        )
        (args.reports_dir / f"{method}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        reports.append(report)
        inputs.append(
            {
                "method": method,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "records": len(predictions),
            }
        )

    headers = ["Method", *[label for label, _ in METRICS]]
    result_stem = (
        f"{args.method}_results" if args.method else "student_main_results"
    )
    csv_path = args.reports_dir / f"{result_stem}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for report in reports:
            writer.writerow(
                [
                    report["display_name"],
                    *[report[key] for _, key in METRICS],
                ]
            )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(METRICS)) + " |",
    ]
    for report in reports:
        values = [
            report["display_name"],
            *[f"{100 * report[key]:.2f}%" for _, key in METRICS],
        ]
        lines.append("| " + " | ".join(values) + " |")
    md_path = args.reports_dir / f"{result_stem}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "experiment": (
            f"student_single_method_{args.method}"
            if args.method
            else "student_main_experiment"
        ),
        "test": {
            "path": str(args.test.resolve()),
            "sha256": sha256(args.test),
            "records": len(gold),
        },
        "ontology": {
            "path": str(args.ontology.resolve()),
            "sha256": sha256(args.ontology),
        },
        "decoding": {
            "do_sample": False,
            "max_new_tokens": 1536,
            "thinking": False,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
        },
        "gate_methods": ["kd_wsr_post", "ours"],
        "inputs": inputs,
        "results_csv": str(csv_path.resolve()),
        "results_markdown": str(md_path.resolve()),
    }
    manifest_name = (
        f"{args.method}_experiment_manifest.json"
        if args.method
        else "experiment_manifest.json"
    )
    (args.reports_dir / manifest_name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

