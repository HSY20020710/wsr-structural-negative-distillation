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

from src.gold_validation import validate_gold_record  # noqa: E402
from src.metrics.teacher_metrics import evaluate_teacher  # noqa: E402


ENHANCED_COMPONENTS = [
    "wsr_ontology",
    "entity_refinement_second_pass",
    "entity_consensus_filter",
    "relation_endpoint_remapping",
    "wsr_gate",
]

SETTINGS = [
    {
        "key": "without_ontology",
        "name": "Qwen3.6-27B（无本体）",
        "default_input": "outputs/teacher/qwen3_6_27b_without_ontology_new_relations.jsonl",
        "report": "teacher_without_ontology_consistency_85.json",
    },
    {
        "key": "with_wsr_ontology",
        "name": "Qwen3.6-27B + WSR本体",
        "default_input": "outputs/teacher/qwen3_6_27b_with_wsr_new_relations.jsonl",
        "report": "teacher_with_wsr_consistency_85.json",
    },
    {
        "key": "with_wsr_ontology_gate_enhanced",
        "name": "Qwen3.6-27B + WSR本体 + Gate（完整系统）",
        "default_input": "outputs/teacher/qwen3_6_27b_with_wsr_gate_enhanced_final.jsonl",
        "report": "teacher_with_wsr_gate_enhanced_consistency_85.json",
    },
]

COMPARISON_COLUMNS = [
    ("Teacher setting", None),
    ("Parse success", "parse_success_rate"),
    ("Entity F1", "entity_span_type_f1"),
    ("Relation F1", "relation_f1"),
    ("Cond. Relation F1", "conditional_relation_f1"),
    ("Triple F1", "triple_f1"),
    ("CVR-All", "cvr_all"),
]


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_records(records: list[dict], label: str) -> list[str]:
    case_ids = [record.get("case_id") or record.get("record_id") for record in records]
    if any(not case_id for case_id in case_ids):
        raise ValueError(f"{label}: every record must have a case_id")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"{label}: duplicate case_id values found")
    return case_ids


def standardize_enhanced_cache(records: list[dict], source: Path) -> list[dict]:
    standardized = []
    for record in records:
        if "entities" not in record or "relations" not in record:
            raise ValueError(
                f"{record.get('case_id')}: enhanced cache lacks entities or relations"
            )
        standardized.append(
            {
                **record,
                "mode": "with_wsr_ontology_gate_enhanced",
                "pipeline_components": ENHANCED_COMPONENTS,
                "result_provenance": {
                    "source_prediction_file": str(source.resolve()),
                    "teacher_recalled": False,
                    "gold_used_during_prediction": False,
                    "note": (
                        "Frozen second-pass entity-refinement cache evaluated through "
                        "the declared complete WSR+Gate system."
                    ),
                },
            }
        )
    return standardized


def validate_gold(records: list[dict]) -> None:
    invalid = []
    for record in records:
        errors = validate_gold_record(record)
        if errors:
            invalid.append(
                {
                    "case_id": record.get("case_id") or record.get("record_id"),
                    "errors": errors,
                }
            )
    if invalid:
        raise ValueError(f"Gold validation failed: {invalid[0]}")
    unready = [
        record.get("case_id") or record.get("record_id")
        for record in records
        if record.get("annotation", {}).get("status")
        not in {"reviewed", "adjudicated"}
    ]
    if unready:
        raise ValueError(f"Gold has {len(unready)} unreviewed records")


def write_comparison(
    reports: list[tuple[str, dict]], output_csv: Path, output_md: Path
) -> None:
    headers = [label for label, _ in COMPARISON_COLUMNS]
    rows = []
    for name, report in reports:
        row = {"Teacher setting": name}
        for label, key in COMPARISON_COLUMNS[1:]:
            row[label] = report[key]
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for row in rows:
        values = [
            row[header]
            if isinstance(row[header], str)
            else f"{100 * row[header]:.2f}%"
            for header in headers
        ]
        lines.append("| " + " | ".join(values) + " |")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run the complete teacher comparison from frozen Qwen caches. "
            "This command never calls the teacher API."
        )
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "data/gold/private_evaluation.jsonl",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=ROOT / "configs/wsr_ontology.yaml",
    )
    parser.add_argument("--without_ontology", type=Path)
    parser.add_argument("--with_wsr", type=Path)
    parser.add_argument("--with_wsr_gate_complete", type=Path)
    parser.add_argument(
        "--enhanced_output",
        type=Path,
        default=ROOT
        / "outputs/teacher/qwen3_6_27b_with_wsr_gate_enhanced_final.jsonl",
    )
    parser.add_argument(
        "--reports_dir", type=Path, default=ROOT / "reports/frozen_teacher_experiment"
    )
    args = parser.parse_args()

    overrides = [
        args.without_ontology,
        args.with_wsr,
        args.with_wsr_gate_complete,
    ]
    inputs = [
        override or ROOT / setting["default_input"]
        for setting, override in zip(SETTINGS, overrides)
    ]
    for path in [args.gold, args.ontology, *inputs]:
        if not path.exists():
            raise FileNotFoundError(path)

    gold = read_jsonl(args.gold)
    gold_ids = validate_records(gold, "gold")
    validate_gold(gold)
    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))

    reports = []
    manifest_inputs = []
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    for setting, input_path in zip(SETTINGS, inputs):
        predictions = read_jsonl(input_path)
        prediction_ids = validate_records(predictions, setting["key"])
        if prediction_ids != gold_ids:
            raise ValueError(
                f"{setting['key']}: case order or membership differs from Gold"
            )
        if setting["key"] == "with_wsr_ontology_gate_enhanced":
            predictions = standardize_enhanced_cache(predictions, input_path)
            if input_path.resolve() != args.enhanced_output.resolve():
                write_jsonl(args.enhanced_output, predictions)

        report = evaluate_teacher(predictions, gold, ontology)
        report.update(
            {
                "predictions_path": str(
                    (
                        args.enhanced_output
                        if setting["key"] == "with_wsr_ontology_gate_enhanced"
                        else input_path
                    ).resolve()
                ),
                "gold_path": str(args.gold.resolve()),
                "teacher_mode": setting["key"],
                "model": predictions[0].get("model", "qwen3.6-27b"),
                "evaluation_provenance": {
                    "teacher_api_called": False,
                    "input_cache_sha256": sha256(input_path),
                    "gold_sha256": sha256(args.gold),
                    "ontology_sha256": sha256(args.ontology),
                },
            }
        )
        report_path = args.reports_dir / setting["report"]
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        reports.append((setting["name"], report))
        manifest_inputs.append(
            {
                "setting": setting["key"],
                "source": str(input_path.resolve()),
                "sha256": sha256(input_path),
                "records": len(predictions),
                "report": str(report_path.resolve()),
            }
        )

    comparison_csv = args.reports_dir / "teacher_quality_comparison.csv"
    comparison_md = args.reports_dir / "teacher_quality_comparison.md"
    write_comparison(reports, comparison_csv, comparison_md)
    manifest = {
        "experiment": "frozen_teacher_quality_comparison",
        "teacher_api_called": False,
        "gold": {
            "path": str(args.gold.resolve()),
            "sha256": sha256(args.gold),
            "records": len(gold),
        },
        "ontology": {
            "path": str(args.ontology.resolve()),
            "sha256": sha256(args.ontology),
        },
        "enhanced_gate_components": ENHANCED_COMPONENTS,
        "inputs": manifest_inputs,
        "comparison_csv": str(comparison_csv.resolve()),
        "comparison_md": str(comparison_md.resolve()),
    }
    (args.reports_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(comparison_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
