from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.teacher_metrics import evaluate_teacher  # noqa: E402
from src.ontology.gate import apply_wsr_gate  # noqa: E402


METHODS = {
    "standard_kd": "Standard KD",
    "contrastive_kd": "Contrastive KD",
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ontology(path: Path) -> dict:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        pass

    ontology = {"entity_types": {}, "allowed_relations": {}}
    section = ""
    current_entity = ""
    current_relation = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            section = line.rstrip(":")
            current_entity = ""
            current_relation = ""
            continue
        if section == "entity_types":
            if line.startswith("  ") and not line.startswith("    "):
                current_entity = line.strip().rstrip(":")
                ontology["entity_types"][current_entity] = {}
            elif current_entity and line.strip().startswith("layer:"):
                ontology["entity_types"][current_entity]["layer"] = (
                    line.split(":", 1)[1].strip()
                )
        elif section == "allowed_relations":
            if line.startswith("  ") and not line.startswith("    "):
                current_relation = line.strip().rstrip(":")
                ontology["allowed_relations"][current_relation] = []
            elif current_relation and line.strip().startswith("- ["):
                pair = line.strip().removeprefix("- [").removesuffix("]")
                ontology["allowed_relations"][current_relation].append(
                    [item.strip() for item in pair.split(",")]
                )
    return ontology

def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    prediction_dir = ROOT / "outputs/baselines_teacher_consistency_364_predictions"
    output_dir = ROOT / "outputs/table5_student_gate_predictions"
    report_dir = ROOT / "outputs/table5_student_gate_completion"
    gold_path = ROOT / "data/gold/teacher_quality_gold.manual_review_v2.frozen.jsonl"
    ontology_path = ROOT / "configs/wsr_ontology.yaml"

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    gold = read_jsonl(gold_path)
    ontology = read_ontology(ontology_path)
    reports = []

    for method, display_name in METHODS.items():
        source_path = prediction_dir / f"{method}.jsonl"
        predictions = read_jsonl(source_path)
        gated_predictions = [
            apply_wsr_gate(record, ontology)
            if record.get("parse_success", True)
            else record
            for record in predictions
        ]
        gated_path = output_dir / f"{method}_student_gate.jsonl"
        gated_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in gated_predictions
            ),
            encoding="utf-8",
        )

        report = evaluate_teacher(gated_predictions, gold, ontology)
        report.update(
            {
                "method": method,
                "display_name": display_name,
                "inference": "Student+Gate",
                "source_predictions": str(source_path.resolve()),
                "source_sha256": sha256(source_path),
                "gated_predictions": str(gated_path.resolve()),
                "gated_sha256": sha256(gated_path),
                "gold_path": str(gold_path.resolve()),
                "gold_sha256": sha256(gold_path),
                "ontology_path": str(ontology_path.resolve()),
                "ontology_sha256": sha256(ontology_path),
            }
        )
        (report_dir / f"{method}_student_gate.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        reports.append(report)

    headers = [
        "Model",
        "Inference",
        "Entity F1",
        "Relation F1",
        "Cond. Rel F1",
        "Triple-F1",
        "CVR-All",
        "Removed by Gate",
        "Pred rel before",
        "Pred rel after",
    ]
    rows = []
    for report in reports:
        removed_by_gate = 1 - report["gate_relation_retention"]
        rows.append(
            {
                "Model": report["display_name"],
                "Inference": report["inference"],
                "Entity F1": report["entity_span_type_f1"],
                "Relation F1": report["relation_f1"],
                "Cond. Rel F1": report["conditional_relation_f1"],
                "Triple-F1": report["triple_f1"],
                "CVR-All": report["cvr_all"],
                "Removed by Gate": removed_by_gate,
                "Pred rel before": report["predicted_relations_before_gate"],
                "Pred rel after": report["predicted_relations"],
            }
        )

    csv_path = report_dir / "table5_student_gate_completion.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---"] + ["---:"] * (len(headers) - 2)) + " |",
    ]
    for row in rows:
        values = [
            str(row["Model"]),
            str(row["Inference"]),
            pct(row["Entity F1"]),
            pct(row["Relation F1"]),
            pct(row["Cond. Rel F1"]),
            pct(row["Triple-F1"]),
            pct(row["CVR-All"]),
            pct(row["Removed by Gate"]),
            str(row["Pred rel before"]),
            str(row["Pred rel after"]),
        ]
        lines.append("| " + " | ".join(values) + " |")

    md_path = report_dir / "table5_student_gate_completion.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

