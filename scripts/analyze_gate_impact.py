from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._ontology_loader import read_ontology  # noqa: E402
from src.metrics.teacher_metrics import evaluate_teacher, normalize_gold_record  # noqa: E402
from src.ontology.gate import apply_wsr_gate  # noqa: E402


METHODS = {
    "standard_kd": {
        "display": "Standard KD",
        "before": ROOT
        / "outputs/baselines_teacher_consistency_364_predictions/standard_kd.jsonl",
    },
    "contrastive_kd": {
        "display": "Contrastive KD",
        "before": ROOT
        / "outputs/baselines_teacher_consistency_364_predictions/contrastive_kd.jsonl",
    },
    "ours": {
        "display": "Ours",
        "before": ROOT
        / "outputs/ablations_manual_gold_v2/wo_inference_gate_predictions/ours.jsonl",
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def relation_keys(record: dict) -> list[tuple]:
    entities = {entity["id"]: entity for entity in record.get("entities", [])}
    keys = []
    for relation in record.get("relations", []):
        head = entities.get(relation.get("head"))
        tail = entities.get(relation.get("tail"))
        if head and tail:
            keys.append(
                (
                    head.get("start"),
                    head.get("end"),
                    head.get("type"),
                    relation.get("type"),
                    tail.get("start"),
                    tail.get("end"),
                    tail.get("type"),
                )
            )
    return keys


def overlap(left: list[tuple], right: list[tuple]) -> int:
    return sum((Counter(left) & Counter(right)).values())


def gate_deletion_stats(before: list[dict], after: list[dict], gold: list[dict]) -> dict:
    gold_by_id = {
        record.get("case_id") or record.get("record_id"): normalize_gold_record(record)
        for record in gold
    }
    after_by_id = {record.get("case_id") or record.get("record_id"): record for record in after}
    before_total = 0
    after_total = 0
    removed_total = 0
    before_correct = 0
    before_wrong = 0
    removed_correct = 0
    removed_wrong = 0

    for before_record in before:
        if not before_record.get("parse_success"):
            continue
        case_id = before_record.get("case_id") or before_record.get("record_id")
        gold_record = gold_by_id.get(case_id)
        after_record = after_by_id.get(case_id)
        if not gold_record or not after_record:
            continue

        before_keys = relation_keys(before_record)
        after_keys = relation_keys(after_record)
        gold_keys = relation_keys(gold_record)

        before_counter = Counter(before_keys)
        after_counter = Counter(after_keys)
        gold_counter = Counter(gold_keys)
        removed_counter = before_counter - after_counter

        before_total += sum(before_counter.values())
        after_total += sum(after_counter.values())
        removed_total += sum(removed_counter.values())

        correct_counter = before_counter & gold_counter
        removed_correct_counter = removed_counter & gold_counter
        before_correct += sum(correct_counter.values())
        removed_correct += sum(removed_correct_counter.values())

        # Count a removed relation as wrong when it does not consume a matching gold relation.
        removed_wrong += sum((removed_counter - gold_counter).values())
        before_wrong += sum((before_counter - gold_counter).values())

    return {
        "before_relations": before_total,
        "after_relations": after_total,
        "removed_relations": removed_total,
        "relation_kept": after_total / before_total if before_total else 0.0,
        "wrong_removed": removed_wrong / before_wrong if before_wrong else 0.0,
        "correct_wrongly_removed": (
            removed_correct / before_correct if before_correct else 0.0
        ),
        "removed_wrong_count": removed_wrong,
        "before_wrong_count": before_wrong,
        "removed_correct_count": removed_correct,
        "before_correct_count": before_correct,
    }


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    gold_path = ROOT / "data/gold/teacher_quality_gold.manual_review_v2.frozen.jsonl"
    ontology_path = ROOT / "configs/wsr_ontology.yaml"
    output_dir = ROOT / "outputs/table6_gate_impact_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    gold = read_jsonl(gold_path)
    ontology = read_ontology(ontology_path)
    rows = []

    for method, spec in METHODS.items():
        before_records = read_jsonl(spec["before"])
        after_records = [
            apply_wsr_gate(record, ontology)
            if record.get("parse_success", True)
            else record
            for record in before_records
        ]
        after_path = output_dir / f"{method}_after_gate.jsonl"
        after_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in after_records
            ),
            encoding="utf-8",
        )

        before_report = evaluate_teacher(before_records, gold, ontology)
        after_report = evaluate_teacher(after_records, gold, ontology)
        deletion = gate_deletion_stats(before_records, after_records, gold)
        row = {
            "model": spec["display"],
            "before_f1": before_report["triple_f1"],
            "after_f1": after_report["triple_f1"],
            "delta_f1": after_report["triple_f1"] - before_report["triple_f1"],
            "before_cvr": before_report["cvr_all"],
            "after_cvr": after_report["cvr_all"],
            **deletion,
        }
        rows.append(row)
        (output_dir / f"{method}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    headers = [
        "Model",
        "Before Gate F1",
        "After Gate F1",
        "Delta F1",
        "Before CVR",
        "After CVR",
        "Relation kept",
        "Wrong removed",
        "Correct wrongly removed",
    ]
    csv_path = output_dir / "table6_gate_impact_analysis.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(
                [
                    row["model"],
                    row["before_f1"],
                    row["after_f1"],
                    row["delta_f1"],
                    row["before_cvr"],
                    row["after_cvr"],
                    row["relation_kept"],
                    row["wrong_removed"],
                    row["correct_wrongly_removed"],
                ]
            )

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for row in rows:
        delta = row["delta_f1"]
        values = [
            row["model"],
            pct(row["before_f1"]),
            pct(row["after_f1"]),
            f"{delta * 100:+.2f}",
            pct(row["before_cvr"]),
            pct(row["after_cvr"]),
            pct(row["relation_kept"]),
            pct(row["wrong_removed"]),
            pct(row["correct_wrongly_removed"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    md_path = output_dir / "table6_gate_impact_analysis.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
