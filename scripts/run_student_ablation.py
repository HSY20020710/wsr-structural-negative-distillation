from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._ontology_loader import read_ontology  # noqa: E402
from src.metrics.teacher_metrics import evaluate_teacher  # noqa: E402


VARIANTS = [
    {
        "variant": "Full training",
        "teacher_target": "Yes",
        "rep_contrast": "Yes",
        "cf_negatives": "Yes",
        "gate_rejected_neg": "Yes",
        "ranking_loss": "Yes",
        "predictions": ROOT
        / "outputs/ablations_manual_gold_v2/wo_inference_gate_predictions/ours.jsonl",
        "restore_before_gate": False,
    },
    {
        "variant": "w/o teacher targets",
        "teacher_target": "No",
        "rep_contrast": "Yes",
        "cf_negatives": "Yes",
        "gate_rejected_neg": "Yes",
        "ranking_loss": "Yes",
        "predictions": ROOT
        / "outputs/ablations_manual_gold_v2/gold_only_predictions/qwen3_supervised.jsonl",
        "restore_before_gate": False,
    },
    {
        "variant": "w/o representation contrastive",
        "teacher_target": "Yes",
        "rep_contrast": "No",
        "cf_negatives": "Yes",
        "gate_rejected_neg": "Yes",
        "ranking_loss": "Yes",
        "predictions": ROOT
        / "outputs/ablations_manual_gold_v2/wo_representation_contrastive_predictions/ours.jsonl",
        "restore_before_gate": True,
    },
    {
        "variant": "w/o counterfactual negatives",
        "teacher_target": "Yes",
        "rep_contrast": "Yes",
        "cf_negatives": "No",
        "gate_rejected_neg": "Yes",
        "ranking_loss": "Yes",
        "predictions": ROOT
        / "outputs/ablations_manual_gold_v2/wo_counterfactual_negatives_predictions/ours.jsonl",
        "restore_before_gate": True,
    },
    {
        "variant": "w/o Gate-rejected negatives",
        "teacher_target": "Yes",
        "rep_contrast": "Yes",
        "cf_negatives": "Yes",
        "gate_rejected_neg": "No",
        "ranking_loss": "Yes",
        "predictions": ROOT
        / "outputs/ablations_manual_gold_v2/wo_gate_rejected_negatives_predictions/ours.jsonl",
        "restore_before_gate": True,
    },
    {
        "variant": "random negatives",
        "teacher_target": "Yes",
        "rep_contrast": "Yes",
        "cf_negatives": "Random",
        "gate_rejected_neg": "Random",
        "ranking_loss": "Yes",
        "predictions": ROOT
        / "outputs/ablations_manual_gold_v2/random_negatives_predictions/ours.jsonl",
        "restore_before_gate": True,
    },
    {
        "variant": "w/o structure ranking loss",
        "teacher_target": "Yes",
        "rep_contrast": "Yes",
        "cf_negatives": "Yes",
        "gate_rejected_neg": "Yes",
        "ranking_loss": "No",
        "predictions": ROOT
        / "outputs/ablations_manual_gold_v2/wo_structure_loss_predictions/ours.jsonl",
        "restore_before_gate": True,
    },
]


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def triples_from_relations(record: dict) -> list[dict]:
    entities = {entity["id"]: entity for entity in record.get("entities", [])}
    triples = []
    for relation in record.get("relations", []):
        head = entities.get(relation.get("head"))
        tail = entities.get(relation.get("tail"))
        if head and tail:
            triples.append(
                {
                    "head_text": head.get("text"),
                    "relation": relation.get("type"),
                    "tail_text": tail.get("text"),
                    "confidence": relation.get("confidence", 0.0),
                }
            )
    return triples


def restore_student_only(records: list[dict]) -> list[dict]:
    restored = []
    for record in records:
        clean = dict(record)
        if "relations_before_gate" in clean:
            clean["relations"] = list(clean.get("relations_before_gate") or [])
            clean["triples"] = triples_from_relations(clean)
            clean.pop("rejected_relations", None)
            clean.pop("gate_stats", None)
        restored.append(clean)
    return restored


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    gold_path = ROOT / "data/gold/teacher_quality_gold.manual_review_v2.frozen.jsonl"
    ontology_path = ROOT / "configs/wsr_ontology.yaml"
    output_dir = ROOT / "outputs/table7_student_only_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    gold = read_jsonl(gold_path)
    ontology = read_ontology(ontology_path)
    rows = []

    for spec in VARIANTS:
        records = read_jsonl(spec["predictions"])
        if spec["restore_before_gate"]:
            records = restore_student_only(records)
        prediction_out = output_dir / (
            spec["variant"]
            .replace("/", "without")
            .replace(" ", "_")
            .replace("-", "_")
            + ".jsonl"
        )
        prediction_out.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        report = evaluate_teacher(records, gold, ontology)
        row = {
            **{key: spec[key] for key in [
                "variant",
                "teacher_target",
                "rep_contrast",
                "cf_negatives",
                "gate_rejected_neg",
                "ranking_loss",
            ]},
            "triple_f1": report["triple_f1"],
            "conditional_relation_f1": report["conditional_relation_f1"],
            "cvr_all": report["cvr_all"],
            "entity_f1": report["entity_span_type_f1"],
            "relation_f1": report["relation_f1"],
            "prediction_source": str(spec["predictions"].resolve()),
            "student_only_predictions": str(prediction_out.resolve()),
        }
        rows.append(row)
        (output_dir / f"{prediction_out.stem}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    full_triple = rows[0]["triple_f1"]
    headers = [
        "Variant",
        "Teacher target",
        "Rep. contrast",
        "CF negatives",
        "Gate-rejected neg.",
        "Ranking loss",
        "Triple-F1",
        "Cond. Rel F1",
        "CVR-All",
        "Delta Triple-F1",
    ]
    csv_path = output_dir / "table7_student_only_ablation.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(
                [
                    row["variant"],
                    row["teacher_target"],
                    row["rep_contrast"],
                    row["cf_negatives"],
                    row["gate_rejected_neg"],
                    row["ranking_loss"],
                    row["triple_f1"],
                    row["conditional_relation_f1"],
                    row["cvr_all"],
                    row["triple_f1"] - full_triple,
                ]
            )

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * 6 + ["---:"] * 4) + " |",
    ]
    for row in rows:
        delta = row["triple_f1"] - full_triple
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    row["teacher_target"],
                    row["rep_contrast"],
                    row["cf_negatives"],
                    row["gate_rejected_neg"],
                    row["ranking_loss"],
                    pct(row["triple_f1"]),
                    pct(row["conditional_relation_f1"]),
                    pct(row["cvr_all"]),
                    f"{100 * delta:+.2f}",
                ]
            )
            + " |"
        )
    md_path = output_dir / "table7_student_only_ablation.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
