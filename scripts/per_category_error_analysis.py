from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._ontology_loader import read_ontology  # noqa: E402
from src.metrics.teacher_metrics import evaluate_teacher  # noqa: E402


GOLD_PATH = ROOT / "data/gold/teacher_quality_gold.manual_review_v2.frozen.jsonl"
PRED_PATH = ROOT / "outputs/table6_gate_impact_analysis/ours_after_gate.jsonl"
ONTOLOGY_PATH = ROOT / "configs/wsr_ontology.yaml"
OUTPUT_DIR = ROOT / "outputs/table8_per_category_analysis"


CATEGORY_GROUPS = [
    {
        "name": "Porosity / gas pore",
        "field": "defect_label",
        "labels": {"气孔"},
        "error": "Carrier or location boundary errors",
    },
    {
        "name": "Slag inclusion",
        "field": "defect_label",
        "labels": {"夹渣"},
        "error": "Confusion with residue or process cause",
    },
    {
        "name": "Crack / incomplete welding / rare defects",
        "field": "defect_label",
        "labels": {"裂纹", "漏焊", "变形", "漏装"},
        "error": "Low support and implicit defect description",
    },
    {
        "name": "Inspection / rework event",
        "field": "event_label",
        "labels": {"检测返修问题"},
        "error": "Missing inspection-rework target",
    },
    {
        "name": "Responsibility-attribution event",
        "field": "event_label",
        "labels": {"责任管理问题", "人员能力不足"},
        "error": "Implicit or omitted responsible subject",
    },
]


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def record_id(record: dict) -> str:
    return str(record.get("case_id") or record.get("record_id"))


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gold = read_jsonl(GOLD_PATH)
    predictions = read_jsonl(PRED_PATH)
    ontology = read_ontology(ONTOLOGY_PATH)
    pred_by_id = {record_id(record): record for record in predictions}

    rows = []
    for group in CATEGORY_GROUPS:
        group_gold = [
            record
            for record in gold
            if record.get(group["field"]) in group["labels"]
        ]
        group_ids = {record_id(record) for record in group_gold}
        group_predictions = [
            pred_by_id[cid] for cid in group_ids if cid in pred_by_id
        ]
        report = evaluate_teacher(group_predictions, group_gold, ontology)
        rows.append(
            {
                "Category group": group["name"],
                "Gold labels": ", ".join(sorted(group["labels"])),
                "Support": len(group_gold),
                "Entity F1": report["entity_span_type_f1"],
                "Relation F1": report["relation_f1"],
                "Text Triple-F1": report["triple_f1"],
                "Main error pattern": group["error"],
            }
        )

    json_path = OUTPUT_DIR / "table8_per_category_analysis.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = OUTPUT_DIR / "table8_per_category_analysis.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    headers = [
        "Category group",
        "Support",
        "Entity F1",
        "Relation F1",
        "Text Triple-F1",
        "Main error pattern",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---:"] + ["---:"] * 3 + ["---"]) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Category group"]),
                    str(row["Support"]),
                    pct(row["Entity F1"]),
                    pct(row["Relation F1"]),
                    pct(row["Text Triple-F1"]),
                    str(row["Main error pattern"]),
                ]
            )
            + " |"
        )
    md_path = OUTPUT_DIR / "table8_per_category_analysis.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
