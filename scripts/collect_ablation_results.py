from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


VARIANTS = [
    (
        "full",
        "完整方法",
        "全部训练组件与推理 Gate",
        "ours.json",
    ),
    (
        "gold_only",
        "w/o teacher targets",
        "仅使用人工 Gold 监督",
        "qwen3_supervised.json",
    ),
    (
        "wo_representation_contrastive",
        "w/o representation contrastive",
        "移除表示级对比损失",
        "ours.json",
    ),
    (
        "wo_counterfactual_negatives",
        "w/o counterfactual negatives",
        "移除全部反事实负样本",
        "ours.json",
    ),
    (
        "wo_gate_rejected_negatives",
        "w/o Gate-rejected negatives",
        "移除 Gate 拒绝的真实困难负样本",
        "ours.json",
    ),
    (
        "random_negatives",
        "random negatives",
        "用等量随机负样本替换 WSR 反事实负样本",
        "ours.json",
    ),
    (
        "wo_structure_loss",
        "w/o structure loss",
        "移除反事实结构间隔损失",
        "ours.json",
    ),
    (
        "wo_inference_gate",
        "Ours w/o inference gate",
        "保留训练组件，推理阶段不使用 Gate",
        "ours.json",
    ),
]


def load_report(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing ablation report: {path}. Run run_student_ablations.sh first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect student ablation reports into paper-ready tables."
    )
    parser.add_argument(
        "--reports_root",
        type=Path,
        default=Path("reports/ablations_manual_gold_v2"),
    )
    args = parser.parse_args()

    rows = []
    for directory, variant, changed_component, filename in VARIANTS:
        report_path = args.reports_root / directory / filename
        report = load_report(report_path)
        rows.append(
            {
                "variant": variant,
                "changed_component": changed_component,
                "entity_f1": float(report["entity_span_type_f1"]),
                "relation_f1": float(report["relation_f1"]),
                "triple_f1": float(report["triple_f1"]),
                "conditional_relation_f1": float(
                    report["conditional_relation_f1"]
                ),
                "cvr_all": float(report["cvr_all"]),
            }
        )

    full_triple_f1 = rows[0]["triple_f1"]
    for row in rows:
        row["delta_triple_f1"] = row["triple_f1"] - full_triple_f1

    args.reports_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.reports_root / "ablation_results.csv"
    headers = [
        "Variant",
        "Changed component",
        "Entity F1",
        "Relation F1",
        "Triple F1",
        "Cond. Relation F1",
        "CVR-All",
        "Delta Triple F1",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(
                [
                    row["variant"],
                    row["changed_component"],
                    row["entity_f1"],
                    row["relation_f1"],
                    row["triple_f1"],
                    row["conditional_relation_f1"],
                    row["cvr_all"],
                    row["delta_triple_f1"],
                ]
            )

    md_path = args.reports_root / "ablation_results.md"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---"] + ["---:"] * 6) + " |",
    ]
    for row in rows:
        values = [
            row["variant"],
            row["changed_component"],
            f'{100 * row["entity_f1"]:.2f}%',
            f'{100 * row["relation_f1"]:.2f}%',
            f'{100 * row["triple_f1"]:.2f}%',
            f'{100 * row["conditional_relation_f1"]:.2f}%',
            f'{100 * row["cvr_all"]:.2f}%',
            f'{100 * row["delta_triple_f1"]:+.2f}',
        ]
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(md_path.read_text(encoding="utf-8"))
    print(f"Markdown: {md_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
