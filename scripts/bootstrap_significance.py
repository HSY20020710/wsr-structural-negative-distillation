from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._ontology_loader import read_ontology  # noqa: E402
from src.metrics.teacher_metrics import _triple_keys, normalize_gold_record  # noqa: E402
from src.ontology.gate import apply_wsr_gate  # noqa: E402


GOLD_PATH = ROOT / "data/gold/teacher_quality_gold.manual_review_v2.frozen.jsonl"
ONTOLOGY_PATH = ROOT / "configs/wsr_ontology.yaml"
OUTPUT_DIR = ROOT / "outputs/table3_bootstrap_significance"
ITERATIONS = 10_000
SEED = 42


PREDICTIONS = {
    "standard_kd_only": ROOT
    / "outputs/baselines_teacher_consistency_364_predictions/standard_kd.jsonl",
    "contrastive_kd_only": ROOT
    / "outputs/baselines_teacher_consistency_364_predictions/contrastive_kd.jsonl",
    "ours_only": ROOT
    / "outputs/ablations_manual_gold_v2/wo_inference_gate_predictions/ours.jsonl",
    "standard_kd_gate": ROOT
    / "outputs/table5_student_gate_predictions/standard_kd_student_gate.jsonl",
    "contrastive_kd_gate": ROOT
    / "outputs/table5_student_gate_predictions/contrastive_kd_student_gate.jsonl",
    "ours_gate": ROOT / "outputs/table6_gate_impact_analysis/ours_after_gate.jsonl",
}


COMPARISONS = [
    (
        "Ours Student-only vs Standard KD Student-only",
        "Text Triple-F1",
        "ours_only",
        "standard_kd_only",
    ),
    (
        "Ours Student-only vs Contrastive KD Student-only",
        "Text Triple-F1",
        "ours_only",
        "contrastive_kd_only",
    ),
    (
        "Ours Student+Gate vs Standard KD Student+Gate",
        "Text Triple-F1",
        "ours_gate",
        "standard_kd_gate",
    ),
    (
        "Ours Student+Gate vs Contrastive KD Student+Gate",
        "Text Triple-F1",
        "ours_gate",
        "contrastive_kd_gate",
    ),
]


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def case_id(record: dict) -> str:
    return str(record.get("case_id") or record.get("record_id"))


def prf(tp: int, pred: int, gold: int) -> float:
    precision = tp / pred if pred else 0.0
    recall = tp / gold if gold else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def overlap(left: list[tuple], right: list[tuple]) -> int:
    return sum((Counter(left) & Counter(right)).values())


def per_case_counts(predictions: list[dict], gold: list[dict]) -> dict[str, tuple[int, int, int]]:
    pred_by_id = {case_id(record): record for record in predictions}
    counts: dict[str, tuple[int, int, int]] = {}
    for raw_gold in gold:
        cid = case_id(raw_gold)
        pred = pred_by_id.get(cid)
        if not pred or not pred.get("parse_success", True):
            # Match evaluate_teacher(): records without a successful parse are
            # skipped from both numerator and denominator.
            counts[cid] = (0, 0, 0)
            continue
        gold_record = normalize_gold_record(raw_gold)
        pred_keys = _triple_keys(pred)
        gold_keys = _triple_keys(gold_record)
        counts[cid] = (overlap(pred_keys, gold_keys), len(pred_keys), len(gold_keys))
    return counts


def f1_for_sample(counts: dict[str, tuple[int, int, int]], sample_ids: list[str]) -> float:
    tp = pred = gold = 0
    for cid in sample_ids:
        c_tp, c_pred, c_gold = counts[cid]
        tp += c_tp
        pred += c_pred
        gold += c_gold
    return prf(tp, pred, gold)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    frac = index - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gold = read_jsonl(GOLD_PATH)
    ontology = read_ontology(ONTOLOGY_PATH)

    predictions: dict[str, list[dict]] = {}
    for name, path in PREDICTIONS.items():
        records = read_jsonl(path)
        if name.endswith("_gate") and "table5_student_gate_predictions" not in str(path):
            records = [
                apply_wsr_gate(record, ontology)
                if record.get("parse_success", True)
                else record
                for record in records
            ]
        predictions[name] = records

    counts = {name: per_case_counts(records, gold) for name, records in predictions.items()}
    ids = [case_id(record) for record in gold]
    rng = random.Random(SEED)

    rows = []
    for label, metric, ours_key, baseline_key in COMPARISONS:
        ours_counts = counts[ours_key]
        baseline_counts = counts[baseline_key]
        observed_ours = f1_for_sample(ours_counts, ids)
        observed_baseline = f1_for_sample(baseline_counts, ids)
        observed_diff = observed_ours - observed_baseline

        diffs = []
        for _ in range(ITERATIONS):
            sample = [ids[rng.randrange(len(ids))] for _ in ids]
            diffs.append(
                f1_for_sample(ours_counts, sample)
                - f1_for_sample(baseline_counts, sample)
            )

        ci_low = percentile(diffs, 0.025)
        ci_high = percentile(diffs, 0.975)
        p_value = (1 + sum(diff <= 0 for diff in diffs)) / (ITERATIONS + 1)
        rows.append(
            {
                "comparison": label,
                "metric": metric,
                "baseline_f1": observed_baseline,
                "ours_f1": observed_ours,
                "observed_difference": observed_diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
                "iterations": ITERATIONS,
                "seed": SEED,
            }
        )

    json_path = OUTPUT_DIR / "bootstrap_text_triple_significance.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = OUTPUT_DIR / "bootstrap_text_triple_significance.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    headers = [
        "Comparison",
        "Metric",
        "Baseline F1",
        "Ours F1",
        "Observed difference",
        "95% CI",
        "p-value",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---"] + ["---:"] * (len(headers) - 2)) + " |",
    ]
    for row in rows:
        p_text = "<0.001" if row["p_value"] < 0.001 else f"{row['p_value']:.4f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["comparison"],
                    row["metric"],
                    f"{100 * row['baseline_f1']:.2f}%",
                    f"{100 * row['ours_f1']:.2f}%",
                    f"{100 * row['observed_difference']:+.2f} pp",
                    f"[{100 * row['ci_low']:+.2f}, {100 * row['ci_high']:+.2f}] pp",
                    p_text,
                ]
            )
            + " |"
        )
    md_path = OUTPUT_DIR / "bootstrap_text_triple_significance.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

