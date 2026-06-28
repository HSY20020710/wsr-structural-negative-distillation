from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.parse_welding_cases import write_jsonl  # noqa: E402
from src.student.data import (  # noqa: E402
    canonical_target,
    make_wsr_counterfactual_targets,
    read_jsonl,
    record_id,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_template(record: dict, split: str) -> dict:
    return {
        **record,
        "annotation_split": split,
        "annotation_status": "pending_human_annotation",
        "defect_label": "",
        "event_label": "",
        "entities": [],
        "relations": [],
        "annotator": "",
        "reviewer": "",
        "annotation_notes": "",
    }


def validate_labeled(
    records: list[dict], expected_ids: set[str], allow_subset: bool = False
) -> list[str]:
    errors = []
    ids = [record_id(record) for record in records]
    if len(ids) != len(set(ids)):
        errors.append("duplicate case_id")
    missing = sorted(expected_ids - set(ids))
    extra = sorted(set(ids) - expected_ids)
    if missing and not allow_subset:
        errors.append(f"missing ids: {missing[:5]} ({len(missing)} total)")
    if extra:
        errors.append(f"unexpected ids: {extra[:5]} ({len(extra)} total)")
    for record in records:
        target = canonical_target(record)
        if not record.get("defect_label") or not record.get("event_label"):
            errors.append(f"{record_id(record)}: missing class label")
        if len(target["entities"]) != len(record.get("entities", [])):
            errors.append(f"{record_id(record)}: invalid entity span")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and audit every input required before student training."
    )
    parser.add_argument("--student_dir", type=Path, default=ROOT / "data/student")
    parser.add_argument(
        "--ontology", type=Path, default=ROOT / "configs/wsr_ontology.yaml"
    )
    parser.add_argument("--gold_train_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_train_path = args.student_dir / "raw_train.jsonl"
    raw_dev_path = args.student_dir / "raw_dev.jsonl"
    test_path = args.student_dir / "test.jsonl"
    raw_train = read_jsonl(raw_train_path)
    raw_dev = read_jsonl(raw_dev_path)
    test = read_jsonl(test_path)
    train_ids = {record_id(record) for record in raw_train}
    dev_ids = {record_id(record) for record in raw_dev}
    test_ids = {record_id(record) for record in test}
    if train_ids & dev_ids or train_ids & test_ids or dev_ids & test_ids:
        raise ValueError("Train/dev/test leakage detected")
    if not 0 < args.gold_train_size < len(raw_train):
        raise ValueError("gold_train_size must be between 1 and train_size - 1")

    rng = random.Random(args.seed)
    gold_candidates = list(raw_train)
    rng.shuffle(gold_candidates)
    gold_candidates = sorted(
        gold_candidates[: args.gold_train_size], key=record_id
    )
    annotation_dir = args.student_dir / "annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        annotation_dir / "gold_train_to_annotate.jsonl",
        [annotation_template(record, "train") for record in gold_candidates],
    )
    write_jsonl(
        annotation_dir / "dev_to_annotate.jsonl",
        [annotation_template(record, "dev") for record in raw_dev],
    )
    write_jsonl(args.student_dir / "teacher_train_input.jsonl", raw_train)

    required = {
        "teacher_train": args.student_dir / "teacher_train_enhanced.jsonl",
        "gold_train": args.student_dir / "gold_train.jsonl",
        "dev": args.student_dir / "dev.jsonl",
        "pseudo_train": args.student_dir / "pseudo_train.jsonl",
    }
    checks = {}
    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    for name, path in required.items():
        check = {"path": str(path.relative_to(ROOT)), "exists": path.is_file()}
        if path.is_file():
            records = read_jsonl(path)
            expected = (
                train_ids
                if name in {"teacher_train", "pseudo_train"}
                else {record_id(record) for record in gold_candidates}
                if name == "gold_train"
                else dev_ids
            )
            check["records"] = len(records)
            check["errors"] = validate_labeled(
                records, expected, allow_subset=name == "pseudo_train"
            )
            if name == "pseudo_train" and not records:
                check["errors"].append("pseudo_train is empty")
            if name == "teacher_train":
                check["parse_success"] = sum(
                    record.get("parse_success", False) for record in records
                )
                counter = Counter()
                for record in records:
                    for item in make_wsr_counterfactual_targets(
                        record, ontology, random.Random(args.seed)
                    ):
                        counter[item["counterfactual_type"]] += 1
                check["counterfactuals"] = dict(sorted(counter.items()))
        checks[name] = check

    pre_kd_ready = all(
        checks[name]["exists"] and not checks[name].get("errors")
        for name in ["teacher_train", "gold_train", "dev"]
    )
    all_methods_ready = pre_kd_ready and (
        checks["pseudo_train"]["exists"]
        and not checks["pseudo_train"].get("errors")
    )
    manifest = {
        "status": "READY" if all_methods_ready else "BLOCKED",
        "pre_kd_ready": pre_kd_ready,
        "all_six_methods_ready": all_methods_ready,
        "split": {
            "train": len(raw_train),
            "human_gold_train_subset": len(gold_candidates),
            "teacher_train_expected": len(raw_train),
            "dev": len(raw_dev),
            "test": len(test),
        },
        "policy": {
            "test_is_never_training_data": True,
            "gold_is_independent_human_annotation": True,
            "teacher_train_uses_complete_wsr_gate_system": True,
            "pseudo_train_is_created_only_after_supervised_training": True,
            "counterfactuals_are_derived_structures_not_new_reports": True,
        },
        "checks": checks,
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [raw_train_path, raw_dev_path, test_path, args.ontology]
        },
    }
    (args.student_dir / "pre_distillation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    commands = f"""# 蒸馏前数据完成顺序

## 1. 生成 {len(raw_train)} 条训练集教师一阶段抽取

```bash
bash run_teacher_train_extraction.sh
```

该脚本自动完成一阶段抽取、实体二次校正、端点重映射、WSR Gate 和最终校验，
并支持中断后继续。最终输出为
`data/student/teacher_train_enhanced.jsonl`。

## 2. 人工标注

- 标注 `data/student/annotation/gold_train_to_annotate.jsonl`，审核后保存为
  `data/student/gold_train.jsonl`。
- 标注全部 `data/student/annotation/dev_to_annotate.jsonl`，审核后保存为
  `data/student/dev.jsonl`。
- 禁止参考教师预测修改这两份人工标注。

## 3. 再次审计

```bash
python scripts/prepare_distillation_inputs.py
```

`pseudo_train.jsonl` 必须在 `qwen3_supervised` 训练完成后生成：

```bash
python scripts/generate_student_pseudo.py \\
  --model outputs/student/qwen3_supervised/final \\
  --input data/student/teacher_train_input.jsonl \\
  --output data/student/pseudo_train.jsonl
```

因此它不是知识蒸馏启动前可以伪造补齐的数据。

## 4. 仅对已有标签的 Train 做实体保持增强

教师训练集完成后可生成一份包装变体：

```bash
python scripts/augment_labeled_train.py \\
  --input data/student/teacher_train_enhanced.jsonl \\
  --output data/student/teacher_train_augmented.jsonl \\
  --copies 1
```

人工 Gold 也可用同一脚本单独增强。Dev/Test 严禁增强。论文应分别报告
原始报告数和增强训练实例数。
"""
    (args.student_dir / "PRE_DISTILLATION_STEPS.md").write_text(
        commands, encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
