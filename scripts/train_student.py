from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.student.data import (  # noqa: E402
    SUPPORTED_METHODS,
    SUPPORTED_STAGES,
    SUPPORTED_TARGET_SCHEMAS,
    build_training_records,
    calibrate_teacher_against_gold,
    read_jsonl,
    validate_enhanced_teacher_records,
)
from src.student.trainer import LossConfig, build_runtime_classes  # noqa: E402


def load_optional(path: Path | None) -> list[dict]:
    return read_jsonl(path) if path else []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one Qwen3 student baseline or the proposed WSR-KD method."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(SUPPORTED_METHODS), required=True)
    parser.add_argument(
        "--stage", choices=sorted(SUPPORTED_STAGES), default="joint"
    )
    parser.add_argument(
        "--target_schema",
        choices=sorted(SUPPORTED_TARGET_SCHEMAS),
        default="full",
    )
    parser.add_argument("--gold_train", type=Path)
    parser.add_argument("--teacher_train", type=Path)
    parser.add_argument("--pseudo_train", type=Path)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=ROOT / "configs/wsr_ontology.yaml")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--student_model")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        help="Override per-device train batch size from the config.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        help="Override per-device evaluation batch size from the config.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        help="Override gradient accumulation from the config.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Override learning rate from the config.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        help="Override the number of training epochs from the config.",
    )
    parser.add_argument("--resume_from_checkpoint", nargs="?", const=True)
    parser.add_argument(
        "--require_enhanced_teacher",
        action="store_true",
        help="Reject teacher data not produced by the complete five-stage system.",
    )
    parser.add_argument(
        "--calibrate_teacher_with_gold",
        action="store_true",
        help="Estimate teacher sample weight from strict agreement with Gold.",
    )
    parser.add_argument("--teacher_weight", type=float)
    parser.add_argument("--gold_weight", type=float, default=2.0)
    parser.add_argument("--gold_repeats", type=int, default=3)
    parser.add_argument(
        "--negative_policy",
        choices=["full", "none", "synthetic_only", "random"],
        default="full",
        help="Counterfactual source used by Ours.",
    )
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        help="Override representation contrastive-loss weight.",
    )
    parser.add_argument(
        "--structure_weight",
        type=float,
        help="Override counterfactual structure-margin-loss weight.",
    )
    parser.add_argument(
        "--teacher_aligned",
        action="store_true",
        help=(
            "Train Ours only on complete enhanced teacher targets. Gold is "
            "used for an agreement report but never replaces teacher labels."
        ),
    )
    parser.add_argument(
        "--teacher_aligned_validation_ratio",
        type=float,
        default=0.05,
        help="Fraction of teacher records reserved for teacher-style model selection.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate and summarize data without GPU training libraries.",
    )
    args = parser.parse_args()

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the WSR ontology.") from exc

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("training_strategy") != "full_parameter":
        raise ValueError(
            "This training pipeline requires training_strategy=full_parameter"
        )
    train_batch_size = (
        args.train_batch_size
        if args.train_batch_size is not None
        else int(config["per_device_train_batch_size"])
    )
    eval_batch_size = (
        args.eval_batch_size
        if args.eval_batch_size is not None
        else int(config["per_device_eval_batch_size"])
    )
    gradient_accumulation_steps = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else int(config["gradient_accumulation_steps"])
    )
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(config["learning_rate"])
    )
    num_train_epochs = (
        args.num_train_epochs
        if args.num_train_epochs is not None
        else float(config["num_train_epochs"])
    )
    if args.method in {"contrastive_kd", "ours"} and train_batch_size < 2:
        raise ValueError(
            "contrastive_kd and ours require --train_batch_size >= 2"
        )
    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    dev_records = read_jsonl(args.dev)
    test_records = read_jsonl(args.test)
    gold_records = load_optional(args.gold_train)
    teacher_records = load_optional(args.teacher_train)
    if args.require_enhanced_teacher:
        if not teacher_records:
            raise ValueError("--require_enhanced_teacher requires --teacher_train")
        validate_enhanced_teacher_records(teacher_records)
    teacher_calibration = None
    teacher_gold_agreement = None
    teacher_weight = args.teacher_weight
    if args.teacher_aligned and args.method != "ours":
        raise ValueError("--teacher_aligned is only valid for ours")
    if args.teacher_aligned and args.calibrate_teacher_with_gold:
        raise ValueError(
            "--teacher_aligned cannot be combined with "
            "--calibrate_teacher_with_gold"
        )
    if args.teacher_aligned:
        if not 0.0 < args.teacher_aligned_validation_ratio < 0.5:
            raise ValueError(
                "--teacher_aligned_validation_ratio must be between 0 and 0.5"
            )
        teacher_weight = 1.0 if teacher_weight is None else teacher_weight
        if gold_records:
            teacher_gold_agreement = calibrate_teacher_against_gold(
                teacher_records,
                gold_records,
                minimum_weight=0.0,
                maximum_weight=1.0,
            )
    teacher_aligned_eval_records = []
    teacher_training_records = teacher_records
    if args.teacher_aligned:
        shuffled_teacher = sorted(
            teacher_records,
            key=lambda record: str(
                record.get("case_id") or record.get("record_id") or ""
            ),
        )
        random.Random(int(config["seed"])).shuffle(shuffled_teacher)
        validation_size = max(
            1,
            round(
                len(shuffled_teacher)
                * args.teacher_aligned_validation_ratio
            ),
        )
        teacher_aligned_eval_records = shuffled_teacher[:validation_size]
        teacher_training_records = shuffled_teacher[validation_size:]
    heldout = [
        *dev_records,
        *test_records,
        *teacher_aligned_eval_records,
    ]
    if args.calibrate_teacher_with_gold:
        if args.method != "ours":
            raise ValueError("--calibrate_teacher_with_gold is only valid for ours")
        teacher_calibration = calibrate_teacher_against_gold(
            teacher_records,
            gold_records,
        )
        if teacher_weight is None:
            teacher_weight = float(teacher_calibration["teacher_weight"])
    records = build_training_records(
        args.method,
        gold_records=gold_records,
        teacher_records=teacher_training_records,
        pseudo_records=load_optional(args.pseudo_train),
        heldout_records=heldout,
        ontology=ontology,
        seed=int(config["seed"]),
        stage=args.stage,
        teacher_weight=teacher_weight,
        gold_weight=args.gold_weight,
        gold_repeats=args.gold_repeats,
        teacher_aligned=args.teacher_aligned,
        target_schema=args.target_schema,
        negative_policy=args.negative_policy,
    )
    if not records:
        raise ValueError("No usable training records")
    summary = {
        "method": args.method,
        "stage": args.stage,
        "target_schema": args.target_schema,
        "records": len(records),
        "unique_cases": len({item["case_id"] for item in records}),
        "sources": {
            source: sum(item["source"] == source for item in records)
            for source in sorted({item["source"] for item in records})
        },
        "heldout_records": len(heldout),
        "dev_records": len(dev_records),
        "test_records": len(test_records),
        "teacher_aligned_eval_records": len(teacher_aligned_eval_records),
        "negative_records": sum(
            item.get("negative_target") is not None for item in records
        ),
        "counterfactual_types": {
            kind: sum(item.get("counterfactual_type") == kind for item in records)
            for kind in sorted(
                {
                    item["counterfactual_type"]
                    for item in records
                    if item.get("counterfactual_type")
                }
            )
        },
        "negative_sources": {
            source: sum(item.get("negative_source") == source for item in records)
            for source in sorted(
                {
                    item["negative_source"]
                    for item in records
                    if item.get("negative_source")
                }
            )
        },
        "available_gate_rejected_negatives": sum(
            item.get("available_gate_rejected_negatives", 0) for item in records
        ),
        "available_synthetic_negatives": sum(
            item.get("available_synthetic_negatives", 0) for item in records
        ),
        "available_recall_negatives": sum(
            item.get("available_recall_negatives", 0) for item in records
        ),
        "available_random_negatives": sum(
            item.get("available_random_negatives", 0) for item in records
        ),
        "selected_gate_rejected_relations": sum(
            item.get("selected_gate_rejected_relations", 0) for item in records
        ),
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "enhanced_teacher_required": args.require_enhanced_teacher,
        "teacher_calibration": teacher_calibration,
        "teacher_gold_agreement": teacher_gold_agreement,
        "teacher_weight": teacher_weight,
        "gold_weight": args.gold_weight,
        "gold_repeats": args.gold_repeats,
        "negative_policy": args.negative_policy,
        "contrastive_weight": (
            args.contrastive_weight
            if args.contrastive_weight is not None
            else float(config["contrastive_weight"])
        ),
        "structure_weight": (
            args.structure_weight
            if args.structure_weight is not None
            else float(config["structure_weight"])
        ),
        "teacher_aligned": args.teacher_aligned,
        "teacher_aligned_validation_ratio": (
            args.teacher_aligned_validation_ratio
            if args.teacher_aligned
            else None
        ),
        "gold_used_as_training_target": bool(gold_records)
        and not args.teacher_aligned,
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements-student.txt on the GPU training server."
        ) from exc

    set_seed(int(config["seed"]))
    student_model = args.student_model or config["student_model"]
    tokenizer = AutoTokenizer.from_pretrained(
        student_model,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        student_model,
        trust_remote_code=True,
        torch_dtype=getattr(torch, config["torch_dtype"]),
        attn_implementation=config.get("attn_implementation", "sdpa"),
    )
    if config.get("gradient_checkpointing", True):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Full-parameter training requires every model parameter")

    StudentDataset, StudentCollator, StructuredKDTrainer = build_runtime_classes()
    dataset = StudentDataset(
        records,
        tokenizer,
        int(config["max_length"]),
        int(config["max_target_length"]),
    )
    model_selection_records = (
        teacher_aligned_eval_records
        if args.teacher_aligned
        else dev_records
    )
    eval_records = build_training_records(
        "qwen3_supervised",
        gold_records=model_selection_records,
        heldout_records=[],
        ontology=ontology,
        seed=int(config["seed"]),
        stage=args.stage,
        target_schema=args.target_schema,
    )
    eval_dataset = StudentDataset(
        eval_records,
        tokenizer,
        int(config["max_length"]),
        int(config["max_target_length"]),
    )
    collator = StudentCollator(tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                **summary,
                "student_model": student_model,
                "training_strategy": "full_parameter",
                "teacher_logits_available": False,
                "standard_kd_type": "hard-label sequence distillation",
                "apply_wsr_gate_at_inference": args.method
                in {"kd_wsr_post", "ours"},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=float(config["warmup_ratio"]),
        weight_decay=float(config["weight_decay"]),
        logging_steps=int(config["logging_steps"]),
        save_strategy="epoch",
        eval_strategy="epoch",
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_eval_batch_size=eval_batch_size,
        prediction_loss_only=True,
        save_total_limit=int(config["save_total_limit"]),
        bf16=config["torch_dtype"] == "bfloat16",
        fp16=config["torch_dtype"] == "float16",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=int(config.get("dataloader_num_workers", 2)),
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        ddp_find_unused_parameters=False,
    )
    trainer = StructuredKDTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        loss_config=LossConfig(
            method=args.method,
            contrastive_weight=(
                args.contrastive_weight
                if args.contrastive_weight is not None
                else float(config["contrastive_weight"])
            ),
            structure_weight=(
                args.structure_weight
                if args.structure_weight is not None
                else float(config["structure_weight"])
            ),
            contrastive_temperature=float(config["contrastive_temperature"]),
            structure_margin=float(config["structure_margin"]),
        ),
    )
    resume = args.resume_from_checkpoint
    if isinstance(resume, str):
        resume = str(Path(resume))
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))


if __name__ == "__main__":
    main()
