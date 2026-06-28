#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

: "${METHOD:?Set METHOD to qwen3_supervised, self_training, standard_kd, contrastive_kd, kd_wsr_post, or ours}"
: "${DEV_DATA:?Set DEV_DATA}"
: "${TEST_DATA:?Set TEST_DATA}"

CONFIG="${CONFIG:-configs/student_training.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
STUDENT_MODEL="${STUDENT_MODEL:-models/Qwen3-1.7B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/student_full_v2}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-${OUTPUT_ROOT}_predictions}"
REPORTS_DIR="${REPORTS_DIR:-reports/$(basename "$OUTPUT_ROOT")_experiments}"
GPU="${GPU:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
TARGET_SCHEMA="${TARGET_SCHEMA:-full}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
OVERWRITE_PREDICTION="${OVERWRITE_PREDICTION:-0}"
REQUIRE_ENHANCED_TEACHER="${REQUIRE_ENHANCED_TEACHER:-0}"
CALIBRATE_TEACHER_WITH_GOLD="${CALIBRATE_TEACHER_WITH_GOLD:-0}"
TEACHER_WEIGHT="${TEACHER_WEIGHT:-}"
GOLD_WEIGHT="${GOLD_WEIGHT:-2.0}"
GOLD_REPEATS="${GOLD_REPEATS:-3}"
NEGATIVE_POLICY="${NEGATIVE_POLICY:-}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-}"
STRUCTURE_WEIGHT="${STRUCTURE_WEIGHT:-}"
TEACHER_ALIGNED="${TEACHER_ALIGNED:-0}"
TEACHER_ALIGNED_VALIDATION_RATIO="${TEACHER_ALIGNED_VALIDATION_RATIO:-0.05}"

case "$METHOD" in
  qwen3_supervised)
    : "${GOLD_TRAIN:?Set GOLD_TRAIN}"
    method_args=(--gold_train "$GOLD_TRAIN")
    ;;
  self_training)
    : "${GOLD_TRAIN:?Set GOLD_TRAIN}"
    : "${PSEUDO_TRAIN:?Set PSEUDO_TRAIN}"
    method_args=(
      --gold_train "$GOLD_TRAIN"
      --pseudo_train "$PSEUDO_TRAIN"
    )
    ;;
  standard_kd|contrastive_kd|kd_wsr_post)
    : "${TEACHER_TRAIN:?Set TEACHER_TRAIN}"
    method_args=(--teacher_train "$TEACHER_TRAIN")
    ;;
  ours)
    : "${GOLD_TRAIN:?Set GOLD_TRAIN}"
    : "${TEACHER_TRAIN:?Set TEACHER_TRAIN}"
    method_args=(
      --gold_train "$GOLD_TRAIN"
      --teacher_train "$TEACHER_TRAIN"
    )
    ;;
  *)
    echo "Unsupported METHOD: $METHOD" >&2
    exit 2
    ;;
esac

if [[ "$METHOD" == "contrastive_kd" || "$METHOD" == "ours" ]]; then
  if (( TRAIN_BATCH_SIZE < 2 )); then
    echo "$METHOD requires TRAIN_BATCH_SIZE>=2" >&2
    exit 2
  fi
fi

model_dir="$OUTPUT_ROOT/$METHOD"
final_dir="$model_dir/final"
prediction="$PREDICTIONS_DIR/$METHOD.jsonl"

common=(
  --config "$CONFIG"
  --method "$METHOD"
  --target_schema "$TARGET_SCHEMA"
  --dev "$DEV_DATA"
  --test "$TEST_DATA"
  --student_model "$STUDENT_MODEL"
  --output_dir "$model_dir"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  "${method_args[@]}"
)
if [[ -n "$NUM_TRAIN_EPOCHS" ]]; then
  common+=(--num_train_epochs "$NUM_TRAIN_EPOCHS")
fi
if [[ -n "$LEARNING_RATE" ]]; then
  common+=(--learning_rate "$LEARNING_RATE")
fi
if [[ "$REQUIRE_ENHANCED_TEACHER" == "1" ]]; then
  common+=(--require_enhanced_teacher)
fi
if [[ "$CALIBRATE_TEACHER_WITH_GOLD" == "1" ]]; then
  common+=(--calibrate_teacher_with_gold)
fi
if [[ -n "$TEACHER_WEIGHT" ]]; then
  common+=(--teacher_weight "$TEACHER_WEIGHT")
fi
if [[ "$METHOD" == "ours" ]]; then
  common+=(--gold_weight "$GOLD_WEIGHT" --gold_repeats "$GOLD_REPEATS")
fi
if [[ -n "$NEGATIVE_POLICY" ]]; then
  common+=(--negative_policy "$NEGATIVE_POLICY")
fi
if [[ -n "$CONTRASTIVE_WEIGHT" ]]; then
  common+=(--contrastive_weight "$CONTRASTIVE_WEIGHT")
fi
if [[ -n "$STRUCTURE_WEIGHT" ]]; then
  common+=(--structure_weight "$STRUCTURE_WEIGHT")
fi
if [[ "$TEACHER_ALIGNED" == "1" ]]; then
  common+=(
    --teacher_aligned
    --teacher_aligned_validation_ratio "$TEACHER_ALIGNED_VALIDATION_RATIO"
  )
fi

"$PYTHON_BIN" scripts/train_student.py "${common[@]}" --dry_run

if [[ "$SKIP_TRAIN" != "1" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$ACCELERATE_BIN" launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    scripts/train_student.py "${common[@]}"
fi

if [[ ! -f "$final_dir/config.json" ]]; then
  echo "Missing completed model: $final_dir/config.json" >&2
  exit 1
fi

mkdir -p "$PREDICTIONS_DIR" "$REPORTS_DIR"
if [[ -f "$prediction" ]]; then
  if [[ "$OVERWRITE_PREDICTION" == "1" ]]; then
    rm -f "$prediction"
  else
    echo "Prediction already exists: $prediction" >&2
    echo "Set OVERWRITE_PREDICTION=1 to replace it." >&2
    exit 1
  fi
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/predict_student.py \
  --method "$METHOD" \
  --target_schema "$TARGET_SCHEMA" \
  --model "$final_dir" \
  --test "$TEST_DATA" \
  --output "$prediction" \
  --batch_size "$PREDICT_BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS"

"$PYTHON_BIN" scripts/evaluate_student.py \
  --method "$METHOD" \
  --predictions_dir "$PREDICTIONS_DIR" \
  --test "$TEST_DATA" \
  --reports_dir "$REPORTS_DIR"

echo "Report: $REPORTS_DIR/${METHOD}_results.md"
