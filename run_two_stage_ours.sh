#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

: "${GOLD_TRAIN:?Set GOLD_TRAIN}"
: "${TEACHER_TRAIN:?Set TEACHER_TRAIN}"
: "${DEV_DATA:?Set DEV_DATA}"
: "${TEST_DATA:?Set TEST_DATA}"
: "${ENTITY_MODEL:?Set ENTITY_MODEL to the best completed joint Ours model}"

CONFIG="${CONFIG:-configs/student_training.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/student_relation_refine_v1}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-${OUTPUT_ROOT}_predictions}"
REPORTS_DIR="${REPORTS_DIR:-reports/$(basename "$OUTPUT_ROOT")_experiments}"
GPU="${GPU:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
RELATION_EPOCHS="${RELATION_EPOCHS:-3}"
RELATION_LEARNING_RATE="${RELATION_LEARNING_RATE:-5e-6}"
SKIP_RELATION_TRAIN="${SKIP_RELATION_TRAIN:-0}"
OVERWRITE_PREDICTION="${OVERWRITE_PREDICTION:-0}"

if (( TRAIN_BATCH_SIZE < 2 )); then
  echo "Relation refinement requires TRAIN_BATCH_SIZE>=2" >&2
  exit 2
fi
if [[ ! -f "$ENTITY_MODEL/config.json" ]]; then
  echo "Missing completed entity/joint model: $ENTITY_MODEL/config.json" >&2
  exit 1
fi

relation_dir="$OUTPUT_ROOT/ours_relation_refine"
relation_final="$relation_dir/final"
prediction="$PREDICTIONS_DIR/ours.jsonl"

common=(
  --config "$CONFIG"
  --method ours
  --stage relation
  --gold_train "$GOLD_TRAIN"
  --teacher_train "$TEACHER_TRAIN"
  --dev "$DEV_DATA"
  --test "$TEST_DATA"
  --student_model "$ENTITY_MODEL"
  --output_dir "$relation_dir"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$RELATION_LEARNING_RATE"
  --num_train_epochs "$RELATION_EPOCHS"
)

python scripts/train_student.py "${common[@]}" --dry_run

if [[ "$SKIP_RELATION_TRAIN" != "1" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    scripts/train_student.py "${common[@]}"
fi

if [[ ! -f "$relation_final/config.json" ]]; then
  echo "Missing relation model: $relation_final/config.json" >&2
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

CUDA_VISIBLE_DEVICES="$GPU" python scripts/predict_student_two_stage.py \
  --entity_model "$ENTITY_MODEL" \
  --relation_model "$relation_final" \
  --test "$TEST_DATA" \
  --output "$prediction"

python scripts/evaluate_student.py \
  --method ours \
  --predictions_dir "$PREDICTIONS_DIR" \
  --test "$TEST_DATA" \
  --reports_dir "$REPORTS_DIR"

echo "Report: $REPORTS_DIR/ours_results.md"
