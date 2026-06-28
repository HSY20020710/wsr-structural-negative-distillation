#!/usr/bin/env bash
set -euo pipefail

: "${GOLD_TRAIN:?Set GOLD_TRAIN}"
: "${TEACHER_TRAIN:?Set TEACHER_TRAIN}"
: "${DEV_DATA:?Set DEV_DATA}"
: "${TEST_DATA:?Set TEST_DATA}"

CONFIG="${CONFIG:-configs/student_training.json}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-1.7B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/student_full}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
DRY_RUN="${DRY_RUN:-0}"

python scripts/check_student_server.py \
  --config "$CONFIG" \
  --gold_train "$GOLD_TRAIN" \
  --teacher_train "$TEACHER_TRAIN" \
  --dev "$DEV_DATA" \
  --test "$TEST_DATA"

run_method() {
  local method="$1"
  shift
  local output_dir="$OUTPUT_ROOT/$method"
  local common=(
    --config "$CONFIG"
    --method "$method"
    --dev "$DEV_DATA"
    --test "$TEST_DATA"
    --student_model "$STUDENT_MODEL"
    --output_dir "$output_dir"
  )
  python scripts/train_student.py "${common[@]}" "$@" --dry_run
  if [[ "$DRY_RUN" != "1" ]]; then
    accelerate launch --num_processes "$NUM_PROCESSES" \
      scripts/train_student.py "${common[@]}" "$@"
  fi
}

run_method qwen3_supervised --gold_train "$GOLD_TRAIN"
if [[ -n "${PSEUDO_TRAIN:-}" && -f "$PSEUDO_TRAIN" ]]; then
  run_method self_training --pseudo_train "$PSEUDO_TRAIN"
else
  echo "Skipping self_training: generate PSEUDO_TRAIN after qwen3_supervised."
fi
run_method standard_kd --teacher_train "$TEACHER_TRAIN"
run_method contrastive_kd --teacher_train "$TEACHER_TRAIN"
run_method kd_wsr_post --teacher_train "$TEACHER_TRAIN"
run_method ours \
  --teacher_train "$TEACHER_TRAIN" \
  --gold_train "$GOLD_TRAIN"
