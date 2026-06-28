#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU="${GPU:-2}"
PYTHON_BIN="${PYTHON_BIN:-/home/justjg/miniconda3/bin/python}"
TEST_DATA="${TEST_DATA:-data/gold/teacher_quality_gold.manual_review_v2.frozen.jsonl}"
MODEL_ROOT="${MODEL_ROOT:-outputs/baselines_teacher_consistency_364}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/efficiency_benchmark}"
REPORT_ROOT="${REPORT_ROOT:-reports/efficiency_benchmark}"
WARMUP_BATCHES="${WARMUP_BATCHES:-5}"

run_method() {
  local method="$1"
  local model="$MODEL_ROOT/$method/final"
  local prediction_dir="$OUTPUT_ROOT/predictions"
  local prediction="$prediction_dir/$method.jsonl"
  mkdir -p "$prediction_dir" "$OUTPUT_ROOT/benchmarks" "$REPORT_ROOT/$method"
  rm -f "$prediction"

  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/predict_student.py \
    --method "$method" \
    --target_schema full \
    --model "$model" \
    --test "$TEST_DATA" \
    --output "$prediction" \
    --batch_size 1 \
    --max_new_tokens 1536 \
    --benchmark_warmup_batches "$WARMUP_BATCHES" \
    --benchmark_output "$OUTPUT_ROOT/benchmarks/$method.json"

  "$PYTHON_BIN" scripts/evaluate_student.py \
    --method "$method" \
    --predictions_dir "$prediction_dir" \
    --test "$TEST_DATA" \
    --reports_dir "$REPORT_ROOT/$method"
}

run_method qwen3_supervised
run_method standard_kd
run_method ours

"$PYTHON_BIN" scripts/collect_efficiency_results.py \
  --benchmark_dir "$OUTPUT_ROOT/benchmarks" \
  --reports_dir "$REPORT_ROOT" \
  --output_dir "$REPORT_ROOT"

echo "Efficiency table: $REPORT_ROOT/efficiency_results.md"
