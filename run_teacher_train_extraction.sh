#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

API_URL="${API_URL:-http://localhost:8000/v1/chat/completions}"
MODEL="${MODEL:-qwen3.6-27b}"
PROTOCOL="${PROTOCOL:-openai}"
INPUT="${INPUT:-data/student/teacher_train_input.jsonl}"
FIRST_PASS="${FIRST_PASS:-work/student_teacher_train_first_pass.jsonl}"
OUTPUT="${OUTPUT:-data/student/teacher_train_enhanced.jsonl}"
ONTOLOGY="${ONTOLOGY:-configs/wsr_ontology.yaml}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
MAX_RETRIES="${MAX_RETRIES:-3}"
MAX_TOKENS="${MAX_TOKENS:-4096}"

for path in "$INPUT" "$ONTOLOGY"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

EXPECTED="$(wc -l < "$INPUT" | tr -d ' ')"
echo "Teacher extraction input: $EXPECTED records"
echo "API: $API_URL"
echo "Model: $MODEL"

python scripts/teacher_extract.py \
  --input "$INPUT" \
  --output "$FIRST_PASS" \
  --mode with_wsr_ontology \
  --model "$MODEL" \
  --api_url "$API_URL" \
  --protocol "$PROTOCOL" \
  --ontology "$ONTOLOGY" \
  --timeout_seconds "$TIMEOUT_SECONDS" \
  --max_retries "$MAX_RETRIES" \
  --max_tokens "$MAX_TOKENS" \
  --resume

FIRST_COUNT="$(wc -l < "$FIRST_PASS" | tr -d ' ')"
if [[ "$FIRST_COUNT" != "$EXPECTED" ]]; then
  echo "First pass incomplete: expected=$EXPECTED actual=$FIRST_COUNT" >&2
  exit 1
fi

python scripts/refine_teacher_entities.py \
  --input "$FIRST_PASS" \
  --output "$OUTPUT" \
  --ontology "$ONTOLOGY" \
  --model "$MODEL" \
  --api_url "$API_URL" \
  --protocol "$PROTOCOL" \
  --timeout_seconds "$TIMEOUT_SECONDS" \
  --max_retries "$MAX_RETRIES" \
  --max_tokens "$MAX_TOKENS" \
  --resume

FINAL_COUNT="$(wc -l < "$OUTPUT" | tr -d ' ')"
if [[ "$FINAL_COUNT" != "$EXPECTED" ]]; then
  echo "Final teacher data incomplete: expected=$EXPECTED actual=$FINAL_COUNT" >&2
  exit 1
fi

python scripts/validate_teacher_train.py \
  --input "$OUTPUT" \
  --source "$INPUT" \
  --ontology "$ONTOLOGY"

echo "Teacher train extraction passed: $OUTPUT"
