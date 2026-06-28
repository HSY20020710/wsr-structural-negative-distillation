# Reproduction Guide

This document maps paper results to the scripts that produced them. The commands below assume access to the private data and frozen prediction caches described in `docs/data.md`.

## Teacher Quality Table

Evaluates teacher settings against the reviewed evaluation set:

```bash
python scripts/run_teacher_experiment.py \
  --gold data/gold/private_evaluation.jsonl \
  --ontology configs/wsr_ontology.yaml \
  --reports_dir reports/frozen_teacher_experiment
```

## Student Training

Prepare distillation inputs:

```bash
python scripts/prepare_distillation_inputs.py --student_dir data/student
```

Train a single method:

```bash
METHOD=ours \
GOLD_TRAIN=data/student/gold_train.jsonl \
TEACHER_TRAIN=data/student/teacher_train.jsonl \
DEV_DATA=data/student/dev.jsonl \
TEST_DATA=data/student/test.jsonl \
STUDENT_MODEL=Qwen/Qwen3-1.7B \
OUTPUT_ROOT=outputs/student_full \
bash run_one_student_method.sh
```

Evaluate completed predictions:

```bash
python scripts/evaluate_student.py \
  --predictions_dir outputs/student_full_predictions \
  --test data/student/test.jsonl \
  --reports_dir reports/student_full_experiments
```

## Inference-time WSR Gate Impact

```bash
python scripts/analyze_gate_impact.py
python scripts/complete_gate_metrics.py
```

## Student-only Ablation

```bash
python scripts/run_student_ablation.py
python scripts/collect_ablation_results.py
```

## Significance and Error Analysis

```bash
python scripts/bootstrap_significance.py
python scripts/per_category_error_analysis.py
```

## Efficiency Analysis

```bash
bash run_student_efficiency_benchmark.sh
python scripts/collect_efficiency_results.py
```

The public GitHub repository does not include private outputs or checkpoints. The reported aggregate numbers are summarized in `docs/paper_results.md`.
