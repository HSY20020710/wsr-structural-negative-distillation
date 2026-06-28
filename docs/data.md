# Data Availability

The full ship welding quality report dataset is not included in this repository because it contains confidential production records and cannot be publicly redistributed.

This repository provides anonymized synthetic examples under `examples/input/` to demonstrate the expected input format and pipeline execution. These examples are not the private evaluation dataset used in the paper.

## Public Files

```text
examples/input/example_reports.txt
examples/expected/example_gold.jsonl
examples/expected/example_teacher_output.jsonl
examples/expected/example_student_prediction.jsonl
```

## Private Inputs Required for Full Reproduction

Full reproduction of the paper tables requires:

- the private ship welding quality report corpus;
- the manually reviewed Gold annotations;
- frozen teacher prediction caches;
- student model checkpoints or the compute resources to retrain them.

## Provenance Note

The internal evaluation set was produced through human annotation and human review. It serves as the unified benchmark for model evaluation and ablation comparisons.
