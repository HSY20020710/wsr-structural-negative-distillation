# From WSR Constraints to Learnable Structural Negatives: Lightweight Knowledge Distillation for Shipbuilding Welding Quality Extraction

Official code release for the paper "From WSR Constraints to Learnable Structural Negatives: Lightweight Knowledge Distillation for Shipbuilding Welding Quality Extraction".

The repository provides the implementation of the Wuli-Shili-Renli (WSR) consistency gate, teacher-output refinement, structural negative construction, student distillation training, and evaluation scripts used in the paper.

## Method Pipeline

1. Parse ship welding quality reports into structured report records.
2. Generate offline teacher JSON outputs with a large Qwen teacher model.
3. Apply WSR-guided entity refinement, endpoint remapping, and consistency gating.
4. Route Gate-passed outputs as positive supervision and Gate-rejected/counterfactual outputs as structural negatives.
5. Train a lightweight Qwen3-1.7B student with sequence targets, label-aware contrastive learning, and structural ranking loss.
6. Evaluate Student-only and Student+Gate inference modes.

## Repository Structure

```text
configs/      WSR ontology, extraction schema, and student-training config
src/          Reusable Python modules for data, teacher parsing, WSR Gate, metrics, and student training
scripts/      Command-line entry points for the paper pipeline
examples/     Anonymized synthetic examples for format and smoke testing
docs/         Data policy, reproduction guide, and paper-result summary
tests/        Unit tests for parsing, training data, teacher stages, and metrics
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For full student training, install a CUDA-enabled PyTorch build that matches your GPU and driver. The scripts expect a Hugging Face model path or model id such as `Qwen/Qwen3-1.7B`.

## Quick Start with Anonymized Examples

The full ship welding quality report dataset is not included because it contains confidential production records. A small synthetic example is provided to demonstrate the input format:

```text
examples/input/example_reports.txt
```

Parse example reports:

```bash
python scripts/prepare_student_data.py \
  --raw examples/input/example_reports.txt \
  --test_gold examples/expected/example_gold.jsonl \
  --output_dir data/student_example
```

Inspect the expected JSON format:

```text
examples/expected/example_teacher_output.jsonl
examples/expected/example_student_prediction.jsonl
```

## Main Commands

Teacher extraction and WSR refinement:

```bash
python scripts/teacher_extract.py --help
python scripts/refine_teacher_entities.py --help
python scripts/run_teacher_experiment.py --help
```

Student data, training, prediction, and evaluation:

```bash
python scripts/prepare_distillation_inputs.py --help
python scripts/train_student.py --help
python scripts/predict_student.py --help
python scripts/evaluate_student.py --help
```

Ablation and analysis:

```bash
python scripts/analyze_gate_impact.py --help
python scripts/run_student_ablation.py --help
python scripts/bootstrap_significance.py --help
python scripts/per_category_error_analysis.py --help
```

## Reproducing Paper Results

The complete private dataset, full teacher caches, predictions, and checkpoints are not distributed in this repository. `docs/reproduction.md` explains which script generated each table in the paper and what private inputs are required.

The public repository is intended to provide:

- the implementation of the method;
- the WSR ontology and validation logic;
- the training/evaluation entry points;
- anonymized examples for format inspection and smoke tests;
- a traceable description of how the reported paper results were produced.

## Data and Model Availability

See `docs/data.md` for the data-release policy. Model weights and training checkpoints are not committed to GitHub. Users should download base models from their original providers and train/evaluate with their own available data or the private dataset under authorized access.

## Citation

If you use this code, please cite the accompanying paper. A `CITATION.cff` file is included for citation metadata.

## License

Code is released under the MIT License. Dataset access is not granted by this license.
