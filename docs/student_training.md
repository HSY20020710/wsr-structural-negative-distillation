# Student Distillation Training

The unified entry point is:

```powershell
python scripts/train_student.py --config configs/student_training.json ...
```

All student experiments use full-parameter fine-tuning. LoRA/PEFT adapters are
not used, and each `final` directory contains a standalone full model.

Supported methods:

- `qwen3_supervised`: human Gold only.
- `self_training`: filtered student pseudo labels.
- `standard_kd`: frozen teacher structured outputs; hard-label sequence KD.
- `contrastive_kd`: Standard KD plus class-aware two-view contrastive loss.
  Samples with the same defect/event class are treated as positives rather than
  accidental in-batch negatives.
- `kd_wsr_post`: Standard KD; WSR Gate is applied only during inference.
- `ours`: teacher structured outputs, optional high-weight Gold, contrastive loss,
  real teacher relations rejected by WSR Gate, relation-type, direction,
  endpoint and entity-type synthetic counterfactual negatives, and WSR Gate
  during inference.

Each source report appears once per epoch. For `ours`, one balanced WSR
negative is attached to each report so reports with more available negative
transformations do not receive duplicated positive-label training. A valid
teacher relation rejected by Gate is preferred as a real hard negative; a
synthetic counterfactual is used only when the report has no usable rejected
relation.

Before any training, run:

```powershell
python scripts/prepare_distillation_inputs.py
```

This creates fixed human-annotation templates, the complete teacher-train input,
hashes and a readiness manifest at
`data/student/pre_distillation_manifest.json`. Training must remain blocked until
the manifest confirms that teacher train, independent Gold train and human dev
labels are complete. `pseudo_train.jsonl` is produced only after the supervised
student exists.

Generate the self-training labels after the supervised run:

```bash
python scripts/generate_student_pseudo.py \
  --model outputs/student_full/qwen3_supervised/final \
  --input data/student/teacher_train_input.jsonl \
  --output data/student/pseudo_train.jsonl
```

Optional entity-preserving augmentation is available only after labels exist:

```bash
python scripts/augment_labeled_train.py \
  --input data/student/teacher_train.jsonl \
  --output data/student/teacher_train_augmented.jsonl \
  --copies 1
```

Never augment Dev or Test. Report original reports and augmented training
instances separately.

The private manually annotated and reviewed evaluation set must be passed
through `--heldout` and must not be reused as `--teacher_train` or
`--gold_train`.

Validate data without importing GPU libraries:

```powershell
python scripts/train_student.py ... --dry_run
```

On a shared 48 GB GPU, force one visible GPU. Full-parameter training has a
higher memory cost than adapter training:

```bash
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --num_processes 1 --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no \
  scripts/train_student.py ... \
  --train_batch_size 2 --eval_batch_size 1 \
  --gradient_accumulation_steps 8
```

Do not expose multiple GPUs to a one-process launch, because Transformers may
fall back to DataParallel and multiply the effective batch and memory use.
The contrastive methods require a per-device training batch size of at least 2.

Example:

```powershell
python scripts/train_student.py `
  --config configs/student_training.json `
  --method ours `
  --teacher_train data/student/teacher_train.jsonl `
  --gold_train data/student/gold_train.jsonl `
  --dev data/student/dev.jsonl `
  --test data/student/test.jsonl `
  --output_dir outputs/student_full/ours
```

Before training on Linux:

```bash
python -m pip install -r requirements-student.txt
export GOLD_TRAIN=data/student/gold_train.jsonl
export TEACHER_TRAIN=data/student/teacher_train.jsonl
export PSEUDO_TRAIN=data/student/pseudo_train.jsonl
export DEV_DATA=data/student/dev.jsonl
export TEST_DATA=data/student/test.jsonl
export DRY_RUN=1
bash run_student_experiments.sh
```

After all six dry-runs pass:

```bash
export DRY_RUN=0
bash run_student_experiments.sh
```

Set `NUM_PROCESSES` to the number of GPUs. Interrupted individual runs can be
continued with `--resume_from_checkpoint` when invoking `train_student.py`
directly.

The batch scripts write full-parameter checkpoints under `outputs/student_full`
predictions under `outputs/student_full_predictions`, and reports under
`reports/student_full_experiments`. Existing `outputs/student*` adapter
checkpoints are not compatible with this pipeline.

JSON generation does not use repetition penalties or n-gram blocking. Those
decoding controls incorrectly suppress repeated JSON field patterns and can
truncate entity and relation lists. Evaluation reports relation F1 both before
and after WSR Gate, plus the Gate relation-retention rate.

The existing vLLM teacher cache contains structured labels and self-reported
confidence, but no token-level teacher logits. Therefore `standard_kd` is
implemented as hard-label sequence distillation rather than fabricated soft-logit
distillation.

## Train and evaluate one method

Use `run_one_student_method.sh` to train one model, immediately run fixed-Test
inference, and print its result table without waiting for the other methods.

Example for the supervised baseline:

```bash
export METHOD=qwen3_supervised
export GPU=2
export GOLD_TRAIN=data/student/gold_train.jsonl
export DEV_DATA=data/student/dev.jsonl
export TEST_DATA=data/student/test.jsonl
export STUDENT_MODEL=models/Qwen3-1.7B
export OUTPUT_ROOT=outputs/student_full_v2
export TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=16
bash run_one_student_method.sh
```

To evaluate an already completed model without retraining:

```bash
export SKIP_TRAIN=1
export OVERWRITE_PREDICTION=1
bash run_one_student_method.sh
```

For `standard_kd`, `contrastive_kd`, and `kd_wsr_post`, set
`TEACHER_TRAIN`. For `ours`, set both `TEACHER_TRAIN` and `GOLD_TRAIN`.
For `self_training`, set `PSEUDO_TRAIN` and `GOLD_TRAIN`. Contrastive methods
require `TRAIN_BATCH_SIZE=2` or greater.

Single-method results are written to:

```text
reports/<output-root-name>_experiments/<method>_results.md
```

For the complete enhanced teacher pipeline, first run:

```bash
bash run_teacher_train_extraction.sh
```

This writes `data/student/teacher_train_enhanced.jsonl`. Train Ours with:

```bash
METHOD=ours \
GOLD_TRAIN=data/student/gold_train.jsonl \
TEACHER_TRAIN=data/student/teacher_train_enhanced.jsonl \
DEV_DATA=data/student/dev.jsonl \
TEST_DATA=data/student/test.jsonl \
REQUIRE_ENHANCED_TEACHER=1 \
OUTPUT_ROOT=outputs/student_enhanced_teacher_v1 \
GPU=2 \
TRAIN_BATCH_SIZE=2 \
EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=8 \
NUM_TRAIN_EPOCHS=3 \
bash run_one_student_method.sh
```

`REQUIRE_ENHANCED_TEACHER=1` checks every record for the declared five-stage
pipeline, successful second-pass refinement, pre-Gate relations, and rejected
relations before training starts.

When enhanced teacher labels overlap with independently annotated Gold, enable
strict reliability calibration and Gold oversampling:

```bash
CALIBRATE_TEACHER_WITH_GOLD=1 \
GOLD_WEIGHT=2.0 \
GOLD_REPEATS=3 \
bash run_one_student_method.sh
```

The calibration uses strict entity spans/types and relation endpoints/types.
It lowers the absolute teacher gradient weight when teacher labels disagree
with Gold. It never reads the dev or test annotations.

To reproduce the enhanced teacher annotation style instead of optimizing for
the independent Gold convention, use teacher-aligned distillation:

```bash
TEACHER_ALIGNED=1 \
TEACHER_WEIGHT=1.0 \
CALIBRATE_TEACHER_WITH_GOLD=0 \
GOLD_REPEATS=1 \
bash run_one_student_method.sh
```

This mode trains on every enhanced teacher target, including teacher records
whose case IDs overlap the training Gold. Gold is read only to write an
agreement report; it does not replace or modify teacher targets. Dev and test
records remain excluded from training.

## Refine relations with a second stage

Use the best completed joint Ours checkpoint for entity prediction. Train only
a relation-refinement checkpoint from that model at a low learning rate. This
avoids the task interference observed when one checkpoint is trained on two
different JSON schemas simultaneously.

```bash
GOLD_TRAIN=data/student/gold_train.jsonl \
TEACHER_TRAIN=data/student/teacher_train.jsonl \
DEV_DATA=data/student/dev.jsonl \
TEST_DATA=data/student/test.jsonl \
ENTITY_MODEL=outputs/student_full_v6/ours/final \
OUTPUT_ROOT=outputs/student_relation_refine_v1 \
GPU=2 \
TRAIN_BATCH_SIZE=2 \
EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=8 \
RELATION_EPOCHS=3 \
RELATION_LEARNING_RATE=5e-6 \
bash run_two_stage_ours.sh
```

Outputs:

```text
outputs/student_relation_refine_v1/ours_relation_refine/final
outputs/student_relation_refine_v1_predictions/ours.jsonl
reports/student_relation_refine_v1_experiments/ours_results.md
```
