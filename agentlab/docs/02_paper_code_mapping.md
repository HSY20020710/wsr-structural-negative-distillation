# 论文—真实代码—平台 Tool 映射

| 论文环节 | 冻结代码入口 | 平台 Tool / Agent |
|---|---|---|
| Teacher candidate generation | `scripts/teacher_extract.py`, `scripts/run_teacher_experiment.py` | TeacherRunnerTool / TeacherAgent |
| entity refinement | `src/teacher/entity_refinement.py` | TeacherPipelineTool |
| WSR ontology | `configs/wsr_ontology.yaml` | OntologyTool |
| Gate | `src/ontology/gate.py` | GateTool / GateAuditAgent |
| distillation input | `scripts/prepare_distillation_inputs.py`, `scripts/prepare_student_data.py` | DataPreparationTool |
| Student training | `scripts/train_student.py` | StudentTrainTool / StudentAgent |
| prediction | `scripts/predict_student.py`, `predict_student_two_stage.py` | StudentPredictTool |
| evaluation | `scripts/evaluate_student.py` | MetricTool |
| Gate impact | `scripts/analyze_gate_impact.py` | GateImpactTool |
| ablation | `scripts/run_student_ablation.py`, `collect_ablation_results.py` | AblationTool |
| bootstrap | `scripts/bootstrap_significance.py` | BootstrapTool |
| efficiency | `scripts/collect_efficiency_results.py` | EfficiencyTool |

**边界提醒**：当前 `gate.py` 可直接确认的主要是允许的 relation signatures。论文中更细的因果方向、物理兼容和责任上下文约束，不应在平台中伪造；需要按实际 executable rules 逐项接入。
