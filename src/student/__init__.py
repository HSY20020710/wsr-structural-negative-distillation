"""Student-model distillation utilities."""

from .data import (
    SUPPORTED_METHODS,
    SUPPORTED_STAGES,
    SUPPORTED_TARGET_SCHEMAS,
    ENHANCED_TEACHER_COMPONENTS,
    build_training_records,
    calibrate_teacher_against_gold,
    canonical_target,
    make_recall_counterfactual_targets,
    make_wsr_negative_target,
    validate_enhanced_teacher_records,
)

__all__ = [
    "SUPPORTED_METHODS",
    "SUPPORTED_STAGES",
    "SUPPORTED_TARGET_SCHEMAS",
    "ENHANCED_TEACHER_COMPONENTS",
    "build_training_records",
    "calibrate_teacher_against_gold",
    "canonical_target",
    "make_recall_counterfactual_targets",
    "make_wsr_negative_target",
    "validate_enhanced_teacher_records",
]
