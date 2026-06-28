from __future__ import annotations

import random
import unittest

import yaml

from src.student.data import (
    assert_disjoint,
    build_system_prompt,
    build_training_records,
    build_relation_user_prompt,
    calibrate_teacher_against_gold,
    canonical_target,
    text_schema_target,
    make_gate_rejected_targets,
    make_recall_counterfactual_targets,
    make_wsr_counterfactual_targets,
    make_wsr_negative_target,
    validate_enhanced_teacher_records,
)


class StudentTrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open("configs/wsr_ontology.yaml", encoding="utf-8") as handle:
            cls.ontology = yaml.safe_load(handle)
        cls.record = {
            "case_id": "train_1",
            "text": "焊前未清理导致气孔。",
            "defect_label": "气孔",
            "event_label": "焊接管控不到位",
            "parse_success": True,
            "entities": [
                {
                    "id": "E1",
                    "text": "未清理",
                    "start": 2,
                    "end": 5,
                    "type": "S_CAUSE",
                    "wsr_layer": "Shili",
                    "confidence": 0.9,
                },
                {
                    "id": "E2",
                    "text": "气孔",
                    "start": 7,
                    "end": 9,
                    "type": "W_DEFECT",
                    "wsr_layer": "Wuli",
                    "confidence": 0.9,
                },
            ],
            "relations": [
                {
                    "head": "E1",
                    "tail": "E2",
                    "type": "CAUSES",
                    "confidence": 0.8,
                }
            ],
        }

    def test_canonical_target_removes_raw_fields(self) -> None:
        record = {**self.record, "raw_response": "large response"}
        target = canonical_target(record)
        self.assertNotIn("raw_response", target)
        self.assertEqual(target["relations"][0]["type"], "CAUSES")

    def test_text_schema_target_removes_ids_and_offsets(self) -> None:
        target = text_schema_target(self.record)
        self.assertEqual(
            target["entities"][0],
            {"text": "未清理", "type": "S_CAUSE", "wsr_layer": "Shili"},
        )
        self.assertEqual(
            target["relations"][0],
            {
                "head_text": "未清理",
                "relation": "CAUSES",
                "tail_text": "气孔",
            },
        )

    def test_system_prompt_contains_closed_schema(self) -> None:
        prompt = build_system_prompt(self.ontology)
        self.assertIn("W_DEFECT", prompt)
        self.assertIn("RESPONSIBLE_FOR", prompt)
        self.assertIn("不得输出WPS", prompt)

    def test_invalid_span_entity_and_relation_are_removed(self) -> None:
        record = {
            **self.record,
            "entities": [
                *self.record["entities"],
                {
                    "id": "E3",
                    "text": "missing",
                    "start": None,
                    "end": None,
                    "type": "W_LOCATION",
                    "wsr_layer": "Wuli",
                },
            ],
            "relations": [
                *self.record["relations"],
                {"head": "E3", "tail": "E2", "type": "OCCURS_AT"},
            ],
        }
        target = canonical_target(record)
        self.assertEqual(len(target["entities"]), 2)
        self.assertEqual(len(target["relations"]), 1)

    def test_wsr_negative_changes_relation_to_invalid_type(self) -> None:
        negative = make_wsr_negative_target(
            self.record, self.ontology, random.Random(7)
        )
        self.assertIsNotNone(negative)
        self.assertNotEqual(
            negative["relations"][0]["type"],
            self.record["relations"][0]["type"],
        )
        self.assertNotEqual(negative["relations"][0]["type"], "CAUSES")

    def test_wsr_counterfactuals_cover_multiple_error_types(self) -> None:
        record = {
            **self.record,
            "text": "焊前未清理导致气孔出现在焊缝。",
            "entities": [
                *self.record["entities"],
                {
                    "id": "E3",
                    "text": "焊缝",
                    "start": 12,
                    "end": 14,
                    "type": "W_LOCATION",
                    "wsr_layer": "Wuli",
                },
            ],
        }
        negatives = make_wsr_counterfactual_targets(
            record, self.ontology, random.Random(7), max_negatives=10
        )
        kinds = {item["counterfactual_type"] for item in negatives}
        self.assertIn("relation_type_swap", kinds)
        self.assertIn("relation_direction_reverse", kinds)
        self.assertIn("endpoint_replacement", kinds)
        self.assertIn("entity_type_corruption", kinds)

    def test_semantic_relation_swap_can_remain_gate_valid(self) -> None:
        negatives = make_wsr_counterfactual_targets(
            self.record,
            self.ontology,
            random.Random(7),
            max_negatives=10,
        )
        semantic = [
            item for item in negatives
            if item["counterfactual_type"] == "relation_semantic_swap"
        ]
        self.assertEqual(len(semantic), 1)
        self.assertEqual(
            semantic[0]["target"]["relations"][0]["type"],
            "CONTRIBUTES_TO",
        )

    def test_recall_counterfactuals_cover_omission_and_boundary_errors(self) -> None:
        negatives = make_recall_counterfactual_targets(
            self.record, random.Random(7)
        )
        kinds = {item["counterfactual_type"] for item in negatives}
        self.assertIn("relation_omission", kinds)
        self.assertIn("entity_omission", kinds)
        self.assertIn("entity_boundary_corruption", kinds)
        for item in negatives:
            for entity in item["target"]["entities"]:
                self.assertEqual(
                    entity["text"],
                    self.record["text"][entity["start"]:entity["end"]],
                )

    def test_gate_rejected_relation_becomes_real_negative(self) -> None:
        record = {
            **self.record,
            "rejected_relations": [
                {
                    "head": "E2",
                    "tail": "E1",
                    "type": "CAUSES",
                    "gate_result": {
                        "passed": False,
                        "conflict_type": "shili",
                    },
                }
            ],
        }
        negatives = make_gate_rejected_targets(record, self.ontology)
        self.assertEqual(len(negatives), 1)
        self.assertEqual(negatives[0]["negative_source"], "gate_rejected")
        self.assertEqual(
            negatives[0]["counterfactual_type"], "gate_rejected_shili"
        )
        self.assertIn(
            {"head": "E2", "tail": "E1", "type": "CAUSES"},
            negatives[0]["target"]["relations"],
        )

    def test_ours_prefers_gate_rejected_negative(self) -> None:
        record = {
            **self.record,
            "rejected_relations": [
                {
                    "head": "E2",
                    "tail": "E1",
                    "type": "CAUSES",
                    "gate_result": {
                        "passed": False,
                        "conflict_type": "shili",
                    },
                }
            ],
        }
        items = build_training_records(
            "ours",
            teacher_records=[record],
            heldout_records=[],
            ontology=self.ontology,
        )
        self.assertEqual(items[0]["negative_source"], "gate_rejected")
        self.assertEqual(items[0]["available_gate_rejected_negatives"], 1)
        self.assertEqual(items[0]["selected_gate_rejected_relations"], 1)

    def test_ours_keeps_all_gate_rejected_relations_in_one_negative(self) -> None:
        record = {
            **self.record,
            "rejected_relations": [
                {
                    "head": "E2",
                    "tail": "E1",
                    "type": "CAUSES",
                    "gate_result": {
                        "passed": False,
                        "conflict_type": "shili",
                    },
                },
                {
                    "head": "E2",
                    "tail": "E1",
                    "type": "AFFECTS",
                    "gate_result": {
                        "passed": False,
                        "conflict_type": "wuli",
                    },
                },
            ],
        }
        item = build_training_records(
            "ours",
            teacher_records=[record],
            heldout_records=[],
            ontology=self.ontology,
        )[0]
        negative = __import__("json").loads(item["negative_target"])
        self.assertEqual(item["counterfactual_type"], "gate_rejected_mixed")
        self.assertEqual(item["selected_gate_rejected_relations"], 2)
        self.assertEqual(len(negative["relations"]), 3)

    def test_heldout_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_disjoint([self.record], [self.record])

    def test_standard_kd_builds_teacher_records(self) -> None:
        items = build_training_records(
            "standard_kd",
            teacher_records=[self.record],
            heldout_records=[],
            ontology=self.ontology,
        )
        self.assertEqual(items[0]["source"], "teacher")
        self.assertIsNone(items[0]["negative_target"])

    def test_text_schema_training_target_is_simplified(self) -> None:
        item = build_training_records(
            "standard_kd",
            teacher_records=[self.record],
            heldout_records=[],
            ontology=self.ontology,
            target_schema="text",
        )[0]
        target = __import__("json").loads(item["target"])
        self.assertNotIn("id", target["entities"][0])
        self.assertNotIn("start", target["entities"][0])
        self.assertIn("head_text", target["relations"][0])

    def test_contrastive_key_uses_defect_and_event_labels(self) -> None:
        items = build_training_records(
            "contrastive_kd",
            teacher_records=[self.record],
            heldout_records=[],
            ontology=self.ontology,
        )
        self.assertEqual(
            items[0]["contrastive_key"],
            "气孔::焊接管控不到位",
        )

    def test_ours_prefers_gold_on_duplicate_case_id(self) -> None:
        gold = {**self.record, "event_label": "工艺纪律违规"}
        items = build_training_records(
            "ours",
            teacher_records=[self.record],
            gold_records=[gold],
            heldout_records=[],
            ontology=self.ontology,
        )
        self.assertEqual(len(items), 1)
        self.assertTrue(all(item["source"] == "gold" for item in items))
        self.assertAlmostEqual(items[0]["sample_weight"], 1.5)
        self.assertTrue(all(item["negative_target"] for item in items))
        self.assertGreaterEqual(
            len(items[0]["available_counterfactual_types"]), 2
        )

    def test_teacher_calibration_uses_strict_gold_agreement(self) -> None:
        calibration = calibrate_teacher_against_gold(
            [self.record],
            [self.record],
        )
        self.assertEqual(calibration["overlap_records"], 1)
        self.assertEqual(calibration["entity_precision"], 1.0)
        self.assertEqual(calibration["relation_precision"], 1.0)
        self.assertEqual(calibration["teacher_weight"], 0.7)

    def test_ours_repeats_and_upweights_gold(self) -> None:
        gold = {**self.record, "case_id": "gold_1"}
        teacher = {**self.record, "case_id": "teacher_1"}
        items = build_training_records(
            "ours",
            teacher_records=[teacher],
            gold_records=[gold],
            heldout_records=[],
            ontology=self.ontology,
            teacher_weight=0.25,
            gold_weight=2.0,
            gold_repeats=3,
        )
        self.assertEqual(len(items), 4)
        gold_items = [item for item in items if item["source"] == "gold"]
        teacher_items = [
            item for item in items if item["source"] == "teacher_structured"
        ]
        self.assertEqual(len(gold_items), 3)
        self.assertTrue(all(item["sample_weight"] == 2.0 for item in gold_items))
        self.assertEqual(teacher_items[0]["sample_weight"], 0.25)

    def test_teacher_aligned_uses_teacher_target_on_gold_overlap(self) -> None:
        gold = {**self.record, "event_label": "工艺纪律违规"}
        items = build_training_records(
            "ours",
            teacher_records=[self.record],
            gold_records=[gold],
            heldout_records=[],
            ontology=self.ontology,
            teacher_aligned=True,
            teacher_weight=1.0,
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["source"], "teacher_aligned")
        target = __import__("json").loads(items[0]["target"])
        self.assertEqual(target["event_label"], self.record["event_label"])
        self.assertAlmostEqual(items[0]["sample_weight"], 0.8666666667)

    def test_entity_stage_removes_relations_from_targets(self) -> None:
        item = build_training_records(
            "ours",
            teacher_records=[self.record],
            heldout_records=[],
            ontology=self.ontology,
            stage="entity",
        )[0]
        target = __import__("json").loads(item["target"])
        negative = __import__("json").loads(item["negative_target"])
        self.assertEqual(target["relations"], [])
        self.assertEqual(negative["relations"], [])
        self.assertIn("只抽取分类和实体", item["system"])

    def test_relation_stage_fixes_entities_in_prompt(self) -> None:
        item = build_training_records(
            "ours",
            teacher_records=[self.record],
            heldout_records=[],
            ontology=self.ontology,
            stage="relation",
        )[0]
        target = __import__("json").loads(item["target"])
        self.assertEqual(
            target,
            {"relations": [{"head": "E1", "tail": "E2", "type": "CAUSES"}]},
        )
        self.assertIn("固定实体清单", item["prompt"])
        self.assertEqual(item["prompt"], build_relation_user_prompt(self.record))

    def test_two_stage_builds_two_tasks_per_case(self) -> None:
        items = build_training_records(
            "ours",
            teacher_records=[self.record],
            heldout_records=[],
            ontology=self.ontology,
            stage="two_stage",
        )
        self.assertEqual(len(items), 2)
        self.assertEqual({item["stage"] for item in items}, {"entity", "relation"})

    def test_enhanced_teacher_validation_requires_complete_pipeline(self) -> None:
        enhanced = {
            **self.record,
            "mode": "with_wsr_ontology_gate_enhanced",
            "pipeline_components": [
                "wsr_ontology",
                "entity_refinement_second_pass",
                "entity_consensus_filter",
                "relation_endpoint_remapping",
                "wsr_gate",
            ],
            "entity_refinement_success": True,
            "relations_before_gate": self.record["relations"],
            "rejected_relations": [],
        }
        validate_enhanced_teacher_records([enhanced])
        with self.assertRaisesRegex(ValueError, "Enhanced teacher"):
            validate_enhanced_teacher_records([self.record])


if __name__ == "__main__":
    unittest.main()
