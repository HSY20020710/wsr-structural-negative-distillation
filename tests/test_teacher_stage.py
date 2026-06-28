from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.data.parse_welding_cases import parse_welding_cases
from src.gold_validation import validate_gold_record
from src.metrics.teacher_metrics import evaluate_teacher
from src.ontology.gate import apply_wsr_gate, check_relation
from src.teacher.parser import extract_json_from_response, normalize_teacher_output
from src.teacher.conditioned_relations import (
    build_relation_candidates,
    parse_conditioned_relation_output,
)
from src.teacher.entity_refinement import (
    build_entity_refinement_prompt,
    filter_consensus_entities,
    parse_refined_entities,
    remap_relations,
)
from src.teacher.prompt_templates import (
    build_with_wsr_ontology_prompt,
    build_without_ontology_prompt,
)
from src.teacher.qwen_client import normalize_api_url


ROOT = Path(__file__).resolve().parents[1]


class TeacherStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ontology = yaml.safe_load(
            (ROOT / "configs" / "wsr_ontology.yaml").read_text(encoding="utf-8")
        )
        cls.record = {
            "case_id": "case_1",
            "original_case_no": "案例1",
            "occurrence_time": "2020年4月",
            "occurrence_stage": "分段阶段",
            "problem_phenomenon": "焊缝出现气孔。",
            "cause_analysis": "焊前未清除氧化皮。",
            "text": (
                "发生时间：2020年4月\n"
                "发生阶段：分段阶段\n"
                "问题现象：焊缝出现气孔。\n"
                "问题原因分析：焊前未清除氧化皮。"
            ),
        }

    def test_legacy_parser_and_missing_warning(self) -> None:
        content = """案例110
一、发生时间
2020年4月
二、发生阶段
分段阶段
三、问题现象
焊缝出现气孔。
四、问题原因
焊前未清除氧化皮。

案例111
一、发生时间
2020年5月
二、发生阶段
加工阶段
三、问题现象
焊缝开裂。
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.txt"
            path.write_text(content, encoding="utf-8")
            records, stats = parse_welding_cases(path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["case_id"], "case_110")
        self.assertEqual(records[0]["cause_analysis"], "焊前未清除氧化皮。")
        self.assertIn("case_111", stats["missing_cause_cases"])

    def test_control_prompt_has_no_wsr_words(self) -> None:
        prompt = build_without_ontology_prompt(self.record)
        self.assertNotIn("WSR", prompt)
        self.assertNotIn("Wuli", prompt)
        self.assertNotIn("Shili", prompt)
        self.assertNotIn("Renli", prompt)
        self.assertNotIn("allowed_relations", prompt)
        ontology_prompt = build_with_wsr_ontology_prompt(self.record, self.ontology)
        self.assertIn("WSR", ontology_prompt)
        self.assertIn("W_DEFECT", ontology_prompt)
        self.assertIn("CONTRIBUTES_TO", ontology_prompt)
        self.assertIn("OCCURS_AT", ontology_prompt)
        self.assertNotIn("CONTAINS、", ontology_prompt)

    def test_json_extraction_and_span_repair(self) -> None:
        response = """结果如下：
```json
{"defect_label":"气孔","event_label":"工艺纪律违规","entities":[
{"id":"x","text":"未清除氧化皮","start":null,"end":null,
"type":"S_CAUSE","wsr_layer":"Shili","confidence":0.9},
{"id":"y","text":"气孔","start":999,"end":1000,
"type":"W_DEFECT","wsr_layer":"Wuli","confidence":0.9}],
"relations":[{"head":"x","tail":"y","type":"CAUSES","confidence":0.8}]}
```
"""
        parsed = extract_json_from_response(response)
        self.assertTrue(parsed["parse_success"])
        normalized = normalize_teacher_output(
            parsed["data"], self.record, self.ontology
        )
        self.assertEqual(len(normalized["entities"]), 2)
        for entity in normalized["entities"]:
            self.assertEqual(
                self.record["text"][entity["start"]:entity["end"]], entity["text"]
            )

    def test_span_repair_uses_nearest_repeated_occurrence(self) -> None:
        record = {
            "case_id": "repeated",
            "text": "气孔已发现，返修后再次发现气孔。",
        }
        second_start = record["text"].rindex("气孔")
        normalized = normalize_teacher_output(
            {
                "defect_label": "气孔",
                "event_label": "其他",
                "entities": [
                    {
                        "id": "x",
                        "text": "气孔",
                        "start": second_start + 1,
                        "end": second_start + 3,
                        "type": "W_DEFECT",
                    }
                ],
                "relations": [],
            },
            record,
            self.ontology,
        )
        self.assertEqual(normalized["entities"][0]["start"], second_start)

    def test_gate_keeps_allowed_and_rejects_reverse_cause(self) -> None:
        allowed = check_relation(
            "S_CAUSE", "CAUSES", "W_DEFECT", self.ontology
        )
        rejected = check_relation(
            "W_DEFECT", "CAUSES", "S_CAUSE", self.ontology
        )
        self.assertTrue(allowed["passed"])
        self.assertFalse(rejected["passed"])
        record = {
            "entities": [
                {"id": "E1", "text": "原因", "type": "S_CAUSE"},
                {"id": "E2", "text": "气孔", "type": "W_DEFECT"},
            ],
            "relations": [
                {"head": "E1", "tail": "E2", "type": "CAUSES"},
                {"head": "E2", "tail": "E1", "type": "CAUSES"},
            ],
        }
        gated = apply_wsr_gate(record, self.ontology)
        self.assertEqual(len(gated["relations"]), 1)
        self.assertEqual(len(gated["rejected_relations"]), 1)

    def test_gate_enforces_new_relation_semantics(self) -> None:
        self.assertTrue(
            check_relation(
                "R_RESPONSIBILITY",
                "CONTRIBUTES_TO",
                "W_DEFECT",
                self.ontology,
            )["passed"]
        )
        self.assertFalse(
            check_relation(
                "R_RESPONSIBILITY", "CAUSES", "W_DEFECT", self.ontology
            )["passed"]
        )
        self.assertTrue(
            check_relation(
                "W_DEFECT", "OCCURS_AT", "W_LOCATION", self.ontology
            )["passed"]
        )
        self.assertFalse(
            check_relation(
                "W_LOCATION", "OCCURS_AT", "W_DEFECT", self.ontology
            )["passed"]
        )

    def test_api_url_typo_is_repaired(self) -> None:
        url, warning = normalize_api_url(
            "http://localhost:8000V1/api/generate"
        )
        self.assertEqual(url, "http://localhost:8000/V1/api/generate")
        self.assertIsNotNone(warning)

    def test_perfect_metrics(self) -> None:
        start_cause = self.record["text"].index("未清除氧化皮")
        start_defect = self.record["text"].index("气孔")
        prediction = {
            "case_id": "case_1",
            "parse_success": True,
            "defect_label": "气孔",
            "event_label": "工艺纪律违规",
            "entities": [
                {
                    "id": "E1",
                    "text": "未清除氧化皮",
                    "start": start_cause,
                    "end": start_cause + len("未清除氧化皮"),
                    "type": "S_CAUSE",
                },
                {
                    "id": "E2",
                    "text": "气孔",
                    "start": start_defect,
                    "end": start_defect + 2,
                    "type": "W_DEFECT",
                },
            ],
            "relations": [{"head": "E1", "tail": "E2", "type": "CAUSES"}],
            "triples": [
                {"head_text": "未清除氧化皮", "relation": "CAUSES", "tail_text": "气孔"}
            ],
        }
        gold = {**self.record, **prediction}
        report = evaluate_teacher([prediction], [gold], self.ontology)
        self.assertEqual(report["entity_span_type_f1"], 1.0)
        self.assertEqual(report["triple_f1"], 1.0)
        self.assertEqual(report["cvr_all"], 0.0)

    def test_metrics_report_gate_retention_and_pre_gate_f1(self) -> None:
        start_cause = self.record["text"].index("未清除氧化皮")
        start_defect = self.record["text"].index("气孔")
        entities = [
            {
                "id": "E1",
                "text": "未清除氧化皮",
                "start": start_cause,
                "end": start_cause + len("未清除氧化皮"),
                "type": "S_CAUSE",
            },
            {
                "id": "E2",
                "text": "气孔",
                "start": start_defect,
                "end": start_defect + len("气孔"),
                "type": "W_DEFECT",
            },
        ]
        gold = {
            **self.record,
            "entities": entities,
            "relations": [{"head": "E1", "tail": "E2", "type": "CAUSES"}],
        }
        prediction = {
            **gold,
            "parse_success": True,
            "relations": [],
            "relations_before_gate": gold["relations"],
        }
        report = evaluate_teacher([prediction], [gold], self.ontology)
        self.assertEqual(report["relation_f1"], 0.0)
        self.assertEqual(report["relation_f1_before_gate"], 1.0)
        self.assertEqual(report["gate_relation_retention"], 0.0)

    def test_gold_validator_accepts_long_and_short_wsr_names(self) -> None:
        base = {
            "case_id": "case_validation",
            "text": "abc",
            "relations": [],
            "annotation": {"status": "reviewed"},
        }
        for field, value in (("wsr", "W"), ("wsr_layer", "Wuli")):
            record = {
                **base,
                "entities": [
                    {
                        "id": "E1",
                        "text": "abc",
                        "start": 0,
                        "end": 3,
                        "type": "W_DEFECT",
                        field: value,
                    }
                ],
            }
            self.assertEqual(validate_gold_record(record), [])

    def test_conditioned_relation_candidates_and_parser(self) -> None:
        record = {
            "entities": [
                {"id": "E1", "text": "未清理", "type": "S_CAUSE"},
                {"id": "E2", "text": "气孔", "type": "W_DEFECT"},
                {"id": "E3", "text": "焊缝", "type": "W_LOCATION"},
            ]
        }
        candidates = build_relation_candidates(record, self.ontology)
        cause = next(
            candidate
            for candidate in candidates
            if candidate["head"] == "E1" and candidate["tail"] == "E2"
        )
        location = next(
            candidate
            for candidate in candidates
            if candidate["head"] == "E2" and candidate["tail"] == "E3"
        )
        self.assertIn("CAUSES", cause["allowed_labels"])
        self.assertIn("NO_RELATION", cause["allowed_labels"])
        self.assertEqual(
            location["allowed_labels"], ["OCCURS_AT", "NO_RELATION"]
        )
        parsed = parse_conditioned_relation_output(
            json.dumps(
                {
                    "decisions": [
                        {
                            "candidate_id": cause["candidate_id"],
                            "label": "CAUSES",
                            "confidence": 1,
                        }
                    ]
                }
            ),
            candidates,
        )
        self.assertTrue(parsed["parse_success"])
        self.assertEqual(len(parsed["relations"]), 1)

    def test_entity_refinement_prompt_and_parser(self) -> None:
        prompt = build_entity_refinement_prompt(
            {
                **self.record,
                "defect_label": "气孔",
                "event_label": "工艺纪律违规",
                "entities": [],
            },
            self.ontology,
        )
        self.assertIn("实体边界校准器", prompt)
        self.assertIn("不同语义类型允许嵌套", prompt)
        entity_text = self.record["problem_phenomenon"][-3:-1]
        start = self.record["text"].index(entity_text)
        response = json.dumps(
            {
                "defect_label": "气孔",
                "event_label": "工艺纪律违规",
                "entities": [
                    {
                        "id": "x",
                        "text": entity_text,
                        "start": start,
                        "end": start + len(entity_text),
                        "type": "W_DEFECT",
                        "confidence": 1,
                    }
                ],
                "relations": [],
            },
            ensure_ascii=False,
        )
        parsed = parse_refined_entities(response, self.record, self.ontology)
        self.assertTrue(parsed["parse_success"])
        self.assertEqual(parsed["entities"][0]["text"], entity_text)

    def test_entity_consensus_and_relation_remap(self) -> None:
        first_pass = [
            {
                "id": "old1",
                "text": "焊缝气孔较多",
                "start": 10,
                "end": 16,
                "type": "W_DEFECT",
            },
            {
                "id": "old2",
                "text": "施工人员",
                "start": 20,
                "end": 24,
                "type": "R_PERSON",
            },
        ]
        refined = [
            {
                "id": "E1",
                "text": "气孔",
                "start": 12,
                "end": 14,
                "type": "W_DEFECT",
            },
            {
                "id": "E2",
                "text": "锈",
                "start": 30,
                "end": 31,
                "type": "W_PHYSICAL_CONDITION",
            },
            {
                "id": "E3",
                "text": "额外部件",
                "start": 40,
                "end": 44,
                "type": "W_COMPONENT",
            },
        ]
        filtered = filter_consensus_entities(refined, first_pass)
        self.assertEqual([entity["id"] for entity in filtered], ["E1", "E2"])
        relations = remap_relations(
            first_pass,
            [{"head": "old2", "tail": "old1", "type": "CONTRIBUTES_TO"}],
            [
                *filtered,
                {
                    "id": "E4",
                    "text": "施工人员",
                    "start": 20,
                    "end": 24,
                    "type": "R_PERSON",
                },
            ],
        )
        self.assertEqual(relations[0]["head"], "E4")
        self.assertEqual(relations[0]["tail"], "E1")


if __name__ == "__main__":
    unittest.main()
