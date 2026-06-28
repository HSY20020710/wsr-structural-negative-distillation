from __future__ import annotations

import unittest

from src.data_pipeline import (
    SplitRatios,
    exact_deduplicate,
    group_near_duplicates,
    grouped_split,
    parse_new_cases,
    parse_old_cases,
)
from src.extraction_schema import validate_prediction_structure
from src.data.parse_welding_cases import _parse_annual_cases


class DataPipelineTests(unittest.TestCase):
    def test_parse_legacy_case(self) -> None:
        text = """案例1
一、发生时间
2021年1月
二、发生阶段
分段阶段
三、问题现象
焊缝出现气孔。
四、问题原因分析
焊前未清理。
"""
        record = parse_old_cases(text)[0]
        self.assertEqual(record["year"], 2021)
        self.assertIn("气孔", record["text"])
        self.assertIn("未清理", record["text"])

    def test_parse_structured_case(self) -> None:
        text = """【2024年-案例1】
• 项目类型：造船
• 工程项目：测试船
• 问题分类：图纸工艺
• 问题简述：焊缝出现气孔。
• 责任部门：制造部
• 发生时间：2024-04
##########
"""
        record = parse_new_cases(text)[0]
        self.assertEqual(record["project"], "测试船")
        self.assertEqual(record["year"], 2024)

    def test_project_parser_keeps_only_welding_annual_cases(self) -> None:
        text = """【2024年-案例1】
• 项目类型：造船
• 问题简述：焊缝出现气孔。
• 发生时间：2024-04
##########
【2024年-案例2】
• 项目类型：修船
• 问题简述：吊装设备损坏。
• 发生时间：2024-04
"""
        records = _parse_annual_cases(text)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["case_id"], "annual_2024_1")

    def test_grouped_split_has_no_group_leakage(self) -> None:
        records = []
        for index, text in enumerate(["焊缝出现气孔", "焊缝出现气孔", "焊缝出现裂纹"]):
            records.append(
                {
                    "record_id": f"r{index}",
                    "source_format": "legacy",
                    "source_case_id": str(index),
                    "year": 2024,
                    "text": text,
                    "dedup_text": text,
                }
            )
        unique, _ = exact_deduplicate(records)
        grouped = group_near_duplicates(unique, threshold=0.9)
        splits, _ = grouped_split(grouped, SplitRatios(0.6, 0.2, 0.2), seed=1)
        group_sets = [
            {record["group_id"] for record in splits[name]}
            for name in ["train", "dev", "test"]
        ]
        self.assertFalse(group_sets[0] & group_sets[1])
        self.assertFalse(group_sets[0] & group_sets[2])
        self.assertFalse(group_sets[1] & group_sets[2])

    def test_teacher_structure_validation(self) -> None:
        text = "焊前未清除氧化皮，导致焊缝出现气孔。"
        schema = {
            "entity_types": {
                "ProcessViolation": "S",
                "WeldingDefect": "W",
            },
            "relation_types": ["Causes"],
        }
        prediction = {
            "defect_labels": ["气孔"],
            "event_labels": [],
            "entities": [
                {
                    "id": "e1",
                    "text": "未清除氧化皮",
                    "start": 2,
                    "end": 8,
                    "type": "ProcessViolation",
                    "wsr": "S",
                },
                {
                    "id": "e2",
                    "text": "气孔",
                    "start": 15,
                    "end": 17,
                    "type": "WeldingDefect",
                    "wsr": "W",
                },
            ],
            "relations": [{"head": "e1", "type": "Causes", "tail": "e2"}],
        }
        self.assertEqual(validate_prediction_structure(prediction, text, schema), [])


if __name__ == "__main__":
    unittest.main()
