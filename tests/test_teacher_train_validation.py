from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TeacherTrainValidationTests(unittest.TestCase):
    def test_complete_teacher_record_passes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = {
            "case_id": "case_train",
            "text": "焊前未清理导致气孔。",
        }
        prediction = {
            **source,
            "parse_success": True,
            "pipeline_components": [
                "wsr_ontology",
                "entity_refinement_second_pass",
                "entity_consensus_filter",
                "relation_endpoint_remapping",
                "wsr_gate",
            ],
            "entities": [
                {
                    "id": "E1",
                    "text": "未清理",
                    "start": 2,
                    "end": 5,
                    "type": "S_CAUSE",
                    "wsr_layer": "Shili",
                },
                {
                    "id": "E2",
                    "text": "气孔",
                    "start": 7,
                    "end": 9,
                    "type": "W_DEFECT",
                    "wsr_layer": "Wuli",
                },
            ],
            "relations": [{"head": "E1", "tail": "E2", "type": "CAUSES"}],
            "rejected_relations": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_path = directory / "source.jsonl"
            prediction_path = directory / "prediction.jsonl"
            source_path.write_text(
                json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            prediction_path.write_text(
                json.dumps(prediction, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/validate_teacher_train.py"),
                    "--input",
                    str(prediction_path),
                    "--source",
                    str(source_path),
                ],
                cwd=root,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(
                prediction_path.with_suffix(".validation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
