from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class StudentSplitTests(unittest.TestCase):
    def test_example_split_is_disjoint(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/prepare_student_data.py"),
                    "--raw",
                    str(root / "examples/input/example_reports.txt"),
                    "--test_gold",
                    str(root / "examples/expected/example_gold.jsonl"),
                    "--output_dir",
                    str(output),
                    "--dev_size",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(
                (output / "split_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["raw_cases"], 3)
            self.assertEqual(manifest["unique_raw_case_ids"], 3)
            self.assertEqual(manifest["fixed_test"], 1)
            self.assertEqual(manifest["dev_candidates"], 1)
            self.assertEqual(manifest["train_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
