from __future__ import annotations

from typing import Any

# Demo-only synthetic cases. Private ship welding reports and annotations are not
# distributed in the public repository.

_SAMPLE_ENTITIES = [
    {"id": "E1", "text": "焊前清理不到位", "type": "S_CAUSE"},
    {"id": "E2", "text": "气孔", "type": "W_DEFECT"},
    {"id": "E3", "text": "责任心不足", "type": "R_RESPONSIBILITY"},
    {"id": "E4", "text": "施工人员", "type": "R_PERSON"},
]

_SAMPLE_RELATIONS = [
    {"head": "E1", "tail": "E2", "type": "CAUSES"},
    {"head": "E2", "tail": "E1", "type": "CAUSES"},
    {"head": "E3", "tail": "E4", "type": "RESPONSIBLE_FOR"},
]

_CASES = [
    {
        "case_id": "demo_001",
        "text": "焊前清理不到位，存在油污，局部出现气孔。",
        "split": "demo",
        "entities": _SAMPLE_ENTITIES,
        "relations": _SAMPLE_RELATIONS,
    },
    {
        "case_id": "demo_002",
        "text": "施工人员未按工艺要求施焊，管理人员检查不到位。",
        "split": "demo",
        "entities": [
            {"id": "F1", "text": "未按工艺施焊", "type": "S_CAUSE"},
            {"id": "F2", "text": "夹渣", "type": "W_DEFECT"},
            {"id": "F3", "text": "施工人员", "type": "R_PERSON"},
            {"id": "F4", "text": "管理检查缺位", "type": "R_RESPONSIBILITY"},
        ],
        "relations": [
            {"head": "F1", "tail": "F2", "type": "CAUSES"},
            {"head": "F3", "tail": "F4", "type": "RESPONSIBLE_FOR"},
        ],
    },
]


class DemoRepository:
    """Lightweight read-only repository for public demo cases."""

    def list_cases(self) -> list[dict[str, Any]]:
        return [
            {"case_id": c["case_id"], "text": c["text"], "split": c["split"]}
            for c in _CASES
        ]

    def get(self, case_id: str) -> dict[str, Any] | None:
        for c in _CASES:
            if c["case_id"] == case_id:
                return c
        return None