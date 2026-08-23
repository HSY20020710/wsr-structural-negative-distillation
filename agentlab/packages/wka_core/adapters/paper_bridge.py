from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import yaml

from ..models import GateDecision, GateState


class PaperCodeBridge:
    """Thin adapter over the public paper reproduction code.

    The bridge expects a directory that contains `src/ontology/gate.py` and
    `configs/wsr_ontology.yaml`. In the public GitHub layout this is normally
    the repository root, while standalone development can still use a local
    `vendor/paper_reproduction/` copy.
    """

    def __init__(self, paper_code_dir: str | Path):
        self.paper_code_dir = Path(paper_code_dir).resolve()
        if not (self.paper_code_dir / "src/ontology/gate.py").exists():
            raise FileNotFoundError(
                f"Cannot find public paper reproduction code under {self.paper_code_dir}"
            )
        if str(self.paper_code_dir) not in sys.path:
            sys.path.insert(0, str(self.paper_code_dir))
        from src.ontology.gate import check_relation

        self._check_relation = check_relation
        ontology_path = self.paper_code_dir / "configs/wsr_ontology.yaml"
        self.ontology = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))

    def gate_relation(self, relation: dict[str, Any], entities: list[dict[str, Any]]) -> GateDecision:
        emap = {e.get("id"): e for e in entities}
        head = emap.get(relation.get("head"))
        tail = emap.get(relation.get("tail"))
        rel_type = relation.get("type")
        known_rel = set(self.ontology.get("relation_types", []))
        if not head or not tail:
            return GateDecision(
                state=GateState.UNKNOWN,
                layer="Unknown",
                reason="missing relation endpoint",
                rule_id="PRECHECK_ENDPOINT",
                relation=relation,
            )
        if not head.get("type") or not tail.get("type") or rel_type not in known_rel:
            return GateDecision(
                state=GateState.UNKNOWN,
                layer="Unknown",
                reason="unmapped entity/relation type",
                rule_id="PRECHECK_TYPE",
                relation=relation,
            )
        result = self._check_relation(head["type"], rel_type, tail["type"], self.ontology)
        state = GateState.PASS if result["passed"] else GateState.CONFLICT
        conflict_type = result.get("conflict_type", "")
        layer = {
            "wuli": "Wuli",
            "shili": "Shili",
            "renli": "Renli",
            "multi_conflict": "Multi",
            "positive": "WSR",
        }.get(conflict_type, "WSR")
        return GateDecision(
            state=state,
            layer=layer,
            reason=result["reason"],
            rule_id=f"DOMAIN_RANGE::{rel_type}",
            relation=relation,
        )