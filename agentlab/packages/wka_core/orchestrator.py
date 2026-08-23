from __future__ import annotations

from pathlib import Path
import uuid
from .models import ExperimentRequest, ExperimentRun, EvidenceEvent
from .evidence import EvidenceStore
from .adapters.paper_bridge import PaperCodeBridge


PUBLIC_RESULT_NOTE = {
    "status": "not_included",
    "reason": "Exact paper experiment outputs are private and are not distributed in the public repository.",
}


class AgentLabOrchestrator:
    def __init__(self, base_dir: str | Path):
        self.base = Path(base_dir).resolve()
        self.store = EvidenceStore(self.base / "runs")
        self.paper = PaperCodeBridge(self._resolve_paper_code_dir())
        self.runs = {}

    def _resolve_paper_code_dir(self) -> Path:
        candidates = [
            self.base / "vendor/paper_reproduction",
            self.base.parent,
            self.base,
        ]
        for candidate in candidates:
            if (candidate / "src/ontology/gate.py").exists() and (
                candidate / "configs/wsr_ontology.yaml"
            ).exists():
                return candidate
        raise FileNotFoundError("Cannot locate public paper reproduction code.")

    def _event(self, run_id, actor, event_type, status="PASS", provenance=None):
        ev = EvidenceEvent(
            event_id="EVT-" + uuid.uuid4().hex[:8],
            run_id=run_id,
            actor=actor,
            event_type=event_type,
            status=status,
            provenance=provenance or {},
        )
        self.store.append(ev)
        return ev

    def run(self, req: ExperimentRequest):
        run_id = "EXP-" + uuid.uuid4().hex[:10].upper()
        events = []
        events.append(
            self._event(
                run_id,
                "ResearchAgent",
                "parse_research_question",
                provenance={"question": req.question},
            )
        )
        events.append(
            self._event(
                run_id,
                "ExperimentAgent",
                "select_template",
                provenance={"template": req.template, "seed": req.seed},
            )
        )
        candidate = {
            "entities": [
                {"id": "E1", "text": "焊前清理不到位", "type": "S_CAUSE"},
                {"id": "E2", "text": "气孔", "type": "W_DEFECT"},
                {"id": "E3", "text": "施工人员", "type": "R_PERSON"},
                {"id": "E4", "text": "责任心不足", "type": "R_RESPONSIBILITY"},
            ],
            "relations": [
                {"head": "E1", "tail": "E2", "type": "CAUSES"},
                {"head": "E3", "tail": "E4", "type": "RESPONSIBLE_FOR"},
                {"head": "E2", "tail": "E1", "type": "CAUSES"},
            ],
        }
        events.append(
            self._event(
                run_id,
                "TeacherAgent",
                "load_demo_candidate",
                provenance={"mode": "synthetic_demo" if req.mode == "demo" else "real_adapter_required"},
            )
        )
        decisions = [
            self.paper.gate_relation(r, candidate["entities"]).model_dump(mode="json")
            for r in candidate["relations"]
        ]
        status = "CONFLICT" if any(d["state"] == "CONFLICT" for d in decisions) else "PASS"
        events.append(
            self._event(
                run_id,
                "GateAuditAgent",
                "gate_route",
                status=status,
                provenance={"decisions": decisions},
            )
        )
        conflicts = [d for d in decisions if d["state"] == "CONFLICT"]
        events.append(
            self._event(
                run_id,
                "NegativeBuilderAgent",
                "build_structural_negatives",
                provenance={"conflict_count": len(conflicts), "policy": "demo / provenance-retained"},
            )
        )
        if req.mode == "real":
            events.append(
                self._event(
                    run_id,
                    "StudentAgent",
                    "real_runner_required",
                    status="UNKNOWN",
                    provenance={"note": "Configure private data and model paths before long-running training."},
                )
            )
        else:
            events.append(
                self._event(
                    run_id,
                    "StudentAgent",
                    "demo_flow_completed",
                    provenance={"paper_results": PUBLIC_RESULT_NOTE},
                )
            )
        events.append(
            self._event(
                run_id,
                "EvaluationAgent",
                "public_result_policy",
                provenance={"paper_results": PUBLIC_RESULT_NOTE},
            )
        )
        summary = {
            "paper_results": PUBLIC_RESULT_NOTE,
            "gate_demo": decisions,
            "candidate": candidate,
            "data_policy": "private manually annotated and reviewed evaluation data is not included",
        }
        run = ExperimentRun(
            run_id=run_id,
            status="completed" if req.mode == "demo" else "adapter_ready",
            question=req.question,
            template=req.template,
            mode=req.mode,
            seed=req.seed,
            summary=summary,
            events=events,
        )
        self.store.write_json(run_id, "run_manifest.json", run.model_dump(mode="json"))
        self.store.write_json(run_id, "gate_trace.json", decisions)
        self.store.write_json(run_id, "public_result_policy.json", PUBLIC_RESULT_NOTE)
        self.runs[run_id] = run
        return run

    def get(self, run_id):
        return self.runs.get(run_id)