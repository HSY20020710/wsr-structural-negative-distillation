import unittest
from pathlib import Path

from packages.wka_core import AgentLabOrchestrator, ExperimentRequest

BASE = Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def test_gate_uses_public_paper_rule(self):
        bridge = AgentLabOrchestrator(BASE).paper
        entities = [
            {"id": "a", "type": "S_CAUSE", "text": "x"},
            {"id": "b", "type": "W_DEFECT", "text": "y"},
        ]
        decision = bridge.gate_relation({"head": "a", "tail": "b", "type": "CAUSES"}, entities)
        self.assertEqual(decision.state.value, "PASS")
        conflict = bridge.gate_relation({"head": "b", "tail": "a", "type": "CAUSES"}, entities)
        self.assertEqual(conflict.state.value, "CONFLICT")

    def test_unknown_precheck(self):
        bridge = AgentLabOrchestrator(BASE).paper
        decision = bridge.gate_relation({"head": "x", "tail": "y", "type": "CAUSES"}, [])
        self.assertEqual(decision.state.value, "UNKNOWN")

    def test_demo_run_persists_evidence(self):
        orchestrator = AgentLabOrchestrator(BASE)
        run = orchestrator.run(ExperimentRequest(question="demo"))
        self.assertEqual(run.status, "completed")
        self.assertTrue((BASE / "runs" / run.run_id / "run_manifest.json").exists())
        self.assertTrue(any(event.actor == "GateAuditAgent" for event in run.events))


if __name__ == "__main__":
    unittest.main()