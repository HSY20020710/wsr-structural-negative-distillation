from pathlib import Path
import sys
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from packages.wka_core import AgentLabOrchestrator, ExperimentRequest
run=AgentLabOrchestrator(BASE).run(ExperimentRequest(question='复现结构冲突路由并展示证据链',mode='demo'))
print(run.model_dump_json(indent=2))
