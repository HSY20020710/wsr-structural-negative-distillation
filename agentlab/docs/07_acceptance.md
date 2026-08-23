# Acceptance Checklist

Before publishing the AgentLab add-on, check that:

- `pytest -q` passes from `agentlab/`;
- `python scripts/run_demo.py` completes without private data;
- no private reports, annotations, caches, checkpoints, logs, or manuscripts are present;
- no exact private paper result table is embedded in the UI or API;
- `data/private/` and `runs/` are ignored by Git;
- the bridge calls the public paper code from the repository root or a local standalone copy.