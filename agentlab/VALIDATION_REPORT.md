# Validation Report

Public-release validation scope:

- The AgentLab package contains only public code, configuration, documentation, and anonymized demo examples.
- Private report data, private annotations, prediction caches, checkpoints, logs, paper manuscripts, and exact paper experiment outputs are excluded.
- The demo runner exercises the public WSR Gate path and writes local evidence artifacts under `runs/`, which is ignored by Git.

Recommended local checks:

```bash
cd agentlab
pytest -q
python scripts/run_demo.py
```