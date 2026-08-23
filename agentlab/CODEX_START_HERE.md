# AgentLab Public Release Notes

Start here if you are extending the paper-linked AgentLab.

## Public Boundary

Allowed in GitHub:

- Agent orchestration and evidence logging code.
- API and minimal web starter code.
- Configuration schemas and demo templates.
- Anonymized synthetic examples.
- Documentation that describes how the system connects to the public paper code.

Do not commit:

- Private ship welding reports or annotations.
- Full teacher/student prediction caches.
- Model weights, checkpoints, logs, or run outputs.
- Manuscript Word/PDF files.
- API keys or local environment files.
- Exact private paper result tables inside UI demos.

## Development Entry Points

```bash
pytest -q
python scripts/run_demo.py
python scripts/start_api.py
```

The demo path should remain runnable without private data.