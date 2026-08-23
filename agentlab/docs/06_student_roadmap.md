# Student Roadmap

A student or collaborator can use this package in three stages.

## Stage 1: Public Demo

Run the demo flow with synthetic examples:

```bash
pytest -q
python scripts/run_demo.py
python scripts/start_api.py
```

Expected outcome: understand the AgentLab event flow, WSR Gate decisions, and evidence files.

## Stage 2: Authorized Private Experiments

Mount private data locally under `data/private/`. Do not commit that directory. Configure model paths and run the paper reproduction scripts from the repository root.

## Stage 3: Platform Extension

Add new tools, agents, or UI views only after preserving these boundaries:

- demo outputs are not paper results;
- private data remains local;
- every Gate decision should be traceable to code or configuration;
- every long-running experiment should write a manifest.