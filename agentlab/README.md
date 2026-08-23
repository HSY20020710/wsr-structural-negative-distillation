# Welding Knowledge AgentLab

Welding Knowledge AgentLab is a paper-linked research workbench for the WSR-guided structural negative distillation pipeline. It wraps the public reproduction code in this repository with lightweight agent orchestration, evidence logging, a demo API, and a small React/Vite starter interface.

This public package is for code inspection and workflow demonstration. It does not include private ship welding reports, private annotations, full prediction caches, model checkpoints, training logs, manuscript files, or paper experiment outputs.

## What Is Included

- `packages/wka_core/`: agent orchestration, evidence store, demo repository, and paper-code bridge.
- `apps/api/`: FastAPI service for demo cases, WSR Gate checks, and experiment runs.
- `apps/web/`: minimal React/Vite starter frontend.
- `scripts/`: demo runner, API launcher, and private-data import helper.
- `configs/`: platform and agent graph configuration.
- `data/demo/`: anonymized synthetic examples for format and smoke testing.
- `docs/`: architecture, paper-code mapping, API contract, and acceptance notes.

## Relationship To The Paper Code

The AgentLab does not reimplement the paper method. It calls the repository's existing paper reproduction modules, especially the WSR ontology and Gate logic under the repository root.

When this folder is placed inside `wsr-structural-negative-distillation/agentlab`, the bridge resolves the paper code from the parent repository. If you run AgentLab as a standalone package, provide a local `vendor/paper_reproduction/` folder with the same public reproduction code.

## Quick Start

```bash
cd agentlab
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-platform.txt
pytest -q
python scripts/run_demo.py
python scripts/start_api.py
```

For the optional web starter:

```bash
cd agentlab/apps/web
npm install
npm run dev
```

## Private Data

Private evaluation data can be mounted locally under `data/private/` for authorized experiments. That directory is ignored by `.gitignore` and should not be committed.

```bash
python scripts/import_private_data.py path/to/private_data_pack.zip
```

The public repository only provides anonymized synthetic examples.