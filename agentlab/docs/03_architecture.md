# Architecture

The public AgentLab has five layers:

1. **API Layer**: FastAPI endpoints for demo cases, Gate checks, and experiment runs.
2. **Agent Orchestration**: a lightweight coordinator that turns a research request into tool-backed events.
3. **Paper Bridge**: an adapter that calls the public WSR ontology and Gate logic from the parent repository.
4. **Evidence Store**: local JSON/JSONL run artifacts under `runs/`, ignored by Git.
5. **Demo Data**: anonymized synthetic records under `data/demo/`.

Private reports, annotations, caches, checkpoints, and exact paper result outputs are external to this public package.