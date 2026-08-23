"""FastAPI entry point for the public AgentLab demo.

The API wraps the public paper reproduction code through packages/wka_core and
keeps private data, exact paper results, and run artifacts out of GitHub.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from packages.wka_core import ExperimentRequest, AgentLabOrchestrator
from packages.wka_core.demo_repository import DemoRepository

BASE = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE / "frontend"
RUNS_DIR = BASE / "runs"

app = FastAPI(title="Welding Knowledge AgentLab API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

orch = AgentLabOrchestrator(BASE)
repo = DemoRepository()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "mode": "public-demo",
        "frontend": "mounted" if FRONTEND_DIR.exists() else "not_mounted",
    }


@app.get("/api/v1/project/summary")
def summary() -> dict:
    return {
        "name": "Welding Knowledge AgentLab",
        "data": "private manually annotated and reviewed evaluation data is not included",
        "teacher": "configured externally",
        "student": "configured externally",
        "paper_results": "not included in the public repository",
    }


@app.get("/api/v1/cases")
def cases() -> list[dict]:
    return repo.list_cases()


@app.get("/api/v1/cases/{case_id}")
def case(case_id: str) -> dict:
    item = repo.get(case_id)
    if not item:
        raise HTTPException(404, "case not found")
    return item


@app.post("/api/v1/gate/check")
def gate_check(payload: dict) -> dict:
    entities = payload.get("entities", [])
    relations = payload.get("relations", [])
    decisions = [
        orch.paper.gate_relation(r, entities).model_dump(mode="json")
        for r in relations
    ]
    counts = {"PASS": 0, "CONFLICT": 0, "UNKNOWN": 0}
    for d in decisions:
        counts[d["state"]] = counts.get(d["state"], 0) + 1
    return {"decisions": decisions, "counts": counts}


@app.post("/api/v1/experiments/run")
def run(req: ExperimentRequest) -> dict:
    return orch.run(req).model_dump(mode="json")


def _read_manifest(run_id: str) -> dict:
    p = RUNS_DIR / run_id / "run_manifest.json"
    if not p.exists():
        raise HTTPException(404, f"run {run_id} not found on disk")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/v1/runs")
def list_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    items: list[dict] = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        manifest = d / "run_manifest.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        items.append(
            {
                "run_id": data.get("run_id", d.name),
                "status": data.get("status"),
                "question": data.get("question", ""),
                "template": data.get("template"),
                "mode": data.get("mode"),
                "seed": data.get("seed"),
                "event_count": len(data.get("events", [])),
                "ts": data.get("events", [{}])[0].get("ts") if data.get("events") else None,
            }
        )
    return items


@app.get("/api/v1/runs/{run_id}")
def get_run(run_id: str) -> dict:
    in_mem = orch.get(run_id)
    if in_mem is not None:
        return in_mem.model_dump(mode="json")
    return _read_manifest(run_id)


@app.get("/api/v1/runs/{run_id}/events")
def get_run_events(run_id: str) -> list[dict]:
    p = RUNS_DIR / run_id / "agent_events.jsonl"
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


@app.get("/api/v1/runs/{run_id}/gate_trace")
def get_run_gate_trace(run_id: str) -> list[dict]:
    p = RUNS_DIR / run_id / "gate_trace.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


if FRONTEND_DIR.exists():
    @app.get("/")
    def index_root() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")