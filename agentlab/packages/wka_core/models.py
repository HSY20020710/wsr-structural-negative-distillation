from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Literal

class GateState(str, Enum):
    PASS = "PASS"
    CONFLICT = "CONFLICT"
    UNKNOWN = "UNKNOWN"

class EvidenceEvent(BaseModel):
    event_id: str
    run_id: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    actor: str
    event_type: str
    input_ref: str | None = None
    output_ref: str | None = None
    status: str = "PASS"
    provenance: dict[str, Any] = Field(default_factory=dict)

class GateDecision(BaseModel):
    state: GateState
    layer: str = "Unknown"
    reason: str
    rule_id: str
    relation: dict[str, Any]

class ExperimentRequest(BaseModel):
    question: str
    template: str = "paper_main"
    case_id: str | None = None
    seed: int = 42
    mode: Literal["demo", "real"] = "demo"

class ExperimentRun(BaseModel):
    run_id: str
    status: str
    question: str
    template: str
    mode: str
    seed: int
    summary: dict[str, Any] = Field(default_factory=dict)
    events: list[EvidenceEvent] = Field(default_factory=list)
