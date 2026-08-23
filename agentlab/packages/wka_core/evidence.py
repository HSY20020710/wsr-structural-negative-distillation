from __future__ import annotations
import hashlib, json
from pathlib import Path
from .models import EvidenceEvent

class EvidenceStore:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
    @staticmethod
    def sha256_obj(obj) -> str:
        raw=json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()
    def append(self, event: EvidenceEvent):
        d=self.run_dir/event.run_id
        d.mkdir(parents=True, exist_ok=True)
        p=d/'agent_events.jsonl'
        with p.open('a',encoding='utf-8') as f:
            f.write(event.model_dump_json()+"\n")
    def write_json(self, run_id:str, name:str, obj):
        d=self.run_dir/run_id; d.mkdir(parents=True, exist_ok=True)
        p=d/name
        p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        return {"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
