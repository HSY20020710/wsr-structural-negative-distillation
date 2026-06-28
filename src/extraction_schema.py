from __future__ import annotations

import json
from pathlib import Path


def load_wsr_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extraction_json_schema(wsr_schema: dict) -> dict:
    entity_types = list(wsr_schema["entity_types"])
    relation_types = wsr_schema["relation_types"]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["defect_labels", "event_labels", "entities", "relations"],
        "properties": {
            "defect_labels": {
                "type": "array",
                "items": {"type": "string"},
            },
            "event_labels": {
                "type": "array",
                "items": {"type": "string"},
            },
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "text", "start", "end", "type", "wsr"],
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string"},
                        "start": {"type": "integer", "minimum": 0},
                        "end": {"type": "integer", "minimum": 1},
                        "type": {"type": "string", "enum": entity_types},
                        "wsr": {"type": "string", "enum": ["W", "S", "R"]},
                    },
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["head", "type", "tail"],
                    "properties": {
                        "head": {"type": "string"},
                        "type": {"type": "string", "enum": relation_types},
                        "tail": {"type": "string"},
                    },
                },
            },
        },
    }


def build_teacher_messages(text: str, wsr_schema: dict) -> list[dict]:
    entity_lines = "\n".join(
        f"- {entity_type}: {wsr}"
        for entity_type, wsr in wsr_schema["entity_types"].items()
    )
    relation_lines = "\n".join(f"- {name}" for name in wsr_schema["relation_types"])
    system = f"""You are a ship-welding quality information extraction system.
Extract only facts explicitly supported by the input report.
Return JSON only. Do not add explanations or markdown.

Entity types and their WSR dimensions:
{entity_lines}

Allowed relation types:
{relation_lines}

Rules:
1. Character offsets use Python slicing: start inclusive, end exclusive.
2. entity.text must exactly equal report[start:end].
3. Causes is directed from cause to result.
4. Every relation endpoint must refer to an entity ID in entities.
5. Omit uncertain entities and relations. Never invent missing facts.
6. The wsr value must match the entity type definition."""
    user = f"""Extract the structured welding-quality information from this report:

{text}"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def validate_prediction_structure(
    prediction: dict, text: str, wsr_schema: dict
) -> list[str]:
    errors = []
    if not isinstance(prediction, dict):
        return ["prediction must be an object"]
    for field in ["defect_labels", "event_labels", "entities", "relations"]:
        if not isinstance(prediction.get(field), list):
            errors.append(f"{field} must be an array")
    if errors:
        return errors
    entity_ids = set()
    for index, entity in enumerate(prediction["entities"]):
        prefix = f"entity[{index}]"
        if not isinstance(entity, dict):
            errors.append(f"{prefix} must be an object")
            continue
        entity_id = entity.get("id")
        if not isinstance(entity_id, str) or not entity_id or entity_id in entity_ids:
            errors.append(f"{prefix}.id is missing or duplicated")
        entity_ids.add(entity_id)
        entity_type = entity.get("type")
        if entity_type not in wsr_schema["entity_types"]:
            errors.append(f"{prefix}.type is not allowed")
        elif entity.get("wsr") != wsr_schema["entity_types"][entity_type]:
            errors.append(f"{prefix}.wsr does not match entity type")
        start, end = entity.get("start"), entity.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            errors.append(f"{prefix} offsets must be integers")
        elif not (0 <= start < end <= len(text)):
            errors.append(f"{prefix} offsets are outside the report")
        elif text[start:end] != entity.get("text"):
            errors.append(f"{prefix}.text does not match report span")
    for index, relation in enumerate(prediction["relations"]):
        prefix = f"relation[{index}]"
        if not isinstance(relation, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if relation.get("head") not in entity_ids:
            errors.append(f"{prefix}.head is unknown")
        if relation.get("tail") not in entity_ids:
            errors.append(f"{prefix}.tail is unknown")
        if relation.get("type") not in wsr_schema["relation_types"]:
            errors.append(f"{prefix}.type is not allowed")
    return errors
