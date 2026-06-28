from __future__ import annotations

import json
import re
from copy import deepcopy


ALLOWED_RELATIONS = {
    "CAUSES",
    "CONTRIBUTES_TO",
    "OCCURS_AT",
    "AFFECTS",
    "VIOLATES",
    "RESPONSIBLE_FOR",
}


def extract_json_from_response(response_text: str) -> dict:
    text = (response_text or "").strip()
    errors = []
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
            if isinstance(parsed, dict) and {
                "defect_label",
                "event_label",
                "entities",
                "relations",
            }.issubset(parsed):
                return {"parse_success": True, "data": parsed, "errors": errors}
            if isinstance(parsed, dict):
                errors.append(f"json_at_{index}_missing_required_fields")
        except json.JSONDecodeError as error:
            errors.append(f"json_decode_at_{index}: {error.msg}")
    return {
        "parse_success": False,
        "data": None,
        "errors": errors[-5:] or ["No JSON object found"],
    }


def _confidence(value: object) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _normalize_label(value: object, labels: list[str] | None) -> str:
    text = str(value or "").strip()
    if not labels:
        return text
    if text in labels:
        return text
    for label in labels:
        if label != "其他" and label in text:
            return label
    return "其他" if text else ""


def _locate_entity(text: str, entity_text: str, start: object, end: object) -> tuple:
    if isinstance(start, int) and isinstance(end, int):
        if 0 <= start < end <= len(text) and text[start:end] == entity_text:
            return start, end
    occurrences = [
        match.start() for match in re.finditer(re.escape(entity_text), text)
    ]
    if occurrences and isinstance(start, int):
        found = min(occurrences, key=lambda position: abs(position - start))
    else:
        found = occurrences[0] if occurrences else -1
    return (found, found + len(entity_text)) if found >= 0 else (None, None)


def _endpoint_key(text: str) -> str:
    key = re.sub(r"[\s　，。、“”‘’：:；;（）()\[\]【】《》<>]+", "", text or "")
    # Domain shorthand seen in student outputs: "焊角" and "焊脚" often refer
    # to the same weld-leg/fillet-weld expression in these reports.
    key = key.replace("焊角", "焊脚")
    return key


def _resolve_text_endpoint(
    endpoint_text: str,
    entities_by_text: dict[str, list[str]],
) -> str | None:
    endpoint_text = endpoint_text.strip()
    if not endpoint_text:
        return None
    exact = entities_by_text.get(endpoint_text, [])
    if exact:
        return exact[0]
    endpoint_key = _endpoint_key(endpoint_text)
    keyed_exact = entities_by_text.get(endpoint_key, [])
    if keyed_exact:
        return keyed_exact[0]
    candidates = []
    for entity_text, ids in entities_by_text.items():
        entity_key = _endpoint_key(entity_text)
        if (
            endpoint_text in entity_text
            or entity_text in endpoint_text
            or (endpoint_key and endpoint_key in entity_key)
            or (entity_key and entity_key in endpoint_key)
        ):
            # Prefer the closest text length match. This maps "焊角" to a
            # longer predicted defect entity when the student uses shorthand
            # relation endpoints.
            candidates.append((abs(len(entity_key) - len(endpoint_key)), ids[0]))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def normalize_teacher_output(
    parsed_data: dict, record: dict, ontology: dict | None = None
) -> dict:
    data = deepcopy(parsed_data or {})
    errors = []
    text = record["text"]
    entity_specs = ontology.get("entity_types", {}) if ontology else {}
    defect_labels = ontology.get("defect_labels") if ontology else None
    event_labels = ontology.get("event_labels") if ontology else None
    entities = []
    seen_entities = {}
    id_aliases = {}
    for index, raw in enumerate(data.get("entities") or [], start=1):
        if not isinstance(raw, dict):
            errors.append(f"entity_{index}_not_object")
            continue
        entity_text = str(raw.get("text") or "").strip()
        if not entity_text:
            errors.append(f"entity_{index}_empty_text")
            continue
        start, end = _locate_entity(
            text, entity_text, raw.get("start"), raw.get("end")
        )
        if start is None:
            errors.append(f"entity_{index}_not_in_source:{entity_text}")
            continue
        entity_type = str(raw.get("type") or "").strip().upper()
        if ontology and entity_type not in entity_specs:
            errors.append(f"entity_{index}_invalid_type:{entity_type}")
            continue
        key = (start, end, entity_type)
        original_id = str(raw.get("id") or f"E{index}")
        if key in seen_entities:
            id_aliases[original_id] = seen_entities[key]
            continue
        entity_id = f"E{len(entities) + 1}"
        id_aliases[original_id] = entity_id
        seen_entities[key] = entity_id
        layer = (
            entity_specs[entity_type]["layer"]
            if ontology and entity_type in entity_specs
            else str(raw.get("wsr_layer") or "").strip()
        )
        entities.append(
            {
                "id": entity_id,
                "text": entity_text,
                "start": start,
                "end": end,
                "type": entity_type,
                "wsr_layer": layer,
                "confidence": _confidence(raw.get("confidence")),
            }
        )

    entity_ids = {item["id"] for item in entities}
    entities_by_text: dict[str, list[str]] = {}
    for item in entities:
        entities_by_text.setdefault(item["text"], []).append(item["id"])
        key = _endpoint_key(item["text"])
        if key:
            entities_by_text.setdefault(key, []).append(item["id"])
    relations = []
    seen_relations = set()
    for index, raw in enumerate(data.get("relations") or [], start=1):
        if not isinstance(raw, dict):
            errors.append(f"relation_{index}_not_object")
            continue
        raw_head = str(raw.get("head") or raw.get("head_id") or "").strip()
        raw_tail = str(raw.get("tail") or raw.get("tail_id") or "").strip()
        head = id_aliases.get(raw_head)
        tail = id_aliases.get(raw_tail)
        if head is None and raw.get("head_text") is not None:
            head = _resolve_text_endpoint(
                str(raw.get("head_text")).strip(),
                entities_by_text,
            )
        if tail is None and raw.get("tail_text") is not None:
            tail = _resolve_text_endpoint(
                str(raw.get("tail_text")).strip(),
                entities_by_text,
            )
        # Text-schema students sometimes keep the keys "head"/"tail" but put
        # entity text in them. Treat those as text endpoints before rejecting.
        if head is None and raw_head:
            head = _resolve_text_endpoint(raw_head, entities_by_text)
        if tail is None and raw_tail:
            tail = _resolve_text_endpoint(raw_tail, entities_by_text)
        relation_type = str(
            raw.get("type") or raw.get("relation") or raw.get("predicate") or ""
        ).strip().upper()
        if head not in entity_ids or tail not in entity_ids:
            errors.append(f"relation_{index}_unknown_endpoint")
            continue
        allowed_relations = (
            {
                item
                for item in ontology.get("relation_types", [])
                if item != "NO_RELATION"
            }
            if ontology
            else ALLOWED_RELATIONS
        )
        if relation_type not in allowed_relations:
            errors.append(f"relation_{index}_invalid_type:{relation_type}")
            continue
        key = (head, relation_type, tail)
        if key in seen_relations:
            continue
        seen_relations.add(key)
        relations.append(
            {
                "head": head,
                "tail": tail,
                "type": relation_type,
                "confidence": _confidence(raw.get("confidence")),
            }
        )

    entity_by_id = {item["id"]: item for item in entities}
    triples = [
        {
            "head_text": entity_by_id[item["head"]]["text"],
            "relation": item["type"],
            "tail_text": entity_by_id[item["tail"]]["text"],
            "confidence": item["confidence"],
        }
        for item in relations
    ]
    teacher_probs = data.get("teacher_probs")
    if not isinstance(teacher_probs, dict):
        teacher_probs = {}
    teacher_probs = {
        "defect": teacher_probs.get("defect", {}),
        "event": teacher_probs.get("event", {}),
        "triple_validity": teacher_probs.get("triple_validity", {}),
    }
    return {
        "defect_label": _normalize_label(data.get("defect_label"), defect_labels),
        "event_label": _normalize_label(data.get("event_label"), event_labels),
        "entities": entities,
        "relations": relations,
        "triples": triples,
        "causal_chain": data.get("causal_chain")
        if isinstance(data.get("causal_chain"), list)
        else [],
        "teacher_probs": teacher_probs,
        "errors": errors,
    }
