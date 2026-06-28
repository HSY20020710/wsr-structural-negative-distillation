from __future__ import annotations

from copy import deepcopy


def _layer(entity_type: str, ontology: dict) -> str:
    return ontology.get("entity_types", {}).get(entity_type, {}).get("layer", "")


def _conflict_type(
    head_type: str, relation_type: str, tail_type: str, ontology: dict
) -> str:
    layers = {_layer(head_type, ontology), _layer(tail_type, ontology)}
    layers.discard("")
    if relation_type in {
        "CAUSES",
        "CONTRIBUTES_TO",
        "VIOLATES",
        "DETECTED_BY",
        "TREATED_BY",
    } and "Shili" in layers:
        return "shili" if len(layers) <= 2 else "multi_conflict"
    if "Renli" in layers:
        return "renli" if layers <= {"Renli", "Shili"} else "multi_conflict"
    if layers and layers <= {"Wuli"}:
        return "wuli"
    if len(layers) > 1:
        return "multi_conflict"
    return {"Wuli": "wuli", "Shili": "shili", "Renli": "renli"}.get(
        next(iter(layers), ""), "multi_conflict"
    )


def check_relation(
    head_type: str, relation_type: str, tail_type: str, ontology: dict
) -> dict:
    if relation_type == "NO_RELATION":
        return {
            "passed": False,
            "conflict_type": "multi_conflict",
            "reason": "NO_RELATION is not a positive relation",
        }
    allowed_pairs = {
        tuple(pair)
        for pair in ontology.get("allowed_relations", {}).get(relation_type, [])
    }
    passed = (head_type, tail_type) in allowed_pairs
    return {
        "passed": passed,
        "conflict_type": "positive"
        if passed
        else _conflict_type(head_type, relation_type, tail_type, ontology),
        "reason": (
            "allowed relation signature"
            if passed
            else f"{head_type} -[{relation_type}]-> {tail_type} is not allowed"
        ),
    }


def apply_wsr_gate(record: dict, ontology: dict) -> dict:
    clean = deepcopy(record)
    entities = {item["id"]: item for item in clean.get("entities", [])}
    accepted = []
    rejected = []
    stats = {
        "predicted_relations_before_gate": len(clean.get("relations", [])),
        "predicted_relations_after_gate": 0,
        "rejected_total": 0,
        "rejected_wuli": 0,
        "rejected_shili": 0,
        "rejected_renli": 0,
        "rejected_multi_conflict": 0,
    }
    for relation in clean.get("relations", []):
        head = entities.get(relation.get("head"))
        tail = entities.get(relation.get("tail"))
        if not head or not tail:
            result = {
                "passed": False,
                "conflict_type": "multi_conflict",
                "reason": "relation endpoint is missing",
            }
        else:
            result = check_relation(
                head["type"], relation["type"], tail["type"], ontology
            )
        enriched = {**relation, "gate_result": result}
        if result["passed"]:
            accepted.append(relation)
        else:
            rejected.append(enriched)
            stats["rejected_total"] += 1
            stats[f"rejected_{result['conflict_type']}"] += 1
    clean["relations_before_gate"] = deepcopy(clean.get("relations", []))
    clean["relations"] = accepted
    clean["rejected_relations"] = rejected
    stats["predicted_relations_after_gate"] = len(accepted)
    clean["gate_stats"] = stats
    clean["triples"] = [
        {
            "head_text": entities[item["head"]]["text"],
            "relation": item["type"],
            "tail_text": entities[item["tail"]]["text"],
            "confidence": item.get("confidence", 0.0),
        }
        for item in accepted
        if item.get("head") in entities and item.get("tail") in entities
    ]
    return clean
