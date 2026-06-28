from __future__ import annotations


VALID_WSR_VALUES = {
    "W": "W",
    "S": "S",
    "R": "R",
    "Wuli": "W",
    "Shili": "S",
    "Renli": "R",
}


def validate_gold_record(record: dict) -> list[str]:
    errors = []
    text = record.get("text", "")
    entity_ids = set()
    for index, entity in enumerate(record.get("entities", [])):
        prefix = f"entity[{index}]"
        entity_id = entity.get("id")
        if not entity_id or entity_id in entity_ids:
            errors.append(f"{prefix}: missing or duplicate id")
        entity_ids.add(entity_id)
        start, end = entity.get("start"), entity.get("end")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not (0 <= start < end <= len(text))
        ):
            errors.append(f"{prefix}: invalid span")
        elif text[start:end] != entity.get("text"):
            errors.append(f"{prefix}: span text mismatch")
        wsr_value = entity.get("wsr", entity.get("wsr_layer"))
        if wsr_value not in VALID_WSR_VALUES:
            errors.append(
                f"{prefix}: wsr/wsr_layer must be W/S/R or "
                "Wuli/Shili/Renli"
            )
        if not entity.get("type"):
            errors.append(f"{prefix}: missing type")
    for index, relation in enumerate(record.get("relations", [])):
        prefix = f"relation[{index}]"
        if relation.get("head") not in entity_ids:
            errors.append(f"{prefix}: unknown head entity")
        if relation.get("tail") not in entity_ids:
            errors.append(f"{prefix}: unknown tail entity")
        if not relation.get("type"):
            errors.append(f"{prefix}: missing relation type")
    if record.get("annotation", {}).get("status") not in {
        "unannotated",
        "in_progress",
        "reviewed",
        "adjudicated",
    }:
        errors.append("annotation.status is invalid")
    return errors
