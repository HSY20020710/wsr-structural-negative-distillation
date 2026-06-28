from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ontology.gate import check_relation  # noqa: E402
from src.student.data import read_jsonl, record_id  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate teacher labels before student distillation."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--ontology", type=Path, default=ROOT / "configs/wsr_ontology.yaml"
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    source = read_jsonl(args.source)
    predictions = read_jsonl(args.input)
    ontology = yaml.safe_load(args.ontology.read_text(encoding="utf-8"))
    source_ids = [record_id(record) for record in source]
    prediction_ids = [record_id(record) for record in predictions]
    source_by_id = {record_id(record): record for record in source}
    duplicate_ids = sorted(
        case_id
        for case_id, count in Counter(prediction_ids).items()
        if count > 1
    )
    missing_ids = sorted(set(source_ids) - set(prediction_ids))
    unexpected_ids = sorted(set(prediction_ids) - set(source_ids))

    errors = []
    warnings = []
    parse_failures = []
    valid_entities = 0
    valid_relations = 0
    rejected_relations = 0
    relation_types = Counter()
    entity_types = Counter()
    for record in predictions:
        case_id = record_id(record)
        source_record = source_by_id.get(case_id)
        if not source_record:
            continue
        if not record.get("parse_success"):
            parse_failures.append(case_id)
        if record.get("pipeline_components") != [
            "wsr_ontology",
            "entity_refinement_second_pass",
            "entity_consensus_filter",
            "relation_endpoint_remapping",
            "wsr_gate",
        ]:
            errors.append(f"{case_id}: incomplete pipeline_components")
        text = str(source_record.get("text", ""))
        entities = {}
        for entity in record.get("entities", []):
            entity_id = str(entity.get("id", ""))
            start, end = entity.get("start"), entity.get("end")
            entity_type = str(entity.get("type", ""))
            expected_layer = ontology.get("entity_types", {}).get(
                entity_type, {}
            ).get("layer")
            if entity_id in entities:
                errors.append(f"{case_id}: duplicate entity id {entity_id}")
                continue
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or not 0 <= start < end <= len(text)
                or text[start:end] != entity.get("text")
            ):
                errors.append(f"{case_id}: invalid span for {entity_id}")
                continue
            if not expected_layer or entity.get("wsr_layer") != expected_layer:
                errors.append(f"{case_id}: invalid type/layer for {entity_id}")
                continue
            entities[entity_id] = entity
            valid_entities += 1
            entity_types[entity_type] += 1
        for relation in record.get("relations", []):
            head = entities.get(str(relation.get("head")))
            tail = entities.get(str(relation.get("tail")))
            if not head or not tail:
                errors.append(f"{case_id}: relation endpoint missing")
                continue
            result = check_relation(
                head["type"], str(relation.get("type")), tail["type"], ontology
            )
            if not result["passed"]:
                errors.append(f"{case_id}: relation failed WSR Gate")
                continue
            valid_relations += 1
            relation_types[str(relation["type"])] += 1
        rejected_relations += len(record.get("rejected_relations", []))
        if not record.get("entities"):
            warnings.append(f"{case_id}: no accepted entities")

    fatal = []
    if duplicate_ids:
        fatal.append(f"duplicate prediction ids: {duplicate_ids[:10]}")
    if missing_ids:
        fatal.append(f"missing source ids: {missing_ids[:10]}")
    if unexpected_ids:
        fatal.append(f"unexpected prediction ids: {unexpected_ids[:10]}")
    if parse_failures:
        fatal.append(f"parse failures: {parse_failures[:10]}")
    fatal.extend(errors)
    report = {
        "status": "PASS" if not fatal else "FAIL",
        "source_records": len(source),
        "prediction_records": len(predictions),
        "unique_prediction_ids": len(set(prediction_ids)),
        "parse_success": len(predictions) - len(parse_failures),
        "valid_entities": valid_entities,
        "valid_relations": valid_relations,
        "rejected_relations": rejected_relations,
        "entity_types": dict(sorted(entity_types.items())),
        "relation_types": dict(sorted(relation_types.items())),
        "missing_ids": missing_ids,
        "unexpected_ids": unexpected_ids,
        "duplicate_ids": duplicate_ids,
        "parse_failures": parse_failures,
        "errors": errors,
        "warnings": warnings,
    }
    report_path = args.report or args.input.with_suffix(".validation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if fatal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
