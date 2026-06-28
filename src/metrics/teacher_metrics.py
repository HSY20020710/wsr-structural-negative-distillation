from __future__ import annotations

from collections import Counter
from copy import deepcopy

from src.ontology.gate import check_relation


def _prf(tp: int, predicted: int, gold: int) -> tuple[float, float, float]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def _overlap(left: list[tuple], right: list[tuple]) -> int:
    return sum((Counter(left) & Counter(right)).values())


def _span_overlap(left: dict, right: dict) -> bool:
    return (
        left.get("type") == right.get("type")
        and isinstance(left.get("start"), int)
        and isinstance(left.get("end"), int)
        and isinstance(right.get("start"), int)
        and isinstance(right.get("end"), int)
        and left["start"] < right["end"]
        and right["start"] < left["end"]
    )


def _maximum_overlap_matches(predicted: list[dict], gold: list[dict]) -> int:
    candidates = []
    for pred_index, pred in enumerate(predicted):
        for gold_index, target in enumerate(gold):
            if _span_overlap(pred, target):
                intersection = min(pred["end"], target["end"]) - max(
                    pred["start"], target["start"]
                )
                union = max(pred["end"], target["end"]) - min(
                    pred["start"], target["start"]
                )
                candidates.append((intersection / union, pred_index, gold_index))
    matched_pred, matched_gold = set(), set()
    matches = 0
    for _, pred_index, gold_index in sorted(candidates, reverse=True):
        if pred_index in matched_pred or gold_index in matched_gold:
            continue
        matched_pred.add(pred_index)
        matched_gold.add(gold_index)
        matches += 1
    return matches


def _find_span(text: str, entity_text: str) -> tuple[int | None, int | None]:
    start = text.find(entity_text)
    return (start, start + len(entity_text)) if start >= 0 else (None, None)


def normalize_gold_record(record: dict) -> dict:
    gold = deepcopy(record)
    text = gold.get("text", "")
    entities = []
    id_aliases = {}
    for index, raw in enumerate(gold.get("entities") or [], start=1):
        entity = dict(raw)
        entity_id = str(entity.get("id") or f"E{index}")
        entity_text = str(entity.get("text") or "")
        start, end = entity.get("start"), entity.get("end")
        if not (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(text)
            and text[start:end] == entity_text
        ):
            start, end = _find_span(text, entity_text)
        normalized_id = f"E{len(entities) + 1}"
        id_aliases[entity_id] = normalized_id
        entities.append(
            {
                "id": normalized_id,
                "text": entity_text,
                "start": start,
                "end": end,
                "type": str(entity.get("type") or "").upper(),
            }
        )
    by_text = {}
    for entity in entities:
        by_text.setdefault(entity["text"], entity["id"])
    relations = []
    for raw in gold.get("relations") or []:
        relation_type = str(
            raw.get("type") or raw.get("relation") or raw.get("predicate") or ""
        ).upper()
        head = raw.get("head") or raw.get("head_id")
        tail = raw.get("tail") or raw.get("tail_id")
        head_id = id_aliases.get(str(head)) if head is not None else None
        tail_id = id_aliases.get(str(tail)) if tail is not None else None
        if not head_id:
            head_id = by_text.get(str(raw.get("head_text") or head or ""))
        if not tail_id:
            tail_id = by_text.get(str(raw.get("tail_text") or tail or ""))
        if head_id and tail_id and relation_type:
            relations.append({"head": head_id, "tail": tail_id, "type": relation_type})
    if not relations:
        for raw in gold.get("triples") or []:
            head_id = by_text.get(str(raw.get("head_text") or raw.get("head") or ""))
            tail_id = by_text.get(str(raw.get("tail_text") or raw.get("tail") or ""))
            relation_type = str(
                raw.get("relation") or raw.get("type") or raw.get("predicate") or ""
            ).upper()
            if head_id and tail_id and relation_type:
                relations.append({"head": head_id, "tail": tail_id, "type": relation_type})
    gold["entities"] = entities
    gold["relations"] = relations
    gold["defect_label"] = gold.get("defect_label") or (
        gold.get("defect_labels") or [""]
    )[0]
    gold["event_label"] = gold.get("event_label") or (
        gold.get("event_labels") or [""]
    )[0]
    return gold


def _entity_keys(record: dict) -> list[tuple]:
    return [
        (entity.get("start"), entity.get("end"), entity.get("type"))
        for entity in record.get("entities", [])
    ]


def _relation_keys(record: dict) -> list[tuple]:
    entities = {entity["id"]: entity for entity in record.get("entities", [])}
    keys = []
    for relation in record.get("relations", []):
        head = entities.get(relation.get("head"))
        tail = entities.get(relation.get("tail"))
        if head and tail:
            keys.append(
                (
                    head.get("start"),
                    head.get("end"),
                    head.get("type"),
                    relation.get("type"),
                    tail.get("start"),
                    tail.get("end"),
                    tail.get("type"),
                )
            )
    return keys


def _conditional_relation_keys(record: dict, gold: dict) -> tuple[list[tuple], list[tuple]]:
    gold_entities = {
        (entity.get("start"), entity.get("end"), entity.get("type"))
        for entity in gold.get("entities", [])
    }
    entities = {entity["id"]: entity for entity in record.get("entities", [])}
    predicted_entity_keys = {
        (entity.get("start"), entity.get("end"), entity.get("type"))
        for entity in record.get("entities", [])
    }
    keys = []
    for relation in record.get("relations", []):
        head = entities.get(relation.get("head"))
        tail = entities.get(relation.get("tail"))
        if not head or not tail:
            continue
        head_key = (head.get("start"), head.get("end"), head.get("type"))
        tail_key = (tail.get("start"), tail.get("end"), tail.get("type"))
        if head_key in gold_entities and tail_key in gold_entities:
            keys.append((head_key, relation.get("type"), tail_key))
    gold_by_id = {entity["id"]: entity for entity in gold.get("entities", [])}
    gold_keys = []
    for relation in gold.get("relations", []):
        head = gold_by_id.get(relation.get("head"))
        tail = gold_by_id.get(relation.get("tail"))
        if head and tail:
            head_key = (head.get("start"), head.get("end"), head.get("type"))
            tail_key = (tail.get("start"), tail.get("end"), tail.get("type"))
            if head_key in predicted_entity_keys and tail_key in predicted_entity_keys:
                gold_keys.append(
                    (head_key, relation.get("type"), tail_key)
                )
    return keys, gold_keys


def _triple_keys(record: dict) -> list[tuple]:
    if record.get("triples"):
        return [
            (
                triple.get("head_text"),
                str(triple.get("relation") or triple.get("type") or "").upper(),
                triple.get("tail_text"),
            )
            for triple in record["triples"]
        ]
    entities = {entity["id"]: entity for entity in record.get("entities", [])}
    return [
        (
            entities[relation["head"]]["text"],
            relation["type"],
            entities[relation["tail"]]["text"],
        )
        for relation in record.get("relations", [])
        if relation.get("head") in entities and relation.get("tail") in entities
    ]


def _cvr_for_relations(relations: list[dict], entities: list[dict], ontology: dict) -> dict:
    by_id = {entity["id"]: entity for entity in entities}
    counts = Counter()
    total = 0
    for relation in relations:
        head = by_id.get(relation.get("head"))
        tail = by_id.get(relation.get("tail"))
        if not head or not tail:
            counts["multi_conflict"] += 1
            total += 1
            continue
        result = check_relation(head["type"], relation["type"], tail["type"], ontology)
        total += 1
        if not result["passed"]:
            counts[result["conflict_type"]] += 1
    invalid = sum(counts.values())
    return {
        "total": total,
        "invalid": invalid,
        "wuli": counts["wuli"],
        "shili": counts["shili"],
        "renli": counts["renli"],
        "multi_conflict": counts["multi_conflict"],
    }


def evaluate_teacher(
    predictions: list[dict], gold_records: list[dict], ontology: dict
) -> dict:
    gold_by_id = {
        record.get("case_id") or record.get("record_id"): normalize_gold_record(record)
        for record in gold_records
    }
    prediction_by_id = {
        record.get("case_id") or record.get("record_id"): record
        for record in predictions
    }
    evaluated = []
    missing = []
    parse_successes = 0
    for case_id, gold in gold_by_id.items():
        prediction = prediction_by_id.get(case_id)
        if not prediction:
            missing.append(case_id)
            continue
        if prediction.get("parse_success"):
            parse_successes += 1
            evaluated.append((prediction, gold))

    defect_correct = sum(
        prediction.get("defect_label") == gold.get("defect_label")
        for prediction, gold in evaluated
    )
    event_correct = sum(
        prediction.get("event_label") == gold.get("event_label")
        for prediction, gold in evaluated
    )
    entity_pred, entity_gold, entity_tp = 0, 0, 0
    relaxed_entity_tp = 0
    relation_pred, relation_gold, relation_tp = 0, 0, 0
    relation_before_pred, relation_before_tp = 0, 0
    conditional_pred, conditional_gold, conditional_tp = 0, 0, 0
    triple_pred, triple_gold, triple_tp = 0, 0, 0
    exact_triples = 0
    cvr_post = Counter()
    cvr_before = Counter()
    for prediction, gold in evaluated:
        pred_keys, gold_keys = _entity_keys(prediction), _entity_keys(gold)
        entity_pred += len(pred_keys)
        entity_gold += len(gold_keys)
        entity_tp += _overlap(pred_keys, gold_keys)
        relaxed_entity_tp += _maximum_overlap_matches(
            prediction.get("entities", []), gold.get("entities", [])
        )

        pred_rel, gold_rel = _relation_keys(prediction), _relation_keys(gold)
        before_record = {
            **prediction,
            "relations": prediction.get(
                "relations_before_gate", prediction.get("relations", [])
            ),
            "triples": [],
        }
        pred_rel_before = _relation_keys(before_record)
        relation_pred += len(pred_rel)
        relation_before_pred += len(pred_rel_before)
        relation_gold += len(gold_rel)
        relation_tp += _overlap(pred_rel, gold_rel)
        relation_before_tp += _overlap(pred_rel_before, gold_rel)

        pred_cond, gold_cond = _conditional_relation_keys(prediction, gold)
        conditional_pred += len(pred_cond)
        conditional_gold += len(gold_cond)
        conditional_tp += _overlap(pred_cond, gold_cond)

        pred_triples, gold_triples = _triple_keys(prediction), _triple_keys(gold)
        triple_pred += len(pred_triples)
        triple_gold += len(gold_triples)
        triple_tp += _overlap(pred_triples, gold_triples)
        exact_triples += Counter(pred_triples) == Counter(gold_triples)

        post = _cvr_for_relations(
            prediction.get("relations", []), prediction.get("entities", []), ontology
        )
        before = _cvr_for_relations(
            prediction.get("relations_before_gate", prediction.get("relations", [])),
            prediction.get("entities", []),
            ontology,
        )
        cvr_post.update(post)
        cvr_before.update(before)

    entity_p, entity_r, entity_f1 = _prf(entity_tp, entity_pred, entity_gold)
    relaxed_entity_p, relaxed_entity_r, relaxed_entity_f1 = _prf(
        relaxed_entity_tp, entity_pred, entity_gold
    )
    relation_p, relation_r, relation_f1 = _prf(
        relation_tp, relation_pred, relation_gold
    )
    relation_before_p, relation_before_r, relation_before_f1 = _prf(
        relation_before_tp, relation_before_pred, relation_gold
    )
    conditional_p, conditional_r, conditional_f1 = _prf(
        conditional_tp, conditional_pred, conditional_gold
    )
    triple_p, triple_r, triple_f1 = _prf(triple_tp, triple_pred, triple_gold)
    records = len(gold_records)
    evaluated_records = len(evaluated)

    def ratio(value: int, total: int) -> float:
        return value / total if total else 0.0

    return {
        "records": records,
        "evaluated_records": evaluated_records,
        "missing_predictions": len(missing),
        "missing_prediction_ids": missing,
        "parse_success_rate": ratio(parse_successes, records),
        "defect_accuracy": ratio(defect_correct, evaluated_records),
        "event_accuracy": ratio(event_correct, evaluated_records),
        "entity_precision": entity_p,
        "entity_recall": entity_r,
        "entity_span_type_f1": entity_f1,
        "relaxed_entity_precision": relaxed_entity_p,
        "relaxed_entity_recall": relaxed_entity_r,
        "relaxed_entity_span_type_f1": relaxed_entity_f1,
        "relation_precision": relation_p,
        "relation_recall": relation_r,
        "relation_f1": relation_f1,
        "relation_precision_before_gate": relation_before_p,
        "relation_recall_before_gate": relation_before_r,
        "relation_f1_before_gate": relation_before_f1,
        "gate_relation_retention": ratio(relation_pred, relation_before_pred),
        "conditional_relation_precision": conditional_p,
        "conditional_relation_recall": conditional_r,
        "conditional_relation_f1": conditional_f1,
        "conditional_relation_note": (
            "Relation type F1 restricted to relations whose two gold entity spans "
            "were exactly extracted. A true gold-entity-conditioned run requires "
            "separate inference."
        ),
        "triple_precision": triple_p,
        "triple_recall": triple_r,
        "triple_f1": triple_f1,
        "triple_exact_accuracy": ratio(exact_triples, evaluated_records),
        "cvr_w": ratio(cvr_post["wuli"], cvr_post["total"]),
        "cvr_s": ratio(cvr_post["shili"], cvr_post["total"]),
        "cvr_r": ratio(cvr_post["renli"], cvr_post["total"]),
        "cvr_multi": ratio(cvr_post["multi_conflict"], cvr_post["total"]),
        "cvr_all": ratio(cvr_post["invalid"], cvr_post["total"]),
        "cvr_w_before_gate": ratio(cvr_before["wuli"], cvr_before["total"]),
        "cvr_s_before_gate": ratio(cvr_before["shili"], cvr_before["total"]),
        "cvr_r_before_gate": ratio(cvr_before["renli"], cvr_before["total"]),
        "cvr_all_before_gate": ratio(cvr_before["invalid"], cvr_before["total"]),
        "predicted_relations": relation_pred,
        "predicted_relations_before_gate": relation_before_pred,
        "gold_relations": relation_gold,
    }
