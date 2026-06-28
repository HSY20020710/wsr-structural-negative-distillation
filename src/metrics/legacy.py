from __future__ import annotations

from collections import Counter


def prf(true_positive: int, predicted: int, gold: int) -> dict:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / gold if gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positive,
        "predicted": predicted,
        "gold": gold,
    }


def entity_key(entity: dict) -> tuple:
    return entity["start"], entity["end"], entity["type"]


def relation_key(relation: dict, entities_by_id: dict[str, dict]) -> tuple | None:
    head = entities_by_id.get(relation.get("head"))
    tail = entities_by_id.get(relation.get("tail"))
    if not head or not tail:
        return None
    return entity_key(head), relation["type"], entity_key(tail)


def multiset_overlap(predicted: list[tuple], gold: list[tuple]) -> int:
    return sum((Counter(predicted) & Counter(gold)).values())


def evaluate_extractions(pairs: list[tuple[dict, dict]]) -> dict:
    entity_predicted, entity_gold, entity_tp = 0, 0, 0
    relation_predicted, relation_gold, relation_tp = 0, 0, 0
    conditional_predicted, conditional_gold, conditional_tp = 0, 0, 0
    hallucinated_entities = 0
    for prediction, gold in pairs:
        pred_entities = prediction.get("entities", [])
        gold_entities = gold.get("entities", [])
        pred_entity_keys = [entity_key(item) for item in pred_entities]
        gold_entity_keys = [entity_key(item) for item in gold_entities]
        entity_predicted += len(pred_entity_keys)
        entity_gold += len(gold_entity_keys)
        entity_tp += multiset_overlap(pred_entity_keys, gold_entity_keys)
        text = gold["text"]
        hallucinated_entities += sum(
            1
            for entity in pred_entities
            if not (
                isinstance(entity.get("start"), int)
                and isinstance(entity.get("end"), int)
                and 0 <= entity["start"] < entity["end"] <= len(text)
                and text[entity["start"]:entity["end"]] == entity.get("text")
            )
        )
        pred_by_id = {item["id"]: item for item in pred_entities if item.get("id")}
        gold_by_id = {item["id"]: item for item in gold_entities if item.get("id")}
        pred_relations = [
            key
            for relation in prediction.get("relations", [])
            if (key := relation_key(relation, pred_by_id)) is not None
        ]
        gold_relations = [
            key
            for relation in gold.get("relations", [])
            if (key := relation_key(relation, gold_by_id)) is not None
        ]
        relation_predicted += len(pred_relations)
        relation_gold += len(gold_relations)
        relation_tp += multiset_overlap(pred_relations, gold_relations)
        gold_span_to_entity = {entity_key(item): item for item in gold_entities}
        projected_pred = []
        for relation in prediction.get("relations", []):
            head = pred_by_id.get(relation.get("head"))
            tail = pred_by_id.get(relation.get("tail"))
            if not head or not tail:
                continue
            head_key, tail_key = entity_key(head), entity_key(tail)
            if head_key in gold_span_to_entity and tail_key in gold_span_to_entity:
                projected_pred.append((head_key, relation["type"], tail_key))
        conditional_predicted += len(projected_pred)
        conditional_gold += len(gold_relations)
        conditional_tp += multiset_overlap(projected_pred, gold_relations)
    return {
        "entity_span_type": prf(entity_tp, entity_predicted, entity_gold),
        "relation": prf(relation_tp, relation_predicted, relation_gold),
        "conditional_relation": prf(
            conditional_tp, conditional_predicted, conditional_gold
        ),
        "triple": prf(relation_tp, relation_predicted, relation_gold),
        "hallucinated_entities": hallucinated_entities,
        "hallucination_rate": (
            hallucinated_entities / entity_predicted if entity_predicted else 0.0
        ),
    }
