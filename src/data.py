from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable


SUPPORTED_METHODS = {
    "qwen3_supervised",
    "self_training",
    "standard_kd",
    "contrastive_kd",
    "kd_wsr_post",
    "ours",
}
SUPPORTED_STAGES = {"joint", "entity", "relation", "two_stage"}
SUPPORTED_TARGET_SCHEMAS = {"full", "text"}
ENHANCED_TEACHER_COMPONENTS = [
    "wsr_ontology",
    "entity_refinement_second_pass",
    "entity_consensus_filter",
    "relation_endpoint_remapping",
    "wsr_gate",
]

SYSTEM_PROMPT = """你是船舶焊接质量报告结构化抽取模型。
请输出一个合法JSON对象，字段必须包含defect_label、event_label、entities和relations。
实体必须保留原文text、start、end、type和wsr_layer。
关系使用实体id作为head和tail，只输出报告原文能够支持的关系。"""


def build_system_prompt(ontology: dict | None = None) -> str:
    if not ontology:
        return SYSTEM_PROMPT
    defect_labels = "、".join(ontology.get("defect_labels", []))
    event_labels = "、".join(ontology.get("event_labels", []))
    entity_lines = "\n".join(
        f"- {name}: wsr_layer={spec['layer']}"
        for name, spec in ontology.get("entity_types", {}).items()
    )
    relation_lines = "\n".join(
        f"- {name}: "
        + "；".join(f"{head}->{tail}" for head, tail in pairs)
        for name, pairs in ontology.get("allowed_relations", {}).items()
    )
    return f"""你是船舶焊接质量报告结构化抽取模型。
只输出一个合法JSON对象，必须包含defect_label、event_label、entities和relations。

defect_label只能选择：{defect_labels}
event_label只能选择：{event_labels}

实体类型和固定WSR层：
{entity_lines}

关系类型及允许的head类型->tail类型：
{relation_lines}

严格要求：
1. 实体text必须是报告中的连续原文片段。
2. start为包含端，end为不包含端，必须满足text=报告[start:end]。
3. 实体type只能使用上述实体类型，wsr_layer必须与类型规定一致。
4. 关系head和tail只能引用entities中的id。
5. 关系type及方向必须符合上述允许签名。
6. 不得输出WPS、DEFECT、EVENT、NOUN、PREDICATE等未定义类型。
7. 顶层只允许defect_label、event_label、entities、relations四个键。
8. 每个实体只允许id、text、start、end、type、wsr_layer六个键。
9. 每个关系只允许head、tail、type三个键。
10. 相同text、start、end、type的实体只能输出一次，实体总数不得超过20。
11. 没有明确实体或关系时输出空数组，不得编造、复读键名或循环生成。
12. 必须完整检查问题现象和原因分析，不得只输出最显眼的一个缺陷或一条关系。
13. 输出前逐项检查缺陷、位置、构件、物理条件、原因、要求、人员、团队和责任行为，原文支持的实体都应保留。
14. 对每个实体检查所有本体允许且有原文证据的关系，不得因已经生成一条关系而提前结束。
15. JSON闭合后立即结束，不输出Markdown、解释、示例或额外文字。"""


def build_text_schema_system_prompt(ontology: dict | None = None) -> str:
    if not ontology:
        return """你是船舶焊接质量报告结构化抽取模型。
只输出合法JSON对象，包含defect_label、event_label、entities和relations。
entities只输出text、type、wsr_layer；relations只输出head_text、relation、tail_text。"""
    defect_labels = "、".join(ontology.get("defect_labels", []))
    event_labels = "、".join(ontology.get("event_labels", []))
    entity_lines = "\n".join(
        f"- {name}: wsr_layer={spec['layer']}"
        for name, spec in ontology.get("entity_types", {}).items()
    )
    relation_lines = "\n".join(
        f"- {name}: "
        + "；".join(f"{head}->{tail}" for head, tail in pairs)
        for name, pairs in ontology.get("allowed_relations", {}).items()
    )
    return f"""你是船舶焊接质量报告结构化抽取模型。
只输出一个合法JSON对象，必须包含defect_label、event_label、entities和relations。

defect_label只能选择：{defect_labels}
event_label只能选择：{event_labels}

实体类型和固定WSR层：
{entity_lines}

关系类型及允许的head类型->tail类型：
{relation_lines}

严格要求：
1. entities中每个实体只输出text、type、wsr_layer三个键。
2. 不要输出实体id、start、end或confidence。
3. 实体text必须是报告中的连续原文片段。
4. relations中每个关系只输出head_text、relation、tail_text三个键。
5. head_text和tail_text必须等于entities中某个实体的text。
6. relation必须符合上面的关系类型、方向和实体类型签名。
7. 顶层只允许defect_label、event_label、entities、relations四个键。
8. 不得输出Markdown、解释、示例或额外文字。
9. JSON闭合后立即结束。"""


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def record_id(record: dict) -> str:
    value = record.get("case_id") or record.get("record_id")
    if not value:
        raise ValueError("Every record must have case_id or record_id")
    return str(value)


def canonical_target(record: dict) -> dict:
    entities = []
    for entity in record.get("entities", []):
        start = entity.get("start")
        end = entity.get("end")
        text = str(entity.get("text", ""))
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end <= start or record.get("text", "")[start:end] != text:
            continue
        entities.append(
            {
                "id": str(entity["id"]),
                "text": text,
                "start": start,
                "end": end,
                "type": str(entity["type"]),
                "wsr_layer": str(entity["wsr_layer"]),
            }
        )
    entity_ids = {entity["id"] for entity in entities}
    relations = [
        {
            "head": str(relation["head"]),
            "tail": str(relation["tail"]),
            "type": str(relation["type"]),
        }
        for relation in record.get("relations", [])
        if str(relation.get("head")) in entity_ids
        and str(relation.get("tail")) in entity_ids
    ]
    return {
        "defect_label": str(record.get("defect_label", "")),
        "event_label": str(record.get("event_label", "")),
        "entities": entities,
        "relations": relations,
    }


def target_text(record: dict) -> str:
    return json.dumps(canonical_target(record), ensure_ascii=False, separators=(",", ":"))


def text_schema_target(record: dict) -> dict:
    target = canonical_target(record)
    entity_by_id = {entity["id"]: entity for entity in target["entities"]}
    return {
        "defect_label": target["defect_label"],
        "event_label": target["event_label"],
        "entities": [
            {
                "text": entity["text"],
                "type": entity["type"],
                "wsr_layer": entity["wsr_layer"],
            }
            for entity in target["entities"]
        ],
        "relations": [
            {
                "head_text": entity_by_id[relation["head"]]["text"],
                "relation": relation["type"],
                "tail_text": entity_by_id[relation["tail"]]["text"],
            }
            for relation in target["relations"]
            if relation["head"] in entity_by_id and relation["tail"] in entity_by_id
        ],
    }


def build_user_prompt(record: dict) -> str:
    return f"""请抽取下面焊接质量报告中的缺陷类别、事件类别、实体和关系。

报告：
{record["text"]}
"""


def build_entity_system_prompt(ontology: dict) -> str:
    prompt = build_system_prompt(ontology)
    return prompt.replace(
        "必须包含defect_label、event_label、entities和relations。",
        "必须包含defect_label、event_label、entities和relations，"
        "其中relations必须是空数组。",
    ) + "\n本阶段只抽取分类和实体，不预测关系。"


def build_relation_system_prompt(ontology: dict) -> str:
    relation_lines = "\n".join(
        f"- {name}: "
        + "；".join(f"{head}->{tail}" for head, tail in pairs)
        for name, pairs in ontology.get("allowed_relations", {}).items()
    )
    return f"""你是船舶焊接关系抽取模型。
输入会提供报告和已经固定的实体清单。实体不得增删、改名、改边界或改类型。
只输出一个合法JSON对象，顶层只允许relations一个键。
每个关系只允许head、tail、type三个键。

关系类型及允许的head类型->tail类型：
{relation_lines}

严格要求：
1. head和tail只能引用输入实体清单中的id。
2. 关系方向和类型必须符合本体签名及报告语义。
3. 区分直接技术原因CAUSES和间接责任因素CONTRIBUTES_TO。
4. 穷尽原文明确支持的关系，不得编造。
5. 没有关系时输出{{"relations":[]}}。
6. JSON闭合后立即结束，不输出解释或Markdown。"""


def build_relation_user_prompt(record: dict) -> str:
    target = canonical_target(record)
    entities = json.dumps(
        target["entities"], ensure_ascii=False, separators=(",", ":")
    )
    return f"""请根据固定实体清单抽取报告中的全部关系。

报告：
{record["text"]}

固定实体清单：
{entities}
"""


def entity_target(record: dict) -> dict:
    target = canonical_target(record)
    target["relations"] = []
    return target


def relation_target(record: dict) -> dict:
    return {"relations": canonical_target(record)["relations"]}


def text_schema_entity_target(record: dict) -> dict:
    target = text_schema_target(record)
    target["relations"] = []
    return target


def text_schema_relation_target(record: dict) -> dict:
    return {"relations": text_schema_target(record)["relations"]}


def _confidence(record: dict) -> float:
    values = []
    for item in [*record.get("entities", []), *record.get("relations", [])]:
        value = item.get("confidence")
        if isinstance(value, (int, float)):
            values.append(float(value))
    confidence = sum(values) / len(values) if values else 1.0
    if not math.isfinite(confidence):
        return 1.0
    return min(1.0, max(0.1, confidence))


def _class_key(record: dict) -> str:
    return f"{record.get('defect_label', '')}::{record.get('event_label', '')}"


def _index_by_id(records: Iterable[dict], label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for record in records:
        case_id = record_id(record)
        if case_id in indexed:
            raise ValueError(f"{label}: duplicate case_id {case_id}")
        indexed[case_id] = record
    return indexed


def assert_disjoint(records: Iterable[dict], heldout_records: Iterable[dict]) -> None:
    train_ids = {record_id(record) for record in records}
    heldout_ids = {record_id(record) for record in heldout_records}
    overlap = sorted(train_ids & heldout_ids)
    if overlap:
        raise ValueError(
            f"Training/held-out leakage: {len(overlap)} overlapping case_ids; "
            f"first={overlap[:5]}"
        )


def validate_enhanced_teacher_records(records: Iterable[dict]) -> None:
    invalid = []
    for record in records:
        reasons = []
        if record.get("pipeline_components") != ENHANCED_TEACHER_COMPONENTS:
            reasons.append("incomplete_pipeline_components")
        if record.get("mode") != "with_wsr_ontology_gate_enhanced":
            reasons.append("invalid_mode")
        if not record.get("entity_refinement_success"):
            reasons.append("entity_refinement_failed")
        if "relations_before_gate" not in record:
            reasons.append("missing_relations_before_gate")
        if "rejected_relations" not in record:
            reasons.append("missing_rejected_relations")
        if reasons:
            invalid.append((record_id(record), reasons))
    if invalid:
        case_id, reasons = invalid[0]
        raise ValueError(
            "Enhanced teacher validation failed for "
            f"{len(invalid)} records; first={case_id}:{','.join(reasons)}"
        )


def _entity_signature(record: dict) -> set[tuple[int, int, str]]:
    return {
        (entity["start"], entity["end"], str(entity["type"]))
        for entity in canonical_target(record)["entities"]
    }


def _relation_signature(
    record: dict,
) -> set[tuple[tuple[int, int, str], str, tuple[int, int, str]]]:
    target = canonical_target(record)
    entities = {
        entity["id"]: (entity["start"], entity["end"], entity["type"])
        for entity in target["entities"]
    }
    return {
        (entities[relation["head"]], relation["type"], entities[relation["tail"]])
        for relation in target["relations"]
        if relation["head"] in entities and relation["tail"] in entities
    }


def calibrate_teacher_against_gold(
    teacher_records: Iterable[dict],
    gold_records: Iterable[dict],
    *,
    minimum_weight: float = 0.2,
    maximum_weight: float = 0.7,
) -> dict:
    """Estimate strict teacher reliability on independently annotated Gold."""
    teacher_by_id = _index_by_id(teacher_records, "teacher calibration data")
    gold_by_id = _index_by_id(gold_records, "gold calibration data")
    overlap = sorted(teacher_by_id.keys() & gold_by_id.keys())
    if not overlap:
        raise ValueError(
            "Teacher calibration requires overlapping case_ids between "
            "--teacher_train and --gold_train"
        )

    entity_tp = entity_predicted = entity_gold = 0
    relation_tp = relation_predicted = relation_gold = 0
    for case_id in overlap:
        teacher_entities = _entity_signature(teacher_by_id[case_id])
        gold_entities = _entity_signature(gold_by_id[case_id])
        entity_tp += len(teacher_entities & gold_entities)
        entity_predicted += len(teacher_entities)
        entity_gold += len(gold_entities)

        teacher_relations = _relation_signature(teacher_by_id[case_id])
        gold_relations = _relation_signature(gold_by_id[case_id])
        relation_tp += len(teacher_relations & gold_relations)
        relation_predicted += len(teacher_relations)
        relation_gold += len(gold_relations)

    entity_precision = entity_tp / max(1, entity_predicted)
    entity_recall = entity_tp / max(1, entity_gold)
    relation_precision = relation_tp / max(1, relation_predicted)
    relation_recall = relation_tp / max(1, relation_gold)
    raw_weight = 0.5 * (entity_precision + relation_precision)
    teacher_weight = min(maximum_weight, max(minimum_weight, raw_weight))
    return {
        "overlap_records": len(overlap),
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "relation_precision": relation_precision,
        "relation_recall": relation_recall,
        "raw_teacher_weight": raw_weight,
        "teacher_weight": teacher_weight,
    }


def make_wsr_negative_target(
    record: dict, ontology: dict, rng: random.Random | None = None
) -> dict | None:
    rng = rng or random.Random(0)
    target = canonical_target(record)
    entities = {entity["id"]: entity for entity in target["entities"]}
    relations = target["relations"]
    if not relations:
        return None

    candidates = list(relations)
    rng.shuffle(candidates)
    relation_types = [
        item
        for item in ontology.get("relation_types", [])
        if item != "NO_RELATION"
    ]
    for relation in candidates:
        head = entities.get(relation["head"])
        tail = entities.get(relation["tail"])
        if not head or not tail:
            continue
        valid_types = {
            relation_type
            for relation_type, pairs in ontology.get("allowed_relations", {}).items()
            if [head["type"], tail["type"]] in pairs
            or (head["type"], tail["type"]) in {tuple(pair) for pair in pairs}
        }
        invalid_types = [
            relation_type
            for relation_type in relation_types
            if relation_type not in valid_types
        ]
        if not invalid_types:
            continue
        negative = json.loads(json.dumps(target, ensure_ascii=False))
        relation_index = relations.index(relation)
        negative["relations"][relation_index]["type"] = rng.choice(invalid_types)
        return negative
    return None


def _relation_is_allowed(
    relation_type: str, head_type: str, tail_type: str, ontology: dict
) -> bool:
    return [head_type, tail_type] in ontology.get("allowed_relations", {}).get(
        relation_type, []
    )


def make_gate_rejected_targets(
    record: dict,
    ontology: dict,
) -> list[dict]:
    target = canonical_target(record)
    entities = {entity["id"]: entity for entity in target["entities"]}
    positive_relations = {
        (relation["head"], relation["type"], relation["tail"])
        for relation in target["relations"]
    }
    candidates = []
    seen = set()
    for rejected in record.get("rejected_relations", []):
        head_id = str(rejected.get("head") or "")
        tail_id = str(rejected.get("tail") or "")
        relation_type = str(rejected.get("type") or "").upper()
        head = entities.get(head_id)
        tail = entities.get(tail_id)
        key = (head_id, relation_type, tail_id)
        if (
            not head
            or not tail
            or relation_type == "NO_RELATION"
            or relation_type not in ontology.get("relation_types", [])
            or key in positive_relations
            or key in seen
            or _relation_is_allowed(
                relation_type, head["type"], tail["type"], ontology
            )
        ):
            continue
        negative = json.loads(json.dumps(target, ensure_ascii=False))
        negative["relations"].append(
            {
                "head": head_id,
                "tail": tail_id,
                "type": relation_type,
            }
        )
        conflict_type = str(
            (rejected.get("gate_result") or {}).get("conflict_type")
            or "multi_conflict"
        )
        candidates.append(
            {
                "counterfactual_type": f"gate_rejected_{conflict_type}",
                "negative_source": "gate_rejected",
                "target": negative,
            }
        )
        seen.add(key)
    return candidates


def make_recall_counterfactual_targets(
    record: dict,
    rng: random.Random | None = None,
) -> list[dict]:
    """Build valid-looking incomplete or boundary-corrupted structures."""
    rng = rng or random.Random(0)
    target = canonical_target(record)
    candidates: list[dict] = []

    if target["relations"]:
        negative = json.loads(json.dumps(target, ensure_ascii=False))
        negative["relations"].pop(rng.randrange(len(negative["relations"])))
        candidates.append(
            {
                "counterfactual_type": "relation_omission",
                "negative_source": "recall_counterfactual",
                "target": negative,
            }
        )

    relation_entity_ids = {
        relation[endpoint]
        for relation in target["relations"]
        for endpoint in ("head", "tail")
    }
    removable_entities = [
        entity
        for entity in target["entities"]
        if entity["id"] in relation_entity_ids
    ] or list(target["entities"])
    if removable_entities:
        removed = rng.choice(removable_entities)
        negative = json.loads(json.dumps(target, ensure_ascii=False))
        negative["entities"] = [
            entity
            for entity in negative["entities"]
            if entity["id"] != removed["id"]
        ]
        negative["relations"] = [
            relation
            for relation in negative["relations"]
            if removed["id"] not in {relation["head"], relation["tail"]}
        ]
        candidates.append(
            {
                "counterfactual_type": "entity_omission",
                "negative_source": "recall_counterfactual",
                "target": negative,
            }
        )

    text = str(record.get("text", ""))
    boundary_options = []
    for entity in target["entities"]:
        start, end = entity["start"], entity["end"]
        if start > 0 and text[start - 1] not in "\r\n":
            boundary_options.append((entity["id"], start - 1, end))
        if end < len(text) and text[end] not in "\r\n":
            boundary_options.append((entity["id"], start, end + 1))
        if end - start > 1:
            boundary_options.append((entity["id"], start + 1, end))
            boundary_options.append((entity["id"], start, end - 1))
    if boundary_options:
        entity_id, start, end = rng.choice(boundary_options)
        negative = json.loads(json.dumps(target, ensure_ascii=False))
        for entity in negative["entities"]:
            if entity["id"] == entity_id:
                entity["start"] = start
                entity["end"] = end
                entity["text"] = text[start:end]
                break
        candidates.append(
            {
                "counterfactual_type": "entity_boundary_corruption",
                "negative_source": "boundary_counterfactual",
                "target": negative,
            }
        )

    return candidates


def make_wsr_counterfactual_targets(
    record: dict,
    ontology: dict,
    rng: random.Random | None = None,
    max_negatives: int = 4,
) -> list[dict]:
    rng = rng or random.Random(0)
    target = canonical_target(record)
    entities = {entity["id"]: entity for entity in target["entities"]}
    relation_types = [
        item for item in ontology.get("relation_types", []) if item != "NO_RELATION"
    ]
    entity_types = list(ontology.get("entity_types", {}))
    candidates: list[dict] = []

    for relation_index, relation in enumerate(target["relations"]):
        head = entities.get(relation["head"])
        tail = entities.get(relation["tail"])
        if not head or not tail:
            continue

        invalid_types = [
            relation_type
            for relation_type in relation_types
            if relation_type != relation["type"]
            and not _relation_is_allowed(
                relation_type, head["type"], tail["type"], ontology
            )
        ]
        if invalid_types:
            negative = json.loads(json.dumps(target, ensure_ascii=False))
            negative["relations"][relation_index]["type"] = rng.choice(invalid_types)
            candidates.append(
                {
                    "counterfactual_type": "relation_type_swap",
                    "negative_source": "synthetic_counterfactual",
                    "target": negative,
                }
            )

        semantic_alternatives = [
            relation_type
            for relation_type in relation_types
            if relation_type != relation["type"]
            and _relation_is_allowed(
                relation_type, head["type"], tail["type"], ontology
            )
        ]
        if semantic_alternatives:
            negative = json.loads(json.dumps(target, ensure_ascii=False))
            negative["relations"][relation_index]["type"] = rng.choice(
                semantic_alternatives
            )
            candidates.append(
                {
                    "counterfactual_type": "relation_semantic_swap",
                    "negative_source": "semantic_counterfactual",
                    "target": negative,
                }
            )

        if not _relation_is_allowed(
            relation["type"], tail["type"], head["type"], ontology
        ):
            negative = json.loads(json.dumps(target, ensure_ascii=False))
            negative["relations"][relation_index]["head"] = relation["tail"]
            negative["relations"][relation_index]["tail"] = relation["head"]
            candidates.append(
                {
                    "counterfactual_type": "relation_direction_reverse",
                    "negative_source": "synthetic_counterfactual",
                    "target": negative,
                }
            )

        replacement_options = []
        for entity_id, entity in entities.items():
            if entity_id not in {relation["head"], relation["tail"]}:
                if not _relation_is_allowed(
                    relation["type"], entity["type"], tail["type"], ontology
                ):
                    replacement_options.append(("head", entity_id))
                if not _relation_is_allowed(
                    relation["type"], head["type"], entity["type"], ontology
                ):
                    replacement_options.append(("tail", entity_id))
        if replacement_options:
            endpoint, entity_id = rng.choice(replacement_options)
            negative = json.loads(json.dumps(target, ensure_ascii=False))
            negative["relations"][relation_index][endpoint] = entity_id
            candidates.append(
                {
                    "counterfactual_type": "endpoint_replacement",
                    "negative_source": "synthetic_counterfactual",
                    "target": negative,
                }
            )

        corruptions = []
        for endpoint, entity in (("head", head), ("tail", tail)):
            for entity_type in entity_types:
                if entity_type == entity["type"]:
                    continue
                head_type = entity_type if endpoint == "head" else head["type"]
                tail_type = entity_type if endpoint == "tail" else tail["type"]
                if not _relation_is_allowed(
                    relation["type"], head_type, tail_type, ontology
                ):
                    corruptions.append((endpoint, entity["id"], entity_type))
        if corruptions:
            _, entity_id, entity_type = rng.choice(corruptions)
            negative = json.loads(json.dumps(target, ensure_ascii=False))
            for entity in negative["entities"]:
                if entity["id"] == entity_id:
                    entity["type"] = entity_type
                    entity["wsr_layer"] = str(
                        ontology["entity_types"][entity_type]["layer"]
                    )
                    break
            candidates.append(
                {
                    "counterfactual_type": "entity_type_corruption",
                    "negative_source": "synthetic_counterfactual",
                    "target": negative,
                }
            )

    unique = []
    seen = set()
    for item in candidates:
        key = json.dumps(item["target"], ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    rng.shuffle(unique)
    selected = unique[:max_negatives]
    semantic = next(
        (
            item
            for item in unique
            if item["counterfactual_type"] == "relation_semantic_swap"
        ),
        None,
    )
    if (
        semantic is not None
        and semantic not in selected
        and selected
    ):
        selected[-1] = semantic
    return selected


def _as_training_items(
    record: dict,
    source: str,
    sample_weight: float,
    ontology: dict | None,
    need_negative: bool,
    rng: random.Random,
    stage: str = "joint",
    target_schema: str = "full",
) -> list[dict]:
    if target_schema == "text":
        need_negative = False
    gate_negatives = (
        make_gate_rejected_targets(record, ontology)
        if need_negative and ontology
        else []
    )
    synthetic_negatives = (
        make_wsr_counterfactual_targets(record, ontology, rng)
        if need_negative and ontology
        else []
    )
    recall_negatives = (
        make_recall_counterfactual_targets(record, rng)
        if need_negative
        else []
    )
    if gate_negatives:
        aggregate_target = json.loads(
            json.dumps(canonical_target(record), ensure_ascii=False)
        )
        existing_relations = {
            (relation["head"], relation["type"], relation["tail"])
            for relation in aggregate_target["relations"]
        }
        conflict_types = set()
        for item in gate_negatives:
            conflict_types.add(
                item["counterfactual_type"].removeprefix("gate_rejected_")
            )
            for relation in item["target"]["relations"]:
                key = (relation["head"], relation["type"], relation["tail"])
                if key not in existing_relations:
                    aggregate_target["relations"].append(relation)
                    existing_relations.add(key)
        selected_negative = {
            "counterfactual_type": (
                f"gate_rejected_{next(iter(conflict_types))}"
                if len(conflict_types) == 1
                else "gate_rejected_mixed"
            ),
            "negative_source": "gate_rejected",
            "target": aggregate_target,
        }
    elif recall_negatives or synthetic_negatives:
        negative_groups = [
            group
            for group in (recall_negatives, synthetic_negatives)
            if group
        ]
        selected_negative = rng.choice(rng.choice(negative_groups))
    else:
        selected_negative = None
    if stage == "entity":
        positive_target = (
            text_schema_entity_target(record)
            if target_schema == "text"
            else entity_target(record)
        )
        stage_negatives = [
            item
            for item in recall_negatives
            if item["counterfactual_type"]
            in {"entity_omission", "entity_boundary_corruption"}
        ]
        selected_negative = rng.choice(stage_negatives) if stage_negatives else None
        if selected_negative:
            selected_negative = {
                **selected_negative,
                "target": {
                    **selected_negative["target"],
                    "relations": [],
                },
            }
        system = (
            build_text_schema_system_prompt(ontology)
            + "\n本阶段只抽取分类和实体，relations必须为空数组。"
            if target_schema == "text"
            else build_entity_system_prompt(ontology or {})
        )
        prompt = build_user_prompt(record)
    elif stage == "relation":
        positive_target = (
            text_schema_relation_target(record)
            if target_schema == "text"
            else relation_target(record)
        )
        relation_negatives = [
            item
            for item in [*recall_negatives, *synthetic_negatives, *gate_negatives]
            if item["counterfactual_type"] != "entity_omission"
            and item["counterfactual_type"] != "entity_boundary_corruption"
            and item["counterfactual_type"] != "entity_type_corruption"
        ]
        selected_negative = rng.choice(relation_negatives) if relation_negatives else None
        if selected_negative:
            selected_negative = {
                **selected_negative,
                "target": {
                    "relations": selected_negative["target"]["relations"]
                },
            }
        system = build_relation_system_prompt(ontology or {})
        prompt = build_relation_user_prompt(record)
    else:
        positive_target = (
            text_schema_target(record)
            if target_schema == "text"
            else canonical_target(record)
        )
        system = (
            build_text_schema_system_prompt(ontology)
            if target_schema == "text"
            else build_system_prompt(ontology)
        )
        prompt = build_user_prompt(record)

    base = {
        "case_id": record_id(record),
        "system": system,
        "prompt": prompt,
        "target": json.dumps(
            positive_target, ensure_ascii=False, separators=(",", ":")
        ),
        "contrastive_key": _class_key(record),
        "source": source,
        "stage": stage,
    }
    if selected_negative is None:
        return [{**base, "sample_weight": float(sample_weight), "negative_target": None}]
    available_negatives = (
        gate_negatives
        if gate_negatives
        else [*recall_negatives, *synthetic_negatives]
    )
    return [
        {
            **base,
            "sample_weight": float(sample_weight),
            "negative_target": json.dumps(
                selected_negative["target"],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "counterfactual_type": selected_negative["counterfactual_type"],
            "negative_source": selected_negative["negative_source"],
            "available_counterfactual_types": sorted(
                {item["counterfactual_type"] for item in available_negatives}
            ),
            "available_gate_rejected_negatives": len(gate_negatives),
            "available_synthetic_negatives": len(synthetic_negatives),
            "available_recall_negatives": len(recall_negatives),
            "selected_gate_rejected_relations": (
                len(gate_negatives)
                if selected_negative["negative_source"] == "gate_rejected"
                else 0
            ),
        }
    ]


def build_training_records(
    method: str,
    *,
    gold_records: list[dict] | None = None,
    teacher_records: list[dict] | None = None,
    pseudo_records: list[dict] | None = None,
    heldout_records: list[dict] | None = None,
    ontology: dict | None = None,
    seed: int = 42,
    stage: str = "joint",
    teacher_weight: float | None = None,
    gold_weight: float = 1.5,
    gold_repeats: int = 1,
    teacher_aligned: bool = False,
    target_schema: str = "full",
) -> list[dict]:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}")
    if stage not in SUPPORTED_STAGES:
        raise ValueError(f"Unsupported training stage: {stage}")
    if target_schema not in SUPPORTED_TARGET_SCHEMAS:
        raise ValueError(f"Unsupported target schema: {target_schema}")
    gold_records = gold_records or []
    teacher_records = teacher_records or []
    pseudo_records = pseudo_records or []
    heldout_records = heldout_records or []
    rng = random.Random(seed)
    if teacher_weight is not None and teacher_weight <= 0:
        raise ValueError("teacher_weight must be positive")
    if gold_weight <= 0:
        raise ValueError("gold_weight must be positive")
    if gold_repeats < 1:
        raise ValueError("gold_repeats must be at least 1")

    if method == "qwen3_supervised":
        if not gold_records:
            raise ValueError("qwen3_supervised requires --gold_train")
        selected = [(record, "gold", 1.0) for record in gold_records]
    elif method == "self_training":
        if not pseudo_records:
            raise ValueError("self_training requires --pseudo_train")
        if not gold_records:
            raise ValueError("self_training requires --gold_train")
        gold_ids = {record_id(record) for record in gold_records}
        selected = [
            (record, "student_pseudo", _confidence(record))
            for record in pseudo_records
            if record.get("parse_success", True)
            and record_id(record) not in gold_ids
        ]
        selected.extend((record, "gold", 1.0) for record in gold_records)
    elif method in {"standard_kd", "contrastive_kd", "kd_wsr_post"}:
        if not teacher_records:
            raise ValueError(f"{method} requires --teacher_train")
        selected = [
            (record, "teacher", _confidence(record))
            for record in teacher_records
            if record.get("parse_success", True)
        ]
    else:
        if not teacher_records:
            raise ValueError("ours requires --teacher_train")
        if teacher_aligned:
            aligned_weight = 1.0 if teacher_weight is None else teacher_weight
            selected = [
                (
                    record,
                    "teacher_aligned",
                    min(_confidence(record), aligned_weight),
                )
                for record in teacher_records
                if record.get("parse_success", True)
            ]
        else:
            gold_ids = {record_id(record) for record in gold_records}
            selected = [
                (
                    record,
                    "teacher_structured",
                    (
                        _confidence(record)
                        if teacher_weight is None
                        else min(_confidence(record), teacher_weight)
                    ),
                )
                for record in teacher_records
                if record.get("parse_success", True)
                and record_id(record) not in gold_ids
            ]
            selected.extend(
                (record, "gold", gold_weight)
                for record in gold_records
                for _ in range(gold_repeats)
            )

    source_records = list(
        {record_id(record): record for record, _, _ in selected}.values()
    )
    _index_by_id(source_records, f"{method} training data")
    assert_disjoint(source_records, heldout_records)
    need_negative = method == "ours"
    items = []
    for record, source, weight in selected:
        stages = ("entity", "relation") if stage == "two_stage" else (stage,)
        for item_stage in stages:
            items.extend(
                _as_training_items(
                    record,
                    source,
                    weight,
                    ontology,
                    need_negative,
                    rng,
                    item_stage,
                    target_schema,
                )
            )
    rng.shuffle(items)
    return items
