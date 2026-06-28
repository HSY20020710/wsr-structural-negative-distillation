from __future__ import annotations

import json

from src.teacher.parser import extract_json_from_response, normalize_teacher_output


CONSENSUS_EXEMPT_TYPES = {
    "R_PERSON",
    "R_TEAM",
    "R_RESPONSIBILITY",
    "S_REQUIREMENT",
    "W_PHYSICAL_CONDITION",
}


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


def filter_consensus_entities(
    refined_entities: list[dict], first_pass_entities: list[dict]
) -> list[dict]:
    kept = []
    for entity in refined_entities:
        supported = any(
            _span_overlap(entity, first_pass) for first_pass in first_pass_entities
        )
        if supported or entity.get("type") in CONSENSUS_EXEMPT_TYPES:
            kept.append(entity)
    return kept


def _mapping_score(source: dict, target: dict) -> float:
    if not _span_overlap(source, target):
        return -1.0
    intersection = min(source["end"], target["end"]) - max(
        source["start"], target["start"]
    )
    union = max(source["end"], target["end"]) - min(
        source["start"], target["start"]
    )
    text_bonus = 0.0
    if source.get("text") == target.get("text"):
        text_bonus = 2.0
    elif (
        str(source.get("text") or "") in str(target.get("text") or "")
        or str(target.get("text") or "") in str(source.get("text") or "")
    ):
        text_bonus = 1.0
    return text_bonus + intersection / union


def remap_relations(
    first_pass_entities: list[dict],
    first_pass_relations: list[dict],
    refined_entities: list[dict],
) -> list[dict]:
    source_by_id = {
        entity.get("id"): entity for entity in first_pass_entities
    }

    def best_target(source: dict | None) -> dict | None:
        if not source:
            return None
        ranked = sorted(
            (
                (_mapping_score(source, target), target)
                for target in refined_entities
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return ranked[0][1] if ranked and ranked[0][0] >= 0 else None

    relations = []
    seen = set()
    for relation in first_pass_relations:
        head = best_target(source_by_id.get(relation.get("head")))
        tail = best_target(source_by_id.get(relation.get("tail")))
        if not head or not tail:
            continue
        key = (head["id"], relation.get("type"), tail["id"])
        if key in seen:
            continue
        seen.add(key)
        relations.append(
            {
                "head": head["id"],
                "tail": tail["id"],
                "type": relation.get("type"),
                "confidence": relation.get("confidence", 0.0),
            }
        )
    return relations


def build_entity_refinement_prompt(record: dict, ontology: dict) -> str:
    entity_types = "\n".join(
        f"- {name}: {spec['description']}"
        for name, spec in ontology["entity_types"].items()
    )
    current_entities = [
        {
            "id": entity.get("id"),
            "text": entity.get("text"),
            "start": entity.get("start"),
            "end": entity.get("end"),
            "type": entity.get("type"),
        }
        for entity in record.get("entities", [])
    ]
    output = {
        "defect_label": record.get("defect_label", ""),
        "event_label": record.get("event_label", ""),
        "entities": [
            {
                "id": "E1",
                "text": "气孔",
                "start": 10,
                "end": 12,
                "type": "W_DEFECT",
                "wsr_layer": "Wuli",
                "confidence": 0.95,
            }
        ],
        "relations": [],
    }
    return f"""你是船舶焊接质量报告的实体边界校准器。
第一阶段已经给出候选实体，但可能存在边界过长、边界过短、类型错误、漏抽和重复词定位错误。
请重新阅读原文，输出最终实体。可以修改、拆分、删除或新增候选实体，但不得改写原文。

实体类型：
{entity_types}

人工Gold标注采用以下边界规范：
1. W_DEFECT抽取缺陷或质量结果的核心短语，去掉位置、部件和无必要程度词。
   例如“焊缝气孔较多”抽“气孔”；“结构变形较多”抽“变形”；
   “焊接成型差”抽“成型差”；“漏装结构”抽“漏装”。
   但必须保留能区分缺陷亚型的修饰词，如“边缘裂纹”不能缩成“裂纹”；
   “缺陷较多”“焊接指标偏低”“合格率较低”等整体质量结果应完整保留；
   “焊角与标注焊角不符合图纸要求”等比较型缺陷应保留完整比较对象和判断。
2. W_COMPONENT通常抽取通用部件核心名称。
   例如“货舱T型材”抽“型材”；“矩形风管”抽“风管”；
   “结构及板材”只抽“板材”。
3. W_LOCATION通常抽取位置核心名称。
   例如“船台合拢缝”抽“合拢缝”；“大成型作业区型材角焊缝”抽“角焊缝”。
4. W_PHYSICAL_CONDITION抽取物理条件名词本身。
   例如“带锈施焊”除抽完整S_CAUSE外，还抽“锈”；
   “油脂、水渍、氧化皮等垃圾物没有处理”分别抽“油脂”“水渍”“氧化皮”。
5. S_CAUSE保留能够完整表达原因的原文片段，不要只留下动作对象；
   原因中出现的R_PERSON、W_PHYSICAL_CONDITION或S_REQUIREMENT可以另外嵌套标注。
6. R_PERSON只抽“施工人员”“管理人员”“新员工”等主体词；
   R_RESPONSIBILITY抽完整失职行为，但不包含句末额外处置动作。
7. S_REQUIREMENT抽原文明确出现的要求核心短语，如“图纸要求”“工艺要求”
   “焊接工艺要求”“互检要求”，不能补写原文没有的内容。
   若要求是一个明确动作，如“按要求采用三合一小车自动化焊接”，保留该完整动作。
8. 不同语义类型允许嵌套或重叠。不要因为已经抽取完整S_CAUSE，
   就漏掉其中的人员、物理条件或要求实体。
9. 同一文本出现多次时，start/end必须指向承担当前语义角色的那一次，
   不能一律选择第一次出现。
10. start为包含端，end为不包含端，并且原文[start:end]必须等于text。
11. 输出全部最终实体，不输出关系。只输出合法JSON，不输出解释。

报告原文：
{record["text"]}

第一阶段候选实体：
{json.dumps(current_entities, ensure_ascii=False, indent=2)}

输出格式：
{json.dumps(output, ensure_ascii=False, indent=2)}
"""


def parse_refined_entities(
    response_text: str, source_record: dict, ontology: dict
) -> dict:
    parsed = extract_json_from_response(response_text)
    if not parsed["parse_success"]:
        return {
            "parse_success": False,
            "entities": [],
            "errors": parsed["errors"],
        }
    normalized = normalize_teacher_output(
        {
            **parsed["data"],
            "defect_label": parsed["data"].get(
                "defect_label", source_record.get("defect_label", "")
            ),
            "event_label": parsed["data"].get(
                "event_label", source_record.get("event_label", "")
            ),
            "relations": [],
        },
        source_record,
        ontology,
    )
    return {
        "parse_success": True,
        "entities": normalized["entities"],
        "errors": normalized["errors"],
    }
