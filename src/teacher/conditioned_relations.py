from __future__ import annotations

import json


def build_relation_candidates(record: dict, ontology: dict) -> list[dict]:
    allowed_by_signature: dict[tuple[str, str], list[str]] = {}
    for relation_type, signatures in ontology["allowed_relations"].items():
        for head_type, tail_type in signatures:
            allowed_by_signature.setdefault((head_type, tail_type), []).append(
                relation_type
            )

    candidates = []
    for head in record.get("entities", []):
        for tail in record.get("entities", []):
            if head["id"] == tail["id"]:
                continue
            allowed = allowed_by_signature.get((head["type"], tail["type"]), [])
            if not allowed:
                continue
            candidates.append(
                {
                    "candidate_id": f"C{len(candidates) + 1}",
                    "head": head["id"],
                    "head_text": head["text"],
                    "head_type": head["type"],
                    "tail": tail["id"],
                    "tail_text": tail["text"],
                    "tail_type": tail["type"],
                    "allowed_labels": allowed + ["NO_RELATION"],
                }
            )
    return candidates


def build_conditioned_relation_prompt(
    record: dict, ontology: dict, candidates: list[dict]
) -> str:
    entity_rows = [
        {
            "id": entity["id"],
            "text": entity["text"],
            "type": entity["type"],
        }
        for entity in record["entities"]
    ]
    candidate_rows = [
        {
            "candidate_id": candidate["candidate_id"],
            "head": candidate["head"],
            "head_text": candidate["head_text"],
            "head_type": candidate["head_type"],
            "tail": candidate["tail"],
            "tail_text": candidate["tail_text"],
            "tail_type": candidate["tail_type"],
            "allowed_labels": candidate["allowed_labels"],
        }
        for candidate in candidates
    ]
    definitions = "\n".join(
        f"- {name}: {description}"
        for name, description in ontology["relation_definitions"].items()
    )
    return f"""你是船舶焊接质量报告的关系分类器。

实体已经由人工确定，禁止修改、删除、新增实体。你只需要判断给出的每个候选实体对是否存在关系。

关系定义：
{definitions}
- NO_RELATION：原文不足以支持候选关系，或两个实体只是共同出现。

判定规则：
1. 必须逐个返回所有candidate_id，每个候选只能选择allowed_labels中的一个标签。
2. CAUSES仅用于直接技术或物理成因；人员、培训、责任、检查和管理因素使用CONTRIBUTES_TO。
3. RESPONSIBLE_FOR只表示人员或团队承担某项责任行为，不能连接人员与普通技术原因。
4. VIOLATES必须有原文明确的要求、图纸、工艺或标准证据。
5. OCCURS_AT和AFFECTS必须有明确位置或部件证据。
6. 多个原因和多个缺陷不能自动做笛卡尔积，逐对判断原文是否支持。
7. 没有充分证据时输出NO_RELATION，不要为了增加关系数量而猜测。
8. 人员与缺陷之间不直接建立关系；人员只通过RESPONSIBLE_FOR连接其责任行为。
9. “没有检查/未及时发现”不能自动VIOLATES任意要求。只有责任文本明确提及同一要求（如“未按照互检要求检查”）才标VIOLATES。
10. 同时存在“开裂、夹渣”等具体缺陷与“缺陷较多”等汇总结果时，原因优先连接具体缺陷；除非原文只明确支持汇总结果，否则汇总实体输出NO_RELATION。
11. AFFECTS只连接缺陷与原文明确承载该缺陷的部件；不能因为部件和缺陷同在报告中就连接。
12. 在输出前进行一次否定复核：若删除该关系不影响原文事实表达，则倾向NO_RELATION。
13. 只输出合法JSON，不输出Markdown或解释文字。

输出格式：
{{
  "decisions": [
    {{
      "candidate_id": "C1",
      "label": "CAUSES",
      "confidence": 0.95
    }}
  ]
}}

报告原文：
{record["text"]}

人工实体：
{json.dumps(entity_rows, ensure_ascii=False, indent=2)}

候选实体对：
{json.dumps(candidate_rows, ensure_ascii=False, indent=2)}
"""


def parse_conditioned_relation_output(
    response_text: str, candidates: list[dict]
) -> dict:
    decoder = json.JSONDecoder()
    parsed = None
    errors = []
    text = (response_text or "").strip()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("decisions"), list):
            parsed = value
            break
    if parsed is None:
        return {
            "parse_success": False,
            "relations": [],
            "decisions": [],
            "errors": ["No decisions JSON object found"],
        }

    candidate_by_id = {
        candidate["candidate_id"]: candidate for candidate in candidates
    }
    raw_by_id = {}
    for raw in parsed["decisions"]:
        if not isinstance(raw, dict):
            continue
        candidate_id = str(raw.get("candidate_id") or "").strip()
        if candidate_id in candidate_by_id and candidate_id not in raw_by_id:
            raw_by_id[candidate_id] = raw

    decisions = []
    relations = []
    for candidate in candidates:
        raw = raw_by_id.get(candidate["candidate_id"], {})
        label = str(raw.get("label") or "NO_RELATION").strip().upper()
        if label not in candidate["allowed_labels"]:
            errors.append(
                f"{candidate['candidate_id']}_invalid_label:{label}"
            )
            label = "NO_RELATION"
        try:
            confidence = min(1.0, max(0.0, float(raw.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        decision = {
            "candidate_id": candidate["candidate_id"],
            "head": candidate["head"],
            "tail": candidate["tail"],
            "label": label,
            "confidence": confidence,
        }
        decisions.append(decision)
        if label != "NO_RELATION":
            relations.append(
                {
                    "head": candidate["head"],
                    "tail": candidate["tail"],
                    "type": label,
                    "confidence": confidence,
                }
            )
    if len(raw_by_id) != len(candidates):
        errors.append(
            f"missing_decisions:{len(candidates) - len(raw_by_id)}"
        )
    return {
        "parse_success": True,
        "relations": relations,
        "decisions": decisions,
        "errors": errors,
    }
