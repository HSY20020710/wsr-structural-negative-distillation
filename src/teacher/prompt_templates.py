from __future__ import annotations

import json


OUTPUT_TEMPLATE = {
    "defect_label": "",
    "event_label": "",
    "entities": [
        {
            "id": "E1",
            "text": "",
            "start": None,
            "end": None,
            "type": "",
            "wsr_layer": "",
            "confidence": 0.0,
        }
    ],
    "relations": [
        {
            "head": "E1",
            "tail": "E2",
            "type": "",
            "confidence": 0.0,
        }
    ],
    "triples": [
        {
            "head_text": "",
            "relation": "",
            "tail_text": "",
            "confidence": 0.0,
        }
    ],
    "causal_chain": [],
    "teacher_probs": {"defect": {}, "event": {}, "triple_validity": {}},
}

FLAT_DEFECT_LABELS = ["漏焊", "漏装", "裂纹", "气孔", "夹渣", "变形", "未除锈", "其他"]
FLAT_EVENT_LABELS = [
    "工艺纪律违规",
    "焊接管控不到位",
    "保护不到位",
    "检测返修问题",
    "责任管理问题",
    "设计工艺问题",
    "设备材料问题",
    "环境条件问题",
    "人员能力不足",
    "其他",
]
FLAT_ENTITY_TYPES = [
    "W_DEFECT",
    "W_LOCATION",
    "W_COMPONENT",
    "W_PHYSICAL_CONDITION",
    "S_PROCESS",
    "S_CAUSE",
    "S_REQUIREMENT",
    "S_INSPECTION",
    "S_REPAIR",
    "R_PERSON",
    "R_TEAM",
    "R_RESPONSIBILITY",
]
FLAT_RELATION_TYPES = [
    "CAUSES",
    "CONTRIBUTES_TO",
    "OCCURS_AT",
    "AFFECTS",
    "VIOLATES",
    "RESPONSIBLE_FOR",
]

RELATION_GUIDANCE = """关系语义规则：
- CAUSES：直接技术原因或物理条件 -> 缺陷。只有能解释缺陷形成机理时使用。
- CONTRIBUTES_TO：人员、能力、责任或管理因素 -> 缺陷，表示间接促成。
- OCCURS_AT：缺陷 -> 发生位置。
- AFFECTS：缺陷 -> 受影响部件。
- VIOLATES：原因、过程或责任行为 -> 原文明确出现的要求。
- RESPONSIBLE_FOR：人员或团队 -> 责任行为或管理失误。
不要使用 CONTAINS 或 EVOLVES_INTO。不要将多个原因和多个缺陷做笛卡尔积；逐条判断每个关系。"""

ENTITY_BOUNDARY_GUIDANCE = """实体边界规则：
- W_DEFECT：优先抽取缺陷或质量结果的核心短语，去掉位置、部件和无必要程度词。例如“焊缝气孔较多”抽“气孔”，“结构变形较多”抽“变形”。必须保留缺陷亚型修饰词，如“边缘裂纹”；“缺陷较多”“焊接指标偏低”等整体质量结果以及比较型缺陷应完整保留。
- W_LOCATION：优先抽取位置核心名称，如“船台合拢缝”抽“合拢缝”，“大成型作业区型材角焊缝”抽“角焊缝”；只有核心词无法表达位置语义时才保留完整限定。
- W_COMPONENT：抽取具体部件名，不把“焊接质量差”等结果当作部件。
- W_PHYSICAL_CONDITION：只抽物理或环境状态本身，如“焊丝返锈”“焊条返潮”，不要同时复制成S_CAUSE。
- S_CAUSE：抽取能够独立表达原因的最短完整行为或状态；不要包含“导致、造成”后的结果。
- S_REQUIREMENT：仅抽原文实际出现的图纸、工艺、标准或要求短语，不得补写原文不存在的“某某要求”。
- R_PERSON/R_TEAM：只抽人员或组织主体。
- R_RESPONSIBILITY：只抽责任、检查、传达、监管等失职行为，不把其中的人员名称并入实体。
- 不同语义类型允许嵌套或重叠。例如完整S_CAUSE中仍可另外标注R_PERSON、W_PHYSICAL_CONDITION或S_REQUIREMENT；仅禁止语义重复的同类型实体。
- 优先复用原文中的核心名称；原因和责任实体保持语义完整。"""


def _record_text(record: dict) -> str:
    return "\n".join(
        [
            f"发生时间：{record.get('occurrence_time', '')}",
            f"发生阶段：{record.get('occurrence_stage', '')}",
            f"问题现象：{record.get('problem_phenomenon', '')}",
            f"问题原因分析：{record.get('cause_analysis', '')}",
        ]
    )


def build_without_ontology_prompt(record: dict) -> str:
    # A flat shared label inventory keeps metrics comparable. No hierarchy,
    # definitions, layer semantics, or legal relation signatures are exposed.
    return f"""你是船舶焊接质量报告结构化抽取助手。

请抽取 defect_label、event_label、entities、relations、triples、
causal_chain 和 teacher_probs。问题现象通常描述结果，问题原因分析通常
描述原因；因果关系应从原因指向结果，不要把缺陷结果反向作为原因。

defect_label 从以下名称中选择一个：
{"、".join(FLAT_DEFECT_LABELS)}

event_label 从以下名称中选择一个：
{"、".join(FLAT_EVENT_LABELS)}

entity type 从以下名称中选择：
{"、".join(FLAT_ENTITY_TYPES)}

relation type 从以下名称中选择：
{"、".join(FLAT_RELATION_TYPES)}

{RELATION_GUIDANCE}

{ENTITY_BOUNDARY_GUIDANCE}

要求：
1. 只输出一个合法 JSON 对象，不输出 Markdown 或解释。
2. 实体 text 必须是原文中的连续片段。
3. start 为包含端，end 为不包含端；无法确定时可填 null。
4. 关系必须引用 entities 中已经出现的实体 id。
5. 没有明确关系时不要强行生成。
6. 不确定时降低 confidence，不要编造。
7. relation type 只能使用上述 relation type。
8. wsr_layer 填空字符串。
9. 实体使用最小但语义完整的连续片段。例如分别抽取“施工人员”、
   “责任心不足”和“未按图施工”，不要把整句合并为一个实体。
10. 不要生成互相重叠且语义重复的长短实体。

输出格式：
{json.dumps(OUTPUT_TEMPLATE, ensure_ascii=False, indent=2)}

报告：
{_record_text(record)}"""


def build_with_wsr_ontology_prompt(record: dict, ontology: dict) -> str:
    entity_definitions = "\n".join(
        f"- {name} ({spec['layer']}): {spec['description']}"
        for name, spec in ontology["entity_types"].items()
    )
    relation_types = "、".join(
        item for item in ontology["relation_types"] if item != "NO_RELATION"
    )
    relation_definitions = "\n".join(
        f"- {name}: {description}"
        for name, description in ontology.get("relation_definitions", {}).items()
    )
    defect_labels = "、".join(ontology["defect_labels"])
    event_labels = "、".join(ontology["event_labels"])
    allowed_relations = "\n".join(
        f"- {relation}: "
        + "；".join(f"{head} -> {tail}" for head, tail in pairs)
        for relation, pairs in ontology["allowed_relations"].items()
    )
    return f"""你是船舶焊接质量报告的 WSR 结构化抽取系统。

字段理解规则：
- 发生时间一般不作为实体抽取。
- 发生阶段可作为阶段信息，不强制抽取。
- 问题现象重点抽取缺陷、部位、构件和质量结果。
- 问题原因分析重点抽取原因、工艺过程、责任、管理、设计、设备、环境和人员能力因素。
- 因果方向一般为：问题原因分析中的原因、行为、管理因素或环境状态 -> 问题现象中的缺陷或质量结果。
- 不要把缺陷结果反向作为原因。

缺陷类别（defect_label 只能选一个）：
{defect_labels}

质量事件类别（event_label 只能选一个）：
{event_labels}

WSR 实体类型：
{entity_definitions}

关系类型：
{relation_types}

关系定义：
{relation_definitions}

允许的关系类型签名：
{allowed_relations}

{ENTITY_BOUNDARY_GUIDANCE}

严格要求：
1. 只输出一个合法 JSON 对象。
2. 不输出 Markdown，不输出解释。
3. 实体 text 必须来自原文连续片段。
4. 无法确定字符位置时 start/end 可以填 null。
5. 实体 type 必须来自上述 WSR 实体类型。
6. wsr_layer 必须与实体类型定义一致。
7. 关系必须引用已抽取实体 id。
8. 关系类型只能从 {relation_types} 中选择。
9. 文本中没有明确关系时不要强行生成。
10. 不确定时降低 confidence，不要编造实体或关系。
11. 关系必须符合上述允许的 head type -> tail type 签名。
12. 实体使用最小但语义完整的连续片段。例如分别抽取“施工人员”、
    “责任心不足”和“未按图施工”，不要把整句合并为一个实体。
13. 不要生成互相重叠且语义重复的长短实体。
14. CAUSES 只表示直接技术或物理原因；人员、能力、责任和管理因素使用 CONTRIBUTES_TO。
15. 缺陷与位置使用“缺陷 -> OCCURS_AT -> 位置”；缺陷与部件使用“缺陷 -> AFFECTS -> 部件”。
16. VIOLATES 的尾实体必须是原文中明确出现的要求、图纸、工艺或标准，不得凭空概括。
17. RESPONSIBLE_FOR 只连接人员/团队与其明确承担的责任行为。
18. 不要把多个原因与多个缺陷做笛卡尔积，必须逐条判断原文证据。

输出格式：
{json.dumps(OUTPUT_TEMPLATE, ensure_ascii=False, indent=2)}

报告：
{_record_text(record)}"""
