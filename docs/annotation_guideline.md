# 教师质量 Gold 标注规范

## 用途

`data/gold/teacher_quality_gold.jsonl`是独立评价Qwen3.6-27B教师输出的
人工金标准。不得将教师预测直接复制为Gold。

## 标注字段

- `defect_label`：必须来自`configs/wsr_ontology.yaml`的`defect_labels`。
- `event_label`：必须来自`event_labels`。
- `entities`：文本实体、字符跨度、类型与WSR层。
- `relations`：明确出现在报告中的实体关系。
- `triples`：可以由`relations`自动生成，也可人工填写。

## 字符跨度

- `start`包含，`end`不包含。
- 必须满足`text[start:end] == entity.text`。
- 实体必须是原文连续片段，不修正原文错别字。
- 相同文字在不同位置出现时，应分别标注。

## 实体类型

### Wuli

- `W_DEFECT`：缺陷或质量结果。
- `W_LOCATION`：焊缝、坡口、舱室、区域和位置。
- `W_COMPONENT`：板材、构件、管路、支座等物理对象。
- `W_PHYSICAL_CONDITION`：低温、雨水、油污、氧化皮和返潮等状态。

### Shili

- `S_PROCESS`：施工、焊接、打磨、检查、转序等过程行为。
- `S_CAUSE`：直接导致问题的工艺或操作原因。
- `S_REQUIREMENT`：图纸、工艺、标准和制度要求。
- `S_INSPECTION`：抽查、探伤、密性试验和复检。
- `S_REPAIR`：补焊、返修、整改和修理。

### Renli

- `R_PERSON`：施工人员、焊工、检验员和管理人员。
- `R_TEAM`：班组、部门、施工队和车间。
- `R_RESPONSIBILITY`：责任心不足、监管缺位、未检查和未培训等因素。

## 关系

- `CONTAINS`：构件或位置对缺陷/位置的包含与挂载。
- `CAUSES`：原因、过程、物理状态或责任因素指向结果。
- `EVOLVES_INTO`：检测、缺陷和返修之间的过程演化。
- `VIOLATES`：行为、原因或责任主体违反明确要求。

关系方向必须与`configs/wsr_ontology.yaml`中的`allowed_relations`一致。
文本不能明确支持关系时不要标注。

## 标注流程

1. 标注员A独立标注。
2. 标注员B复核实体跨度、类型和关系方向。
3. 分歧由领域专家裁决。
4. 完成后将`annotation.status`设为`reviewed`或`adjudicated`。

只有这两种状态能够进入正式评价。
