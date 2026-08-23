# Evidence Chain 规范

每次关键动作至少写入：`event_id, run_id, timestamp, actor, event_type, input_ref, output_ref, status, provenance`。

必须覆盖：
- 数据版本/划分/输入 hash；
- 代码 revision 与配置；
- Agent 规划决定；
- Tool 名称、参数、返回状态；
- Gate 规则/冲突来源；
- 负样本的 parent positive 与 perturbation；
- checkpoint / prediction / metrics artifact hash；
- replay 差异。

禁止：用 Agent 的自然语言总结替代 raw tool output。
