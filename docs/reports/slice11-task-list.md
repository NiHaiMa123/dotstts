# Slice 11 task list：训练可行性与 tiny overfit

- 状态：已完成（6/6）
- 入口：Slice 10 已验收；Slice 10 最终 approved 参考为 6 个，其中 angry/sad 池暂无通过多数票的候选。
- 收口：Slice 11 已验收，后续已进入 Slice 12；此列表不再表示当前停止点。

## 任务

- [x] S11-1 调研并固定 dots.tts 训练入口、base/soar checkpoint 与显存/依赖契约。
- [x] S11-2 设计冻结模块、LoRA target layers、梯度检查点和 16 GB 显存方案。
- [x] S11-3 完成单 batch 前向与单 step smoke test，记录峰值显存和 checkpoint 写入路径。
- [x] S11-4 用 8–32 条数据做 tiny overfit，记录 loss 曲线、seed、参考和生成结果。
- [x] S11-5 测试保存/恢复，与基线做固定句盲比；若涉及显式情感控制另开实验。
- [x] S11-6 写验收报告并决定是否进入 Slice 12 正式实验。

## 约束

Slice 10 的模型闭环排行只用于参考选择，不能把情感字段写入 JSONL 就宣称官方 basic
pipeline 已启用情感控制；任何情感控制实验必须单独记录配置和对照组。
