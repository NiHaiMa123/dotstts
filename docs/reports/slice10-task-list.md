# Slice 10 task list：模型闭环参考排行

- 状态：已完成（8/8）
- 范围：`fuxuan@1` Slice 9 冻结的 30 个去重参考，固定中文测试句、固定模型和 seed。
- 停止条件：完成 S10-8 后停在 Slice 11 入口；不提前开展训练或 tiny overfit。

## 任务

- [x] S10-1 冻结评测契约：输入 manifest、模型/ASR 身份、测试句、seed、指标和失败/稳定性规则。
- [x] S10-2 生成并校验本次 benchmark manifest，记录所有输入哈希和候选池归属。
- [x] S10-3 实现可恢复的单进程推理 runner，逐项生成并记录 runtime、seed、参考文本及音频哈希。
- [x] S10-4 执行全量生成，确保每个参考×测试句×seed 都有成功或可解释失败状态。
- [x] S10-5 对生成结果计算 CER、说话人相似度、音质、失败率和 RTF，并分开保存静态分数。
- [x] S10-6 汇总 seed/句子稳定性，生成模型闭环排行和可复算的 tie-break 规则。
- [x] S10-7 生成盲听包与人工审核队列；若存在需人工确认项，在此停下并标明原因。
- [x] S10-8 完成 Slice 10 验收报告、哈希/回归检查和 task list 更新，明确 Slice 11 入口。

## Slice 10 完成说明

已接收并校验 120 条人工决策（keep 42、reject 78、uncertain 0），按严格多数票生成
最终聚合排行；Slice 10 验收报告已冻结。Slice 11 仅从可行性调研入口开始，不在本 task
list 中提前执行训练或 tiny overfit。

## 口径

静态分数来自 Slice 9 冻结报告；模型闭环分数只来自本配置中固定的 `dots.tts-mf-1step`
和 faster-whisper large-v3-turbo。任何缺失、异常或未完成的组合都不能以 0 分伪装成成功。
