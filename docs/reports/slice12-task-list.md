# Slice 12 task list：正式实验与自动评测

- 状态：8/8 已执行；step-500 拒绝，后续按 validation loss 预选的 step-400 经 Plan2 优化后验收通过
- 入口：Slice 11 已验收；Slice 12 正式训练、自动评测和人工盲审均已完成。
- 当前停止点：Slice 12 已于 2026-09-11 收口，step-400 三层门禁全部通过，已进入 Slice 13。

## 任务

- [x] S12-1 固定 v2 正式实验 contract：区分 mf 部署基线、未训练 SOAR control 与 SOAR LoRA candidate，并明确聚合口径。
- [x] S12-2 完成 checkpoint、Docker、核心/可选依赖、冻结数据可读性 preflight，并记录哈希。
- [x] S12-3 建立未训练部署 baseline 评测矩阵：中性、长短句、数字/专名、参考外文本、多 seed；情感控制另开实验。
- [x] S12-4 运行 baseline，记录 CER、说话人相似度、音质、失败率、RTF、显存。
- [x] S12-5 先运行未训练 SOAR 同矩阵 control，再按冻结/LoRA 方案启动正式训练，绑定数据、配置、checkpoint 与日志。
  - [x] 技术前置、真实 1-step 训练和 compact checkpoint 保存/恢复 smoke。
  - [x] 未训练 SOAR control：生成、ASR、指标均为 72/72 成功。
  - [x] 正式 500-step LoRA 训练与 checkpoint 恢复验证。
- [x] S12-6 用同一评测矩阵评估训练后 checkpoint，并生成可追溯输出 sidecar；生成、ASR、指标均为 72/72 成功。
- [x] S12-7 自动做 control/训练后差异和回归阈值检查；7/10 通过，质量与绝对 RTF 共 3 项失败，已标记人工复核。
- [x] S12-8 完成 12 对盲审、解盲和 Slice 12 验收报告；control 胜 3、训练后胜 1、平局 5、两者都差 3，决定性样本中训练后胜率为 25%，step-500 候选拒绝。

## 验收结论

- 自动 gate：7/10 通过；训练后 quality pass rate 为 `0.0833`，低于 `0.5`，且相对 control 下降 `0.0972`，超过 `0.05` 上限；绝对 mean RTF 为 `1.9309`，超过 `1.0`。
- 人工 gate：拒绝；control 在决定性样本中以 3:1 胜出。前导静音 10/12 判为相同，control/训练后各有 1 条更差。
- 状态解释：Slice 12 的既定任务已全部执行，但候选模型没有通过验收；step-500 只作为研究产物保留，不发布、不进入 Slice 13。
- 下一候选：保留的 step-400 checkpoint 具有最低 validation loss（`0.6849`），可在下一轮按验证集选择规则独立评测；不得用本次 test/盲审结果反向挑 checkpoint。

## Step-400 独立复评

- 状态：4/4 已完成；step-400 人工盲审及 Plan2 推理/后处理正式复评均已通过。
- [x] S12-R1 按最低 validation loss 固定 step-400 checkpoint、权重哈希和独立输出路径。
- [x] S12-R2 用与 control 相同的 72 条矩阵生成 step-400 音频；72/72 成功、0 失败，配置哈希为 `62381c7af8b66854688830120909fd78f5f2670045ab3ee2965df51fe4760cbc`。
- [x] S12-R3 完成 72 条 ASR 与 CER、说话人相似度、质量、RTF、显存汇总；ASR 与指标均为 72/72 成功、0 错误。
- [x] S12-R4 原始自动 gate 为 6/10；人工盲审训练后 5 胜、control 3 胜、平局 2、两者都差 2。随后按预先固定的 Plan2 统一裁边和推理优化路线正式复评，模型、后处理、性能三层 gate 全部通过。

Step-400 最终正式结果：failure rate `0`、raw mean CER `0.05627`、raw speaker cosine
`0.79937`、postprocessed quality pass rate `0.8333`、optimized mean RTF `0.3004`、相对
control RTF `0.9566`、峰值显存 `6.4583 GiB`。全部阈值及输入哈希见
`docs/reports/plan2-acceptance-v1.json`；已允许进入 Slice 13。

## 约束

情感控制仍需单独实验和对照；不能仅因 JSONL 存在 emotion 字段，就宣称 basic
pipeline 已启用情感控制。
