# Slice 12 验收报告（含 step-400 最终复评）

- 日期：2026-09-10
- 实验：`slice12_fuxuan_formal_v2`
- 初始候选：SOAR LoRA step-500（拒绝）
- 最终候选：SOAR LoRA step-400（按最低 validation loss 预选）
- 最终结论：**step-400 验收通过，进入 Slice 13**
- 执行状态：Slice 12 的 S12-1 至 S12-8 均已执行完；“任务完成”不代表“模型通过验收”。

## 自动门禁

训练前 control 与训练后 candidate 均完成 72/72 生成、ASR 和指标计算。10 项门禁通过 7 项，
失败 3 项：

| 指标 | Control | Step-500 | 门槛/结果 |
|---|---:|---:|---|
| mean CER | 0.0581 | 0.0563 | 小幅改善 |
| mean speaker cosine | 0.7818 | 0.7953 | 小幅改善 |
| quality pass rate | 0.1806 | 0.0833 | 低于 0.5，失败 |
| quality pass rate drop | - | 0.0972 | 超过 0.05，失败 |
| mean RTF | 1.6860 | 1.9309 | 超过 1.0，失败 |
| relative RTF ratio | 1.0 | 1.1452 | 不超过 1.15，通过 |
| peak CUDA memory | 5.5415 GiB | 5.5448 GiB | 通过 |

## 人工盲审

12 对样本覆盖 6 个句型、每句 2 个 seed；解盲前已完成全部选择。

| 结果 | 数量 |
|---|---:|
| Control 胜 | 3 |
| Step-500 胜 | 1 |
| 平局 | 5 |
| 两者都差 | 3 |

决定性样本共 4 对，step-500 胜率为 25%，人工 gate 按预定规则拒绝候选。前导静音判断中，
10/12 为相同，control 与 step-500 各有 1 条更差，因此无法把整体主观退化仅归因于前导静音。

## 决策与下一步

step-500 仅保留为可复现研究产物，不作为发布模型，不进入 Slice 13。训练过程中 step-400 的
validation loss 最低（`0.6849`，step-500 为 `0.7639`），且该 checkpoint 已保留。下一轮应
依据验证集这一预先可解释的选择规则评测 step-400；不得利用本次 test 或人工盲审结果反向
挑选 checkpoint。若 step-400 仍失败，再调整训练方案和质量目标，而不是放宽现有验收门槛。

后续记录：step-400 的初始 72/72 自动评测暴露了前导静音与未优化推理造成的质量/RTF
问题；没有改 checkpoint、test 集、seed 或采样步数。独立 12 对盲审完成后，训练后 5 胜、
control 3 胜、平局 2、两者都差 2，人工 gate 通过。Plan2 对 control 与 candidate 使用同一
确定性安全裁边，并保持 Euler 10 steps，仅启用 LoRA 数学合并和 runtime optimize。

最终正式复评两组均 72/72 成功。step-400 raw CER `0.05627`、speaker cosine `0.79937`；
裁边后 quality pass rate `0.8333`；mean RTF `0.3004`、相对 control `0.9566`、峰值 CUDA
`6.4583 GiB`。模型、后处理、性能三层 gate 的 13 个检查全部通过，因此最终接受 step-400
并进入 Slice 13。step-500 的拒绝结论不变。

## 可追溯证据

- 自动评测：`docs/reports/slice12-evaluation-v1.json`
- Step-400 复评：`docs/reports/slice12-step400-evaluation-v1.json`
- 训练报告：`docs/reports/slice12-soar-lora-training-v1.json`
- 本地规范决策：`data/reports/datasets/fuxuan_v1/slice12/slice12_blind_review_decisions.json`
- 本地解盲报告：`data/reports/datasets/fuxuan_v1/slice12/slice12_manual_review.json`
- Step-400 规范决策：`data/reports/datasets/fuxuan_v1/slice12/slice12_step400_blind_review_decisions.json`
- Step-400 解盲报告：`data/reports/datasets/fuxuan_v1/slice12/slice12_step400_manual_review.json`
- Plan2 最终验收：`docs/reports/plan2-acceptance-v1.json`
- 决策文件 SHA-256：`56235e1273bf83926eb4495dc2a8a34e512fc3fb3f11c0a49668e2132072ec4a`
- 解盲报告 SHA-256：`77f680f19d246285496b10360bbfd18d3814401fead1c0c941c08b14f13ae186`
