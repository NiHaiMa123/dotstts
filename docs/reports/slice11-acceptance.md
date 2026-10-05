# Slice 11 验收：训练可行性与 tiny overfit

- 状态：已完成（6/6）
- 验收日期：2026-09-08
- 实验 ID：`slice11_tiny_overfit_v1`

## 结果

| 验收项 | 结果 | 证据 |
|---|---|---|
| 训练入口/依赖契约 | 通过（已记录缺失的 base/SOAR 与 PEFT 依赖） | [`slice11-training-feasibility.md`](slice11-training-feasibility.md) |
| 冻结/显存方案 | 通过（output-layer smoke；全量训练明确不通过 16 GB 预算） | [`tiny_overfit_v1.yaml`](../../configs/lab/slice11/tiny_overfit_v1.yaml) |
| 单 batch + 单 step | 通过 | `tiny_overfit_metrics.json`：峰值 allocated 约 9.53 GiB，checkpoint 写入成功 |
| 8 条 tiny overfit | 通过 | 32 步平均 loss `1.147500 → 1.010239`（约下降 11.96%） |
| 保存/恢复 | 通过 | `restore_max_abs_diff=0.0` |
| 固定句盲比材料 | 已准备 | `compare/blind_compare.html` 与两个 opaque WAV；主观分数未伪造 |
| 回归测试 | 通过 | `158 tests OK`；新增脚本 `py_compile` 通过；`git diff --check` 无错误 |

## 进入 Slice 12 的边界

可以进入 Slice 12 的正式实验规划，但正式训练开跑前必须完成 checkpoint/依赖
preflight：当前只有 `models/dots.tts-mf-1step` 推理 artifact，没有官方
`dots.tts-base`/`dots.tts-soar`。正式 LoRA 路径需要 `accelerate` 与 `peft`；
`bitsandbytes` 和 `flash_attn` 只是可选优化，首轮不依赖。本 Slice 的
2.23M-parameter output-layer adapter 仅证明链路和
可学习性，不能作为正式模型或情感控制结果。

Slice 12 的 baseline、训练配置、checkpoint 和评测必须重新绑定，不能复用这个
tiny adapter 的指标作为正式基线。
