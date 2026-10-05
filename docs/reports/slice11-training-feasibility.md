# Slice 11：训练可行性与 tiny overfit

## S11-1 入口与依赖契约

核对日期：2026-09-08。

固定的官方训练入口是 `scripts/train_dots_tts.py`，通过
`accelerate launch ... --config <yaml>` 启动；MeanFlow 蒸馏入口是
`scripts/train_dots_tts_meanflow.py`。训练 JSONL 的最小字段为 `fid`、`audio`、
`text`，基础 pipeline 不读取情感字段作为控制信号。

当前仓库状态：

| 项目 | 结论 |
|---|---|
| 官方 base checkpoint | 未下载：`pretrained_models/pretrain_cpt_decay/latest/model/` 不存在 |
| 官方 SOAR checkpoint | 未下载：`pretrained_models/dots.tts-soar/` 不存在 |
| 可用本地权重 | `models/dots.tts-mf-1step/`，完整推理 artifact，采样率 48 kHz、patch size 4、audio samples/token 7680 |
| GPU | NVIDIA GeForce RTX 5080，CUDA 可用；按项目约束按 16 GB 显存预算 |
| 参数高效训练依赖 | 正式 LoRA 必需 `accelerate`、`peft`；`bitsandbytes`、`flash_attn` 为首轮不启用的可选优化 |
| 训练器现状 | 默认 AdamW 选择所有 `requires_grad` 参数；模型构造时 vocoder 与 speaker encoder 已冻结，未内置 LoRA/4-bit |

因此 Slice 11 采用“本地 mf-1step artifact + 冻结大模块”的可行性 smoke
路径，不把它宣称为官方 base/SOAR 正式微调。正式训练仍需另行取得
`dots.tts-base` 或 `dots.tts-soar` 并按 16 GB 方案补齐 `accelerate`、`peft`。

## S11-2 冻结与显存方案

已固定配置：`configs/lab/slice11/tiny_overfit_v1.yaml`。

- smoke 路径冻结 `core.llm`、`core.patch_encoder`、DiT blocks、所有投影层、
  `core.eos_proj`、vocoder 和 speaker encoder，仅训练
  `core.velocity_field_predictor.output_layer`，先验证损失/保存恢复链路。
- 正式 16 GB 方案建议继续冻结 LLM、patch encoder、vocoder 和 speaker encoder，
  在 `core.velocity_field_predictor.blocks.*` 的 attention
  `q_proj/k_proj/v_proj/o_proj` 与 FFN `fc1/fc2` 注入 rank=8、alpha=16 LoRA；
  仅按需解冻 `hidden_proj/latent_proj/coordinate_proj/xvec_proj/eos_proj`。
- DiT blocks 用 `torch.utils.checkpoint` 包裹，关闭 LLM cache，bf16 autocast，
  batch size 1、梯度累积 1；Slice 11 当时未安装 `peft`，所以本轮不伪造 LoRA 已执行的结论。

参数盘点：模型总参数 2,386,134,622；构造后默认可训练 2,198,091,778，
明显不能以默认 AdamW 在 16 GB 上直接全量微调。tiny smoke 的 output layer
参数约 2.1M，作为安全的单步/过拟合验证面。

tiny manifest 已生成 8 条、无重复、无合成重复行：
`data/reports/datasets/fuxuan_v1/slice11/tiny_overfit_train.jsonl`，并有同名
`.metadata.json` 记录来源与音频哈希。

## S11-3 单 batch / 单 step smoke

已用真实 `BasicTtsPipeline`、vocoder latent 提取、speaker fbank、模型 loss 和
AdamW 完成 1 个 batch/1 个 optimizer step。结果写入
`data/work/slice11/tiny_overfit_v1/tiny_overfit_metrics.json`，并写出 adapter
checkpoint（仅 2,230,400 个 output-layer 参数）。

- step 1：`fm_loss=1.254555`，`eos_loss=0.088325`，总 loss `1.342880`，梯度范数 `0.424766`
- CUDA 峰值 allocated：`10,228,272,640` bytes（约 9.53 GiB）
- CUDA 峰值 reserved：`10,389,291,008` bytes（约 9.68 GiB）
- 1-step checkpoint：`data/work/slice11/tiny_overfit_v1/checkpoint-step00000001.pt`

这证明真实 batch/forward/backward/checkpoint 链路可运行，且该冻结 smoke 在 16 GB
卡上留有余量；不等同于全量训练可行。

## S11-4 8 条数据 tiny overfit

已用同一配置、seed `20260908`、8 条真实 Fuxuan v1 音频循环训练 32 步，结果仍在
`data/work/slice11/tiny_overfit_v1/tiny_overfit_metrics.json`。每 8 步的平均总 loss
为 `1.147500 → 1.089858 → 1.015817 → 1.010239`，下降约 11.96%；单条样本的
loss 会因逐条轮换而上下波动，这是预期现象。32-step adapter checkpoint 为
`data/work/slice11/tiny_overfit_v1/checkpoint-step00000032.pt`。

本实验只验证可学习性和训练链路，训练行中的情感标签没有进入 basic pipeline，
不能据此声称已经学会显式情感控制。

## S11-5 保存/恢复与固定句盲比

已验证：从 `checkpoint-step00000032.pt` 恢复到全新模型时，所有 adapter 参数的
最大绝对差为 `0.0`。固定句、固定 seed `20260908` 和同一参考音频已分别生成
baseline/adapted WAV，并准备了不泄露标签的盲比包：
`data/work/slice11/tiny_overfit_v1/compare/blind_compare.html`（同时有
`blind_compare.json`、`blind-S11-A.wav`、`blind-S11-B.wav`）。映射只放在同目录
`blind_mapping.json`，供审计而不出现在盲听页面。

审计摘要 `restore_and_blind_compare.json` 显示：baseline 3.04 s、adapted
2.56 s，均为 48 kHz、有限且非空音频。此处没有把无人值守的音质数值当成人工
盲听结论；如要形成主观偏好分数，仍需人工回听该页面。
