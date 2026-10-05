# Slice 12 正式实验 contract

当前合同为 `configs/lab/slice12/experiment_contract_v2.yaml`；v1 作为已运行的
`mf-1step` 基线历史合同保留，不覆盖。

- 数据：Fuxuan v1，参考池绑定 Slice10 最终 6 个 approved reference。
- 部署基线：现有本地 `dots.tts-mf-1step`，保留已经完成的 72 条可复现推理结果。
- 训练对照：正式训练前先用未训练 `dots.tts-soar` 跑同一矩阵；训练后只与这个同源
  SOAR control 做主回归比较，避免把不同模型家族误当作 before/after。
- 训练起点：`dots.tts-soar` 已按 immutable revision
  `2f9b3e18d70d670d4c701da2dc55ded5755815ce` 下载并逐文件哈希；LoRA checkpoint 只保存
  trainable delta，恢复时必须叠加该官方基座。
- 评测矩阵：6 个参考 × 6 个句型 × 2 个 seed，覆盖中性、专名、长句、数字、短句和参考外文本。
- 指标口径已显式固定为均值/比例或最大值，不再用 `max_cer` 之类歧义字段；推理显存
  上限 12 GiB，训练峰值显存上限 15 GiB，且另设相对 SOAR control 的回归阈值。
- 现有 6 类句型不包含可控情感输入；情感控制仍是单独实验，不把参考音频目录标签当作
  模型具备显式情感控制的证据。

阈值是 Slice12 的预设回归门槛，不是对尚未运行的训练结果的结论。

真实 1-step smoke 已验证配置、显存、保存和恢复链路，但不属于正式 500-step 实验结果。
