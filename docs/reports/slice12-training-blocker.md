# Slice 12 S12-5 正式训练前置状态

S12-5 的技术阻塞已解除。官方 SOAR 上的真实 1-step LoRA 训练、compact checkpoint 保存和
恢复均已通过；正式 500-step 训练尚未启动，也没有把 smoke 结果冒充正式训练后指标。

## 已解除的阻塞项

1. contract 绑定的官方训练起点 `pretrained_models/dots.tts-soar/` 已按 immutable revision
   `2f9b3e18d70d670d4c701da2dc55ded5755815ce` 下载并逐文件哈希，不再是 blocker。
2. Docker Desktop + WSL2 Linux engine 已修复并通过容器内 RTX 5080/CUDA 可见性检查；
   训练镜像已构建，真实 1-step smoke 已通过。
3. Docker image 已把 `accelerate`、`peft` 固定为核心依赖；宿主 `.venv` 是否安装它们不再
   决定 Docker 训练能否启动。`bitsandbytes`、`flash_attn` 是首轮不启用的可选优化，
   不再误列为 blocker。
4. 冻结 JSONL 的 Windows 绝对路径已通过可追溯派生清单改写为 `/workspace/...`，不再需要
   修改冻结数据。
5. LoRA 注入、输出层可选解冻、DiT gradient checkpointing、compact trainable-delta
   checkpoint 保存恢复接线已实现并通过单元测试及真实 SOAR + GPU smoke。

## 可复现证据

- `data/reports/datasets/fuxuan_v1/slice12/preflight_v2.json`
- `data/reports/datasets/fuxuan_v1/slice12/soar_checkpoint.json`
- `configs/lab/slice12/experiment_contract_v2.yaml`
- `docker/slice12/Dockerfile` 与 `docker/slice12/compose.yaml`
- `data/work/slice12/container_manifests/metadata.json`
- `docs/reports/slice12-soar-lora-smoke-v1.json`
- `docs/reports/slice11-training-feasibility.md`
- `scripts/train_dots_tts.py`
- 官方入口要求：`accelerate launch scripts/train_dots_tts.py --config ...`

未训练 SOAR 同矩阵 control 已完成，生成、ASR 和指标均为 72/72 成功；汇总证据见
`docs/reports/slice12-soar-control-v1.json`。修复 10.0 秒 batch 预算会跳过 4 条 64–65-token
样本的问题后，正式 500-step 和 step-500 恢复验证均已完成，见
`docs/reports/slice12-soar-lora-training-v1.json`。随后 S12-6 至 S12-8 均已执行；自动与人工
gate 均拒绝 step-500，因此不得进入 Slice 13。完整结论见 `docs/reports/slice12-acceptance.md`。
