# Slice 7–12 一致性审计与修复记录

- 日期：2026-09-08
- 范围：Slice 7、8、9、10、11、12 的冻结产物、任务状态、执行脚本、实验合同和训练前置条件
- 结论：Slice 7–11 的已验收数据/人工结论保持有效；发现的内部代码与文档问题已修复。
  Slice 12 的正式训练、同源评测和人工 gate 均已执行，但 step-500 候选被拒绝，不能进入 Slice 13。

## Slice 7

- 8 个关键审核/决策产物 SHA-256 与验收报告全部一致。
- Review 页面相关测试包含显式“保存成功”反馈、草稿恢复和导出校验，未发现数据漂移。
- `slice7-progress-snapshot.md` 是中断时的历史快照，不作为当前状态；最终状态以 task list 和 acceptance 为准。

## Slice 8

- canonical 8 个数据文件和 tree SHA-256
  `77754574c4048a20682b0e8f9be95c322cf0848afedaf90d7c81b1f00df029af` 与验收一致。
- train/validation/test 仍为 219/27/27；未修改 `datasets/fuxuan/v1/`。
- 修复两份设计文档仍写“待冻结/待校准/缺 pyarrow”的过时状态，补入最终校准和冻结结果。
- 冻结目录在受限进程中需要提升只读权限才能访问，但 Docker 后台和已授权校验可读取；未扩大本机 Users ACL。

## Slice 9

- reference pool、provenance、独立 rebuild 三项核验全部通过，4/4 产物 byte-identical。
- `pypinyin` 缺失是冻结报告中明确记录的可选特征降级；评分已按可用权重重归一化。
  现在补装并重算会改变已冻结 v1，因此没有静默改写。

## Slice 10

- 240/240 生成、240/240 ASR、30 个候选和 120 条人工决策均完整；所有输出 WAV 哈希匹配。
- 修复 `mean_cer or 1.0` 等逻辑：合法的数值 `0.0` 不再被当作缺失值，speaker/RTF 同理。
- 盲听最终页的多数票说明改为从配置生成，并明确 `keep > reject`；实际 4 句严格多数仍为至少 3 keep。
- ASR runner 现在自动切换到 `metrics.asr_python` 指定的隔离环境，不再依赖调用者手动选解释器。

## Slice 11

- 32-step history、首/末 8 步均值、2,230,400 可训练参数、checkpoint 和 restore diff=0 均复核通过。
- 修复 task list 仍称“只进入 Slice11”的过时停止条件。
- 依赖口径修正为：首轮 LoRA 必需 `accelerate`、`peft`；`bitsandbytes`、`flash_attn` 是不启用的可选优化。

## Slice 12

- 保留 v1 的 `mf-1step` 72 条部署基线及其固定 manifest，不覆盖历史产物。
- 新建 v2 合同：主比较改为未训练 SOAR control 与同源 SOAR LoRA candidate；指标聚合从歧义的
  `max_cer/min_speaker_cosine` 改为显式 mean/rate/max 字段，并分开推理/训练显存门槛。
- 官方 SOAR 固定到 revision `2f9b3e18d70d670d4c701da2dc55ded5755815ce`，下载后逐文件记录大小和 SHA-256。
- 新增 Docker Desktop/WSL2 配置、核心依赖、训练配置和 runner；首轮不依赖量化或 FlashAttention。
- 生成 Linux 容器三字段 manifest，源 JSONL 哈希仍为 train
  `317406266213d920cea2c3a2713b41210195beb858d130e97602c100227a08cc`、validation
  `a4af63be4b3da3f80538781784c1007b767bf4099164b883b8fb58776cf78e56`、test
  `02a2dd1894caf0cd323dc9d11fd9d5849ad1b67b6f1c19f05831446ff2272299`。
- 新增任意 PyTorch 模块的低层 PEFT LoRA 注入、全基座冻结、output layer 可选训练、DiT
  gradient checkpointing 和带 LoRA 架构的 checkpoint 重建路径。
- Docker 镜像已构建并确认 PyTorch 2.8.0+cu128、Transformers 4.57.6、Accelerate 1.12.0、
  PEFT 0.20.0 与 RTX 5080 可用；真实 1-step 训练、约 58.9 MB compact checkpoint 和恢复通过。
- 1-step 峰值 allocated 约 9.55 GiB、reserved 约 9.70 GiB，低于 15 GiB 训练门槛；
  原始 runtime metrics 在零步恢复后哈希不变。
- 未训练 SOAR 同矩阵 control 已完成：生成、ASR、指标均为 72/72 成功；mean CER
  `0.0581`、mean speaker cosine `0.7818`、quality pass rate `0.1806`、mean RTF `1.6860`、
  inference peak `5.5415 GiB`。回归门禁因缺训练后输入继续 fail-closed。
- 正式训练首轮发现 10.0 秒 batch 预算会因 token 对齐跳过 4 条样本，已归档该不完整尝试并
  修正为 10.5 秒/66 tokens 后从头重跑；正式 500-step 无漏样本警告，耗时 286.25 秒，
  峰值 allocated 约 9.59 GiB，step-500 compact checkpoint 恢复通过。
- 训练后同矩阵生成、ASR、指标均为 72/72 成功；CER 与 speaker cosine 小幅改善，但
  quality pass rate 从 `0.1806` 降至 `0.0833`，mean RTF 从 `1.6860` 升至 `1.9309`。
  自动门禁 7/10 通过，质量绝对值、绝对 RTF、质量相对下降 3 项失败。12 对人工盲审解盲后
  control 胜 3、训练后胜 1、平局 5、两者都差 3，人工 gate 拒绝 step-500；前导静音 10/12
  判为相同，control/训练后各有 1 条更差。
- 新增 fail-closed regression gate：缺少 SOAR control 或 trained 输入时明确输出
  `blocked_missing_inputs`。
- Slice 12 的 8 项任务已执行完，但候选验收失败；下一候选应按最低 validation loss 预先选择
  保留的 step-400 并独立评测，不能使用 test/盲审结果反向选点。

## 验证

- 全量 unittest：170/170 通过。
- 最新训练/PEFT 与 Slice12 工具针对性回归：8/8 通过。
- `compileall -q src scripts tests`：通过。
- `git diff --check`：通过（仅有现有 Windows LF/CRLF 提示）。
- Docker Compose 静态解析：通过。
- Docker 容器 GPU：RTX 5080 / 16,303 MiB 可见；修改文件 Ruff 检查通过。

Ruff 使用 Docker 训练环境执行，不依赖宿主 `.venv`。
