# Slice 13 验收报告

- 日期：2026-09-11
- 候选：SOAR LoRA step-400
- 状态：**通过**
- 输入：Slice12 已验收的 optimized formal generation manifest
- 配置哈希：`83a1609ba5d6295ea489cdc158d1e92e31ddcebb552a9193e6ae456c04cabe00`
- 输入 manifest 哈希：`a3f9eefb8150053ef7602873969067e58e0b8903a61b25df08cf15e642f0f2a5`

## 完整导出

| preset | job | 输出 | 格式 | 验证 |
|---|---:|---:|---|---|
| raw | 72 | 72 | WAV | 与来源逐文件字节哈希一致 |
| training | 72 | 72 | PCM24 WAV | 无信号处理且与来源逐文件字节哈希一致 |
| preview | 72 | 72 | PCM24 WAV | -18.11 至 -18.00 LUFS；最大 -1.0000 dBTP |
| release | 72 | 216 | WAV/FLAC/MP3 | -18.21 至 -15.97 LUFS；最大 -0.9996 dBTP |

完整导出连续执行两次，四个 manifest 哈希均保持一致。逐文件验证覆盖输出与 sidecar
哈希、格式集合、来源 manifest、模型/adapter、参考、文本、seed、采样参数、响度和
true peak，错误数为 0。release 最低响度低于 -16 LUFS 的项目由 -1 dBTP 峰值优先规则
约束，符合预设合同，不存在硬削波。

## 验证结果

- 完整项目测试：183/183 通过。
- Slice13 抽样保护变更后专项测试：2/2 通过。
- 本次新增/修改的 P2 与 Slice13 文件 Ruff：通过。
- `compileall`：通过。
- `git diff --check`：通过。
- 仓库全量 Ruff 仍报告 82 个历史/其他工作区问题，主要为旧文件 import 排序和既有
  未定义符号；未批量修改这些用户/其他 agent 变更，不影响本次 183 项运行测试及
  Slice13 产物验收。

## 可追溯证据

- Slice12 三层 gate：`docs/reports/plan2-acceptance-v1.json`
- 输出合同：`configs/lab/slice13/release_presets_v1.yaml`
- 输出合同说明：`docs/reports/slice13-output-contract.md`
- 完整逐文件验证：`docs/reports/slice13-export-validation-v1.json`
- 使用说明：`docs/reports/slice13-release-guide.md`
- 正式导出根目录：`data/exports/fuxuan_step400_v1/`
