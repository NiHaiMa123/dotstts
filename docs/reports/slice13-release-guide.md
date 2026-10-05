# Slice 13 发布导出使用说明

最终候选是 SOAR LoRA step-400。正式输出位于
`data/exports/fuxuan_step400_v1/`，配置合同为
`configs/lab/slice13/release_presets_v1.yaml`。

## 常用命令

导出或原样重建完整四预设：

```powershell
.\.venv\Scripts\python.exe scripts/export_slice13_release.py
```

只重建完整 release 预设：

```powershell
.\.venv\Scripts\python.exe scripts/export_slice13_release.py --preset release
```

抽样或指定 job 时必须提供独立 run label，防止覆盖正式 manifest：

```powershell
.\.venv\Scripts\python.exe scripts/export_slice13_release.py --preset release --limit 2 --run-label smoke
.\.venv\Scripts\python.exe scripts/export_slice13_release.py --preset preview --job-id JOB_ID --run-label debug-job
```

抽样输出位于 `data/exports/fuxuan_step400_v1/runs/<run-label>/`。不带筛选条件的
导出才会更新正式 preset 目录。

验证当前正式导出：

```powershell
.\.venv\Scripts\python.exe scripts/verify_slice13_exports.py --output docs/reports/slice13-export-validation-v1.json
```

## 目录与选择

- `raw/`：72 个来源 WAV 的字节级副本，用于归档与复现。
- `training/`：72 个不做信号处理的 PCM24 WAV，用于后续训练/分析。
- `preview/`：72 个安全裁边并匹配约 -18 LUFS 的 PCM24 WAV，用于审听。
- `release/`：每个 job 各有 WAV、FLAC、MP3，共 216 个文件，目标约 -16 LUFS，
  true peak 不高于 -1 dBTP。

每个音频旁都有 `<filename>.<ext>.json` sidecar；preset 目录包含 `manifest.jsonl` 和
`summary.json`。sidecar 记录来源、模型/adapter、参考、文本、seed、推理参数、后处理、
格式和哈希。若响度目标与峰值上限冲突，系统优先满足峰值安全，因此少数 release 文件
可能低于 -16 LUFS；当前完整集合最低约 -18.21 LUFS。

## 不变量

- 输入 manifest、Slice12 验收状态和裁边配置哈希不匹配时拒绝导出。
- raw/training 不裁边、不增益、不限幅、不重采样；最终哈希必须等于来源音频。
- preview/release 只使用固定裁边和全局增益，不做硬削波。
- 所有写入采用临时文件后原子替换；相同输入重复执行必须得到相同 manifest 哈希。
