# Slice 13 输出预设合同

配置固定于 `configs/lab/slice13/release_presets_v1.yaml`，输入只能是已通过 Slice 12
三层门禁的 step-400 optimized formal manifest。导出器必须在开始前验证输入 manifest、
验收报告和裁边配置哈希。

## 预设

| preset | 信号处理 | 格式 | 用途 |
|---|---|---|---|
| raw | 无；原 WAV 按字节复制 | WAV | 归档和复现 |
| training | 无裁边、增益、限幅、重采样或有损编码 | PCM24 WAV | 后续训练/分析 |
| preview | 已验证安全裁边；目标 -18 LUFS；最大上调 12 dB；-1 dBTP 全局峰值保护 | PCM24 WAV | 审听 |
| release | 已验证安全裁边；目标 -16 LUFS；最大上调 12 dB；-1 dBTP 全局峰值保护 | PCM24 WAV、FLAC、MP3 | 发布 |

“安全限幅”固定为只降低整体增益的 peak-safe global gain，不使用会改变瞬态形状的硬削波。
当目标响度与峰值上限冲突时优先满足 -1 dBTP，允许实际响度低于目标。raw 与 training
不得执行任何波形处理；raw 还必须与来源 WAV 字节哈希一致。

每个输出必须带 JSON sidecar，记录源 manifest/音频哈希、模型与 adapter、参考音频和文本、
目标文本、seed、采样参数、实际后处理操作、输出编码信息与最终文件哈希。
