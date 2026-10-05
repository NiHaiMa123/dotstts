# Slice 13 task list：后处理与发布

- 入口：Slice 12 的 step-400 最终候选已通过模型、后处理和性能三层 gate。
- 当前状态：4/4 完成；Slice 13 已验收。
- 发布模型：SOAR LoRA step-400；推理固定为 LoRA merge、runtime optimize、BF16、Euler 10 steps、guidance 1.2。

## 任务

- [x] S13-1 固定输出预设：raw/training 不做破坏性处理；preview/release 使用已验证裁边、响度匹配与安全限幅。
- [x] S13-2 实现可复跑导出器；WAV/FLAC/MP3 每个文件带模型、adapter、参考、文本、seed、采样参数、后处理链和哈希 sidecar。
- [x] S13-3 建立幂等、格式、削波、响度、哈希和来源追溯测试，并做实际导出 smoke。
- [x] S13-4 写 Slice 13 验收报告和使用说明，更新总 PLAN。

## 验收结论

完整 72-job 四预设共 432 个音频已导出，两次运行 manifest 哈希一致；逐文件格式、
响度、true peak、哈希和来源追溯扫描 0 错误。完整结论见
`docs/reports/slice13-acceptance.md`。

## 固定证据

- Slice 12 最终 gate：`docs/reports/plan2-acceptance-v1.json`
- 推理选择：`docs/reports/plan2-inference-selection-v1.json`
- 裁边配置：`configs/lab/postprocess/generated_audio_edge_trim_v1.yaml`
- 正式候选 manifest：`data/reports/datasets/fuxuan_v1/slice12/soar_lora_step400_optimized_formal_generation.jsonl`
