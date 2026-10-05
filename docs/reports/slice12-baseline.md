# Slice 12 baseline 结果

S12-4 已完成。baseline 使用本地 `dots-studio/dots.tts-mf-1step`，固定
6 个 approved reference、6 类测试句、2 个 seed，共 72 个 job。生成与 ASR
均为 72/72 成功，自动指标文件和排名文件均为 `succeeded`。

## 固定产物

- generation：`data/reports/datasets/fuxuan_v1/slice12/baseline_generation.jsonl`
- ASR：`data/reports/datasets/fuxuan_v1/slice12/baseline_asr.jsonl`
- metrics：`data/reports/datasets/fuxuan_v1/slice12/baseline_metrics.json`
- ranking：`data/reports/datasets/fuxuan_v1/slice12/baseline_ranking.json`
- ranking HTML：`data/reports/datasets/fuxuan_v1/slice12/baseline_ranking.html`
- baseline manifest SHA-256：`ddfb42d6211f1c882bf77fe3c273dc7423fd52a8be3824a5650e66a6f43927b1`

## 汇总

| 指标 | 结果 | Slice12 预设阈值 | 说明 |
|---|---:|---:|---|
| generation / ASR failure rate | 0 / 0 | ≤ 5% | 本次无失败项 |
| CER（72 条平均） | 0.06398 | ≤ 0.20 | 平均值；最大单条为 0.31579 |
| speaker cosine（72 条平均） | 0.79532 | ≥ 0.75 | 平均值；最小单条为 0.33798 |
| quality pass rate | 16/72 = 22.22% | ≥ 50% | 56 条为 `review`，0 条 `fail` |
| RTF（平均） | 0.3981 | ≤ 1.0 | 最大单条为 1.3020，需训练后按同口径复核 |
| 峰值 CUDA 显存 | 5.54 GiB | ≤ 12 GiB | `peak_torch_cuda_bytes` 最大值 |

这些数值是未训练 baseline 的对照基线，不是训练后验收结论。质量 `review`
主要由静音段规则触发；因此 S12-7 仍需按 contract 明确逐条回归和人工复核。

