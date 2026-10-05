# Slice 9 参考音频预选验收报告

- 状态：accepted（2026-09-08）
- 配置：`configs/lab/datasets/fuxuan_slice9_v1.yaml`
- 数据：`fuxuan@1`，dataset item 共 273 条；train=219、validation=27、test=27。
- 选择范围：仅 train 219 条，明确排除 validation/test 54 条，保持评测隔离。

## 结果

- 五个独立池：`main_neutral=10`、`emotion_neutral=6`、`emotion_happy=6`、
  `emotion_angry=6`、`emotion_sad=5`，共 33 个名额、30 个不同 asset。
- `main_neutral` 与 `emotion_neutral` 重叠 3 条，四个 emotion 池之间无交集。
- 47 条 selected/boundary 已人工确认 approved，并以 batch/round/provenance 追加到 catalog。
- 文本可信度：人工确认 29 条（tier A），文件名候选 190 条（tier B）；未提供匹配 ASR，全部保持 unknown。
- 拼音依赖 `pypinyin==0.53.0` 当前未安装；字符、标点和韵律特征已生成，拼音字段保持 null，评分按可用权重重归一化。

## 可复核产物

- [任务清单](slice9-task-list.md)
- [HTML 审听报告](../../data/reports/datasets/fuxuan_v1/analysis/slice9_candidates_v1.html)
- [JSON 审听报告](../../data/reports/datasets/fuxuan_v1/analysis/slice9_review_report_v1.json)
- [最终 manifest](../../data/references/fuxuan/v1/manifest.json)
- [最终 stats](../../data/references/fuxuan/v1/stats.json)
- [checksums](../../data/references/fuxuan/v1/checksums.txt)
- [端到端验收审计](../../data/reports/datasets/fuxuan_v1/audit/slice9_pipeline_acceptance_v1.json)

冻结 manifest SHA-256：`1a33c738c1b807337b97eaed278aee05c1458aebc1f4d16da32ba7dee879c085`；
stats SHA-256：`e706bada599dd786baab2b4507b8ed5f0a842af3248ae116ff8d62c10683f32d`；
checksums SHA-256：`85bbb83092020a56b40e22d805a9d25c218270853a551252f1afc2ee2d48afe0`。

## 验收

- 质量、speaker、文本 provenance、池隔离、pool-local 归一化、综合分和 deterministic diversity
  rerank 均有版本化报告。
- catalog 路径/哈希/文本/speaker/pool/rank 核验通过；独立临时目录 4 个冻结产物字节一致。
- 全量 unittest 158/158 通过，Slice9 新增测试 9/9 通过，compileall 通过，pip check 无破损依赖。

Slice 9 已完成。下一步是 Slice 10：固定文本/seed 的模型闭环生成与 CER、speaker 相似度、音质、失败率和稳定性评测；不在本报告中提前执行。
