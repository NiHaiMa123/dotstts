# Slice 8 数据就绪与准入策略

- 状态：基线已确认
- 日期：2026-09-07
- 范围：`fuxuan` 单说话人数据 277 条

## 当前输入

| 层 | 数量 | 结论 |
|---|---:|---|
| verified raw asset | 277 | content-addressed raw 均已摄取 |
| `training_audio@1` | 277 | 48 kHz mono PCM24，文件均存在 |
| filename transcript candidate | 277 | 均为非空文本 |
| quality pass | 261 | 可直接进入后续候选解析 |
| quality review | 16 | 16 条全部包含在正式人工审核的 40 条中且均 approved |
| quality reject | 0 | 无 |
| latest human review decision | 40 | approved 40 / rejected 0 / pending 0 |
| 未逐条人工听审文本 | 237 | 只能标记为 filename 来源，不能标记为 human reviewed |

当前数据只有一个目录解析 speaker，共 277 条。弱标签分布为 neutral 81 / happy 68 /
angry 122 / sad 6；应用 latest 人工决策后的 primary 分布将在冻结前重新计算并写入 stats，
不能回写覆盖原始弱标签。

## 训练 JSONL 契约

dots.tts 仓库 README、`JsonlManifestSourceAdapter` 与 `BasicTtsPipeline` 共同确认最小字段为：

```json
{"fid":"<stable id>","audio":"<absolute filesystem path>","text":"<non-empty text>"}
```

adapter 会透传额外字段，但训练所需派生 JSONL 固定只输出以上三字段；完整 lineage 放在
canonical Parquet、manifest 和 catalog 中。Slice 8 冻结器额外要求 `fid` 唯一、`audio`
存在且哈希匹配、`text` 去除首尾空白后非空。

## 文本准入与 provenance

冻结候选按以下优先级解析最终文本：

1. 有 latest review 且为 `approved`：使用 `review_decision.text_final`，记录 decision id、
   round、batch、`text_decision`，`text_source=human_review`。
2. 有 latest review 且为 `rejected` 或 `pending`：fail closed，不进入 frozen dataset。
3. 没有 review decision：使用 source metadata 的 `transcript_candidate`，记录 metadata profile
   id/version/config hash，`text_source=filename_candidate_unreviewed`。

因此当前冻结输入预期为 40 条 `human_review`（39 accept reference、1 accept edited）和
237 条 `filename_candidate_unreviewed`。后者虽可用于第一版训练冻结，但统计与 lineage 必须
显式暴露，不能写成“人工确认”；将来新增 review round 后必须提升 dataset version，不能
静默改变 v1。

## 标签准入与 provenance

1. latest review 为 `approved` 时，使用 review 的 primary/secondary/intensity，并保留原始
   `emotion_weak_label`；来源沿用 `weak_label_confirmed` 或 `human_corrected`。
2. 无 review decision 时，primary 使用原始 weak label，secondary/intensity 为空，来源标记
   `weak_label_unreviewed`。
3. review 为 `rejected`/`pending` 时不准入，不回退到 weak label。

当前预期为 31 条 `weak_label_confirmed`、9 条 `human_corrected`、237 条
`weak_label_unreviewed`。

## 质量 review 准入边界

当前 16 条 quality review 均已在真实浏览器中听审并获得 approved 决策，因此 v1 可保留，
但 canonical manifest 必须携带原 quality reason。通用规则为：quality `pass` 可准入；quality
`review` 仅在存在 latest approved 人工决策时准入；quality `reject` 或无人工批准的 review
均排除。这个例外不能扩展成以后自动放行所有 quality review。
