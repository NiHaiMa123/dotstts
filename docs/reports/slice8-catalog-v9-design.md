# Slice 8 Catalog v9 设计（含 v10 calibration overlay 扩展）

- 状态：v9 与 v10 扩展均已实现并完成真实 catalog 迁移
- 日期：2026-09-08
- 当前 schema：10

## v10 calibration overlay 扩展

v9 的分析表结构保留不变。v10 只新增两个表，用显式、不可变的 calibration identity 把
audit-only dataset config 提升为可运行分析的有效配置；不会原地修改已登记的 build config。

### `dataset_threshold_calibration`

主键 `(dataset_id, dataset_version, calibration_id, calibration_version)`，外键指向
`dataset_build_config`。保存 calibration config/report SHA-256、candidate snapshot SHA-256、
canonical thresholds JSON、报告路径和登记时间。同一 identity 的哈希或阈值变化必须拒绝；
相同内容重放幂等。

### `dataset_analysis_calibration`

以 `run_id` 为主键并外键指向 `dataset_analysis_run`；复合外键绑定上面的已登记 calibration
identity 及全部 SHA-256。开始 analysis run 时，run 与 calibration binding 在同一事务插入，
任何未登记、错版本、错 config/report SHA 或 candidate snapshot 漂移都必须在写入 run 前失败。

同一 dataset id/version 的多个 analysis run 必须使用完全相同的 calibration identity 与哈希，
避免后续 run 隐式切换到目录中另一个“最新”报告。读取 overlay 时还要交叉核验 calibration
报告内的 dataset identity、config SHA、candidate snapshot SHA、calibration identity、summary 与
detail 阈值；只有显式传入并通过这些核验的 overlay 才能解除 audit-only 门禁。

## 表与职责

### `dataset_build_config`

冻结完整 YAML 的 identity，主键 `(dataset_id, dataset_version)`。保存 `config_sha256`、canonical
`config_json`、`implementation_version`、`source_path`、`registered_at`。同 id/version 内容变化
必须拒绝，不能 update hash。

### `dataset_analysis_run`

一次候选与特征分析运行，主键 `run_id`，外键指向 build config。保存
`candidate_snapshot_sha256`、开始/结束时间、`running|succeeded|completed_with_errors|failed`
状态、候选/成功/错误数和错误信息。run 一旦完成，其 identity 和计数不可修改。

### `dataset_asset_feature`

主键 `(run_id, asset_sha256)`，外键到 analysis run、asset 和 derived audio。保存
`derived_id`、重新核验的 `standardized_sha256`、duration，以及以下两个独立特征：

- fingerprint little-endian float32 blob、dimension、feature SHA-256；
- L2-normalized speaker embedding little-endian float32 blob、dimension、feature SHA-256。

维度分别受 CHECK 约束为配置期望的 4096 和 512；blob hash 用于检测 catalog 损坏。

### `dataset_similarity_edge`

主键 `(run_id, left_asset_sha256, right_asset_sha256, evidence_type)`，强制
`left_asset_sha256 < right_asset_sha256`。`evidence_type` 为 `exact_audio|exact_text|near_audio|
near_text`；保存 score、threshold、evidence JSON。候选 edge 是不可变分析输出。

exact evidence 在生成时可直接标为 `accepted_exact`；near evidence 初始为 `pending_review`。
最终是否用于 group 不覆盖原 edge。

### `dataset_similarity_edge_review`

near edge 的 append-only 人工结论。主键 `review_id`，唯一键为 run/pair/evidence/round；保存
`accepted|rejected`、note、created_at 和 batch id。latest round 决定 edge 是否进入 group。
同 batch 重放必须幂等，冲突内容失败。

### `dataset_similarity_group` / `dataset_similarity_group_member`

group 表主键 `(run_id, group_id)`，保存 member count、evidence-review snapshot SHA-256 和创建
时间。member 表主键 `(run_id, group_id, asset_sha256)`，每个资产在同 run 只能属于一个 group。
`group_id` 是排序后 member asset hashes 的 canonical JSON SHA-256，边插入顺序不能改变它。
所有冻结候选（包括没有 accepted edge 的资产）都必须落入一个 group；因此孤立资产形成
singleton group。exact edge 直接参与连通分量，near edge 只有 latest review 为 `accepted` 时
参与；latest review snapshot 变化时，在同一事务删除旧 materialization 并写入完整新结果。

### `dataset_speaker_assessment`

主键 `(run_id, asset_sha256)`；保存 center cosine、kNN cosine、k、robust z/MAD 分数、
`candidate_outlier` 和 evidence JSON。该表只表达模型建议，不修改 source `speaker_id`。

### `dataset_speaker_review`

speaker outlier 的 append-only 人工结论，主键 `review_id`，唯一 run/asset/round；保存
`confirmed_same_speaker|exclude_wrong_speaker|exclude_uncertain`、note、时间和 batch。没有人工
结论的 candidate outlier 会阻止 freeze。

### `dataset_version`

成功发布的不可变版本，主键 `(dataset_id, dataset_version)`，外键到 build config 与 selected
analysis run。保存 candidate snapshot、edge review snapshot、speaker review snapshot、每个
canonical artifact 的路径/hash、总 item/duration 与 `published_at`。只在 staging 全部验证并
原子发布后插入；同内容重放是 verify-only，不同内容冲突。

### `dataset_item`

主键 `(dataset_id, dataset_version, asset_sha256)`，并对 `(dataset_id, dataset_version,
ordinal)`、`fid` 唯一。保存：

- `split` (`train|validation|test`) 与 `group_id`；
- raw asset、source root/path、derived id/path/hash、quality run/decision/reasons；
- exact final text/hash、`text_source` (`human_review|filename_candidate_unreviewed`)；
- original weak label、final primary/secondary/intensity、label source；
- nullable review decision id/round/batch；
- canonical `lineage_json`。

外键绑定 asset、source location、derived audio、quality run、review decision 和 similarity group。
CHECK 保证 human sources 必须有 decision id，unreviewed sources 必须没有 review id。

## 索引

- analysis run：build config + started time、status；
- feature：asset、derived id；
- edge：run + evidence/status、左右资产；
- edge/speaker review：lookup keys + round descending；
- group member：run + asset unique、group；
- speaker assessment：run + candidate_outlier、center/kNN score；
- dataset item：dataset + split/ordinal、group、text/label source、quality decision。

## 事务与不变量

- config 注册、run 开始/完成、feature/edge 批量落库、review append、group materialization、
  dataset publish 各自使用明确事务；group round 分配与 review append 使用 `BEGIN IMMEDIATE`。
- failed run 可留审计行，但不能被 dataset version 引用；只有 succeeded analysis 可冻结。
- feature/edge/group 必须属于同一 run 和 candidate snapshot。
- dataset item 集合必须等于 published manifest 集合；插入 version 与全部 item 在一个事务中。
- migration 只新增表和索引，不重写 v1–v8 数据；旧 catalog 原地升级后 40 条 Slice 7 history
  必须逐字节/逐字段保持不变。

## 实现测试要求

- 空 v0 → v9 与真实/fixture v8 → v9；重复 `initialize()` 幂等。
- foreign key、CHECK、unique、left/right canonical order、feature dimension/blob hash。
- 同 config/version 漂移拒绝；失败 run 不能发布；同 dataset version 同内容幂等、异内容冲突。
- edge 和 speaker review append-only round、batch 幂等与并发写入。
- 删除/篡改输入不会级联删除已冻结 lineage；读取时必须报告完整性错误。
- v9 → v10 只新增 calibration 表与索引；登记幂等、身份/哈希漂移拒绝，analysis run 与 binding
  原子写入；基础 dataset config 保持 audit-only，只有通过完整交叉核验的显式 overlay 可运行。
