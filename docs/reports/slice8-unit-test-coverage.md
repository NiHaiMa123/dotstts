# Slice 8 单元测试覆盖矩阵

- 状态：通过
- 测试文件：`tests/test_dataset_freeze.py`
- 基线：148 项全量测试通过

| 退出要求 | 主要测试类 | 关键失败边界 |
| --- | --- | --- |
| 严格配置与官方 JSONL 契约 | `DatasetFreezeConfigTests`、`TrainingManifestContractTests` | 未校准配置、字段顺序、比例和空字段拒绝 |
| v8→v10 migration | `CatalogV10MigrationTests` | 旧数据保留、重复初始化幂等、CHECK/索引存在 |
| 候选查询与解析 | `DatasetCandidateQueryTests`、`DatasetCandidateResolutionTests` | run 隔离、latest review、缺文件/来源、哈希漂移 |
| 候选快照 | `DatasetCandidateSnapshotTests` | 顺序稳定、语义漂移、重复资产、原子幂等/冲突 |
| 精确/近似文本与音频 | `ExactAudioAnalysisTests`、`ExactTextAnalysisTests`、`NearTextAnalysisTests`、`AcousticFingerprintTests` | 文件漂移、空文本、Unicode、顺序稳定、缓存损坏 |
| speaker embedding/outlier | `SpeakerEmbeddingTests` | 模型权重漂移、采样率、静音、缓存损坏、排序稳定 |
| edge/group id | `DuplicateGraphTests` | typed edge、人工结论、canonical component id、边顺序 |
| group-aware/stratified split | `GroupAwareSplitTests`、`ReviewedSelectionTests`、`SplitAuditTests` | 同组跨 split、低资源分层、代表缺失、五类泄漏 |
| 原子发布与版本冲突 | `AtomicDatasetPublishTests` | staging 验证、幂等缓存、内容冲突、失败清理 |
| Parquet/JSONL/checksums | `DatasetArtifactTests` | 显式 schema、独立重建、checksum 篡改、音频哈希漂移 |

本矩阵只声明已由可执行测试覆盖的行为；真实 273 条数据的重建和 catalog 对账结果另见
Slice 8 验收报告。
