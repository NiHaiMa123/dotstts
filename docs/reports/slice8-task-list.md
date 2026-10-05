# Slice 8 去重、说话人一致性与数据集冻结任务清单

- 状态：完成（41/41）
- 创建：2026-09-07
- 前置：Slice 7 已通过；正式 catalog 有 40 条 approved 人工决策
- 目标：无跨 split 精确/近重复；冻结版本可完全重建；每条样本可追溯到 raw、标准化音频、
  文本与标签决策

## 已确认基线

- [x] 确认 Slice 7 退出条件全部完成后再启动 Slice 8。（Slice 7 清单 36/36，验收报告
  状态“通过”，`PLAN.md` 已写完成记录）
- [x] 盘点真实数据就绪度。（277 raw / 277 standardized / 277 非空 filename transcript；
  quality 为 261 pass / 16 review / 0 reject；40 条人工决策均 approved）
- [x] 盘点当前重复与分层基线。（单一 speaker 共 277 条；当前规范化文本精确重复组 0；
  weak emotion 为 neutral 81 / happy 68 / angry 122 / sad 6）
- [x] 盘点本地依赖与模型缓存。（`pandas`、`sklearn`、`transformers`、`torchaudio` 可用；
  `pyarrow`、`speechbrain`、`resemblyzer` 不可用；后续在本地
  `models/dots.tts-mf-1step/` 找到并验证官方 CAM++ speaker encoder）

## P0：冻结契约与风险决策

- [x] 核对 dots.tts 官方训练器的三字段 JSONL 契约、路径语义和文本要求；测试锁定字段名。
  （README、`JsonlManifestSourceAdapter` 和 `BasicTtsPipeline` 确认最少字段为 `fid`、
  `audio`、`text`；冻结输出采用绝对音频路径并额外拒绝空文本，2 项契约测试通过）
- [x] 明确 237 条未逐条人工听审文本的准入与 provenance：不能伪装为人工确认；40 条使用
  review latest，其他条目保留 filename candidate 来源。（已写入
  `docs/reports/slice8-data-readiness.md`；固定 `human_review` 40 / 
  `filename_candidate_unreviewed` 237，rejected/pending fail closed）
- [x] 明确 16 条 quality review 的准入策略：仅因其全部进入正式人工 approved 范围才可保留，
  同时在 frozen manifest 中保留质量原因；未来未人工批准的 review 默认不准入。（通用规则
  已固化：pass 准入，review 仅 latest approved 准入，reject 排除）
- [x] 选择并冻结声学近重复特征、speaker embedding 模型/版本/revision、预处理和阈值校准方法；
  若需下载或新增依赖，先固定许可、哈希与缓存策略。（见
  `docs/reports/slice8-feature-selection.md`；选择本地官方 CAM++ revision
  `4e872aa8f47ee67887468e7494e36aad30908b1e`，权重严格加载及真实音频原型通过；fingerprint
  与阈值校准协议已固定）
- [x] 明确 deterministic group-aware split 策略、比例、seed 与低样本 sad 约束；相似连通分量
  不得跨 split，不用随机过采样伪造分布。（见
  `docs/reports/slice8-split-strategy.md`；80/10/10、seed `20260907`、277 条总目标
  221/28/28，sad 当前 final-primary 7 条目标 5/1/1）
- [x] 完成 `docs/adr/0008-dataset-dedup-freeze.md`，记录上述取舍、失败边界和版本变更规则。
  （2026-09-08：ADR 已 Accepted，覆盖全量准入但不夸大 review、独立相似证据、CAM++、
  group-aware split、canonical Parquet/derived JSONL、原子不可变发布和重建退出条件）
- [x] 新增严格版本化 `configs/lab/datasets/fuxuan_v1.yaml`，覆盖输入 provenance、eligibility、
  文本/声学阈值、embedding、split 和输出契约。（已增加 Pydantic strict/frozen schema；
  阈值未校准期间 `assert_freeze_ready()` fail closed，配置与 JSONL 契约测试 6/6 通过）

## P1：Catalog 与候选快照

- [x] 设计 catalog v9 表：analysis run、per-asset fingerprint/embedding、duplicate edge/group、
  speaker score/outlier、dataset version/item；所有记录绑定 config/implementation provenance。
  （见 `docs/reports/slice8-catalog-v9-design.md`；含 11 类表、键/外键/CHECK/索引、
  append-only 人工 edge/speaker 结论、事务和迁移测试要求）
- [x] 实现 v9 migration、索引、外键和 CHECK 约束，并测试旧 v8 catalog 原地升级及幂等初始化。
  （9 项 Slice 8 定向测试和 91 项全量测试通过；真实 catalog 已备份为
  `data/catalog/catalog-v8-before-slice8-20260908.sqlite` 后升至 v9，11 张 dataset 表、
  40 条既有 review decision、`PRAGMA integrity_check=ok`）
- [x] 实现冻结候选查询：绑定 verified raw、`training_audio@1` derived、latest quality、可用来源、
  latest review decision，禁止隐式读取“最新但未冻结”的其他 run。（quality 使用 report 指定
  run id，review latest 限定指定 benchmark/version；2 项隔离测试通过；真实查询返回 277 条、
  261 pass / 16 review、40 approved review、0 缺来源、277 条 48 kHz mono PCM24）
- [x] 实现最终文本/标签解析和显式 `text_source`、`label_source`、review round/batch lineage；
  rejected/pending、缺文件、哈希漂移和缺字段必须 fail closed。（3 项解析门禁测试通过；
  真实 277 条全文件哈希核验后 eligible 277 / excluded 0，文本来源 40 human + 237 filename，
  标签来源 31 confirmed + 9 corrected + 237 unreviewed，总时长 29.333364 分钟）
- [x] 冻结候选输入快照与 SHA-256；同 dataset id/version 输入或配置漂移必须在写产物前拒绝。
  （`fuxuan@1` 配置已登记，config SHA-256
  `316b45f34f21a7b4a80455bacd574ba51512753be77ad4ed717c8143cdf9a43f`；真实 277 条候选、
  0 排除、1760.0018125 秒已原子写入
  `data/reports/datasets/fuxuan_v1/audit/candidate_snapshot.json`，snapshot SHA-256
  `5fd831600c498344166af3834a5809e0bba2a7831f5bea63f498deaac1daceea`；幂等重跑返回
  `cached`，无残留 partial，配置/输入漂移与覆盖拒绝共 4 项测试通过）
- [x] 增加只读 `dataset-audit` CLI，输出候选覆盖、准入/排除原因和 provenance 报告。
  （CLI 通过 SQLite `mode=ro` 打开 catalog，不运行 migration；交叉核验已登记配置、quality
  report/run、资产集合、逐条 decision/reasons、标准化音频实文件哈希和冻结 snapshot；真实运行
  raw/standardized/quality/candidate/eligible 均为 277、excluded 0，报告写入
  `data/reports/datasets/fuxuan_v1/audit/dataset_audit.json`；21 项定向测试与 `compileall` 通过，
  幂等重跑前后 catalog SHA-256 均为
  `10003a414a70e568415af288bf25150ab23de5d9ad2267f2262fcfce27d8ea04`）

## P2：精确与近重复分析

- [x] 实现标准化音频字节哈希核验与 exact-audio 分组；不得只信 catalog 中的旧哈希。
  （分析器逐文件重新读取标准化音频字节、计算 SHA-256 并与冻结候选值比较后再分组；3 项测试
  覆盖顺序稳定、相同字节分组、哈希漂移/文件缺失/重复资产拒绝及原子报告写入；真实 277 条
  全部核验通过，unique audio SHA-256 277、exact duplicate group/item 均为 0；报告
  `data/reports/datasets/fuxuan_v1/analysis/exact_audio.json` SHA-256 为
  `451b50830d4965828d8f209d95da9a8234d80403bd8949814188da78e720273a`）
- [x] 实现 Unicode/标点可配置的文本规范化、精确文本分组与可解释 diff；空文本不得参与相似度。
  （严格使用冻结配置的 NFKC、casefold 和 Unicode P/Z/C 类别过滤；逐条记录 Unicode/casefold
  是否变化、被删除字符的索引/码点/类别/名称及 original→normalized diff；规范化为空的条目只
  报告、不进入分组。4 项测试覆盖配置语义、稳定分组、空文本隔离、坏输入拒绝及确定性写入；
  真实 277 条规范化后全部非空、unique normalized text 277、exact duplicate group/item 均为 0；
  `data/reports/datasets/fuxuan_v1/analysis/exact_text.json` SHA-256 为
  `d01846f7f40d0f5fb525b994c57ac87b19672aa4acaafeacfb303d865957a1b3`）
- [x] 实现文本近重复候选召回与对称相似度评分；用真实数据检查专名、短句和标点导致的误合并。
  （按冻结的 normalized-length ratio ≥0.8 全召回，使用方向无关的 normalized Levenshtein
  similarity；阈值仍为 `null`，本步不自动成边或删样本。4 项测试覆盖正反向同分、长度过滤、
  exact/空文本隔离、坏输入拒绝、稳定顺序与确定性写入；真实 277 条共 38,226 对，过滤
  25,871 对、评分 12,355 对，exact collision 0；≥0.8 仅 3 对，均为共享长句片段，
  高分专名差异 0，唯一 short-text 候选仅 0.142857。报告绑定 `fuxuan_domain@1`
  (`0181a81e67e4cb0707c64edf2861e7284a50764d4eca44db71dc25092a03c9be`)；
  `data/reports/datasets/fuxuan_v1/analysis/near_text.json` SHA-256 为
  `ae362457486aa16b6fc35cb955214887797003d1cd71fee410b55e3239e6e7fd`）
- [x] 实现可复现的声学 fingerprint 特征缓存和距离；覆盖轻微静音/增益/重采样变体。
  （实现相对峰值 30 dB 边缘裁剪、soxr HQ→16 kHz、-20 dBFS RMS、显式无 padding 25/10 ms
  STFT、64-bin HTK log-mel、64 帧插值、逐频带中心化、4096 维 float32-le L2 fingerprint
  和对称 cosine；identity 固定算法、配置及 numpy/scipy/soundfile/soxr 版本。内容寻址缓存命中前
  重验音频 SHA，每个 blob 由 sidecar 绑定 audio/config/fingerprint SHA，同尺寸位翻转亦拒绝。
  4 项测试覆盖增益、边缘静音、48k→44.1k 重采样、缓存复用/损坏和全静音门禁；真实 277 条
  首轮 computed 277、次轮 cached 277，blob/sidecar 各 277、distinct fingerprint 277、无 partial；
  fingerprint config SHA-256 `a6afa482ce956d7e27215431e3156605a0d4a1128eb2cea7cb58cda47c8e20e4`，
  `data/reports/datasets/fuxuan_v1/analysis/acoustic_fingerprints.json` SHA-256
  `560b7cf0860bfc29e1f7f7f9f5febba7b780c9acffafe79be41c377f42a669c8`）
- [x] 实现 pinned speaker embedding 提取、L2 归一化、缓存、中心/邻域相似度与离群评分。
  （加载前核验本地 CAM++ weights/config SHA，`SpeakerXVectorFeatures` CPU float32、完整音频、
  `strict=True`；输出 512 维 float32-le L2 embedding。内容寻址 blob+sidecar 绑定
  audio/config/embedding SHA，命中前仍重验音频；全命中时 lazy encoder 不加载模型。使用
  coordinate median→L2 鲁棒中心、top-5 邻居均值与
  `1-(center_cosine+knn_cosine)/2` 离群分数，阈值未校准时 `candidate_outlier=null`。
  5 项测试覆盖真实 pinned 模型、权重漂移、音频门禁、缓存损坏、稳定中心/KNN/排序；
  单条真实音频两次独立计算 embedding SHA 完全一致。真实 277 条首轮 computed 277、逆序
  次轮 cached 277，distinct embedding/blob/sidecar 各 277、无 partial；center cosine
  0.387058–0.945408、5-NN cosine 0.597400–0.917719。embedding config SHA-256
  `1445b409333982a6efb80717c667bf3befd741c40e6d9e60015a3c17655248ec`，
  `data/reports/datasets/fuxuan_v1/analysis/speaker_embeddings.json` SHA-256
  `c126c40c3d5aa07af3b9aa34a819a0aaa12d01f12ff29b7d4721524e72a9e499`）
- [x] 用真实 277 条分布校准文本、声学和 speaker 阈值；阈值来源写入报告，不凭单个样本拍值。
  （独立版本化 `fuxuan_dataset_thresholds@1` 绑定 dataset config、candidate snapshot、near-text、
  fingerprint 与 speaker 报告 SHA。文本使用 3 对已人工检查正例与 12,352 对背景的分离中点：
  positive min 0.842105、background max 0.617647、threshold 0.729876，候选 3 对。声学按四类
  emotion/时长分位选 16 条、生成 gain/edge-silence/resample-roundtrip 共 48 正例，与 7,077
  个满足时长比的真实背景对取分离中点：positive min 0.9999999994、background max 0.797179、
  threshold 0.898590，真实候选 0 对。speaker 使用全 277 条 median/MAD，`median -
  4.5*1.4826*MAD` 得 center 0.654570、5-NN 0.676602，待复核 6 条。4 项校准契约测试通过；
  二次运行报告字节一致。calibration config SHA-256
  `8d2b6f4003f0936f0aa9b03df3ae63baeab8b1469cf9195ec80e54089abf38c2`；
  `data/reports/datasets/fuxuan_v1/analysis/threshold_calibration_v1.json` SHA-256
  `c9703fd572e1e58f6776d94a085a4e4440917011f63ae22261c591e34bf41677`；报告补充绑定
  `dataset_id/version` 后重建，阈值及候选集合不变）
- [x] 实现 calibration overlay/promotion 契约：analysis run 显式绑定 calibration id/version/SHA，
  使已登记的 audit-only `fuxuan@1` 配置无需原地改写即可使用已校准阈值；禁止隐式读取其他报告。
  （新增严格 overlay loader，交叉核验 dataset config、candidate snapshot、calibration config/report
  SHA、报告身份与 summary/detail 阈值；基础配置继续保持 audit-only。catalog v10 新增 calibration
  注册表及 run 绑定表，未登记、身份漂移和跨 run calibration 漂移均在写 run 前拒绝。真实 catalog
  备份为 `data/catalog/catalog-v9-before-calibration-overlay-20260908.sqlite` 后升至 v10；已登记
  `fuxuan_dataset_thresholds@1`，正式 analysis run
  `9e2ff68f-b473-4e3a-acf9-b333c54c3535` 显式绑定 config SHA
  `8d2b6f4003f0936f0aa9b03df3ae63baeab8b1469cf9195ec80e54089abf38c2`、report SHA
  `c9703fd572e1e58f6776d94a085a4e4440917011f63ae22261c591e34bf41677`；12 项定向测试通过，
  `PRAGMA integrity_check=ok`）
- [x] duplicate edge 按证据类型和分数落库，再求稳定连通分量；边顺序变化不得改变 group id。
  （新增显式 SHA 校验的 `materialize_duplicate_graph.py`，统一生成 `exact_audio`、`exact_text`、
  `near_audio`、`near_text` typed/scored edge；exact 自动纳入图，near 固定为 `pending_review`，
  仅 latest accepted 人工结论可参与连通分量。group id 为排序后 member SHA-256 列表的 canonical
  JSON SHA-256，包含 singleton，边/候选顺序不影响结果；edge set 同 run 漂移整笔拒绝，人工结论
  变化可原子重算 group。真实 catalog 写入前备份为
  `data/catalog/catalog-v10-before-duplicate-graph-20260908.sqlite`；正式 run 写入 near-text 3 条、
  exact 0 条、near-audio 0 条（按时长比实际评分 7,077 对），accepted pair 0，生成 277 个
  singleton group / 277 个 member；review snapshot SHA-256
  `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`，重放返回 `cached`。
  4 项新契约测试及 132 项全量测试通过，`compileall`、catalog integrity、外键和 group id 重算
  均通过）
- [x] 生成 JSON/CSV/HTML 分析报告，列出 exact/near duplicate、speaker outlier、可回听路径和理由。
  （新增只读 `generate_dataset_analysis_report.py`，在生成前交叉核验正式 run 的 calibration
  binding、candidate/speaker/calibration SHA、group 全覆盖与 canonical group id、latest edge review
  snapshot，并重验所有展示音频的实文件 SHA；speaker 候选按 overlay 阈值重新计算并与 calibration
  报告集合核对。真实报告状态 `review_required`，含 near-text edge 3 条、speaker outlier 6 条、
  CSV 9 行；HTML 明示 pending 状态并提供 42 个回听控件（duplicate 两侧、outlier 本体及各 5 个
  最近邻），不提供会误导为已保存的按钮。产物
  `data/reports/datasets/fuxuan_v1/analysis/dataset_analysis.json` SHA-256
  `7b3342bb1116794e53ce6db41aef34240713fe4c592c7f8a2759f06ad4110857`，CSV SHA-256
  `ba5ec6bc7af8bd4237c57f879fb5ed13abdfbf8622328de45b3d0052000f3358`，HTML SHA-256
  `62376b889993e920a50626d30c2a26ab60da8cbba08a9a7ac601dffb719602c0`；确定性重跑哈希不变，
  2 项新报告测试及 134 项全量测试、`compileall`、行宽和 diff 检查通过）
- [x] 对 near duplicate 与 speaker outlier 边界样本做人工复核；自动分析只建议，不静默删样本。
  （用户确认 3 条 `near_text` edge 均为相同内容的完整/截取版本：保留
  `562645be…dea98`、`6d5e96e6…b13954`、`f773a229…ac2c5`，冻结时排除对应截短样本
  `0e73b650…bdcc7`、`230edf8e…fe5b7`、`fcb3062c…ba4c3`；speaker outlier rank 4
  `407a6e00…cca1` 因音质很差标记 `exclude_uncertain`，其余 5 个候选确认为同一 speaker。
  raw/standardized 源文件保持不可变，不做物理删除。正式 catalog 已写入 277 条 feature、277 条
  speaker assessment、3 条 accepted edge review 和 6 条 speaker review，重放全部幂等；形成
  274 个 group / 271 个 singleton，review snapshot SHA-256
  `34002c03383950819b018cc1c5fe1a5fdf8ca13243a057e5730fc836b51345df`。人工复核产物 SHA-256
  `7ec4dafa5d439612036f441c0e2f09e9b5082d3d8705a6824f974c8a5d740bc5`；更新后的分析报告状态
  `reviewed`，JSON/CSV/HTML SHA-256 分别为
  `24b0f16cb9827b2ac8d0c0f9ae4c6b5b5fa2f401cd1f97d3d0ce6c64a5361638`、
  `d8cbe0f6639aed796ecc9ad452e8d4792030ce00bd4b3675252d97caf9eab7b0`、
  `e2e1b1e00a8f9dad96d358828646ab55fc6a6e4dd2009b9a4551e56e4267848a`。远程审听页已增加
  中文“复核完成”“保留此条”“排除此条”醒目标记并更新至 GitHub commit
  `ae0b0adccd4f232e967471ebf4dccb9ce0c9c1dd`。报告 3 项定向测试及 136 项全量测试、
  `compileall`、catalog integrity/foreign key、校验和与格式检查通过）

## P3：分组划分与冻结产物

- [x] 实现以 duplicate/similarity group 为最小单位的 deterministic split；严禁同组跨 split。
  （新增 `dataset_split.py` 与只读 `plan_group_split.py`：按资产数/时长/seeded stable key
  排序完整 group，以 item count、duration、group count 的 lexicographic objective 贪心分配，
  再做最多 64 轮严格改进的 deterministic group swap；输入覆盖、canonical group id 或同组
  split 漂移均 fail closed。真实 277 条 / 274 组预演精确达到 train 221 / validation 28 /
  test 28，3 个 multi-asset group 均未跨 split；assignment SHA-256
  `95f9a5ae94a67560d8e916d056144650e754f1f2f477564d0a026db92388a7a3`，预演报告 SHA-256
  `1387c4b06df2d51db11d1efdab2da495c2bce3bf676590e3c0a84273ab727b67`，确定性重跑不变。
  该产物明确标记为 pre-stratification contract check，待后续人工排除与 emotion strata 后重算；
  3 项新定向测试及 139 项全量测试、`compileall`、行宽和 diff 检查通过）
- [x] 在 group 约束下做 emotion strata，单独报告 sad 6 条在 train/val/test 的实际分布和偏差。
  （新增全局 row/column-balanced strata target：每个 emotion 行总量与 train/validation/test
  列总量同时守恒，并以 low-resource group 覆盖、总 item、primary emotion、duration、group
  count 的 lexicographic objective 做最多 64 轮 deterministic whole-group swap。真实 277 条
  预演仍为 221/28/28 且无 group 跨 split，final `emotion_primary` 四类全部零偏差；人工校正后的
  primary sad 实为 7 条，精确分为 5/1/1，每个 eval 各 1 个独立 group。原始 weak-label sad
  才是清单所写的 6 条，实际为 5/0/1，对目标 4/1/1 的偏差为 +1/-1/0；报告保留该偏差，
  未复制或伪造样本。assignment SHA-256
  `b31c8c05927147ae3f0791abd92e642e3724c91a832b626bae7ad454a9ef896d`，预演报告 SHA-256
  `0f7068f25b17052a8967afde56057179dc08376f0585a2bbb252d505fd46ad7f`，确定性重跑不变；
  2 项新 strata 测试、5 项 split 定向测试及 141 项全量测试、`compileall`、行宽和 diff 检查通过）
- [x] 明确 duplicate group 的保留代表与排除理由；训练集不得通过复制/过采样改变 frozen 样本数。
  （新增 fail-closed `apply_reviewed_exclusions` 与 `select_reviewed_dataset.py`：每个 multi-asset
  group 必须由版本化人工结论唯一指定 representative，否则拒绝继续；保留原 catalog canonical
  group id 和完整 member lineage，同时只把代表送入最终候选。真实数据固定保留
  `562645be…dea98`、`6d5e96e6…b13954`、`f773a229…ac2c5`，排除其 3 个截短重复版本及 speaker
  rank 4 `407a6e00…cca1`，由 277 条得到 273 条唯一候选 / 273 个非空 group，excluded 4、
  oversampled 0。排除后重算 split 为 219/27/27，final primary 四类与 sad 5/1/1 均零偏差；
  selection SHA-256 `2fc65c705cc90168db38cfc0d8aed32680ea9a706df307013edd118d96eb1802`，
  selection 文件 SHA-256 `7e4ced7ddc2bba3ac68347e6e0de286f8764cf3011b856bb9dcaf192c3107a5d`，
  assignment SHA-256 `802a0906a0bc301309e4678bdfe7d7b35dbf080a7344cea222e111f4d334de37`，
  split 文件 SHA-256 `cf14bf91a25dd491ea337eee1eed42c588cfe20c798beeb4f2098c4bc2731d21`；
  7 项 selection/split 定向测试及 143 项全量测试、`compileall`、行宽和 diff 检查通过）
- [x] 实现 split 审计：精确音频、规范化文本、近重复边、来源组均不得跨 split。
  （新增 `dataset_split_audit.py` 与 SHA-pinned `audit_dataset_split.py`，逐项核验 catalog
  similarity group、标准化音频 SHA、按冻结配置规范化后的非空文本、全部 typed similarity edge，
  以及 `(source_root_key, source_path_key, raw_relative_path)` 源 provenance identity；任一泄漏均
  fail closed。真实 273 条审计结果为 273 group、exact audio duplicate 0、exact normalized text
  duplicate 0、source identity duplicate 0；catalog edge 3 条均因一个 endpoint 已按人工结论排除，
  selected-endpoint edge 0，五类 violation 全部为 0。audit SHA-256
  `9ba2836e92d4238e4cc32f0b1982801ce75ca9f0c1b6cff48c05fa51ca7c37f5`，报告 SHA-256
  `e402b7a7171e0049ee68db19ea1dc7715b6e02266fdcd62aea00a4ae170e6e2f`，重跑字节不变；
  2 项新审计测试及 145 项全量测试、`compileall`、行宽和 diff 检查通过）
- [x] 实现原子 `dataset-freeze`：先 staging 验证，再发布不可变 `datasets/fuxuan/v1/`；同版本
  同内容幂等，不同内容拒绝覆盖。（新增 `dataset_publish.py`：staging 必须与版本目录同父、名称
  受限且不可为目标本身；build 完成后先调用验证器，再生成按 POSIX 相对路径排序的 size/SHA-256
  tree inventory，目标不存在时以同卷 `os.replace` 原子发布。目标已存在且 inventory 完全一致返回
  `cached`，任一文件不同抛出 `DatasetVersionConflict` 且不触碰已发布目录；构建/验证失败只清理
  已校验属于本次目标的 staging，并拒绝 symlink。2 项新定向测试覆盖 publish/cached、内容冲突、
  validation failure 与 staging 清理；147 项全量测试、`compileall`、行宽和 diff 检查通过。
  真实 v1 将在下一项生成完整 artifact set 后通过此机制发布）
- [x] 输出 canonical Parquet（完整 lineage）、官方三字段 train/val/test JSONL、stats.json、
  manifest.json、冻结 build config 和 `checksums.txt`。（新增显式 Arrow schema 和固定 Parquet
  2.6/zstd 写入参数，`pyarrow` 固定为 25.0.1；273 条按 split/asset 稳定排序，Parquet 保留
  raw/source/derived/quality/text/label/review 全链路，训练 JSONL 严格仅含 `fid/audio/text` 且
  使用标准化音频绝对路径。真实 `datasets/fuxuan/v1/` 已原子发布 8 个文件，总计 240,927
  bytes，tree SHA-256 `77754574c4048a20682b0e8f9be95c322cf0848afedaf90d7c81b1f00df029af`；
  dataset.parquet SHA-256 `6bf5d0e29a16c62cf92087af9380bb9f62e724f964700c428c000fa8fa4c0c2e`，
  manifest artifact-set SHA-256 `9ef664c6825e53cd9d6681315b0e0ce0542d1325152524db643ff1992b8bb052`，
  checksums SHA-256 `5ce2910a47b11964b0421535508772e9e20c96afde74db0cbca60c23882dbbcf`；
  第二次真实运行返回 `cached` 且 tree hash 不变。1 项新 artifact 测试、3 项 artifact/publish
  定向测试及 148 项全量测试、`compileall` 通过。）
- [x] 验证每个 JSONL 音频路径存在且哈希匹配、文本非空、speaker 合法；Parquet/JSONL 与 catalog
  dataset_item 的资产集合和 split 完全一致。（新增发布后深度验证与不可变 catalog 登记：重新读取
  273 个绝对音频路径并逐字节计算 SHA-256，全部存在且匹配；文本均非空，唯一 speaker 为
  `崩铁符玄`。Parquet 显式 schema、273 条行顺序和三份 JSONL 的 `fid/audio/text` 全部逐行一致；
  正式 catalog 写入 1 条 `dataset_version` 和 273 条 `dataset_item`，再按全部 lineage 字段逐行
  与 Parquet 对比一致，split 为 219/27/27。speaker review snapshot SHA-256
  `9d21147cf038190f709f969bb1f652a4e25627f8cc2c9f4a01bd899e8bfde813`；登记前备份
  `catalog-v10-before-dataset-freeze-20260908.sqlite` SHA-256
  `895a188bfd47dd02bebf9e3f3b468e720897bd66757222d8eaf4442aaecea696`。重复登记返回
  `cached`；`integrity_check=ok`、foreign-key violations 0。artifact 测试新增音频内容漂移拒绝，
  148 项全量测试及 `compileall` 通过。）
- [x] 在独立临时目录重建 v1，除允许变化的运行时间外，规范化产物与 checksums 完全一致。
  （新增 `verify_dataset_rebuild.py`，从三个 SHA-pinned freeze 输入在独立临时目录重新执行 Arrow、
  JSONL、JSON、YAML 和 checksums 全构建，并在临时目录清理前完成结构验证和逐文件
  size/SHA-256 inventory 比较；不读取已发布文件作为构建输入，也没有允许忽略的运行时间字段。
  真实重建 8/8 文件完全一致，tree SHA-256
  `77754574c4048a20682b0e8f9be95c322cf0848afedaf90d7c81b1f00df029af`；报告
  `data/reports/datasets/fuxuan_v1/freeze/rebuild_verification.json` SHA-256
  `e5a235dc64c4ca3f787d0ab211a04889db3162741e9c8515d4f7424c2e736c41`。单元测试亦在两个
  独立目录生成并比较完整 inventory。）

## 测试、验收与收口

- [x] 单元测试覆盖 config、migration、候选解析、漂移拒绝、文本/声学分组、speaker outlier、
  group id、split、幂等/冲突和 checksums。（完成
  `docs/reports/slice8-unit-test-coverage.md`，逐项映射 16 个测试类与 fail-closed 边界；补充
  checksum 文件所列产物发生字节篡改时必须拒绝，以及音频源内容漂移拒绝。artifact 定向测试
  和 148 项全量测试、`compileall` 通过。）
- [x] 集成测试用合成音频覆盖 exact duplicate、增益/静音近重复、speaker outlier、相似文本组与
  low-sample strata，验证无跨 split 泄漏。（新增 `tests/test_slice8_integration.py`：9 条真实写盘
  PCM24 合成音频同时覆盖字节完全相同、0.45 倍增益、首尾静音；增益/静音 fingerprint cosine
  均 >0.99；近似中文文本得分 >0.8；独立 speaker 向量被排为 outlier rank 1。把 exact/人工接受
  near evidence 合成 canonical connected components 后进行 emotion-stratified whole-group split，
  low-resource sad 在 validation/test 各 1 条，最终 group assertion 与五类 split audit 全部通过。
  1 项端到端集成测试及 149 项全量测试、`compileall` 通过。）
- [x] 对真实 277 条运行 audit → analysis → freeze，人工复核边界报告并记录最终统计。（最终流水线
  从只读 candidate audit 开始复跑：277 candidate / 277 eligible / 0 初始排除，candidate snapshot
  SHA 不变；analysis 报告为 `reviewed`，3 条 near-text 和 6 条 speaker outlier 的 pending 均为 0，
  固定人工结论排除 3 个重复截短版本和 1 个低音质 speaker 样本。selection/split 重放哈希完全
  不变，得到 273 条、无过采样、train/validation/test 219/27/27，总时长 1738.3898958333334 秒；
  final emotion 为 train 68/51/95/5、validation 8/6/12/1、test 8/6/12/1（neutral/happy/
  angry/sad），全部命中目标。五类 split leakage violation 为 0，freeze 与 catalog 均返回
  `cached`，并再次验证 273 个音频哈希和 catalog item。）
- [x] 运行全量 tests、`compileall`、`pip check`；记录实际项数与结果。（最终执行
  `python -m unittest discover -s tests`：149/149 通过，16.871 秒；`python -m compileall -q
  src scripts tests` 通过；`uv pip check --python .venv/Scripts/python.exe` 检查 92 个包，结果
  `All installed packages are compatible`。正式 catalog `integrity_check=ok`、foreign-key
  violations 0。）
- [x] 完成 `docs/reports/slice8-dataset-freeze-acceptance.md` 和产物哈希，证明可重建与来源链。
  （验收报告集中记录用户人工边界结论、273 条最终统计、8 个冻结文件 SHA-256、五类零泄漏、
  Parquet/JSONL/catalog 对账、独立重建和 149 项测试证据，并显式披露 234 条文本仍为未逐条人工
  听审的 filename candidate。报告 SHA-256
  `2aadca02f86ed134ae48be31226daf6a6e243092320d61ed401ade1ff25985a0`，状态“通过”。）
- [x] 更新 `PLAN.md` 的 Slice 8 完成记录；只有全部退出条件完成后再进入 Slice 9。（`PLAN.md`
  已记录 277→273 的人工排除、最终 split/emotion 统计、零泄漏、冻结 tree hash、独立重建、
  catalog 对账和 149 项测试结果；新建 `docs/reports/slice9-task-list.md` 并将状态设为“已进入，
  待执行”。Slice 8 清单 41/41，正式进入 Slice 9，但尚未执行参考音频预选任务。）

## 执行顺序

1. P0 契约与 ADR。
2. P1 catalog 与候选快照。
3. P2 去重和 speaker 分析。
4. P3 分组划分与冻结。
5. 全量测试、真实数据验收、报告与 `PLAN.md` 收口。
