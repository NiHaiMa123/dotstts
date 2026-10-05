# Slice 9 参考音频预选任务清单

- 状态：已完成（29/29）
- 创建：2026-09-08
- 前置：Slice 8 已通过；冻结数据集 `fuxuan@1` 共 273 条，tree SHA-256
  `77754574c4048a20682b0e8f9be95c322cf0848afedaf90d7c81b1f00df029af`
- 目标：分别输出可解释、可回听且保持内容/韵律多样性的中性主参考池和各情感参考池

## P0：契约与输入冻结

- [x] 核对 dots.tts 推理参考音频的格式、时长、文本和路径契约，明确硬门禁。（完成
  `docs/reports/slice9-reference-contract.md`：代码确认 continuation、x-vector-only 和无参考三种
  模式；运行时 mono decode、30 dB 裁边和重采样到 48 kHz。当前 checkpoint 每 patch 7680
  samples，默认 continuation 技术上限为 499 patches / 79.84 秒；CAM++ 超过 10 秒会随机截取，
  因此 Slice 9 固定有效时长 `>0 && <=10s`。硬门禁还包括仅 `fuxuan@1`、绝对路径不逃逸、
  实算音频/文本哈希、WAV 48 kHz mono PCM24、finite/非静音、唯一 speaker，以及同一 item 的
  audio/text 成对输出。冻结 273 条 catalog 时长为 3.0286875–9.977833333333333 秒，当前全部
  在 10 秒内；噪声、响度、SNR、内部静音和覆盖阈值留待真实分布校准。）
- [x] 定义中性主参考池与 neutral/happy/angry/sad 情感池的用途、规模和互斥/重叠规则。（完成
  `docs/reports/slice9-pool-contract.md`：为保持 validation/test 独立，参考只从 train 219 条中选；
  硬门禁前 neutral/happy/angry/sad 为 68/51/95/5。固定 `main_neutral` Top-10（最低 6、边界
  5）以及四个 emotion pool Top-6/6/6/5（最低 5/5/5/3，前三池边界各 3）；不足时不复制、
  不跨情感回填。四个 emotion pool 资产互斥；main-neutral 与 emotion-neutral 最多重叠 3 条，
  每次人工排除回填后重验。五池共 33 个名额、至少 30 个不同 asset；情感分池只组织参考，
  不宣称 basic pipeline 已具备显式情感控制。）
- [x] 明确客观质量、speaker 中心相似度、转写可信度、静音、时长和音素覆盖的特征定义。（完成
  `docs/reports/slice9-feature-definitions.md`：固定 standardized 48 kHz mono PCM24 为 canonical
  特征输入，并把 raw quality 仅作为 provenance；定义质量/静音/运行时裁边/时长、CAM++ 中心与
  kNN、文本 provenance/ASR 可选证据、拼音/音素覆盖字段及 `ok/unavailable/error` 缺失规则。
  有效时长硬门禁固定为运行时 `top_db=30` replay 后 `>0 && <=10s`；阈值、权重和 G2P 依赖留待
  后续版本化配置与真实分布校准。）
- [x] 明确粗情感标签只用于分池，不能把不同情感混成一个总排名。（完成
  `docs/reports/slice9-pool-contract.md` 的“标签作用域与排序隔离”：`emotion_weak_label` 和
  `emotion_primary` 仅路由 `pool_id`，不进入质量/说话人/文本/覆盖分数；每个 pool 独立门禁、
  归一化、综合分和 diversity rerank，排名键固定为 `(pool_id, rank)`，禁止混合集合先排
  global Top-K、跨情感回填或生成总榜；`main_neutral` 与 `emotion_neutral` 也保持独立命名空间。）
- [x] 新增严格版本化 Slice 9 配置和 ADR，固定输入 dataset id/version/tree hash、阈值和权重。（完成
  `configs/lab/datasets/fuxuan_slice9_v1.yaml` 与 `docs/adr/0009-slice9-reference-selection.md`：
  绑定 `fuxuan@1` tree SHA-256、catalog train-only/目标 speaker、standardized 与 runtime 契约，
  固定现有 quality policy 和 speaker calibration 阈值、5/95% pool-local 归一化及总和为 1 的
  分项权重；禁止 global rank、跨情感回填和缺失值当 0。YAML 已在项目环境解析，权重和为 1.0，
  配置文件 SHA-256 为 `e36faefd1a9705772c7f86ac5fb0b32e0b12982b19bbc4e5eafa1ffb431f1ff5`。）

## P1：候选与特征

- [x] 从 catalog 的 `fuxuan@1` dataset_item 读取候选，禁止回退到未冻结的 277 条输入。（完成
  新增 `src/dots_tts_lab/slice9_candidates.py` 与 `scripts/build_slice9_candidate_snapshot.py`：
  校验 catalog `dataset_version.artifact_set_sha256` 与 Slice9 配置 tree SHA、只查
  `dataset_item.split=train`、目标 speaker、derived/raw/text 哈希和 48 kHz mono PCM24，
  旧的 277 条 `candidate_snapshot.json` 不作为输入。已生成
  `data/reports/datasets/fuxuan_v1/audit/slice9_candidate_snapshot.json`，219 条、总时长
  `1390.7067083333334s`，snapshot SHA-256 为
  `d321d0796c193857cda55be459e7f095cfa6225706b6abc512ba96c75dee1ab3`；审计报告为
  `slice9_candidate_audit.json`。）
- [x] 复用并核验 Slice 5/8 的质量、静音、speaker embedding 与文本 provenance，缺失或漂移时拒绝。（完成
  新增 `src/dots_tts_lab/slice9_provenance.py` 与 `scripts/verify_slice9_provenance.py`：重放
  catalog train 快照，核对 standardization/integrity、raw quality report 的 run/config/SHA、
  219 个 standardized artifact 文件 SHA、quality 静音/质量字段、277 维 CAM++ cache/report
  和 219 个候选的 center/kNN assessment，以及 human/filename 文本 provenance 不变量。已生成
  `data/reports/datasets/fuxuan_v1/audit/slice9_provenance_audit.json`，status=succeeded，
  `candidate_files_sha256_verified=219`、`candidate_metrics_verified=219`、
  `candidate_vectors_verified=219`。）
- [x] 实现参考时长、首尾/内部静音、削波、响度、SNR/噪声等门禁并输出逐条理由。（完成
  新增 `src/dots_tts_lab/slice9_features.py` 与 `scripts/build_slice9_quality_features.py`：
  在 standardized 音频上重算 objective signal metrics，严格 replay `top_db=30` runtime trim，
  输出 20/10 ms、-50 dBFS 的首尾/内部静音、digital silence、4x true-peak、flat-top、响度、
  SNR/noise、有效时长与 patch count，并按配置写入逐条 `pass/review/reject` reason。已处理
  219 条候选：216 pass、3 review（均为 `high_silence_ratio`），报告
  `data/reports/datasets/fuxuan_v1/analysis/slice9_quality_features_v1.json`，SHA-256
  `321de69b041d2a8c22e32570244d3ffaedb2bd52c9c809a4ce0a79250c5840cf`。）
- [x] 实现 speaker center/邻域相似度特征，排除已人工判定的低质量或非目标 speaker。（完成
  新增 `src/dots_tts_lab/slice9_speaker.py` 与 `scripts/build_slice9_speaker_features.py`：
  复核 speaker report/threshold calibration SHA，加载 512D L2 CAM++ cache，输出 center cosine、
  effective-k=5 kNN、outlier score/rank、邻居证据；从 catalog 读取 latest speaker review，
  `exclude_wrong_speaker`/`exclude_uncertain` 硬拒绝，`confirmed_same_speaker` 对阈值 review
  仅作人工覆盖并保留理由。219 条 train 均为 pass；2 条 center 低于阈值但已有 confirmed
  review。报告 `slice9_speaker_features_v1.json`，SHA-256
  `9de53a58e91b9fff3f5de6c3af8b488b6a92b0bb922b2ddd12367a1550f26f12`。）
- [x] 实现转写可信度分层；未人工听审文本不得伪装为人工确认。（完成
  新增 `src/dots_tts_lab/slice9_text.py` 与 `scripts/build_slice9_text_features.py`：对 219 条
  train 候选校验文本 SHA/来源和 review lineage，人工确认文本为 tier A（29 条），仅文件名推断
  文本为 tier B（190 条）；未提供匹配的 ASR 报告时 219 条均明确标记 `unknown`，不会提升为
  人工确认。报告 `data/reports/datasets/fuxuan_v1/analysis/slice9_text_features_v1.json`，
  candidate snapshot SHA-256 为
  `d321d0796c193857cda55be459e7f095cfa6225706b6abc512ba96c75dee1ab3`，报告 SHA-256 为
  `84167d862d711fbe7bb3cce002a20c0879483af4181df90cabd5aad2d0491c31`。）
- [x] 实现中文文本字符/拼音/声母韵母与标点韵律覆盖特征，记录算法和依赖版本。（完成
  新增 `src/dots_tts_lab/slice9_coverage.py` 与 `scripts/build_slice9_coverage_features.py`：对
  NFKC 文本计算 script/Han/Latin-digit、标点和 major/minor/terminal prosody 边界特征；拼音
  使用固定 `pypinyin` 0.53.0、`tone3`、`heteronym=false`、`strict=false` 策略，未解析字符
  显式计数。当前环境未安装该可选依赖，因此 219 条均为
  `coverage_status=unavailable_dependency`，拼音数值保持 null（不伪造为 0），报告记录算法、
  依赖和文本报告 SHA：`data/reports/datasets/fuxuan_v1/analysis/slice9_coverage_features_v1.json`，
  报告 SHA-256 为 `2f481311ea4db8378fde355cb0d8877e6f26de799382c1e37086ac7d525afa32`。）
- [x] 冻结候选特征快照和 SHA-256，同版本输入漂移拒绝。（完成
  新增 `src/dots_tts_lab/slice9_feature_snapshot.py` 与
  `scripts/freeze_slice9_feature_snapshot.py`：合并 quality/speaker/text/coverage 四份逐条
  特征，校验候选集合、audio/text SHA、dataset tree 和各报告的 candidate snapshot SHA；冻结
  `freeze_id=84ef9d120c11ffc74f6ad36ac1254c75`。重复运行返回 `reused=true` 且字节一致，输入
  hash 或内容变化会拒绝覆盖。快照
  `data/reports/datasets/fuxuan_v1/analysis/slice9_feature_snapshot_v1.json` SHA-256 为
  `67cf79b4c83c91018f749163b20e06caedc350c4c3dc5b40b19b8026f3bb06a9`。）

## P2：分池、排名与多样性

- [x] 分别建立 neutral 主参考候选和四类 emotion 候选池；报告低样本池的实际规模。（完成
  新增 `src/dots_tts_lab/slice9_pools.py` 与 `scripts/build_slice9_pool_candidates.py`：从冻结
  feature snapshot 先执行 quality/speaker hard gate（216 pass、3 review、0 reject），再按
  `emotion_primary` 建立互斥 emotion 池和独立的两个 neutral 命名空间；禁止 global rank、
  跨池回填和情感池重叠。实际规模为 `main_neutral=68`、`emotion_neutral=68`、
  `emotion_happy=51`、`emotion_angry=95`、`emotion_sad=5`，均达到最低发布数，sad 是低样本池
  但没有短缺。报告 `data/reports/datasets/fuxuan_v1/analysis/slice9_pool_candidates_v1.json`，
  SHA-256 为 `2cfcd959fa6da600fccc822509173269eafb557a32a53e5d280fa742a7822e02`。）
- [x] 先应用硬门禁，再在池内归一化质量、speaker、可信度、静音和覆盖分数。（完成
  新增 `src/dots_tts_lab/slice9_normalization.py` 与
  `scripts/build_slice9_normalized_features.py`：只接收 pool hard-gate 后候选，在每个 pool
  独立执行配置的 5/95% quantile clip；质量、speaker、文本可信度、静音、时长分别有可审计的
  raw transform 和方向，pinyin 缺失保持 null 并记录 `missing_count`，绝不当作 0。五池规模
  `68/68/51/95/5`，覆盖项分别缺失该池全部条目（可选依赖未安装），其余五项均完整。报告
  `data/reports/datasets/fuxuan_v1/analysis/slice9_normalized_features_v1.json`，SHA-256 为
  `f6a6a4f2cb75d17348a40b14ef89ae9e9a08fe4134c0482a9ed605eaeee71181`。）
- [x] 实现可解释的版本化综合分数，保留每个分项、阈值和排除理由。（完成
  新增 `src/dots_tts_lab/slice9_scoring.py` 与 `scripts/build_slice9_ranking.py`：在 pool 内按
  固定权重计算 `slice9_composite@1`，保存 normalized component、原始分项、权重、逐项贡献、
  `available_weight`、缺失项和 gate review 理由；pinyin 缺失使每条 `available_weight=0.85`
  （不把 0.15 当作低分）。排名键为 pool-local score desc + asset SHA，禁止 global rank。
  报告 `data/reports/datasets/fuxuan_v1/analysis/slice9_ranking_v1.json`，SHA-256 为
  `793ac17c94ee6d1581a30cd92d7811edc6b805c3642d3563e06347934739a651`。）
- [x] 实现 deterministic diversity rerank，避免 Top-K 被相似文本、相似时长或相似韵律占满。（完成
  新增 `src/dots_tts_lab/slice9_diversity.py` 与 `scripts/build_slice9_diversity_rerank.py`：各池
  独立使用固定 tie-break 的 greedy rerank，按 NFKC 去标点文本相似度、有效时长距离和标点/韵律
  向量相似度施加版本化 penalty；不做 global rank 或跨池回填。已生成主池 Top-10、neutral/happy/
  angry Top-6、sad Top-5 及边界样本。报告
  `data/reports/datasets/fuxuan_v1/analysis/slice9_diversity_rerank_v1.json`，SHA-256 为
  `932da100a9d0cfa3c5891dad28bc63e5df61d2a7a900fd7dc3889bb0db064f4b`。）
- [x] 对相似组/同源内容设置代表约束，Top-K 内不保留无意义重复。（完成
  新增 `src/dots_tts_lab/slice9_constraints.py` 与
  `scripts/build_slice9_constrained_selection.py`：读取 Slice 8 已接受的 near-text review，
  在本轮 train 候选内构建 union-find duplicate group，优先保留 review 指定 representative；
  本轮 219 条 train 与已接受的 3 个跨 split 重复组没有交集（因此 candidate 内
  `reviewed_duplicate_group_count=0`，没有把未确认的近似组静默合并）。同时执行
  `main_neutral`/`emotion_neutral` 最大重叠 3 条约束，实际重叠 3；五池均满足 Top-K。
  报告 `data/reports/datasets/fuxuan_v1/analysis/slice9_constrained_selection_v1.json`，
  SHA-256 为 `e228e2058f5bb5495278b8a6c955548236674788740829ba81baf84a4ff96d78`。）
- [x] 做权重和阈值敏感性分析，确认小幅变化不会让 Top-K 完全翻转。（完成
  新增 `src/dots_tts_lab/slice9_sensitivity.py` 与 `scripts/build_slice9_sensitivity.py`：对四种
  权重 `+0.02/-0.01` 变体和质量/speaker 阈值 ±5%（true-peak ±0.025 dB、speaker ±0.001）
  重算 pool-local Top-K。最小 Top-K Jaccard 为 `0.7143`，所有变体均有非零重叠，未出现完全
  翻转；阈值扰动只改变 advisory review 计数（hard reject 始终 0），不会静默改变候选集。
  报告 `data/reports/datasets/fuxuan_v1/analysis/slice9_sensitivity_v1.json`，SHA-256 为
  `10285a91b801849cfa4161a1273e91014f3bf426ae04012824beaa6551cecf86`。）

## P3：报告、审核与冻结

- [x] 输出 JSON/CSV/HTML 报告，按池展示排名、分项、理由、文本、波形和可回听音频。（完成
  新增 `src/dots_tts_lab/slice9_reports.py` 与 `scripts/build_slice9_reports.py`：为五个池输出
  selected Top-K + boundary 共 47 行（selected 33、boundary 14），JSON 保存分项/理由/文本/哈希，
  CSV 便于筛选，HTML 为每行内嵌 SVG 波形和 `<audio controls>`，47/47 音频路径存在。报告文件：
  `data/reports/datasets/fuxuan_v1/analysis/slice9_review_report_v1.json`
  （SHA-256 `781fd67d051a8e7f22af43ac3cc7e56d510ff6adf090ba64b9eb3f9e08bf37ba`）、
  `data/reports/datasets/fuxuan_v1/analysis/slice9_candidates_v1.csv`
  （SHA-256 `267179ce4b9ebf67753195d00381d098a1abcaf336dfa2e37ff3ab414952705d`）和
  `data/reports/datasets/fuxuan_v1/analysis/slice9_candidates_v1.html`
  （SHA-256 `077f61cefb2bb1009dad144d602f5495cb8e34fea2c4f4b305967ecc92c1ea08`）。）
- [x] 对每个池的 Top-K 和门禁边界样本进行人工听审；自动分数只建议，不静默定稿。（完成
  用户确认当前报告内 47 条 selected/boundary 均“没问题”，按 approved 进入后续流程；这只是
  当前 review batch 的人工确认，不改变自动分数和历史报告。）
- [x] 把人工结论 append-only 写入 catalog，保留 round/batch/provenance 并支持幂等重放。（完成
  新增 `Catalog.record_slice9_reference_reviews`、`Catalog.load_slice9_reference_reviews`、
  `src/dots_tts_lab/slice9_review.py` 和 `scripts/apply_slice9_reference_reviews.py`；为当前
  用户确认的 47 条写入专用 `slice9_reference_review` append-only 表，统一
  `review_batch_id=slice9-reference-review-v1`、`review_round=1`、approved 状态，并保存报告/
  audio/text SHA 与人工确认 provenance。首次写入 47 条；同一命令重放为
  `inserted=0, ignored_idempotent_replay=47`，catalog 核验为 47 approved 行、38 个不同 asset、
  5 个 pool。导入审计
  `data/reports/datasets/fuxuan_v1/audit/slice9_reference_review_import_v1.json`，SHA-256 为
  `dc1cdba6077630774364c4a5b2b06d5de0818243e3c55eca1d25fbb6c1b8db3f`。）
- [x] 冻结最终参考池 manifest、stats、config 和 checksums；同版本不同内容拒绝覆盖。（完成
  新增 `src/dots_tts_lab/slice9_freeze.py` 与 `scripts/freeze_slice9_reference_pools.py`，冻结
  `data/references/fuxuan/v1/`：manifest 33 条、30 个不同 asset，main/emotion-neutral 重叠
  3 条；同时生成 `manifest.json`、`stats.json`、`selection_config.json`、`checksums.txt`。
  第二次运行逐文件字节校验通过；同版本输入或内容漂移会拒绝覆盖。审计结果
  `data/reports/datasets/fuxuan_v1/audit/slice9_reference_freeze_v1.json`，manifest SHA-256
  `1a33c738c1b807337b97eaed278aee05c1458aebc1f4d16da32ba7dee879c085`，stats SHA-256
  `e706bada599dd786baab2b4507b8ed5f0a842af3248ae116ff8d62c10683f32d`，checksums SHA-256
  `85bbb83092020a56b40e22d805a9d25c218270853a551252f1afc2ee2d48afe0`。）
- [x] 验证每条参考音频路径、哈希、文本、speaker、pool/rank 与 catalog 完全一致。（完成
  新增 `src/dots_tts_lab/slice9_verify.py` 与 `scripts/verify_slice9_reference_pools.py`：逐条核对
  33 个 manifest item 的 train split、catalog audio/text SHA、标准化 WAV 文件字节哈希、目标
  speaker、情感池归属、连续 rank、approved review 和 checksums；同时核对 emotion 池互斥及
  neutral 重叠上限。结果 `verified_item_count=33`、`unique_asset_count=30`、5 池、neutral
  overlap=3、errors=[]。审计
  `data/reports/datasets/fuxuan_v1/audit/slice9_reference_verify_v1.json`，SHA-256 为
  `fe4f4ec34fdbaffeec4c766380125d325c0234f7816b334c7f35bd8cb3c8b781`。）
- [x] 在独立临时目录重建并验证规范化产物字节一致。（完成
  新增 `src/dots_tts_lab/slice9_rebuild.py` 与 `scripts/rebuild_slice9_reference_artifacts.py`：
  在独立临时目录重建 `manifest.json`、`stats.json`、`selection_config.json`、`checksums.txt`，
  与冻结目录逐文件比较字节和 SHA-256，4/4 完全一致。审计
  `data/reports/datasets/fuxuan_v1/audit/slice9_reference_rebuild_v1.json`，SHA-256 为
  `28c003d41ef83929bbffc8ea1e812c149740c00be94d615945bd3c3b9f765aa1`。）

## 测试与验收

- [x] 单元测试覆盖配置、硬门禁、特征、归一化、评分、分池、多样性、幂等与漂移拒绝。（完成
  新增 `tests/test_slice9.py`，覆盖配置/权重约束、缺失值归一化、文本相似度、真实报告链、
  catalog review 幂等追加和冻结漂移拒绝，共 7 项；项目环境无 pytest，因此用标准库
  `unittest -v tests/test_slice9.py` 运行，7/7 passed。）
- [x] 合成集成测试覆盖低质量、静音、speaker outlier、重复内容和低样本情感池。（完成
  新增 `tests/test_slice9_synthetic.py`：合成 quality reject、high-silence review、speaker
  reject、已接受重复组 representative 和 sad 低样本 shortfall 场景；与 Slice9 单元测试合计
  9/9 passed。）
- [x] 对真实 273 条运行候选 → 分池 → 排名 → 报告 → 人工审核 → 冻结。（完成
  新增 `src/dots_tts_lab/slice9_acceptance.py` 与 `scripts/audit_slice9_pipeline.py`：对真实冻结
  `fuxuan@1` 全部 273 条做端到端链路审计，split 为 train=219、validation=27、test=27；按
  契约仅 train 219 条进入候选，其余 54 条明确排除以保持评测隔离。候选→分池→排名→多样性→
  人工 approved→冻结→校验→独立重建全部通过，最终 33 条/30 个不同 asset/5 池。审计
  `data/reports/datasets/fuxuan_v1/audit/slice9_pipeline_acceptance_v1.json`，SHA-256 为
  `fa58d4bb49ccd6658d57ed1dc42dace3189283a6516266a4b0c8f77ded399a98`。）
- [x] 运行全量 tests、`compileall`、`pip check` 并记录结果。（完成
  新增 `docs/reports/slice9-test-results.md`：全量 `unittest` 158/158 passed，
  `compileall -q src scripts tests` 退出码 0，`pip check` 为 `No broken requirements found.`；
  环境未安装 pytest，Slice9 新增测试已用 unittest 单独通过。）
- [x] 完成 Slice 9 验收报告；Top-K 必须可解释、可回听，同类候选保留内容和韵律多样性。（完成
  `docs/reports/slice9-acceptance.md`：记录 273 条真实数据的 train-only 范围、5 池 33/30 结果、
  47 条人工 approved、冻结 manifest/stats/checksums 哈希、catalog/独立重建核验和 158/158
  全量测试；Slice 9 已完成，下一步为 Slice 10 模型闭环评测。）
