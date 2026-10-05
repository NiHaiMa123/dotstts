# Slice 9 候选特征定义

- 状态：定义冻结；阈值、权重和归一化方法留待后续校准
- 日期：2026-09-08
- 适用范围：冻结数据集 `fuxuan@1` 的 train 候选（只用于参考池，不读取未冻结的 277 条输入）

本文件只规定“每条候选要测什么、从哪里测、缺失如何表示”。它不在本轮决定
Top-K 阈值，也不把粗情感标签混进一个总分。所有数值都必须带单位、算法版本和
来源 run/config SHA-256，才能进入后续候选快照。

## 1. 候选行和输入优先级

每条特征行以 `dataset_item` 的以下键组成稳定身份：

`dataset_id`, `dataset_version`, `fid`, `asset_sha256`, `derived_id`, `audio_sha256`,
`text_sha256`。

输入优先级固定为：

1. 从 catalog 的 `dataset_item` 读取 train 行；`audio_relative_path` 必须解析到已验证的
   standardized artifact，重新计算得到的文件 SHA-256 必须等于 `audio_sha256`。
2. 候选音频的 canonical 数值特征在 standardized WAV 上计算：48 kHz、mono、PCM 24-bit，
   并保留 standardization `run_id`、`config_id/version`、`config_sha256` 和
   `implementation_version`。
3. dots.tts 运行时行为另行重放一次：`librosa.load(sr=None, mono=True)`，
   `librosa.effects.trim(top_db=30)`，再以项目的高质量重采样到 checkpoint 的 48 kHz。
   因此“标准化时长”和“推理有效时长”必须是两个字段，不能互相覆盖。
4. `data/reports/quality/quality.json` 是 raw source 的质量 provenance；它的数值不得在没有
   明确标记的情况下冒充 standardized 音频的数值。若要把质量特征用于 Slice 9 排名，必须
   在同一 analysis config 上对 standardized 音频重算；raw 结果只作为溯源和漂移诊断。
5. speaker 特征复用已冻结的 CAM++ report/cache，文本直接使用 dataset item 的
   `text_exact` 和 `text_sha256`。任何跨 asset、跨 split 或未记录的回退都视为输入漂移。

每条输出行建议采用以下顶层分组，并给每组一个 `status`：`ok`、`unavailable` 或
`error`。字段缺失时保留 `null` 和状态，不删除列。

## 2. 硬门禁与软特征的边界

| 类别 | 进入门禁的内容 | 作为排序/诊断的内容 |
| --- | --- | --- |
| 身份与完整性 | dataset/tree、路径边界、audio/text SHA、speaker、有限样本、非空文本 | 无 |
| 时长 | dots.tts 运行时裁边后的有效时长 `>0 && <=10s` | 时长在候选分布中的位置、裁边比例 |
| 质量分析错误 | decode、非 finite、特征计算失败 | policy 的 `review` 原因及各数值 |
| speaker | 没有与冻结报告/目标 speaker 对上的向量时拒绝 | center/kNN 相似度、outlier 分数和邻居 |
| 文本 | 文本为空、hash 不匹配或 audio/text 不成对时拒绝 | provenance 等级、可选 ASR 一致度、覆盖特征 |
| 静音/削波/响度/SNR | 全静音、无 active frame 等不可用输入拒绝 | 首尾/内部静音、silence ratio、响度、噪声代理 |
| 音素覆盖 | 只因覆盖率低不自动拒绝 | 拼音/声母/韵母/声调及脚本、标点覆盖 |

旧 quality policy 的阈值穿越仍表示 `pass/review` provenance，不自动升级成 Slice 9 的
reject；Slice 9 的新阈值必须在后续真实分布校准后写入版本化配置。

## 3. 客观质量特征

以下字段属于 `quality` 组。除 `source_decision` 和 `source_reason_codes` 外，canonical
值均指 standardized 音频上、`objective_signal_metrics@1` 算法的重算结果。raw report 的
同名值可放在 `source_metrics` 子对象中，但不能混用。

| 字段 | 定义和单位 | 解释/方向 |
| --- | --- | --- |
| `analysis_status` | 解码、有限性和全部指标是否成功 | `error` 是门禁失败；不是 0 分 |
| `sample_rate_hz`, `channels`, `frames` | 实际 standardized 文件的采样率、声道数、帧数 | 48 kHz/mono 是格式门禁 |
| `duration_seconds` | `frames / sample_rate_hz`，未按推理再次裁边 | 仅作 canonical artifact 诊断；推理门禁看 `duration.effective_seconds` |
| `sample_peak_dbfs` | `20 log10(max(abs(x)))` | 越接近 0 dBFS 越需关注；有限样本外不定义 |
| `true_peak_estimate_dbtp` | 4 倍 polyphase 插值后的最大绝对值的 dB 估计 | 这是 estimate，不声称 BS.1770 认证；越高越危险 |
| `rms_dbfs` | 全部样本 RMS 的 dBFS | 不能单独当响度或 SNR |
| `integrated_loudness_lufs` | `pyloudnorm.Meter(sample_rate)` 的 integrated loudness | 短于实现所需窗口时为 `null`，不是静音 |
| `crest_factor_db` | `sample_peak_dbfs - rms_dbfs` | 只作动态范围诊断，不直接等于音质 |
| `abs_dc_offset` | 各声道均值绝对值的最大值 | 越高越异常；不对音频静默修正后再冒充原值 |
| `noise_floor_proxy_dbfs`, `speech_level_proxy_dbfs` | 按静音阈值分出的 frame level 的 50th/90th percentile | 相对代理，不是校准的声学测量 |
| `snr_proxy_db` | `max(0, speech_level_proxy_dbfs - noise_floor_proxy_dbfs)` | 相对代理；缺一端则 `null` |
| `near_peak_sample_count`, `near_peak_sample_ratio` | `abs(x) >= -1 dBFS` 的样本数及占全部样本比例 | 削波线索，不等同 flat-top |
| `flat_top_run_count`, `max_flat_top_run_samples` | 近峰、量化容差内相邻稳定样本的连续平台数/最长长度；最短 3 samples | `flat_top_run_count > 0` 只产生 review 证据 |
| `source_decision` | raw quality report 的 `pass/review/reject` | provenance；不替代 standardized 重算 |
| `source_reason_codes` | raw policy reason code 的稳定排序列表 | 必须保留 `high_silence_ratio` 等原始理由，不能只存布尔值 |

质量实现必须记录 `objective_signal_metrics@1` 的 config SHA-256、实现版本和输入
`audio_sha256`。任何指标为 `null` 时同时记录原因（例如 `not_defined`、`analysis_error`），
禁止将 `null` 当作 0、最好或最差。

## 4. 静音特征

静音 frame 的定义固定为 `frame_length=20 ms`、`hop=10 ms`、frame RMS dBFS `<= -50 dBFS`。
与 quality 实现一致，短文件仍至少形成一个 frame，边界 frame 不重复填充。

| 字段 | 定义 |
| --- | --- |
| `leading_silence_seconds` | 第一个 active frame 起点之前的秒数；无 active frame 时等于整段时长 |
| `trailing_silence_seconds` | 最后一个 active frame 结束到文件末尾的秒数；裁剪到不小于 0 |
| `silence_ratio` | `inactive_frame_count / frame_count`，阈值为 `-50 dBFS` |
| `digital_silence_frame_ratio` | frame level `<= -119 dBFS` 的比例（实现的 `-120 dBFS` floor 加 1 dB） |
| `internal_silence_run_count` | 第一个和最后一个 active frame 之间的连续 inactive frame 段数量 |
| `max_internal_silence_seconds` | 上述内部连续段的最大时长；没有内部段为 `0` |
| `runtime_trim_leading_seconds`, `runtime_trim_trailing_seconds` | 完全按推理 `top_db=30` replay 得到的两端移除量 |

`internal_*` 是 Slice 9 新的可解释诊断，不能从 `silence_ratio` 猜回。全静音或无法
得到 active frame 是硬门禁；非全静音但静音比例高只保留为 review/排序证据。quality
的 `-50 dBFS` frame 静音与运行时的 `top_db=30` 裁边是两个不同阈值，报告中必须分别保留。

## 5. 时长特征

| 字段 | 定义 |
| --- | --- |
| `source_duration_seconds` | raw quality/catalog 记录的原始源时长，仅作 provenance |
| `standardized_duration_seconds` | derived artifact 的 `output_frames / output_sample_rate` |
| `effective_seconds` | 重放 dots.tts `mono -> top_db=30 trim -> 48 kHz resample` 后的样本数除以 48 kHz |
| `trimmed_fraction` | `effective_seconds / standardized_duration_seconds`；分母为 0 时为 `null` |
| `patch_count` | `ceil(effective_samples / (patch_size * hop_size))`，当前 checkpoint 为 `patch_size=4`、`hop_size=1920` |

参考候选的硬时长条件是 `effective_seconds > 0 && effective_seconds <= 10.0`；10 秒是
CAM++ x-vector 的随机 crop 上限，不是“越接近 10 秒越好”的排名目标。`patch_count` 只用于
解释 continuation 预算，不能替换秒数门禁。

## 6. Speaker 中心相似度特征

复用 `speaker_embeddings.json` 的 512 维 L2-normalized CAM++ 向量和 assessment，不另引
一个 speaker 模型。report/cache、model weights/config、输入 derived audio 的 SHA-256
必须逐条复核；当前 report 使用 `center_method=coordinate_median_then_l2`、`k=5`。

| 字段 | 定义 | 趋势 |
| --- | --- | --- |
| `center_cosine` | 候选向量与全体候选 coordinate-median 后 L2 center 的 cosine | 越高越接近中心 |
| `knn_cosine` | 排除自身后 top-5 邻居 cosine 的算术平均（不足时记录 effective k） | 越高越稳定 |
| `outlier_score` | `1 - (center_cosine + knn_cosine) / 2` | 越低越好 |
| `outlier_rank` | 按 `outlier_score` 降序、asset SHA 作为 tie-break 的排名 | 仅诊断，不是拒绝理由 |
| `candidate_outlier` | report 的候选标记；本 Slice 不擅自填阈值 | `null` 仍是合法值 |
| `neighbor_asset_sha256s`, `neighbor_cosines` | 产生 kNN 分数的完整邻居证据 | 必须保留，便于回听/复核 |

当前冻结 report 的 `center_cosine`/`knn_cosine` threshold 为 `null`；Slice 9 不在本定义
阶段发明阈值。向量缺失、维度/finite/L2 校验失败或 speaker 不唯一时拒绝，不以中心值
插补。

## 7. 转写可信度与文本 provenance

文本特征归入 `text` 组，音频和文本必须是同一 dataset item。`text_exact` 先按冻结时的
规则取值（人工 review 的 `review_text_final`，否则 filename candidate），只做 `strip`，
并对 UTF-8 原文重算 `text_sha256`。

| 字段 | 定义 |
| --- | --- |
| `text_source` | 只允许 `human_review` 或 `filename_candidate_unreviewed` |
| `confidence_tier` | `A` = approved human review；`B` = filename candidate、未人工确认；无文本不发 tier |
| `label_source` | 保留 `weak_label_confirmed`、`human_corrected` 或 `weak_label_unreviewed` |
| `review_decision_id`, `review_round`, `review_batch_id` | human review 的 append-only 证据；B tier 必须为 `null` |
| `text_sha256` | `text_exact.encode('utf-8')` 的 SHA-256；与 catalog 不一致即拒绝 |
| `asr_backend`, `asr_report_id`, `asr_cer`, `asr_edit_distance` | 可选 benchmark agreement；没有跑 ASR 时为 `null`，表示 unknown，不表示 CER=0 |
| `text_char_count`, `han_char_count`, `punctuation_ratio` | 对 coverage 规范化视图计算的基础计数；保留 exact 文本供展示 |

ASR agreement 只能作为后续 soft evidence，不能把自动转写结果改写成
`human_review`，也不能在缺失时给满分。批准的文本若被后续人工编辑，仍以新的
`review_decision_id/round/batch` 追踪，不覆盖旧 provenance。

## 8. 音素/拼音覆盖特征

模型本身吃 BPE 文本，不把音素作为模型输入；这里的音素特征只用于参考音频的文本/发音
多样性诊断和后续 diversity rerank。实现必须在后续配置中钉死 G2P/拼音库、版本、词典
和多音字策略；当前环境未预装 `pypinyin`、`jieba` 或 phonemizer，因此本轮不伪造数值。

对每条非空文本生成一个 versioned `coverage_view`：

| 字段 | 定义 |
| --- | --- |
| `script_char_count` | 去除 whitespace 后的 Unicode code-point 数；标点另计，不从 exact 文本删除 |
| `han_char_count` | 上述字符中 CJK Han 数量 |
| `latin_digit_ratio` | Latin 字母和 ASCII/Unicode 数字数除以 `script_char_count` |
| `punctuation_ratio` | Unicode punctuation 数除以 `script_char_count` |
| `pinyin_syllable_count` | 每个可解析 Han 字符产生一个 canonical `(initial, final, tone)` 三元组的数量 |
| `unique_syllable_type_count` | 去重后的完整 pinyin 三元组数量 |
| `initial_type_count`, `final_type_count`, `tone_type_count` | 去重后的声母、韵母、声调类型数 |
| `unresolved_char_count` | 词典/多音字策略无法确定的字符数；不当作“没有音素” |
| `coverage_status` | `ok`、`unavailable_dependency`、`unresolved` 或 `unsupported_script` |

`pool_inventory_*`（候选池中出现的 syllable/initial/final/tone 类型）和
`selected_novelty_*`（相对已选 Top-K 的新增类型）是池级/迭代级派生量，不能写进单条
音频的静态特征而假装与选择顺序无关。覆盖率低、脚本混合或有 unresolved 字符只影响
后续分数并触发人工关注；只有空文本、hash 不匹配或 coverage 实现错误才走相应硬门禁。

## 9. 缺失、错误和评分输入规则

1. **硬完整性缺失**：身份、路径、audio/text hash、目标 speaker、非空/finite 音频、有效
   `effective_seconds` 缺失或不一致，直接 `reject`，并写稳定 reason code。
2. **可选特征缺失**：ASR、integrated loudness（算法未定义）、音素依赖未安装等写
   `null`，组状态为 `unavailable`，记录 `unavailable_reason`；不填 0、不填中位数、不
   通过排序把缺失项当作最佳/最差。
3. **预期分析失败**：已具备依赖但发生 decode、非 finite、维度、SHA 或算法错误时为
   `error`；质量/ speaker 主特征的 error 不能进入候选排名。
4. 后续综合分数只对当前存在的分项归一化，并同时输出 `available_weight`、总权重和每项
   原始值。阈值、权重、clip/winsorize 和 tie-break 必须在 Slice 9 配置/ADR 中另行冻结。
5. 每条特征行还要保存 `feature_schema_version`、各组实现版本、输入 artifact/report
   SHA-256 和生成 `run_id`，保证同一 dataset/tree 下的内容变化能被漂移拒绝。

## 10. 当前可复核的基线身份

本定义引用的已有 provenance（不是本轮新阈值）如下：

- standardization：`training_audio@1`，run `6c1b4a80-198d-45c2-ab24-617ffb662fe1`，
  config SHA-256 `a2a2c7d00ec638dc3e420692785c604b86d0997202dd35abdcff17c8abdda4d6`，
  输出 48 kHz mono PCM24。
- raw quality：`objective_signal_metrics@1` / `training_source_review@1`，run
  `ce39a56f-009f-45b7-8a4c-8600355b71b7`；其数值是 raw source provenance，不能替代
  standardized 重算。
- speaker：report `data/reports/datasets/fuxuan_v1/analysis/speaker_embeddings.json`，
  config identity SHA-256 `1445b409333982a6efb80717c667bf3befd741c40e6d9e60015a3c17655248ec`，
  center SHA-256 `23a792f6051f50d689786304cf6dabd19cb2a66e7af18696f099686e14e28956`，
  512 维、effective k=5、thresholds 尚为 `null`。

