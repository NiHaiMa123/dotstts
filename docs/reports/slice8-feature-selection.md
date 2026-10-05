# Slice 8 声学特征与 Speaker Embedding 选择

- 状态：已完成真实数据校准并用于 `fuxuan@1` 冻结
- 日期：2026-09-07

## 选择结果

### Speaker embedding

复用 dots.tts 自身的 CAM++ `SpeakerXVectorFeatures`，不引入另一个说话人模型：

- artifact：`dots-studio/dots.tts-mf-1step`
- Hugging Face revision：`4e872aa8f47ee67887468e7494e36aad30908b1e`
- 权重：`models/dots.tts-mf-1step/speaker_encoder.safetensors`
- 权重 SHA-256：`1cf3861c9dee79e4db34bd0b8a4155e68bed27a7c6274e168bb6ee4fed191c85`
- 模型 config SHA-256：`f3388119ec80a559013092e1875d9df8cda812bd4517476b5fc744d4e05c7785`
- 许可：Apache-2.0（artifact README）
- 输入：48 kHz mono float32；encoder 内部以 16 kHz、80-bin fbank、dither 0、mean norm true
  处理
- 输出：512 维 float32，落库/比较前做 L2 normalization
- 裁剪：`max_audio_seconds=0`，使用完整音频，避免随机裁剪破坏可复现性
- 设备：CPU float32 作为 v1 canonical 实现；GPU 只能作为结果逐元素容差验证后的加速实现

真实原型已验证权重 `strict=True` 完整加载，一条 48 kHz 标准化音频输出 shape `(1, 512)`，
全部有限，原始 L2 norm 为 `23.895906448364258`。

### 声学 near-duplicate fingerprint

使用无需额外模型的确定性时频 fingerprint，目标是发现“同一录音经边缘静音、增益或重采样
变化”的近重复，不把 speaker embedding 当作内容重复证据：

1. 读取已哈希验证的 `training_audio@1`，转 mono float32。
2. 以相对峰值 30 dB 阈值去除边缘静音；全静音输入 fail closed。
3. 高质量重采样到 16 kHz，按 RMS 归一化，记录实现和 soxr 版本。
4. 计算 64-bin log-mel（25 ms window、10 ms hop）。
5. 沿时间轴确定性插值/池化为 64 帧，逐频带中心化后展平为 4096 维并 L2 normalize。
6. fingerprint blob 使用 little-endian float32；同时保存参数化特征 SHA-256。
7. 配对时同时要求时长比例门槛和 fingerprint cosine 门槛；分数只生成 review edge，不能
   直接删除资产。

exact audio 仍以重新读取文件所得 SHA-256 为唯一依据。文本 exact/near、声学 fingerprint 和
speaker embedding 是三类独立证据，报告和 catalog 不合并丢失证据类型。

## 阈值校准协议

阈值未在实现前拍定，而是按以下固定协议由真实数据和合成正例共同决定；最终参数、
分位数和输入哈希已记录在
`data/reports/datasets/fuxuan_v1/analysis/threshold_calibration_v1.json`：

- 从 277 条按时长与四类 weak emotion 确定性抽样；对每条构造增益、边缘静音和
  48k→44.1k→48k 重采样变体，形成已知 positive pairs。
- 全量 277 条互不相同资产形成 negative/background pair 分布；规范化文本完全相同的 pair
  单独报告，避免把潜在真重复当成普通负例。
- fingerprint 阈值必须覆盖全部预先声明的变换类型，并把阈值附近的真实 pair 全部送人工
  回听；任何自动排除必须同时有人工结论或 exact hash 证据。
- speaker outlier 同时使用“到鲁棒中心的 cosine”和“k-nearest-neighbor cosine”；以分布的
  median/MAD 与尾部候选生成 review 队列。单角色只有 277 条，v1 不用聚类标签自动改
  `speaker_id`。
- 校准样本、变换参数、分布分位数、候选 pair 和最终阈值写入 versioned report；阈值变化
  必须提升 analysis config/version。

## 依赖结论

CAM++、`torch`、`torchaudio`、`soundfile` 和 `safetensors` 已在项目环境中可用，不需要
`speechbrain` 或 `resemblyzer`。Slice 8 冻结时 `pyarrow` 已可用，canonical Parquet 已成功
生成并通过独立重建校验；这里不再保留“待解决”的依赖状态。
