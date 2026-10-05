# Slice 9 参考音频输入契约

- 状态：冻结
- 日期：2026-09-08
- 适用数据集：`fuxuan@1`
- 适用入口：`DotsTtsRuntime.generate()` / `generate_stream()` 与 `dots.tts` CLI
- 默认本地 checkpoint：`models/dots.tts-mf-1step/`

## 代码事实

标准 TTS 有三种参考模式：

1. `prompt_audio + prompt_text`：continuation voice cloning。README 将其列为推荐模式和最高
   speaker similarity 路径；本 Slice 的最终参考池以此模式为主。
2. 仅 `prompt_audio`：只使用 CAM++ x-vector 的音色条件，不做参考音频 latent continuation。
3. 不提供参考音频：随机 voice；只对已微调的单 speaker checkpoint 有意义，不属于本 Slice。

`runtime.py` 使用 `librosa.load(..., sr=None, mono=True)` 解码参考音频，按相对峰值 30 dB 裁剪
首尾静音，再高质量重采样到 checkpoint vocoder sample rate。当前 checkpoint 为 48 kHz；因此
运行时可以接收其他采样率和多声道，但会隐式转换。

只要 `prompt_text` 非空就必须同时提供 `prompt_audio`。prompt text 会 `strip()` 后追加换行；空白
prompt text 等价于未提供，进入 x-vector-only 路径。README 明确要求 continuation transcript 与音频
实际内容精确匹配；不匹配会降低稳定性并造成词级错误。

当前 checkpoint 的 `patch_size=4`，vocoder hop size 为
`2×2×2×4×6×10=1920` samples，因此每个 prompt patch 是 7680 samples（48 kHz 下 0.16 秒）。
默认 `max_generate_length=500` 要求 continuation prompt patch count 小于 500，即 30 dB 裁边后
最多 499 patches / 79.84 秒。这个值是运行时技术上限，不是质量建议。

CAM++ 配置的 `xvec_max_audio_seconds=10.0`。超过 10 秒时 speaker encoder 会随机截取一个 10 秒
窗口；README 同时建议参考音频保持在约 10 秒，超过该长度不会带来更好结果。为保证同一参考在
固定 seed 外也有确定的 speaker conditioning，本 Slice 不接受超过 10 秒的参考。

## Slice 9 硬门禁

以下条件任一不满足即排除，不进入后续综合评分：

| 类别 | 硬门禁 | 理由 |
| --- | --- | --- |
| 冻结身份 | 必须来自 catalog 已登记的 `fuxuan@1` `dataset_item` | 禁止回退到被排除或未冻结的 277 条输入 |
| 路径 | 使用 Parquet/JSONL 中冻结的绝对路径；规范化后仍位于冻结 standardization root | 避免相对路径、路径逃逸和误读其他文件 |
| 完整性 | 文件存在，实算 SHA-256 与 `dataset_item.audio_sha256` 相同 | 禁止静默使用漂移音频 |
| 容器/PCM | WAV、48,000 Hz、mono、PCM24 | 与冻结训练音频和当前 checkpoint 原生采样率一致，避免参考时隐式转换 |
| 数值 | 可解码、样本数大于 0、全部 finite、非全静音 | 防止解码失败、NaN/Inf 和无有效说话内容进入模型 |
| 30 dB 裁边 | 复现 runtime 的 `librosa.effects.trim(top_db=30)` 后仍有有效非静音内容 | 排名特征必须与实际送入模型的参考一致 |
| 时长 | 30 dB 裁边后的有效时长 `>0` 且 `<=10.0` 秒 | 避免 CAM++ 随机 10 秒截取；同时远低于默认 79.84 秒技术上限 |
| speaker | `speaker_id` 非空，且属于冻结数据集唯一 speaker `崩铁符玄`；不得包含 Slice 8 人工排除项 | 参考池不能重新引入错误/不确定 speaker |
| 文本 | `text_exact.strip()` 非空，SHA-256 与 catalog 一致 | continuation 必须有冻结 transcript |
| continuation | `prompt_audio` 与同一 dataset item 的完整 `text_exact` 成对输出，不允许跨条目拼配或自动改写 | 保证 transcript 与实际语音一一对应 |

`fuxuan@1` 当前 273 条 catalog 时长范围为 3.0286875–9.977833333333333 秒，因此冻结时长字段
下 273/273 均满足 10 秒上限；后续仍须以复现 30 dB 裁边后的波形重新计算有效时长。

## 不在本步提前固定的质量规则

背景噪声、尾噪、削波、响度、SNR、内部长静音、文本可信度、speaker center 相似度和音素/文本
覆盖度会影响参考质量，但其阈值尚需查看真实 273 条分布。它们在后续 Slice 9 配置与校准任务中
定义，不能在本契约核对步骤凭单条样本拍值。

同理，“约 10 秒”是最大时长与推荐区间的依据，不代表越接近 10 秒排名越高。较短且内容完整、
干净、韵律自然的参考仍可能优先。

## 验收断言

后续候选 CLI 和测试至少必须证明：

- 变更文件字节、相对路径逃逸、错误格式/采样率/声道/位深、空音频、全静音和超过 10 秒均拒绝；
- 空白 transcript、文本哈希漂移、未知 speaker 和非 `fuxuan@1` item 均拒绝；
- 输出的 continuation pair 始终来自同一 `dataset_item`；
- 同一冻结输入重复运行，门禁结果和理由字节一致。
