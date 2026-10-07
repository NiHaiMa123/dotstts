# Devin Task — 锁暝「磨砂 / 声码器颗粒感」根因隔离诊断

## 目的

当前锁暝 `suoming_step500_v1` 已经缓解明显沙哑，但生成音仍存在一听就能察觉的「磨砂、塑料、声码器颗粒感」；原声本身明显更清澈。

本任务**只做根因隔离**，不要继续通过叠 EQ、exciter、reverb、spectral stabilizer 等方式“调听感”。

最终必须回答：

1. 磨砂是否已经存在于 **SOAR + LoRA 的 raw waveform**；
2. 提高 ODE/NFE steps 是否能显著改善；
3. BF16 / optimize 编译路径是否引入或放大该问题；
4. 问题主要来自 LoRA，还是官方 SOAR / vocoder 本身；
5. 当前 v11 后处理究竟是在改善，还是在放大 raw 的高频颗粒。

---

## 当前基线

仓库：`NiHaiMa123/dotstts`

当前锁暝 profile：

- `configs/voices/suoming_step500_v1.yaml`
- base：`pretrained_models/dots.tts-soar`
- adapter：`data/work/suoming/suoming_lora_v1/checkpoint-00000500/model`
- Euler 16 steps
- guidance 1.2
- speaker scale 1.5
- BF16
- optimize=true
- voice polish：`configs/lab/postprocess/suoming_voice_polish_v11.yaml`

当前 v11 中包含：

- 100 Hz low cut
- -2 dB @ 340 Hz
- +1.5 dB @ 3.2 kHz
- +2 dB above 6 kHz
- parallel compressor
- de-esser
- 220 ms synthetic ambience
- spectral stabilizer：400–9000 Hz / 7 frames / strength 0.4
- prompt loudness calibration

**不要假设这些处理是正确的。**

---

## 固定诊断文本

使用数据集 validation 中这条真实原声：

```text
午后凉风拂过，雨云渐聚，细雨敲在屋檐上。我坐在窗边听雨，拭去玄朱锁上的薄尘。这确实是有些凄迷的场景，但……我很喜欢。
```

validation 条目：

- fid：`bf38867d624e76e12d06097c1abb4c29dfd0f3e8353b045c31f95df1ae72a045`
- standardized WAV：
  `data/work/standardized/fb/fb8e3c080298ab2d0a404789e6973d68cf5af7bfadb8486851fae677f1aac1ca.wav`

这条样本位于 validation，不在 train，因此适合作为对照。

生成时固定使用 `suoming_step500_v1.yaml` 当前 prompt：

`data/inbox/锁暝/中立_neutral/【中立_neutral】谛天鉴除了处理岁主事务，也掌管天文、历法。各州分鉴各有职责，梦州也有自己——唔……先前御者说对谛天鉴了解不多，不知不觉话多了些。总之，谛天鉴为岁主尽责，那自然，.wav`

以及该 profile 中完整 prompt text。

所有可比较 case 固定：

- seed = 42
- language = chinese
- template = tts
- guidance_scale = 1.2
- speaker_scale = 1.5
- normalize_text = false
- 相同目标文本
- 相同 prompt audio/text

除指定变量外，其他参数不得变化。

---

## 必做实验

新建独立诊断脚本，例如：

`scripts/diagnose_suoming_grit.py`

输出目录：

`outputs/diagnostics/suoming_grit_v1/`

### REF — 原声

复制或引用 validation WAV：

`00_reference_validation.wav`

不要做 voice-polish。

### Case 1 — LoRA / 16 steps / BF16 / optimize=true / RAW

输出：

`01_lora16_bf16_opt_raw.wav`

条件：

- 当前 Step500 LoRA
- BF16
- optimize=true
- Euler 16
- **禁止 voice-polish**
- 只允许必要的 edge trim；最好同时保留完全 raw 文件
- 不做响度归一化，不做 EQ，不做 de-esser，不做 ambience，不做 stabilizer

这是最重要的 baseline。

### Case 2 — LoRA / 32 steps / BF16 / optimize=true / RAW

输出：

`02_lora32_bf16_opt_raw.wav`

除 `num_steps=32` 外，与 Case 1 完全一致。

目的：判断 flow sampling 精度提高能否降低颗粒感。

### Case 3 — LoRA / 32 steps / FP32 / optimize=false / RAW

输出：

`03_lora32_fp32_eager_raw.wav`

条件：

- Step500 LoRA
- precision=float32
- optimize=false
- Euler 32
- 无 voice-polish

目的：隔离 BF16 / compile / optimized inference 数值路径。

如果 FP32 因显存或当前实现无法运行：

- 不得静默退回 BF16；
- 在报告中明确记录 BLOCKED；
- 可追加 `FP16 eager` 或 `BF16 eager` 作为辅助 case，但不能伪装成 FP32。

### Case 4 — 官方 SOAR / 32 steps / 相同 prompt / RAW

输出：

`04_base32_bf16_raw.wav`

条件：

- **不加载 LoRA**
- 官方 `dots.tts-soar`
- Euler 32
- BF16
- 优先 optimize=true；若需要同时跑 eager，可作为附加 case
- 相同 prompt audio/text
- 无 voice-polish

目的：判断高频磨砂是 LoRA 导致，还是 base / latent decoder / vocoder 已经存在。

### Case 5 — 当前生产 v11 control

输出：

`05_lora16_current_v11.wav`

完全按当前 `suoming_step500_v1` 正式路径生成。

目的：与 Case 1 直接做 raw → v11 对照，判断后处理净贡献。

### Case 6 — 极简后处理

**先完成前 5 个 case，再做此 case。**

从 Case 1/2/3 中主观和指标最好的 raw 作为输入，输出：

`06_best_raw_minimal.wav`

极简后处理仅允许：

- 100 Hz low-cut；
- 固定 gain / loudness match；
- true-peak safety。

必须关闭：

- brightness shelf；
- presence EQ；
- body EQ（本轮先关闭）；
- parallel compressor；
- de-esser；
- dynamic EQ；
- exciter；
- ambience；
- spectral stabilizer。

不要创建新的“修复算法”。

---

## 必做客观分析

不要只看一个 spectral-flatness 数字。

对 REF 与全部 case，至少计算：

### 1. 基本量

- duration
- RMS dBFS
- integrated LUFS
- true peak

### 2. 分频带相对能量

以 200 Hz–4 kHz 作为 voice reference band，至少输出：

- 4–8 kHz / reference band
- 8–12 kHz / reference band
- 12–18 kHz / reference band

单位 dB。

### 3. 高频结构

至少分别对 4–8 kHz、8–12 kHz 计算：

- spectral flatness：mean / median / p90
- spectral crest factor
- spectral entropy 或等价“谱峰集中程度”指标

仅在 voiced frames 上统计；voiced frame 的选择规则必须固定并写进报告。

### 4. 时间稳定性

针对 2–9 kHz：

- 计算相邻 STFT frame 的 log-magnitude 差异；
- 给出 median / p90 temporal delta；
- 用来判断是否真有 frame-to-frame spectral wobble。

不要直接把“flatness 更低”定义为“听感更干净”。

### 5. 同文本对齐可视化

生成至少两张图：

- REF vs Case1 vs Case2 vs Case4 的平均 log spectrum；
- REF vs Case1 vs Case5 的 4–12 kHz spectrogram。

图仅用于诊断，最终仍以试听为第一优先级。

---

## 必做试听包

生成：

`outputs/diagnostics/suoming_grit_v1/listen/`

里面放：

- `00_reference_validation.wav`
- `01_lora16_bf16_opt_raw.wav`
- `02_lora32_bf16_opt_raw.wav`
- `03_lora32_fp32_eager_raw.wav`（若成功）
- `04_base32_bf16_raw.wav`
- `05_lora16_current_v11.wav`
- `06_best_raw_minimal.wav`

不要匿名。

再生成一个简单 `index.html`，每条音频显示：

- case 名称
- 参数
- audio player
- 用户可填：
  - 清澈度 1–5
  - 磨砂/颗粒 1–5
  - 像锁暝 1–5
  - 备注

不需要复杂前端。

---

## 判定逻辑

报告必须按下面逻辑明确给结论，不要只罗列指标。

### A. Case1 已明显磨砂

说明问题已经在 raw synthesis 中存在。

此时**不要继续优化 voice-polish**，优先比较 Case2/3/4。

### B. Case2 明显优于 Case1

说明 16 steps 的 flow sampling 精度是主要变量之一。

下一轮才值得测试：

- 20
- 24
- 28
- 32

本任务不要自动扩展 sweep，先提交本轮结论。

### C. Case3 明显优于 Case2

说明 BF16 / optimize / compile 数值路径是重要变量。

下一轮再拆：

- BF16 eager
- FP32 optimize（如支持）
- BF16 optimize

本轮不要无限展开。

### D. Case4 明显比 LoRA case 清澈

说明 Step500 LoRA 或其训练设置改变 latent 分布，是主要嫌疑。

下一轮优先比较：

- step100/200/300/400/500；
- LoRA rank / target modules；
- train_output_layer；
- cfg/xvec drop rate。

**本任务不重新训练。**

### E. Case4 也存在近似磨砂

说明问题更可能来自：

- SOAR 本身在该 speaker / prompt 下的生成上限；
- latent representation；
- AudioVAE / BigVGAN decoder；
- prompt conditioning。

此时报告要明确写出：继续堆后处理只能遮掩，不能从根本恢复原声那种干净谐波结构。

### F. Case1 比 Case5 清澈

说明当前 v11 后处理在放大或引入磨砂。

必须逐项指出最可疑模块，但**本轮不要创建 v12**。

重点检查：

- +2 dB above 6 kHz brightness；
- 3.2 kHz presence；
- 400–9000 Hz spectral stabilizer；
- synthetic-noise ambience。

### G. Case6 比 Case5 更自然/清澈

说明生产链应优先向“少处理”收缩，而不是继续增加插件。

---

## 关于当前 spectral stabilizer 的专项检查

当前实现：

- STFT magnitude
- log magnitude
- 沿时间 median filter
- 原 phase 重建

必须做一个简单 unit/offline check：

1. 把 validation 原声输入 stabilizer；
2. 只开 stabilizer，其余全部关闭；
3. 比较处理前后：
   - 4–8 kHz flatness
   - 8–12 kHz flatness
   - crest
   - temporal delta
4. 保存：
   - `stabilizer_reference_before.wav`
   - `stabilizer_reference_after.wav`

如果 stabilizer 在干净原声上显著降低谱峰对比、提高宽带平坦度，报告中明确标记为：

`risk: spectral smearing`

不要因为名字叫 stabilizer 就默认它能去除“磨砂”。

---

## 关于 ambience 的专项检查

当前 ambience 是：

- seeded random noise IR
- bandpass 250–7000 Hz
- exponential decay
- convolution
- mix 0.08

必须在报告中明确说明这是 **synthetic noise impulse response**，不是实测 room IR。

用 validation 原声做 before/after：

- 只开 ambience；
- 保存 before / after；
- 检查 2–7 kHz 宽带能量和 spectral flatness 是否上升。

若上升且主观更“散/砂”，标记：

`risk: diffuse broadband texture`

---

## 禁止事项

本轮禁止：

- 新增 v12/v13 voice-polish；
- 新增 exciter；
- 新增 denoiser；
- 新增神经后处理模型；
- 重新训练 LoRA；
- 修改训练集；
- 因某个指标变好就宣称问题解决；
- 使用 spectral flatness 作为唯一“干净度”判据；
- 隐藏失败 case；
- 自动进行几十组参数 sweep；
- 修改正式 `suoming_step500_v1.yaml` 的默认行为。

诊断脚本和报告可以新增；生产 profile 暂时不要改。

---

## 输出

必须新增：

1. `scripts/diagnose_suoming_grit.py`
2. `docs/reports/suoming-grit-diagnostic-v1.md`
3. `outputs/diagnostics/suoming_grit_v1/metrics.json`
4. `outputs/diagnostics/suoming_grit_v1/listen/index.html`
5. 必要的图表和 metadata

WAV 如果仓库体积策略不适合提交，可以不 commit，但报告中必须写绝对/相对输出路径，并确保本地实际生成成功。

---

## 报告最后必须有这个表

| Case | 清澈/磨砂主观结论 | 8–12k 相对能量 | 8–12k flatness | 8–12k crest | temporal delta | 结论 |
|---|---|---:|---:|---:|---:|---|
| REF | | | | | | |
| LoRA16 raw | | | | | | |
| LoRA32 raw | | | | | | |
| LoRA32 FP32 eager | | | | | | |
| Base32 raw | | | | | | |
| Current v11 | | | | | | |
| Minimal | | | | | | |

如果没有用户试听结果，主观栏写：

`PENDING_USER_LISTENING`

不要替用户捏造评分。

---

## 完成标准

只有同时满足以下条件才算完成：

- 所有可运行 case 实际生成；
- raw 与 v11 路径明确分离；
- 不再把“磨砂”简单等同于高频多或少；
- 能明确判断下一步应该查 sampling、precision/optimize、LoRA、base/vocoder 还是 postprocess；
- 报告明确写出**最可能根因排序**；
- 没有改正式生产 profile；
- 所有新增脚本可复现；
- tests 至少覆盖诊断指标函数与关键 config/case wiring；
- 提交到 GitHub。

最终回复只需要给：

1. commit SHA；
2. 诊断报告路径；
3. 试听页路径；
4. 最可能根因 Top 3；
5. 哪一个下一步实验最值得做。

不要在没有试听证据时宣称“已经修好”。
