# Devin Task — 守岸人 dots.tts Zero-shot / MF 盲听诊断

## 目标

当前守岸人有两个独立问题要拆开确认：

1. 音色不像守岸人：可能来自 prompt conditioning 链路、参考音频、MF / MF-1step 蒸馏后的 speaker fidelity。
2. 听感“沙 / 粗糙 / 沙哑”：可能来自声学模型、VAE/vocoder、参考音频或后处理。此前单一的 4–12 kHz 平均 spectral flatness 不能作为“已经不沙”的充分证据。

本任务只做可复现诊断 + 盲听样本生成。不训练、不重训 LoRA、不调正式 WebUI 参数。

最终主观听感由用户判断。Devin 不要用“频谱指标正常”替代人工试听结论。

## 当前仓库状态（执行前先复核）

当前 main 已有：

- configs/voices/shouanren_mf_zero_v1.yaml
  - base_model.path = models/dots.tts-mf-1step
  - 无 adapter
  - prompt 音频：
    data/inbox/守岸人/中立_neutral/【中立_neutral】或许这就是泰缇斯系统冒风险也要将悲鸣与噬亡星结合的原因…….wav
  - prompt 精确文本：
    或许这就是泰缇斯系统冒风险也要将悲鸣与噬亡星结合的原因……
  - 当前 profile 为 fixed 1-step、speaker_scale 1.5
- configs/voices/shouanren_step500_v1.yaml
  - SOAR + Step-500 LoRA
  - 守岸人另一条标准化参考：
    data/work/standardized/56/56784f46414575ff5789a2857fcb764ab5dcad40ce7522790997cb9d889418cd.wav
  - 精确文本：
    欢迎回到，这片属于你的海岸。
- configs/voices/fuxuan_step400_v1.yaml
  - 扶玄参考音频可作为明显不同 speaker 的 conditioning sanity check
- src/dots_tts_lab/fuxuan_batch.py::load_runtime
  - adapter 已可选；adapter=None 时走 DotsTtsRuntime.from_pretrained
- src/dots_tts_lab/fuxuan_webui.py
  - profile 的 prompt.audio_path / prompt.text 已传到生成链
- src/dots_tts/runtime.py
  - 支持 prompt_text=None，可直接做 x-vector-only 诊断
- inputs/shouanren_text/睡眠.txt

注意：不要把文件名中的“【中立_neutral】”写进 prompt_text。prompt_text 必须只包含参考音频实际说出的内容。

## 上游采样契约

执行时以官方当前模型卡为准，并把实际模型 revision / config SHA-256 记录到报告。

- SOAR：flow-matching，NFE 10–32。本项目为稳定性一直使用 Euler 16 + CFG 1.2，本次继续用 16。
- dots.tts-mf：MeanFlow，官方推荐 NFE=4；CFG 已蒸馏进 student，guidance_scale 对该 checkpoint 无效。
- dots.tts-mf-1step：fixed one-step artifact。调用时不要人为塞多步参数；优先让 artifact 自己选择固定采样契约。

如果本地没有 dots.tts-mf：
- 下载官方 dots-studio/dots.tts-mf 到独立目录，建议 models/dots.tts-mf。
- 记录来源、revision / commit、config.json SHA-256。
- 不要把 mf-1step 改成 num_steps=4 来冒充 MF。

参考：
- https://github.com/studio-dots-ai/dots.tts
- https://huggingface.co/dots-studio/dots.tts-mf
- https://huggingface.co/dots-studio/dots.tts-mf-1step

## 禁止事项

本任务中不要：

- 训练或继续训练任何 LoRA / full model；
- 改写现有冻结数据集、checkpoint、参考音频；
- 覆盖 outputs/shouanren_audio 或 outputs/shouanren_mf_audio；
- 调 voice_polish 来“修”试听结果；
- 在 raw 诊断音频上做降噪、EQ、压缩、de-esser、响度标准化；
- 只根据 spectral flatness 宣布问题解决；
- 为了得到好结果不断换 seed。核心矩阵必须固定 seed。

如果发现明确的生产链 bug，先在报告里给出最小复现和证据；不要顺手大改正式 WebUI。

## 固定测试文本

只用短句，避免 5 分钟长文把变量混在一起。

T1 — 中性叙述：

森林里的小动物们都说，月亮最近失眠了。

T2 — 较软的角色对白：

我睡不着。风把云朵吹散了，天空太亮，我找不到做梦的枕头。

两条都来自现有 inputs/shouanren_text/睡眠.txt。

## Reference 定义

### R1 — 当前 MF 零样本参考

音频：

data/inbox/守岸人/中立_neutral/【中立_neutral】或许这就是泰缇斯系统冒风险也要将悲鸣与噬亡星结合的原因…….wav

文本：

或许这就是泰缇斯系统冒风险也要将悲鸣与噬亡星结合的原因……

### R2 — 守岸人标准化参考

音频：

data/work/standardized/56/56784f46414575ff5789a2857fcb764ab5dcad40ce7522790997cb9d889418cd.wav

文本：

欢迎回到，这片属于你的海岸。

### R3 — 扶玄 sanity-check 参考

从 configs/voices/fuxuan_step400_v1.yaml 读取当前已冻结的 prompt.audio_path 与 prompt.text，不要复制旧值硬编码。

## Phase 0 — Preflight

1. git status，保留用户已有的无关改动。
2. 校验 R1/R2/R3 均存在。
3. 校验 profile 内记录的 SHA-256。
4. 对 R1/R2/R3 输出 sample rate、channels、duration、peak、RMS / LUFS（若现有依赖可用）。
5. 明确打印实际传给 runtime 的 base model path、adapter 是否为 None、prompt path、prompt_text、seed、sampling contract。
6. 若 R1 的实际语音与上述 transcript 不一致，停止核心实验并报告；不要用 ASR 猜一个文本后继续。

## Phase 1 — 新增独立诊断脚本

新增：

scripts/diagnose_shouanren_zero_shot.py

要求：

- 直接调用 DotsTtsRuntime；
- 不经过正式 voice-polish；
- raw WAV 原样保存；
- 每个 case 调用前固定 seed_everything(42)；
- 输出 WAV 建议 PCM24；
- 每条音频旁边保存 metadata JSON；
- metadata 至少包含 case_id、model / model revision / config hash、prompt audio SHA-256、prompt_text、target_text、seed、actual sampling args、sample rate / duration、output SHA-256、wall time / RTF（若方便）；
- 失败必须保留异常和 case_id，不能静默跳过。

输出目录使用新版本目录：

outputs/diagnostics/shouanren_zero_shot_v1/
- raw/
- listen/
- metadata/
- report/

不得覆盖历史输出。

## Phase 2 — 核心 12 条 A/B

固定 seed=42、speaker_scale=1.5。

| Case | 模型 | Adapter | Reference | Target | 采样 |
|---|---|---|---|---|---|
| S-R1-T1 | SOAR | None | R1 | T1 | Euler 16, CFG 1.2 |
| S-R1-T2 | SOAR | None | R1 | T2 | Euler 16, CFG 1.2 |
| S-R2-T1 | SOAR | None | R2 | T1 | Euler 16, CFG 1.2 |
| S-R2-T2 | SOAR | None | R2 | T2 | Euler 16, CFG 1.2 |
| M-R1-T1 | MF | None | R1 | T1 | NFE 4，CFG fused |
| M-R1-T2 | MF | None | R1 | T2 | NFE 4，CFG fused |
| M-R2-T1 | MF | None | R2 | T1 | NFE 4，CFG fused |
| M-R2-T2 | MF | None | R2 | T2 | NFE 4，CFG fused |
| O-R1-T1 | MF-1step | None | R1 | T1 | artifact fixed contract |
| O-R1-T2 | MF-1step | None | R1 | T2 | artifact fixed contract |
| O-R2-T1 | MF-1step | None | R2 | T1 | artifact fixed contract |
| O-R2-T2 | MF-1step | None | R2 | T2 | artifact fixed contract |

重要：SOAR case 必须是无 LoRA。不要因为仓库默认有守岸人 Step-500 就误挂 adapter。

这 12 条的目的：
- 同一参考跨 SOAR / MF / MF1：看蒸馏层级对音色与粗糙感的影响。
- 同一模型跨 R1 / R2：看问题是否主要来自 reference。
- 同一 case 跨 T1 / T2：看问题是否只在某种韵律下暴露。

## Phase 3 — Conditioning sanity check

### C1：MF-1step + 扶玄 R3

固定 T1、seed=42，直接用 R3 生成。

结果与 O-R1-T1 一起进入盲听包。

目的：确认更换 reference 后输出 speaker 是否明显改变。

如果 R1 和 R3 的生成在 CAM++ / 人工听感上都几乎不变，优先怀疑：
- prompt audio 未真正进入模型；
- x-vector 路径失效；
- wrapper / profile / runtime conditioning 链路有 bug。

不要先归因于“MF1 本来就不像”。

### C2：MF-1step x-vector-only

固定 R1 + T1 + seed=42，调用 runtime：
- prompt_audio_path=R1
- prompt_text=None

这是上游支持的 x-vector-only clone。

与 O-R1-T1（audio + exact transcript）比较，判断 continuation conditioning 与纯 x-vector conditioning 的差异。

不要为此修改 fuxuan_batch.py 的生产校验；诊断脚本直接调用 runtime 即可。

## Phase 4 — 当前 WebUI 路径复现

额外生成 1 条：
- profile：shouanren_mf_zero_v1
- 文本：T1
- seed：42
- 按当前 WebUI / batch 正式路径执行

保留正式路径产生的最终文件，同时保留 direct-runtime 的 O-R1-T1 raw。

比较：
1. direct runtime raw；
2. 当前 batch/WebUI path 的生成前 raw（若现有链路拿不到，允许加最小诊断 hook，但不要改变声音）；
3. 当前 postprocess 后 final。

目的：判断“沙”是否由后处理引入或放大，以及 wrapper 是否改变了模型调用。

## Phase 5 — 客观指标（只辅助，不代替试听）

优先复用仓库现有代码，不额外引入大型依赖。

### 5.1 Speaker similarity

复用：

src/dots_tts_lab/speaker_embedding.py

使用仓库现有 CAM++ speaker embedding 计算：
- 每条守岸人生成 vs R1 cosine；
- 每条守岸人生成 vs R2 cosine；
- C1 扶玄生成 vs R3 cosine；
- C1 扶玄生成 vs R1/R2 cosine。

报告同时保留两条守岸人 reference 的结果，避免单条 reference 偏差。

### 5.2 粗糙度辅助指标

至少记录：
- integrated LUFS / RMS / peak；
- 4–12 kHz spectral flatness；
- 短时 spectral flatness 分布（median / p90 / max），不要只算整句平均；
- 若现有依赖可以无风险实现，再加 HNR / CPP；否则不要为了指标安装复杂新栈。

禁止定义“flatness 接近原声 = 不沙”。该指标只能作为相关证据。

## Phase 6 — 生成盲听页面

生成：

outputs/diagnostics/shouanren_zero_shot_v1/listen/index.html

outputs/diagnostics/shouanren_zero_shot_v1/listen/answer_key.json

页面要求：
- 隐藏真实模型和 reference 名称，只显示随机匿名编号 A01.wav、A02.wav 等；
- 同一 target 的样本放在一起，但顺序随机化；
- 页面显示 target text；
- 每条提供三个 1–5 分评分项：
  1. 像守岸人程度；
  2. 沙 / 粗糙程度（1=不沙，5=非常沙）；
  3. 自然度；
- 提供自由备注框；
- 提供导出 JSON 按钮；
- 默认不要显示 answer key；
- answer_key.json 单独保存真实 case 映射。

可额外生成一套仅做响度匹配的 listen/level_matched/ 副本帮助公平试听，但：
- raw 必须保留；
- 不得 EQ / 降噪 / 动态压缩；
- 报告里明确区分 raw 与 level-matched。

## Phase 7 — 报告

生成：

docs/reports/shouanren-zero-shot-diagnostic-v1.md

报告必须包含：

1. 当前 main commit SHA。
2. 运行环境：GPU、CUDA、PyTorch、dots.tts commit/package version。
3. 三个 checkpoint 的精确来源与 hash / revision。
4. R1/R2/R3 的路径、SHA-256、时长和 transcript。
5. 所有生成 case 的完整表格：case_id、model、ref、target、seed、sampling、output path、output sha256、duration、RTF、speaker cosine、roughness auxiliary metrics。
6. blind listening 页路径。
7. 不要填写用户主观结论，除非用户已经提交了试听 JSON。
8. 给出诊断分支判断，但写成“证据支持/不支持”，不要武断。

判断规则：

- 若 R1→R3 换 speaker 后输出基本不变：conditioning 链路列为 P0 blocker。
- 若 R2 普遍明显优于 R1：reference 质量/风格是主要变量之一。
- 若 SOAR > MF > MF1 的 speaker cosine 与人工“像”评分呈连续下降：支持蒸馏/fixed-step speaker fidelity trade-off。
- 若 direct MF1 raw 不沙，但 WebUI final 沙：后处理链优先排查。
- 若 direct MF1 raw 已经沙，而 postprocess 仅轻微改变：问题在模型/解码路径，不要继续堆后处理。
- 若 MF NFE4 同时比 SOAR 干净、又比 MF1 更像：把 dots.tts-mf 作为下一阶段首选候选；此任务仍不修改正式 profile。
- 若 SOAR zero-shot 已经不像，而 R1/R2 都验证正确：不要把问题全部归因于 MF 蒸馏；需要重新评估 reference / speaker encoder / dots.tts 对该角色的 zero-shot 上限。

## 最低验收条件

- [ ] 所有 preflight 资产都验证过；若缺失有明确 blocker。
- [ ] R1/R2 transcript 没混入“【中立_neutral】”等文件名元数据。
- [ ] SOAR zero-shot 明确无 adapter。
- [ ] MF 使用真正 dots.tts-mf NFE4，而不是 mf-1step 强行 4 steps。
- [ ] MF-1step direct API 使用 artifact fixed contract。
- [ ] 核心 12 条全部生成成功，或逐条记录失败原因。
- [ ] 完成 R3 conditioning sanity check。
- [ ] 完成 x-vector-only case。
- [ ] 至少有一条 WebUI/current-profile path 与 direct runtime 对照。
- [ ] raw 音频未经过 voice polish / EQ / 降噪 / LUFS normalization。
- [ ] blind listening HTML + answer key 已生成。
- [ ] 客观指标表已生成，但没有用单一指标代替试听。
- [ ] docs/reports/shouanren-zero-shot-diagnostic-v1.md 已写。
- [ ] 新增/修改代码有最小测试，运行相关单测和 compileall。
- [ ] 提交 commit，commit message 明确写 diagnostic/listening test，不要声称已解决音质问题。

## 停止条件

出现以下任何情况时不要继续大范围调参：

1. prompt transcript 不确定；
2. reference 文件/hash 与 profile 不一致；
3. SOAR/MF/MF1 checkpoint 身份不确定；
4. conditioning sanity check 表明 reference 根本没有影响输出；
5. GPU/模型资产不足以真实运行。

遇到停止条件，提交现有诊断代码与报告，明确 blocker，等待用户决定。

本任务优先级是把变量拆干净并生成可听的盲测包，不是追求某个漂亮分数，也不是在一次任务里把 TTS 最终方案定死。
