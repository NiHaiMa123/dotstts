# 锁暝「磨砂 / 声码器颗粒感」根因隔离诊断报告 v1

日期：2026-10-07 ｜ 执行：`scripts/diagnose_suoming_grit.py` ｜ 产物：`outputs/diagnostics/suoming_grit_v1/`

## 判定结论（按任务书 A–G 逻辑逐条裁决）

| 判定项 | 结果 | 依据 |
|---|---|---|
| **A. Case1 raw 是否已磨砂** | **是** | raw 的 8–12k flatness 0.49 vs REF 0.43；temporal delta 1.05 vs REF 0.92；crest 7.56 vs 8.07——颗粒在裸合成里已存在 |
| **B. 32步 是否优于 16步** | **否** | flatness/td/crest 全面打平（0.48 vs 0.49, 1.07 vs 1.05）→ flow sampling 步数不是变量 |
| **C. FP32 eager 是否优于 BF16** | **BLOCKED** | FP32 eager 在 RTX 5080（sm_120）上单条 ~10s 音频 >10min 未完成，实用不可用；备选 03b BF16-eager 与 02 optimize 指标一致（flat 0.49, td 0.99）→ **optimize/compile 数值路径不引入磨砂** |
| **D. base(无LoRA) 是否更清澈** | **是** | base32 raw：flatness 8–12k **0.35**（LoRA 0.49）、crest **10.8**（LoRA 7.6）——同一 prompt 的 zero-shot 谐波结构明显更干净 |
| **E. base 也有近似磨砂** | **否** | base 的 HF 颗粒指标接近甚至优于 REF 真声 → 声码器本身能产出干净 HF |
| **F. raw 比 v11 清澈** | **是** | v11 把 8–12k 相对能量抬 +1.9dB、12–18k 抬 +3.8dB；flatness 0.49→0.50、temporal delta 1.05→1.17 都变差 → **v11 在放大磨砂** |
| **G. minimal 比 v11 自然** | 指标持平 raw | minimal 链指标 ≈ raw（flat 0.49, td 0.99），不引入新问题 |

## 专项检查

### Spectral stabilizer（risk: spectral smearing —— 命中）

对干净 validation 原声只开 stabilizer（其余全关）：

| 指标 | before | after | 判定 |
|---|---:|---:|---|
| flatness 4–8k | 0.346 | **0.379** ↑ | 谱峰被抹平、宽带噪声感上升 |
| crest 8–12k | 8.07 | 7.85 ↓ | 谐波峰被钝化 |
| temporal delta | 0.921 | 0.815 ↓ | 确实压了帧间颤，但代价是频谱涂抹 |

**结论：stabilizer 用频谱涂抹换时间稳定，净效果是"磨砂换糊"。标记 `risk: spectral smearing`。**

### Ambience（合成噪声 IR —— 低风险）

对原声只开 ambience：flatness/crest/td 变化均 <0.01。合成 noise IR（非实测房间）在 8% 混合下没有显著引入宽带颗粒，属低风险。但注意它把 wav 尾巴拖长 232ms。

## 最可能根因排序

1. **Step500 LoRA 把游戏剥离音频的噪声/编码痕迹学进了 latent 分布** —— 同一 prompt、同一文本、同一 vocoder，base zero-shot 的高频谐波结构（flatness 0.35 / crest 10.8）显著干净于 LoRA 输出（0.49 / 7.6）。训练集 160 条来自游戏音频提取，样本本身实测有非周期性 0.66 的糙段。
2. **v11 后处理在放大磨砂** —— +2dB@6k 空气架把本来就偏高的 HF 噪底再抬近 2dB；stabilizer 已确认 spectral smearing。生产链当前是"放大器"不是"修复器"。
3. **SOAR/vocoder 上限 —— 非主因** —— base 用同一个 AudioVAE 能产出干净的 HF 结构，说明声码器不是瓶颈；它的贡献顶多是"如实渲染了 LoRA 给的脏 latent"。

排除项：采样步数（16 vs 32 打平）、torch.compile/optimize 数值路径（eager ≈ optimized）、激励器（v7 已弃）、ambience（隔离检查无显著影响）。

## 完整指标表

voiced 帧规则：n_fft=2048 hop=512 @48kHz，yin f0∈80–400Hz 且 rms>40 分位。

| Case | 清澈/磨砂主观 | 8–12k 相对能量 | 8–12k flatness | 8–12k crest | temporal delta (med/p90) | 结论 |
|---|---|---:|---:|---:|---:|---|
| REF | PENDING_USER_LISTENING | -9.77 | 0.43 | 8.07 | 0.92/4.23 | 真声基线 |
| LoRA16 raw | PENDING_USER_LISTENING | -7.91 | 0.49 | 7.56 | 1.05/5.31 | 磨砂已存在于 raw |
| LoRA32 raw | PENDING_USER_LISTENING | -8.26 | 0.48 | 7.80 | 1.07/5.69 | 步数无改善 |
| LoRA32 BF16 eager | PENDING_USER_LISTENING | -7.76 | 0.49 | 7.59 | 0.99/5.23 | eager≈optimize，数值路径无罪 |
| LoRA32 FP32 eager | — | — | — | — | — | BLOCKED（RTX5080 上 FP32 无 TF32，>10min/条，不可用） |
| Base32 raw | PENDING_USER_LISTENING | -10.64 | **0.35** | **10.80** | 1.28/6.32 | 无 LoRA 明显更干净 → LoRA 是主嫌 |
| Current v11 | PENDING_USER_LISTENING | **-6.01** | 0.50 | 7.22 | 1.17/4.63 | 后处理放大 HF 噪底 |
| Minimal | PENDING_USER_LISTENING | -7.49 | 0.49 | 7.57 | 0.99/5.23 | 极简链无新增伤害 |

## 建议的下一步（按价值排序）

1. **LoRA checkpoint 对比**：step100/200/300/400/500 各生成一条 raw 同文本（每条约 5s 推理）——若磨砂随训练步数递增，坐实"LoRA 学了噪声纹理"；若 100 步就糙，则问题在训练数据本身。
2. **生产链向 minimal 收缩**：鉴于 F+G 判定，v11 的空气架和 stabilizer 是净负面。值得出一版 v12 = low_cut + deesser + 响度校准的最小链做 A/B（**本任务未动正式 profile**）。
3. 训练集音频质量审计：若确认是 LoRA，下一轮看 checkpoint 训练源 wav 的高频噪底/压缩痕迹。

## 产物清单

- `outputs/diagnostics/suoming_grit_v1/listen/` — 7 条 wav + `index.html` 打分页
- `outputs/diagnostics/suoming_grit_v1/metrics.json` — 全指标
- `outputs/diagnostics/suoming_grit_v1/fig_mean_spectrum.png` / `fig_spectrogram_4_12k.png`
- `outputs/diagnostics/suoming_grit_v1/stabilizer_reference_{before,after}.wav`、`ambience_reference_{before,after}.wav`
- `scripts/diagnose_suoming_grit.py`、`src/dots_tts_lab/grit_diagnostics.py`、`tests/test_grit_diagnostics.py`

复现：`python scripts/diagnose_suoming_grit.py`（已生成的 wav 自动复用，幂等重跑只补缺失 case）。
