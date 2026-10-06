# 守岸人 zero-shot / MF 盲听诊断报告 v1

任务书：`DEVIN_SHOUANREN_ZERO_SHOT_LISTENING.md`（诊断 + 盲听包，非定性结论）。

## 环境

- 主分支 commit：`3f9a4903022bdfdfebb17a922804bbb353e3f857`
- GPU：NVIDIA GeForce RTX 5080；CUDA 12.8；PyTorch 2.8.0+cu128
- 采样契约：SOAR = Euler 16 / CFG 1.2；MF = Euler NFE 4 / CFG fused（guidance_scale=0）；MF-1step = 固定 1 步 / g=0（artifact 契约）
- 全部 case：seed=42、speaker_scale=1.5、无 adapter（SOAR zero-shot 确认未挂 LoRA）、bf16

## Checkpoint 来源与指纹

| 模型 | 路径 | 来源 | 权重 SHA-256(前16) |
|---|---|---|---|
| SOAR | `pretrained_models/dots.tts-soar` | 本地既有 | 2787e6d4fe0b27ac |
| MF | `pretrained_models/dots.tts-mf` | HF `dots-studio/dots.tts-mf` @ `c28105a`（本次补下 model+vocoder safetensors） | a16d5798da197bf6 |
| MF-1step | `models/dots.tts-mf-1step` | 本地既有（官方固定一步 artifact） | 6776f62bbd5332c2 |

## Reference

| ID | 路径 | SHA-256(前16) | 时长 | 响度 | transcript |
|---|---|---|---|---|---|
| R1 | `data/inbox/守岸人/中立_neutral/【中立_neutral】或许这就是…….wav` | d18a5ac54a2b9d6d | 5.74s | -21.97 LUFS | 或许这就是泰缇斯系统冒风险也要将悲鸣与噬亡星结合的原因…… |
| R2 | `data/work/standardized/56/56784f46…418cd.wav` | da7e86684a7b397c | 3.38s | -21.11 LUFS | 欢迎回到，这片属于你的海岸。 |
| R3 | `data/work/standardized/6b/6bdb5989…610e9.wav` | 76676771bf7b8d9f | 9.48s | -16.23 LUFS | 各位启程前，请先好好休养歇息一番。有什么想去的地方，尽可以逛逛。我要暂代云骑事务，无法奉陪了。 |

prompt_text 均只含实际台词，未混入 `【中立_neutral】` 文件名元数据。R1 的 transcript 来自导出管线 voice_map 映射（与文件名一致），未经 ASR 复核。

## 生成结果（14/14 成功）

`outputs/diagnostics/shouanren_zero_shot_v1/`：`raw/`（PCM24 raw）、`metadata/`（逐 case 参数+hash+RTF）、`listen/`（盲听页）、`webui_final/T1.wav`（当前 profile 正式路径）、`report/metrics.json`。

| case | cos vs R1 | cos vs R2 | cos vs R3 | flat med | flat p90 | LUFS | dur(s) |
|---|---|---|---|---|---|---|---|
| S-R1-T1 | 0.884 | 0.626 | 0.275 | 0.274 | 0.421 | -21.3 | 4.0 |
| S-R1-T2 | 0.836 | 0.652 | 0.306 | 0.293 | 0.402 | -23.5 | 6.1 |
| S-R2-T1 | 0.715 | 0.824 | 0.326 | 0.299 | 0.389 | -18.3 | 4.3 |
| S-R2-T2 | 0.657 | 0.784 | 0.340 | 0.306 | 0.415 | -22.1 | 6.1 |
| M-R1-T1 | 0.851 | 0.672 | 0.243 | 0.228 | 0.387 | -21.0 | 4.2 |
| M-R1-T2 | 0.853 | 0.662 | 0.292 | 0.248 | 0.375 | -21.2 | 6.4 |
| M-R2-T1 | 0.675 | 0.806 | 0.345 | 0.277 | 0.432 | -18.9 | 4.2 |
| M-R2-T2 | 0.640 | 0.797 | 0.405 | 0.274 | 0.375 | -21.3 | 7.0 |
| O-R1-T1 | 0.847 | 0.649 | 0.280 | 0.227 | 0.387 | -19.3 | 4.5 |
| O-R1-T2 | 0.810 | 0.701 | 0.325 | 0.257 | 0.392 | -21.1 | 6.4 |
| O-R2-T1 | 0.673 | 0.820 | 0.255 | 0.290 | 0.406 | -17.3 | 4.5 |
| O-R2-T2 | 0.682 | 0.832 | 0.405 | 0.317 | 0.409 | -20.1 | 7.2 |
| C1-O-R3-T1 | 0.349 | 0.417 | **0.864** | 0.193 | 0.315 | -14.4 | 3.8 |
| C2-O-R1-T1-xvec | 0.829 | 0.622 | 0.265 | **0.121** | 0.233 | -12.4 | 3.8 |
| webui T1 (final) | 0.841 | 0.645 | 0.280 | 0.227 | 0.387 | -25.7 | 4.5 |

## 诊断分支判断（证据支持/不支持）

1. **conditioning 链路健康，非 P0 blocker。** C1 换扶玄 R3 后输出 speaker 明显改变（cos vs R3 = 0.864，vs 守岸人参考仅 0.35-0.42）。prompt audio 与 x-vector 路径都在起作用。
2. **reference 质量是主要变量之一（证据支持）。** 同一模型下，R2 条件生成的 4-12kHz flatness 普遍高于 R1（SOAR 0.299/0.306 vs 0.274/0.293；MF 0.277/0.274 vs 0.228/0.248；MF1 0.290/0.317 vs 0.227/0.257），且 cos vs 自身参考也更低。R2（3.4s 标准化短 clip）是更差的参考。
3. **continuation conditioning 是"沙"的主要来源（证据较强）。** C2 x-vector-only 输出 flatness median 0.121、p90 0.233——远低于一切 audio-continuation case（0.19-0.34），同时 cos vs R1 仍有 0.829。即"音色相似度保留 + 噪声大幅下降"同时成立。是否过度发闷需盲听确认（指标不能替代）。
4. **后处理不是"沙"的来源（证据支持）。** WebUI 正式路径 final 与 direct-runtime raw 的 flatness median 完全一致（0.227），仅响度被拉到 -26 LUFS。问题在模型/解码路径，堆叠 EQ/降噪只是掩盖。
5. **蒸馏 trade-off 证据弱。** SOAR > MF ≈ MF1 的 speaker cosine（0.884 > 0.851 ≈ 0.847），MF/MF1 flatness 反而略低于 SOAR。MF1 的"不像"更可能来自 reference/continuation 路径而非蒸馏本身。
6. **MF(NFE4) 作为下一阶段候选（弱支持）。** MF 在 R1 条件下 cos 与 MF1 持平、flatness 略低且 RTF 远优于 MF1（0.8-1.1 vs 6-33）。值得在下一阶段对比 MF 多步与 x-vector-only。
7. **未回答的问题（待盲听 JSON）。** "像守岸人"与"沙"的主观判断未填写；C2 xvec-only 是否发闷、R3-conditioned 输出为何 flatness 低（0.193）都需人工确认。

## 盲听

- 页面：`http://127.0.0.1:8788/outputs/diagnostics/shouanren_zero_shot_v1/listen/index.html`
- 15 条匿名 A01–A15（12 核心 + C1 + C2 + webui-final），每条提供 raw 与 level-matched（-20 LUFS 仅响度）两个版本切换
- `listen/answer_key.json` 为真实映射，页面默认不显示

## 复现

- 生成：`python scripts/diagnose_shouanren_zero_shot.py`
- 指标：`python scripts/diagnose_shouanren_metrics.py`
- 盲听包：`python scripts/build_shouanren_listen_pack.py`

## 已知限制

- MF-1step 在本机 bf16 下 RTF 6-33（远慢于 SOAR/MF），未查明原因，不影响样本有效性
- x-vector-only 路径目前只在诊断脚本直接调用 runtime；生产 `run_batch`/`voice_profile` 尚不支持 `prompt_text=None`，如要采用需后续工程接入

## 盲听结果（用户已提交）

**重要**：用户评分约定 `沙` 分数越高=越干净（1=最沙）。以下按此方向解读。

### v1（15 条，listen/shouanren_blind_v1.json）

- **参考音频决定"像不像"**：所有 R1 条件的 case 像=3-4，所有 R2 条件像=1，R3/xvec 像=1。R2（3.4s 标准化短 clip）作为参考完全失败，应退役。
- **最沙的两条都是 MF-1step+R1+T1**（O-R1-T1 与其 WebUI 成品，干净分=1）。mf-1step 的 1-step 采样存在固有沙哑伪影，与 flatness 指标（0.227/0.228，其实并不高）不一致——**谱平坦度与主观"沙"不相关**，不能单独作为判据。
- **SOAR 零样本+R1（S-R1-T1）像=4/干净=5/自然=4**，客观 flatness 0.274 排倒数但主观干净——再次证明平坦度不能当结论。
- **xvec-only（C2）像=1**：声纹向量保留了音色频率特征，但丢了守岸人的咬字韵律，主观上"不像"。该方向证伪。
- MF(NFE4) 与 mf-1step 在 R1 上各有"像4/干净5"的样本，MF 质量不差且 RTF 快 30 倍。

### v2 补充（6 条，listen_v2/shouanren_blind_v2.json）

针对"LoRA+R1"未覆盖的空白补测 3 条 + 3 条 v1 锚点重匿名：

| 匿名 | case | 像 | 干净 | 自然 |
|---|---|---|---|---|
| B01 | SL-R1-T2 | 5 | 5 | 5 |
| B02 | O-R1-T1（锚点） | 5 | 1 | 5 |
| B03 | SL-R1-T1 | 5 | 5 | 4 |
| B04 | S-R1-T1（锚点） | 5 | 5 | 5 |
| B05 | SL-R1-long（27s 长文） | 5 | 4 | 5 |
| B06 | M-R1-T1（锚点） | 5 | 5 | 5 |

锚点复现稳定（O-R1-T1 两轮均干净=1）。**最终结论：SOAR + Step-500 LoRA + R1 参考为最优组合**，短句/软语气/长文（干净 4-5）均达标；之前"又沙哑又不像"的成品病根是 R2 参考而非 LoRA 或底模。

### 已落地的改动

- `configs/voices/shouanren_step500_v1.yaml`：prompt 由 R2（标准化短 clip）换为 R1（原始长句 + 完整 transcript），profile_version → 2
- `configs/voices/shouanren_mf_zero_v1.yaml`：标注为实验档案，不建议日常使用
