# dotstts 本地多音色 LoRA 工作区

> 2026-09-22：仓库已从 `D:\project\dotstts` 迁移至 `E:\project\dotstts`。迁移时移除了训练集产物（`datasets/`、`data/references/`、`data/raw/`、`data/work/standardized/` 中除 WebUI 参考音频外的语料、`data/work/long_form/exports/strict_v1/`），均可经流水线重建；`exports/suisui/v1/` 因冻结参考包 `reference_audio_root` 依赖而保留。

本仓库基于 dots.tts，增加了从语料治理、LoRA 训练、评测、后处理到本地多音色 WebUI 的完整工作流。当前扶玄音色已经完成 Slice 1–14，可直接生成；后续音色应复用同一流程，并通过音色注册表接入 WebUI。

## 给后续 agent 的约定

- 本文件是项目入口和当前状态的唯一摘要；详细证据保留在 `docs/`。
- 原始语料、冻结数据集、已验收 checkpoint 和验收报告不可原地覆盖。改动必须使用新版本、新配置或新输出目录。
- 训练、评测与后处理由 agent/脚本执行；WebUI 只消费已验收并登记的音色配置。
- WebUI 切换“模型”时，必须同时切换基础模型、LoRA、参考音频与文本、推理参数、后处理和输入/输出目录，不能只换 adapter。
- 当前运行配置以 `configs/voices/*.yaml` 为准；Slice13 导出文件是历史验收快照，不代表 WebUI 的最新参数。
- 修改后至少运行单元测试和 `compileall`，并保留用户已有的无关改动。

## 当前状态

| 项目 | 结果 |
| --- | --- |
| 流程 | Slice 1–14 全部完成 |
| 数据集 | `fuxuan@1`，273 条；训练/验证/测试为 219/27/27 |
| 冻结数据 | `datasets/fuxuan/v1/`（2026-09-22 迁移到 E: 时已删除，训练集可经流水线重建） |
| 入选模型 | 官方 SOAR 基座 + Slice12 step400 LoRA |
| step500 | 已评测并淘汰，不应替换 step400 |
| 正式评测 | 72/72 成功；CER 0.05627；speaker cosine 0.79937 |
| 性能 | 平均 RTF 0.3004；峰值 CUDA 6.4583 GiB |
| 自动测试 | 最近一次 263/263 通过（2026-09-12） |
| 运行方式 | Windows 本地推理；Docker 仅用于训练环境复现 |

数据集 tree hash：`77754574c4048a20682b0e8f9be95c322cf0848afedaf90d7c81b1f00df029af`。

## 快速使用

Windows 下双击根目录的 `启动TTS.vbs`。浏览器是唯一常驻界面；关闭 WebUI 页面后，服务及其子进程会退出。旧的 `启动扶玄TTS.vbs` 仅作为兼容入口保留。

也可以手动启动：

```powershell
.\.venv\Scripts\python.exe scripts\launch_voice_webui.py
```

WebUI 只有模型选择和 TTS 操作。它会显示任务进度与持久化状态；异常退出后，下次启动会标记中断并清理 `.partial.wav`。生成成功后可直接打开输出目录。

当前扶玄音色使用：

- 输入目录：`inputs/fuxuan_text/`
- 输出目录：`outputs/fuxuan_audio/`
- 一个 `.txt` 生成一个同名 `.wav`
- 多个 `.txt` 批量生成多个最终音频
- 单句中间音频不会保留

兼容的命令行批处理：

```powershell
.\.venv\Scripts\python.exe scripts\generate_fuxuan_from_txt.py --force
```

## 当前扶玄运行配置

音色定义：`configs/voices/fuxuan_step400_v1.yaml`
注册表：`configs/voices/registry.yaml`

关键绑定如下：

| 组件 | 当前值 |
| --- | --- |
| 基础模型 | `pretrained_models/dots.tts-soar` |
| 基座 revision | `2f9b3e18d70d670d4c701da2dc55ded5755815ce` |
| LoRA | `data/work/slice12/soar_lora_v1/checkpoint-00000400/model` |
| adapter hash | `de5b3bb59e510d622cf65813f75fd0f5c287c3ea8f043ea132c213d3f0d0ed54` |
| 采样 | Euler，16 steps，guidance 1.2，speaker scale 1.5 |
| 精度 | bfloat16；Windows eager 模式 |
| 文本切分 | 最长 120 字；段间停顿 250 ms |
| 后处理 | `configs/lab/postprocess/fuxuan_voice_polish_roughness_control_v2.yaml` |
| 边缘裁切 | `configs/lab/postprocess/generated_audio_edge_trim_v1.yaml` |

16 steps 是当前稳定默认值，用于规避 10 steps 固定种子下偶发的重声/回音。扶玄后处理 v2 保留轻微清亮度和厚度，同时使用动态存在感控制与齿音控制，降低持续粗糙感；这些参数属于扶玄音色，不可直接套用到其他 LoRA。

## 添加新音色

后续 agent 应按以下顺序执行：

1. 将原始素材放入独立目录，完成 catalog、规范化、去重、切分和冻结；不得修改已有冻结数据。
2. 选择固定参考音频，并保存其精确文本和 SHA-256。
3. 使用官方 SOAR 基座训练版本化 LoRA。训练入口可参考 `scripts/run_slice12_container_train.ps1`；Docker 不是本地推理依赖。
4. 在冻结测试集上比较 checkpoint，检查可懂度、音色、稳定性、RTF 和显存，不凭单条试听决定。
5. 为该音色单独调整后处理；配置放入 `configs/lab/postprocess/`，不要修改扶玄配置来兼容其他音色。
6. 新建 `configs/voices/<voice_id>.yaml`，绑定所有生成相关资产、参数、哈希及独立输入/输出目录。
7. 将 profile 加入 `configs/voices/registry.yaml`。正常情况下无需修改 Python 或前端代码。
8. 运行测试、哈希校验和真实 HTTP smoke，再将 profile 标记为可用。

字段说明和接入细节见 `docs/voice-model-registry.md`。切换音色时，服务会释放旧模型与 CUDA cache，再加载新 profile，避免 LoRA、参考音频或后处理串用。

### 30–60 分钟长音频

长音频、多角色、轻语和 3D 麦素材必须先经过独立 long-form 流程，不能直接交给原有“一个文件一条语句”的入口：

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_long_form.py "data/inbox/<音色目录>"
```

流程会分块读取，先生成宽松 VAD/ASR 候选，再按 1.5 秒语义停顿重切；同一句中的长空白会被压缩并以多个源采样区间保留追溯。随后只把 normal、主声线、较高 ASR/SNR、无明显空间声场或密集背景风险的前 40 条/源送首轮人工审核，其余合格项进入候补，技术不合格项只留审计记录。人工仍需确认残留噪声、其他人声、文本与风格；详细任务和当前审核页见 `PLAN.md`。导出的决定通过 `scripts/apply_long_form_review.py` 导入后，才可进入训练数据导出。

新版默认使用 `configs/lab/long_form/refinement_v2.yaml`，候选要求 2–15 秒，并在带上下文的读取后做零交叉调整、首尾淡化和句内拼接 de-click。精筛只保存特征，入选后逐条重读渲染。旧版 v1 配置和已审核片段保持原样。

运行目录绑定源文件、配置、实现版本及 ASR 配置内容；`--force` 创建独立运行，不覆盖已有结果。ASR 部分失败时保存成功结果，重跑只补失败项；有未完成 ASR 的源不会进入精筛，CLI 返回非零退出码并列出 `pending_asr_sources`。人工事件绑定清单和音频哈希，内容变化后需在新版本审核目录重新审核；旧日志只有匹配原清单的快照才能继续导入。

用户已报告 UVR5 失败，当前改用 **DeepFilterNet3**，取消等待 Vocal stem。已有 0.5.6 隔离环境定义、模型缓存和短片 AB 入口；生产接入计划复用 8 dB 衰减上限、关闭 post-filter、启用延迟补偿，新增带上下文的分块降噪和最终 gate。旧 205 条不恢复大批审核，旧 AB 不自动批准。接入范围、已有资产和验收见 [DFN3 补充方案](docs/reports/long-form-deepfilternet3-v2-plan.md)；已完成的通用修复见 `docs/reports/long-form-reliability-v2.md`。

2026-09-12 联网复查及后端调整后，当前待办为 [PLAN 的 LF-12 严格 gate 与 LF-13 DFN3 接入](PLAN.md)：先预筛正常语音区域、按需 DFN3 降噪，再验证目标参考声线、角色/空间、重叠/拟音、双 ASR 与最终音质。灰区默认隔离；新 6 源试点计划整批最多送审 24 条、每源最多 6 条，合格不足不凑数。新 gate、预算和长音频 DFN3 接入尚未实现；当前命令仍执行前述每源 40 条的旧策略。自动免逐条审核需先通过独立精度验收，不能把 ASR 高分或最大说话人簇视为安全证明。依据、模型边界与分阶段验收见 [严格准入研究](docs/reports/long-form-strict-gate-research-20260912.md)。

审核完成后使用 `scripts/export_long_form_dataset.py` 生成版本化数据片段。每个片段包含直接训练用 `trainer.jsonl`、后续汇总多源用 `catalog_candidates.jsonl`、目标参考、排除审计和全量校验和；单个源文件的少量片段不应直接拿去训练，先汇总足够的同音色素材再统一去重与冻结。

## 目录说明

| 路径 | 用途 |
| --- | --- |
| `data/inbox/` | 原始导入素材，只追加、不覆盖 |
| `data/catalog/catalog.sqlite` | 素材目录与来源记录 |
| `datasets/<voice>/vN/` | 冻结、版本化的数据集 |
| `data/references/` | 参考音频选择结果 |
| `data/work/` | 训练 checkpoint 与阶段性产物 |
| `data/reports/` | 评测、盲听和分析报告 |
| `configs/voices/` | WebUI 音色注册表及完整运行 profile |
| `configs/lab/postprocess/` | 每个音色独立的后处理配置 |
| `inputs/<voice>/` | 待生成文本 |
| `outputs/<voice>/` | 最终音频 |
| `docs/reports/` | 决策记录与验收证据 |

流程概览：

```text
原始素材 -> catalog -> 规范化/去重/切分 -> 冻结数据集
        -> 参考音频 -> LoRA 训练 -> checkpoint 评测
        -> 音色专属后处理 -> voice profile -> WebUI/批量生成
```

## 环境与验证

推荐 Python 3.10–3.12、NVIDIA CUDA 环境。首次安装可使用：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[full]" -c constraints/recommended.txt
```

常规验证：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
.\.venv\Scripts\python.exe -m pip check
```

训练前应确认官方 checkpoint、PEFT 依赖、CUDA、显存和容器挂载均可用。训练中断应从明确的 checkpoint 恢复，不覆盖已验收目录。

## 验收证据

- `docs/reports/slice7-12-audit.md`：Slice7–12 审计
- `docs/reports/plan2-acceptance-v1.json`：优化路线最终验收
- `docs/reports/slice12-acceptance.md`：step400 训练与选择
- `docs/reports/slice13-acceptance.md`：发布导出验收
- `docs/reports/slice14-acceptance.md`：WebUI 初始验收（注册表改造前的历史记录）
- `docs/voice-model-registry.md`：多音色绑定规范

## 上游项目

本仓库基于 [dots.tts](https://github.com/rednote-hilab/dots.tts)，当前包版本为 0.3.1，遵循 Apache-2.0 License。基础模型与第三方组件仍受各自许可证约束。
