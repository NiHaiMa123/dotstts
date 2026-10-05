# 长音频训练素材预处理计划

目标：从多个 30–60 分钟、混有多角色、轻语、耳语、3D 麦、底噪、拟音和语气词的音声中，优先挑选目标正常声线的干净完整句用于训练。流程可恢复、可审计，原始文件保持不变，最终片段遵守 2–15 秒契约。

## 当前优先级：严格准入，减少人工审核（2026-09-12 更新）

现有代码已支持多长文件流式处理；待解决的是降噪尚未正式集成、角色/正常发声识别证据不足，以及灰区进入人工队列过多。当前执行 **LF-12 严格 gate + LF-13 DeepFilterNet3 接入**：用户报告 UVR5 失败并明确改用 DeepFilterNet3，取消 UVR 等待依赖。研究、逐项 gate、模型依据和验收方法见 [严格准入研究](docs/reports/long-form-strict-gate-research-20260912.md)，后端参数、已有缓存与长音频适配见 [DFN3 补充方案](docs/reports/long-form-deepfilternet3-v2-plan.md)。以下新增工作尚未实现或取得准确率验证。

- 流程改为：多源分块预筛 → 找连续正常语音区域 → 原音直通或按需 DeepFilterNet3 降噪 → 说话人/角色/空间判别 → 自然句切片 → 最终 WAV 的事件、音质、双 ASR 和边界 gate → 去重/多样性 → 小批确认。
- 不再用“最大说话人簇”“未命中耳语规则”代表目标正常声线。增加固定多参考、时间轴重叠检测、同演员角色负例和 normal 正例验证；原始双声道证据在下混前保留。
- 所有必要内容 gate 必须通过；分数不能相互抵消，缺模型、缺特征、分歧、偏域均进入隔离区。内容通过但联合认证未完成时只进小批人工确认，不能自动入库。灰区保留记录但默认不送审，不强行修复全部素材。
- 试点整批最多 **24 条、每源最多 6 条**，对当前新 6 源同时生效；合格不足不凑数。相较旧 205 条，默认送审上限目标降低约 88.3%，不是已实现的质量提升。初期优先 3–12 秒连续自然句，其余合法时长和压缩拼接另行校准。
- 自动准入目标：联合通过集的片段级合格率单侧 95% 置信下界 ≥99%，关键缺陷验收为零。当前未达到；先输出少量 `strict_candidate` 人工确认，有独立验收证据后再实现/启用 `auto_verified` 与抽检。不能承诺任何模型通过的片段 100% 无误。
- UVR5 路线关闭，不等待 Vocal stem。DeepFilterNet3 作为选定降噪后端，先复用 0.5.6 与已有权重，以 8 dB 衰减上限、关闭 post-filter、启用延迟补偿为试点起点；拍击/音乐/重叠残留靠 gate 隔离。旧 AB 结论保留，旧 205 条不恢复大批人工审核，历史 31 条导出保持原样。

## 不可变规则

- `data/inbox/` 只读；所有切片和报告写入 `data/work/long_form/` 与 `data/reports/long_form/`。
- 所有产物绑定源文件 SHA-256、起止采样点、配置 SHA-256 和实现版本。
- 采用分块解码，不允许将整段长音频一次性读入内存。
- 自动规则可以排除不符合目标集的片段；多角色、轻语和 3D 麦等歧义保留证据、默认隔离、按需复核，不全部推送人工。未校准模型不能自动放行训练。
- 当前 ASR 文本仍是候选文本，不自动覆盖人工确认文本。未来 `auto_verified` 必须另立版本化契约、认证报告和导出校验，不能冒充 `human_confirmed`。
- 正常、轻语、耳语、角色音、3D/双耳必须分别标记，默认训练集只接收目标音色的正常说话。

## 已完成的基础任务（LF-1～LF-10）

以下勾选表示基础实现完成，不表示已通过 LF-12 对复杂音声的高精度准入验收。

- [x] LF-1：冻结长音频配置、数据契约、目录和状态机。
- [x] LF-2：实现分块音频读取与流式声学扫描，验证长音频内存有界。
- [x] LF-3：实现 VAD 候选区间、停顿合并及 2–15 秒句段切分。
- [x] LF-4：实现双声道/3D、噪声、削波、静音和重叠风险特征。
- [x] LF-5：接入说话人 embedding 与风格分组，生成目标音色筛选建议。
- [x] LF-6：接入带时间信息的 ASR 候选文本，并保留独立文本审核状态。
- [x] LF-7：实现可恢复批处理 CLI、原子写入、缓存和逐源文件状态记录。
- [x] LF-8：生成可试听的 HTML 审核页，并支持导入 append-only 人工结论。
- [x] LF-8A：分析真实 ASR 停顿与质量分布，冻结二次截断和首轮精筛策略。
- [x] LF-8B：实现 ASR 句级二次截断，去掉句间长空白并自动淘汰过短碎片。
- [x] LF-8C：实现纯人声优先的质量分层、每源首轮审核上限和候补区。
- [x] LF-8D：在“岁岁”真实样本上重建精简审核页并验证可追溯性。
- [x] LF-9：导出审核通过的标准短音频和 manifest，生成训练与 catalog/质量/去重/冻结交接清单。
- [x] LF-10：对 `data/inbox/岁岁` 做真实运行、资源统计和全项目回归测试。

## 当前样本基线

- 文件：`data/inbox/岁岁/林三歲-姐姐的憋尿野營.mp3`
- 时长：2004.432 秒（33 分 24 秒）
- 音频：48 kHz、双声道 MP3
- 引入 long-form 前的旧入口基线：可 inventory，但元数据规则不匹配；质量分析和标准化整段读入内存；单条策略上限 15 秒。现有 long-form 已另行实现分块处理。

## 执行记录

- LF-1（2026-09-12）：新增 `configs/lab/long_form/training_source_v1.yaml`、严格配置/片段契约 `src/dots_tts_lab/long_form_contract.py` 和 ADR 0010。配置只允许 `normal` 自动保留建议，ASR 文本必须人工确认，输出禁止写入 inbox；6 项单元测试和编译检查通过。
- LF-2（2026-09-12）：新增 `src/dots_tts_lab/long_form_audio.py`，使用 `SoundFile.read(frames=...)` 固定块读取并流式累计源哈希、峰值、声道 RMS、近削波比例和立体声相关性。4 项新增测试通过；与 LF-1 合计 10 项测试通过。真实扫描“岁岁”样本共 67 个 30 秒块，最大解码块 11,520,000 bytes，而不是整段约 770 MB float32 数据；源文件未改动。
- LF-3（2026-09-12）：新增 `src/dots_tts_lab/long_form_segmentation.py`。采用固定内存帧能量直方图和启停双阈值 VAD，合并 0.35 秒内停顿，输出 2–15 秒候选；超过硬上限的等分片段标记为必须复核，等待 LF-6 用 ASR 边界细化。14 项长音频测试通过。真实只读 dry-run 得到 158 个候选、18.85 分钟，时长 2.01–14.34 秒；67 个强制边界已正确进入复核，不直接准入训练。
- LF-4（2026-09-12）：新增 `src/dots_tts_lab/long_form_features.py`，每次只随机读取一个最长 15 秒候选，计算 SNR/静音/近削波、频谱平坦度、未校准重叠代理，以及局部声道相关性、声道差、声像移动和 Mid/Side 风险；空间异常不自动下混。19 项长音频测试与编译检查通过。真实样本均匀抽查 12 段，其中 3 段出现明确局部空间风险，证明不能依据整文件平均相关性决定下混；原始文件未改动。
- 阶段回归（LF-1～LF-4）：全项目 220/220 单元测试通过，`compileall` 通过。
- LF-5（2026-09-12）：新增 `src/dots_tts_lab/long_form_grouping.py`，复用已固定 SHA-256 的 CAM++ 编码器，支持安全下混/分析声道 embedding、确定性说话人聚类、参考音色 cosine、基频/周期性特征、稳健风格聚类和保守语义建议。真实均匀抽查 12 段得到 5 个说话人声学簇、2 个风格簇，建议 normal 4、binaural_3d 3、unknown 5；因尚无岁岁目标参考，12/12 均为 `reference_required`，没有自动准入。全项目 225/225 测试与 `compileall` 通过。
- LF-6（2026-09-12）：新增 faster-whisper long-form 配置、词级时间戳序列校验、候选文本绑定、边界触碰标记和受 15 秒硬上限约束的 ASR 词边界重切函数。ASR 只产生 `candidate` 文本，`human_confirmed_text` 保持空值；真实运行 158/158 成功、0 空文本、0 失败，平均词置信度 0.7844。
- LF-7（2026-09-12）：新增 `scripts/preprocess_long_form.py` 与可恢复 pipeline。每源文件保存 running/interrupted/failed/awaiting_review 状态，启动时清理 `.partial`，候选 PCM24 和 JSON 原子写入，embedding 按音频/模型哈希缓存。真实“岁岁”构建完成 158 个候选（155.3 MiB）和 158 个 embedding，最终状态 `awaiting_review`；原始 MP3 未修改。
- LF-8（2026-09-12）：新增 `scripts/build_long_form_review.py`、`scripts/apply_long_form_review.py` 和审核页面。页面支持“优先参考候选”等筛选、逐条试听、修改候选文本、确认风格、保留/排除、选择一个目标参考、“保存本条”可见状态、localStorage 恢复及 JSON 导入/导出；导入端校验 source/config/manifest 哈希，并以 batch ID 幂等追加人工事件。真实审核页已生成，共 158 条。全项目 234/234 测试与 `compileall` 通过。
- LF-8A（2026-09-12）：真实 158 条 ASR 时间戳显示，1.5 秒句间停顿可把首条按语义拆为 3 段，但全量会扩张为 219 个片段；仅切分不能降低审核量。冻结“1.5 秒停顿重切 + 有效字数/有效语音时长淘汰 + normal/主声线/ASR/SNR/空间风险联合排序 + 每源首轮上限 + 其余保留为候补”的可追溯策略。现有父片段严格联合筛选约 50 条，目标首轮审核控制在 40 条以内。
- LF-8B（2026-09-12）：新增 `long_form_refinement.py`、版本化 `refinement_v1.yaml` 和 `refine_long_form_review.py`。按 1.5 秒语义停顿重切；同一句内超过 0.35 秒的空白改为带独立 `source_spans` 的短连接，避免保留长空白且保持采样级追溯；有效人声、输出时长和文本过短的碎片只进入审计记录。首条测试固定为 3 个语义段，中间句由两段源区间合成约 2 秒；7 项相关回归测试通过。
- LF-8C（2026-09-12）：精筛同时要求父片段为 normal、主说话人簇、ASR 均值不低于 0.78，并检查重切后音频的 SNR、静音、削波、空间声场和密集背景风险；每源首轮最多 40 条，额外合格项进入候补而非删除。`preprocess_long_form.py` 默认在 ASR 后批量执行精筛并生成审核页，可用 `--skip-refinement` 显式跳过。审核页显示首轮/候补/排除数量及停顿压缩标记；35 项长音频回归测试通过。
- LF-8D（2026-09-12）：真实样本由 158 个宽松候选细化为 219 个语义单元，其中首轮审核 40、候补 12、自动排除 167；首轮总时长 2.54 分钟、单条 1.52–8.57 秒，15 条压缩了句内长停顿。40/40 WAV 均为 48 kHz 单声道 PCM24，文件哈希、合成帧数、源 `source_spans`、时长及页面数量一致，无 `.partial`、无残留自动质量风险；全项目 236/236 测试及 `compileall` 通过。
- LF-9（2026-09-12）：完整导入 40 条人工决定（保留 32、排除 8），默认 normal 集进一步隔离 1 条人工标记 whisper，最终导出 31 条、117.768 秒。新增版本化 `long_form_export.py`、`export_suisui_v1.yaml` 与 CLI；产物含 48 kHz 单声道 PCM24、`manifest.json`、`trainer.jsonl`、`catalog_candidates.jsonl`、排除审计、参考绑定和全量 `checksums.txt`。精确音频/文本均无重复，幂等重跑返回 cached。
- LF-10（2026-09-12）：完成 33 分 24 秒真实源文件的端到端运行；流式扫描固定为最多 30 秒解码块，原文件不变。最终导出 manifest SHA-256 为 `78d82ee46e2aaf6bb095a4e910ea6d9a6bd99d2526864444faa6c082675ef359`，checksums SHA-256 为 `8533a4c74e6a6c6362e34404a5343de30bba7e89618ef298b95654d8175055c8`；全项目 237/237 测试及 `compileall` 通过。
- LF-12A（部分，2026-09-13）：新增标注规范 `docs/long-form-labeling-v1.md`（发声/角色、空间、事件三类标签、关键缺陷清单、参考包确认与拆分纪律）；新增 `src/dots_tts_lab/long_form_reference.py` 参考包契约（6–12 条、每条 3–10 秒、绑定音频/源区间/审核批哈希，confirmed 包必须覆盖同演员角色、有字耳语、近讲双耳、夹拍击四类难负例）；新增 `scripts/build_long_form_reference_pack.py`（propose/freeze 两步）。已用岁岁 v1 导出生成 12 条草案 `data/reports/long_form/reference_pack_proposal_v1.json`（status=draft_pending_confirmation，逐条校验磁盘音频哈希），并生成确认页 `data/reports/long_form/reference_pack_review_v1.html`（每条可确认为 normal 参考、标为四类难负例或排除，导出 freeze 契约 JSON）；新增 `src/dots_tts_lab/long_form_splits.py` 与 `configs/lab/long_form/split_v1.yaml`，7 源整源拆分：岁岁已审源为参考集，按 sha256 排序前 4 源（15/2c/55/98）为开发集、后 2 源（d4/ff）为留出集；`check_split_isolation` 拒绝相邻切片、raw/DFN3 对、重复文本跨拆分。**已解冻（2026-09-13）**：人工确认 8 条参考（余 4 条排除）+ 12 条难负例（角色音×2、耳语×1、夹拍击×3、近讲双耳×6——首批双耳提名未获确认，放宽池并排除已审后补 8 条选出 6 条；other_speaker_normal 池为空故 0 条，非必需项）。`configs/lab/long_form/reference_pack_suisui_v1.yaml` 已冻结，pack_sha256 `0ea3fc6765bec50b7d0e2aa2076ca1421b68a0383dcee165e44292ab5cbe460a`，loader 校验内嵌哈希一致性。审核工具链：`scripts/serve_long_form_review.py` 本地审核服务器（127.0.0.1:8899），入口页列出审核页与已收导出，页面经 HTTP 打开时导出直接 POST 写入 `data/reports/long_form/incoming/` 固定路径（`reference-*` 白名单、原子写入），file:// 打开回退浏览器下载；负例侦察脚本支持 `--only-kind`/`--exclude`/`--save-name` 增量补选。
- LF-12B（2026-09-13）：新增 `src/dots_tts_lab/long_form_strict_gate.py` 与 `configs/lab/long_form/strict_gate_v1.yaml`。G0–G9 逐项 `pass/fail/unknown` 判定：任一 fail → reject；任一 unknown 或证据缺失 → quarantine；未校准评估器在生产模式不能 pass；G9 偏域/认证缺失 → quarantine；内容全过但联合认证未完成 → strict_candidate；认证通过 → auto_verified；calibration 模式只产 calibration_sample。`apply_batch_budget` 执行 24/6 双上限，超额转 reserve_budget；`verify_approval_binding` 逐字段比对，音频/区间/路由/gate 配置/参考包/校准报告/文本任一变化均阻断旧批准迁移。
- LF-13A（2026-09-13）：新增 `configs/lab/long_form/deepfilternet3_v2.yaml` 生产配置（独立于 `denoise_ab_v1.yaml`）与 `src/dots_tts_lab/long_form_dfn3.py` 严格契约：锁定 0.5.6/DeepFilterNet3、8 dB 衰减、post_filter=false、compensate_delay=true；核心块 ≤30 秒、上下文 ≤2 秒、逐窗重置、跨缝句重处理或隔离；先空间 gate 再下混；输出 `data/work/long_form/enhancement/dfn3/<run_hash>` 结构；失败策略只能 block_candidate/abort_run，契约层面无法表达"静默回退 raw 伪称降噪"。`scripts/verify_dfn3_backend.py` 实跑验证：config.ini 与 model_120.ckpt.best 哈希与文档指纹一致；容器内复验通过（deepfilternet 0.5.6 可导入、image `dotstts-long-form-denoise:deepfilternet-0.5.6` / ID `9d499f041e9f`），报告写入 `data/reports/long_form/dfn3/backend_verification.json`。
- 验证（2026-09-13）：新增 40 项测试；全项目 303/303 通过（31.8 秒），`compileall -q src scripts tests` 通过。
- LF-12C（2026-09-13）：新增 `src/dots_tts_lab/long_form_prefilter.py` 与 `configs/lab/long_form/prefilter_v1.yaml`。廉价预筛：Silero VAD 6.2.0（ONNX、512 样本窗、16 kHz，状态跨块连续、跨源显式 reset、模型实例跨源复用）产出连续语音区域（滞回门限 + 0.35 s 间隙合并 + ≤34 s 谷值切分，与 DFN3 30s+2s 窗口对齐）；原始 L/R 以 0.25 s 窗统计声道电平差/相关性/pan/side-mid 得空间风险占比（下混前证据）；16 k 单声道峰值能量包络正微分做粗瞬态密度（非事件分类器，只标记密度供后续事件 gate）。区域处置 `usable`/`prefilter_quarantine`（spatial_risk_region/transient_dense_region），预筛不产生训练候选。按源原子写 `regions.json`+`state.json`（config_sha256+impl_version 绑定），已完成源续跑跳过、失败源不冒充完成；`scripts/run_long_form_prefilter.py` 多源入口，启动时清理 `.partial`。silero-vad==6.2.0/onnxruntime==1.29.0 已加入 pyproject 依赖与 constraints 钉版。
- LF-13B（2026-09-13）：新增 `src/dots_tts_lab/long_form_dfn3_regions.py` 与 `scripts/run_dfn3_regions.py`（容器内入口）。`plan_windows` 把区域切成 ≤30 s 连续平铺核心、两侧各 ≤2 s 真实上下文（文件边界夹紧、不零填充）；`core_output_bounds` 在延迟补偿后的上下文长度输出中精确切回核心；`classify_sentence_spans`/`sentence_window` 落实跨缝句整句重处理、超长句隔离、禁止拼裁残词；`verify_delay_alignment` 用脉冲探针实测残差延迟（容器入口校验 ±2 样本）。`enhance_source_regions` 逐区域原子提交（.partial→rename）、区域级 wav+records/state 落盘、`completed_regions` 断点续跑只补缺、失败区域记 failed 并中止该源剩余、空间证据缺失的立体声区 fail-closed 阻断（safe_mean_only 下混需显式 gate 通过）；`make_df_enhancer` 内 init_df 一次加载跨窗复用、每次 enhance 调用天然逐窗重置。
- 验证（2026-09-13）：新增 25 项测试；全项目 328/328 通过（22.5 秒），compileall 通过，ruff check 通过（仓库未做 ruff format 基线，新增文件维持现状）；真实 SileroOnnxVAD 端到端跑通。注：全量命令为仓库根目录 `python -m unittest discover -s tests`（tests/ 非包、同级 import 依赖 discover 注入 sys.path）。
- LF-12D（2026-09-13）：新增 `src/dots_tts_lab/long_form_identity.py`、`configs/lab/long_form/identity_v1.yaml`、`scripts/run_long_form_identity.py`。对每区域按 1.5 s 窗 / 0.75 s hop 用钉死 CAM++（与 training_source_v1 同权重）嵌窗：能量门限以下不嵌入；相邻窗余弦低于转折阈值切 turn、短 turn 按原型相似回并；turn 原型贪心匹配产出**片内一致**的 speaker_NNN 标签（不跨文件同义）。每区域产出 G2/G3 证据：turn 数、speaker 标签、目标参考 cosine（max-per-window 的 median/min/p10）、按难负例类别取最大 cosine、相邻窗最小余弦、dominant_turn_share、status ∈ single_speaker_candidate/speaker_change_suspect/insufficient_speech——只产证据不判 pass，阈值标定前 G3 保持 unknown。参考向量 `load_reference_vectors` 逐条校验冻结包音频哈希（包 sha 不符即中止）。岁岁源真实冒烟：311 区域 8 秒；294 insufficient_speech（区域 0.3–5.8 s，单窗无法证稳定）、15 single、2 speaker_change_suspect；speaker_000–002。**校准警示**：参考片互余弦 0.479–0.849（中位 0.715），same_actor_roleplay 负例对参考最高 0.747——正负区间重叠，证实单 CAM++ 余弦不能区分同演员正常/角色音，身份判定必须等域内标定+风格 gate。
- 严格批次人工闭环（2026-09-13）：岁岁源 153 句全部人工审核——86 确认 / 67 拒绝（确认中位 2.83s vs 拒绝 0.86s，时长分离清晰）；校订页对 86 条逐条出 text_final（30 条手改、56 条维持、51 条 ASR 分歧全部人工消解），导出落 `incoming/` 并绑定 batch_sha256。`scripts/export_strict_batch.py` 终导出 `exports/strict_v1/`：**86 条 / 255.4s / 4.26 分钟**，逐条边界修复（零交叉+淡入淡出，首尾归零验证）、音频 sha256、确认文本 sha256、决策工件 sha 全绑定，decision=human_confirmed。注意：导出全部走 raw 路由（usable 孤岛多被 DFN3 路由但确认集主要来自跨区句）；warnings 字段标注 81 条 crosses_prefilter_quarantine、51 条 asr_disagreement_human_resolved、1 条 spatial_blocked_region_raw_downmix（region 52，人工听过 mono 预览确认）——这些人工确认只对本批有效，不迁移。
- LF-12H（2026-09-13）：新增 `src/dots_tts_lab/long_form_batch.py`、`scripts/build_strict_review.py`、4 项测试。`assemble_sentence_candidates` 把 transcript 句 span 与 routes/identity/style_event 三层证据按包含区域 join，逐句产 G0–G8 GateVerdict（当前评估器全 uncalibrated → 只能出 fail/unknown；covers_quarantine→G6 fail、非 candidate span→G7 fail、换人嫌疑→G2 fail、空间风险→G5 fail），`evaluate_candidate` 汇总 disposition，排序（双ASR一致→时长在域→事件风险升序）+ 文本去重 + `apply_batch_budget`（24/6）。审核页 `strict_review_<sha>.html`：raw/路由成品音频并排、主副 ASR 文本对照、fail/unknown gate 徽标、确认/拒绝/灰区三选，灰区默认隐藏，导出 POST 至 `incoming/strict-review-<sha12>.json`（服务器白名单已扩 `strict-review-` 前缀）。岁岁源真实汇总：153 条去重句 → strict_candidate 0（全部 fail/unknown → quarantine/reject），与证据链一致。transcript 记录补 `secondary_text` 字段（impl v4）供文本差异展示。
- LF-12G（2026-09-13）：新增 `src/dots_tts_lab/long_form_quality.py`、`configs/lab/long_form/quality_v1.yaml`、`scripts/run_long_form_quality.py`、4 项测试。DNSMOS P.835（sig_bak_ovr.onnx + model_v8.onnx，官方 DNS-Challenge 模型 sha256 钉死，onnxruntime CPU）按上游 dnsmos_local.py 精确复刻：16k 重采样、<9.01s 自拼接填充、1s hop 逐窗均值、非个性化三次多项式校准；`padded`/`short_clip` 标记不隐藏。逐区域对照 routed 输出 vs 同源区间 raw，输出 sig/bak/ovr/p808 增量并联事件风险分（区分背景降噪收益 vs 拟音损伤：BAK↑SIG↓+高事件风险=增强伤内容）。岁岁源真实验收：79 个 DFN3 区域 SIG 中位 +0.20 / BAK +0.40 / OVR +0.13；4 区域 OVR 回归且全部事件风险 >1.0（region 148/187/205/218）——拟音损伤证据成立。**关键限制**：全部区域 <9s → 100% padded 自拼接，分数方向有效、幅度不可作标定依据；`thresholds.calibrated=false` 保持。插曲：nisqa pip 包依赖过旧曾降级 torch 2.8→2.2.1+cpu，已恢复 cu128 并弃用 nisqa 改 DNSMOS-ONNX（onnxruntime 已钉版）。
- LF-13C（2026-09-13）：新增 `src/dots_tts_lab/long_form_route.py`、`configs/lab/long_form/route_v1.yaml`、`scripts/run_long_form_route.py`、8 项测试。路由规则：非 usable 区域不路由；立体声无空间下混许可 → blocked_spatial；稳态噪声分（style_event 新增 `noise_label_patterns`/`noise_scores`：hiss/static/hum/buzz/rumble/noise/wind/air conditioning，独立于污染风险组）≥0.30 → dfn3_denoised，其余 raw；噪声证据缺失默认 raw。raw 直接切源（mono 或 safe_mean），dfn3 委派 LF-13B 内核、失败即中止无回退。逐区域原子写 + routes.json 统一清单 + state.json 续跑。修了三处真 DFN3 路径 bug（init_df 需模型目录非 cache_root、enhance 需 [C,T] 张量、官方 config.ini 依赖 upstream 默认项 → config_allow_defaults=True），DFN3_REGION_IMPLEMENTATION_VERSION→2。岁岁源容器真实跑：311 区域 → 79 dfn3_denoised / 0 raw / 21 blocked_spatial / 211 skipped，15 分钟，产物哈希+帧数抽样全对。**注意**：usable 区 noise_max 中位 0.846（buzz 0.77/hum 0.72 在语音上虚高），阈值 0.30 致全部 usable 走 DFN3——该阈值未标定，LF-12G 须在开发集重定。
- LF-12F（2026-09-13）：新增 `src/dots_tts_lab/long_form_transcript.py`、`configs/lab/long_form/transcript_v1.yaml`、`scripts/run_long_form_transcript.py`、13 项测试。设计：faster-whisper（large-v3-turbo，词级时间戳，独立 env）做主转写+界标，SenseVoice Small 对每个最终句 span 独立复述；规范化文本须逐字一致方可候选，数字/拉丁/专名分歧单列状态；句界由词时间戳构成——先累积至 ≥3s 再按句末标点/停顿切，>12s 句在内部最深停顿处裂解；边界严格落在词起止为 clean，插入词内为 dirty；覆盖预筛隔离区的句标 `overlaps_quarantine` 并降为 `covers_quarantine`；`transcribe_quarantined` 开启时全部语音区域（含隔离区——隔离非拒绝，证据仍需）参与转写合并。ASR 输出一律 `candidate_text_is_confirmed: false` + `requires_final_recheck: true`，不视为真值。岁岁源真实冒烟：137 转写单元 → 166 句 span，50 条双 ASR match、96 mismatch、20 coverage_gap；56 条落 3–12s 区间但**全部 covers_quarantine**（usable 孤岛最大 1.25s，≥3s 句必跨隔离段）→ 候选 0——对该事件密集源为诚实结果，亦提示预筛瞬态阈值对耳语内容可能偏严，属待标定项。
- LF-12E（2026-09-13）：新增 `src/dots_tts_lab/long_form_style_event.py`、`configs/lab/long_form/style_event_v1.yaml`、`scripts/run_long_form_style_event.py`。①风格证据：9 维声学剖面（电平/静音比/谱平坦/周期性/基频/side-mid/pan 稳定度/声道相关/瞬态密度）对冻结包做分布距离——`load_style_anchors` 逐条校验参考音频哈希后提取 normal 与每类难负例锚点，`score_style_evidence` 输出 normal_mean_z/max_z/nearest_anchor 与按类别 negative 距离（全 NaN 特征列按缺失罚分处理）；只产证据不判 pass。②事件证据：接入 PANNs CNN14 AudioSet（`panns-inference==0.1.1`，CPU；标签 csv 与 327MB 权重 sha256 钉死在配置，漂移即中止；upstream 硬编码 `~/panns_data/` 路径，已注明）；28 个风险类别子串匹配（male/female/child speech、hubbub、whisper、singing、music、laugh、crying、slap、clap、bang、rustle、scratch、rub 等）逐窗取最大分+命中时间；短区域向两侧借真实上下文至 ≥1 s 再打标并标记 context_padded，不零填充；后端不可用时 `NoEventBackend` 让所有区域 evidence_missing → gate 保持 unknown。③空间证据透传 prefilter 的 L/R 统计。岁岁源真实冒烟：311 区域 63 秒、全部 evidence_complete；usable 区 normal_mean_z 中位 4.9（锚点仅 8 条 MAD 极小→z 膨胀，待标定）；nearest_negative_kind 全为 nearfield_binaural（双耳片风格近正常，区分靠空间证据）；usable 区事件风险高发（rub 100/100、music 99、clap 84、whisper 45）证实内容事件密集——未标定前仅作证据。新增依赖 panns-inference/torchlibrosa/matplotlib 已钉版。

## 已生成数据状态（严格 gate 上线前）

首个源 `岁岁/林三歲-姐姐的憋尿野營.mp3` 已完成端到端处理，最终数据片段位于 `data/work/long_form/exports/suisui/v1/`，含 31 条、约 1.96 分钟 normal 训练音频。

追加批次（2026-09-12）新增 6 个源、约 4.75 小时，全部完成流式扫描、宽松切分、embedding、词级 ASR、句级重切和自动精筛，无失败或中断：

- 宽松候选 1,683 条，重切后形成 2,057 个语义单元。
- 自动排除 1,704 条，候补 148 条，首轮人工审核 205 条、共 824.969 秒（13.75 分钟）。
- 各源首轮条数为 15、30、40、40、40、40；没有为了凑满 40 条而放宽质量阈值。
- 批量审核总览：`data/reports/long_form/review_index.html`。旧源显示已完成，新 6 源显示待审核。

## 背景降噪 AB

- LF-10A（已完成，2026-09-12）：新增独立 DeepFilterNet3 保守降噪盲听 AB 实现。固定 8 dB 最大衰减、关闭额外 post-filter，不覆盖原始候选；20 条样本按高底噪/中底噪/干净对照分层并跨源轮选。新增 3 项测试，并与相关回归共 10/10 通过，编译检查通过。
- LF-10B（已完成，2026-09-12）：隔离镜像及 DeepFilterNet3 固定权重已就绪，0.5.6 API 兼容回归 4/4 通过；20 组盲听音频已生成。40 个文件均通过 PCM24/48 kHz/时长/哈希验证；CAM++ 原音/增强音余弦最低 0.9832、中位 0.9901；faster-whisper 14/20 完全一致，总字符变化率 6.49%。高/中底噪 16 条全部降低噪声底，平均约 -5.35 dB；干净对照证明不应全量降噪。全项目 242/242 测试与编译检查通过。当时状态为 `awaiting_human_review`；后续 LF-10C 已记录人工否决，技术通过不代表允许接入导出。
- LF-10C（人工结论，2026-09-12）：DeepFilterNet3 仅能减弱稳定底噪，不能可靠移除拍击等前景拟音；20 组主观听感仍不可接受。多数候选又存在非零波形硬截断，首尾缺少零交叉/de-click/fade，降噪后边界滋声更明显。DeepFilterNet AB 保留为技术审计，不接入训练数据或正式导出。

## LF-11：UVR5 历史尝试与已完成的独立修复

最新状态（2026-09-12）：用户报告 UVR5 失败，明确改用 DeepFilterNet3。UVR 分离路线停止排期；失败原因未经本项目诊断，不推断为具体模型或硬件故障。LF-11A/B/C/D/F 的 UVR 工作由 LF-13 替代，已完成的通用可靠性与边界修复继续复用。

UVR 等待期间的独立修复（2026-09-12，已完成）：

- [x] LF-11R1：人工事件绑定 source/config/manifest 与片段音频哈希；拒绝跨版本继承批准。旧日志仅能由匹配当前清单的原快照提供绑定，保留日志原字节和已导出的 v1 数据。
- [x] LF-11R2：精筛不再累计候选 PCM，首遍仅保留特征，入选后逐条重读；64 条合成候选回归验证没有数组累积。
- [x] LF-11R3：ASR 区分 succeeded/partial/failed；保留逐条成功结果和已生成音频，失败或中断后只补缺失项。未完成源跳过精筛并在 CLI 明确返回失败状态。
- [x] LF-11R4：所有主要输出入口在创建、恢复、写入前验证实际路径；拒绝输出与 inbox/输入重叠，拒绝 Windows 盘符绕过配置相对路径约束。
- [x] LF-11R5：运行缓存绑定源、配置、实现版本及 ASR 配置文件哈希；精筛绑定父清单和 ASR 结果哈希。配置变化、实现升级和强制重跑均保留旧版本，不能覆盖已审核清单。
- [x] LF-11R6：默认启用版本化 `refinement_v2.yaml`，新候选及导出执行 2–15 秒契约；v1 的 1.5 秒历史配置和首批 31 条已验收片段保持不变。

验证：62 项长音频专项、263 项全项目测试通过；`compileall`、`pip check` 通过。合成 WAV 已贯通边界渲染、人工事件导入和导出验证；首批 31 条导出全部校验和通过，原 manifest/checksums/config 哈希保持一致。详细记录：`docs/reports/long-form-reliability-v2.md`。

- LF-11A（失败，用户报告）：UVR5 分离未成功，当前不重试此路线。
- LF-11B（已替代）：原计划验证 Vocal stem，改由 LF-13B/C 验证 DFN3 区域音频、时间映射和产物哈希。
- LF-11C（已替代）：原计划从 Vocal stem 重切，改由 LF-13C 接入 raw/DFN3 并受 LF-12 gate 约束。
- LF-11D（已替代）：原计划 raw/uvr_vocal 双来源，改为 raw/dfn3_denoised 双来源；残留拟音、音乐、其他人声或增强损伤仍不准入。
- LF-11E（代码完成，真实 DFN3 成品验证纳入 LF-13D）：新增版本化边界渲染，读取目标句前后 180 ms 上下文，在边界附近 5 ms 内寻找零交叉并裁回，首尾做 25 ms 淡入淡出，句内源区间接静音处做 15 ms de-click；保留原有 80 ms 连接停顿。实际源采样区间、上下文、淡化长度和重采样前输出映射写入候选及导出；48 kHz 最终文件首尾为零。合成音频验证通过，未重建或审核旧 205 条原音候选。
- LF-11F（已替代）：原计划 raw/UVR 盲听，改由 LF-13D 的有限 raw/DFN3 试点验收；通用人声完整性、事件残留与音色要求保留。
- LF-13D（2026-09-13）：`scripts/build_dfn3_ab_review.py` 句子级三列对照页（原音/8dB/12dB）。12 句选自 transcript 3–12s 真实文本句并按覆盖区域证据分层（4 干净/4 稳态底噪/4 难负例）；DFN3 走 sentence-window 等价路径（整句核心+真实上下文、延迟补偿残差 0），8dB 与 12dB 两臂分别独立 work_root 实跑（各 96–98s、零 blocked）。验收判决：8dB 难负例全 tie（DNSMOS OVR 回归不构成可听损伤）、稳态底噪全 tie、干净组 2 better/1 tie/1 worse；人工结论「tie 源于源本身干净而非力度不足」→ **冻结 8dB、维持 raw 优先路由**，12dB dev 臂保留供审计不转生产（`configs/lab/long_form/deepfilternet3_dev12db.yaml`）。决定记录 `calibration/lf13d_decision_46d41e37fc4f.json`。
- LF-13E（进行中）：`scripts/run_lf13e_sources.py` 对岁岁其余 6 源顺序跑全管线（prefilter→style_event→identity→transcript→route→容器 DFN3→quality→strict review 页），逐源日志 `data/reports/long_form/lf13e_runs/<sha12>.log`、summary.json 汇总。

当前停止条件：LF-12/LF-13 产出小批严格候选后，未取得自动认证时停在人工确认；不自动开始训练。增强分支证据不足的片段隔离，干净原音可独立通过全部 gate 后继续。

原先按约 79% 保留率估算 12–14 分钟的产量不再用于排期；新标准可能显著减少产量。依据严格 gate 后实际合格分钟数、文本/声学多样性和审核结果，再决定补料与训练规模。

## LF-12：当前待办与依赖

以下任务全部待执行。本次只更新设计与文档，不把旧代理特征、模型配置存在或预算限流记作新 gate 已完成。

| 顺序 | 任务 | 依赖与完成条件 |
| --- | --- | --- |
| P0 | [x] LF-12A：标注规范、目标正常参考包与数据拆分 | 可立即执行。固定 6–12 条正常参考，定义角色/气声/耳语/普通换气/非语言事件；复用历史结论但重新核对新标签；参考、开发、留出按源/时间块隔离；新增标注分不超过 12 条的小批 |
| P0 | [x] LF-12B：严格 gate 契约与状态机 | 可立即执行。固定 G0–G9 的证据/版本/原因，`pass/fail/unknown`，缺失不放行；定义 strict_candidate、reserve_budget、quarantine、reject；测试高分不能抵消否决，旧批准不能迁移 |
| P1 | [x] LF-12C：多源廉价预筛与阶段缓存 | 神经 VAD + 原始 L/R + 粗事件/瞬态；分块保状态/跨文件重置；先筛后重计算；模型跨源复用；缓存按 stage 恢复，内存有界 |
| P1 | [x] LF-12D：单人时间轴与目标身份 | 常规 diarization/重叠检测、跨块标签衔接、固定 CAM++ 参考和局部稳定性；不以 exclusive 或最大簇放行；同片换人/两人重叠回归与真实难例验收 |
| P1 | [x] LF-12E：正常发声、角色、空间与事件 gate | normal 正例判别 + 同演员角色难负例；多标签事件时间轴 + 瞬态；空间证据来自原始 L/R，不能由 mono 覆盖；DFN3 未去除的拟音/音乐/其他人声必须隔离或排除 |
| P1 | [x] LF-12F：双 ASR、中文对齐与最终切片复检 | 可用原音开发。复用 Whisper/SenseVoice 配置，保留事件标签；选定一种中文 aligner；最初规范化文本完全一致，数字/专名/语气词分歧隔离；最终 3–12 秒自然句重新测各项，不继承父通过 |
| P1 | [x] LF-12G：音质校准与 DFN3 路由验收 | 使用 LF-13B/C 的 raw/dfn3_denoised 产物；比较 DNSMOS/NISQA 域内增益，验证短音频评分；区分背景与拟音；全程哈希和延迟/漂移映射，DFN3 路线单独验收 |
| P1 | [x] LF-12H：整批预算与精简审核页 | 先完成 A/B，可用 fixture 开发；真实启用依赖 C–G。联合 gate 后才排序、去重、多样性；当前 6 源总上限 24、每源 6；隔离区默认隐藏，灰区校准每批最多 6 且默认关闭；展示原音/成品、文本差异和局部风险 |
| P2 | [ ] LF-12I：冻结阈值并验证联合精度 | 依赖 A–H。开发集定阈值，独立留出验收；报告 precision/coverage、按源/路线的误收、有效分钟、人工分钟；小样本不得宣称 ≥99% 已认证，关键假阳性阻断发布 |
| P2 | [ ] LF-12J：长文件性能与恢复验收 | C–H 后与 LF-13F 合并执行，避免重复基准。30/60 分钟及现有 6 源记录 RTF、RSS/VRAM、模型加载、缓存命中；中断/OOM 只补缺失；单 GPU 重阶段串行调度 |
| P3 | [ ] LF-12K：认证后自动验证与抽检 | 仅在 I 达标后进入。新增 ADR/导出 schema，`auto_verified` 绑定认证与最终音频/文本；抽检关键失败即暂停路线、隔离受影响未冻结批次；不伪造人工确认、不自动启动训练 |

实施顺序：先 LF-12A/B/C/H 的独立部分与 LF-13A/B，再完成 LF-12D/E/F、LF-13C 和 LF-12G；LF-13D 试点验收后执行 LF-13E/F 与 LF-12I/J。联合质量与性能验收完成后再考虑 LF-12K。所有工作均不再依赖 UVR，不得绕过尚未成立的 gate 给旧 205 条补“自动通过”。

详细 gate：G0 来源文件、G1 有效语言、G2 单人时间稳定、G3 目标身份、G4 正常声线/空间、G5 事件/音质、G6 文本/对齐、G7 增强保真、G8 最终成品、G9 校准域/发布资格，见 [研究报告第 5 节](docs/reports/long-form-strict-gate-research-20260912.md#5-必须全部成立的-gate)。

## LF-13：DeepFilterNet3 接入（替代 UVR）

已完成路线选择与现有配置/模型缓存核对，以下生产任务仍待执行。具体参数、文件指纹、分块规则与验收见 [DFN3 补充方案](docs/reports/long-form-deepfilternet3-v2-plan.md)。旧 AB 不覆盖、不自动批准；本次选用后端无需再次确认。

| 顺序 | 任务 | 完成条件 |
| --- | --- | --- |
| P0 | [x] LF-13A：固定后端与新生产配置 | 复用独立 0.5.6 环境和已缓存 DeepFilterNet3，校验 config/权重哈希及 image ID；创建独立版本配置，8 dB/post-filter off/延迟补偿为试点起点；不改 v1 AB 配置或 TTS 主环境 |
| P1 | [x] LF-13B：带上下文的有界降噪适配 | 核心块最多 30 秒、前后各最多 2 秒真实上下文为待验证起点；每窗明确重置状态、模型跨窗复用；正确补偿延迟、映射采样范围；跨缝句子完整重处理或隔离，不拼裁残词 |
| P1 | [x] LF-13C：raw/DFN3 路由、恢复与 pipeline 交接 | 原始 L/R 先判空间；干净原音优先，需降噪区域只处理一次；逐区域原子提交/哈希缓存/失败续跑，新输出目录；失败不得静默回退并伪称降噪通过；进入最终切片复检 |
| P1 | [x] LF-13D：不超过 12 组的新版对照验收 | 4 干净、4 稳定底噪、4 难负例，跨源；比较原音/DFN3 的尾音、事件、音色、文本和边界。8 dB 不足时仅在开发集增加 12 dB 对照并冻结选择；不过就淘汰，不反复增强 |
| P2 | [ ] LF-13E：新 6 源接入严格筛选 | 依赖 LF-12C–H 与 LF-13D；多文件恢复处理，只输出全部内容 gate 通过的候选；总送审上限 24、每源 6，灰区隔离；报告实际有效分钟，不恢复旧 205 条大队列 |
| P2 | [ ] LF-13F：回归、资源与可重建验收 | 与 LF-12J 共用基准；覆盖 30/60 分钟、跨块、中断、非 48 kHz、文件边缘、错误输出、缓存失效、旧产物保护；检查首中尾对齐与内存有界；保留 gate 精度与人工预算报告，不自动训练 |

## 完成标准

- 多个长文件可顺序或受控并发处理；中断后可从已完成阶段恢复。
- 峰值内存不随源文件总时长线性增长。
- 每个候选片段都可追溯回源文件精确时间范围。
- 必须检测目标身份、正常发声、原始空间、重叠、事件、文本和最终音质；任何必要证据缺失或不满足均隔离/排除，分数不互相抵消。
- 新 6 源试点默认送审不超过 24 条，单源不超过 6 条，超额合格项与灰区分开统计；记录实际人工分钟和合格产量，不能把限流当作准确率提升。
- 联合 gate 有按源/路线拆分的开发与独立验收证据；自动发布前达到精度目标和关键缺陷门槛，未达到就保持小批人工确认。
- 人工确认或有效自动验证结论绑定同一最终音频、文本与配置，可确定性重建；任何模型/参考/音频/文本变化使相关旧结论失效。
