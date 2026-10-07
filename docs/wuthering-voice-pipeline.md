# 鸣潮角色语音 → TTS 数据集流水线

以守岸人（v2，24.1min）和锁暝（270 条，24.5min）两轮实际跑通的流程为准。
角色 exporter：`scripts/export_shouanren_voice.py`、`scripts/export_suoming_voice.py`（后者同时覆盖好感语音 + 剧情对话两条来源）。

## 0. 跳过预处理约定

用户要求「跳过预处理」时：导出的 WAV 直接入库，不跑降噪/增强（DeepFilterNet、noisereduce 一律不用）。
注意 pipeline 里的 `standardize` 阶段不是预处理——它只做响度/格式归一化（切片、采样率、LUFS），属于管线固有步骤，正常执行。

## 1. 角色标识定位

三条线索交叉确认：

- `WutheringData/ConfigDB/lang_speaker.db` → `Speaker_<id>_Name`，锁暝 = 400073 / 400074（双形态）
- `roleinfolang` / `role_lang_<name>` → 角色语音包注册名（守岸人 = shouanren / RoleId 1505；锁暝没有注册条目）
- `favorword` BinData → `FavorWord_<RoleId><idx>_Content`，锁暝好感语音 RoleId = **1312**
  （1311 是别的角色，判定标准：谁的 Wwise 事件名出现在谁的条目里——`play_favor_word_suoming_*`）

## 2. 语音来源一：好感/战斗语音（favor + vo/role 事件）

### 解包工具链

- **FModelCLI**：`E:\project\Ludiglot\tools\FModelCLI.exe`，用法：
  - 提取：`FModelCLI.exe <GameDir> <AES> <OutputDir> [Filter]`（GameDir = `D:\game\Wuthering Waves\Wuthering Waves Game`；AES 存 `Ludiglot/cache/aes_archive.md`）
  - 列文件：`FModelCLI.exe <GameDir> <AES> --list <Filter>`
  - 每次调用都全量挂 pak（~250 万文件，~10s），Filter 按路径子串匹配；单 media id 提取每次 ~2s
  - 输出目录还原原始路径：bnk → `Client/Content/Aki/WwiseAudio_Generated/Event/<event>.bnk`；wem → `.../Media/<id>.wem`；剧情外部源 → `.../WwiseExternalSource/zh_vo_<key>.wem`
- **vgmstream**：`E:\project\Ludiglot\tools\.data\vgmstream-cli.exe -o <out.wav> <in.wem>`（游戏 wem 是 opus 编码）
- **media id 全集**：`--list "WwiseAudio"` 输出 `[File]` 行 → 正则 `Media/(\d+)\.wem` 收集存 `E:\project\temp\known_media_ids.txt`（锁暝一轮 109 个，全库几十万行；`[File]` 行不按 pak 分组，**不要按 pakchunk 号猜**）
- **bnk→media id**：`.bnk` 文件 4 字节 LE 扫描，与 known_media 集合取交集（`media_ids_in_bnk`）

### 提取步骤

1. `WutheringDialog` 的 `voice_map_v6.json` 或 FModel CLI 搜游戏 pak：
   `FModelCLI.exe <GameDir> <AES> <out> play_favor_word_suoming_`（前缀还有 `play_vo_suoming_`、`play_role_suoming_`）
2. `.bnk` 文件解出 media id 集合 → 全部落在某个 `pakchunkNN`（锁暝 = pakchunk43，109 个 media）
3. 逐 id `FModelCLI <GameDir> <AES> <MEDIA_OUT> <media_id>` 提取 `.wem` → vgmstream 解码 WAV
4. 文本从 `lang_multi_text.db` 取（注意：`lang_favor.db` 同键可能返回空值，要 fallback 到 multi_text）

## 3. 语音来源二：剧情对话（FlowState + PlotAudio）★ 本轮新增

好感语音只是冰山一角——锁暝 favor 只有 71 条，剧情对话多出 199 条。

0. **ConfigDB 快照**：`FModelCLI <GameDir> <AES> <out> ConfigDB` 整包解出（锁暝一轮解到 `E:\project\temp\suoming_cfg`）；新版文本/剧情库全是 sqlite `.db`（`db_flowState.db`、`db_plot_audio.db`、`db_ShippingHiddenMultiText.db`、`<lang>/lang_multi_text.db`）
1. **FlowState.json**（旧版 `WutheringData/ConfigDB/FlowState.json`；新版 `db_flowState.db` flowstate.BinData）：protobuf-ish 格式，
   但 blob 里嵌着明文 JSON 片段，直接搜 `"WhoId":<speaker_id>` + `"TalkItems"` 就能拿到对话 key 列表。
   锁暝 WhoId = **400074**（不是 400073），命中 327 个唯一 key。
2. **PlotAudio.json**（同目录）：`dialog_key → vo_事件名` 映射。327 个 key 里 199 个有映射；
   **剩下 128 个没映射的是无配音台词（回忆/旁白/心声），属预期，不是提取失败**。
3. **音频路径**：映射到的事件对应的 media 是外部源，路径固定为
   `WwiseExternalSource/zh_vo_<dialog_key>.wem`（`E:\project\Ludiglot\data\WwiseAudio_Generated\WwiseExternalSource`），
   同样 vgmstream 解码。

## 4. 导出与命名

- 输出：`data/inbox/<角色名>/<情绪>/【<情绪>】<台词文本>.wav`——**文件名即 transcript，不跑 ASR**
- 文本含 `/\:*?"<>|` 等非法字符需清洗；重名（空文本/标点被洗光）追加 `_<key尾号>` 防碰撞
- 导出报告：`E:\project\temp\<role>_export_report.json`（成功/失败/key 对照）

锁暝结果：270/398 成功 = 71 favor + 199 剧情，128 个失败全部是无配音条目。

## 5. Pipeline 阶段（dots.tts.lab CLI）

```bat
:: 1) 入库（幂等，重复自动去重）
dots.tts.lab.exe ingest --config configs/lab/ingest/ww_voice_v1.yaml --source-root "data/inbox/锁暝"

:: 2) 质量门（v2 游戏语音策略）
dots.tts.lab.exe analyze --config configs/lab/quality/signal_analysis_v1.yaml --policy configs/lab/quality/training_source_review_v2.yaml --report-dir data/reports/quality_<role>_v2

:: 3) 标准化（响度/格式归一化，不是降噪）
dots.tts.lab.exe standardize --config configs/lab/standardize/training_audio_v1.yaml --asset-manifest <manifest.json>

:: 4) 说话人聚类（保留最大簇 = 角色本音，剔除 NPC 混录/短喝）
python -X utf8 scripts/filter_shouanren_speaker.py --speaker-id 锁暝

:: 5) 人工评审页 → 6) 冻结数据集 → 7) LoRA 训练 → 8) 注册音色档案
```

锁暝 269 条入库后：质量门 186 pass / 83 review / 0 reject；聚类 39 簇、keep 214 + 救回 2 = 216 候选。

## 6. 已知坑

- **当版本主线剧情文本整体遮蔽**：`Main_Mengzhou_3_7_*` 在 `lang_multi_text.db` 全部为 `*****`（`RedirectDbIndex=1` → `db_ShippingHiddenMultiText.db`，但该表只有 TextId→台词键映射，不存文本；重定向键 `ZYHSGST_*` 也全部遮蔽）。3_6 及更早文本正常。`*` 是文件名非法字符 → 清洗后空文件名 → `】_N.wav` 碰撞后缀 → transcript 只剩 `_N`。**修复：对遮蔽条目跑 SenseVoice ASR 补文本**（`data/reports/asr/suoming_masked_transcripts.json`），再做专名同音修正（谛天鉴/岁主/神宫/新月湖/梦枢天罗等）。ASR 文本无标点、专名需人工在审核时以耳为准。
- **参考音频（prompt）质量决定成败**：守岸人 R2 短 clip 让爱弥斯/守岸人输出全毁，换长句干净参考立刻解决——盲听验证。
- **谱平坦度不能当"干净"判据**：对讲机特效音频带窄→平坦度低，反而更脏。参考必须人工听。
- **聚类会吞掉 <1s 短喝**（散击！/三碎。类），嵌入不可靠，自动剔除属预期。
- 静音资产会卡死 embedding，筛选脚本已加 skip。
- **训练 batcher 会静默丢弃超长样本**：`max_audio_seconds_in_batch` 是按 batch 算的，但单条样本超限时会被 skip（RuntimeWarning）。锁暝有 10/128 条 >15.5s（最长 30.2s），配置需提到 ~32 否则丢 8% 数据。
- 浏览器缓存坑：替换音频必须改文件名（`_v2` 后缀）或加版本参数。
