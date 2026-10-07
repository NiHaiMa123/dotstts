# Local voice model registry

The local WebUI consumes accepted voice deployments. Training, checkpoint selection,
prompt selection, and post-processing tuning remain agent-run workflows; the WebUI
does not train or tune a model.

## Registry

`configs/voices/registry.yaml` defines the default model and the ordered list shown
by the WebUI. Every entry points to one strict voice profile. The server validates
all profiles and their bound artifact checksums before it opens the browser.

A profile binds every setting that can change the generated voice or final audio:

- official base model path and revision;
- trainable-delta/LoRA directory, selected training step, weights hash, and metadata
  hash;
- prompt WAV, its hash, and the exact prompt transcript;
- runtime precision, optimization mode, sequence limits, vocoder merge setting, and
  LoRA merge policy;
- language, template, text normalization, speaker scale, ODE method, step count,
  guidance, seed, sentence pause, and text split size;
- edge-trim and voice-polish configurations with file hashes;
- model-specific TXT input and final WAV output directories.

The selected profile is resolved once when a job starts. The state machine records
its version and canonical SHA-256. Switching models releases the previously loaded
runtime before loading the new base/adapter pair, so CUDA memory and settings cannot
leak from one voice to another.

## Adding a newly accepted LoRA

After the agent has finished training and post-processing review:

1. Freeze the chosen checkpoint, prompt WAV/transcript, inference parameters, and
   post-processing YAML. Do not rewrite historical Slice acceptance files.
2. Copy `configs/voices/fuxuan_step400_v1.yaml` to a new uniquely named profile and
   replace every model-specific field and checksum.
3. Give the model its own input and output directories so identically named TXT files
   cannot overwrite another voice's WAV files. Convention: TXT files for a model go
   in `inputs/<model>_text/` and generated WAVs land in `outputs/<model>_audio/`.
   Create both folders when the profile is created; the WebUI also creates any
   missing profile directories at startup as a safety net.
4. Add the profile path to `configs/voices/registry.yaml`; change
   `default_model_id` only when the new model has been explicitly accepted as the
   default.
5. Run `python -m unittest tests.test_voice_registry tests.test_fuxuan_webui` and
   then the complete test suite.
6. Restart the WebUI. The dropdown is populated from the registry automatically;
   HTML and Python source changes are not required.

If any artifact is missing or its bytes differ from the recorded hash, startup or
model loading fails explicitly instead of silently mixing a LoRA with another
voice's prompt or post-processing chain.

## 角色语音提取（鸣潮）

提取脚本：`scripts/export_shouanren_voice.py`（旧版数据）、`scripts/export_suoming_voice.py`（新版数据）。产物落到 `data/inbox/<角色>/<情绪>/【<情绪>】<原文>.wav`，文件名即文本，跳过 ASR 预处理。

工具链：Ludiglot 的 `tools/FModelCLI.exe`（解包，AES 缓存在 `Ludiglot/cache/aes_archive.md`）+ `tools/.data/vgmstream-cli.exe`（wem→wav）。

### 旧版数据结构（守岸人，2026-09 快照）

- `db_favor.db` favorword 表按 RoleId 出好感语音；BinData 含 `play_favor_word_<角色>_*` 事件名。
- `FlowState.json` 里 TalkItem.WhoId ↔ `Speaker_<WhoId>_Name`（lang_multi_text.db）定角色，`TidTalk` 为文本键；`PlotAudio.json` 把文本键映射到 `vo_<key>`。
- 角色语音包：整包 Media/*.wem 已解到 `E:\project\temp\<role>_pak`；bnk→media id 用 `known_media` 集合交集（4 字节扫描）。

### 新版数据结构（锁暝，游戏现版本）

- FlowState.json / PlotAudio.json 取消，改为 `db_flowState.db`（BinData 尾部嵌 JSON Actions，正则提取 `"WhoId":<id>` + `"TidTalk"`）和 `db_plot_audio.db`（BinData 内含 `vo_<key>` 文件名）。
- lang 文本：好感语音文本 `FavorWord_<fid%100+RoleId*100>_Content` 在 `lang_multi_text.db`（注意新版 fid 是 7 位，文本键是 6 位；`lang_favor.db` 有同键空值，load_textmap 要跳过空串）。
- 角色语音 pak：bnk 里 4 字节扫描的 media id 与全量 Media id 表（`E:\project\temp\known_media_ids.txt`，由 `--list "WwiseAudio"` 生成）取交集，再逐 id `FModelCLI <GameDir> <AES> <out> <id>` 提取（每次 ~2s）。**不要**按 pakchunk 号猜——`[File]` 行不按 pak 分组。
- 剧情语音：`zh_vo_<TidTalk>.wem` 走 `WwiseExternalSource` 路径逐 key 提取。
- 文件名须清空白/换行并截断到 80 字符，否则长台词写盘失败。

### 角色身份定位

- `lang_multi_text.db` `Speaker_<id>_Name = <角色名>` → TalkItem WhoId。
- `db_favor.db` favorword BinData 含 `play_favor_word_<拼音>_*` → RoleId。
- `db_audio.db` roleinfolang 的 `role_lang_<拼音>` 字段可交叉验证（未实装角色没有此项）。
