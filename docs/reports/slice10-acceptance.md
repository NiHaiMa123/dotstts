# Slice 10 stop-gate report：模型闭环参考排行

- 状态：已验收（人工 gate 已完成）
- benchmark：`fuxuan_slice10_reference_rank@1`
- benchmark manifest：`data/reports/datasets/fuxuan_v1/slice10/benchmark_manifest.jsonl`
- manifest SHA-256：`9f3db2bc86a55b5ef8daf8daf5ded693b33ef4e9498aa883e4196e7bf849a762`
- 生成时执行配置 SHA-256：`fe98beddb318eb16f6a674f123cbcee583984c8eca27b87c089b3a85e9ae6cb8`
  （后续只增加人工 gate 输出字段，原始执行哈希保持钉扎）

## 已完成的自动部分

- 30 个 Slice 9 冻结、去重且 approved 的参考；4 条固定中文句；2 个固定 seed。
- TTS：`dots-studio/dots.tts-mf-1step`，本地 config SHA-256
  `f3388119ec80a559013092e1875d9df8cda812bd4517476b5fc744d4e05c7785`，模型权重、speaker
  encoder、vocoder 哈希已写入 benchmark config。
- 生成矩阵：240/240 成功，0 generation error；每项都有音频和 SHA-256，平均 RTF 约 0.3631。
- ASR：固定 faster-whisper large-v3-turbo revision
  `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`；240/240 转写成功。
- 自动指标：平均 CER `0.072697`，平均 CAM++ center cosine `0.837927`，无 metric error。
- 句子 CER：`neutral_report 0.0563`、`named_entity 0.0021`、`long_decision 0.0026`、
  `date_number 0.2298`。
- 音质策略：112/240 pass、128/240 review；128 条全部因 `long_leading_silence` 触发，
  不能在没有听感确认的情况下自动剔除或接受。

## 排行结果

闭环排行与 Slice 9 静态排行已分开保存于
`data/reports/datasets/fuxuan_v1/slice10/ranking.json`、`.csv`、`.html`。当前闭环第一名
资产前缀为 `af54e63834bb`（闭环分 `0.844523`，静态 rank `16`）；这只说明自动指标的
结果，不替代盲听。人工 gate 后的最终聚合排行保存于
`data/reports/datasets/fuxuan_v1/slice10/final_ranking.json`、`.csv`、`.html`，自动排行和
人工评级仍分别保留。

## 人工 gate（已完成）

盲听页面：`data/reports/datasets/fuxuan_v1/slice10/blind_listening.html`。
共有 4 个句子区块、每区块 30 个匿名条目，共 120 条；实际导出结果为 keep `42`、reject
`78`、uncertain `0`。每条均有 `saved_at`，并已复制到
`data/reports/datasets/fuxuan_v1/slice10/blind_listening_decisions.json`。

页面对应的匿名清单是 `blind_listening.json`；排名映射单独封存在
`blind_listening_key.json`。按严格多数票（4 句至少 3 个 keep；2-2 平票不通过）后，30 个
候选中 6 个 approved、24 个 rejected。approved 池分布为：`emotion_neutral` 2、
`main_neutral` 4、`emotion_happy` 1；`emotion_angry` 和 `emotion_sad` 没有通过多数票，
没有自动回填。Slice 10 已完成验收，下一步只进入 Slice 11 的可行性调研入口。
