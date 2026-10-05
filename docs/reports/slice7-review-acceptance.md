# Slice 7 审核与情感标签校正验收报告

- 状态：通过
- 更新：2026-09-07
- Benchmark：`fuxuan_asr@2`
- Review package：`652b5dd7143573d26574b908d6d081a637fb7b8ba960f58e6fa62aba87150894`
- Decision batch：`5cd01dd0-a01c-44a1-a7e7-8ec731c18704`

## 当前结论

Slice 7 的代码、数据约束、provenance、静态页面、自动端到端链路和正式人工听审均已
通过验收。用户在真实浏览器中完成 40 条逐条复核并导出决策包；`review-import` 严格校验
后向正式 catalog 插入 40 条 round 1 决策，无忽略或冲突。

最终结果为 approved 40 / rejected 0 / pending 0；修正文本 1 条，情感主标签校正或补充
secondary 共 9 条。catalog latest、完整 history 和 decision report 的 40 个资产完全一致。

## 冻结范围与证据

- 样本：40
- 审核队列：31（high 18 / medium 9 / low 4）
- 未入队：9
- 原始弱标签：neutral 10 / happy 10 / angry 14 / sad 6
- 原始弱标签映射基线 SHA-256：
  `69ea7c0c74105235d8282cb7560472cb26283a5a6df4efbb11411df959a6fd63`
- 同音异形自动裁定候选：1
- 全后端语气词提示：4
- 推荐后端：`qwen3_asr`
- ASR runs：
  - `qwen3_asr`: `5b4e999f-5dbf-4530-878e-b05e115e2c01`
  - `sensevoice`: `d8141572-b234-45eb-95a8-7df4707e68a0`
  - `faster_whisper`: `6a728964-c58c-4a92-9fd0-05ffb1c92fdf`

## 自动化验收

- 全量单元测试：82/82 通过
- `compileall`：通过
- `pip check`：`No broken requirements found`
- Slice 7 专项：32/32 通过
- 页面 JavaScript harness：通过
- 三条临时 catalog E2E：通过
  - 第一批：accept reference、edited text、pending defect
  - 输出：decision_report JSON/CSV/HTML
  - 第二批：新 batch 修改已有资产
  - 历史：round 1/2 均保留

## 产物哈希

| Artifact | SHA-256 |
|---|---|
| `data/reports/asr/review/review.html` | `a33ce310f0e38727c8284789de88afe35b636f80ae016ebbebf3498a11021779` |
| `data/reports/asr/review/review.json` | `0b7803ca87de81a615e0fa2a415026f5ed3f67c310f4b0be09e1118ced625ce1` |
| `data/reports/asr/review/export.json` | `ae524cc634482f03e6ff02955e76598ede210feb91a3f4fc525321361a3132df` |
| `data/reports/asr/comparison/comparison.json` | `49ec0c71734565543332140ee59a1816be6fe30b4db57f7fc26f89bb70429b5d` |
| `data/reports/asr/review/decisions-5cd01dd0.json` | `c7fb0776b220411ecf5a98de670000a6d50488106ca32cd27a44b69d49a8775c` |
| `data/reports/asr/review/decision_report.json` | `affdcba49092ec4853a429ae6e94367c90664cc66e080566242026399d4b9548` |
| `data/reports/asr/review/decision_report.csv` | `93d0101d9fca87cef9d77c135cf6775537bdd2a6e6343801df73b3572b788850` |
| `data/reports/asr/review/decision_report.html` | `39bd683d7687cc081f28e6507f057df873eacbc07d3289314e36cf1a41b52f2e` |

## 已验证的不变量

- 页面不再使用 `file://` 下会被拦截的 `fetch(review.json)`；package 数据安全内嵌，
  独立 JSON 仍用于哈希和审计。
- comparison schema/config/lexicon、benchmark config/manifest、backend run、样本与队列
  均在导出前校验。
- import 绑定完整 package hash，不信任客户端的 reference 接受声明或 auto-rule 字段。
- 同一 batch 完全重放是 no-op；内容变化报错；新 batch 追加 round。
- round 分配和插入在一个 immediate transaction 内完成。
- 原始弱标签只读；校正写入 append-only `review_decision`。
- 导入后 40 条弱标签映射与导入前 package 逐资产一致，冻结 SHA-256 仍为
  `69ea7c0c74105235d8282cb7560472cb26283a5a6df4efbb11411df959a6fd63`。
- 当前 latest/history 均为 40 条 round 1；`label_source` 为 31 条
  `weak_label_confirmed` 和 9 条 `human_corrected`。
- pending/rejected/approved 在页面导航中可区分。

## 正式人工审核结果

| 指标 | 结果 |
|---|---:|
| 已人工听审 | 40 / 40 |
| approved | 40 |
| rejected | 0 |
| pending | 0 |
| 文本修正 | 1（`哼哼` → `哼`，备注“确实只有一个哼”） |
| 情感修正/补充 | 9 |
| 同音规则确认 | 1 / 1（保留 reference `不只`） |
| 语气词/叠词听判 | 4 条全后端语气词提示均覆盖；叠词修正 1 条 |

## 最终退出条件

- [x] 真实浏览器视觉/点击验收完成。
- [x] 40 条均由人工听音频完成审核。
- [x] 18 high、9 medium、4 low 和 9 unflagged 均覆盖。
- [x] sad 6 条和语气词/叠词/同音异形边界样本有明确结论。
- [x] 原始弱标签复核前后完全一致。
- [x] 最终 decision report 与 catalog latest/history 一致。
- [x] 重新运行全量测试、compileall、pip check 并更新本报告实际数字。
- [x] 更新 `PLAN.md` 后才进入 Slice 8。
