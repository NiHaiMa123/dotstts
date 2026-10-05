# Slice 7 Review 与任务清单

更新时间：2026-08-28

## 当前结论

Slice 7 已有可用的骨架：catalog v8、静态审核页、localStorage 草稿、JSON
导出/CLI 导入、append-only review round 和三种报告均已接线。但当前实现还不适合开始
40 条正式人工审核；否则可能出现审核结果无法保存、修改被静默忽略，或页面证据来自不同
ASR 快照的问题。

当前 `data/reports/asr/review/` 仍是 `fuxuan_asr@1` 的产物，而当前默认 benchmark
和 comparison 已升级为 v2。现有页面应视为过期，不应继续录入正式裁定。

## Review 发现

### P0：开始正式人工审核前必须修复

- [x] 修复 `flag_defect` 无法保存：页面把该决策的 `text_final` 设为 `null`，但
  `save()` 只保存 `text_final !== null` 的决策，导致缺陷标记不会进入 localStorage，
  也不会被导出。（已修复并通过定向回归测试）
- [x] 修复 decision HTML 报告生成崩溃：表格行有 class + 9 个单元格占位符，实际
  少传 `review_status`。当前 13 项 Slice 7 测试中有 3 项因此报错，导入在写完
  catalog 后生成 HTML 时失败。（缺失列已补齐，原 3 项失败测试通过）
- [x] 明确并实现 batch 幂等语义：页面会复用同一个 `export_batch_id`，并提示重复
  导出会“覆盖导入”；catalog 实际使用 `INSERT OR IGNORE`。同批次中已经导入过的
  裁定若被修改，会被静默忽略。建议每次导出生成新 batch id；同 batch 重放仅在内容
  完全一致时判为幂等，否则显式报冲突。（已实现并通过 3 项定向测试）
- [x] 将 review package 绑定到 comparison 中记录的精确 ASR `run_id`。当前导出取
  catalog 的“最新 run”，却从已有 comparison 报告读取队列和推荐后端；新增一次
  ASR run 后，页面会混用旧比较结论和新假设。（已绑定精确 run 并通过 superseded-run 测试）
- [x] 重新基于 `fuxuan_asr@2` 和 comparison v2 生成 review package；废弃当前
  benchmark v1 的 `review.html`、`review.json`、`export.json`。（已生成 40 条 / 31 队列，
  18/9/4 优先级，三个 run id 与 comparison 一致，音频路径全部可达）

### P1：导入前补齐的数据约束与审计安全

- [x] 把 review round 分配和插入放进同一事务；并发导入不能因相同 round 触发
  `INSERT OR IGNORE` 后静默丢失一批真实裁定。（已用 `BEGIN IMMEDIATE` 串行化分配，
  两个并发写入得到 round 1/2，测试通过）
- [x] 校验 `accept_reference` 的 `text_final` 必须与冻结参考文本完全一致；否则应使用
  `accept_edited`。不要仅凭客户端声明分类。（服务端按 review package 冻结参考校验，
  伪接受测试确认拒绝且零写入）
- [x] 收紧 `flag_defect` 语义：规定允许的 `review_status`、`text_final` 和必填备注，
  避免出现“approved + flag_defect”或无理由缺陷。（仅允许 pending/rejected、null final、
  非空备注；三类非法和一类合法测试通过）
- [x] 将 `emotion_primary` / `emotion_secondary` 限定为允许值；校验真实 UUID 与 ISO-8601
  时间戳，避免绕过页面手写 JSON 写入任意标签或无效审计时间。（四类 emotion Literal、
  canonical UUID、带 UTC offset 的 ISO-8601 均已校验并通过拒绝测试）
- [x] import 时校验 decision 中的 `auto_rules_applied` 与 review package 原值一致，
  不信任客户端回传的规则审计字段。（逐资产精确比对，篡改测试确认拒绝且零写入）
- [x] comparison/package provenance 校验至少覆盖 schema、comparison config hash、
  benchmark manifest/config hash、backend/run 集合、样本 ordinal/asset 集合和推荐后端。
  （已补齐 comparison provenance 字段、CLI config/lexicon 参数与六类漂移测试，并重生成
  comparison/review v2）
- [x] 让 review history 查询显式带 benchmark id/version，避免同一资产跨 benchmark
  版本的 round 在审计展示中混在一起。（API 已要求显式 scope，两轮历史测试通过）

### P2：页面语义与可用性

- [x] 修正语气词提示条件或文案。函数说明写的是“每个后端都听到”，实现却是任一后端
  命中即提示，页面又显示“多家后端听到”；应采用明确的共识阈值并测试。Slice 6 v2
  的跨后端结论是 4 条，旧 review 页面显示 7 条，不能直接沿用。（已要求全后端命中，
  参考已含任一语气词则不提示；0/1/2/全部边界测试通过，真实结果恢复为 4 条）
- [x] 补全三个 `<select>` 的闭合标签，确保浏览器 DOM 不依赖容错修复。（三个控件均已
  显式闭合，页面回归测试通过）
- [x] 修复静态模板遗留的双花括号：`str.replace` 注入后统一还原 CSS/JS 的 `{}`，避免
  生成页面包含无效的 `{{ ... }}`。（页面生成回归测试通过）
- [x] 修复 `file://` 数据加载：审核页不能依赖浏览器通常会拦截的本地 `fetch`；安全内嵌
  当前 package 数据，同时保留独立 `review.json` 供哈希和审计。（`<>&` 已转义；单文件、
  注入安全和 JavaScript harness 三项测试通过）
- [x] 页面保存前验证：`accept_edited` 不得为空、primary/secondary 不得相同、缺陷备注
  按规则必填；错误应就地提示，不能等 CLI import 才失败。（校验失败会阻断导航/导出，
  flag_defect 自动切换 pending；3 项页面回归测试通过）
- [x] 区分 `approved`、`rejected`、`pending` 的导航状态；当前 pending 也显示为绿色 done。
  （pending 改为黄色，进度文案改为“已保存”，页面测试通过）
- [x] 增加显式保存反馈：保存按钮旁显示本地草稿状态、review status 和保存时间，按钮短暂
  显示“已保存 ✓”，顶部解释左侧状态圆点颜色。
  （2026-09-07：定向页面测试 3/3 通过，已重新生成 40 条样本的 `review.html`）
- [x] 校验 localStorage 恢复的数据结构和 package 资产集合；损坏或旧数据应提示并隔离。
  （校验资产、决策/status、文本、emotion/intensity；异常草稿备份到 invalid key 后清空，
  页面测试通过）
- [x] decision report 的计数应以实际插入/幂等确认后的 catalog 记录为准，不能按本次
  输入 rows 统计被忽略或冲突的修改。（写库后按 batch 重读持久化 rows，幂等重放仍报告
  原 round；报告测试通过）

## 测试任务

- [x] 修复现有 3 个错误，使全部 13 项 Slice 7 测试通过。（测试持续扩展，
  最终 `test_review.py` 32/32 通过）
- [x] 增加页面逻辑测试：`flag_defect` 保存/恢复/导出、空编辑拦截、pending 状态、每次
  导出的 batch 行为、非法 localStorage。（新增无第三方依赖的 Node VM harness，实际执行
  生成 JavaScript；测试通过）
- [x] 增加幂等冲突测试：同 batch 同内容重放为 no-op；同 batch 改内容必须失败；新 batch
  改主意追加 round；并发导入不丢记录。（4 条路径全部有回归测试并通过）
- [x] 增加 provenance 测试：comparison run 不是 latest 时仍读取指定 run；run 缺失、
  hash 漂移、样本集合不一致时拒绝导出。（完整矩阵通过）
- [x] 增加导入约束测试：伪 `accept_reference`、非法 emotion、非法 UUID/时间戳、篡改
  auto rules、无备注 defect 均被拒绝。（4 组矩阵测试通过）
- [x] 增加语气词共识边界测试：0/1/2/全部后端命中及后端缺失场景。（规则边界与
  缺样本后端的导出拒绝测试均通过）
- [x] 运行全部单元测试、`compileall` 与 `pip check`，记录实际项数和结果。（最新 82/82 tests
  通过；compileall 通过；pip check：No broken requirements found）

## 正式审核与验收

- [x] 用修复后的代码导出 v2 审核包，并核对：40 条样本、31 条队列、18/9/4 优先级、
  三个固定 run id、音频相对路径均可访问。（全部核对通过；4 条全后端语气词提示；
  package `652b5dd7143573d26574b908d6d081a637fb7b8ba960f58e6fa62aba87150894`）
- [x] 先做 3–5 条端到端冒烟：保存、关闭/重开恢复、导出、导入、改主意后二次导入、
  检查 JSON/CSV/HTML 和完整 history。（Node VM 验证保存/恢复/导出；临时 catalog 上完成
  3 条导入、三报告和新 batch 二次修改，history 为 round 1/2）
- [x] 在真实浏览器完成一次视觉/点击复核。（2026-09-07：用户从生成页面完成逐条保存并
  导出 batch `5cd01dd0-a01c-44a1-a7e7-8ec731c18704`，40 条均通过页面校验）
- [x] 按顺序审核 18 条 high、9 条 medium、4 条 low，再检查未入队的 9 条；sad 6 条和
  语气词/叠词/同音异形边界样本必须人工确认。（正式 batch 已覆盖 package 全部 40 个
  唯一资产；导入 40、忽略 0，approved 40 / rejected 0 / pending 0）
- [x] 核对原始 `emotion_weak_label` 未被修改；所有人工校正只进入 review_decision，且
  `label_source` 可解释。（2026-09-07 正式导入后逐资产复算：40/40 映射一致、差异 0，
  SHA-256 仍为 `69ea7c0c74105235d8282cb7560472cb26283a5a6df4efbb11411df959a6fd63`；
  `weak_label_confirmed` 31 条、`human_corrected` 9 条）
- [x] 完成 `docs/adr/0007-review-decisions.md`，记录 append-only round、batch 幂等、
  静态页方案、40 条先行和规则自动裁定边界。（已 Accepted，包含 provenance、事务、
  localStorage 隔离和退出条件）
- [x] 完成 `docs/reports/slice7-review-acceptance.md`，列出最终裁定数、文本/情感修正、
  reject/pending、规则命中、测试结果及产物哈希。（2026-09-07：40/40 正式决策、
  catalog latest/history 一致性、82/82 测试和 8 个最终产物哈希均已记录）
- [x] 更新 `PLAN.md` 的 Slice 7 完成记录；只有在上述验收项完成后再进入 Slice 8。
  （2026-09-07：已记录正式 batch、40 条裁定统计、不变量、报告一致性与最终验证结果）

## 建议执行顺序

1. P0 数据安全与 provenance。
2. P1 导入约束、事务和审计。
3. P2 页面体验与语义一致性。
4. 自动化测试全绿。
5. v2 端到端冒烟。
6. 40 条正式人工审核、ADR、验收报告和 PLAN 收口。
