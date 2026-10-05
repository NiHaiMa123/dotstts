# Slice 7 进度快照（2026-08-27 中断时）

> 本文件固化 Slice 7「审核与情感标签校正」执行到一半的状态。
> 原因：会话额度中断。下次继续时从「剩余工作」直接接手，不需要重新调研。

## 已批准的计划（完整版见 `C:\Users\Administrator\.claude\plans\merry-giggling-moon.md`）

范围（用户已确认的三个决策）：

1. **只做基准集 40 条**（重点 31 条队列），不扩展到全量 277；
2. **纯静态 HTML+JS**（Python 生成、file:// 打开、localStorage 暂存、导出
   decisions.json、CLI 导入），不新增任何 web 框架依赖；
3. **同音异形以参考文本为准自动裁定**；语气词缺失/叠词需人工听音频裁定。

数据流：

```
catalog (v7) + comparison.json
  -> dots.tts.lab review-export
       ├─ data/reports/asr/review/review.html   静态审核页
       ├─ data/reports/asr/review/review.json    页面数据包（含 diff spans）
       └─ data/reports/asr/review/export.json    导出摘要
人工在浏览器审核（localStorage 可暂停恢复）
  -> 导出 decisions.json
  -> dots.tts.lab review-import
       ├─ catalog schema v8: review_decision 表（append-only）
       └─ decision_report.{json,csv,html}
```

## 已完成（代码在盘上，可直接用）

### 1. Schema v8 — `src/dots_tts_lab/catalog.py` ✅

- `_MIGRATIONS` 增加 migration 8：`review_decision` 表
  （append-only，UNIQUE(asset, benchmark, round) + UNIQUE(batch_id, row_index)
  双幂等键；text_decision CHECK 三值；emotion_primary/secondary/intensity/
  label_source/review_status；auto_rules_applied_json）
- `SCHEMA_VERSION = 8`
- 新方法：`record_review_decisions()`（INSERT OR IGNORE 返回 inserted/ignored）、
  `load_latest_review_decisions()`（每资产最新 round）、
  `load_review_decision_history()`、`next_review_rounds()`
- **注意：真实库 `data/catalog/catalog.sqlite` 尚未跑过 initialize()，还在 v7。
  下次运行任何 review 命令会自动迁移，无风险。**

### 2. `src/dots_tts_lab/review.py` ✅（约 1100 行，模块可导入、已过冒烟）

- `ReviewRulesConfig`：严格 pydantic 校验（homophone_pairs 为 list[list[str]]，
  strict 模式不接受 tuple —— 这是踩过的坑，已修）
- `diff_spans()`：字符级 diff 合并 span（DP 回溯 + run 合并），已验证
  替换/插入/删除三态正确
- `homophone_only_difference()`：全部后端差异可由规则表解释时才自动裁定
- `interjection_hint()`：句首语气词提示（不自动裁定，只标黄提示听判）
- `review_export()`：读 catalog + comparison.json，写 review.json /
  review.html / export.json。**已在真实数据上跑通**：
  40 条、31 条带优先级、7 条语气词提示、1 条同音异形自动裁定（#12 不只/不止）
- `review_import()`：package_sha256 校验（防过期数据包）、资产校验、
  幂等导入（round 推进）、生成 decision_report 三件套
- `ReviewDecision` / `ReviewDecisionBatch`：导入侧严格模型
- 静态页面模板 `_PAGE_TEMPLATE`：完整单文件页面（分组导航、音频播放、
  diff 高亮、自动裁定提示、文本三选一决策、情感五字段、导出+剪贴板兜底）
  — 用 `str.replace` 注入参数而非 `.format`（JS 模板字面量花括号冲突，踩过坑）

### 3. `configs/lab/asr/review_rules_v1.yaml` ✅

**注意方向**：左=参考写法（保留），右=后端产出变体（检查时替换回左）。
写反过一次（[不止, 不只]），已修正为 [不只, 不止] 等 5 对。

### 4. CLI — `src/dots_tts_lab/cli.py` ✅

`review-export` / `review-import` 两个子命令已接线（parser + main dispatch）。

### 5. `tests/test_review.py` ✅ 已写（13 项新测试）

RuleTests×4 + DiffSpanTests×1 + ReviewFlowTests×8（含幂等、历史 append、
过期包拒绝、重复行拒绝、报告内容、migration 幂等）。

### 6. 真实数据 export 已跑通 ✅

`data/reports/asr/review/` 下 review.html / review.json / export.json 已生成，
音频相对路径 `../../../work/standardized/...` 已验证可达。

## 中断时的确切位置

正在修 `_decision_report_html()` 的一个 `str.format` 占位符/参数数不匹配 bug。
**修完这处测试就能全绿。** 具体状态：

- 错误：`IndexError: Replacement index 9 out of range`（10 个 `{}` 占位符、
  9 个参数 —— 表头 9 列：# / Text decision / Original / Final / Changed /
  Weak label / Primary / Secondary / Status，行需要 class + 9 td = 10 个值）
- 中断前最后一次 Edit 已写入正确版本：class 用 review_status、9 个 td 依次为
  ordinal / decision / original / final / changed / weak / primary / secondary
  （注意：最后一个 td 目前是 **emotion_secondary**，而表头最后一列是
  **Status** —— 需确认第 9 个 td 应为 review_status，或表头去掉 Status 列
  改为 Secondary 结尾，二者取一，与 CSV/JSON 报告对齐即可）

### 已知待修清单（按顺序）

1. **修 `_decision_report_html` 行/表头列对齐**（见上）→ 跑
   `.venv/Scripts/python.exe -m unittest discover -s tests -t tests`
   期望 60 项全绿（47 原有 + 13 新增）
2. 全量测试 + `pip check` + `compileall`
3. 真实数据端到端演练：review-export → 浏览器打开 review.html 人工裁定几条
   （重点 #29/#31/#36/#40 语气词、#5 叠词、#7 药材名、#40 符玄）→
   review-import → 检查 decision_report 与 catalog v8
4. ADR `docs/adr/0007-review-decisions.md`（决策：单表 append-only + round 制
   vs 独立文本修正表；localStorage+导出导入 vs 轻后端；为何 40 条先行）
5. 验收报告 `docs/reports/slice7-review-acceptance.md`
6. PLAN.md Slice 7 完成记录 + git add/commit（用户要求过 slice 完成后关机，
   见对话；commit 前确认）

## 关键踩坑记录（不要重复踩）

- pydantic strict 模式下 YAML 的 pair 要声明 `list[list[str]]`，不是 tuple
- 静态页注入参数用 `str.replace`，别用 `.format`（JS `${...}` 花括号地狱）
- `diff_spans` DP 回溯是倒序的：run 合并要 prepend，最后整体 reverse 一次
  （已实现，勿再动）
- homophone 规则表方向：**参考写法在左**，`residual.replace(variant, reference_form)`
- `load_asr_benchmark_items()` 不返回 `reference_normalized`，要用
  `normalize_asr_text(text_exact, config)` 现算
- Windows 控制台输出中文要 `sys.stdout.reconfigure(encoding='utf-8')`
- 测试运行：`.venv/Scripts/python.exe -m unittest discover -s tests -t tests`
  （`-t tests` 必须带上，否则 import 失败）

## 验收标准（PLAN.md 原文）

所有人工修改有历史；原始标签永不丢失；审核可暂停和恢复。

—— 具体到实现的验收点：round 制 append-only（改主意 = 新 round，旧行保留）；
`emotion_weak_label` 原字段零改动；localStorage 键含 package_sha256 前 16 位，
包更新自动隔离旧草稿。
