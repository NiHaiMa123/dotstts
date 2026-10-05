# Slice 9 参考池用途、规模与重叠契约

- 状态：冻结
- 日期：2026-09-08
- 输入：`fuxuan@1`，tree SHA-256
  `77754574c4048a20682b0e8f9be95c322cf0848afedaf90d7c81b1f00df029af`

## 只从 train split 选参考

参考候选只允许来自 `dataset_item.split == "train"`。validation/test 的 54 条音频保持为后续
参考排行、模型闭环和训练实验的独立评估数据，不得进入参考池或作为候补回填。

冻结数据共有 train 219 条：neutral 68、happy 51、angry 95、sad 5；文本来源为人工审核 29、
未逐条听审 filename candidate 190，quality 为 pass 208、人工批准的 review 11。上述只是硬门禁前
人口，后续质量门禁可以继续减少各池可用数。

## 五个池

| pool id | 输入标签 | 用途 | 发布 Top-K | 最低可发布数 | 边界候补 |
| --- | --- | --- | ---: | ---: | ---: |
| `main_neutral` | `中立_neutral` | 通用 voice cloning 主参考；优先音色稳定、文本可信和低噪，不追求强烈韵律 | 10 | 6 | 下一名 5 条 |
| `emotion_neutral` | `中立_neutral` | 中立语气基准；优先“典型中立”和自然韵律，用于与其他情感池对照 | 6 | 5 | 下一名 3 条 |
| `emotion_happy` | `开心_happy` | 开心/轻快参考候选，用于后续固定文本闭环比较 | 6 | 5 | 下一名 3 条 |
| `emotion_angry` | `生气_angry` | 生气/强势参考候选，用于后续固定文本闭环比较 | 6 | 5 | 下一名 3 条 |
| `emotion_sad` | `难过_sad` | 低资源难过参考候选；完整保留可用多样性 | 5 | 3 | train 仅 5 条，不预留固定候补 |

“发布 Top-K”是自动预选报告送交人工审核的目标规模，不是最终主参考。Slice 10 才会通过相同
文本、seed 和模型闭环决定最终参考。emotion 字段当前不会作为官方 basic pipeline 的独立控制
输入；情感池只是按人工/弱标签组织参考音频，不能声称模型已经具备显式情感控制。

## 标签作用域与排序隔离

`emotion_weak_label`（目录/文件名推断的粗标签）和审核后得到的 `emotion_primary` 都只承担
一个作用：把候选路由到本契约中的 `pool_id`。它们不是客观质量、speaker 相似度、文本可信度
或音素覆盖的分数项，也不能作为“情感越强，排名越高”的隐式权重。即使
`emotion_primary` 已由人工确认，本 Slice 仍只把它当作分池标签，不把它解释为模型已具备
显式情感控制。

排名必须遵守以下隔离规则：

1. 每个 pool 独立执行硬门禁、特征缺失处理、分项归一化、综合分数和 deterministic diversity
   rerank；分布的分位数、有效权重和 tie-break 不能跨 pool 共享成一个 global rank。
2. 报告的排名键固定为 `(pool_id, rank)`。不得先在 neutral/happy/angry/sad 的混合集合上
   选一个 Top-K 再切情感；也不得为了填满某池从另一情感池回填。低资源池必须报告实际数量和
   `shortfall`。
3. `main_neutral` 与 `emotion_neutral` 虽共享 `中立_neutral` 路由标签，仍是两个不同用途的
   独立排名命名空间；允许的最多 3 条交集不意味着共享名次或共享综合分数。
4. 若需要跨池看板，只能展示各池的数量、分数分布和审核状态，不输出把不同情感排成先后顺序
   的总榜。任何跨池比较须标注为诊断，不得用于发布参考选择。

实现和验收时应检查这些不变量：每个候选的 `pool_id` 可由其冻结标签确定；同一
`emotion_*` pool 内不重复 asset；`emotion_*` 之间不出现 asset 交集；所有 rank 都带 pool
命名空间；任一 shortfall 都没有静默跨情感补齐。

## 重叠与互斥

1. 四个 `emotion_*` 池按单值 `emotion_primary` 严格分池，同一 asset 不得跨 emotion 池出现，
   不允许为了补足数量从其他情感回填。
2. `main_neutral` 与 `emotion_neutral` 的用途不同，允许同一优质 neutral asset 同时出现，但
   Top-K 交集最多 3 条。也就是说，main 至少 7/10 条、emotion-neutral 至少 3/6 条必须是各自
   独有候选，避免两个榜单退化为同一排名。
3. 同一 pool 内 asset、音频 SHA、source identity 和 Slice 8 similarity group 均必须唯一；
   reviewed near edge 的两个 endpoint 不得同时进入 review window。
4. 一个 asset 在同一 pool 只能有一个 rank。人工审核排除后，仅按该 pool 的稳定候补顺序回填，
   不改变其他 pool 的名次。
5. 中性两个池的交集约束在每次回填后重新检查；如果某次回填使交集超过 3，跳过该候补并记录
   `neutral_overlap_cap` 理由。

## 数量不足规则

- 硬门禁后可用数达到 Top-K：输出完整 Top-K 和指定边界候补。
- 可用数低于 Top-K 但不低于最低可发布数：不复制、不跨情感回填，输出实际数量和 `shortfall`。
- 可用数低于最低可发布数：该 pool 状态为 `insufficient_candidates`，Slice 9 不能整体验收；需
  调整有证据支持的门禁、补充新数据版本或由用户决定缩小产品目标。
- sad 的 5 条 train 样本全部进入排名和人工审核范围；任何排除都会显式降低实际池规模。

## 排名与审核窗口

报告必须列出各池全部硬门禁通过候选，但默认人工窗口为 Top-K 加边界候补。边界候补用于检查
阈值和权重是否把相近样本错误切开，也用于人工排除后的确定性回填。

按当前硬门禁前数量，五池目标总名额为 33；考虑 `main_neutral`/`emotion_neutral` 最多 3 条重叠，
最终预选至少包含 30 个不同 asset。此规模既能支撑 Slice 10 的闭环排行，也把首轮人工审听控制
在可操作范围内。
