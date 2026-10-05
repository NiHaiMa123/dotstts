# Slice 8 Deterministic Group-aware Split 策略

- 状态：已完成并冻结为 `fuxuan@1`；本文后半保留冻结前的算法设计依据
- 日期：2026-09-07
- dataset：`fuxuan@1`
- 比例：train / val / test = 0.80 / 0.10 / 0.10
- seed：`20260907`

`deterministic_group_greedy_swap_v1` 已实现 largest-remainder 总量目标、稳定 group 排序、
完整 group 分配、有限 deterministic swap 和跨 split fail-closed 审计；
`deterministic_group_emotion_swap_v1` 在此基础上加入全局 row/column-balanced strata target、
low-resource eval 覆盖和 primary/weak 双口径偏差报告。冻结前的两个预演文件在 277 条候选上
运行；随后应用人工排除，正式 `fuxuan@1` 保留 273 条，train/validation/test 为
219/27/27。canonical tree SHA-256 为
`77754574c4048a20682b0e8f9be95c322cf0848afedaf90d7c81b1f00df029af`；验收以
`slice8-dataset-freeze-acceptance.md` 为准，预演文件不是最终数据集。

## 分组硬约束

- exact audio、exact normalized text、人工确认的 near-audio edge 和应合并的 near-text edge
  共同形成无向图；每个 connected component 是不可拆分的 `similarity_group`。
- 没有 edge 的资产各自形成单元素 group。
- group id 由组内已排序 asset SHA-256 的 canonical JSON 求 SHA-256，不依赖遍历或插入顺序。
- 同一 group 绝不跨 split；若任何审计发现跨 split，冻结失败。
- source path/目录只保留 provenance，不作为硬 group。当前目录层级编码情感类别，按目录分组会
  把整个 emotion 放进单一 split，违反评测覆盖目标。
- 当前只有一个 speaker，因此不做 speaker-disjoint split；speaker outlier 必须在准入阶段先复核。

## 文本与标签分层

- 分层标签使用冻结候选最终 `emotion_primary`，同时保留原始 weak label 供审计。
- 当前基于 Slice 7 latest decision 的预期 primary 分布为 neutral 86 / happy 64 / angry 120 /
  sad 7；实现必须从 catalog 重新计算，不能把此数字写死。
- secondary emotion 不参与主分层，只进入 stats 中的交叉分布。
- 不复制、不随机过采样、不伪造样本数。

## 确定性分配算法

1. 对 split 总样本数用 largest-remainder method 生成整数目标；277 条的总目标为
   train 221 / val 28 / test 28。
2. 对每个 primary emotion 计算比例目标；整数余数在全局总目标约束下统一分配，避免各 strata
   独立取整后破坏 split 总数。
3. group 按以下稳定键排序：资产数降序、总时长降序、
   `sha256(dataset_config_sha256 + seed + group_id)` 升序。
4. 逐 group 尝试三个 split，选择硬约束满足且代价最小者。代价按顺序衡量：split 总样本数
   偏差、primary emotion 偏差、总时长偏差、group 数偏差；同分由 train/val/test 固定顺序
   和稳定哈希打破。
5. 完成后做有限 deterministic local-swap 改善目标偏差；只能交换完整 group，且每轮必须严格
   降低同一目标函数，因此结果与运行顺序无关。

## 低样本 sad 规则

- 不以 PLAN 中历史 weak-label 数字 6 写死；以冻结候选的最终 primary 数和 group 数为准。
- 当 sad 至少有 3 个独立 group 时，val 与 test 各至少包含 1 个 sad group，其余优先进入
  train。当前预期 7 条且尚无重复 group，目标为 5 / 1 / 1。
- 若 sad 少于 3 个 group，不拆 group、不复制样本；按 train→val→test 优先级保留可训练性，
  在 stats 与验收报告中明确缺失的 split 覆盖。
- 报告 weak-label sad 与 final-primary sad 两套分布，避免把人工校正后的 7 条误写成原始 6 条。

## 重建与版本规则

- 相同 candidate snapshot、analysis config、人工 edge 结论、dataset config 与实现版本必须产生
  相同 group id、split 和规范化 manifest 字节。
- 任何输入资产、最终文本/标签、duplicate edge、阈值、seed 或 split 算法变化都必须提升
  dataset version；不得覆盖 `datasets/fuxuan/v1/`。
- 运行时间、临时 staging 路径不进入 canonical checksums；所有影响语义的 provenance 均进入。
