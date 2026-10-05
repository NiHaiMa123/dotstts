# Slice 8 数据集冻结验收报告

- 状态：通过
- 数据集：`fuxuan@1`
- 正式目录：`datasets/fuxuan/v1/`
- analysis run：`9e2ff68f-b473-4e3a-acf9-b333c54c3535`
- 候选快照 SHA-256：`5fd831600c498344166af3834a5809e0bba2a7831f5bea63f498deaac1daceea`
- 冻结 tree SHA-256：`77754574c4048a20682b0e8f9be95c322cf0848afedaf90d7c81b1f00df029af`

## 结论

Slice 8 的退出条件全部满足：最终 273 条样本没有跨 split 的精确音频、规范化文本、
人工接受近重复、similarity group 或来源 identity 泄漏；发布目录不可变且同内容重放幂等；
独立临时目录重建得到 8/8 个完全相同的文件；Parquet 每行均保留 raw、source、derived、
quality、text、label、review 和 analysis/group lineage。

## 人工边界结论

原始候选 277 条。用户确认 3 组 near-text 均为相同内容的完整/截取版本，保留较完整代表并
排除 3 个截短版本；speaker outlier rank 4 因音质很差排除，其余 5 个候选确认为同一 speaker。
源 raw 与标准化音频没有物理删除。最终排除 4 条、保留 273 条、过采样 0 条。

正式分析报告状态为 `reviewed`：near-text edge 3、speaker outlier 6、待审核 duplicate 0、
待审核 speaker 0。edge review snapshot SHA-256 为
`34002c03383950819b018cc1c5fe1a5fdf8ca13243a057e5730fc836b51345df`；speaker review
snapshot SHA-256 为 `9d21147cf038190f709f969bb1f652a4e25627f8cc2c9f4a01bd899e8bfde813`。

## 最终数据统计

| split | 条数 | 时长（秒） | neutral | happy | angry | sad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 219 | 1390.706708333333 | 68 | 51 | 95 | 5 |
| validation | 27 | 173.843104166667 | 8 | 6 | 12 | 1 |
| test | 27 | 173.840083333333 | 8 | 6 | 12 | 1 |
| 合计 | 273 | 1738.389895833333 | 84 | 63 | 119 | 7 |

final-primary 四类在三个 split 均精确命中整数目标；低资源 sad 为 5/1/1。原 weak-label sad
仍如实报告为 5/0/1，没有复制样本来伪造平衡。

准入样本的 text provenance 为 `human_review` 39 条、
`filename_candidate_unreviewed` 234 条；quality 为 pass 258、review 15。未逐条听审的 234 条
没有被标成已人工确认；15 条 quality review 仅因其均有 latest approved 人工文本审核才准入。
这是 ADR 0008 明确接受的数据版本边界，不影响本次冻结完整性，但后续仍可通过新数据版本扩大
人工覆盖。

## 冻结产物

| 文件 | SHA-256 |
| --- | --- |
| `build_config.yaml` | `ff106c9d7a3ada5aacd2b0d50ab36bc97bd4cf5cd5ab97beb07ce75afdf3fbb7` |
| `checksums.txt` | `5ce2910a47b11964b0421535508772e9e20c96afde74db0cbca60c23882dbbcf` |
| `dataset.parquet` | `6bf5d0e29a16c62cf92087af9380bb9f62e724f964700c428c000fa8fa4c0c2e` |
| `manifest.json` | `4b2b6ab4cc242ee023661f7a1eb26fed383aa838fb585f9d0a84c68be63a9da6` |
| `stats.json` | `4044ff69aca5c2a28cfb3aaafae2c6598d863de7f622ef2cad8d617204dad0d0` |
| `train.jsonl` | `317406266213d920cea2c3a2713b41210195beb858d130e97602c100227a08cc` |
| `validation.jsonl` | `a4af63be4b3da3f80538781784c1007b767bf4099164b883b8fb58776cf78e56` |
| `test.jsonl` | `02a2dd1894caf0cd323dc9d11fd9d5849ad1b67b6f1c19f05831446ff2272299` |

`manifest.json` 内的 core artifact-set SHA-256 为
`9ef664c6825e53cd9d6681315b0e0ce0542d1325152524db643ff1992b8bb052`。三个训练 JSONL 严格
只有官方 `fid`、`audio`、`text` 字段，音频均为标准化文件绝对路径。

## 泄漏、对账与重建证据

- split audit：273 个 group、273 个音频 SHA、273 个规范化文本、273 个来源 identity 全部检查；
  3 条 catalog near edge 均有一个人工排除 endpoint，selected-endpoint edge 为 0；五类 violation
  全部为 0。报告 SHA-256：
  `e402b7a7171e0049ee68db19ea1dc7715b6e02266fdcd62aea00a4ae170e6e2f`。
- 发布后逐条重读 273 个音频，路径存在且 SHA-256 匹配；唯一 speaker 为 `崩铁符玄`，文本均非空。
- Parquet、三个 JSONL 和正式 catalog 的 273 个 asset/split 及全部 catalog lineage 字段逐行一致。
- catalog 已登记 1 条 `dataset_version`、273 条 `dataset_item`；重复登记返回 `cached`，
  `integrity_check=ok`、foreign-key violation 0。
- 独立重建没有允许忽略的运行时间字段，8/8 文件 size/SHA-256 完全一致。重建报告 SHA-256：
  `e5a235dc64c4ca3f787d0ab211a04889db3162741e9c8515d4f7424c2e736c41`。

## 验证命令结果

- `python -m unittest discover -s tests`：149/149 通过。
- `python -m compileall -q src scripts tests`：通过。
- `uv pip check --python .venv/Scripts/python.exe`：92 个包兼容。
- 合成端到端测试覆盖 exact duplicate、增益/静音近重复、speaker outlier、相似文本组和
  low-resource strata，并通过最终 split leakage audit。

因此可以关闭 Slice 8，并开始 Slice 9 的参考音频预选。
