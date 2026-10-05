# Slice 12 S12-2 preflight

- 状态：ready；正式训练技术前置条件已满足
- 报告：`data/reports/datasets/fuxuan_v1/slice12/preflight_v2.json`

已核对 baseline `models/dots.tts-mf-1step`、官方 SOAR artifact、哈希、冻结数据、当前
Python 环境、Docker 和 CUDA。`pretrained_models/dots.tts-soar` 已按 immutable revision
`2f9b3e18d70d670d4c701da2dc55ded5755815ce` 下载；Docker Linux engine 与 RTX 5080 可用，
镜像 `dotstts-slice12:torch2.8-cu128` 已构建。宿主 `.venv` 缺少 `accelerate`、`peft`
不构成 blocker，因为 Docker image 已安装；`bitsandbytes`、`flash_attn` 为首轮明确不启用
的可选优化。

冻结 JSONL 中的音频是 Windows 绝对路径。现已生成只供容器使用的三字段派生清单到
`data/work/slice12/container_manifests/`，保留源 manifest、逐音频集合和输出哈希，未修改
`datasets/fuxuan/v1/`。

真实官方 SOAR LoRA 1-step、compact checkpoint 保存和恢复均已通过；峰值 allocated
`10,249,214,976` bytes、reserved `10,416,553,984` bytes，低于 15 GiB 合同上限。证据见
`docs/reports/slice12-soar-lora-smoke-v1.json`。

## S12-3 baseline matrix

已由 `scripts/build_slice12_baseline_manifest.py` 生成固定清单：

- 6 个 Slice10 approved reference
- 6 类测试句：中性、专名、长句、日期/数字、短句、参考外文本
- 2 个 seed：`20260908`、`20260909`
- 预计 72 个 generation jobs
- 清单哈希：`ddfb42d6211f1c882bf77fe3c273dc7423fd52a8be3824a5650e66a6f43927b1`
