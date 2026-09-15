# 下一阶段本地执行证据

日期：2026-09-16。工程实现 `f217bac`；拆分集成 `2b0575a`；比较基线 `a35a580`；
上游 `48d07c5`。完整 SHA、测试集合和分支文件列表见
[`final-validation.json`](../../../../runs/async-policy-rewrite/next-stage/final-validation.json)。

## 环境和结果

Python 3.12、Torch 2.11.0+cu128。CPU 运行均设置
`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。
复用 `runs/async-policy-rewrite/venv/bin/python`，通过各 worktree 的 cwd 和
`PYTHONPATH=.` 导入其自身源代码。基线使用独立 detached worktree。

| 检查 | 实际结果 |
|---|---|
| `a35a580` 完整 CPU | 889 passed、12 failed、10 skipped、22 deselected |
| 第一轮候选完整 CPU | 920 passed、相同 12 failed、10 skipped；随后补了一个 S1 用例 |
| `2b0575a` 完整 CPU | 921 passed、相同 12 failed、10 skipped、22 deselected |
| PR 1 / PR 2 / PR 3 独立检查 | 5 / 33 / 90 passed；命令见各 PR 草稿 |
| CPU 收集比较 | 911 → 943；没有丢失旧 node ID，新增 32 项 |
| 变更涉及的测试模块 | 149 项全部通过，包含 unittest 类方法；由完整运行的 JUnit 核对 |
| Sphinx HTML | 原基线、工程候选、PR 3、PR 4 均成功且无警告 |
| Ruff / Git | 本轮修改的 20 个 Python 文件通过；各 review 分支 `git diff --check` 通过 |
| 工具入口 | 14 个帮助/矩阵预览成功；另逐一检查 17 种 Modal phase 的资源预览 |

完整 CPU 命令在各自 worktree 执行：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=. python -m pytest tests/ -k cpu
```

原始输出包括 `baseline-pytest.log`、`candidate-pytest.log`、`pr4-pytest.log`、
`pr4-pytest.xml`、`baseline-collection.log`、`final-collection.log`；失败集合比较
另存 `cpu-failure-comparison.json`。这些文件在上述 JSON 所在目录。

## 不能写成“全套通过”的 12 个失败

| 模块 | 失败数 |
|---|---:|
| `test_agentic_cpu.py` | 1 |
| `test_ave_event_recognition_cpu.py` | 1 |
| `test_bailing_kv_cache_cpu.py` | 5 |
| `test_gemma4_multimodal_cpu.py` | 1 |
| `test_moe_sequence_parallel_cpu.py` | 4 |

基线与候选的失败 node ID 集合完全相同。此结论限于本次 CPU 环境与命令；
没有为本项目修改上述无关模块，也没有把环境下的失败隐藏成 skip。

## 分支和边界核对

- 四个 review 分支相对 upstream/main 均没有 `docs/projects/`、`runs/`、
  `REWRITE_START.md`。
- `f217bac` 与 `2b0575a` 的运行代码、示例、测试及英文用户文档相同。
- native 代码已无计划列出的 `_pending`、result pump、rendezvous、私有启动/中止访问。
- `done()/wait()` 不取消请求；有限 close 共享预算；启动失败清理有独立共享 5 秒预算。
- 本轮没有修改公开配置 dataclass、主 CLI 选项或项目依赖。

## 归档与计费核查

H1 的资料分支提交 `612bffc` 保存原型审查及证据的 8 个原文件，逐项核对 SHA-256。
原 `feat-async-trainer` dirty worktree 保持原状，归档 README 标明历史用途。

H2 复制并校验了 760 个原始结果文件，共 7,616,137,087 bytes。目标在实验盘镜像
之外；具体路径及排除项记录于 `backup-summary.json`。这是镜像外备份，不代表已做
独立物理介质的灾备。另补归档 next-stage 的 753 个文件、50,596,664 bytes，逐项
核对 SHA-256；路径与独立清单见
[`next-stage-backup-summary.json`](../../../../runs/async-policy-rewrite/next-stage/next-stage-backup-summary.json)。

H3 已查询本项目 Modal 卷及活动 App；没有本项目活动任务，卷保留供后续实验。
2026-09-16 查询的[官方价格](https://modal.com/pricing)为 Volumes $0.09/GiB/月，
含 1 TiB/月免费额。账号总存储占用与最终账单未核定，不能据此保证永久免费。

## 待验收

本轮没有启动 GPU 训练、推送分支、发送导师消息或创建对外 PR/issue。
新的 protocol/native/default attention/loss 尚缺双 GPU 结果；旧 R4 数据仅对应旧提交。
后续命令、停机条件与矩阵规模见 [GPU 计划](../GPU_VALIDATION_PLAN.md)。
