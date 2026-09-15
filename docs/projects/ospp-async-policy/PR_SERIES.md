# 本地 PR 拆分与验证

2026-09-16：四个本地 review 分支均已提交并验证。未推送、未创建 GitHub PR；
维护者评审与 OSPP 发布时间另行确认。

共同基线为 `upstream/main` 的 `48d07c5`。四个分支排除 `docs/projects/`、`runs/` 和
`REWRITE_START.md`；本页与沟通稿保留在工程工作分支，不带入功能 PR。

| PR | 本地分支 / 提交 | 范围 | 依赖 / 描述草稿 |
|---|---|---|---|
| 1 | `review/async-seed` / `b84e509` | sampling seed 透传与直接 CPU 测试 | 无；[PR 1](drafts/PR1_SEED.md) |
| 2 | `review/async-engine` / `ac0cd7c` | cluster lifecycle 公共接口、训练 state 保存/恢复及 CPU 测试 | 无；[PR 2](drafts/PR2_ENGINE.md) |
| 3 | `review/async-core` / `44fe9ac` | pipeline/bridge/coordinator、数据契约、loss 消融、CPU oracle、概念文档 | 无；[PR 3](drafts/PR3_CORE.md) |
| 4 | `review/async-native` / `2b0575a` | 原生适配、示例、GPU/benchmark 工具、续训、cookbook | 合并 1/2/3；[PR 4](drafts/PR4_NATIVE.md) |

worktree 分别为本工作目录的同级 `async-policy-pr1`、`async-policy-pr2`、
`async-policy-pr3`、`async-policy-pr4`。PR 4 合并依赖后的起点是 `79a8798`，
其自身改动可用 `git diff 79a8798..2b0575a` 查看；对 upstream/main 的差异包含
依赖内容。待依赖合入后再更新 PR 4 的比较基线。

工程实现提交为 `feat/async-policy-rewrite` 上的 `f217bac`。已核对该提交与
`2b0575a` 在 `areno/`、`examples/`、`tests/`、英文概念页/cookbook/index 的
内容完全相同；项目审查和交付草稿保留在工程分支。

## 验证约定

每个分支从自己的 worktree 执行测试，CPU 命令设置
`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。
同一 Python 工具环境可复用，但 `areno` 导入须指向各分支自身。

各 PR 草稿包含对应命令。最终实际结果：

| 分支 / 检查 | 结果 | 原始记录 |
|---|---|---|
| PR 1 generation CPU | 5 passed | `pr1-pytest.log` |
| PR 2 lifecycle/checkpoint/backend/import CPU | 33 passed | `pr2-repro-pytest.log`；前一次检查记录为 31 passed |
| PR 3 contracts/core/loss/registry/import CPU | 90 passed | `pr3-repro-pytest.log` |
| PR 4 完整 CPU | 921 passed、12 failed、10 skipped、22 deselected | `pr4-pytest.log`、`pr4-pytest.xml` |
| 原基线 `a35a580` 完整 CPU | 889 passed、同样 12 failed、10 skipped、22 deselected | `baseline-pytest.log` |
| CPU 收集比较 | 原 911 项全部保留，新增 32 项，最终 943 项 | `final-validation.json` |
| 变更涉及的测试模块 | 149 项全部通过，包含 unittest 类方法 | 同上及 PR 4 JUnit |
| 文档 | 基线、候选、PR 3、PR 4 HTML 均无警告 | `docs-*.log` |
| 静态与边界检查 | 本轮 20 个 Python 文件 Ruff 通过；四分支 diff check 通过；native 无所列 engine 私有访问 | `final-ruff.log`、`final-validation.json` |

记录位于 `runs/async-policy-rewrite/next-stage/`。完整 CPU 套件未全绿：
12 个失败的 node ID 与基线完全一致，见 [执行证据](evidence/next-stage-validation.md)。
工具的 17 种 Modal phase 已做资源预览；这不证明其真实 GPU 阶段成功。

下一步是 [GPU 分阶段验收](GPU_VALIDATION_PLAN.md)。faults、extended-faults、
example、full-parameter resume、质量和吞吐矩阵均尚无本轮 GPU 结果。
