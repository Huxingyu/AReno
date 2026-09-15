# 本地 PR 拆分与验证

本轮先准备本地 review 分支。未推送、未创建 GitHub PR；维护者评审与 OSPP 发布时间另行确认。

共同基线为 `upstream/main` 的 `48d07c5`。四个分支排除 `docs/projects/`、`runs/` 和
`REWRITE_START.md`；本页与沟通稿只保留在工程资料分支。

| PR | 本地分支 | 范围 | 依赖 |
|---|---|---|---|
| 1 | `review/async-seed` | sampling seed 透传与直接 CPU 测试 | 无 |
| 2 | `review/async-engine` | cluster lifecycle 公共接口、训练 state 保存/恢复及 CPU 测试 | 无 |
| 3 | `review/async-core` | pipeline/bridge/coordinator、数据契约、loss 消融、CPU oracle、概念文档 | 无 |
| 4 | `review/async-native` | 原生适配、示例、GPU/benchmark 工具、续训、cookbook | 合并 1/2/3，包含同步 seed 回归前置 |

## 验证约定

每个分支从自己的 worktree 执行测试，CPU 命令设置
`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。
同一 Python 工具环境可复用，但 `areno` 导入须指向各分支自身。

具体 commit、各分支测试数量、diff 范围和最终候选验证在分支准备后追加。
GPU faults、extended-faults、example、full-parameter resume 和质量矩阵尚未运行，
不会在 PR 描述中标记为已通过。
