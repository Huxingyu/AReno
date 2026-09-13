AReno 原生异步策略训练器：OSPP 项目资料

本目录维护课题 267bf0030 的需求、实现状态与实验记录。工作仓库为
[Huxingyu/AReno](https://github.com/Huxingyu/AReno)，官方上游为
[inclusionAI/AReno](https://github.com/inclusionAI/AReno)。

当前阶段已经完成同步 GRPO 基线和需求核对，完整异步 trainer 尚待实现。
材料中的结果固定于各自记录日期；后续进展以实际实现和实验为准。
后续开发按 8 周目标、另留 2 周缓冲安排；先阅读实施方案，再从 TODO 的 T01 开始。

| 文档 | 内容 |
|---|---|
| [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) | 可供评审与交接的完整方案：范围、架构、版本与同步协议、验收、工期、资源和交付流程 |
| [TODO.md](./TODO.md) | 16 项核心任务的依赖、执行内容、完成条件和证据要求；单独列出后续扩展 |
| [PROJECT_REQUIREMENTS.md](./PROJECT_REQUIREMENTS.md) | 官方中英文合并需求、原文、交付范围和执行清单 |
| [REQUIREMENTS_AUDIT.md](./REQUIREMENTS_AUDIT.md) | 主线已有能力、缺口、未合并 PR 与本地协调器的边界 |
| [BASELINE_REPORT.md](./BASELINE_REPORT.md) | Qwen3-0.6B LoRA 同步 GRPO 三步训练结果和复现参数 |
| [官网需求字段快照](./sources/ospp-project-267bf0030.json) | 需求原文、接口来源和完整本地响应的校验值 |
| [验证记录](./evidence/README.md) | 指标、显存采样、重载校验与 CPU 检查摘要 |
| [算术输入](./data/arithmetic-smoke.jsonl) | 同步基线使用的 8 条测试数据 |

后续在 Fork 上开发，采用以下远端与分支约定：

| 名称 | 用途 |
|---|---|
| `origin` | `https://github.com/Huxingyu/AReno.git`，推送自己的工作分支 |
| `upstream` | `https://github.com/inclusionAI/AReno.git`，获取官方更新 |
| `main` | 保持与官方主线同步 |
| `docs/ospp-async-policy` | 持续维护本目录的课题材料 |
| `feat/...`、`fix/...`、`test/...` | 按可审查的功能或修复拆分开发 |

本机已将默认推送远端设置为 origin，并按 CONTRIBUTING 的约定禁用 upstream 推送。
新环境可以采用相同的远端布局。更新 main 时使用快进合并；有分叉时先检查差异。
功能分支从确认的上游版本建立，把实现提交推送到 origin；向官方提交 PR 时选择所需的功能分支。

```bash
git fetch upstream
git switch main
git merge --ff-only upstream/main
git push origin main
git switch -c feat/async-policy-trainer upstream/main
```

资料分支与功能分支分别维护。需要查看其他分支上的资料时，可以切回资料分支，
或在 GitHub 上选择 `docs/ospp-async-policy`。远端云机器从 Fork 获取已提交的功能分支，
源代码修改在本地完成，遵循仓库的 [AGENTS.md](../../../AGENTS.md)。

模型、虚拟环境、下载缓存、完整运行日志和 checkpoint 由实验存储管理。
本目录保存可读文档、输入样本和小型证据文件，便于在本机与云端之间复用。
