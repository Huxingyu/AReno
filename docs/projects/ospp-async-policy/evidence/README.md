验证记录说明

本目录记录 2026-09-13 已执行的同步训练与代码审查结果。归档过程只读取已有结果，
没有重新运行 GPU 训练，也没有将 CPU 测试结果视为多 GPU 异步验证。

| 文件 | 原始记录 | 处理方式 |
|---|---|---|
| `requirements-audit.json` | `requirements-audit-evidence.json` | 保留检查命令、结果和协调器探针结果 |
| `sync-baseline-metrics.json` | `sync-train-nkIhSy/metrics-summary.json` | 保留所有标量序列，把 `path` 改为实验内相对路径 `metrics` |
| `sync-baseline-gpu.json` | `sync-train-nkIhSy/gpu-summary.json` | 保留全部字段 |
| `sync-baseline-reload.json` | `sync-train-nkIhSy/reload-verification/verification.json` | 保留全部字段 |
| `sync-baseline-sample.json` | `sync-train-nkIhSy/reloaded-sample.json` | 保留回答和奖励，把 `adapter` 改为实验内相对路径 |

同步训练的代码版本为 `60e2fa73f331ea5f5f7245f0793084fbebe9d76c`。
协调器测试依赖本地原型提交 `5248ac2`；原型代码尚未通过本资料分支发布。
原始日志与 checkpoint 保留在实验环境中，数据解释和测试限制见上一级报告。
