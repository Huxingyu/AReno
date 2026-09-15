# 异步策略训练重写：从这里开始

R0–R3 已完成：从干净上游重新实现异步核心，交付设计、统一 bridge、有界调度、失败收尾和真实 CPU 模型验收。没有带入旧异步生产实现。

R4 首轮真实 GPU 验证也已完成：Modal 单机双 L4 上的同步 / 异步 LoRA 短训练、实际 GPU 重叠、NCCL 权重同步、保存重载、两种故障与同容器重启均通过。完整 R4 仍有未验收项目，见 TODO。

- 分支：`feat/async-policy-rewrite`。
- 上游起点：`48d07c54051c41bf36218f99bce1c3697e9ba63c`，2026-09-15 fetch 后创建。
- 原型：`1f73e23` / `ca97cad`，均不在本分支祖先链上。
- 实现与测试提交：`ecb8f3bfe0db9726708a2bad9aeef42b7b4a5e2a`。
- 当前验收：75 项专项契约、66 项选定上游 CPU 回归通过；216 组并发矩阵、80 次生命周期检查通过；固定输入 9 类数值与干净上游完全一致。

先遵循根目录 [AGENTS.md](AGENTS.md)，接续工作只需读这三份材料：

1. [TODO 与验收记录](docs/projects/ospp-async-policy/REWRITE_TODO.md)：阶段状态、实测结果和复现命令。
2. [设计说明](docs/projects/ospp-async-policy/DESIGN.md)：所有权、状态转换、同步和失败契约。
3. [已知问题与测试说明](docs/projects/ospp-async-policy/KNOWN_ISSUES.md)：原反例、迁移说明和补测修复。

实现入口是 `areno.experimental.async_policy.AsyncPolicyPipeline`。模型操作全部经过同一个 `DualEngineBridge`，测试替身只在 tests 中。正常 CPU 测试入口是 `tests/test_async_policy_contracts_cpu.py` 和 `tests/test_async_policy_core_cpu.py`；隔离执行与数值对照工具在 `tests/async_policy_validation/`。

本次原始结果留在忽略的 `runs/async-policy-rewrite/acceptance/`，`overview.json` 是汇总，子目录保存 JUnit、事件、数值和源码 SHA。记录针对提交时的干净工作区；随后只更新验收文档。

原生 GPU 接入位于 `areno/experimental/async_policy/native.py`，仍由同一个 bridge / coordinator 管理准入和版本。GPU 实现提交为 `11c7e3b`；补充样本记录与故障断言的工具提交为 `54bf560`，已推送到本分支。GPU 原始结果位于 `runs/async-policy-rewrite/modal/overview.json`，配套工具在 `tests/async_policy_validation/`，没有另加一批参考文档。

**R4 尚未全部完成。** 本轮只验证 Qwen3-0.6B LoRA、TP=1 / DP=1、短序列，关闭 torch.compile 并使用 eager decode。全参数、CUDA graph / 编译路径、独立上游 GRPO / GSPO GPU 回归、外部中断 / 通信超时以及长时间性能 / 质量实验仍待完成。既有 GRPO loss 保持不变；S1 仍是另一目标函数的独立诊断。

本机验证环境是 `runs/async-policy-rewrite/venv/`，复用已有 Python 3.12 / Torch，只补装项目已声明的 safetensors。公开 config / CLI、项目依赖和 GPU 操作仍遵循根目录 AGENTS。

旧方案和原型审计档案保留在旧工作区，按需查证；没有复制成一大批开工参考文件。
