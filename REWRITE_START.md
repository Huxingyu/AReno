# 异步策略训练重写：从这里开始

R0–R3 已完成：从干净上游重新实现异步核心，交付设计、统一 bridge、有界调度、失败收尾和真实 CPU 模型验收。没有带入旧异步生产实现。

R4 的首版工程交付和计划内验证已完成：Modal 双 L4 上的 LoRA / 全参数训练、实际 GPU 重叠、编译 / CUDA graph、NCCL 同步、故障清理、优化器续训和独立上游 GRPO / GSPO 回归均通过。三种配置各运行三次，共 495 次真实更新，其中 450 次进入性能测量。

`lag=1` 更新吞吐平均为同步的 1.396 倍；`lag=0` 为 0.721 倍。质量存在回退：种子 43 的 `lag=1` 在 32 题、64 token 预算下从训练前 21 题正确降至 16 题，同步结果为 23 题。工程验收完成不代表质量稳定性已证明；完整数据和边界见 TODO。

- 分支：`feat/async-policy-rewrite`。
- 上游起点：`48d07c54051c41bf36218f99bce1c3697e9ba63c`，2026-09-15 fetch 后创建。
- 原型：`1f73e23` / `ca97cad`，均不在本分支祖先链上。
- R0–R3 实现与测试提交：`ecb8f3bfe0db9726708a2bad9aeef42b7b4a5e2a`。
- R0–R3 验收：75 项专项契约、66 项选定上游 CPU 回归；216 组并发矩阵、80 次生命周期；固定输入 9 类数值与干净上游完全一致。
- R4 当前实现 / CPU 验收提交：`9619e1e`；185 项选定 CPU 测试通过，各 GPU 用例的实际 SHA 见 TODO 和运行记录。

先遵循根目录 [AGENTS.md](AGENTS.md)，接续工作只需读这三份材料：

1. [TODO 与验收记录](docs/projects/ospp-async-policy/REWRITE_TODO.md)：阶段状态、实测结果和复现命令。
2. [设计说明](docs/projects/ospp-async-policy/DESIGN.md)：所有权、状态转换、同步和失败契约。
3. [已知问题与测试说明](docs/projects/ospp-async-policy/KNOWN_ISSUES.md)：原反例、迁移说明和补测修复。

实现入口是 `areno.experimental.async_policy.AsyncPolicyPipeline`。模型操作全部经过同一个 `DualEngineBridge`，测试替身只在 tests 中。正常 CPU 测试入口是 `tests/test_async_policy_contracts_cpu.py` 和 `tests/test_async_policy_core_cpu.py`；隔离执行与数值对照工具在 `tests/async_policy_validation/`。

本次原始结果留在忽略的 `runs/async-policy-rewrite/acceptance/`，`overview.json` 是汇总，子目录保存 JUnit、事件、数值和源码 SHA。记录针对提交时的干净工作区；随后只更新验收文档。

原生 GPU 接入位于 `areno/experimental/async_policy/native.py`，仍由同一个 bridge / coordinator 管理准入和版本。正式示例与训练 / 评测数据位于 `examples/async_policy/`，包括 `--save-training-state` / `--resume-from`。验证工具在 `tests/async_policy_validation/`；首轮和补全结果分别保存在忽略的 `runs/async-policy-rewrite/modal/` 与 `runs/async-policy-rewrite/r4-completion/`，各自的 `overview.json` 是结果入口。

首版范围是 Qwen3-0.6B 文本 GRPO completion、每角色单 GPU、TP=1 / DP=1；包含 LoRA 与小模型全参数验证。续训恢复权重、优化器、训练 RNG 和版本，数据迭代器重新开始，不恢复在途 batch。多机、复杂 TP / DP、Agentic、多模态和异步 GSPO 扩展不在首版范围。既有 GRPO loss 保持不变；S1 仍是另一目标函数的独立诊断。

本机验证环境是 `runs/async-policy-rewrite/venv/`，复用已有 Python 3.12 / Torch，只补装项目已声明的 safetensors。公开 config / CLI、项目依赖和 GPU 操作仍遵循根目录 AGENTS。

旧方案和原型审计档案保留在旧工作区，按需查证；没有复制成一大批开工参考文件。
