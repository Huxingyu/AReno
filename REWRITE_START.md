# 异步策略训练重写：从这里开始

用户已决定从干净上游重新实现异步核心，携带精简后的方案、问题和测试作为先验。本分支没有旧异步生产实现。

- 分支：`feat/async-policy-rewrite`。
- 上游起点：`48d07c54051c41bf36218f99bce1c3697e9ba63c`，2026-09-15 fetch 后创建。
- 原型：`1f73e23` / `ca97cad`，均不在本分支祖先链上。
- 起点实测：选取的 42 项上游 CPU 回归通过；固定输入的一步同步计算，9 类输出与历史 `60e2fa7` 同步基线最大绝对差均为 0。没有验证新异步实现或 GPU。

先遵循根目录 [AGENTS.md](AGENTS.md)，然后只需读两份开工材料：

1. [重写 TODO](docs/projects/ospp-async-policy/REWRITE_TODO.md)：实现范围、顺序和每一步的验收。
2. [已知问题与验收要求](docs/projects/ospp-async-policy/KNOWN_ISSUES.md)：必须避免的错误，以及如何迁移测试。

测试输入集中在 `tests/async_policy_validation/`，共六个 Python 文件。按当前开发阶段阅读对应测试即可；不需要先读历史日志。新实现的内部类名和结构可以重新设计，保持外部行为要求和数值基准。

首个交付目标是 CPU 上真正闭环的异步控制逻辑：统一状态所有权、完整 prompt group、有界 Q / K、版本和 lag、真实更新计数、同步及失败退出。随后让同一 bridge 路径连接真实双 GPU。复用上游数据语义和 GRPO loss，不把重写扩成算法改造。

**现在从 TODO 的 R1 开始。** 先形成简短的状态所有权和转换说明，再实现并逐项运行测试。不要为了让旧测试导入成功而复制旧生产模块。

当前可用 `python3`、Torch、NumPy、Pydantic、pytest。运行前确认 `areno.__file__` 指向当前 worktree；GPU 环境尚未完整准备。公开 config / CLI、依赖和 GPU 操作按根目录 AGENTS 与当前用户授权执行。

完整旧方案仍在 `docs/ospp-async-policy@1481411`。原始审计日志保留在旧 worktree 的 `runs/async-policy-cpu/`；本次起点预检日志保留在旧 worktree 的 `runs/async-policy-rewrite-start/`。这些是按需查证的档案，不属于开工必读材料，也未复制进本分支。
