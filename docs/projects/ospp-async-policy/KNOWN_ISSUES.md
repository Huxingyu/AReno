# 已知问题与测试使用

原型的专项审计共有 31 项：15 通过、16 失败。其中 30 项契约用例有 15 项失败，去重为 4 项已知问题 K1–K4、10 项新发现的实现 / 契约缺口 N1–N10；另外 S1 是不同 loss 目标的诊断差异。它们不是同等严重的 16 个 bug，也不能据此估算生产故障率。

| 类别 / 原编号 | 之前发生了什么 | 新实现必须证明 |
|---|---|---|
| 同步准入 K1 / K2 / K4 | 超时预算重复累计、多个同步调用同时拿到许可、SYNC 模式只限制一个调用方向 | 统一单调时钟截止时间；同步单一持有者；两个方向均原子互斥 |
| 生命周期 K3 / N3 / N4 | 收尾超时仍成功返回；iter(data) 抛错后消费者一直等；next(data) 抛错泄漏许可 | 初始化到收尾都在异常边界内；根因可见、等待可退出、许可总能归还；未完成清理就报告失败 |
| 队列关闭 N1 / N2 | abort 后继续交付缓存；close 与 release 的竞态放行新任务 | EOF 可以 drain，abort 不再消费；关闭后所有等待者拒绝新准入 |
| 实际权重 N7 / N8 | initialize 只改标志；复制一半失败后仍可 rollout，版本号却保持旧值 | 初始实际对齐；部分同步失败后禁止使用模型，不能把版本没变当作回滚 |
| 数据所有权 N9 / N10 | frozen 外壳内部共享可变路由张量；features 绕过 CPU / 无图检查 | 全部受支持 payload 遵循 CPU / 无图约束，发布后的 batch 不被生产端后续操作改写 |
| 指标 N5 / N6 | 异常后报告 not_started；等待时间漏掉空轮询 | 退出原因对应真实状态，等待统计覆盖完整过程 |

N7 属于尚未完成的初始化接入；N9 的具体验收依赖快照 / 所有权转移契约。不能把这些描述成所有正常运行都会触发的故障，也不能只改说明文字来掩盖缺口。

上表描述原型历史。新实现已逐项运行原 30 项契约并通过，结果见 [本次验收](REWRITE_TODO.md)。K3 改为验证无法在预算内结束的后端会明确报关闭超时，并在释放 gate 后完成清理；初始同步现在真实执行，因此运行期 / 部分同步故障在初始对齐成功后注入。

重写补测实际暴露并修复了两处边界错误：producer 线程启动失败时，join 未启动线程导致已初始化引擎未被关闭；失败退出时，消费者的空轮询等待未计入总时间。对应回归为 `test_producer_start_failure_still_closes_initialized_engines` 和 `test_failed_run_wait_metric_includes_empty_polls`。追加测试也固定了初始化参数校验、单次关闭预算、首因与次生错误、真实零梯度 step、内存复用隔离和所有等待者的关闭唤醒。

**S1：保留既定 loss**

现有 GRPO 的 `exp(logp - logp.detach())` 数值为 1，但梯度仍存在。它与使用行为策略 logprob 作分母的目标不同。原方案已明确保留现有 loss，限制 lag 并做质量观察。S1 不计新增 bug，也不要求在调度重写中改成另一套目标；有限 lag 不保证质量不退化。

**测试入口**

| 文件，均在 `tests/async_policy_validation/` | 用途 |
|---|---|
| `known_cases.py` | 原四个反例 |
| `protocol_cases.py` | 队列、许可、故障、生命周期、指标、216 组矩阵 |
| `tiny_backend.py` | 425 参数真实 CPU 模型、采样、独立数值 oracle、SGD 和同步适配 |
| `torch_cases.py` | CPU 闭环、初始对齐、部分同步失败、payload 和 S1 |
| `reference.py` | 指定 checkout 的同步 / 异步 materialization 和一步数值对照 |
| `run.py` | 子进程隔离和超时；默认运行契约测试，S1 需显式加入 |
| `fakes.py` | 仅用于测试的确定性适配器与 gate / 故障注入 |

实现位于 `areno/experimental/async_policy/`；测试替身没有放入生产包。`tests/test_async_policy_contracts_cpu.py` 将原 30 项纳入正常 CPU 收集，`tests/test_async_policy_core_cpu.py` 增加 45 项契约。部分测试访问 `_cond`、`_queue` 等内部对象以精确观察等待或注入故障，生产调用仍统一经过 bridge。

新 reward callback 使用上游 `RewardRecord`，可直接读取实际生成 tokens；小模型已移除通过 prompt ID 查奖励的旧适配。N9 采用 CPU payload 快照契约，并额外验证源迭代器、生成缓冲区和 reward 内部的后续修改不会污染训练 rows。

从仓库根目录运行；这里使用已准备的验证 venv，环境准备及完整对照命令见 TODO：

```bash
runs/async-policy-rewrite/venv/bin/python tests/async_policy_validation/run.py --include-core --output-dir runs/async-policy-rewrite/check
runs/async-policy-rewrite/venv/bin/python tests/async_policy_validation/run.py --include-semantics-probe --case stale_loss --output-dir runs/async-policy-rewrite/loss-diagnostic
```

第二条诊断预期显示既定目标与另一目标的差异，本次实际非零退出。新运行记录自己的 SHA、环境和输出目录；不放松断言 / 容差来修绿，不用历史通过状态代替新实现验证。
