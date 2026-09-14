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

**S1：保留既定 loss**

现有 GRPO 的 `exp(logp - logp.detach())` 数值为 1，但梯度仍存在。它与使用行为策略 logprob 作分母的目标不同。原方案已明确保留现有 loss，限制 lag 并做质量观察。S1 不计新增 bug，也不要求在调度重写中改成另一套目标；有限 lag 不保证质量不退化。

**只保留六个必要测试文件**

| 文件，均在 `tests/async_policy_validation/` | 用途 |
|---|---|
| `known_cases.py` | 原四个反例 |
| `protocol_cases.py` | 队列、许可、故障、生命周期、指标、216 组矩阵 |
| `tiny_backend.py` | 425 参数真实 CPU 模型、采样、独立数值 oracle、SGD 和同步适配 |
| `torch_cases.py` | CPU 闭环、初始对齐、部分同步失败、payload 和 S1 |
| `reference.py` | 指定 checkout 的同步 / 异步 materialization 和一步数值对照 |
| `run.py` | 子进程隔离和超时；默认运行契约测试，S1 需显式加入 |

这些测试依赖待实现的异步模块，干净上游阶段出现模块缺失表示尚未实现。它们来自原型审计，不是新分支已通过的结果。旧测试访问 `_cond`、`_queue`、`_producer` 等私有对象；可以改为新接口或故障注入点，保留行为要求，不需要重建旧结构。

旧 reward callback 的 `(prompt, sample_index)` 和小模型通过 prompt ID 查奖励的方式只是测试适配，不是新 API 的要求。N9 当前反例采用快照期望；选择其他所有权方式时，应提供同等可检验的交接保证。

实现对应接口后，从仓库根目录运行：

```bash
python3 tests/async_policy_validation/run.py --output-dir runs/async-policy-rewrite/check
python3 tests/async_policy_validation/run.py --include-semantics-probe --case stale_loss --output-dir runs/async-policy-rewrite/loss-diagnostic
```

第二条诊断预期显示既定目标与另一目标的差异，并可能非零退出。新运行记录自己的 SHA、环境和输出目录；不放松断言 / 容差来修绿，不用过去的 85 项常规通过或历史数值对照代替新实现验证。
