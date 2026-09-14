# R0–R3：异步策略流水线设计

本阶段交付实验性 completion 调度核心及可替换的引擎适配协议，保持上游 GRPO 的数据和 loss 语义。入口是 `areno.experimental.async_policy.AsyncPolicyPipeline`。GPU / NCCL 适配属于 R4；不新增算法名称、CLI 或公开 Trainer 配置。

| 唯一所有者 | 管理内容 |
|---|---|
| Supervisor（pipeline 与 bridge 共享） | 生命周期、停止信号、首个异常；失败唤醒所有等待者 |
| Coordinator | Vt / Vr、生成 / 训练 lease、同步请求合并；同一条件锁保护准入与版本提交 |
| Producer | 唯一源迭代器、至多 K 个生产任务；取得许可后才读源，任务入队或终止才归还 |
| 训练主线程 | 消费、batch 决策和指标；后续 checkpoint 也归此线程所有 |

```mermaid
flowchart LR
  A[取得 K 许可] --> B[读源并快照]
  B --> C[Bridge 生成 lease]
  C --> D[CPU 评分与整组 materialization]
  D --> E[有界队列 Q]
  E --> F[归还 K 许可]
  E --> G[Bridge 训练准入及 lag 检查]
  G --> H[至多一次 optimizer update]
  H --> I[同锁提交 Vt 与待同步标记]
  I --> J[关闭新准入并等现有模型操作结束]
  J --> K[独占复制、接收确认、推理状态准备]
  K --> L[发布 Vr 并重新开放准入]
```

状态：`NEW → INITIALIZING → READY → RUNNING → DRAINING → STOPPING → CLOSED`。初始化必须实际复制，即使初始 Vt = Vr = 0。EOF 先等所有已接收任务入队再 drain，最后补齐权重；max_steps / 外部 stop / abort 关闭准入并丢弃剩余工作。任何阶段异常进入 `FAILED`，清理成功后进入 `CLOSED`，但报告始终保留 failed 与原始异常。关闭超时也是失败；允许在操作退出后再次 close 完成清理，不能先成功返回再后台收尾。

生成引擎 session 串行；ASYNC 模式允许一份生成与一份训练重叠，SYNC 模式两个方向都互斥。CPU 评分和 queue / permit 等待不占模型 lease。Vb 在生成准入时捕获；训练准入后检查 `0 ≤ Vt − Vb ≤ lag`，过期整批丢弃，future version 报错。仅适配器明确返回 `stepped=True` 才递增 Vt；零梯度也可能真实 step，不能看 loss 或参数差猜测。提交 Vt 的同一锁内设置同步需求：达到 C，或继续生成将超过 lag。多个同步调用等待同一次传输；超时使用单个单调时钟截止时间。传输返回必须意味着完整复制、接收确认及推理状态准备结束；部分失败不回滚版本冒充恢复，而是禁止后续模型操作。

所有发布数据必须是 CPU、无计算图。读源时复制 prompt，释放生成 lease 前复制 rollout，materialization 再复制训练 payload；普通容器、数值 NumPy 数组、无梯度 CPU tensor 可用，其他对象拒绝。冻结 envelope 只保护元数据；其内部 TrainSequence 由训练消费者独占，生产者不得保留可修改的别名。完整 prompt group 先评分，再复用上游 group advantage 构造 rows；reward 使用上游 `RewardRecord`，可直接读取实际 tokens，文本评分可传入 decoder。

适配器的 initialize / generate / train / transfer / close 必须遵守传入的剩余超时；reward 与源迭代器必须及时返回或合作取消（可观察 pipeline.stop_event）。Python 不能安全强杀任意线程。停止依次唤醒队列 / 许可 / lease 等待者，等待 producer 与其有界任务退出，再等待模型操作并关闭引擎。超时报告仍存活的资源，不提前关闭正在使用的模型。

`pipeline.run()` 无论初始化是否成功都执行一次统一收尾，整个收尾共用一个截止时间；不会失败后自动重置预算再试。单独使用 bridge 时，调用者须把 initialize 也放进 `try/finally: close()` 范围。首个异常原样抛出，收尾等次生异常以及剩余线程 / lease 另存入 report；显式再次 close 可完成后续回收。

验收入口见 [重写 TODO](REWRITE_TODO.md) 与 [已知问题](KNOWN_ISSUES.md)。原始运行输出只写到忽略的 `runs/async-policy-rewrite/`。
