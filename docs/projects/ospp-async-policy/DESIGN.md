# R0–R4：异步策略流水线设计

本阶段交付实验性 completion 调度核心及可替换的引擎适配协议，保持上游 GRPO 的数据和 loss 语义。入口是 `areno.experimental.async_policy.AsyncPolicyPipeline`。R4 已增加原生 GPU / NCCL 适配、训练状态 checkpoint 和独立示例脚本；不新增算法名称、主 CLI 入口或公开 Trainer 配置。

| 唯一所有者 | 管理内容 |
|---|---|
| Supervisor（pipeline 与 bridge 共享） | 生命周期、停止信号、首个异常；失败唤醒所有等待者 |
| Coordinator | Vt / Vr、生成 / 训练 lease、同步请求合并；同一条件锁保护准入与版本提交 |
| Producer | 唯一源迭代器、至多 K 个生产任务；取得许可后才读源，任务入队或终止才归还 |
| 训练主线程 | 消费、batch 决策、指标与 checkpoint |

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

**R4 原生 GPU 接入**

`native.NativeCudaEnginePair` 为两个独立单卡分区提供 train / rollout / weight_sync 适配器，复用现有 `TPCluster`、`ArenoWorker`、training pack、既定 GRPO loss 和 NCCL policy transfer。首版只接受已解析到本地的真实 checkpoint、文本 completion、TP=1 / DP=1 和互不重叠的设备；远程模型由启动器先通过 ModelScope 获取。

pair 的 train 端点拥有两组 worker 和共同的 TCPStore；rollout 端点不重复关闭它们。初始化两个分区共用一个截止时间；各次原生 RPC 使用剩余预算。多 microbatch 累积成一次更新，直接读取 worker 的 stepped / global_step，不从 loss 或参数变化猜测更新。配对同步同时发出 publish / receive，并检查双方完成事件，任一方失败即可传播，避免先阻塞等发送者而漏看接收者错误。

收尾先请求两个分区退出，再在同一个预算内 terminate / kill 未合作的进程并 join，最后关闭结果泵、队列和进程句柄。无法确认资源退出就报告超时。pair 不另外维护 Vt / Vr 或同步准入，所有使用均通过 bridge；部分传输失败后由共享 supervisor 禁止继续运行。

bridge 将同一个 supervisor stop event 绑定到支持 `bind_stop_event` 的端点。原生启动、RPC 和配对同步等待检查它；取消先让等待线程以 `PipelineClosed` 退出、释放 lease，再由统一收尾关闭 worker。这修复了原先先 join producer、却无法打断其长 RPC 等待的清理缺口。普通 CPU 协议适配器无需实现这个可选绑定。

GPU 验证工具中的 checkpoint 保存由训练消费者在 lease 内发起；同步核验在独占同步区内导出两端张量。专用验证 worker 记录真实操作、显存、编译 / graph 重放次数和 profiler 事件，故障注入只存在于验证工具。`max_steps` 仍按既定契约立即停止，因此最后 Vt=5 / Vr=4 合法；最终保存的是训练版本 5。重复性能实验使用 eager 模式；编译 / graph 正确性独立验收。

`save_training_checkpoint` 在同目录的临时路径内保存权重、optimizer state、CPU / CUDA 训练 RNG 与更新计数，最后写入包含参数布局、optimizer 配置和文件校验和的 manifest，再原子重命名发布。失败不会覆盖已有成功 checkpoint。恢复前核对布局、配置与校验和，通过后才载入优化器。新增 worker 操作位于原有枚举末尾，既有命令编号和默认 SDK 路径保持不变。

`NativeCudaEnginePair(resume_from=...)` 提供 `initial_policy_version`，调用者用它初始化唯一的 coordinator；初始化强制同步恢复后的权重，再开始新 batch。正式示例恢复权重、optimizer、训练 RNG 和版本，但数据源从头开始，在途 batch 不持久化。GPU 续训验收使用固定输入比较连续训练与重建进程后的续训，最终 392 个 LoRA 张量和 optimizer state 完全一致。

验收入口见 [重写 TODO](REWRITE_TODO.md) 与 [已知问题](KNOWN_ISSUES.md)。原始运行输出只写到忽略的 `runs/async-policy-rewrite/`。

## Q3：staleness 的精确定义

Vt 是训练侧成功 optimizer update 的计数，Vr 是已经完整复制并确认的生成版本，
Vb 是当前 batch 在生成准入时捕获的版本。训练只接受 `0 <= Vt - Vb <= max_policy_lag`，
过期整组丢弃，未来版本报错。`max_policy_lag` 不会切换损失函数或自动校正分布偏差。

C 按实际 optimizer update 计数；同步既可由 C 到期触发，也可由训练与生成版本差
超过 lag 触发。因此当前实现的正常触发间隔为 `min(C, lag + 1)`，不应在 lag=1 下
把 C=2 与 C=4 当作两个独立有效 cadence 来比较。

lag=0 保证的是训练准入时 batch 使用当前权重。producer 仍可以在 update 前开始
生成，update 后完成的旧 batch 会被丢弃；这会改变实际训练题目、随机数消耗和时序。
因此它不等价于逐步串行的 sync loop，也不存在“每批次必然过期”的规则。

默认 GRPO 数值 ratio 为 1，梯度非零；这是既定目标。可选的
`grpo_offpolicy_loss_fn` 使用记录的 old_logprobs 作为分母，并保留 response mask 与
clipping。该选择与 lag 配置正交，只用于显式消融；它不承诺消除全部异步偏差。

## 2026-09-16 原生接口收敛

native 使用 `ArenoEngine.from_pretrained(start=False)` 构建两侧，并在启动前保留
显式 LoRA seed 和测试 worker。`ClusterCallHandle.done()/wait()` 支持不取消请求的
轮询；结果仍由 `result()` 取得。native 不再访问 pending、结果泵或 rendezvous 私有字段。

关闭先对两分区调用公开 `request_shutdown()`，再分配剩余总预算进行 `close()`。
engine 返回 WorkerExit；超时也保留退出证据，另一分区仍会被尝试清理。失败启动的
等待截止时间不重置，清理另有共享 5 秒预算。该额外清理时间是公开文档中的明确语义。

U7 采用独立 bridge 的 A 路线。公开 Trainer/CLI 算法注册仍未实现；当前上游 reward
已是 RewardRecord，无需新增旧式 callback 适配。性能与拓扑扩展见
[EXTENSION_DESIGN](EXTENSION_DESIGN.md)，维护者决策见 [decisions](evidence/decisions.md)。
