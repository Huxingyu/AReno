AReno 课题要求与实现核对
核对日期：2026-09-13。

当前主线尚未提供可直接运行的 experimental async policy trainer。同步 GRPO/GSPO、异步生成接口、独立 rollout/train engine、设备分组与原生 GPU 权重同步已经存在。课题的主要新增工作是把这些能力组成有界、可控制策略延迟、能够正确退出的异步训练流水线，并用单机多 GPU 实验验证它。

核对范围与版本

- 官方课题：[267bf0030](https://summer.ospp.ac.cn/org/prodetail/267bf0030?lang=zh&list=pro)。核对了中文及英文描述、产出要求；需求字段快照保存在 [ospp-project-267bf0030.json](./sources/ospp-project-267bf0030.json)。
- 审查时重新 fetch 后的官方 main：`60e2fa73f331ea5f5f7245f0793084fbebe9d76c`，提交时间 2026-09-10 19:53:34 +0800。
- 同步基线分支：`exp/sync-smoke-20260913`，审查时 HEAD 与官方 main 相同。
- 本地异步原型：`feat/async-policy-scheduler`，提交 `5248ac2`，在上述主线基础上新增 3 个文件、298 行。
- 另行检查了尚未合并的 [PR #480](https://github.com/inclusionAI/AReno/pull/480) 及 [Issue #487](https://github.com/inclusionAI/AReno/issues/487)。它们与主线、本地原型分别列示，不能合计为已经接通的功能。

官方要求的边界

英文要求进一步明确：第一版放在 `areno.experimental`，优先支持 GRPO，复用现有 `grpo_loss_fn`，为后续 GSPO 留出空间；使用 bounded queues 和显式 staleness 控制，支持本地 rollout/train 资源分离，保留同步 GRPO/GSPO 行为。

中文第 5 项写的是 bounded queue 配置；英文产出要求同时列出 queue size、inflight rollout work、weight sync cadence，应全部纳入实现范围。

`async-grpo` / `async-gspo` 是课题列举的实验入口形式。课题也接受 experimental trainer class，供 CLI/SDK 后续接入，并非必须第一版就新增两个 CLI 算法入口。

“熟悉 Python async、GRPO 语义、checkpoint、多 GPU、backend lifecycle”等五条是技术能力要求，不能理解为需要重新实现五套底层系统。公开课题文字未规定商业级模型规模、从零预训练任务或固定吞吐提升百分比。

| 课题产出 | 当前主线已有实现 | 尚需完成 |
|---|---|---|
| 1. experimental async policy trainer | `AlgorithmSpec`、实验包发现机制、同步 `PolicyOnlyTrainer` 可复用；`areno/experimental` 只有根 `__init__.py` | 完整异步 trainer、生产与消费调度、真正的 rollout/reward 与 train 重叠。当前没有 `async-grpo` / `async-gspo` 注册 |
| 2. rollout worker 与 train loop 生命周期 | SDK init/close、rollout session、GPU worker 进程启动/退出、RPC 异常传播 | 后台样本生产 worker、队列关闭与 drain/abort、max_steps/数据耗尽/中断/异常的统一停止、唤醒等待者和回收线程/进程 |
| 3. AReno-native 权重同步 | 已有 NCCL GPU 间分桶传输、train/rollout 不同 TP 布局映射、全量权重与 LoRA plan、版本比较和推理状态失效处理 | 在并发流水线中接入同步安全点、同步频率控制、阻止新工作持续插队、确保复制期间源权重稳定、同步失败后的终止或恢复协议 |
| 4. 最小单机多 GPU async GRPO smoke recipe 与文档 | 同步 tiny smoke、设备分离的 CLI 示例、拓扑文档；本机同步三步训练已有实测记录 | 可运行的异步示例、完整配置、预期指标与退出行为、实际多 GPU 重叠证据和同步对照 |
| 5. bounded queue 配置 | 推理并发容量 `max_running_prompts`、权重传输桶 `policy_sync_bucket_mb` | 供训练消费的 ready-batch 有界队列及 backpressure；queue size、inflight work、sync cadence 配置。前两种现有容量限制不等同于此队列 |

可以直接复用的能力

| 能力 | 证据与适用范围 |
|---|---|
| 同步 GRPO/GSPO 主循环 | [PolicyOnlyTrainer](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/api/trainers/policy_only.py#L157)：生成、评分/优势计算、训练依次完成 |
| 每个 prompt 的 n_samples 与 token/logprob 数据 | [RolloutResult / TrainSequence](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/api/models.py#L51)；训练样本包含 tokens、logprobs、mask/边界、advantage、reward 等 |
| 普通 completion 的组内优势和样本构造 | [materialize](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/api/trainers/policy_only.py#L723)、[compute_batch_group_advantages](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/api/advantages.py#L15) |
| Agentic 轨迹与 loss mask | [agentic rollout](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/api/trainers/policy_only.py#L317)、[agentic materialize](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/api/trainers/policy_only.py#L572)；异步 trainer 对此仍需接入验证 |
| reward 评分的可选批内并发 | [score_reward_records](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/api/rewards.py#L90) 支持 reward_fn.parallel_workers；调用会等待该批评分完成 |
| 异步生成 API | [rollout_token_batch_async](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/api/trainer.py#L372)、[CUDA rollout_batch_async](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/api/backend/cuda/backend.py#L444) |
| 训练与生成设备分离 | [CudaBackend.initialize](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/api/backend/cuda/backend.py#L180)；独立 engine、TP/DP 进程组；设备配置及 CLI 已存在 |
| 原生权重传输 | [policy_sync.py](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/engine/policy_sync.py#L162)；按 tensor/chunk 传输，直接写入目标布局 |
| 全量权重或 LoRA 同步计划 | [build_policy_plan](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/engine/policy_sync.py#L28) 根据 adapter registry 选择；LoRA 同步 A/B，不复制冻结基座 |
| 接收新权重后的状态更新 | [receive_policy](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/engine/worker.py#L209)；清理 KV cache、decode graphs、派生推理权重并重建 |
| 现有同步统计 | [同步指标](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/api/backend/cuda/backend.py#L300)：耗时、字节数、tensor 数、带宽；异步队列等待、样本策略延迟与丢弃统计尚缺 |

同步与异步的关键边界

现有 loop 在 [第 201 行](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/api/trainers/policy_only.py#L201) 使用 `asyncio.run(...)` 并等待 rollout 完成，然后 materialize，最后在 [第 246 行](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/api/trainers/policy_only.py#L246) 调用 train。因此，生成阶段内部可以有请求并发，训练流水线仍然顺序执行。

更直接的限制在 [CudaBackend.train](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/api/backend/cuda/backend.py#L503)：

```python
if self._separate_rollout and self._rollout_session_active:
    raise RuntimeError("cannot update policy weights during an active rollout session")
```

只给上层加一个线程或队列，无法在这条现有路径上实现目标并发。需要在实验模式下建立合理的并发协议：独立 train engine 可以更新其自身模型，rollout 使用自己的稳定版本；权重复制必须在合适安全点进行。仅删除保护检查不足以证明正确。

现有 [权重同步](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/api/backend/cuda/backend.py#L282) 在下一次 rollout 开始前按需同步新版本。它解决“把权重搬过去”，尚未解决“允许哪些旧版本样本排队、何时训练、何时暂停两边进行同步”的策略。

SDK 的 global_step、step 时序和 metrics 也以顺序 loop 为前提，见 [Trainer.train](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/api/trainer.py#L413)。异步设计需要明确这些共享状态由谁更新，避免生成线程和训练线程混写同一个 step。

策略延迟与 loss 语义

主线 backend 有 train/rollout 版本计数，RolloutResult 有可选 adapter_version，但 [TrainSequence](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/api/models.py#L74) 没有形成用于异步消费的批次策略版本契约，也没有执行 max_staleness 检查或过期批次处置。

异步样本至少应携带生成时的策略版本，消费时比较版本差；不能把“版本字段存在”当作已经实现了 staleness 控制。应先保持完整 prompt group 的奖励、优势计算与 token/logprob/mask 对齐，再以完整 materialized batch 入队。若未来细化队列粒度，也必须另行保留组装语义。

当前 CUDA [GRPO loss](https://github.com/inclusionAI/AReno/blob/60e2fa73f331ea5f5f7245f0793084fbebe9d76c/areno/api/backend/cuda/losses.py#L160) 的 ratio 为：

```python
ratio = torch.exp(logprobs - logprobs.detach())
```

GSPO 使用对应的序列级表达式。对有限 logprobs，ratio 的数值为 1，但梯度不为零；rollout old_logprobs 用于差异指标，未参与行为策略的重要性比值。因此，保留 rollout logprobs 本身不意味着异步旧策略样本已获得重要性修正。

官网又明确要求复用现有 grpo_loss_fn。合理的首版范围是保留原 loss，明确策略延迟上限与适用条件，验证质量及训练行为；若要改变 loss，需要单独论证，而不能作为异步接线时的隐式改动。这是设计和验证问题，不应因贡献者遵循“复用 loss”要求就判定其不合格。

未合并 PR 与本地原型

| 位置 | 已有内容 | 不应计为已完成的内容 |
|---|---|---|
| [PR #480](https://github.com/inclusionAI/AReno/pull/480)，head `01e6baf20b02394f5fabbde8f170cc1ac4c53b57`，open / 未合并 | `AsyncTrainBatch` 元数据、`AsyncTrainBatchQueue` 有界 FIFO/backpressure/timeout/close/error、`staleness_delta/is_stale` 辅助函数及 CPU 测试文件 | 没有完整 trainer、worker、算法注册、真实权重同步接入或 GPU smoke；不在主线，也不在本地协调器分支 |
| [Issue #487](https://github.com/inclusionAI/AReno/issues/487)，open | 描述完整流水线及拟议配置、验证计划 | issue 内的命令和配置是提案，不能当成现有可执行接口 |
| 本地 `5248ac2` | PolicyPipelineCoordinator（本地原型提交 `5248ac2`）：rollout/train/sync 状态协调、版本推进、timeout/close；7 项 CPU 测试 | 没有队列、后台生产、train loop、算法入口、实际传输调用；主线 backend 未引用该协调器 |

本地协调器还存在已验证的同步等待缺口：begin_sync 先等待活跃工作归零，再设置 _sync_active；等待期间 begin_rollout/begin_train 并不知道已有同步请求。新工作持续进入时，同步可能饥饿。本次探针在同步等待期间连续成功申请了 5 次新 rollout，均使用旧版本 0；结束原始 rollout 后同步才完成。

另外，end_sync(succeeded=False) 只保证版本号不推进，并不回滚已经写入的 tensor。对真实 backend，传输失败可以选择立即停止整条流水线，或提供恢复机制；不能仅凭版本没推进，就继续使用可能部分更新的 rollout 模型。

2026-09-13 已执行的验证及边界

以下为已执行的历史检查命令，在当时的 AReno 环境和对应 checkout 中运行。整理文档时未重复执行这些训练相关测试。

主线命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_algorithms_cpu.py tests/test_dual_engine_backend_cpu.py tests/test_policy_tensor_sync_cpu.py -k 'not real_gloo_collectives' -p no:cacheprovider
```

结果：17 passed，2 deselected。覆盖 registry、fake dual backend、权重 layout 和计划等 CPU 行为。两个真实多进程 Gloo 用例未执行，本次未据此宣称 NCCL 或多 GPU 异步集成已经验证。

本地原型命令（工作目录为 `feat/async-policy-scheduler` 对应的原型 checkout；本资料分支没有该原型测试文件）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" python -m pytest -q tests/test_async_policy_coordinator_cpu.py -p no:cacheprovider
```

结果：7 passed。另有上述 pending-sync 探针；单测通过不代表该缺口已经覆盖。PR #480 的代码 diff 已阅读，本次没有执行该 PR 的测试。

动态列出的主线算法为 `dpo, grpo, gspo, ppo, sft`。既有 [同步基线报告](./BASELINE_REPORT.md) 记录了本机 Qwen3-0.6B、LoRA、3 个真实 GRPO optimizer steps、保存重载及约 3.49 GiB 采样显存峰值；此次沿用该记录，没有重新开展 GPU 训练。该基线验证的是短任务同步链路，不是异步吞吐或训练质量提升。

建议的后续交付顺序

1. 保持现有同步基线，确定实验性 batch 版本契约、有界队列与配置、完整 prompt group 语义。
2. 用 CPU fake backend 跑通生产、消费、staleness、周期同步、耗尽/异常/中断退出，并修复 pending-sync 准入问题。
3. 接入真实独立 engine 与已有原生传输，处理安全点、缓存失效、失败终止及共享 metrics 状态。
4. 在单机多 GPU 上提供 GRPO smoke recipe：至少验证参数更新、实际阶段重叠、队列上限、策略延迟和资源回收。
5. 在相同机器、GPU 数量与分配、模型、精度、序列长度、n_samples 和训练方式下比较同步/异步；记录有效训练吞吐、queue wait、policy lag、丢弃样本和同步开销，较长实验再比较达到相近质量的时间。
6. 在 GRPO 路径稳定后，按需要扩展 GSPO、更多 agentic 场景与 CLI/SDK 接入。

其中 CPU fake 流水线是开发里程碑；完整课题成果仍需要实际单机多 GPU 的 AReno-native 实验路径。


检查摘要见 [requirements-audit.json](./evidence/requirements-audit.json)。原型记录和未合并 PR 的状态均固定于审查日期，后续实施应另行更新。
