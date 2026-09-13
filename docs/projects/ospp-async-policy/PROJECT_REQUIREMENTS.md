AReno 原生异步策略训练器：中英文合并需求与执行清单

项目编号：267bf0030。整理日期：2026-09-13。

本文将官网中文与英文的项目说明、产出要求、技术要求合并为本地实施依据，并保留全部相关原文。文中“官方要求”直接对应官网字段；“本地建议”“执行清单”是我们为落实要求拟定的工作安排，具体接口名称、默认值和检查方法属于设计选择。

项目名称为“实现 AReno 原生异步策略训练器”，成果仓库为 [inclusionAI/AReno](https://github.com/inclusionAI/AReno/)。原始来源如下：

- [中文项目链接](https://summer.ospp.ac.cn/org/prodetail/267bf0030?lang=zh&list=pro)。
- [英文项目链接](https://summer.ospp.ac.cn/org/prodetail/267bf0030?lang=en&list=pro)。英文引文依据官网接口中的英文字段，本次文档整理未依赖网页渲染结果。
- 官网详情接口：`POST https://summer.ospp.ac.cn/api/getProDetail`，请求 JSON 为 `{"programId":"267bf0030","type":"org"}`。
- [官网需求字段快照](./sources/ospp-project-267bf0030.json)：`programDesc` / `programDescEN`、`outputRequirement` / `outputRequirementEN`、`techRequirement` / `techRequirementEN`。

中英文内容存在详略差异。本地计划同时纳入两者明确写出的要求；本文不假定某种语言具有更高的验收优先级。官网未给出的默认参数、性能阈值等，不自行标为官方规定。

合并后的项目目标是：实现一个 AReno-native、实验性的异步策略训练器，在单机上使 rollout/reward 样本生产与策略训练重叠执行，并在单机多 GPU 条件下验证对 RL 训练吞吐的影响。

背景中的现有同步流程是：生成每个 prompt 的 n_samples 个 completion 或 agentic trajectories → reward 评分 → group-relative advantages → TrainSequence → backend train / optimizer step。官方列出的动机场景包括长输出 RLVR、慢工具或环境交互的 Agentic RL、单机多 GPU 资源分离，以及较大的 n_samples。它们说明为什么需要流水线；原文没有为每种场景规定独立的速度指标。

首版范围中需要明确保留以下官方约束：

| 编号 | 合并后的约束 | 来源 |
|---|---|---|
| R1 | 能力放在 `areno.experimental`，以实验特性提供 | 中文目标；英文 `live under areno.experimental` |
| R2 | 首版优先支持 GRPO | 英文 `support GRPO first` |
| R3 | 复用现有 `grpo_loss_fn` | 英文 `reuse existing grpo_loss_fn` |
| R4 | 设计保留后续支持 GSPO 的空间 | 英文 `keep the design generic enough for GSPO later` |
| R5 | 使用有界队列，并显式控制策略延迟 staleness | 中英文背景；英文 `use bounded queues and explicit staleness control` |
| R6 | 支持本地 rollout/train 资源分离，使生产与训练可以重叠 | 中英文目标；英文 `support local rollout/train resource separation` |
| R7 | 保持现有同步 GRPO/GSPO 行为 | 英文 `keep existing synchronous GRPO/GSPO behavior unchanged` |
| R8 | 保持 AReno-native，不新增 vLLM/SGLang 等外部推理 runtime 依赖 | 中英文背景 |
| R9 | 保持 single-node、self-contained 的使用体验和本地可调试性 | 中英文背景 |

中文目标列举了 `areno.experimental.async_policy`、`async-grpo` / `async-gspo` 实验算法入口，或先提供 experimental trainer class 供 CLI/SDK 后续接入。因此，首版可以先提供可运行的实验 trainer class；同时新增两个 CLI 算法名称不是官网明确规定的必选交付方式。

以下五项是官方产出要求的逐条合并。中文与英文列保留原文，最后一列说明我们要交付的内容。

| 编号 | 官方中文原文 | 官方英文原文 | 合并后的工作目标 |
|---|---|---|---|
| D1 | 1. 设计并实现 experimental async policy trainer | 1. Experimental async policy trainer implementation | 实现可运行的 experimental async policy trainer；首版 GRPO，复用现有 loss，让生产和训练形成流水线。 |
| D2 | 2. 实现 rollout worker 与 train loop 的生命周期管理 | 2. Rollout worker lifecycle management | 管理后台 rollout 样本生产 worker 与 train loop 的启动、运行和停止，协调它们之间的依赖。 |
| D3 | 3. 实现 AReno-native 权重同步方案 | 3. Weight sync mechanism suitable for AReno’s local backend | 提供适合 AReno 本地 backend 的权重同步方案，接入流水线中的策略版本更新。可复用主线已有原生传输实现。 |
| D4 | 4. 新增最小 smoke recipe，说明如何在单机多 GPU 上试跑 experimental async GRPO | 4. Documentation and a minimal smoke recipe | 交付说明文档和最小 smoke recipe，说明如何在单机多 GPU 上试跑实验性 async GRPO。 |
| D5 | 5.新增 bounded queue 配置 | 5. Config knobs for queue size, inflight rollout work, and weight sync cadence | 配置有界 ready-sample 队列；同时提供 queue size、inflight rollout work、weight sync cadence 三类控制项。 |

英文第 5 项比中文更详细。“有界队列配置”在我们的合并范围内包括三项不同的控制，不能只实现一个推理并发数量限制。

| 控制项 | 官方来源 | 实施时要明确的语义（本地设计任务） |
|---|---|---|
| Queue size：队列容量 | 英文产出第 5 项；中文有界队列要求 | 明确容量单位，限制已准备好但尚未训练的批次数量；队满时对生产端施加 backpressure |
| Inflight rollout work：在途生成工作量 | 英文产出第 5 项 | 明确按任务还是批次计数，限制正在生成及尚未入队的工作，避免只限制队列却放任上游无限生成 |
| Weight sync cadence：权重同步频率 | 英文产出第 5 项；中英文背景中的周期性同步 | 明确触发条件和计数单位，例如按完成的 optimizer updates 计数 |
| Staleness：样本策略延迟 | 英文首版目标；中英文背景 | 记录生成时策略版本，消费时检查延迟并处置过期样本；上限如何表达、丢弃还是报错由设计明确 |

官网没有指定以上参数的字段名、CLI flag、默认值、队列实现库、线程/进程数量或固定同步算法。Issue #487 中出现的配置名称和数值属于提案，不能当成官网已经规定的参数。

以下五项是官方技术要求的中英文对照，它们描述完成项目需要掌握的知识，也指导我们阅读和复用现有模块。

| 官方中文原文 | 官方英文原文 |
|---|---|
| 1. 熟悉 Python async / multiprocessing / queue backpressure | 1. Python async and multiprocessing |
| 2. 熟悉 GRPO/GSPO 的 rollout logprobs、advantages 和 loss 输入语义 | 2. GRPO/GSPO loss input semantics |
| 3. 熟悉 PyTorch checkpoint/state_dict 或分布式权重同步基础 | 3. PyTorch model state or checkpoint synchronization |
| 4. 熟悉单机多 GPU 资源划分和 CUDA worker 生命周期 | 4. Single-node multi-GPU resource management |
| 5. 熟悉 AReno Trainer、PolicyOnlyTrainer、algorithm registry 和 backend lifecycle | 5. AReno trainer/backend lifecycle |

下面是我们的执行清单，依据 2026-09-13 的仓库审查整理。勾选仅表示已经有对应本地证据，不表示整项课题已验收。详细状态见 [仓库要求核对报告](./REQUIREMENTS_AUDIT.md)。

2026-09-14 已进一步整理为可评审的 [完整实施方案](./IMPLEMENTATION_PLAN.md) 和 [逐项 TODO](./TODO.md)，包含任务依赖、完成条件、测试矩阵与 8 周目标排期。下面保留需求层面的阶段概览；日常执行与证据更新以 TODO 为入口。

- [x] 建立本机同步基线：Qwen3-0.6B、LoRA、3 个真实 GRPO optimizer steps、checkpoint 保存与重载。证据见 [BASELINE_REPORT.md](./BASELINE_REPORT.md)。
- [x] 完成中英文要求对照及当前代码能力核对，记录主线已有能力与缺口。
- [ ] D1 / R2–R4：确定实验 trainer 的输入、配置与调用方式；首版复用现有 GRPO loss，为 GSPO 留出扩展点。
- [ ] D1 / R5：定义异步批次数据与生成策略版本，保持 prompt group、tokens、logprobs、mask、reward 和 advantages 的对应关系。
- [ ] D5 / R5：实现并接通有界 ready-batch 队列、生产端 backpressure 和 inflight 工作量限制。
- [ ] D1 / R5：实现训练消费侧的 staleness 检查与过期样本处置，明确计数、丢弃和停止规则。
- [ ] D2：实现后台生产 worker、train loop 和统一生命周期；处理数据耗尽、max_steps、用户中断、reward 异常、worker 失败及等待中的队列操作。
- [ ] D3 / R6：接入真实独立 rollout/train engine 和 AReno 原生权重同步，定义同步安全点、周期和成功后版本推进。
- [ ] D2 / D3：修复本地协调器在同步等待期间仍放行新工作的缺口，防止同步持续被推迟；明确传输失败后终止或恢复的行为。
- [ ] D1 / D2 / D3：先以 CPU fake backend 跑通端到端流水线，再接真实 backend；两阶段都验证清理和错误传播。
- [ ] R7 / R8：完成现有同步路径回归，保持实验能力可选，维持 AReno-native 运行路径。
- [ ] D4：编写并实际运行单机多 GPU async GRPO smoke recipe，保存命令、配置、代码版本、日志、参数更新和退出证据。
- [ ] 项目目标：在相同资源和工作负载下比较同步/异步吞吐，记录策略延迟、同步开销及样本丢弃情况。
- [ ] D4：完成用户文档、设计说明、复现步骤、实验结果和当前限制说明。

上述清单的建议推进顺序是：同步基线 → 批次/队列/版本契约 → CPU 完整流水线 → 真实 backend 与原生同步 → 云端多 GPU smoke → 吞吐对照与文档。GSPO 的具体接入、更多 Agentic 场景和 CLI 体验完善可安排在 GRPO 首版之后；首版应保留相应扩展空间。

现有代码可以复用的部分包括 `PolicyOnlyTrainer` 的 rollout 与样本构造逻辑、组内优势计算、原生生成和训练引擎、GPU 资源分组、NCCL 权重传输及实验算法注册机制。主线审查基线为 `60e2fa73f331ea5f5f7245f0793084fbebe9d76c`。本地协调器提交 `5248ac2` 和未合并的 PR #480 都是组件层工作，尚未形成完整 trainer；整合方案应以实际代码为准。

实施时有三处已知边界需要纳入设计说明：现有 CUDA backend 在独立 rollout session 活跃时拒绝训练；本地协调器的 pending-sync 准入存在缺口；现有 CUDA GRPO/GSPO loss 的 ratio 未采用 rollout old_logprobs 进行行为策略重要性修正。最后一点与“复用 grpo_loss_fn”应同时记录：首版保持现有 loss，明确策略延迟限制并验证影响，不能把自动修正旧策略偏差当作现成能力。细节和代码位置见要求核对报告。

下面是建议采用的本地验收检查。这些是我们验证官方交付的具体方法，官网没有逐字规定每条断言、指标名称或测试规模。

| 检查层次 | 本地检查内容 | 对应官方目标 |
|---|---|---|
| 批次与训练语义 | 保持完整 prompt group 的优势计算；检查 tokens/logprobs/mask 对齐，保证异步调度不破坏现有 loss 输入 | D1；技术要求第 2 项 |
| 队列与 backpressure | 队列和 inflight 工作量不超过配置；满/空等待可结束；正常结束、异常和关闭不会使消费者永久等待 | D2、D5 |
| 策略版本与 staleness | 批次版本对应实际生成权重；过期批次按设计处理；版本不能仅凭任务提交就提前推进 | D1、D3；R5 |
| 权重同步 | 使用 AReno 原生路径；同步期间源和目标处于安全状态；成功后再发布版本；部分失败后按协议终止或恢复 | D3；R6、R8 |
| 生命周期 | 验证正常结束、max_steps、中断、reward 异常和 worker 失败，检查无遗留后台线程/进程 | D2 |
| CPU 流水线 | 用 fake backend 验证生产、排队、消费、同步、停止完整流程及可预期的重叠关系 | D1、D2、D3、D5 的开发验证 |
| 实际多 GPU smoke | 检查有限 loss/gradient、实际参数更新、原生同步、阶段重叠、队列和 staleness 指标以及资源回收 | D1、D3、D4 |
| 同步回归 | 原有 GRPO/GSPO 默认路径继续按原行为工作；实验入口不改变默认训练流程 | R7 |
| 吞吐对照 | 同一机器、GPU 数量与分配、模型、训练方式、精度、token 长度和 n_samples；报告有效训练吞吐、等待时间、同步开销和丢弃样本 | 官方“验证是否提升吞吐”的目标 |

CPU fake backend 是开发阶段验证，不能代替真实单机多 GPU smoke。短程 smoke 证明链路可运行；训练质量或收敛改善需要更充分的实验。官网未规定固定加速百分比，实验应如实记录改善、无改善或退化的情形。

资源安排属于我们的实施建议：本机已验证能运行上述 0.6B LoRA 同步短任务，可继续承担 CPU 调度开发与本地回归。真实多 GPU 验证计划使用用户可提供的云端机器；可从 2×24 GB GPU、64 GB 主机内存估算准备，再按模型、序列和批量实测容量。这个配置、LoRA 训练方式以及模型大小都不是官方规定。本机采用外置存储上的 100 GiB 实验镜像保存模型、环境和运行制品；本文档在 Fork 中持续维护。

以下附录保留官网相关中文与英文原文，去除 HTML 样式、空条目，保留文字内容；便于与上面的合并条目逐项核对。

官方中文项目说明（programDesc）：

> 背景
>
> AReno 当前的 GRPO/GSPO 实现共用 PolicyOnlyTrainer，每个 step 按顺序执行：
>
> rollout 生成 n_samples completions 或 agentic trajectories
>
> reward function 评分
>
> 计算 group-relative advantages
>
> 构造 TrainSequence
>
> 调用 backend train 执行 optimizer step
>
> 这个设计符合 AReno 的 single-node、self-contained 定位，容易理解和调试。但在下面这些场景中，同步 loop 会让 GPU 或 Python Runtime 出现明显等待：
>
> 长输出 RLVR：max_new_tokens 较大，rollout 时间远高于 train 时间
>
> Agentic RL：工具调用、环境交互、sandbox 操作、reward 计算可能成为瓶颈
>
> 单机多 GPU：有机会将 rollout engine 和 train engine 分配到不同设备组，形成本地异步流水线
>
> 大 n_samples：每个 prompt 需要采样多条 completion，rollout 阶段更容易成为端到端 step time 的主导项
>
> 基于对业界其他开源后训练框架的调研分析，我们充分看到了异步 GRPO 的价值：通过后台 rollout worker、样本队列、staleness 控制和周期性权重同步，让 rollout 和 train 重叠执行。AReno 可以借鉴这一类系统思想，但需要保持 AReno-native：不依赖 vLLM/SGLang 等外部推理后端，不破坏本地一体化训练体验。
>
> 目标
>
> 实现 AReno-native async policy trainer，用于验证 rollout/train pipeline 并发是否能提升单机多 GPU 上的 RL 训练吞吐。
>
> 该能力应优先作为 experimental feature 提供，例如：
>
> 新增 areno.experimental.async_policy
>
> 注册实验性算法入口，例如 async-grpo / async-gspo
>
> 或提供 experimental trainer class，供 CLI/SDK 后续接入
>

官方英文项目说明（programDescEN）：

> Background
>
> GSPO and GRPO currently share PolicyOnlyTrainer. Each step performs rollout, reward scoring, group-relative advantage calculation, train batch materialization, and then a backend training step.
>
> This is a good default for AReno’s single-node and self-contained positioning. However, for long-output RLVR, large n_samples, or agentic RL with slow tool/environment calls, the synchronous loop can leave training resources waiting for rollout or reward computation.
>
> Other open source project’s implementation demonstrates the value of decoupling generation and training through rollout workers, a ready-sample queue, staleness control, and periodic weight synchronization. AReno should explore the same system idea in an AReno-native way: no external vLLM/SGLang runtime dependency, no change to the stable default GRPO path, and no loss of local debuggability.
>
> Target
>
> Build an experimental AReno-native async policy trainer that overlaps rollout/reward production with policy training on a single node. The first version should:
>
> live under areno.experimental
>
> support GRPO first
>
> reuse existing grpo_loss_fn
>
> keep the design generic enough for GSPO later
>
> use bounded queues and explicit staleness control
>
> support local rollout/train resource separation
>
> keep existing synchronous GRPO/GSPO behavior unchanged
>

官方中文产出要求（outputRequirement）：

> 1. 设计并实现 experimental async policy trainer
>
> 2. 实现 rollout worker 与 train loop 的生命周期管理
>
> 3. 实现 AReno-native 权重同步方案
>
> 4. 新增最小 smoke recipe，说明如何在单机多 GPU 上试跑 experimental async GRPO
>
> 5.新增 bounded queue 配置
>

官方英文产出要求（outputRequirementEN）：

> 1. Experimental async policy trainer implementation
>
> 2. Rollout worker lifecycle management
>
> 3. Weight sync mechanism suitable for AReno’s local backend
>
> 4. Documentation and a minimal smoke recipe
>
> 5. Config knobs for queue size, inflight rollout work, and weight sync cadence
>

官方中文技术要求（techRequirement）：

> 1. 熟悉 Python async / multiprocessing / queue backpressure
>
> 2. 熟悉 GRPO/GSPO 的 rollout logprobs、advantages 和 loss 输入语义
>
> 3. 熟悉 PyTorch checkpoint/state_dict 或分布式权重同步基础
>
> 4. 熟悉单机多 GPU 资源划分和 CUDA worker 生命周期
>
> 5. 熟悉 AReno Trainer、PolicyOnlyTrainer、algorithm registry 和 backend lifecycle
>

官方英文技术要求（techRequirementEN）：

> 1. Python async and multiprocessing
>
> 2. GRPO/GSPO loss input semantics
>
> 3. PyTorch model state or checkpoint synchronization
>
> 4. Single-node multi-GPU resource management
>
> 5. AReno trainer/backend lifecycle
>

本地完整接口响应的 SHA-256：`7a9188116bf18c9f16ff9459f733f809d842f91ac968be2ee7bf45695f488f1f`。

仓库中的来源 JSON 保留了需求相关字段及其原始文字，并附来源元数据；上述校验值对应本地完整响应。
