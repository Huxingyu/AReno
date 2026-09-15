# 异步策略训练器：下一阶段 TODO（v3）

日期：2026-09-15。基线：`feat/async-policy-rewrite` @ `a35a580`，上游起点 `48d07c5`。
本表接续 [REWRITE_TODO.md](./REWRITE_TODO.md)（R0–R4 已勾选）和原 [TODO.md](./TODO.md)（v2，OSPP 流程与 T00–T16）。
状态纪律不变：区分"计划 / 代码存在 / 已验证"，只有证据到位才勾选。

> 2026-09-16 已完成逐项源码审查并开始执行。详见 [审查报告](NEXT_REVIEW.md)、[扩展设计](EXTENSION_DESIGN.md)、[决策记录](evidence/decisions.md) 和 [PR 分支说明](PR_SERIES.md)。下文保留 a35a580 的现状描述；新代码的 CPU 证据与待补 GPU 验收分别记录，不继承旧提交的 GPU 通过状态。

## 0. 完成度判定

**结论：首版工程交付完成，项目未完成。** 差在三处：质量稳定性未证明、代码尚不具备上游可合入形态、OSPP 流程一项未动。

| 官方交付 | 状态 | 依据 | 缺口 |
|---|---|---|---|
| D1 experimental async trainer | 已验证 | `AsyncPolicyPipeline`，CPU 92 项、双 L4 五组 GPU 用例 | 可选集成尚缺 `register_algorithm` / 公开 `Trainer`；官方允许先提供 experimental class，非 D1 的必选缺口 |
| D2 生命周期管理 | 已验证 | Supervisor / 统一 close；SIGINT、SIGTERM、worker 强退、NCCL 阻塞均回收 | 无 |
| D3 AReno-native 权重同步 | 已验证 | 复用 `policy_sync` NCCL，392 张量逐项相等 | 每次同步等待在途 rollout 结束（约 2 s / update） |
| D4 文档与 smoke recipe | 部分 | `examples/async_policy/train.py`，中文项目文档 | `docs/` 下无面向用户的英文页面；示例硬编码 flash attn 与 base model 名 |
| D5 Q / K / C 配置 | 已验证 | `AsyncPolicyConfig` 四项 + staleness | K>1 只并发 CPU 阶段，GPU 生成仍串行 |

| 官方约束 | 状态 | 备注 |
|---|---|---|
| R1 experimental 包 | 满足 | |
| R2 GRPO 优先 | 满足 | |
| R3 复用 `grpo_loss_fn` | 满足 | 但该 loss 的 ratio 恒为 1，异步下无行为策略修正，见 Q2 |
| R4 GSPO 可扩展 | 未验证 | `native.train` 硬编码 `grpo_loss_fn` |
| R5 有界队列 + staleness | 满足 | |
| R6 资源分离重叠 | 满足 | trace 证明跨卡内核重叠 |
| R7 同步路径不变 | 满足 | 干净上游 GRPO / GSPO 回归逐项一致；核心改动仅 82 行新增 |
| R8 无外部推理依赖 | 满足 | |
| R9 单机可调试 | 满足 | |

## 1. P0：质量与语义（Q）

- [ ] **Q1 种子 43 退化归因。** 三种子、32 题、64 token 上限下 lag=1 从 21 降到 16，同步 23；26/32 回答被截断。
  - 步骤：种子扩到 ≥5；训练长度设 64/256，固定 checkpoint 分别按 64/256 评测；评测集扩到 ≥128 题并保留原 32 题子集；记录 token 上限命中率、平均回答长度、reward 曲线和参与更新的 prompt IDs。同步 / lag=1 / lag=0 同矩阵。
  - 完成条件：分开报告固定 checkpoint 的评测长度效应、固定评测预算的训练长度效应及同预算的调度差异。复现或未复现均如实记录；更大样本未复现不等于证明长期无退化。
  - 证据：`runs/async-policy-rewrite/quality-v2/`，`benchmark_run.py` 参数化。
- [ ] **Q2 行为策略比值消融。** 现有 loss 用 `exp(logp - logp.detach())`，比值恒 1、clip 无效；`layout.old_logprobs` 已在 pack 里但只用于指标。
  - 步骤：在 `areno/experimental/async_policy/` 内新增可选 `grpo_offpolicy_loss_fn`，分母改用 `old_logprobs`，保留 clip；不改上游 `grpo_loss_fn`（R3 / R7）。`native.train` 接受 `loss_fn` 参数。用 S1 用例验证数值差异符合预期。
  - 完成条件：Q1 矩阵上多一列"lag=1 + off-policy ratio"，给出质量对照。
  - 评审点：这是算法决定，先发导师；官方要求复用现有 loss，消融结果只作为建议。
- [x] **Q3 staleness 语义文档化。** lag 是训练准入阈值并可提前触发同步，不选择 loss。lag=0 不等价于串行同步，可能生成并丢弃旧版本数据；不能声称每批必然过期。
  - 完成条件：DESIGN 增加一节；`AsyncPolicyConfig` docstring 说明各字段单位。

## 2. P0：上游可合入性（U）

本节每项都读过对应上游源码后写成；行号以 `a35a580` 为准。标注"未核实"的地方是我没读到的代码，动手前先看。

### U1 去除 `native.py` 对 engine 私有成员的耦合

2026-09-16：公开接口与 native 迁移已实现并通过相关 CPU 测试；GPU faults / extended-faults 待重跑。下表是基线问题，不是新实现仍有的访问。

**现状。** `native.py` 触碰了 `TPCluster` / `ClusterCallHandle` 的 7 处私有成员，并子类化重写了 2 个私有方法。上游 `protocol.py` 任何重构都会静默破坏 GPU 路径，CPU 测试发现不了。

| `native.py` 位置 | 访问的私有成员 | 上游位置 | 为什么要绕过 |
|---|---|---|---|
| `_wait_handles` | `handle._pending.event` | `protocol.py:219` `ClusterCallHandle` | `result(timeout)` 超时会 pop 掉 pending 并抛错，等于取消，不能用来轮询；需要"只等不取消" |
| `_make_cluster` 子类 | 重写 `_wait_for_worker_ready` | `protocol.py:412` | 上游固定 0.2 s 轮询、无截止时间、无 stop event |
| `_make_cluster` 子类 | 重写 `_abort_start` | `protocol.py:389` | 上游 `_abort_start` 清空 `processes`，pair 无法事后核对 exitcode |
| `close` | `_pump_stop`、`_pump_thread`、`_pending_lock`、`_pending_calls`、`_finish_pending_call`、`_rendezvous_store` | `protocol.py:567` `close()` | 上游 `close()` 固定 5 s grace + terminate，无 kill、无超时参数、不返回退出信息、不唤醒等待中的调用 |
| `initialize` | 手工拼 `EngineConfig` 18 行 | `api.py:208` `ArenoEngine.from_pretrained` | 没有绕过的必要，是重复实现 |

**方案：给上游 `protocol.py` 加最小公开接口，全部是带默认值的新参数，默认行为不变。**

1. `ClusterCallHandle.done() -> bool` 与 `ClusterCallHandle.wait(timeout: float | None) -> bool`：只等待 `_pending.event`，不 pop、不抛。`result()` 保持不变。约 8 行。
2. `TPCluster.close(*, timeout_s: float | None = None) -> list[WorkerExit]`：`timeout_s=None` 保留旧 grace / terminate 路径；关闭唤醒和关闭后拒绝提交是明确修正，不声称所有失败行为完全一致。给定时按三段分配预算（SHUTDOWN grace、terminate、kill），返回每个进程的 `(rank, pid, exitcode, alive)`；关闭前用 `_finish_pending_call` 以 `RuntimeError("cluster closed")` 唤醒所有 pending call；仍有 `alive` 时抛 `TimeoutError`。`WorkerExit` 是新 dataclass。实现还需处理提交/关闭竞态、结果重复完成及 pipe feeder，不能按约 30 行估算。另增 `request_shutdown()`，让两分区先发退出通知再等待。
3. `start_partitioned_clusters(..., *, timeout_s: float | None = None, stop_event: threading.Event | None = None)`：透传到 `_wait_for_worker_ready(pending, deadline, stop_event)`；超时或 stop 时先 `_abort_start()` 再抛。`_abort_start` 返回退出信息，失败时保留可观察状态。启动等待共享截止时间；失败清理另有两分区共享的 5 秒预算，必须明确写入接口语义。
4. `native.initialize` 改用 `ArenoEngine.from_pretrained(..., start=False, cluster_kwargs={"world_spec":..., "partition":...})` 构造两个引擎，与 `backend.py:225-259` 同一写法；再调 `start_partitioned_clusters(train.cluster, rollout.cluster, world, timeout_s=..., stop_event=...)`。RPC 仍用 `cluster.submit` 而不是 `ArenoEngine.generate_rollout` / `step`，因为那两个是无超时的 `cluster.call`。
5. `native.close` 先对两个分区调用 `request_shutdown()`，再按剩余预算分配 `cluster.close(timeout_s=...)`；即使一侧失败也尝试另一侧，收集公开退出信息到 `worker_exits`。

- 完成条件：`grep -n "\._pending\|_pump\|_rendezvous\|_wait_for_worker_ready\|_abort_start" areno/experimental/async_policy/native.py` 为空；`tests/test_async_policy_native_cpu.py` 8 项通过；Modal `faults` 与 `extended-faults` 阶段重跑通过。
- 上游测试：在 `tests/test_protocol_cpu.py`（未核实是否用 CPU worker，若不是则参考 `test_dual_engine_backend_cpu.py` 的 fake worker）加 `close(timeout_s)` 返回退出信息、`done()` 不取消、`start_partitioned_clusters` 超时中止三个用例。
- 顺带：`training_state.py` 读写 `worker._global_step`，同在 engine 层可接受，但建议给 `ArenoWorker` 加 `global_step` 只读属性。
- 评审点：接口草案先发导师；这是 PR 4 的前置，也可独立成小 PR。

### U7 与 `CudaBackend` 的关系

**现状。** 流水线绕过 `areno/api/backend/cuda/backend.py`，直接驱动 worker。绕过的不是偶然，是 backend 三处状态与并发训练冲突：

| `CudaBackend` 状态 | 位置 | 冲突 | 异步下的替代 |
|---|---|---|---|
| `train()` 在 `_separate_rollout and _rollout_session_active` 时抛错 | `backend.py:503` | 异步的定义就是 rollout session 活跃时训练 | `coordinator.begin_train`：SYNC 模式保留该互斥，ASYNC 模式允许，因为两侧在不同设备、权重复制只在 sync 独占区 |
| `_train_policy_version` 在 stepped 后自增 | `backend.py:531-535` | 与 coordinator 的 Vt 重复计数 | `coordinator.end_train(stepped)` |
| `_rollout_policy_version` 在 `_sync_policy_if_needed` 设置；同步在下一次 rollout 前 lazy 触发；`.result()` 无超时 | `backend.py:282-323` | 无安全点概念，无法"先关准入再等在途操作"，无 cadence / lag 控制 | `coordinator.begin_sync / end_sync`，C 与 lag 显式触发，`_wait_handles` 带截止时间与 stop event |

`initialize` 部分（`backend.py:135-261`）没有冲突，`native.py` 只是重复了它，U1 第 4 步解决。

**三条路线，建议 PR 3 / 4 走 A，B / C 写进设计 issue 征求意见。**

- **A（现状加说明）**：不动 `CudaBackend`。PR 描述里放上表，说明 bridge + coordinator 是 backend 三处状态在并发下的等价物，SYNC 模式与 backend 行为一致（干净上游 GRPO / GSPO 回归已证明 backend 本身未变）。代价：SDK `Trainer` 无法直接驱动异步，见 U3 的依赖。
- **B（backend 加实验开关）**：`CudaConfig` 新增 `concurrent_rollout: bool = False`（属于 AGENTS "Ask first"）。开启时 `train()` 跳过 503 行守卫；`_sync_policy_if_needed` 改为 no-op，新增公开 `sync_policy(version, timeout_s)` 供 bridge 调用；两个版本号由调用方提供。这让 `Trainer` SDK 能接异步，但改了公开 backend 行为。
- **C（长期）**：把 coordinator 下沉到 backend，同步模式也走 `mode=SYNC` 的 coordinator，503 行守卫由它实现。这是 U3 注册 `async-grpo` 的干净前提，工作量最大。

- 完成条件：PR 3 描述含上表；设计 issue 得到维护者对 B / C 的态度并记入 `evidence/decisions.md`。

### U3 实验入口形式

2026-09-16 核实：暂保留 package/pipeline 入口；`Trainer.init()` 确实启动 backend，init 前 tokenizer 为 None。算法 loop 由 CLI factory 创建并调用 `fit()`，不能混同公开 SDK Trainer 的职责。

**事实。** `AlgorithmSpec(name, trainer_cls | factory, default_loss_fn, requires_rollout, loss_fn_factory)`（`algorithms.py:38-49`）。`build_trainer` 只做一件事：`trainer_cls(config, instance=, dataset=, reward_fn=, loss_fn=)`（`trainer_factory.py:16`）。`load_experimental_algorithms` 自动导入 `experimental/` 下每个包（`algorithms.py:181`），所以在 `async_policy/__init__.py` 里调 `register_algorithm` 即可注册，且该包导入已不加载 torch，满足"轻量"要求。

**要注册 `async-grpo` 需要补的东西。**

1. `AsyncGRPOTrainer(config, *, instance, dataset, reward_fn, loss_fn)` 类，接口对齐 `PolicyOnlyTrainer`（未核实：`Trainer` 对 trainer_cls 要求哪些方法，读 `areno/api/trainer.py` 与 `trainers/policy_only.py` 后再定）。
2. reward 已兼容：当前 `areno/api/rewards.py` 与 `PolicyOnlyTrainer` 都使用 `reward_fn(RewardRecord) -> float`，无需为旧接口描述新增适配层。
3. 数据适配：dataset 记录 → `AsyncPrompt`，tokenize 用 `instance` 的 tokenizer；`examples/async_policy/train.py:load_prompts` 已有雏形。
4. Q / K / C / lag 配置通道：`TrainerConfig` 有没有扩展字段未核实。没有的话需要新增 `async_policy: AsyncPolicyConfig | None = None`，属于 "Ask first"。
5. **依赖 U7**：若 `Trainer.init` 总是初始化 backend（未核实），A 路线下 `async-grpo` 会双重启动 worker。所以 U3 的注册要么等 U7 选 B / C，要么 trainer 自己不经过 `instance` 的 backend，只借 tokenizer。

- 完成条件：与导师确认三种形式选哪种；若选注册，上述 5 点各有测试；`areno train --algo async-grpo` 能跑 `examples/async_policy` 的数据。
- 评审点：官方原文是"例如"，包形式已满足 R1；注册是加分项不是必需项，不要为它改公开 config 而不问。

### U4 面向用户的英文文档

2026-09-16：两页与 toctree 已完成；基线/候选 HTML 构建均无警告。

`docs/` 是 Sphinx（`index.rst` 五个 toctree）。新增两页，英文：

- `docs/concepts/async-policy.rst`，挂 Conceptual Guides，紧接 `backend-topology`。内容：动机（对应官方背景四场景）、两级流水线图、Q / K / C / lag 四个字段的单位与默认值、staleness 语义（Q3 的结论）、支持拓扑（每角色单 GPU、TP=1 / DP=1）、退出语义表（EOF / max_steps / stop / abort / failed）、reward 取消边界、已知限制（K>1 不增加 GPU 生成并发、同步等待在途 rollout、质量未证明）。
- `docs/cookbook/async-grpo-two-gpu.rst`，挂 Cookbook。内容：环境、`examples/async_policy/train.py` 全部参数、预期输出文件（`result.json` / `updates.jsonl` / `samples.jsonl`）、同步对照命令、续训命令、如何读 `benchmark-summary.json`。
- 完成条件：`make -C docs html` 无新增警告；两页从 `index.rst` 可达；不出现 `runs/`、Modal、个人路径。

### U5 验证工具归位

2026-09-16：六个工具已迁到 `examples/async_policy/tools/`，smoke 资产已迁到示例目录；当前仓库没有顶层 tools。工具帮助与预览入口已验证。

`tests/async_policy_validation/` 2,484 行。按用途拆：

| 文件 | 去向 |
|---|---|
| `known_cases.py`、`protocol_cases.py`、`torch_cases.py`、`tiny_backend.py`、`fakes.py`、`reference.py`、`run.py`、`sdk_regression.py`、`sdk_train_fixture.json` | 留在 `tests/async_policy_validation/`，是 pytest 收集或显式运行的测试 |
| `modal_run.py`、`gpu_run.py`、`gpu_worker.py`、`benchmark_run.py`、`example_run.py`、`resume_run.py` | 移到 `tools/async_policy/`（仓库是否已有 `tools/` 未核实；没有则用 `examples/async_policy/tools/`） |
| `smoke_prompts.jsonl`、`smoke_reward.py` | 并入 `examples/async_policy/` |

- 完成条件：`pytest tests/ -k cpu` 不丢失任何原有测试 node ID（新增测试允许总数增加）；`modal_run.py` 的 `--phase` 全部仍可运行；REWRITE_TODO 的复现命令同步更新。

### U6 示例脚本修正

2026-09-16：参数与元数据修正及 CPU 测试已完成；GPU example 阶段待重跑。

`examples/async_policy/train.py:118-121`：

- `base_model_name_or_path="Qwen/Qwen3-0.6B"` → 用 `args.model` 原值（解析前的仓库引用）。
- `runtime={"attn_backend": "flash"}` → 新增 `--attn-backend {flash,native}`，默认 `native`，flash 需要 flash-attn 已装。
- `LoraConfig(rank=8, alpha=16)`、`lr=1e-5`、`n_samples=4`、`max_new_tokens=64` → 提为 CLI 参数，默认值不变，这样 P3 的 256 / 512 token 实验不用改代码。
- 完成条件：`tests/test_async_policy_example_cpu.py` 覆盖参数解析；Modal `example` 阶段重跑通过。

### U2 PR 拆分

本地分支、提交、实际依赖和各分支验证记录见 [PR_SERIES.md](PR_SERIES.md)。尚未对外创建 PR。

建议顺序与依赖：

1. **`fix(cuda)`: `sampling_params.seed` 透传**。`generation.py` 1 行 + 已有 CPU 测试。无依赖，可最先合，也可中选前作为社区贡献。
2. **`feat(engine)`: cluster 公开接口**（U1 第 1–3 步）+ `SAVE_TRAINING_STATE` / `LOAD_TRAINING_STATE` + `training_state.py`。无依赖。
3. **`feat(experimental)`: `async_policy` 核心**：`contracts` / `coordination` / `lifecycle` / `bridge` / `data` / `pipeline` + `test_async_policy_contracts_cpu.py` + `test_async_policy_core_cpu.py` + U4 的 concepts 页。描述含 U7 的对照表。无依赖。
4. **`feat(experimental)`: native 适配 + 示例**：`native.py`（U1 第 4–5 步之后）+ `examples/async_policy/` + U5 归位后的工具 + U4 的 cookbook 页 + `test_async_policy_native_cpu.py` / `example` / `benchmark`。依赖 2 和 3。

- 完成条件：四个分支各自 `git diff upstream/main --stat` 不含 `docs/projects/`、`runs/`、`REWRITE_START.md`；每个 PR 描述写明依赖与验证命令。

## 3. P1：吞吐（P）

- [ ] **P1 生成侧并发。** `begin_rollout` 用单个 bool 互斥，K>1 只让 CPU 评分与生成重叠，GPU 上仍是一个 prompt group 一次 session。L4 生成卡利用率仅 20%。
  - 方案 A：一次 `generate` 打包多个 prompt group（改 `RolloutPayload` 行数），K 语义不变。
  - 方案 B：允许多 session 并发（rollout 引擎自身的 `max_running_prompts` 已支持）。
  - 完成条件：同资源下生成 token/s 提升有数据；stale 率不恶化。
- [x] **P2 同步不等待在途 rollout（本阶段仅设计）。** lag=1 三轮同步准入等待合计约 100 s / 50 updates。
  - 方案：rollout 侧双缓冲权重，生成边界切换；或把同步放到生成结束事件上而不是训练结束。首版明确不做，先写设计草案与预期收益。
- [ ] **P3 对齐官方动机场景的 benchmark。** 现有 benchmark 是 64 token 短输出。补 `max_new_tokens ∈ {256, 512}`、`n_samples ∈ {4, 8}`，这是"长输出 RLVR"与"大 n_samples"的原始动机；不同长度的收益方向不能预设；流水线平衡、同步和内存开销需实测。
- [ ] **P4 恢复原 T14 的扫描矩阵。** C ∈ {1,2,4}、Q ∈ {1,2,4}、K ∈ {1,2}，每配置 ≥3 次；扫描固定 lag=4，避免 C=4 被 lag=1 提前触发。当前仅实验入口已准备。

## 4. P1：范围扩展（E）

- [ ] **E1 GSPO 异步。** `native.train` 的 `loss_fn` 参数化后跑 `gspo_loss_fn`，复用同一 benchmark。
- [ ] **E2 全参数续训。** `training_state` 要求 TP=1/DP=1；`full-01` 只做了训练 + 同步，未做 resume。补一次全参数 save / resume 对照。
- [x] **E3 Agentic 轨迹接入设计。** `features` 目前直接拒绝；官方背景把 agentic 列为动机之一。先写清接入点（`prompt_features_by_dp`、loss mask），不急于实现。
- [x] **E4 TP>1 / DP>1 接入设计。** 首版明确不支持；记录需要改动的位置（`ClusterPartition`、`training_state._identity`）。

## 5. P0：OSPP 流程（O）

按原 TODO v2 第 0 节，这些一项都没动，而它们决定代码是否计入结项。

- [ ] **O1 导师对齐（原 T00.2）。** 带上现有证据：CPU 契约、双 L4 结果、1.396× 与种子 43 退化、U1/U3 两个接口问题。三个问题：PR #480 定位、bridge 直接驱动 worker 是否可接受、`async-grpo` 注册是否期望。
- [ ] **O2 申请书（原 T00.3）。** 以本表第 1–3 节为"计划"部分；已完成部分如实写为"中选前的准备工作"。
- [ ] **O3 开 PR 时机。** 规则：中选前提交的 PR 不计入结项。选项：中选前只开 U2-1（seed 修复，作为社区贡献）与设计 issue；主体 PR 中选后开。与导师确认。
- [x] **O4 结项报告草稿。** 四部分：已完成、问题与解决、测试用例、后续安排；从 REWRITE_TODO 与本表直接抽取。

## 6. 收尾（H）

- [x] **H1 旧 worktree 归档。** `feat-async-trainer` 的未提交 `docs/projects/` 与 `tests/async_policy_validation/` 是原型审计的历史版本；把 `REWRITE_ASSESSMENT.md` 与 `evidence/` 提交到资料分支 `docs/ospp-async-policy`，其余不再维护；分支加 `archive/` 前缀或在 README 标注。
- [x] **H2 原始结果备份。** `runs/async-policy-rewrite/` 含 1.1 GiB trace 与全部 JSON，不在 Git；拷到实验盘并记录路径。
- [ ] **H3 Modal Volume。** `areno-async-policy-r4` 保留模型缓存与结果；确认是否持续计费，不用时删除。

## 7. 建议顺序

1. O1 先发，等待期间做 Q1（纯实验，不改代码）。
2. U1 + U6，两者都是导师第一眼会看的。
3. Q2 消融，结果再决定是否进入 PR 范围。
4. U2 拆分、U4 文档，随中选结果决定开 PR 节奏。
5. P1 / P3 作为中选后的主体工作。

## 8. 2026-09-16 执行记录

- Q2/E1：可选 offpolicy loss、GSPO 注入与原 S1 数值 oracle 已验证；Q1/Q2/P3/P4/E1 的 GPU 矩阵未运行。
- P2/E3/E4：仅完成 [扩展设计](EXTENSION_DESIGN.md)，没有实现生成双缓冲、Agentic 或多 rank 支持。
- O1/O2/O3/O4：见 [OSPP_HANDOFF](drafts/OSPP_HANDOFF.md)；草稿不代表发送、申请提交或正式结项。
- H1：资料分支 `docs/ospp-async-policy` 的 `612bffc` 归档 8 个原文件并核对 hash；原 dirty worktree 保留。
- H2：760 个原始结果文件、7,616,137,087 bytes 已复制到实验盘镜像外并逐项验证 SHA-256。路径与清单见 `runs/async-policy-rewrite/next-stage/backup-summary.json`；本轮仍在生成的 next-stage 与可重建工具环境单列。
- H3：活动 App 列表没有本项目任务，缓存卷保留用于待跑实验。2026-09-16 查询 Modal 官方定价：Volumes $0.09/GiB/月，含 1 TiB/月免费额；账号总存储用量与最终账单未核定，不声称永久免费。
