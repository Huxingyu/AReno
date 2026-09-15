# 异步核心重写 TODO

本表提炼自反复审查过的原 TODO / 实施方案，并加入实测得到的验收要求。旧分支的任务完成状态不继承。当前先交付 CPU 闭环，完整目标仍包括真实双 GPU、故障回收、数值 / 同步回归和同资源性能实验。

**范围与默认起点**

实验性 GRPO completion；一个生成引擎、一个训练写入者；两个独立单 GPU 组，TP=1 / DP=1。默认 Q=2、K=1、同步间隔 C=1、lag=1。K 可以限制多份生产任务，首版同一生成引擎的 session 操作仍串行。每个非空 ready batch 至多一次真实 optimizer update，max_steps 按成功 updates 计数。

复用上游 group advantage、materialization、GRPO loss、GPU worker 和原生权重传输。先用 LoRA 闭环，再补小模型全参数。GSPO 扩展、Agentic、多模态、多机、复杂 TP / DP、CLI 新入口和无暂停权重同步不占首版关键路径。

**R0：干净起点**

- [x] 从上游 `48d07c5` 建立独立分支 / worktree，未带入旧异步生产代码。
- [x] 留下少量开工文档和必要测试，原始档案保留在旧工作区。
- [x] 确认导入当前 checkout，42 项选定 CPU 回归通过；同步数值起点对照通过。

**R1：先明确所有权与失败契约**

- [x] 整体生命周期和首个失败原因由 supervisor 管理；版本 / 准入 / 同步由唯一协调器管理；producer 拥有数据迭代器与生产任务；训练主线程拥有训练指标 / checkpoint。
- [x] 写出初始化、运行、EOF drain、max_steps 停止、abort、失败和关闭的状态转移及清理顺序。
- [x] 明确 CPU payload 快照或可验证的所有权转移；队列不能携带计算图、非 CPU tensor 或可被后续生产操作改写的数据。
- [x] 定义实际 stepped、初始权重对齐、重复同步请求合并和同步失败后停止使用模型的契约。
- [x] 把 [已知问题](KNOWN_ISSUES.md) 映射到验收测试；允许适配旧接口，保留行为要求。

验收：[所有权 / 状态转换说明](DESIGN.md)，每个失败路径有明确结果。首版只接受能及时返回或合作取消的 reward；Python 线程不能安全强杀。checkpoint 的所有者已约定，真实 GPU checkpoint 实现及验收仍在 R4。

**R2：重新实现 CPU 调度核心**

- [x] 完整 prompt group 经评分、advantage 和 materialization 后才入队；stale 整批丢弃。
- [x] Q 限制 ready batch；K 从接收工作直到入队或终止，包括生成、评分、构造和等待 put。先取得许可再读数据，任何异常都归还；不预先创建无限 futures。
- [x] 入训练准入后检查 `Vt - Vb`；future version 直接失败。真实 update 推进 Vt，drop / 空批 / 未 stepped 不推进。
- [x] 同步先关闭新准入，等待活跃模型操作完成，再独占传输；更新版本及触发同步请求形成原子边界。CPU 评分 / 等待队列不持 GPU lease。
- [x] 所有 queue / permit / lease 等待可被关闭唤醒；初始化和数据源异常能传播；成功返回前完成收尾，超时不能声称成功。
- [x] 用真实线程、队列、事件 / barrier 验证重叠、互斥和退出。

验收：迁移第一层 21 项用例；重跑 Q∈{1,2,4}、K∈{1,2,4}、C∈{1,4}、lag∈{0,1,4} 的 4 种子矩阵，共 216 组；30 次空输入、50 次非空生命周期检查。检查不变量，不固定操作系统调度下的丢弃数量。

**R3：统一 bridge 与真实 CPU 小模型**

- [x] pipeline 和 bridge 形成同一条调用路径，避免重复的生命周期或准入状态。
- [x] 实际对齐两端初始权重；源权重在同步期间不变；完整传输 / 接收 / 推理状态准备成功后才发布 Vr。部分复制失败整体停止。
- [x] 两份 CPU 模型真实采样、autograd、SGD 和参数复制，检查模型实际参数与版本相符。
- [x] 迁移第二层 9 项契约用例；固定输入对照同步的 masks、advantages、logprobs、loss、梯度和更新后参数。
- [x] 准确记录 batch 去向、Vb / Vt / Vr、完整等待时间、实际 updates 和退出原因。

验收：合计 30 项契约对应的行为通过；容差保持测试已有定义。S1 是原方案已知的另一目标函数诊断，单独运行，不作为修改 loss 或调度验收失败的理由。

**R4：真实 GPU 与完整交付**

- [x] 同资源双 GPU 同步基线，然后 Qwen3-0.6B LoRA 异步 smoke：Q=2、K=1、C=1、lag=1，prompt 上限 128、生成上限 64、n_samples=4，至少 3–5 次成功更新。
- [x] 用可关联的 GPU 活动证明 rollout / train 实际重叠；完成原生同步、保存 / 重载和清理。
- [ ] 注入 worker / NCCL / 同步失败与中断，验证停止预算、资源回收、重启和最后成功 checkpoint 的保留。
- [ ] 补小模型全参数同步、原同步 GRPO / GSPO 回归，检查默认 API 行为不变。
- [ ] 同资源比较吞吐、等待、lag / drop 和质量；关键配置至少 3 次运行，预热后至少 50 次更新或固定测量窗口，包含无收益 / 退化结果。
- [ ] 正式示例和必需数据随功能交付，干净 checkout 可复现，代码 / 配置 / 日志中的 SHA 一致。

CPU 两份模型不替代双 GPU 证据；短 smoke 不代表收敛或质量保证。每阶段保存当前 SHA 下的新结果，不能用历史通过状态代替验收。

第三项已覆盖 worker 强制退出、同步接收失败、进程回收、最后成功 LoRA checkpoint 保留和同容器重新启动；独立通信超时、外部中断及 optimizer 状态续训尚未验证，保持未勾选。首轮同步基线使用同一原生适配器和 `DeviceMode.SYNC`；独立上游 GRPO / GSPO GPU 回归仍属于第四项。

**本机验收记录：2026-09-15**

实现 / 测试 SHA：`ecb8f3bfe0db9726708a2bad9aeef42b7b4a5e2a`。在该提交的干净工作区运行，随后仅补充文档。同步基线来自独立、干净的 `48d07c54051c41bf36218f99bce1c3697e9ba63c` worktree，脚本检查实际导入路径并记录两端 SHA。

| 验收 | 结果 |
|---|---|
| 专项契约 | 75 通过：原 30 项迁移 + 45 项补充；0 失败 / 跳过 / 超时 |
| 上游选定 CPU 回归 | 66 通过，覆盖算法、reward、materialization、packing、logprobs、同步、导入和注册 |
| 并发矩阵 | 216 组，3,456 个 prompt groups；Q / K / lag / 实际 update 不变量通过 |
| 生命周期 | 30 次空输入、50 次非空；后者共 400 groups，存活 pipeline 对象和遗留线程均为 0 |
| 真实 CPU 模型 | 4 配置、96 groups、48 次真实更新、46 次复制（包含初始对齐）；线程事件证实生成 / 训练重叠且同步互斥 |
| 独立逐行 CPU oracle | 梯度 / 参数最大绝对差均为 `2.9802322387695312e-8`，原容差未放宽 |
| 固定输入 vs 干净同步基线 | 4 prompts / 11 sequences；mask、advantage、logprobs、loss、梯度、参数等 9 类输出最大绝对差均为 0；atol=`1e-7`，rtol=`1e-6` |
| 静态检查 | 新模块及验证工具 Ruff 通过，`git diff --check` 通过 |
| S1 独立诊断 | 预期不等价，最大梯度差约 `0.121379`；单独非零退出，未计入 75 项契约，也未修改既定 loss |

更新 / 复制 / stale 数量受线程调度影响，上表是本次观察值；验收断言检查不变量，不固定这些数量。原始结果位于 `runs/async-policy-rewrite/acceptance/`：`overview.json` 汇总，`metadata.json` 记录环境，子目录保留 JUnit、事件和对照数组。原始结果不入 Git。

环境：Python 3.12.3、Torch 2.11.0+cu128、NumPy 2.5.2、Pydantic 2.13.4、pytest 9.1.1、safetensors 0.8.0。全程隐藏 CUDA、限制 CPU 数值库线程。独立 venv 复用已有 Torch，仅补装项目原已声明的 safetensors，没有修改项目依赖。

**复现**

从仓库根目录执行。本 worktree 的验证 venv 已准备；新机器先在已有 Torch 的 Python 环境中准备：

```bash
async_base_python=$(python3 -c 'import sys; print(sys.executable)')
uv venv --python "$async_base_python" --system-site-packages runs/async-policy-rewrite/venv
uv pip install --python runs/async-policy-rewrite/venv/bin/python 'safetensors>=0.4'
```

75 项契约的隔离执行（默认每组超时 45 秒、同时两组）：

```bash
async_python=runs/async-policy-rewrite/venv/bin/python
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$async_python" tests/async_policy_validation/run.py --include-core --output-dir runs/async-policy-rewrite/recheck
```

66 项上游回归：

```bash
"$async_python" -m pytest -q \
  tests/test_algorithms_cpu.py tests/test_losses_rewards_cpu.py \
  tests/test_materialization_equivalence_cpu.py tests/test_train_pack_equivalence_cpu.py \
  tests/test_logprobs_cpu.py tests/test_policy_tensor_sync_cpu.py tests/test_protocol_cpu.py \
  tests/test_import_boundaries_cpu.py tests/test_registry_cpu.py tests/test_registry_discovery_cpu.py \
  tests/test_dual_engine_backend_cpu.py
```

固定输入数值对照（脚本只导入指定 checkout 的 AReno）：

```bash
git worktree add --detach runs/async-policy-rewrite/upstream 48d07c5
"$async_python" tests/async_policy_validation/reference.py --repo-root runs/async-policy-rewrite/upstream --mode sync --output runs/async-policy-rewrite/recheck/reference/upstream.json
"$async_python" tests/async_policy_validation/reference.py --repo-root . --mode async --output runs/async-policy-rewrite/recheck/reference/async.json --compare runs/async-policy-rewrite/recheck/reference/upstream.json
git worktree remove runs/async-policy-rewrite/upstream
```

正常 `pytest tests/ -k cpu` 也会收集新增 75 项；S1 不自动收集，其显式运行命令见 [测试说明](KNOWN_ISSUES.md)。上述 CPU 验收完成时 R4 尚未开始，后续 GPU 结果见下节。

**Modal 首轮验收记录：2026-09-15**

这是以上 CPU 记录之后执行的 R4 首轮验证。原生 GPU 实现与同步 / 异步 / trace 的源码 SHA 为 `11c7e3bf54e2d4f00960e5585746b974ec19df1e`；故障 / 重启和补充样本记录使用 `54bf560f2f0b2eb23c8a5486a1cbdd54451a6e70`，后者只修改验证工具。所有云端代码均先在本地提交并推送，再由容器 fetch / checkout；每次检查请求 SHA、实际 checkout 与日志 SHA 相同。

配置：同一容器内 NVIDIA L4 × 2，每卡实际可见约 22.03 GiB；训练 GPU 0、生成 GPU 1，TP=1 / DP=1。模型通过 ModelScope 下载 Qwen3-0.6B；LoRA rank=8 / alpha=16，microbatch=1、每组 4 条 completion，Q=2 / K=1 / C=1 / lag=1，prompt≤128 / response≤64。Torch 2.8.0+cu128、FlashAttention 2.8.3、Transformers 5.17.0、ModelScope 1.40.0；关闭 torch.compile，使用 eager decode。数据为随测试交付的 8 条算术 prompt，使用答案命中和小幅 token 多样性奖励，仅验证运行机制。

| 检查 | 实测结果 |
|---|---|
| 双卡同步对照 | 5 次真实 optimizer update，梯度非零；6 次同步包含初始版本 0 |
| 异步流水线 | 5 次真实 update；5 次同步，0 stale drop，Q 峰值 1、K 峰值 1；max_steps 正常停止 |
| GPU 实际重叠 | 独立 trace 运行完成 5 次 update；83,181 个训练内核、377,916 个生成内核，经 CPU 时钟标记对齐，找到至少 846 μs 的跨卡内核重叠 |
| 同步互斥 | 六个运行的 worker 事件中，sync 与 rollout / train 的重叠冲突均为 0 |
| 权重与保存重载 | 每次成功同步逐项比较 392 个 LoRA 张量，值完全一致；四次正常运行的 checkpoint 经公共 Trainer SDK 重载、导出比较均完全一致，并再次生成 |
| 显存 | 正常短训练的 PyTorch 峰值 allocated：训练约 2.693 GiB、生成约 2.711 GiB；不把它当作包含全部驱动 / NCCL 分配的 nvidia-smi 总显存 |
| 接收端同步故障 | 版本 1 接收端主动报错；原异常传播，Vt=1 / Vr=0，checkpoint-1 保留，全部 worker 被回收；用例含初始化共约 17.02 秒 |
| worker 强退 | 第二次训练强制退出，exitcode=23 被识别；checkpoint-1 保留、无存活子进程；用例含初始化共约 18.45 秒 |
| 故障后重启 | 在同一容器重新创建流水线，再完成 5 次 update 和保存重载；没有声称恢复了 optimizer 状态 |
| 本地 CPU 回归 | 在干净 `54bf560` 上，原 75 项 + 8 项原生适配器边界测试，共 83 项通过 |

合计观察到 22 次成功 update（包含两个故障用例各 1 次）、24 次成功同步。注入的两次失败是预期行为；本轮没有暴露新的功能性失败。每次运行均无残留 producer / worker；5 个 Modal App 全部 stopped，最终查询到的活动容器为 0。

运行前后工作区账单快照的计量费用增加约 **$0.35**，当前由已有额度抵扣，`billed_cost` 仍为 0。账单快照和各任务的命令、SHA、指标、样本、checkpoint、原始 trace 位于忽略的 `runs/async-policy-rewrite/modal/`，入口是 `overview.json`。完整 trace 约 1.1 GiB，未加入 Git。单次命令时间包含初始化 / 重载，trace 还引入显著开销；本轮不报告吞吐提升或训练质量结论。

在已有 Modal 配置的机器上，从干净且已推送的本分支复现：

```bash
uv venv --python python3 runs/async-policy-rewrite/modal-venv
uv pip install --python runs/async-policy-rewrite/modal-venv/bin/python 'modal[api-proxy-support]==1.5.5'
modal_python=runs/async-policy-rewrite/modal-venv/bin/python
"$modal_python" tests/async_policy_validation/modal_run.py --phase prepare --output-dir runs/async-policy-rewrite/modal/prepare
"$modal_python" tests/async_policy_validation/modal_run.py --phase baseline --output-dir runs/async-policy-rewrite/modal/baseline
"$modal_python" tests/async_policy_validation/modal_run.py --phase async --output-dir runs/async-policy-rewrite/modal/async
"$modal_python" tests/async_policy_validation/modal_run.py --phase trace --output-dir runs/async-policy-rewrite/modal/trace
"$modal_python" tests/async_policy_validation/modal_run.py --phase faults --output-dir runs/async-policy-rewrite/modal/faults
```

prepare 只使用 CPU，完成 CUDA 扩展构建和 ModelScope 缓存；GPU 任务固定 `L4:2`，最多一个容器、无自动重试、20 分钟硬超时，内部测试子进程为 17.5 分钟超时并终止进程组。模型和结果保存在专用 Modal Volume；原始结果也下载回本地。CUDA 构建输入有变化时，启动器会拒绝复用旧扩展，必须先更新镜像构建。
