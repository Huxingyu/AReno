# NEXT_TODO v3 审查与执行说明

日期：2026-09-16。审查基线 `a35a580`；上游 `48d07c5`。本次重新 fetch
`upstream/main` 后仍为该提交。原始计划见 [NEXT_TODO.md](NEXT_TODO.md)。

## 结论

**方向合理，可以推进；不能原样把所有事项当作立即可验收的开发任务。**
优先解决质量证据、engine 接口边界、示例与用户文档是正确的。实验算法注册、
生成并发、双缓冲权重和多卡拓扑需要独立设计；导师意见、OSPP 状态和 GPU
实测不能由代码存在代替。本次先落实可以本地实现和验证的部分，保留外部验收项。

原文把“首版工程交付”和“项目完成”分开，且没有掩盖种子 43 的退化，这一点应保留。
但“D1 无注册入口”是可选集成能力，不是官方 D1 的确定缺口。资料分支保存的
官方原文允许 experimental package / trainer class 先行。

## 一、代码核实后必须修正的判断

| 原判断 / 建议 | 核实结果与修正 |
|---|---|
| lag=0 等价同步 | 不成立。它约束训练准入时的版本差；后台生成、丢弃、数据迭代和 RNG 消耗仍可能不同。只能说接受的 batch 来自当前训练版本。 |
| “每批次都过期”解释 lag=0 | 过强。已有三轮 54/54/55 个 stale drop 是观测值，不是每批必然过期的契约。应说明更新期间完成的旧 batch 会被丢弃。 |
| ratio 恒 1，因此 loss 无效 | 原文没有直接声称无效，但容易这样理解。`logp.detach()` 只停止分母的梯度，策略梯度仍存在；缺少的是行为策略分母和有效 clipping。这是既定目标差异。 |
| Q1 只需判断截断或异步二选一 | 不足。训练长度、评测长度、实际参与更新的题目和采样随机性均有混杂；扩大样本后不再出现退化，也不等于证明问题不存在。 |
| U3 reward 要从 `(example, completions)` 改配 `RewardRecord` | 已过时。当前 `areno/api/rewards.py` 和 `PolicyOnlyTrainer._score_reward_records` 已统一使用 `RewardRecord`。无需新增这一适配层。 |
| `Trainer` 要求 trainer_cls 的接口待核实 | 所有者需区分：CLI 的 `build_trainer` 创建算法 loop，并调用 `fit()`；公开 SDK `Trainer` 是 loop 使用的执行实例。`Trainer.init()` 确实初始化 backend。 |
| 只借 `instance` tokenizer 可避免双启动 | `get_tokenizer()` 在 init 前返回 None；init 又会启动 backend。需要明确新的生命周期或独立加载 tokenizer，不能直接套现有 loop。 |
| U1 约几十行且默认行为完全不变 | 低估了并发与回收边界。必须处理关闭与提交的竞态、同步/异步等待者、pipe feeder、多个分区和失败后的退出证据。保留旧默认 grace 不等于所有失败行为不变。 |
| 用 `from_pretrained` 完全等价替换手拼配置 | 还需保留 native 显式 seed：该工厂使用 `torch.initial_seed()`。本次在未启动的 engine config 中设置原 pair seed，再选择验证用 worker。 |
| C=1/2/4、lag=1 可以直接比较 cadence | C=4 会被 lag 提前触发；实际触发更新数为 `min(C, lag+1)`。容量扫描固定 lag=4，并保留独立 lag 消融。 |
| 更长输出预期有更大异步收益 | 不是必然。理想两级流水线的上限约为 `(Trollout+Ttrain)/max(Trollout,Ttrain)`；单侧越来越占主导时，收益反而可能接近 1。必须报告实测。 |
| U5 完成后测试收集数不变 | 单纯迁移应不丢失原测试；本阶段新增测试后，总数应增长。验收应比较旧 node ID 是否完整保留。 |
| PR1 可直接 cherry-pick seed 提交 | `14c92f5` 还包含约千行 SDK fixture 与回归脚本；PR1 应只取 generation 与对应 CPU 测试两个文件。 |

上游实际数学实现在 `areno/api/backend/cuda/losses.py`；项目指南中若干
`api/loss_fns/` 导航已过时。本次依据真实定义实现，没有新增主 CLI 分支或公开 config 字段。

## 二、逐项合理性与本轮处理

| 项目 | 判断 | 本轮处理 / 尚缺条件 |
|---|---|---|
| Q1 | P0 合理，实验设计需修正 | 固定 128 题，保留原 32 题为子集；训练 64/256 × 评测 64/256 分离；五 seed、四组设置，40 次训练。实验入口已准备，GPU 结果待取得。 |
| Q2 | 适合作为可选消融 | 实现 experimental `grpo_offpolicy_loss_fn` 与 loss 注入；默认 GRPO 不变。精确 loss/梯度、mask、分母 detach 与原 S1 oracle 均有 CPU 验证。质量结论仍待 Q1。 |
| Q3 | 应立即完成 | config docstring、DESIGN 和英文概念页已说明字段单位、lag 与 loss 的关系及 lag=0 限制。 |
| U1 | P0 合理 | 原生适配改用公开等待、关闭与工厂接口；保留 seed、退出证据和重试。CPU 验证已做；GPU faults / extended-faults 仍须重跑。 |
| U7 | 先采用 A | 不改稳定 backend，不增加第二个版本所有者。B/C 的公开配置和 SDK 生命周期设计留作维护者决策。 |
| U3 | 可选，非本轮阻塞项 | 不注册一个不能遵守 SDK 生命周期的名义入口。核实后的依赖写入设计稿，等待维护者决定入口形式。 |
| U4 | 必需且成本适中 | 两页英文 Sphinx 文档及 toctree 已落地；基线和候选构建均无警告。 |
| U5 | 合理 | 仓库没有顶层 tools，使用 `examples/async_policy/tools/`；测试 oracle 与 SDK fixture 留在 tests。更新复现命令并检查入口。 |
| U6 | 应立即完成 | 保留原模型引用；attention 默认 native；参数化 LoRA、学习率、样本数、长度、Q/K/C 和 loss；默认训练算法保持 GRPO。 |
| U2 | 合理，但拆分须实际验证 | 准备四个本地 review 分支；只选择必要文件，排除 docs/projects、运行结果和 REWRITE_START。具体提交、依赖、命令见 PR_SERIES。 |
| P1 | 有价值，不能无测量优化 | 先保留可执行吞吐矩阵和批量生成设计。K 的任务单位、分组 advantage 和同步准入不能被 batching 悄悄改变。 |
| P2 | 按原文仅做设计 | 记录生成结束时触发同步与双缓冲的成本、权重一致性、graph/cache 约束；不宣称实现。 |
| P3 | 实测目标合理 | 三 seed，256/512 tokens × 4/8 samples × sync/async，共 24 次训练；入口已准备。 |
| P4 | 修正 lag 后合理 | 18 个 Q/K/C 组合各三次，另加三次同步对照，共 57 次；入口已准备。 |
| E1 | 保留扩展空间是必须，完整 GSPO 实验是扩展 | loss 注入与分发 CPU 验证完成；另有 12 次训练的 GSPO 矩阵，未冒充已实跑。 |
| E2 | 补验收合理 | continuation 工具新增 full-parameter 模式；真实 CPU AdamW 续训测试已有，真实双 GPU 全参续训仍待跑。 |
| E3/E4 | 首版明确延后合理 | 接入点与验收边界写入扩展设计；没有为凑完成数解除 text-only 或 TP/DP 限制。 |
| O1/O2/O3/O4 | 项目流程必须单列 | 准备导师邮件、设计 issue、申请技术稿和结项草稿；尚未发送、提交或取得维护者答复。 |
| H1 | 归档有必要 | 把原型审查及 evidence 的副本归档到资料分支，保留原 dirty worktree。 |
| H2 | 应保留真实证据 | 原始验收结果备份到实验盘镜像外，并逐文件核对 SHA-256；工具环境不属于实验结果。详情见执行证据。 |
| H3 | 应核查计费并按需保留 | 已查询本项目卷及活动 App，并查询 Modal 公开存储费率；后续实验仍需缓存，保留卷。不能把活动计算为零等同于存储永远免费。 |

## 三、修正后的实验判读

1. **固定 checkpoint 改评测预算**：对同一 checkpoint 分别评测 64/256 tokens。
   这回答“输出预算是否掩盖答案”，不把训练长度改变混入这一比较。
2. **固定评测预算改训练预算**：同 seed、模型、数据、学习率、attention 和更新数，
   比较训练 64/256 tokens 的 checkpoint；保留每步实际 prompt IDs、版本和 reward。
3. **同预算比较调度与目标**：sync、lag=1、lag=0、lag=1+offpolicy。
   按 seed 配对，报告每个 seed 的准确率差和吞吐比，不只比较平均值。
4. **显式报告结果边界**：token-limit hit 是截断代理指标，不是引擎 stop reason。
   128 道合成算术题、五 seed 和 55 次更新仍不能证明长期收敛；若差异消失，应写
   “该矩阵未复现”，不能写“已证明不退化”。
5. **防止证据错配**：矩阵汇总检查 seed、长度、loss、Q/K/C、更新窗口、source SHA
   和数据 hash；未完成 job 不生成“完成”结论。质量、吞吐与生命周期验收分开报告。

## 四、实现中的接口取舍

`ClusterCallHandle.done()/wait()` 不消费请求；旧 `result(timeout)` 的超时语义保留。
`TPCluster.request_shutdown()` 让两个分区先收到退出通知，再开始 join，避免一个
分区等退出时另一个尚未开始分布式收尾。`close(timeout_s=...)` 在所有 rank 和
结果泵之间共享一个预算，返回 `WorkerExit`；不能证明退出就抛错，证据仍可读取。
队列关闭使用 `cancel_join_thread`，避免已无 reader 的 feeder 把有界关闭变成无限等待。

启动等待使用同一个单调时钟截止时间；失败回收另有全分区共享的 5 秒预算。
这是一项明确的接口语义：启动超时的异常返回可能额外包含清理时间，不声称整个函数
只花 timeout_s。默认无参数 close 保留原每 worker 5 秒 grace / terminate 路径，
但关闭时唤醒 pending call、拒绝后续提交属于有意修正的失败行为，需要维护者评审。

## 五、已验证与未验证

本机 Python 3.12、Torch 2.11.0+cu128，可见一张 GPU；所有 CPU 回归显式隐藏 CUDA。
基线在独立 detached worktree 中运行，未切换用户工作分支。

- 完整 CPU 基线：889 passed、12 failed、10 skipped。
- 第一轮完整候选：920 passed、12 failed、10 skipped；随后补充一个 S1 oracle 用例。
- 最终 PR 4 集成候选：921 passed、12 failed、10 skipped；失败 node ID 集合与基线完全一致。原 911 个 CPU node ID 全部保留，新增 32 个；变更涉及的测试模块 149 项全部通过。
- 四个本地 PR 分支均已提交；PR 1/2/3 独立检查分别为 5/33/90 passed。PR 4 与工程实现 `f217bac` 的代码内容一致，未推送、未创建对外 PR。
- 文档：基线与候选 Sphinx HTML 构建无警告；两页可从 index 到达。
- 示例及迁移工具：14 个帮助/预览命令成功，预览没有启动远端 GPU。
- 17 种 Modal phase 资源预览均成功；不代表 GPU 阶段已经运行。
- 本轮 20 个 Python 文件 Ruff 通过，四个分支 `git diff --check` 通过；最终复查、各 PR 分支证据见 [PR_SERIES.md](PR_SERIES.md) 和[执行证据](evidence/next-stage-validation.md)。

候选代码尚无本轮双 GPU 验证。原 R4 的 GPU 成绩仅证明旧提交，不能直接继承为
本次 protocol、native、attention 默认值或可选 loss 的验收结果。
本轮未确认 OSPP 中选/签约状态，也未获得导师或维护者意见。
资料分支 2026-09-14 的记录说明 2026 采用滚动申请；不能仅因九月就断言申请期结束。

具体执行记录保存在 `runs/async-policy-rewrite/next-stage/`；该目录不进入功能 PR。
下一步的六项验收、种子 43 pilot、完整矩阵与停止条件已写成
[GPU 执行计划](GPU_VALIDATION_PLAN.md)，待资源与预算批准后按阶段执行。
