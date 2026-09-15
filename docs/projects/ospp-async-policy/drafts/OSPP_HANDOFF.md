# OSPP 沟通与交付草稿

2026-09-16。以下均为本地准备稿，未发送、未在报名系统提交、未取得导师答复。
当前中选、协议和开工状态待本人确认；不能把已有准备工作计作中选后的结项成果。

## O1：导师邮件草稿

主题：AReno 原生异步策略训练器：现有准备、质量观察与接口对齐

老师您好：

我已整理本课题官方中英文需求，并完成一版实验性 completion 流水线的准备工作。
实现使用 AReno 原生 worker 与权重同步，默认复用既有 GRPO loss。既有 CPU 契约
和双 L4 短程验证记录已保存；旧基线三 seed 实验的 lag=1 更新吞吐平均约为同步
的 1.396 倍，但种子 43 的评测从 21/32 降到 16/32，同步为 23/32。这一退化尚未
解释，不能据此声称质量稳定。当前正在准备独立控制训练/评测长度的五 seed 对照。

为避免与现有社区工作重复，希望先确认三个问题：

1. PR #480 与 Issue #487 当前在维护者规划中的定位，以及希望复用的组件范围。
2. 是否接受首版 experimental bridge 直接驱动原生 worker，并给 cluster 补公开的
   非取消等待、可取消启动与有界关闭接口；接口草案及 CPU 证据已备好。
3. 首版是否期望注册 async-grpo，还是先提供实验 trainer/pipeline 与独立示例。
   稳定 CudaBackend 的 session 守卫、版本计数和同步触发需要明确唯一所有者。

行为策略 ratio loss 只准备为显式消融，默认 GRPO 未变。请您评估是否纳入正式
研究范围。申请、开工时间与主体 PR 节奏也希望按项目规则核定；已有代码与实验将
如实列作前期准备，不申请提前承诺中选。

附件索引：NEXT_REVIEW、DESIGN、EXTENSION_DESIGN、PR_SERIES、REWRITE_TODO。
发送前将本地索引替换为拟公开、可访问的链接，并核对收件人。

## U7/U3：设计 issue 草稿

标题：Experimental async policy integration and native worker lifecycle APIs

The current CudaBackend prohibits training while a separate rollout session is
active, owns train/rollout version counters, and synchronizes lazily before the
next rollout. The experimental pipeline places these responsibilities in one
coordinator, with overlapping model operations on disjoint devices and exclusive
weight synchronization. We propose keeping this bridge separate for the first
delivery, with explicit public cluster lifecycle APIs.

| Responsibility | Stable backend | Experimental bridge |
|---|---|---|
| Train admission | Reject an active rollout session | SYNC keeps mutual exclusion; ASYNC permits separate-device overlap |
| Version commit | Increment after stepped=True | Coordinator.end_train(stepped) is the only owner |
| Weight publication | Lazy sync before rollout | Close admission, wait for active operations, copy and acknowledge, publish Vr |

The current reward interface already uses RewardRecord on both paths. Registering
async-grpo still needs a configuration channel and explicit SDK lifecycle: calling
Trainer.init solely to get a tokenizer starts the stable backend as well.

Feedback requested: retain the independent experimental entry; add a narrowly
scoped backend mode with one version owner; or move the coordinator into the
backend for both sync and async execution. No public config/CLI changes are
included in the current proposal. The quality regression remains an open result.

## O2：申请书技术草稿

**项目**：实现 AReno 原生异步策略训练器，项目编号 267bf0030。

**个人信息**：姓名、学校、年级、预计毕业日期、联系方式由申请人确认后填写。
本稿不代为声明当前学生资格或签约状态。

**目标**：在单机资源分离下交付 experimental GRPO 流水线、生命周期管理、原生
权重同步、有界 Q/K/C 与 staleness 控制、可复现 smoke 和同步行为回归。

**前期准备**：同步基线、completion 调度实现、CPU 契约与旧提交的双 L4 实验；
质量退化已记录。下一阶段的公开 engine 接口、示例参数化、英文文档与消融入口
也是本次准备工作。是否及如何计入正式成果以项目规则和真实开工记录为准。

**拟研究内容**：质量退化归因及目标消融、长输出/大 sample 场景的吞吐、K/Q/C
耦合与生成批处理、安全同步的开销、GSPO 扩展和全参数续训。多机与 TP/DP 扩展、
双缓冲和 Agentic 轨迹作为设计储备，不能默认承诺首版均完成。

**计划**（自正式开工日起，具体范围须重新与导师确定）：

| 周次 | 工作 | 验收 |
|---|---|---|
| 1–2 | 接口与范围评审、环境复核、同步/异步基线 | 可复现的 commit、输入、退出与数值记录 |
| 3–4 | 五 seed 质量矩阵与行为策略目标消融 | 完整原始回答、版本/样本记录、配对质量与截断分析 |
| 5–6 | 长输出/大样本 benchmark、生成 batching 原型 | CPU 分组/许可契约和同资源 GPU 数据 |
| 7–8 | 容量扫描、GSPO 与全参数 continuation | 对照结果、checkpoint/optimizer 一致性、故障回收 |
| 9–10 | 分批 PR 评审修改、文档与完整回归 | 每个 PR 的独立检查和依赖可复现 |
| 11–12 | 评审缓冲、结果复核与结项 | 实际上游合入证据；未完成项明确列出 |

**资源与风险**：双 GPU 主机及费用上限待确定；先小规模 smoke 再扩矩阵。
质量可能不改善，应交付可解释的结果；PR 合入依赖评审，不能以分支存在替代。
规则来源为资料分支保存的 2026-09-14 官网核查记录，提交前需由申请人复核当前规则。

## O3：PR 时机建议

先准备本地四个 review 分支及说明；seed 修复可作为独立社区贡献。主体 PR 的
创建时机须结合真实中选/签约状态与导师意见。中选前的开发应如实披露；晚发 PR
并不会自动把早期开发变成正式开工后的成果。本轮没有对外创建 PR 或 issue。

## O4：结项报告草稿

### 已完成的工程准备

实验 completion pipeline、单一 coordinator/supervisor、CPU payload 快照、
原生 worker/NCCL 适配与 LoRA continuation；旧提交有双 L4 验证。当前阶段补充
公开 lifecycle 接口、可选目标、参数化示例、用户文档和扩展实验工具。
这些是工程状态，不等同于已满足 OSPP 的时间与合入条件。

### 问题与解决

修复方向包括：并发关闭/等待接口、启动退出证据、默认 attention 可移植性、
模型引用元数据和实验参数不足。质量方面仅补充控制变量与消融，不声称退化已修复。
细节见 NEXT_REVIEW 和 KNOWN_ISSUES。

### 测试用例与结果

当前 CPU 候选与基线失败集合相同，新增契约检查通过；文档和工具入口检查已做。
旧提交 GPU 的吞吐、张量同步、故障与续训结果见 REWRITE_TODO。当前代码的双 GPU
重跑、质量矩阵、GSPO、全参数续训尚缺实测，报告中须保持这一边界。

### 后续安排

确定 GPU 预算与 OSPP 状态，重跑 native 故障/示例回归，再分阶段执行质量、吞吐、
容量与扩展矩阵；同步推进接口评审与主体 PR。最终报告需填写真实合入链接、
起止日期、维护者意见与尚未完成项，不能直接把本草稿提交为结项证明。
