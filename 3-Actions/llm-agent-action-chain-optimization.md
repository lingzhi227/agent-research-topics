# LLM Agent Action Chain 形式化表示、验证与效率优化：研究综述

> 研究日期: 2026-02-23
> 论文数据库: 762 篇候选论文 (Semantic Scholar + arXiv, 2023-2026)
> 筛选入围: 70 篇核心论文 (8 个研究方向)
> 深度阅读: 15 篇，其中 12 篇有详细结构化笔记
> 代码仓库: 14 篇论文对应代码，11 篇有公开仓库
>
> **研究流程**: Phase 1 (前沿搜索) → Phase 2 (762→70 论文筛选与聚类) → Phase 3 (15 篇深度阅读) → Phase 4 (代码仓库分析) → Phase 5 (跨论文综合) → Phase 6 (本报告)

---

## 目录

1. [研究问题与动机](#1-研究问题与动机)
2. [研究方向全景图](#2-研究方向全景图)
3. [方向一：形式语言约束的 Action Generation](#3-方向一形式语言约束的-action-generation)
4. [方向二：代码作为统一 Action 表示](#4-方向二代码作为统一-action-表示)
5. [方向三：有限状态机建模 Agent 工作流](#5-方向三有限状态机建模-agent-工作流)
6. [方向四：形式化验证（LTL / Model Checking）](#6-方向四形式化验证ltl--model-checking)
7. [方向五：MDP/POMDP 形式化与 RL 训练](#7-方向五mdppomdp-形式化与-rl-训练)
8. [方向六：Token 效率量化与优化](#8-方向六token-效率量化与优化)
9. [方向七：结构化/约束解码](#9-方向七结构化约束解码)
10. [综合分析：关键对比表](#10-综合分析关键对比表)
11. [研究空白与机会](#11-研究空白与机会)
12. [你的研究方向的定位与建议](#12-你的研究方向的定位与建议)
13. [参考文献](#13-参考文献)

---

## 1. 研究问题与动机

你提出的研究核心问题可以分解为三个子问题：

1. **Action Chain 的形式化表示**：给 LLM Agent 一种标准化的 action chain representation，让 LLM 生成可验证的格式
2. **Action Chain 的验证**：用形式化验证器/Python 解释器验证 long chain actions 的合法性
3. **Agent 执行效率的定量分析**：研究状态迁移概率、状态转移图、token cost、执行时间等指标

**结论先说**：这三个子问题都已有人在研究，但**尚无统一框架将三者整合**。现有工作大多只解决其中一个子问题。这是一个重要的研究空白。

---

## 2. 研究方向全景图

```
                    LLM Agent Action Chain 优化
                              │
          ┌──────────┬────────┼────────┬──────────┐
          │          │        │        │          │
    事前约束      统一表示    执行建模   事后验证    效率分析
          │          │        │        │          │
    ┌─────┴─────┐    │    ┌───┴───┐   │    ┌─────┴─────┐
    │           │    │    │       │   │    │           │
  文法约束    FSM   Code  MDP    RL  LTL  Token量化  轨迹压缩
  Formal-LLM  State CodeAct Agent Agent Bridge Tokenomics AgentDiet
  GAD         Flow  LLMFP  -R1   -RL FM    AgentTaxo   DEPO
  XGrammar    Meta  AFlow  RAGEN      VeriPlan        APC
              Agent
```

---

## 3. 方向一：形式语言约束的 Action Generation

### 3.1 Formal-LLM [@li2024formal] — CFG + PDA 约束 Agent 规划

**核心思想**: 将 agent 的规划约束编码为**上下文无关文法 (CFG)**，自动转换为**下推自动机 (PDA)**，在生成时约束 LLM 只能选择合法动作。

**技术细节**:
- CFG 四元组: 终结符（具体工具）、非终结符（工具类别）、起始符号、产生式规则
- PDA 三状态: q₀(起始) → q₁ → q₂(接受)
- 栈编码了计划的**树形结构**，支持多输入工具的数据流追踪
- 生成循环: 自动机枚举合法转移 → LLM 从合法选项中选择 → 更新栈 → 重复
- 回溯 (Backtracking) 保证完备性

**关键结果**:
| 模型 | Few-shot 合法率 | Formal-LLM 合法率 | 性能提升 |
|------|---------------|------------------|---------|
| GPT-3.5 | 71% | **100%** | +45% |
| Claude-2 | 47% | **100%** | +44% |
| GPT-4 | 76% | **100%** | +5% |
| Vicuna-7B (RLTF) | 29% | **100%** | +157% |

**局限**: 约束需手动编写；CFG 表达力有限（无法表达上下文相关约束）；无最优性保证。

### 3.2 LLM+AL [@ishay2025llmal] — Action Languages + 符号推理 (AAAI 2025)

将 LLM 语义解析能力与 Action Languages 的符号推理结合。在与 GPT-4、Claude 3 Opus 等对比中，LLM+AL + 人工校正能持续正确，而单独 LLM 即使经人工干预也无法改善。

**与你想法的关联**: Action Languages 本身就是一种成熟的 action chain 形式化表示，可以直接用于编码多种 action 类型的前置条件、效果和约束关系。

---

## 4. 方向二：代码作为统一 Action 表示

### 4.1 CodeAct [@wang2024codeact] — Python 代码作为动作空间 (ICML 2024)

**核心思想**: 用可执行 Python 代码替代 JSON/Text 作为 agent 的统一动作格式。

**为什么代码更优**:
| 维度 | Code | JSON/Text |
|------|------|-----------|
| 复杂操作 | 原生 if/for/变量 | 需要精心工程化，甚至不可行 |
| 工具可用性 | 直接 import Python 包 | 必须预定义工具集 |
| 自动反馈 | traceback 内建 | 需额外实现 |
| 验证 | **解释器即验证器** | 需外部解析器 |

**关键结果**:
- 17 个 LLM 中 12 个在 CodeAct 下取得最佳成功率
- GPT-4: 成功率从 53.7% 提升到 **74.4%** (+20.7%)
- 交互轮次从 7.7 减少到 **5.5** (-28.6%)

**与你想法的直接关联**: Python 解释器本身就是你提到的"验证器"——语法错误、类型错误、API 违规都会产生 traceback，agent 可据此自我修正。这是最 pragmatic 的 action chain 验证方案。

### 4.2 AFlow [@zhang2025aflow] — 自动化工作流搜索 (ICLR 2025 Oral)

**核心创新**: 将 agent 工作流表示为可执行代码，用 MCTS 自动搜索最优工作流结构。

**工作流表示**:
- **节点 (Node)**: LLM 调用单元，参数 = (模型M, 提示词P, 温度τ, 输出格式F)
- **边 (Edge)**: 代码控制的数据流
- **算子 (Operator)**: 8 种预设模式（Generate, Review, Revise, Ensemble, Test 等）

**搜索过程**: 初始化空模板 → 软混合概率选择 → LLM驱动扩展（修改代码） → 执行评估 → 经验回传 → 收敛判定

**关键结果**:
- 6 个基准平均提升 **5.7%**，比 ADAS 提升 **19.5%**
- GPT-4o-mini 优化后超越 GPT-4o，推理成本仅 **4.55%**

**与你想法的关联**: AFlow 本质上就是在优化 action chain 的结构。它证明了：(1) 代码是表示 action chain 的好格式；(2) 自动搜索比人工设计更有效；(3) 优化 action chain 结构比提升模型能力更具性价比。

### 4.3 LLMFP [@hao2025llmfp] — 形式化编程实现零样本规划 (ICLR 2025)

将规划问题形式化为优化问题，由 LLM 生成 Python 代码，求解器执行。GPT-4o 达到 **83.7%** 最优率，比 o1-preview 直接规划高 **37.6%**。

---

## 5. 方向三：有限状态机建模 Agent 工作流

### 5.1 StateFlow [@wu2024stateflow] — FSM 驱动的 LLM 工作流 (COLM 2024)

**核心设计**: 将 agent 工作流建模为**转换器有限状态机** ⟨S, s₀, F, δ, Γ, Ω⟩

**两个正交维度的分离**:
- **过程锚定 (Process Grounding)**: 通过 state + state transition 控制宏观流程
- **子任务求解 (Sub-task Solving)**: 每个 state 内的 LLM/工具调用

**状态设计示例 (InterCode SQL)**:
```
Init → Observe → Solve → Verify → End
                    ↓        ↓
                  Error ←──┘
```

**转移触发机制**:
1. 字符串匹配: 检测输出中的关键词（如 "Error"）
2. LLM 分类: 由 LLM 判断应转入哪个状态

**关键结果**:
| 基准 | StateFlow | ReAct | 提升 | 成本降低 |
|------|-----------|-------|------|---------|
| InterCode SQL | **63.73%** | 50.68% | +13% | **5x** |
| ALFWorld | **83.3%** | 55.5% | +28% | **2.5x** |

**Token 效率**: ReAct 平均 16,456 prompt tokens/任务，StateFlow 仅 3,128 tokens/任务 (**-81%**)

**与你想法的直接关联**: StateFlow 正是你所设想的"状态迁移模型"。它定义了明确的状态类型（plan、执行、验证、错误处理等），用转移函数控制状态间的流转。但它的局限在于状态机需要手动设计，且没有分析状态迁移概率。

### 5.2 MetaAgent [@zhang2025metaagent] — 自动构建 FSM 多 Agent 系统 (ICML 2025)

解决了 StateFlow 的手动设计问题：给定任务描述，**自动生成** FSM 架构。

### 5.3 StateAct [@rozanov2025stateact] — Chain-of-States (ACL 2025 Workshop)

引入**显式状态追踪**到 agent 推理中。在 ALFWorld 上超过 ReAct **10%+**，Textcraft 上超过 **30%**。

---

## 6. 方向四：形式化验证（LTL / Model Checking）

### 6.1 Bridging LLM Planning & Formal Methods [@bridge2025fm] (ASE 2025 Workshop)

**验证流程**: 自然语言计划 → LLM 转换 → Kripke 结构 + LTL 规约 → NuSMV 模型检查 → Valid/Invalid

**可验证的属性**:
- **目标可达性**: F(goal) — 最终到达目标
- **安全性**: G(NOT bad_state) — 永不进入危险状态
- **活性**: G(F(desired)) — 期望条件无限次发生
- **时序约束**: NOT (B U A) — 工具 A 必须在工具 B 之前

**结果**: GPT-5 形式化验证 F1 = **96.3%**，但直接判断更优 (99.65%)。核心问题：形式化转换引入语义损失。

### 6.2 LLMs as Planning Formalizers Survey [@tantakoun2025survey] (ACL 2025 Findings)

系统综述了 LLM → PDDL 的三条路径：模型生成、模型编辑、混合建模。核心发现：
- 简单域成功率高，复杂域出现**级联错误传播**
- 语法正确 ≠ 语义正确（核心瓶颈）
- 反馈循环是关键设计模式: 生成 → 验证 → 错误分析 → 自然语言反馈 → 重新生成

**对 action chain 验证的实用建议**: 分层验证（语法层 → 语义层 → 执行层）+ 增量形式化 + 多模型交叉验证

---

## 7. 方向五：MDP/POMDP 形式化与 RL 训练

### 7.1 Agent-R1 [@agentr1] — 完整的 LLM Agent MDP 形式化

**MDP 定义** (S, A, P, R):
- **S**: `(w_p, T_1, ..., T_k, T_{k+1}^partial)` — prompt + 交互回合序列
- **A**: Token 级别（但特定序列触发工具调用）
- **P**: 双组件分解 — P_G(确定性 token 拼接) + P_E(随机性环境响应)
- **R**: 混合奖励 — outcome reward (EM) + process reward (工具调用) + format penalty

**核心创新 — 动作掩码**: 精确区分 agent 生成的 token（可学习）vs 环境反馈 token（不可学习），实现精准信用分配。

**结果**: GRPO 算法在 HotpotQA 上 EM 从基线 0.08 提升到 **0.44** (+450%)

### 7.2 RAGEN [@ragen2025] — StarPO 框架与 Echo Trap 发现

**核心发现 — Echo Trap**:
- 训练过程中模型从多样推理坍缩为重复模板
- 四个预警指标: 奖励标准差骤降、梯度尖峰、策略熵骤降、输出多样性消失
- 根本原因: RL 强化了表层模式而非通用推理能力

**StarPO-S 稳定化**: 不确定性过滤（仅保留奖励方差最高的 25% prompt）+ 非对称 clipping + critic 基线

**关键洞察**:
- 最优交互粒度为**中等**（5-6 步），过少或过多都不好
- 推理 trace 随训练**自然压缩**（307 tokens → 89.5 tokens）
- 无推理感知奖励时，模型发展"虚假推理"

### 7.3 其他 MDP 工作

| 论文 | 核心贡献 |
|------|---------|
| **Agent Lightning** (Microsoft) | 统一 MDP 数据接口，可与 LangChain/AutoGen 无缝集成 |
| **AgentFly/Memento** | Memory-Augmented MDP (M-MDP)，增加 Case Memory 模块 |
| **RRO** | POMDP 形式化，基于上升奖励轨迹优化 |
| **AGILE** (NeurIPS 2024) | 7B/13B 模型经 PPO 训练后超越 GPT-4 agent |

---

## 8. 方向六：Token 效率量化与优化

### 8.1 Tokenomics [@tokenomics2026] — Token 消耗去向分析 (MSR 2026)

**核心发现**:
- Code Review 阶段占 **59.4%** 的总 token（最大消耗者）
- Input tokens 占总消耗的 **53.9%**（input:output ≈ 2:1）
- Coding 阶段仅占 **8.6%**（初始生成很便宜）
- 主要成本在于**迭代精炼和验证的"对话成本"**

### 8.2 AgentTaxo — Token 分布分类学 (ICLR 2025 Workshop)

**Agent 角色分类**: Planner / Reasoner / Verifier

**核心发现**:
- 验证阶段在 MetaGPT 中占 **72%** 的 token
- 首次定义"通信税 (Communication Tax)"概念
- 线性架构比层级架构通信税更低

### 8.3 效率优化方法

| 方法 | 类型 | 核心机制 | 效果 |
|------|------|---------|------|
| **Agentic Plan Caching** (NeurIPS 2025) | 推理时 | 缓存+复用成功的 plan 模板 | 成本 -50%, 延迟 -27% |
| **DEPO** (AAAI 2026) | 训练时 | 双效率偏好优化(步级+轨迹级) | token -60.9%, 步骤 -26.9%, 性能 +29.3% |
| **AgentDiet** | 推理时 | 轨迹压缩(移除冗余信息) | input token -40~60% |
| **Optima** (ACL 2025) | 训练时 | 多 agent 通信效率优化 | 性能 2.8x, token <10% |
| **TALE** (ACL 2025) | 推理时 | Token-budget-aware 推理 | 动态调节推理 token 数 |

### 8.4 DEPO 的双效率框架值得特别关注

DEPO 基于 POMDP 建模，定义了两个效率维度：
- **步级效率** `T̄_token(τ)`: 每步 token 数（信息密度）
- **轨迹级效率** `T_step(τ)`: 总步数（路径长度）

效率奖励: `b(τ) = α₁/T̄_token(τ) + α₂/T_step(τ)`

这**直接编码了"短路径+紧凑节点"的优化目标**，与你想研究的状态转移效率完全对应。

---

## 9. 方向七：结构化/约束解码

| 论文 | 会议 | 核心贡献 |
|------|------|---------|
| **GAD** (NeurIPS 2024) | 解决文法约束扭曲 LLM 分布的问题，保持分布保真性 |
| **XGrammar** (MLSys 2025) | 高效 CFG 约束解码引擎，10x 加速 |
| **XGrammar 2** (arXiv 2026) | 专为 Agentic LLM 设计的动态结构化生成，6x 加速 |
| **GRAMMAR-LLM** (ACL 2025) | 提出 LL(prefix) 文法类，平衡表达力与效率 |

XGrammar 2 特别值得关注——它是第一个专门为 agent 场景设计的约束解码引擎，支持动态切换约束规则（例如不同 tool 有不同的参数格式）。

---

## 10. 综合分析：关键对比表

### 10.1 Action Chain 表示方法对比

| 表示方法 | 代表论文 | 表达力 | 验证能力 | 自动化程度 | 实用性 |
|---------|---------|--------|---------|-----------|--------|
| CFG/PDA | Formal-LLM | 中 (上下文无关) | 100% 合法性 | 低 (手动编写) | 中 |
| FSM | StateFlow | 中 (有限状态) | 状态合法性 | 低→高 (MetaAgent) | 高 |
| Python Code | CodeAct | 高 (图灵完备) | 运行时验证 | 高 (自动生成) | **最高** |
| PDDL | LLMs as Formalizers | 高 (规划表达力) | 规划器验证 | 中 (LLM 转换) | 中 |
| LTL | Bridge FM | 高 (时序约束) | 模型检查 | 中 (LLM 转换) | 低 |
| Action Languages | LLM+AL | 高 (动作因果) | 符号推理 | 中 | 低 |
| Code Workflow | AFlow | 高 (含搜索) | 执行评估 | **最高** (MCTS) | 高 |
| MDP | Agent-R1 | 最高 (完整建模) | RL 优化 | 高 | 中 |

### 10.2 优化切入点对比

| 优化维度 | 方法 | 优化时机 | Token 减少 | 性能影响 |
|---------|------|---------|-----------|---------|
| 动作格式 | CodeAct | 设计时 | ~30% 步骤减少 | +20% |
| 工作流结构 | AFlow | 搜索时 | 成本 4.55% | +5.7% |
| 状态机约束 | StateFlow | 设计时 | **81%** | +13~28% |
| 文法约束 | XGrammar | 解码时 | 格式错误→0 | 中性 |
| 轨迹压缩 | AgentDiet | 推理时 | 40~60% | 中性 |
| 计划缓存 | APC | 推理时 | ~50% | 中性 |
| 策略优化 | DEPO | 训练时 | **60.9%** | +29.3% |

---

## 11. 研究空白与机会

基于对 762 篇候选论文中筛选出的 70 篇核心论文的系统调研，以及对 15 篇重点论文（12 篇详细结构化笔记）的深度阅读，我识别出以下关键研究空白：

### 空白 1: 状态迁移概率的系统性分析 ⭐⭐⭐

**现状**: 没有论文系统性地研究 P(state_i → state_j) 的分布。

StateFlow 定义了状态和转移，但没有分析实际执行中的转移概率分布。Agent-R1 分解了 P_G 和 P_E，但没有在具体状态类型上统计。RAGEN 发现了 Echo Trap（策略坍缩），但没有从状态转移概率的角度分析。

**机会**: 在 StateFlow 式的 agent 上收集大量执行 trace，统计各状态之间的转移概率矩阵。例如：
- P(Solve → Verify) vs P(Solve → Error) 如何随任务复杂度变化？
- 哪些状态是"吸收态"（agent 容易卡住的状态）？
- 状态停留时间分布是什么样的？

### 空白 2: 按状态类型的 Token Cost 分析 ⭐⭐⭐

**现状**: Tokenomics 按 SDLC 阶段分析了 token 分布，AgentTaxo 按 agent 角色分析。但没有人按**agent 状态类型**（plan、explore、execute、verify、rollback、memory management）分析 token cost。

**机会**: 定义标准化的 agent 状态类型（你提到的 plan、tool 探索、执行、reasoning、检查结果、回滚状态、context memory 管理），然后测量每种状态的:
- 平均 input/output/reasoning token 消耗
- 状态执行时间分布
- 状态成功率（进入该状态后成功完成子任务的概率）
- 状态的 ROI（token 投入 vs 对最终结果的贡献）

### 空白 3: 统一的 "形式化表示 + 验证 + 效率优化" 框架 ⭐⭐⭐

**现状**: 形式化表示（Formal-LLM, CodeAct）、验证（Bridge FM, VeriPlan）、效率优化（DEPO, APC）是三条独立的研究线。

**机会**: 构建统一框架：
1. 用 FSM/Code 作为 action chain 的表示
2. 用约束解码 (XGrammar) 确保生成的合法性
3. 用 LTL/Model Checking 验证关键属性
4. 用 MDP 框架分析和优化状态转移效率
5. 用 token cost 模型指导优化方向

### 空白 4: 动态自适应的 Action Grammar ⭐⭐

**现状**: Formal-LLM 的 CFG 是静态的；XGrammar 2 支持动态切换但仍是预定义的。

**机会**: 让 agent 根据执行历史和环境反馈**动态调整**自己的 action grammar。例如：
- 当连续多次工具调用失败时，自动收窄 action space
- 当验证通过时，允许更多自由探索
- 基于历史成功率调整各 action 的先验概率

### 空白 5: 状态缓存与上下文管理的形式化 ⭐⭐

**现状**: CORAL 提出了认知资源自分配，APC 提出了计划缓存，但都是启发式方法。

**机会**: 将 context window 管理形式化为一个优化问题——在有限的 context 预算下，最大化后续决策的信息量。这可以用信息论框架（mutual information, entropy）来建模。

### 空白 6: 跨任务的状态机迁移学习 ⭐

**现状**: StateFlow 为每个任务手动设计状态机；MetaAgent 自动生成但每次从头开始。

**机会**: 研究不同任务的状态机之间的结构相似性，实现状态机的迁移和复用。深度阅读发现 StateFlow 的 Plan、Error、End 状态在所有任务 FSM 中复现，暗示存在"通用 agent FSM 骨架"的可能。

### 空白 7: 验证成本悖论 ⭐⭐ (深度阅读新发现)

**现状**: 验证/review 消耗 59-72% 的 token，但对前沿模型的价值可疑。

**证据**:
- Bridge FM: GPT-5 直接判断 (99.59%) > GPT-5 + 形式化验证 (95.89%)
- StateFlow 消融: 移除 Verify 状态影响最小
- Tokenomics: Code Review = 59.4% tokens

**机会**: 研究何时形式化验证真正有价值、何时是浪费。开发自适应验证机制（仅对不确定预测触发）。

### 空白 8: Semi-Markov 过程建模 ⭐⭐ (深度阅读新发现)

**现状**: Agent 执行的状态具有可变持续时间，但现有 MDP 建模未考虑这一点。

**证据**:
- StateFlow: 不同状态 token 消耗差异巨大 (Observe ~400 tokens vs Solve 变长)
- Agent-R1: 标准 MDP 假设固定步长
- APC: 计划延迟占总延迟 71%

**机会**: 用 Semi-Markov Process 建模，引入状态停留时间分布 f(t|s) 和状态依赖成本率 c(s,t)，实现状态持续时间感知的优化。

---

## 12. 你的研究方向的定位与建议

### 12.1 你的想法在现有研究中的位置

你提出的方案实际上是一个**多层次的优化框架**，涉及现有研究的多个方向：

| 你的想法 | 最相关的现有工作 | 差距 |
|---------|----------------|------|
| "给 LLM 一种 action chain 的标准格式" | Formal-LLM (CFG), CodeAct (Python), StateFlow (FSM) | 现有工作各有一种表示，缺乏对比和统一 |
| "LLM 生成这种格式" | AFlow (代码工作流), LLMFP (优化问题) | 已有好的方案，Python 代码是当前最优选择 |
| "形式化验证器/Python 解释器验证" | Bridge FM (LTL), CodeAct (解释器), VeriPlan | Python 解释器最实用，LTL 更强但转换有损 |
| "研究状态迁移概率" | **几乎空白** | StateFlow 定义了状态但没分析概率 |
| "状态转移图" | StateFlow, MetaAgent | 有定义但没有量化分析 |
| "token cost per state" | Tokenomics (按 SDLC), AgentTaxo (按角色) | **没有按 agent 状态类型分析** |
| "执行时间 cost" | APC (延迟测量), Efficient Agents | 有测量但没有按状态分解 |

### 12.2 建议的研究路线

**路线 A: "Agent 执行过程的定量分析"（偏分析/理解）**

1. 定义标准化的 agent 状态类型分类学（taxonomy）
2. 在多个 agent 框架（Claude Code, OpenHands, AutoGen）上 instrument 执行 trace
3. 构建状态迁移概率矩阵 P(s_i → s_j)
4. 分析每种状态的 token cost 分布、执行时间分布、成功率
5. 识别效率瓶颈（哪些状态/转移路径是"浪费"最大的）
6. 提出基于数据的优化建议

**路线 B: "Action Chain 的形式化表示与验证"（偏方法/系统）**

1. 基于 StateFlow 的 FSM + CodeAct 的 Python，设计一种混合表示
2. FSM 控制宏观流程（状态迁移），Python 代码实现每个状态内的动作
3. 用 XGrammar 约束解码确保生成合法的 FSM 转移 + Python 代码
4. 用 Python 解释器做运行时验证，LTL 做关键属性的静态验证
5. 在标准 agent benchmark 上评估

**路线 C: "效率导向的 Agent 架构优化"（偏优化/工程）**

1. 结合 StateFlow (FSM 控制) + APC (计划缓存) + DEPO (策略优化)
2. 用状态迁移概率指导缓存决策（高频路径优先缓存）
3. 用 token cost 模型指导状态机设计（合并高成本低收益状态）
4. 用 RL 训练优化转移策略（RAGEN 的 StarPO 方法）
5. 目标：同时提升准确率和效率

### 12.3 最有 novelty 的切入点

基于空白分析，我认为最有发表价值的方向是：

> **"Agent 执行过程的 Markov 状态模型：定量分析与优化"**
>
> 将 LLM Agent 的执行过程建模为 Markov Chain / Semi-Markov Process，其中状态 = {Plan, Explore, Execute, Verify, Error, Rollback, Memory}，定义状态迁移概率、停留时间分布、token cost 函数。基于实证数据拟合模型参数，识别效率瓶颈，并提出优化状态转移策略。

这个方向的 novelty 在于：
1. **没有人**从 Markov 状态模型的角度系统分析 agent 执行效率
2. 结合了 StateFlow 的状态机思想 + Tokenomics 的量化分析 + MDP 的形式化建模
3. 可以产出**定量的、可复现的**分析结果（概率矩阵、cost 曲线、瓶颈识别）
4. 直接指导 agent 架构优化（数据驱动而非直觉驱动）

### 12.4 深度阅读后的补充洞察

基于 12 篇论文的详细结构化分析，以下是跨论文的关键发现：

**发现 1: "结构同时降低成本并提高准确率"原则**
- StateFlow: +28% 成功率 AND -81% token (FSM 结构)
- CodeAct: +20% 成功率 AND -30% 交互轮数 (代码结构)
- DEPO: +29.3% 成功率 AND -60.9% token (效率训练)
- AFlow: +5.7% 准确率 AND 95.45% 成本降低 (工作流结构)

这与朴素直觉相反（更好的结果需要更多计算）。**结构减少了 LLM 每步的决策空间，产生更短的 prompt、更少的错误、更清晰的状态追踪。**

**发现 2: 小模型 + 结构 ≈ 大模型**
- APC: LLaMA-8B 适配缓存计划
- Plan-and-Act: 8B+CoT ≈ 70B (~9x 参数减少)
- AFlow: GPT-4o-mini + 优化工作流 ≈ GPT-4o (4.55% 成本)

**发现 3: DEPO 的非对称效率奖励是关键设计洞察**
- 仅对成功轨迹奖励效率 → 有效
- 对失败轨迹也惩罚效率 → 性能**恶化** (+46.5% 更差)
- 这暗示效率优化必须与任务成功条件耦合

**发现 4: 代码仓库生态已相当成熟**
- 14 篇论文中 11 篇有公开代码
- 关键框架: veRL (RL 训练), AutoGen/LangGraph (FSM 运行时), XGrammar (约束解码)
- RAGEN (~1.9K stars) 和 XGrammar (~1.3K stars) 是最活跃的社区项目

---

## 13. 参考文献

### 形式语言与约束
- [@li2024formal] Zelong Li et al. "Formal-LLM: Integrating Formal Language and Natural Language for Controllable LLM-based Agents." arXiv 2402.00798, 2024.
- [@ishay2025llmal] Adam Ishay, Joohyung Lee. "LLM+AL: Bridging Large Language Models and Action Languages for Complex Reasoning about Actions." **AAAI 2025**.
- [@xu2024core] Shuyuan Xu et al. "CoRE: LLM as Interpreter for Natural Language Programming." arXiv 2405.06907, 2024.

### 代码作为动作
- [@wang2024codeact] Xingyao Wang et al. "CodeAct: Executable Code Actions Elicit Better LLM Agents." **ICML 2024**.
- [@zhang2025aflow] Jiayi Zhang et al. "AFlow: Automating Agentic Workflow Generation." **ICLR 2025 Oral**.
- [@hao2025llmfp] Yilun Hao et al. "Planning Anything with Rigor: General-Purpose Zero-Shot Planning with LLM-based Formalized Programming." **ICLR 2025**.

### 状态机建模
- [@wu2024stateflow] Yiran Wu et al. "StateFlow: Enhancing LLM Task-Solving through State-Driven Workflows." **COLM 2024**.
- [@zhang2025metaagent] Yaolun Zhang et al. "MetaAgent: Automatically Constructing Multi-Agent Systems Based on Finite State Machines." **ICML 2025**.
- [@rozanov2025stateact] Nikolai Rozanov, Marek Rei. "StateAct: Enhancing LLM Base Agents via Self-prompting and State-tracking." **ACL 2025 REALM Workshop**.

### 形式化验证
- [@bridge2025fm] "Bridging LLM Planning Agents and Formal Methods: A Case Study in Plan Verification." arXiv 2510.03469, ASE 2025 Workshop.
- [@veriplan2025] "VeriPlan: Integrating Formal Verification and LLMs into End-User Planning." **CHI 2025**.
- [@verifyllm2025] "Plan Verification for LLM-Based Embodied Task Completion Agents." arXiv 2509.02761, 2025.
- [@ltlcrit2025] "Improving Embodied LLM Agents through Temporal Logic Based Critics." arXiv 2507.03293, 2025.
- [@tantakoun2025survey] Marcus Tantakoun et al. "LLMs as Planning Formalizers: A Survey." **ACL 2025 Findings**.
- [@huang2025limits] Cassie Huang, Li Zhang. "On the Limit of Language Models as Planning Formalizers." **ACL 2025**.

### MDP/RL 训练
- [@agentr1] Mingyue Cheng et al. "Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning." arXiv 2511.14460, 2025.
- [@agentlightning] Xufang Luo et al. "Agent Lightning: Train ANY AI Agents with Reinforcement Learning." arXiv 2508.03680, Microsoft Research, 2025.
- [@agentfly] "AgentFly: Fine-tuning LLM Agents without Fine-tuning LLMs." arXiv 2508.16153, 2025.
- [@ragen2025] "RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning." arXiv 2504.20073, 2025.
- [@agile2024] Peiyuan Feng et al. "AGILE: A Novel Reinforcement Learning Framework of LLM Agents." **NeurIPS 2024**.
- [@agentrl2025] Hanchen Zhang et al. "AgentRL: Scaling Agentic Reinforcement Learning." arXiv 2510.04206, 2025.

### 效率分析与优化
- [@tokenomics2026] Mohamad Salim et al. "Tokenomics: Quantifying Where Tokens Are Used in Agentic Software Engineering." **MSR 2026**.
- [@agenttaxo2025] Qian Wang et al. "AgentTaxo: Dissecting and Benchmarking Token Distribution of LLM Multi-Agent Systems." ICLR 2025 Workshop.
- [@apc2025] Qizheng Zhang et al. "Agentic Plan Caching: Test-Time Memory for Fast and Cost-Efficient LLM Agents." **NeurIPS 2025**.
- [@depo2026] Sirui Chen et al. "DEPO: Dual-Efficiency Preference Optimization for LLM Agents." **AAAI 2026**.
- [@agentdiet2025] Yuan-An Xiao et al. "AgentDiet: Improving the Efficiency of LLM Agent Systems through Trajectory Reduction." arXiv 2509.23586, 2025.
- [@optima2025] Weize Chen et al. "Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System." **ACL 2025 Findings**.
- [@tale2025] Tingxu Han et al. "Token-Budget-Aware LLM Reasoning." **ACL 2025 Findings**.
- [@efficientagents2025] Ningning Wang et al. "Efficient Agents: Building Effective Agents While Reducing Cost." arXiv 2508.02694, 2025.

### 结构化生成
- [@gad2024] Kanghee Park et al. "Grammar-Aligned Decoding." **NeurIPS 2024**.
- [@xgrammar2025] Yixin Dong et al. "XGrammar: Flexible and Efficient Structured Generation Engine." **MLSys 2025**.
- [@xgrammar2_2026] Linzhang Li et al. "XGrammar 2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs." arXiv 2601.04426, 2026.
- [@grammarllm2025] "GRAMMAR-LLM: Grammar-Constrained Natural Language Generation." **ACL 2025 Findings**.

### 长程规划与基准
- [@planandact2025] Lutfi Eren Erdogan et al. "Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks." **ICML 2025**.
- [@coral2025] "CORAL: Don't Lose the Thread - Empowering Long-Horizon LLM Agents with Cognitive Resource Self-Allocation." OpenReview, 2025.
- [@deepplanning2026] Yinger Zhang et al. "DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints." arXiv 2601.18137, Qwen team, 2026.
- [@theagentcompany2025] Frank F. Xu et al. "TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks." **NeurIPS 2025**.
- [@webdreamer2025] Gu, Zheng et al. "WebDreamer: Is Your LLM Secretly a World Model of the Internet?" **TMLR 2025**.
- [@routine2025] Guancheng Zeng et al. "Routine: A Structural Planning Framework for LLM Agent System in Enterprise." arXiv 2507.14447, 2025.

### Benchmark & Survey
- [@eto2024] Yifan Song et al. "Trial and Error: Exploration-Based Trajectory Optimization of LLM Agents." **ACL 2024**.
- [@tpsbench2025] Hanwen Xu et al. "TPS-Bench: Evaluating AI Agents' Tool Planning & Scheduling Abilities." arXiv 2511.01527, 2025.
- [@agentsurvey2025] "Evaluation and Benchmarking of LLM Agents: A Survey." arXiv 2507.21504, 2025.
