# Agent OS: Context Engineering, Memory Management, and Multi-Agent Coordination for LLM Benchmark Enhancement

> **Research Report** — Systematic Literature Review
> Date: 2026-02-25
> Papers surveyed: 85 (76 peer-reviewed, 9 preprints)
> Deep-read papers: 12
> GitHub repositories: 28

---

## Executive Summary

本报告系统综述了 **Agent OS** 范式下提升 LLM agent benchmark 性能的最新研究。核心发现：**不改模型权重，仅通过系统级设计（上下文工程、记忆管理、多智能体协调、推理时计算分配）即可实现显著的 benchmark 提升**，在部分场景下效果超越模型训练。

**关键量化证据**：
- **AFlow** (ICLR 2025 Oral): GPT-4o-mini + 优化 workflow = GPT-4o 性能，仅需 **4.55% 的成本**
- **MLE-bench** (ICLR 2025 Oral): 同一 GPT-4o 模型，AIDE vs MLAB scaffold = **11 倍性能差距**
- **ACE** (ICLR 2026): 自进化上下文 playbook → **+17.0 pp** AppWorld 性能，零重训练
- **MemoryOS** (EMNLP 2025 Oral): OS-inspired 三层记忆 → **+49.11% F1**，零重训练
- **RocketKV** (ICML 2025): KV cache **400 倍压缩**，**3.7 倍加速**，精度损失可忽略

这些结果共同证明：**agent 的系统架构（scaffold）设计对性能的影响可以与甚至超过模型本身的能力**。对于无法自主训练大模型的研究者和开发者，Agent OS 方法提供了高效、可行的能力增强路径。

---

## 1. Introduction

### 1.1 Motivation

LLM agent 在科学推理、工具调用、长上下文任务上仍面临严峻挑战。现有 benchmark 数据表明：
- **BixBench** (生物信息学): 最佳成绩仅 **17%**
- **PaperBench** (论文复现): 仅 **26.6%**
- **ScienceAgentBench**: 多步科学实验任务仍是瓶颈

传统提升路径依赖模型训练：预训练需要海量数据和算力，SFT 依赖高质量标注数据，RL (PPO/GRPO) 的奖励函数设计困难。更关键的是，**这些方法对一般研究者不可行**。

### 1.2 Agent OS 假说

一个替代思路正在学术界获得广泛验证：**将 LLM 视为 CPU，围绕它构建操作系统级的基础设施**——上下文工程（Context Engineering）、记忆管理（Memory Management）、多智能体协调（Multi-Agent Coordination）、推理时资源分配（Inference-Time Compute）——可以在不改变模型权重的前提下显著提升 agent 性能。

本报告系统综述了 2023-2026 年间 85 篇相关论文和 28 个开源项目，构建了 Agent OS 的完整技术图谱。

### 1.3 Report Scope

| 维度 | 范围 |
|------|------|
| 时间跨度 | 2023 Oct – 2026 Feb |
| 论文来源 | ICLR 2025/2026, NeurIPS 2025, ICML 2025, ACL 2025, EMNLP 2025, COLM 2025 + arXiv |
| 论文总量 | 85 篇（76 peer-reviewed, 9 preprints） |
| 深度阅读 | 12 篇核心论文 |
| 开源项目 | 28 个 GitHub 仓库 |

---

## 2. Taxonomy: The Agent OS Architecture Stack

独立的研究团队从不同角度出发，但最终收敛到了一个与传统操作系统高度类比的分层架构：

```
┌─────────────────────────────────────────────────────────┐
│ Layer 5: Workflow Compiler/Optimizer                     │
│   AFlow (MCTS workflow search), MAS-GPT (single-pass)   │
├─────────────────────────────────────────────────────────┤
│ Layer 4: Context Engineering                             │
│   ACE (evolving playbooks), PAACE (plan-aware compress)  │
│   ACON (observation/history compression)                 │
├─────────────────────────────────────────────────────────┤
│ Layer 3: Memory Management                               │
│   MemGPT (virtual context), MemoryOS (3-tier storage)    │
│   A-MEM (Zettelkasten self-organization)                 │
├─────────────────────────────────────────────────────────┤
│ Layer 2: Multi-Agent Coordination                        │
│   MacNet (DAG topology, scaling law)                     │
│   MAS-GPT (auto-generate MAS)                            │
├─────────────────────────────────────────────────────────┤
│ Layer 1: Runtime/Kernel                                  │
│   AIOS (scheduler + context switch + access control)     │
│   RocketKV (KV cache as virtual memory)                  │
│   Test-Time Compute (resource allocation)                │
└─────────────────────────────────────────────────────────┘
```

### 2.1 Full Taxonomy

```
Agent OS Approaches (不改模型权重，通过系统设计提升 benchmark)
│
├── A. Context Engineering (上下文工程)
│   ├── A1. Context Accumulation — ACE (ICLR 2026): evolving playbooks
│   ├── A2. Context Compression — ACON, PAACE, ReSum, CCF
│   ├── A3. Context Steering — CoS (ICLR 2025): amplify context influence
│   └── A4. Context Optimization — Joint Optimization (ACL 2025 Findings)
│
├── B. Memory Management (记忆管理)
│   ├── B1. Virtual Context — MemGPT (ICLR 2024): OS-inspired paging
│   ├── B2. Hierarchical Storage — MemoryOS (EMNLP 2025 Oral): STM/MTM/LPM
│   ├── B3. Self-Organizing Memory — A-MEM (NeurIPS 2025): Zettelkasten
│   ├── B4. Graph Memory — Mem0, AriGraph, SYNAPSE
│   ├── B5. Learnable Memory — AgeMem: RL-trained memory operations
│   └── B6. Memory OS — MemOS: memory as first-class compute resource
│
├── C. Multi-Agent Coordination (多智能体协调)
│   ├── C1. Topology — MacNet (ICLR 2025): DAG + scaling law
│   ├── C2. Orchestration — Evolving Orch. (NeurIPS 2025): RL puppeteer
│   ├── C3. Workflow — AFlow (ICLR 2025 Oral): MCTS code search
│   ├── C4. Auto MAS — MAS-GPT (ICML 2025): single-pass generation
│   └── C5. Debate — DOWN, ReSo (EMNLP 2025): adaptive debate
│
├── D. Agent Runtime & Infrastructure (运行时基础设施)
│   ├── D1. OS Kernel — AIOS (COLM 2025): scheduler + managers
│   ├── D2. Scheduling — Agent.xpu, Astraea: state-aware scheduling
│   ├── D3. Serving — Nalar: workflow-level serving framework
│   └── D4. Protocols — MCP (Anthropic), A2A (Google), LACP
│
├── E. Reasoning Scaffolds (推理脚手架)
│   ├── E1. Topology Evolution — CoT → ToT → GoT → AGoT → FoT
│   ├── E2. MCTS Search — LATS (ICML 2024), SC-MCTS*, ReKG-MCTS
│   ├── E3. Self-Reflection — PR-CoT, Reflexion, MAR
│   └── E4. Cognitive-Inspired — Sketch-of-Thought, Dual-Process
│
└── F. Inference-Time Compute (推理时计算)
    ├── F1. Scaling Laws — Snell et al. (ICLR 2025 Oral)
    ├── F2. Adaptive Allocation — SelfBudgeter, BudgetThinker
    ├── F3. KV Compression — RocketKV (ICML 2025), ChunkKV
    └── F4. Prompt Compression — LLMLingua-2, PromptDistill
```

### 2.2 OS Analogy Mapping

| OS Concept | Agent OS Equivalent | Key Paper(s) |
|-----------|-------------------|------------|
| CPU | LLM inference engine | AIOS |
| Virtual Memory | Virtual context management | MemGPT (ICLR 2024) |
| RAM | Context window | All |
| Page Replacement | Context compression/eviction | ACON (preprint), PAACE (preprint), RocketKV (ICML 2025) |
| File System | Persistent memory storage | Mem0 (preprint), A-MEM (NeurIPS 2025) |
| Process Scheduler | Agent/task scheduler | AIOS (COLM 2025), Agent.xpu (preprint), Astraea (preprint) |
| Compiler | Workflow optimizer | AFlow (ICLR 2025 Oral), MAS-GPT (ICML 2025) |
| System Calls | Tool/API calls | MCP (Anthropic), A2A (Google) |
| IPC | Agent communication | MacNet (ICLR 2025), A2A Protocol |
| Config Files | Evolving playbooks | ACE (ICLR 2026) |
| Memory Hierarchy | STM/MTM/LPM tiers | MemoryOS (EMNLP 2025 Oral) |
| Kernel | AIOS Kernel | AIOS (COLM 2025) |

---

## 3. Context Engineering — 上下文工程

**Context engineering** 已从 2025 年起成为独立于 prompt engineering 的正式学科。它关注的核心问题是：**什么信息应在什么时间以什么形式进入上下文窗口**。

### 3.1 ACE: Agentic Context Engineering (ICLR 2026)

**核心思想**: 将上下文视为可进化的 **playbook**（操作手册），通过 Generator-Reflector-Curator 三角色循环持续积累领域知识。

**机制**:
1. **Generator**: 执行任务，记录成功策略
2. **Reflector**: 分析失败，生成改进建议
3. **Curator**: 整合新知识，避免 context collapse（迭代重写导致的信息侵蚀）

**关键创新**: 解决了 **brevity bias**（LLM 倾向于生成简短摘要丢失关键细节）和 **context collapse**（迭代重写导致上下文退化）两大问题。

**实验结果**:
| Benchmark | Improvement | Note |
|-----------|------------|------|
| AppWorld | **+17.0 pp** | 匹配顶级 agent |
| GAIA | +9.0% avg | 跨 3 level |
| WebArena | 提升显著 | 与 PAACE 互补 |

**对科学推理的意义**: ACE 最直接地验证了 "不改权重，通过上下文设计提升性能" 的范式。其 evolving playbook 机制特别适合科学推理——科学实验中的领域知识（实验参数、分析流程、常见错误）可以自然地积累到 playbook 中。

### 3.2 PAACE: Plan-Aware Agent Context Engineering (preprint)

**核心思想**: 将上下文压缩从 "一刀切" 升级为 **plan-aware**（感知计划状态的）上下文工程。

**统一框架**: PAACE 将以下 4 种策略统一为一个优化框架：
1. **Pruning** — 删除无关历史
2. **Summarization** — 压缩保留语义
3. **Rewriting** — 重组上下文结构
4. **Compression** — Token-level 压缩

**关键创新**: 训练了 4B 的上下文压缩模型 (distilled from larger teacher)，在压缩过程中保留 plan-relevant 信息。

**实验结果**:
- AppWorld: **59.0% SOTA**，超越所有已知方法
- 达到教师模型 **97% 性能** @ **10x 成本削减**

### 3.3 ACON: Context Compression for Long-Horizon Agents (preprint)

**核心思想**: 分别压缩 **observations** (环境观察) 和 **history** (交互历史) 两类上下文。

**关键发现**:
- 对小模型尤其有效：**+46% 性能提升**
- 大模型 (GPT-4o) 受益较小，但仍可节省 **26-54% tokens**
- 95%+ 原始准确率保留

**启示**: 上下文压缩是 "穷人的上下文工程"——不需要复杂的 playbook 演化，只需智能压缩即可显著提升小模型性能。

### 3.4 Context Engineering 小结

| 方法 | 类型 | Benchmark 提升 | 成本影响 | 需要重训练? |
|------|------|---------------|---------|------------|
| ACE | 上下文积累 | +17.0 pp AppWorld | -82-92% latency | 否 |
| PAACE | 计划感知压缩 | SOTA 59.0% AppWorld | 10x 成本削减 | 4B distill |
| ACON | 观察/历史压缩 | +46% (小模型) | -26-54% tokens | 否 |
| CoS | 上下文引导 | 提升个性化 | 零额外成本 | 否 |
| Joint Opt | 工具上下文优化 | 提升工具调用 | 优化时间 | 否 |

**互补关系**: ACE 负责 **知识积累**（write path），ACON/PAACE 负责 **高效压缩**（read path）。两者构成上下文管理的完整读写路径。

---

## 4. Memory Management — 记忆管理

Agent 的记忆系统解决的核心问题是：**如何在有限的上下文窗口中管理跨越长时间范围的信息**。2025-2026 年，这一领域经历了从启发式到 OS-inspired 到可学习的快速演化。

### 4.1 MemGPT: LLMs as Operating Systems (ICLR 2024)

**奠基性工作**: 首次提出将 LLM 类比为 OS，上下文窗口类比为 RAM，外部存储类比为磁盘。

**核心机制**:
- **Virtual context management**: 自动在 "main context"（RAM）和 "external storage"（disk）间分页
- **自触发函数调用**: Agent 自主决定何时 load/save 数据
- **分层记忆**: Conversation buffer (L1 cache) → Recall memory (RAM) → Archival memory (Disk)

**实验结果**:
- Deep retrieval 准确率: **32% → 92.5%** (GPT-3.5 级别模型)
- 多会话对话质量显著提升

**影响**: 开启了 "Memory as OS" 范式，后续的 AIOS、MemoryOS、MemOS 都直接受其影响。

### 4.2 MemoryOS: OS Memory for AI Agents (EMNLP 2025 Oral)

**核心思想**: 将操作系统的内存层级直接映射到 agent 记忆系统。

**三层存储架构**:
```
STM (Short-Term Memory)    ← L1 Cache: 当前对话上下文
    ↕ Updating Module
MTM (Medium-Term Memory)   ← RAM: 最近交互的摘要
    ↕ Updating Module
LPM (Long-Persistent Memory) ← Disk: 持久化人物/知识档案
```

**四大模块**:
1. **Storage Module**: 管理三层存储结构
2. **Updating Module**: 层间信息流转（上升 = 巩固，下降 = 遗忘）
3. **Retrieval Module**: 智能检索相关记忆片段
4. **Generation Module**: 将检索到的记忆注入上下文

**实验结果**:
| Metric | Improvement |
|--------|------------|
| F1 (LoCoMo) | **+49.11%** |
| BLEU-1 | **+46.18%** |
| ROUGE-L | 显著提升 |

**对科学推理的意义**: 科学实验天然需要三层记忆——当前实验步骤 (STM)、本次实验历史 (MTM)、领域知识和先前实验 (LPM)。MemoryOS 的架构可以直接映射。

### 4.3 A-MEM: Agentic Memory (NeurIPS 2025)

**核心思想**: 受 **Zettelkasten**（卡片笔记法）启发，让 agent 自主管理记忆的组织结构。

**关键创新**:
- 记忆不是被动存储，而是 agent 主动的 **activate → link → evolve** 循环
- 每条记忆包含：观察、思考、关联链接
- 形成动态的 **interconnected knowledge network**

**实验结果**:
- Multi-hop QA: **2 倍提升**
- Token 消耗: 仅需 **1/7**

**对比 MemoryOS**: MemoryOS 是确定性的 OS-inspired 层级，A-MEM 是涌现式的自组织网络。两者代表了记忆管理的两种哲学——**结构化 vs 涌现式**。

### 4.4 Other Memory Systems

| System | Venue | Approach | Key Result |
|--------|-------|----------|-----------|
| **AgeMem** | preprint | RL 训练 LTM/STM 统一管理 | 3-stage progressive RL; memory ops as tools |
| **Mem0** | preprint | 生产级图+向量记忆 | +26% over OpenAI; 91% lower p95 latency; 48K GitHub stars |
| **MemOS** | preprint | Memory as first-class compute | +159% temporal reasoning vs OpenAI |
| **SYNAPSE** | preprint | Episodic-semantic spreading activation | Solves "Contextual Tunneling" |
| **CAM** | NeurIPS 2025 | Piaget-inspired schema assimilation | Incremental memory clustering |
| **AriGraph** | IJCAI 2025 | KG world model + episodic memory | Surpasses all RL baselines in TextWorld |

### 4.5 Memory Management 小结

**演化趋势**: Heuristic (手工规则) → OS-inspired (MemGPT/MemoryOS) → Self-organizing (A-MEM) → RL-learned (AgeMem)

**关键 gap**: 现有记忆系统为通用对话设计，缺乏对科学知识的特化处理（公式、实验参数、统计结果）。

---

## 5. Multi-Agent Coordination — 多智能体协调

### 5.1 AFlow: Automating Agentic Workflow Generation (ICLR 2025 Oral)

**核心贡献**: 将 agent workflow 表示为 **code**，使用 **MCTS (Monte Carlo Tree Search)** 在 workflow 空间中自动搜索最优组合。

**方法**:
1. 将 workflow 表示为 Python 代码（node = LLM 操作，edge = 数据流）
2. 使用 MCTS 搜索 workflow 空间：selection → expansion → evaluation → backpropagation
3. LLM 作为 MCTS 的 operator，提议 workflow 修改

**实验结果**:
| 发现 | 数据 |
|------|------|
| 平均提升 | +5.7% across 6 benchmarks |
| 成本 | 仅需人工设计的 **4.55%** |
| **最强结果** | **GPT-4o-mini + AFlow ≈ GPT-4o** |

**对科学推理的意义**: AFlow 是 "scaffold > model" 假设的最强证据。其 MCTS 方法可以扩展到科学任务——将实验 pipeline 表示为 code workflow，自动搜索最优实验流程。

### 5.2 MacNet: Scaling Multi-Agent Collaboration (ICLR 2025)

**核心发现**: **Collaborative Scaling Law** — 多 agent 协作性能遵循 logistic 增长曲线。

**DAG 拓扑**:
- 将任务分解为 DAG（有向无环图），每个 node 分配一个 agent
- Agents 通过 topological ordering 执行
- 支持 1000+ agents 协作

**Scaling Law**:
```
Performance = L / (1 + exp(-k(x - x₀)))
```
- 初始快速增长 → ~32 agents 后饱和 → 边际收益递减

### 5.3 MAS-GPT: Training LLMs to Build MAS (ICML 2025)

**核心思想**: 训练一个 LLM 直接 **一次性生成** 完整的 multi-agent system（角色、工具、workflow）。

**方法**: 收集 MAS 设计-性能对 → fine-tune LLM → 单次推理生成 MAS 配置

**结果**: 9 个 benchmark 上超越 10+ baselines，比手工设计的 MAS 平均提升 3.89%。

### 5.4 Protocols: MCP & A2A

**Model Context Protocol (MCP)** — Anthropic → Linux Foundation:
- JSON-RPC 标准化 LLM-工具交互
- 已被 OpenAI、Google、Microsoft 采纳
- GitHub: 79K+ stars (servers), 22K+ stars (Python SDK)
- 相当于 Agent OS 的 **POSIX syscall interface**

**Agent-to-Agent Protocol (A2A)** — Google → Linux Foundation:
- HTTP/SSE/JSON-RPC 的 agent 间通信协议
- 150+ 组织参与，v0.3 支持 gRPC streaming
- 相当于 Agent OS 的 **IPC 机制**

### 5.5 AIOS: LLM Agent Operating System (COLM 2025)

**最完整的 Agent OS 实现**，显式构建了：
- **Agent Scheduler**: FIFO, Round-Robin, Priority-based
- **Context Manager**: 保存/恢复 agent 上下文
- **Memory Manager**: 短期+长期
- **Storage Manager**: 持久化
- **Tool Manager**: 注册/调用外部工具
- **Access Controller**: 权限管理

**结果**: 2.1x 吞吐量提升，无精度损失。

---

## 6. Reasoning Scaffolds & Inference-Time Compute

### 6.1 Test-Time Compute Scaling (ICLR 2025 Oral)

**奠基性工作** (Snell et al.): 证明在推理时 **自适应分配计算资源** 比盲目增加模型大小更高效。

**核心发现**:
- PRM (Process Reward Model) 引导的搜索 + 自适应分配 → **2-4x 效率提升**
- 小模型 + 充足推理时间 ≥ 大模型（在特定难度区间）
- "Compute-optimal" 策略因问题难度而异

**启示**: 不是所有问题都需要同等的 token budget。Agent OS 应该包含 **task-aware resource allocator**。

### 6.2 RocketKV: KV Cache Compression (ICML 2025)

**问题**: 长上下文推理中，KV cache 占用大量 GPU 内存。

**方法**: 两阶段 — Pruning (粗粒度移除不重要 head) + Sparse Attention (细粒度稀疏化)

**结果**:
- **400x KV cache 压缩**
- **3.7x 解码加速**
- 在 128K context 下精度损失可忽略

**对科学推理的意义**: 科学任务通常涉及长文档（论文、数据集描述、实验日志），RocketKV 使在有限 GPU 上处理这些长上下文成为可能。

### 6.3 Reasoning Topology Evolution

推理结构从线性链进化到动态图：

```
CoT (线性链) → ToT (树搜索) → GoT (图推理) → AGoT (自适应图) → FoT (框架)
```

- **AGoT** (preprint): 动态构建 DAG 推理结构 → **+46.2% on GPQA**
- **Sketch-of-Thought** (EMNLP 2025): 认知启发的简略推理 → **84% token reduction**, 近零精度损失
- **LATS** (ICML 2024): MCTS + LLM → **92.7% pass@1 on HumanEval**

---

## 7. Comparative Analysis

### 7.1 Approach vs Benchmark Impact

| Approach | Representative | Venue | Benchmark Impact | Cost Impact | Retraining? |
|----------|--------------|-------|-----------------|-------------|-------------|
| Context Playbooks | ACE | ICLR 2026 | +17.0 pp AppWorld | -82-92% latency | No |
| Plan-Aware Compress | PAACE | →ICML (preprint) | SOTA 59.0% AppWorld | 10x cost reduction | 4B distill |
| Context Compress | ACON | Preprint | +46% small models | -26-54% tokens | No |
| Virtual Context | MemGPT | ICLR 2024 | 32→92.5% deep retrieval | Moderate | No |
| Hierarchical Memory | MemoryOS | EMNLP 2025 Oral | +49% F1 LoCoMo | Low overhead | No |
| Zettelkasten Memory | A-MEM | NeurIPS 2025 | 2x multi-hop | 1/7 tokens | No |
| MCTS Workflow | AFlow | ICLR 2025 Oral | +5.7% avg 6 benchmarks | 4.55% cost | No |
| Auto MAS Gen | MAS-GPT | ICML 2025 | +3.89% avg 8 benchmarks | Single pass | Fine-tune |
| DAG Coordination | MacNet | ICLR 2025 | Logistic emergence | Scalable | No |
| OS Scheduling | AIOS | COLM 2025 | 2.1x throughput | No acc loss | No |
| KV Compression | RocketKV | ICML 2025 | ~0 acc loss at 400x | 3.7x speedup | No |
| Test-Time Compute | Snell et al. | ICLR 2025 Oral | 2-4x efficiency | Smaller model OK | No |

### 7.2 Timeline of Key Developments

| Year | Milestone | Significance |
|------|-----------|-------------|
| 2023 Oct | MemGPT (arXiv) | OS virtual memory analogy for LLMs |
| 2024 Feb | AIOS v1 (arXiv) | First full Agent OS kernel |
| 2024 Jun | MacNet (arXiv) | Collaborative scaling law |
| 2024 Aug | Test-Time Compute (arXiv) | Inference scaling > model scaling |
| 2024 Oct | AFlow (arXiv) | MCTS workflow optimization |
| 2024 Nov | MCP (Anthropic) | Tool protocol standardization |
| 2025 Feb | A-MEM (arXiv) | Self-organizing memory |
| 2025 Apr | A2A Protocol (Google) | Agent interoperability |
| 2025 May | MemOS, RocketKV | Memory OS + efficient inference |
| 2025 Jun | MemoryOS (arXiv) | OS memory hierarchy for agents |
| 2025 Jul | Context Engineering Survey | Field formalization |
| 2025 Oct | ACE, ACON, PAACE | Context engineering trifecta |
| 2026 Jan | AgeMem, SYNAPSE | RL-learned memory + graph memory |
| 2026 Feb | MemoryArena, Codified Context | Benchmarks + production infra |

---

## 8. Open-Source Ecosystem

### 8.1 Core Agent OS Implementations

| Project | Stars | Description |
|---------|-------|-------------|
| **AIOS** (agiresearch/AIOS) | 5.2K | LLM Agent OS: scheduler + context + memory + tools |
| **MemOS** (MemTensor/MemOS) | 5.9K | AI memory OS with persistent skill memory |
| **Letta/MemGPT** (letta-ai/letta) | 21.3K | Stateful agents with virtual context management |
| **MemoryOS** (BAI-LAB/MemoryOS) | 1.2K | EMNLP 2025 Oral: hierarchical memory OS |

### 8.2 Memory Layer

| Project | Stars | Description |
|---------|-------|-------------|
| **Mem0** (mem0ai/mem0) | 48.0K | Universal memory layer (graph + vector) |
| **SimpleMem** (aiming-lab/SimpleMem) | 3.0K | Efficient lifelong memory via semantic compression |
| **A-MEM** (agiresearch/A-mem) | 850 | NeurIPS 2025: agentic Zettelkasten memory |

### 8.3 Multi-Agent Frameworks

| Project | Stars | Description |
|---------|-------|-------------|
| **MetaGPT** | 64.5K | SOP-driven multi-agent software company |
| **AutoGen** (microsoft/autogen) | 54.8K | Multi-agent conversations framework |
| **CrewAI** | 44.6K | Role-based agent orchestration |
| **LangGraph** (langchain-ai/langgraph) | 25.1K | Stateful agent graphs |
| **AFlow** (FoundationAgents/AFlow) | 436 | ICLR 2025 Oral: MCTS workflow generation |

### 8.4 Tool Infrastructure

| Project | Stars | Description |
|---------|-------|-------------|
| **LangChain** | 127.4K | De facto standard library for agents |
| **MCP Servers** | 79.4K | Standardized tool access protocol |
| **MCP Python SDK** | 21.8K | Build MCP-compatible tools |

### 8.5 OS Concept Mapping

| OS Concept | Agent OS Equivalent | Key Projects |
|-----------|---------------------|-------------|
| Kernel | Agent governance, scheduling | AIOS, Agent OS Kernel |
| Virtual Memory / Paging | Context window management | Letta/MemGPT, MemOS |
| RAM / Heap | Working context window | Context engineering frameworks |
| Disk / Persistent Storage | Long-term agent memory | Mem0, SimpleMem |
| Process Scheduler | Agent orchestration | LangGraph, MetaGPT, AutoGen |
| IPC | Multi-agent messaging | CrewAI, ChatDev/MacNet |
| System Calls | Tool invocation protocol | MCP (Model Context Protocol) |
| Device Drivers | MCP server implementations | MCP Servers |
| Standard Library | Common agent utilities | LangChain |
| Compiler | Workflow optimizer | AFlow |

---

## 9. Gap Analysis & Research Opportunities

### 9.1 Open Problems

**Gap 1: 无统一 Agent OS 实现**
AIOS, MemGPT, MemoryOS, AFlow 各自独立实现 OS 的不同层。没有系统同时整合 scheduling + context engineering + memory management + workflow optimization。

**Gap 2: Context Engineering × Memory 交互未被研究**
ACE 优化 playbook，ACON 压缩 context，MemoryOS 管理 memory，但三者从未在同一系统中测试。

**Gap 3: 科学推理 Benchmarks 缺少 Agent OS 方法测试**
ACE、MemoryOS、AFlow 主要在 coding (SWE-bench)、web (WebArena)、通用 QA 上验证，**未覆盖 ScienceAgentBench、BixBench、PaperBench 等科学推理基准**。

**Gap 4: Memory 系统缺乏科学知识特化**
现有记忆系统为通用对话设计。科学推理需要 structured knowledge（公式、实验参数、统计结果）的专用记忆管理。

**Gap 5: Workflow Optimization 未覆盖科学任务**
AFlow 的 MCTS 方法在 HumanEval、GSM8K 上验证，未在多步实验、数据分析、论文复现等科学任务上测试。

**Gap 6: Multi-Agent 长时间运行退化**
MacNet 的 logistic scaling law 在 ~32 agents 饱和。长时间多 agent 科学任务的退化模式未被建模。

**Gap 7: 评估框架不统一**
没有统一的 "Agent OS Performance Card" 同时评估 task accuracy × context efficiency × memory utilization × coordination overhead × inference cost。

### 9.2 Most Promising Research Directions

**Direction 1: Context Engineering for Scientific Agents ★★★★★**
- **Why**: ACE (+17pp) 已证明 context engineering 潜力巨大，但尚未应用于科学推理。长上下文 + 工具调用的交叉是最大 gap。
- **Approach**: 将 ACE 的 evolving playbook 机制应用于 ScienceAgentBench 或 PaperBench，让 agent 在科学实验过程中自动积累和优化领域 context。

**Direction 2: Memory-Augmented Scientific Reasoning ★★★★★**
- **Why**: BixBench 17%, PaperBench 26.6%。跨步骤、跨文档的知识管理是核心瓶颈。MemoryOS (+49% F1) 证明 OS-inspired memory 有巨大提升空间。
- **Approach**: 设计 ScientificMemoryOS — 融合 MemoryOS 的层级结构 + A-MEM 的自组织能力，专为科学推理定制。

**Direction 3: Unified Agent OS Benchmark ★★★★**
- **Why**: 没有 benchmark 同时评估 context engineering + memory + coordination + efficiency。
- **Approach**: 构建统一评估框架，在 ScienceAgentBench 上同时测量所有 Agent OS 维度。

**Direction 4: Workflow Optimization for Multi-Step Tool Use ★★★★**
- **Why**: 多步工具调用 66-81% 失败是 "Value Not Yet Known" 类型。AFlow 的 MCTS 方法可以自动发现避免状态追踪错误的 workflow。
- **Approach**: 将 AFlow 扩展到 FuncBenchGen/TRAJECT-Bench 类型的多步工具调用任务。

**Direction 5: Adaptive Resource Allocation for Agent Tasks ★★★**
- **Why**: Test-Time Compute (ICLR 2025 Oral) 证明自适应分配可节省 2-4x 计算。
- **Approach**: 设计 task-aware resource allocator，动态分配 context window, token budget, tool calls。

---

## 10. Conclusions & Recommendations

### 10.1 Key Takeaways

1. **Agent OS 范式已获广泛验证**: 从 MemGPT (ICLR 2024) 到 ACE (ICLR 2026)，2 年内产生了 85+ 相关论文，涵盖 6 个顶级会议。

2. **Scaffold > Model 已被反复证明**: MLE-bench 的 11x gap、AFlow 的 4.55% 成本、ACE 的 +17pp 均表明系统设计的影响力可超越模型本身。

3. **技术已走向融合**: Context engineering + memory management + multi-agent coordination 正在融合为统一的 Agent OS 栈。

4. **科学推理是最大的蓝海**: 几乎所有 Agent OS 技术都在 coding/web/QA 上验证，科学推理基准 (ScienceAgentBench, BixBench, PaperBench) 尚未被系统性地应用这些技术。

5. **开源生态已成熟**: 从 LangChain (127K stars) 到 MCP (79K stars) 到 Mem0 (48K stars)，Agent OS 的各层都有生产级开源实现。

### 10.2 Recommended Research Path

对于关注科学推理 + 工具调用 + 长上下文任务的研究者，推荐的优先级排序：

1. **首选**: **Context Engineering for Scientific Agents** — 将 ACE 的 evolving playbook + PAACE 的 plan-aware 压缩应用到科学任务。Impact 最高、技术路径最清晰、且竞争最小。

2. **其次**: **Scientific Memory OS** — 设计针对科学知识的记忆层级（公式记忆、实验状态记忆、领域知识记忆）。MemoryOS + A-MEM 的融合是强基线。

3. **同步推进**: **Workflow Optimization for Science** — 用 AFlow 的 MCTS 方法自动搜索最优的科学实验 workflow，特别是多工具调用场景。

4. **基础设施**: 基于 AIOS + MCP 构建实验平台，利用 LangGraph 做 workflow 管理。

### 10.3 Core Conclusion

> **AFlow 证明 GPT-4o-mini + 优化的 workflow 可以匹配 GPT-4o 的性能，仅需 4.55% 的成本；MLE-bench 证明同一 GPT-4o 在不同 scaffold 下有 11 倍性能差距。这两个结果共同表明：对于 agent 任务，系统架构设计的影响力可以超越模型能力本身的 10 倍以上差异。**

---

## References

See `references.bib` for the complete BibTeX bibliography (85 entries).

### Key Papers (Deep-Read, 12 papers)

1. **ACE: Agentic Context Engineering** — Zhang et al., ICLR 2026
2. **ACON: Context Compression for Long-horizon Agents** — Preprint, 2025
3. **PAACE: Plan-Aware Agent Context Engineering** — Preprint (→ICML), 2025
4. **MemGPT: Towards LLMs as Operating Systems** — Packer et al., ICLR 2024
5. **Memory OS of AI Agent** — Kang et al., EMNLP 2025 Oral
6. **A-MEM: Agentic Memory** — Xu et al., NeurIPS 2025
7. **AIOS: LLM Agent Operating System** — Mei et al., COLM 2025
8. **AFlow: Automating Agentic Workflow Generation** — Zhang et al., ICLR 2025 Oral
9. **MacNet: Scaling Multi-Agent Collaboration** — Chen et al., ICLR 2025
10. **Scaling LLM Test-Time Compute Optimally** — Snell et al., ICLR 2025 Oral
11. **RocketKV: KV Cache Compression** — NVIDIA, ICML 2025
12. **MAS-GPT: Training LLMs to Build MAS** — Chen et al., ICML 2025

### Supplementary Papers (Surveyed)

- **Context Engineering Survey** (arXiv:2507.13334) (preprint) — 1400+ papers surveyed
- **Memory in the Age of AI Agents Survey** (arXiv:2512.13564) (preprint) — 46+ co-authors
- **Agent Interoperability Protocols Survey** (arXiv:2505.02279) (preprint) — MCP, A2A, ACP comparison
- **Stop Overthinking** (TMLR 2025) — Token budget management
- **MLE-bench** — ICLR 2025 Oral; scaffold = 11x performance gap
- **MCP** (Anthropic → Linux Foundation) — Agent tool protocol standard
- **A2A** (Google → Linux Foundation) — Agent-to-agent protocol standard
- **AgeMem** (preprint) — RL-learned memory management
- **Mem0** (preprint) — Production-grade memory with 48K GitHub stars
- **MemOS** (preprint) — Memory as first-class compute resource
- **AriGraph** (IJCAI 2025) — KG world model with episodic memory
- **CAM** (NeurIPS 2025) — Piaget-inspired constructivist memory
- **Sketch-of-Thought** (EMNLP 2025) — 84% token reduction
- **AGoT** (preprint) — Adaptive Graph of Thoughts, +46.2% on GPQA

---

*Generated 2026-02-25. Based on systematic review of 85 papers (76 peer-reviewed, 9 preprints) and 28 GitHub repositories.*
