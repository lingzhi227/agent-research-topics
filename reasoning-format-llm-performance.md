# 推理格式（Reasoning Format）对 LLM 推理性能的影响

## 深度研究报告

> 基于 844 篇论文调研（302 篇顶会）、12 篇论文深度阅读、15 个代码工具分析。
> 日期：2026-02-26

---

## 一、核心发现：格式是第一性变量

推理格式不是次要因素——它是**与模型规模同等重要的第一性变量**。在同一任务上，格式变化可以引起 20-90+ 个百分点的性能差异。

### 关键量化证据

| 实验 | 基准 | 格式对比 | 性能差 |
|------|------|----------|--------|
| JSON vs 自然语言 | GSM8K | 结构化输出约束 | **-27 pp** [@tam2024speak] |
| 格式效应量 | 多任务 | Cohen's d | **1.58** (远超领域效应 d<0.2) [@cot_encyclopedia] |
| 网络 vs 链 | 排序32数 | XNoT vs CoT | **92% vs 0%** [@knot] |
| 代码/文本自适应 | SymBench 37任务 | CodeSteer vs 基线 | **+33 pp** [@codesteer] |
| 潜在 vs 文本 | ProsQA | Coconut vs CoT | **+19.5 pp** [@coconut] |
| 多格式混合 | FOLIO/ProofWriter | MoT vs 单格式 | **+11.7 pp** [@mot] |
| 结构路由 | Loong 200K+ | StructRAG vs Long-Context | **+23 pp** [@structrag] |
| CoT vs No-CoT | 20+数据集（数学） | 链式思维 vs 直接回答 | **+12-14%** [@to_cot] |
| CoT vs No-CoT | 20+数据集（常识） | 链式思维 vs 直接回答 | **<1%** [@to_cot] |

---

## 二、格式-任务交互矩阵

### 没有单一最优格式

最重要的发现是：**最优格式取决于任务类型**。

| 任务类型 | 最佳格式 | 为什么 | Benchmark 证据 |
|----------|---------|--------|----------------|
| 数学/算术 | **代码 (PoT)** | 精确计算，消除算术错误 | GSM8K +12%, CodeSteer SymBench |
| 符号逻辑 | **代码 + 真值表** | 穷举所有逻辑情况 | FOLIO +11.7pp (MoT), ProofWriter |
| 多跳推理 | **图/DAG** | 非线性依赖追踪 | GPQA Diamond +46.2% (AGoT) |
| 规划/约束 | **表格/矩阵** | 系统性约束覆盖 | Table-as-Thought 规划任务 |
| 排序/搜索 | **网络** | 任意拓扑，支持回溯 | Sort-32 92% vs 0% (XNoT) |
| 知识密集 | **自适应混合** | 任务相关结构化 | Loong +23pp (StructRAG) |
| 常识推理 | **自然语言** | 灵活，上下文相关 | 20+ datasets <1% CoT gain |
| 指令跟随 | **无CoT/选择性** | CoT 会分散注意力 | IFEval, ComplexBench (When Thinking Fails) |
| 效率敏感 | **潜在空间** | ~80% 时间减少 | ProsQA 97.0% (Coconut) |

### 为什么结构化有时更好，有时更脆？

用户的直觉完全正确：

**结构化更好的时候**：
- 需要精确计算 → 代码可以执行验证
- 需要穷举搜索 → 图/树支持回溯
- 需要覆盖所有约束 → 表格强制系统化

**结构化更脆的时候**：
- 一个环节出错 → 链式级联崩溃（Intrinsic Stability Limits, 2026）
- 格式序列化不同 → 同一图，不同节点排序，答案不同（Lost in Serialization）
- 添加无关信息 → 高达 -65% 性能（GSM-Symbolic, ICLR 2025）

---

## 三、自适应格式选择：性能最高的方法

### 所有顶级系统都使用自适应格式选择

| 系统 | 选择机制 | 性能提升 | 发表 |
|------|---------|---------|------|
| **StructRAG** | DPO 训练的 7B 路由器 | 仅路由器就贡献 15 pp | ICLR 2025 |
| **CodeSteer** | 8B 转向模型 | 53.3→86.4 (GPT-4o) | ICML 2025 |
| **MoT** | 并行 + 多数投票 | +11.7 pp | ICLR 2026 |
| **AGoT** | 免训练自适应 DAG | +46.2% | arXiv 2025 |
| **AutoForm** | LLM 自选格式 prompt | +3.3-5.7% | EMNLP 2024 |
| **DOTS** | 内化轨迹搜索 | > 静态 prompting | ICLR 2025 |
| **TypedThinker** | 类型化思维分类 | 推理多样化 | ICLR 2025 |

**关键架构模式**：格式选择器通常是**小模型**（7-8B）引导**大模型**（GPT-4o, 70B+）。这在计算上非常高效——选择器开销极低，但解锁巨大性能提升。

### StructRAG 的认知负荷理论验证

StructRAG 基于认知负荷理论（Cognitive Load Theory）：人类根据任务需求将原始信息转换为不同结构：
- 统计比较 → 表格
- 关系追踪 → 图
- 顺序过程 → 算法
- 层级组织 → 目录

消融实验：
- 仅用表格：49.66
- 仅用图：22.71
- 仅用文本块：53.92
- **自适应混合：69.43**（+16-47 pp 超过最佳固定格式）

---

## 四、推理拓扑：链 → 树 → 图 → 网络

### 线性链（CoT）在需要搜索的任务上严重受限

```
Chain (CoT):  A → B → C → D → Answer
              ↑ 无法回头，错误级联

Tree (ToT):   A → B → C
              └→ B'→ C'
              ↑ 可探索替代路径

Graph (GoT):  A → B → C
              ↓   ↑
              D → E
              ↑ 可循环、合并

Network (XNoT): 任意拓扑
              ↑ 最灵活，排序32数: 92% vs CoT 0%
```

### 具体 Benchmark 结果

| 方法 | GPQA Diamond | Sort-32 | Game of 24 |
|------|-------------|---------|------------|
| Direct I/O | 低 | 极低 | 12.5% |
| CoT | 中 | **0%** | ~20% |
| ToT | 中+ | 12% | ~45% |
| GoT | 中+ | 中 | ~40% |
| **AGoT** | **+46.2%** | - | **50%** |
| **XNoT** | - | **92%** | - |
| **PaCoRe** | - | - | **94.5% HMMT** |

PaCoRe (2026) 的并行协调推理在 HMMT 2025 数学竞赛上达到 **94.5%**，超过 GPT-5 的 93.3%——通过消息传递架构将推理计算扩展到每题近 200 万 token。

---

## 五、代码 vs 自然语言推理

### 代码在符号任务上有明确优势

CodeSteer (ICML 2025) 的关键发现：
- GPT-4o 基线: 53.3
- + CodeSteer 代码/文本转向: **86.4**（超过 o1 的 82.7）
- 符号任务（数字乘法、数独、方块堆叠）: 准确率提升 **>30%**

但**代码并非万能**：
- 常识推理：代码无优势
- 模糊/上下文依赖任务：自然语言更灵活
- 代码生成本身有语法错误风险

### 最优方案：混合 + 自适应切换

"Code to Think, Think to Code" 综述 (EMNLP 2025) 总结：
- 代码训练提升数学推理 → 推理训练提升代码能力 → **双向增益**
- 混合方法（代码+文本+总结）在所有方法中表现**最佳**

---

## 六、推理过多反而有害

### "过度思考"问题

三条独立证据线：

1. **When Thinking Fails** (NeurIPS 2025 Spotlight): CoT 在 15 个模型上系统性降低指令跟随性能。机制：推理"分散了"对约束 token 的注意力。

2. **Overthinking** (121 引用): 在 agent 任务中，过度推理延迟行动，甚至导致 agent "说服自己"放弃正确行动。

3. **推理轨迹分析** (2601.23163): 反直觉发现——**更短的推理链 ↔ 更高准确率**。延长推理主要是"难度反应性"的，而非内在有益。

### "怀疑 token" 的发现

CoT Robustness (ICLR 2026) 揭示：
- "wait"、"let me check"、"hmm" 等 token 是**关键恢复机制**
- 它们是互信息峰值（NeurIPS 2025）
- 释义（paraphrase）去除这些 token → **独特地**降低准确率
- 噪声注入反而**触发**更多怀疑 token → 促进恢复

---

## 七、潜在空间推理：超越文本

### Coconut (ICLR 2025): 完全无文本推理

- 使用 LLM 最后一层隐藏状态作为"连续思维"
- ProsQA: **97.0%** vs CoT 77.5%（+19.5 pp）
- 涌现**广度优先搜索** (BFS) — 在顺序文本中不可能
- 推理时间减少 **~80%**

### 局限
- 数学推理仍然落后于文本 CoT
- 隐藏推理不可监控 → 安全隐患
- 需要特殊训练流程（课程学习）

### 趋势：混合潜在-文本推理
- System-1.5 Reasoning (NeurIPS 2025): 语言+潜在空间动态切换
- Token Assorted (ICML 2025): 混合潜在和文本 token
- Hybrid Latent Reasoning (NeurIPS 2025): RL 优化的混合方案

---

## 八、实用工具生态

### 生产就绪工具

| 工具 | Stars | 用途 | 关键特性 |
|------|-------|------|---------|
| **vLLM** | 55K+ | 推理服务 | `--reasoning-parser` 支持 DeepSeek/Qwen3/GLM |
| **SGLang** | 20K+ | 推理服务 | 思考阶段免语法约束，输出阶段强制格式 |
| **DSPy** | 25K+ | 编程框架 | ChainOfThought / ProgramOfThought 模块 |
| **LangGraph** | 15K+ | Agent 框架 | 图基 agent 推理 + "think" 工具 |

### 研究级工具

| 工具 | Stars | 论文 | 功能 |
|------|-------|------|------|
| **Graph of Thoughts** | 2.6K | ETH Zurich | `pip install` 可用的图推理库 |
| **CodeSteer** | 88 | ICML 2025 | 代码/文本转向模型 |
| **StructRAG** | 158 | ICLR 2025 | 推理时结构化 |
| **Coconut** | 1.5K | Meta/ICLR 2025 | 潜在空间推理 |
| **LightThinker** | 133 | EMNLP 2025 | gist token 压缩，-70% token |
| **AGoT** | 128 | arXiv | 自适应图推理 |

### vLLM "先想后格式化" 实现

```python
# vLLM 原生支持 "reason freely, format later"
vllm serve deepseek-ai/DeepSeek-R1 \
  --reasoning-parser deepseek_r1

# Python 客户端
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": question}]
)
reasoning = response.choices[0].message.reasoning  # 自由推理
content = response.choices[0].message.content       # 格式化输出
```

---

## 九、直接回答用户问题

### Q1: 结构化格式（数学公式、图、链、linked-list）是否比自然语言在某些任务上表现好？

**是的，有大量 benchmark 证据。** 具体来说：
- 代码格式在数学上 +12%（To CoT or not, ICLR 2025）
- 图格式在科学推理上 +46.2%（AGoT, GPQA Diamond）
- 表格格式在约束规划上优于链（Table as Thought）
- 网络格式在排序上 92% vs 0%（XNoT）
- 真值表格式在逻辑推理上 +11.7 pp（MoT, ICLR 2026）

### Q2: 结构化更容易崩，一步错全错？

**是的，这也有证据。** 两面性：
- 链式级联：自回归推理的内在稳定性极限（2602.06413）
- 添加一个无关子句 → -65%（GSM-Symbolic, ICLR 2025）
- 图序列化不同 → 答案不同（Lost in Serialization）
- 释义去除"怀疑 token" → 独特地降低准确率（ICLR 2026）

**但自然语言也有其脆弱性**：语义漂移、记忆而非推理（Surface-Form Brittleness）。**没有格式是完全鲁棒的。**

### Q3: 对 agentic reasoning、task、action 能力的影响？

- **推理过多会损害行动能力**（Overthinking, 121 引用）
- **格式奖励与准确率奖励同等重要**（DeepSeek-R1, Logic-RL）
- **选择性推理**是最佳策略：简单任务不推理，复杂任务深度推理（When Thinking Fails）
- **结构化 agent 推理**（Satori 的 Chain-of-Action-Thought, ACL 2025 的 Agentic Reasoning）正在兴起

### Q4: 应该怎样组织代码上下文？

基于 StructRAG 的认知负荷理论：
1. **不要大片塞入 context**——按任务需求结构化
2. **分类存储**是正确方向（用户直觉正确）
3. **推理时路由**：根据当前问题类型选择合适的代码表示
4. **具体建议**：
   - 函数签名 + 文档 → 表格格式
   - 调用关系 → 图格式
   - 执行流程 → 算法/伪代码格式
   - 具体实现 → 原始代码文本

---

## 十、推理格式分类体系

```
推理格式分类
├── 文本类
│   ├── 自然语言 (NL)
│   │   ├── 自由 CoT（标准）
│   │   ├── 结构化 NL（编号列表、要点）
│   │   └── Sketch-of-Thought（压缩 NL，-84% token）
│   ├── 代码
│   │   ├── Program-of-Thought (PoT)
│   │   ├── Code I/O (CodeIO, ICML 2025 Oral)
│   │   └── 混合 code+text (CoRT)
│   └── 符号
│       ├── 逻辑表达式（一阶逻辑、命题逻辑）
│       ├── 真值表（MoT, ICLR 2026）
│       └── 数学公式（LaTeX）
├── 结构类
│   ├── 表格 (Table-as-Thought)
│   ├── 图 (Graph-of-Thought, GoT)
│   ├── 树 (Tree-of-Thought, ToT)
│   ├── DAG (DAG-Math, ICLR 2026)
│   ├── 网络 (XNoT, 任意拓扑)
│   └── 矩阵 (Matrix-as-Plan)
├── 潜在类
│   ├── 连续潜在 (Coconut, ICLR 2025)
│   ├── 压缩 (LightThinker, gist token)
│   └── 激活空间 (ActivationReasoning, ICLR 2026)
└── 混合/自适应
    ├── AutoForm（自选格式）
    ├── CodeSteer（外部转向）
    ├── StructRAG（任务路由）
    └── MoT（并行多格式）
```

---

## 参考文献

见 `references.bib`。
