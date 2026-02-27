# 从专业文档到增强推理：LLM知识蒸馏、自动提示生成与领域推理综述

# From Professional Documents to Enhanced Reasoning: A Survey on LLM Knowledge Distillation, Automatic Prompt Generation, and Domain-Specific Reasoning

## 1. 引言 (Introduction)

大语言模型(LLMs)在通用任务上表现出色，但在数学、生物、基因组学、信息学、计算机科学算法和统计学等专业领域，其推理质量仍存在明显不足。专业教科书和论文包含了大量结构化知识——从定义、定理、证明到练习，具有天然的层级结构和难度递进。然而，一个核心矛盾是：虽然当前LLM的上下文窗口已扩展至256K-1M tokens，但有效召回窗口仅约10K tokens [@liu2024lost]，远不足以处理教科书级别的长文档。

本综述聚焦于一个重要且前沿的研究方向：**如何让LLM通过阅读专业文档（如MD格式的教科书），自动生成优化的提示(prompts)、技能模块(skills)和推理格式(reasoning formats)，从而提升专业任务上的推理质量**。我们系统梳理了2022-2026年间的70篇顶会论文，涵盖自动提示优化(APO)、推理知识蒸馏(KD)、长上下文与检索增强(RAG)、以及领域特定推理四大方向。

## 2. 背景 (Background)

### 2.1 问题定义

给定一组专业文档 $\mathcal{D} = \{d_1, d_2, ..., d_n\}$（如教科书的Markdown格式），目标是自动生成：
- **优化提示** $P^* = \text{APO}(\mathcal{D}, \mathcal{T})$：针对领域任务 $\mathcal{T}$ 的最优指令
- **技能模块** $S = \{s_1, ..., s_k\}$：可复用的领域推理模块
- **推理格式** $R$：领域特定的推理链模板（如"假设→机制→证据→预测"）

### 2.2 核心挑战

**长上下文的有效性悖论**：Liu等人在"Lost in the Middle" [@liu2024lost] (TACL 2024) 中证明，LLM对长上下文中间位置信息的召回率显著下降，呈U形注意力模式。这一发现被多项后续研究证实：即使在100K+ token窗口下，模型性能仍随上下文长度增加而下降 [@hsieh2024ruler]。

**提示脆弱性**：Yang等人 [@yang2024opro] (ICLR 2024) 发现，语义相似的提示可导致截然不同的性能——"Let's think step by step"在GSM8K上达71.8%，而"Let's work together to solve this problem step by step"仅49.4%。这凸显了自动提示优化的必要性。

## 3. 方法分类 (Taxonomy of Approaches)

### 3.1 自动提示优化 (Automatic Prompt Optimization)

自动提示优化旨在用算法替代手工提示工程。目前主要有五种范式：

**元提示 (Meta-Prompting)**：OPRO [@yang2024opro] (ICLR 2024) 提出让LLM自身作为优化器。通过维护一个包含历史提示-分数对的元提示(meta-prompt)，LLM迭代生成新的候选提示。在GSM8K上，OPRO发现的"Take a deep breath and work on this problem step-by-step"达到80.2%，比人工提示提高8个百分点。OPRO的优势在于利用完整优化轨迹，但发现的提示是通用推理触发器，而非领域知识丰富的指令。

**程序化优化 (Programmatic Optimization)**：DSPy [@khattab2024dspy] (ICLR 2024) 提出"编程而非提示"的范式。MIPROv2优化器通过三阶段工作：(1) 引导(Bootstrapping)——运行程序收集成功轨迹作为示例；(2) 基础提案(Grounded Proposal)——利用数据摘要、程序代码和引导示例生成指令候选；(3) 贝叶斯优化——使用TPE采样器搜索最优指令-示例组合。DSPy的关键创新在于领域知识通过**程序结构**而非提示字符串进入系统。

**进化搜索 (Evolutionary Search)**：EvoPrompt [@guo2024evoprompt] (ICLR 2024) 将进化算法应用于离散提示优化，在BBH任务上比人工提示提升高达25%。PromptBreeder [@fernando2024promptbreeder] 则实现自引用的自我改进——提示进化的同时，进化策略本身也在进化。

**文本梯度 (Textual Gradients)**：TextGrad [@yuksekgonul2024textgrad] (Nature 2024) 类比反向传播，使用LLM生成的文本反馈作为"梯度"来优化提示和其他文本变量。

**多臂老虎机策略选择**：Bandit-Based Prompt Design [@huang2025bandit] (ACL 2025) 使用多臂老虎机算法在不同提示策略间动态选择。Tree-of-Prompts [@chen2025tree] (ACL 2025) 将控制流抽象引入提示优化，支持层级化的提示结构。

### 3.2 推理知识蒸馏 (Reasoning Knowledge Distillation)

知识蒸馏旨在将大模型的推理能力转移到小模型。近年来的创新集中在**如何蒸馏**而非仅仅蒸馏什么。

**关键步骤识别与渐进蒸馏**：KPOD [@wang2024kpod] (ICML 2024) 发现推理链中并非所有token同等重要。通过掩码学习模块识别"关键点"token，并按从易到难的顺序进行渐进式蒸馏。在GSM8K等基准上比SOTA提高5-6%，在CommonsenseQA上提高高达12.8%。

**教育学启发的蒸馏 (IOA)** [@chen2026ioa] (ICLR 2026) 是与我们目标最直接相关的工作。该框架将Bloom的掌握学习理论和Vygotsky的最近发展区理论应用于LLM知识蒸馏：
- **Identifier**：通过性能差距量化 $\Delta(k) = [P_T(k) - P_S(k)]/P_T(k)$ 诊断知识缺陷，构建先决条件DAG
- **Organizer**：设计渐进式课程，约束难度增幅 $\leq \tau_{ZPD} \times$ 当前阶段难度，掌握度达0.9才允许进阶
- **Adapter**：通过五种策略适配知识表示——概念具象化、推理分解、认知负荷管理、表示格式优化、语言简化

IOA使学生模型在DollyEval上保留教师94.7%的性能，在MATH上提升19.2%，在HumanEval上提升22.3%。

**Agent蒸馏** [@pan2025agent] (NeurIPS 2025 Spotlight) 超越静态CoT蒸馏，将**工具使用的智能体行为**蒸馏到小模型中。通过First-Thought Prefix (FTP)和Self-Consistent Action Generation (SAG)，0.5B的agent模型可匹配1.5B的CoT模型，3B的agent可超越7B的CoT模型。这种方法教会模型**如何使用知识源**而非记忆知识——与从教科书中检索并推理的模式天然匹配。

**自学推理 (STaR)** [@zelikman2022star] (NeurIPS 2022) 提出了一个优雅的自举循环：生成推理链→评估答案→对错误答案提供正确答案重新生成→在正确推理链上微调→重复。这一方法可应用于从教科书练习题中引导推理能力的提升。

### 3.3 长上下文与检索增强策略

**长上下文强化学习**：LoongRL [@zheng2026loongrl] (ICLR 2026 Oral) 是该领域的突破性工作。通过KeyChain数据合成（在扩展上下文中插入UUID链条要求模型追踪）和GRPO训练，LoongRL-14B在LongBench v1上达到74.2分，接近o3-mini(74.5)和DeepSeek-R1(74.9)。最关键的发现是：**仅在16K token上训练，即可有效泛化到128K token**。同时，模型自发涌现出"计划-检索-推理-复查"的结构化推理模式。相比之下，R1蒸馏模型在长上下文任务上反而退化了17.7个百分点——表明朴素蒸馏会损害长上下文能力。

**自适应检索**：DeepRAG [@li2026deeprag] (ICLR 2026) 将检索决策建模为马尔可夫决策过程，通过二叉树搜索找到最优检索策略。DeepRAG仅使用0.28-1.09次平均检索（对比AutoRAG的4.52-6.26），同时准确率提高21.99%。其知识边界校准(MCC=0.451)远超先前方法——模型能准确判断何时需要查阅外部知识、何时可以依靠自身知识推理。

**层级化文档处理**：HiRAG [@huang2025hirag] (EMNLP 2025 Findings) 和TreeRAG [@liu2025treerag] (ACL 2025) 提出保留文档层级结构的检索方法，对教科书的章节-小节-段落结构特别有效。Anthropic的Contextual Retrieval (2024) 则通过为每个文本块添加上下文信息来提高检索准确性。

### 3.4 领域特定科学推理

**科学验证**：SCI-Verifier [@wang2026sciverifier] (ICLR 2026) 针对科学答案验证问题（如等价数学表达式的识别），通过SFT+DAPO两阶段训练，8B模型在SCI-VerifyBench上达86.28%，超过GPT-5(84.92%)。这证明了在专业领域，精心训练的小模型可以超越通用大模型。

**领域适应预训练**：OmniScience (preprint) 通过科学文献的领域适应预训练+推理蒸馏，构建通用科学推理模型。rbio-1 (preprint) 则使用生物世界模型作为验证器，通过强化学习训练生物推理模型。

**技能模块架构**：Agent Skills框架 (2025) 将过程性专长打包为可组合、可移植、可动态加载的模块。技能可以简单到一个包含Markdown指令的文件，也可以包含脚本和工具。这一抽象解决了通用模型与专业任务需求之间的张力。

## 4. 深度分析 (Detailed Analysis)

### 4.1 从文档到提示的关键技术挑战

| 挑战 | 当前解决方案 | 差距 |
|------|-------------|------|
| 长文档超出上下文窗口 | HiRAG层级分块、LoongRL长上下文RL | 无法处理完整教科书(>128K) |
| 提示优化需要标注数据 | DSPy自动引导、OPRO元提示 | 未能利用文档结构信息 |
| 知识层级构建 | IOA手动定义 | 需要自动化从目录/章节提取 |
| 领域推理格式多样 | KPOD关键步骤提取 | 缺乏领域特定格式发现 |
| 验证领域答案正确性 | SCI-Verifier | 仅覆盖验证，缺乏生成端优化 |

### 4.2 方法间的互补性分析

我们的关键发现是：**现有方法分别解决了管道中的不同阶段，但没有端到端的集成**。

- **OPRO + DSPy** 优化提示，但不处理文档
- **HiRAG + TreeRAG** 处理文档，但不生成提示
- **IOA** 应用教育学原理，但需要手工定义知识层级
- **Agent Distillation** 教会工具使用，但不从文档提取知识
- **LoongRL** 处理长上下文推理，但训练数据不来自教科书
- **SCI-Verifier** 验证领域答案，但不增强推理生成

将这些组件集成为一个管道——**从PDF到增强推理**——是一个重要的研究机会。

### 4.3 推荐的集成架构

```
阶段1: 文档处理
  MinerU (PDF→MD) → HiRAG (层级分块) → 知识图谱构建 → IOA式先决条件DAG

阶段2: 自动提示/技能生成
  DSPy管道定义 → MIPROv2 (从教科书示例引导) → OPRO (优化领域指令)

阶段3: 推理格式生成
  KPOD (提取关键推理步骤) → IOA (教育学课程设计) → STaR (自举推理)

阶段4: 领域模型增强
  Agent蒸馏 (训练工具使用) → DeepRAG (自适应教科书检索) → SCI-Verifier (验证)
```

## 5. 应用场景 (Applications)

### 5.1 数学推理
- **现状**: 前沿推理模型(o1, R1)已在数学上表现优异
- **文档知识的价值**: 教科书的定理-证明结构、例题解法提供了自然的推理模板
- **推荐方法**: DSPy + KPOD从数学教科书提取渐进式推理步骤

### 5.2 生物/基因组学
- **现状**: OmniScience, rbio-1为领域适应模型提供基础
- **文档知识的价值**: 生物教科书包含分类体系、机制通路、实验方法论
- **推荐方法**: IOA式知识层级 + Agent蒸馏(检索基因数据库 + 代码分析)

### 5.3 计算机科学算法与编码
- **现状**: Agent Distillation在代码生成上表现出色(HumanEval提升22%)
- **文档知识的价值**: 算法教科书提供伪代码、复杂度分析、正确性证明
- **推荐方法**: Agent蒸馏 + 代码执行验证 + DSPy算法描述优化

### 5.4 统计学
- **现状**: 统计推理需要精确的数学推导和假设检验
- **文档知识的价值**: 统计教科书提供完整的假设-检验-结论推理链
- **推荐方法**: OPRO优化统计推理触发器 + SCI-Verifier式答案验证

## 6. 开放问题与未来方向 (Open Problems and Future Directions)

### 6.1 关键开放问题

1. **端到端管道**: 目前没有从PDF→知识→提示→推理增强的完整管道，各组件的集成仍是开放问题
2. **文档接地的提示优化**: 如何确保自动生成的提示忠实于源文档知识，而非产生泛化但不准确的指令？
3. **跨文档推理**: 如何从多本教科书中综合知识（如生物+统计→生物信息学）？
4. **评估指标**: 如何评估"文档衍生提示"的质量？现有指标（准确率）不足以衡量知识的忠实度和覆盖度
5. **可扩展性**: 如何处理持续更新的文档（如新版教科书、最新论文）？

### 6.2 最有前景的研究方向

**方向1: 文档接地的自动提示优化 (DG-APO)**
- 扩展OPRO/DSPy使其在优化循环中包含文档摘录
- 开发同时衡量准确性和文档忠实度的评分函数
- **可行性**: 高——构建在成熟的APO方法之上

**方向2: 从教科书结构自动构建知识课程**
- 利用LLM解析教科书MD为IOA式知识层级
- 从章节依赖关系自动提取先决条件DAG
- **可行性**: 高——教科书本身已提供此结构

**方向3: 领域推理格式发现**
- 使用STaR式自举结合领域特定验证器
- 发现领域推理模式（如"基因组学: 序列→比对→注释→功能"）
- **可行性**: 中——需要领域特定验证器

**方向4: 教科书规模的Agent训练**
- 结合LoongRL(长上下文) + DeepRAG(自适应检索) + Agent蒸馏(工具使用)
- 使用KeyChain式合成结合教科书内容训练
- **可行性**: 中——需要大量计算和领域数据

## 7. 结论 (Conclusion)

本综述系统梳理了从专业文档到LLM推理增强的研究前沿，涵盖70篇来自NeurIPS、ICLR、ICML、ACL等顶级会议的论文。我们识别了四个核心研究领域——自动提示优化、推理知识蒸馏、长上下文策略和领域特定推理——并揭示了一个关键洞察：**现有方法分别解决了管道中的不同阶段，但端到端的集成仍是重大开放问题**。

最令人振奋的近期进展包括：
- **DSPy/MIPROv2**: 将提示优化从手工工程转变为自动化程序化过程
- **IOA**: 将教育学理论成功应用于LLM知识蒸馏，学生模型保留94.7%教师性能
- **LoongRL**: 仅在16K训练即可泛化至128K的长上下文推理
- **Agent蒸馏**: 0.5B agent模型匹配1.5B CoT模型，证明工具使用比知识记忆更有效
- **SCI-Verifier**: 8B专业模型超越GPT-5，证明领域特化的价值

对于希望从教科书中自动提取知识以增强LLM推理的实践者，我们推荐的路径是：先用MinerU处理文档，再用HiRAG式层级分块保持结构，然后用DSPy自动优化领域管道，同时结合IOA的教育学原理设计渐进课程。这一组合方案利用了现有最成熟的技术，同时为未来的端到端集成奠定基础。

## References

见 `references.bib`（共70篇论文）

**按方向分类的核心参考文献：**

### 自动提示优化
- [@yang2024opro] Yang et al. "Large Language Models as Optimizers" ICLR 2024
- [@khattab2024dspy] Khattab et al. "DSPy: Compiling Declarative Language Model Calls" ICLR 2024
- [@guo2024evoprompt] Guo et al. "EvoPrompt: Connecting LLMs with Evolutionary Algorithms" ICLR 2024
- [@yuksekgonul2024textgrad] Yuksekgonul et al. "TextGrad: Automatic Differentiation via Text" Nature 2024
- [@pryzant2024ease] Pryzant et al. "Prompt Optimization with EASE" NeurIPS 2024

### 推理知识蒸馏
- [@wang2024kpod] Wang et al. "Keypoint-based Progressive CoT Distillation" ICML 2024
- [@chen2026ioa] Chen et al. "Pedagogically-Inspired Data Synthesis for KD" ICLR 2026
- [@pan2025agent] Pan et al. "Distilling LLM Agent into Small Models" NeurIPS 2025 Spotlight
- [@zelikman2022star] Zelikman et al. "STaR: Bootstrapping Reasoning With Reasoning" NeurIPS 2022
- [@li2025dakd] Li et al. "DA-KD: Difficulty-Aware Knowledge Distillation" ICML 2025

### 长上下文与检索
- [@zheng2026loongrl] Zheng et al. "LoongRL: RL for Long-Context Reasoning" ICLR 2026 Oral
- [@li2026deeprag] Li et al. "DeepRAG: Thinking to Retrieve Step by Step" ICLR 2026
- [@liu2024lost] Liu et al. "Lost in the Middle" TACL 2024
- [@huang2025hirag] Huang et al. "HiRAG: RAG with Hierarchical Knowledge" EMNLP 2025
- [@wu2025longcontext] Wu et al. "Long-Context LLMs Meet RAG" ICLR 2025

### 领域特定推理
- [@wang2026sciverifier] Wang et al. "SCI-Verifier: Scientific Verifier with Thinking" ICLR 2026
- [@zhang2025omniscience] Zhang et al. "OmniScience: Domain-Specialized LLM" (preprint)
- [@huang2025scillm] Huang et al. "Scientific Language Models Survey" EMNLP 2024
