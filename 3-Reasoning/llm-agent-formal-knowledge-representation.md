# 代码作为Agent的知识语言：LLM Agent形式化知识表示的系统性综述

# Code as the Language of Agent Knowledge: A Systematic Survey on Formal Knowledge Representation for LLM Agents

## 1. 引言 (Introduction)

大型语言模型 (LLM) Agent 正在迅速成为自主完成复杂任务的关键技术。然而，一个根本性问题尚未得到系统回答：**Agent 应该以什么形式表示和积累知识？**

当前主流的Agent系统（如ReAct [@yao2023react]、Reflexion [@shinn2023reflexion]）主要使用自然语言来记录Agent的推理过程、经验和知识。这种做法源于一个隐含假设：人类使用自然语言记录知识，因此Agent也应如此。但这个假设忽略了一个关键差异：**LLM Agent可以高效地处理代码和形式化语言，而人类主要依赖自然语言是因为人类阅读代码的速度远低于阅读自然语言。**

本综述系统性地调研了一个新兴的研究方向：**用代码/形式化语言/领域特定语言（DSL）来表示Agent的探索结果和积累的知识，而非自然语言**。我们发现这一方向在2023-2025年间经历了爆发式增长，已形成多个成熟的研究主线，并在ICML 2025上催生了专门的workshop（PRAL: Programmatic Representations for Agent Learning）。

本综述基于74篇论文的系统分析（65篇同行评审，9篇预印本），涵盖NeurIPS、ICML、ICLR、ACL、EMNLP等顶级会议2021-2026年的工作。

---

## 2. 背景：知识表示的人机差异 (Background: Human-Machine Knowledge Representation Gap)

### 2.1 人类的知识表示偏好

人类的知识记录方式受限于认知处理能力：
- **最优**：图表、实体模型（视觉空间推理能力强）
- **次优**：自然语言（语言处理速度约250词/分钟）
- **困难**：形式化语言、代码（阅读速度显著降低，需要专门训练）

因此，人类知识体系以自然语言为主，辅以数学公式和形式化表述。这是一种**受限于人类认知能力的知识表示方式**。

### 2.2 LLM Agent 的知识处理能力

LLM Agent 与人类有根本性差异：
- **代码/数学/DSL**：可以瞬间解析、执行和验证
- **自然语言**：虽然也能处理，但存在歧义、不可执行、不可验证
- **结构化数据**：JSON/XML/YAML 等可以高效处理

这种差异意味着：**为Agent设计的知识表示不应受人类阅读习惯的约束，而应利用代码/形式化语言的固有优势。**

### 2.3 代码作为知识表示的理论优势

代码相比自然语言具有五个根本性优势：
1. **可执行性 (Executability)**：可以被运行、测试、验证
2. **组合性 (Compositionality)**：可以组合、嵌套、抽象
3. **可压缩性 (Compressibility)**：可以重构为可复用组件
4. **无歧义性 (Unambiguity)**：具有形式化语义，无解释歧义
5. **可迁移性 (Transferability)**：可以编辑、版本控制、共享

---

## 3. 方法分类学 (Taxonomy of Approaches)

我们将现有工作分为五大类：

### 3.1 代码作为Agent动作空间 (Code as Agent Action Space)

**核心思想**：用可执行代码替代JSON或自然语言作为Agent的动作表示。

**CodeAct** [@wang2024codeact] (ICML 2024) 是这一方向的标志性工作。它提出用Python代码作为LLM Agent的统一动作空间，替代传统的JSON格式或自然语言指令。Agent生成Python代码并在解释器中执行，利用执行结果（包括错误信息）进行自我调试。

关键实验结果：
- 在复杂多工具任务上，代码动作比文本动作的成功率提高**20.7%**（GPT-4: 74.4% vs. 53.7%）
- 交互轮次减少**约30%**（5.5轮 vs. 7.7轮）
- 在17个模型中的12个上，代码是最优动作格式

**Code as Policies** [@liang2023cap] (ICRA 2023) 和 **ProgPrompt** [@singh2023progprompt] (ICRA 2023) 将这一思想应用于机器人控制，LLM生成可执行的Python策略代码来控制机器人。

### 3.2 代码作为Agent知识库 (Code as Agent Knowledge Store)

**核心思想**：Agent的积累知识以可执行代码的形式存储和检索。

**Voyager** [@wang2023voyager] (NeurIPS 2023) 首次提出了代码技能库的概念。在Minecraft环境中，Agent将学到的技能存储为JavaScript函数，通过自然语言描述的嵌入进行索引和检索。技能可以相互调用，形成层次化的能力积累。

关键结果：
- 获取独特物品数量比AutoGPT多**3.3倍**
- 技术树进展速度比AutoGPT快**15.3倍**
- 使用自然语言推理的基线（ReAct、Reflexion）在相同任务上**完全失败**

**WorldCoder** [@tang2024worldcoder] (NeurIPS 2024) 更进一步，Agent将整个世界模型构建为Python程序。关键创新包括：
- 用代码表示环境的转移函数和奖励函数
- 通过单元测试验证代码是否正确预测观测数据
- 学习后的世界模型可以作为模拟器用于规划，无需进一步LLM调用
- 相比深度RL，样本效率提升约**10,000倍**

**Code World Models** (ICLR 2026) 训练了一个32B参数的模型来理解代码执行，实现了比直接LLM调用快**4-7个数量级**的规划速度。

### 3.3 代码作为推理媒介 (Code as Reasoning Medium)

**核心思想**：用代码/程序作为推理的中间表示，而非自然语言思维链。

**PAL** [@gao2023pal] (ICML 2023) 和 **Program of Thoughts** [@chen2023pot] (TMLR 2023) 是奠基性工作。PAL让LLM将问题分解为Python程序，由解释器执行计算。关键洞察：**将分解（LLM擅长）与执行（代码擅长）分离**。

关键结果：
- 比Chain-of-Thought在GSM8K上提高**15%**（72.0% vs. 57.6%）
- 在13个基准测试中达到最优
- 关键消融实验：让LLM"模拟"代码执行（而非实际运行）只有23.2%准确率 vs. 实际执行的72.0%——**形式化执行是关键，不仅仅是代码格式**

**CodePlan** [@wen2025codeplan] (ICLR 2025) 将这一思想泛化为"代码形式的计划"。LLM生成伪代码计划来指导推理，然后基于计划生成自然语言回答。

关键结果：
- 在13个基准测试上平均提升**25.1%**
- 代码计划 vs. 自然语言计划：代码在所有类别都更优，最大差距在符号推理（+27.2%）和决策（+27.5%）
- **优势随任务复杂度增长**：2-hop问题提升6.3%，4-hop问题提升43.8%
- NLL分析显示代码计划比自然语言计划**更容易学习**（0.237 vs. 0.351）

"Code to Think, Think to Code" [@yang2025survey] (EMNLP 2025) 对这一方向进行了全面综述。

### 3.4 程序合成与库学习 (Program Synthesis and Library Learning)

**核心思想**：知识通过程序压缩和抽象来积累，形成可复用的程序库。

**DreamCoder** [@ellis2021dreamcoder] (PLDI 2021) 提出了wake-sleep贝叶斯程序学习，通过三个阶段循环增长程序库：
1. Wake：搜索解决问题的程序
2. Abstraction sleep：通过版本空间代数将解决方案压缩为可复用库组件
3. Dreaming sleep：用库生成的"幻想"任务训练神经模型

在8个领域的实验中，DreamCoder展示了程序作为知识压缩机制的力量：列表处理中的程序从32个函数调用压缩到5个。

**LILO** [@grand2024lilo] (ICML 2024) 结合了LLM与DreamCoder的思路：
- 用LLM引导程序合成
- 用Stitch算法进行高效压缩（比DreamCoder快1,000-10,000倍）
- 引入AutoDoc：自动为库函数生成自然语言文档

**LILO的关键发现对本综述特别重要**：去除AutoDoc后性能下降30.60分。这表明**最优的知识表示不是纯代码，而是代码+自然语言文档的混合体**。

**Parsel** [@zelikman2023parsel] (NeurIPS 2023) 提出了层次化程序分解，将任务描述为函数层级，然后搜索实现的组合。在竞赛编程问题上比直接采样提高**75%**。

### 3.5 形式化规划语言与神经符号方法 (Formal Planning Languages and Neuro-Symbolic Approaches)

**核心思想**：使用PDDL等形式化规划语言或逻辑程序来表示Agent知识。

[@neurips2024pddl] (NeurIPS 2024) 提出了通过环境交互自动生成PDDL描述的方法。[@iclr2025formplan] (ICLR 2025) 探索了基于LLM的形式化编程规划。

这些方法提供了更强的形式化保证，但灵活性不如可执行代码。新兴的趋势是**混合方法**，结合代码的灵活性与形式化语言的严格性（如Metagent-P, ACL 2025）。

---

## 4. 深度分析 (Deep Analysis)

### 4.1 代码 vs. 自然语言：量化证据

| 比较 | 代码表示 | 自然语言/替代 | 提升幅度 |
|------|---------|-------------|---------|
| WorldCoder vs. DreamerV3 | Python世界模型 | 神经网络世界模型 | 10,000x 样本效率 |
| CodeAct vs. 文本动作 | Python代码 | JSON/文本 | +20.7% 成功率 |
| Voyager vs. AutoGPT | JS技能库 | NL Agent | 15.3x 里程碑速度 |
| PAL vs. CoT | Python程序 | NL思维链 | +15% 准确率 |
| CodePlan vs. NL计划 | 伪代码 | NL计划 | +25.1% 平均提升 |
| CWM vs. LLM调用 | 代码执行轨迹 | 直接LLM | 10^4-10^7x 规划速度 |
| Parsel vs. 直接采样 | 层次化程序 | 直接代码生成 | +75% 通过率 |

### 4.2 为什么代码优于自然语言？机制分析

**1. 可执行性使验证成为可能**
WorldCoder通过单元测试验证世界模型，CodeAct通过错误信息自我调试，Voyager通过自验证确认技能正确性。自然语言描述无法自动验证。

**2. 组合性使知识积累具有倍增效应**
Voyager的技能可以相互调用（函数组合），DreamCoder的库组件可以嵌套使用。自然语言描述缺乏结构化的组合机制。

**3. 压缩性使知识更凝练**
DreamCoder将32个函数调用压缩为5个（6.4倍压缩）。LILO在REGEX任务上将26^5个可能性压缩为一个库函数。自然语言倾向于冗长和重复。

**4. 形式化语义消除歧义**
PAL的关键消融实验（23.2% vs. 72.0%）证明：代码的精确语义（通过解释器执行）远优于LLM对代码的"模拟理解"。自然语言固有地存在歧义。

**5. 预训练对齐**
LLM在大量代码语料上训练，对代码的生成和理解能力强于对形式化推理的自然语言描述能力（CodePlan的NLL分析：代码计划更容易学习）。

### 4.3 混合表示的必要性

LILO的AutoDoc发现揭示了一个重要规律：**最优表示不是纯代码，而是代码+自然语言注释**。这是因为：
- 自然语言提供**语义锚点**：帮助LLM理解代码的意图
- 自然语言提供**检索接口**：Voyager用NL描述的嵌入来检索代码技能
- 自然语言提供**人类可审计性**：WorldCoder强调代码的人类可读性

因此，最优的Agent知识表示可能是：**结构化知识用代码，元知识（意图、适用条件、来源）用自然语言注释**。

### 4.4 Agent架构视角：CoALA框架

CoALA [@sumers2024coala] (TMLR 2024) 提供了一个统一的认知架构框架，将Agent记忆分为：
- **工作记忆**：短期、基于提示
- **情景记忆**：经验轨迹
- **语义记忆**：事实、反思
- **程序性记忆**：代码 + 权重

关键洞察：**知识从情景→语义→程序性渐进形式化**。分析现有Agent系统，最强大的Agent（Voyager）正是将学到的知识存储为可执行代码。

---

## 5. 社区信号与趋势 (Community Signals)

### 5.1 ICML 2025 PRAL Workshop

"Programmatic Representations for Agent Learning" Workshop（710人容量）是这一研究方向获得社区层面认可的标志性事件。Workshop明确定义了五个子方向：
1. Programs as Policies（代码作为策略）
2. Programs as Reward Functions（代码作为奖励函数）
3. Programs as Skill Libraries（代码作为技能库）
4. Programs as Task Generators（代码作为任务生成器）
5. Programs as Environment Models（代码作为环境模型）

受邀报告人包括来自Georgia Tech、UT Austin、Google DeepMind、University of Toronto等的研究者。

### 5.2 ARC-AGI竞赛

Abstraction and Reasoning Corpus (ARC) 竞赛的经验高度相关：**纯神经网络方法在抽象推理上失败，而基于DSL的程序合成方法占据主导地位**。2025年的最佳方法结合了LLM引导的进化程序合成，达到52%的解决率。

### 5.3 产业趋势

CodeAct演化为OpenHands（前OpenDevin），成为生产级的代码Agent平台。SkillRL (2025) 将代码技能库概念引入强化学习。AgentDistill (2024) 提出通过提取可复用代码模块进行无训练Agent蒸馏。

---

## 6. 开放问题与未来方向 (Open Problems and Future Directions)

### 6.1 核心开放问题

**问题1：Agent原生知识格式**
当前的代码表示仍然是为人类设计的编程语言。Agent是否应该发展自己的"Agent原生"形式化语言？一种最大化压缩和组合性的表示，可能对人类不可读，但可以通过翻译层提供人类视图？

**问题2：随机性和部分可观测性**
WorldCoder等方法仅适用于确定性、完全可观测环境。概率编程语言（如Pyro、Gen）作为Agent知识表示是一个重要的未探索方向。

**问题3：可扩展性**
从<100行代码（WorldCoder）到4000+行（PoE-World）的扩展表明了可扩展性的挑战。如何用代码表示百万级的事实和规则？

**问题4：多Agent知识共享**
如果多个Agent都使用代码表示知识，它们如何共享、合并和版本控制这些知识？类似Git的Agent知识版本控制系统可能是未来方向。

**问题5：统一评估基准**
缺少直接比较代码 vs. 自然语言 vs. 知识图谱 vs. 混合表示的统一基准。

### 6.2 研究路线图

**近期（1-2年）**：
- 统一的表示比较基准
- 可扩展的模块化代码架构（微服务风格）
- 概率程序世界模型

**中期（2-5年）**：
- Agent原生知识格式 + 人类翻译层
- 多Agent代码知识共享协议
- 知识表示效率的形式化理论

**长期（5+年）**：
- Agent自主进化的DSL
- Agent之间的知识经济
- Agent知识正确性的形式化验证

---

## 7. 结论 (Conclusion)

本综述系统调研了用代码/形式化语言替代自然语言作为LLM Agent知识表示的研究方向。主要发现如下：

1. **这是一个真实且快速增长的研究方向**，已在ICML 2025上形成专门的workshop（PRAL），有来自WorldCoder (NeurIPS'24)、CodeAct (ICML'24)、Voyager (NeurIPS'23)等高影响力工作的支持。

2. **代码表示在所有评估维度上显著优于自然语言**：样本效率（10,000x）、任务成功率（+20.7%）、推理准确率（+15-25%）、规划速度（10^4-10^7x）。

3. **最优表示是代码+自然语言注释的混合体**（LILO的发现），而非纯代码或纯自然语言。

4. **知识渐进形式化**是一个反复出现的模式：从情景记忆（NL轨迹）→语义记忆（结构化NL）→程序性记忆（可执行代码）→压缩库（DSL抽象）。

5. **关键开放问题**包括：处理随机性和部分可观测性、扩展到复杂环境、多Agent知识共享、以及是否应该发展"Agent原生"的知识格式。

用户提出的核心洞察——LLM Agent应该以代码/形式化语言而非自然语言来表示探索结果——**得到了强有力的文献支持**。这一方向的最新进展（特别是WorldCoder和CodePlan）提供了令人信服的证据。更深层的洞察在于：问题不仅仅是处理速度的差异，而是代码作为知识表示在**可执行性、组合性、可压缩性、无歧义性和可迁移性**方面的根本性优势。

---

## References

[@tang2024worldcoder] Hao Tang, Darren Key, Kevin Ellis. "WorldCoder, a Model-Based LLM Agent: Building World Models by Writing Code and Interacting with the Environment." NeurIPS 2024.

[@wang2024codeact] Xingyao Wang et al. "Executable Code Actions Elicit Better LLM Agents." ICML 2024.

[@wang2023voyager] Guanzhi Wang et al. "Voyager: An Open-Ended Embodied Agent with Large Language Models." NeurIPS 2023.

[@gao2023pal] Luyu Gao et al. "PAL: Program-aided Language Models." ICML 2023.

[@chen2023pot] Wenhu Chen et al. "Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks." TMLR 2023.

[@wen2025codeplan] Jiaxin Wen et al. "CodePlan: Unlocking Reasoning Potential in Large Language Models by Scaling Code-form Planning." ICLR 2025.

[@ellis2021dreamcoder] Kevin Ellis et al. "DreamCoder: Growing Generalizable, Interpretable Knowledge with Wake-Sleep Bayesian Program Learning." PLDI 2021.

[@grand2024lilo] Gabriel Grand et al. "LILO: Learning Interpretable Libraries by Compressing and Documenting Code." ICML 2024.

[@zelikman2023parsel] Eric Zelikman et al. "Parsel: Algorithmic Reasoning with Language Models by Composing Decompositions." NeurIPS 2023.

[@liang2023cap] Jacky Liang et al. "Code as Policies: Language Model Programs for Embodied Control." ICRA 2023.

[@singh2023progprompt] Ishika Singh et al. "ProgPrompt: Program Generation for Situated Robot Task Planning Using Large Language Models." ICRA 2023.

[@sumers2024coala] Theodore Sumers et al. "Cognitive Architectures for Language Agents." TMLR 2024.

[@shinn2023reflexion] Noah Shinn et al. "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023.

[@yang2025survey] Dayu Yang et al. "Code to Think, Think to Code: A Survey on Code-Enhanced Reasoning and Reasoning-Driven Code Intelligence in LLMs." EMNLP 2025.

[@neurips2024pddl] "Leveraging Environment Interaction for Automated PDDL Translation and Planning with Large Language Models." NeurIPS 2024.

[@iclr2025formplan] "Planning Anything with Rigor: General-Purpose Zero-Shot Planning with LLM-based Formalized Programming." ICLR 2025.

[@icml2025codeio] "CodeIO: Condensing Reasoning Patterns via Code Input-Output Prediction." ICML 2025 Oral.

Full bibliography available in `references.bib` (74 entries).
