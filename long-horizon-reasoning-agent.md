# Long-Horizon Reasoning in LLM Agents: A Survey of Inference-Time and System Design Approaches

## Abstract

Large language models exhibit remarkable capabilities on isolated reasoning tasks, yet they struggle when confronted with problems that unfold over many steps, require maintaining coherent state across long interaction histories, or demand exploration of large decision spaces. This survey examines a rapidly maturing body of work demonstrating that **system-level techniques -- without any modification to model weights -- can dramatically extend the effective reasoning horizon of LLM agents**. We organize the landscape into seven technique categories: search and planning, decomposition, reflection and self-correction, memory and context management, multi-agent methods, compute allocation, and cognitive architectures. Drawing on over 60 recent papers (2023--2026), we show that test-time compute scaling can outperform a 14x larger model, that planner-executor separation yields +34 percentage point gains over reactive baselines on web tasks, and that structured memory enables 3.5x performance improvements at 3.7x lower memory cost. We provide a taxonomy with concrete empirical results, analyze technique compositions and scaling properties, survey the benchmark landscape, and offer practical recommendations for building long-horizon reasoning systems. The central insight is that the frontier of LLM reasoning is increasingly defined not by model size alone, but by the sophistication of the inference-time system surrounding it.

## 1. Introduction

### 1.1 The Long-Horizon Problem

When an LLM is asked to solve a single-step question, performance is often impressive. But real-world agent tasks -- debugging a codebase, navigating a website to complete a multi-page form, planning a research project, or coordinating a complex workflow -- require reasoning across dozens or hundreds of sequential decisions. In these **long-horizon** settings, LLMs face compounding challenges: errors accumulate multiplicatively across steps, context windows overflow with interaction history, and the combinatorial decision space grows exponentially with horizon length.

These failures are not primarily failures of knowledge. The same model that cannot complete a 50-step web task can often answer each sub-question in isolation. The bottleneck lies in **system-level orchestration**: how the model's knowledge is sequenced, verified, remembered, and composed over extended interactions.

### 1.2 Scope and Motivation

This survey focuses exclusively on **inference-time techniques** -- methods that improve LLM reasoning without modifying model weights through training, reinforcement learning, or architectural changes to the transformer itself. This constraint is both practically and scientifically motivated:

- **Practical**: Inference-time methods can be deployed with any foundation model, including proprietary APIs where weight modification is impossible. Organizations can upgrade reasoning capabilities without retraining.
- **Scientific**: Understanding what can be achieved through system design alone clarifies the boundary between what LLMs "know" and what they can "do," and illuminates the role of cognitive architecture in intelligent behavior.

We cover work published primarily between 2023 and early 2026, spanning the period from the introduction of Tree of Thoughts [@yao2023tot] and Reflexion [@shinn2023reflexion] through the emergence of provable test-time scaling laws [@provable2024] and agentic cognitive architectures [@sumers2024coala].

### 1.3 Organization

Section 2 establishes theoretical foundations. Section 3 presents a seven-category taxonomy of techniques with representative methods and empirical results. Sections 4--6 address technique composition, scaling analysis, and benchmarks. Section 7 provides practical recommendations, and Section 8 identifies open problems.

## 2. Theoretical Foundations

### 2.1 Test-Time Compute Scaling Laws

The seminal work of Snell et al. [@snell2024scaling] demonstrated that **optimally allocating test-time compute can be more effective than scaling model parameters**. Their key finding: a compute-optimal test-time strategy improves efficiency by more than 4x compared to best-of-N baselines. In FLOPs-matched evaluations, a smaller model with optimal test-time compute outperforms a **14x larger model** on problems where the smaller model achieves non-trivial baseline success rates.

Two primary mechanisms drive these gains: (1) searching against dense **process-based verifier reward models** (PRMs) that score intermediate reasoning steps, and (2) **adaptively updating** the model's output distribution conditioned on the specific prompt at test time. Critically, the optimal strategy is **difficulty-dependent**: easy problems benefit most from iterative refinement of a single solution, while hard problems require broader search across diverse solution approaches.

These empirical findings have been placed on rigorous theoretical footing. Li et al. [@provable2024] proved that a two-stage knockout-style algorithm -- generating multiple candidates and aggregating via tournament selection -- achieves failure probability decay that is **exponential or power-law** as test-time compute grows. This establishes that test-time scaling is not merely an empirical observation but a provable phenomenon with well-characterized rates.

### 2.2 The Search-Verify Paradigm

Process Reward Models (PRMs) assign reward signals to **intermediate reasoning steps** rather than only to final outputs, enabling dense, structured supervision for step-by-step verification [@prm2024scaling]. ThinkPRM [@thinkprm2025] demonstrated that long chain-of-thought verifiers fine-tuned on **only 1% of PRM800K labels** outperform both LLM-as-Judge and discriminative verifiers on ProcessBench, MATH-500, and AIME 2024 benchmarks. This data efficiency makes the search-verify paradigm accessible without massive annotation efforts.

The combination of PRM-guided search with tree-structured exploration forms a powerful inference-time reasoning engine: the model generates candidate reasoning steps, the PRM evaluates each step, and search algorithms (beam search, MCTS) navigate the resulting tree. Positive steps are accepted; negative or neutral steps trigger expansion or backtracking.

### 2.3 Cognitive Architecture Principles

The Cognitive Architectures for Language Agents (CoALA) framework [@sumers2024coala] provides a unifying theoretical lens. CoALA decomposes agent architectures into three components:

1. **Memory System**: Working memory (current context), episodic memory (past experiences), semantic memory (world knowledge), and procedural memory (skills and strategies).
2. **Action Space**: Internal actions (memory read/write, reasoning) and external actions (tool use, environment interaction).
3. **Decision-Making Cycle**: Retrieval + reasoning to propose actions, evaluate candidates, select and execute, observe results, and repeat.

This framework generalizes ReAct, Reflexion, LATS, and many other agent patterns into a common language, revealing that most advances in long-horizon reasoning can be understood as innovations in one or more of these three components. Memory modularity enables persistent context across long horizons; procedural memory enables skill reuse; and the decision-making loop supports iterative refinement with explicit planning.

## 3. Taxonomy of Techniques

### 3.1 Search and Planning

**Core Mechanism.** Rather than generating a single reasoning trajectory, search methods explore a tree or graph of possible reasoning paths, using evaluation functions to guide exploration and backtracking to escape dead ends.

**Representative Methods.**

| Method | Venue | Key Result | Compute |
|--------|-------|------------|---------|
| Tree of Thoughts (ToT) | NeurIPS 2023 | Game of 24: 74% vs. 4% for CoT (GPT-4) | Medium-High |
| LATS | ICML 2024 | 92.7% pass@1 HumanEval (GPT-4); 75.9 WebShop | High |
| Self-Backtracking | arXiv 2025 | +40% over optimal-path supervised fine-tuning | Medium |

**Tree of Thoughts** [@yao2023tot] maintains a tree of "thoughts" -- coherent language sequences serving as intermediate reasoning steps. The LLM both generates candidate thoughts and evaluates their promise, while BFS or DFS algorithms manage exploration. On the Game of 24, ToT achieves 74% success compared to 4% for standard chain-of-thought, demonstrating the power of systematic exploration with backtracking.

**LATS (Language Agent Tree Search)** [@zhou2024lats] integrates Monte Carlo Tree Search with LLM-powered value functions, self-reflection, and environmental feedback. The algorithm follows the classic MCTS cycle -- selection via UCT, expansion, simulation, backpropagation -- augmented with verbal self-reflection on failure to guide future exploration. LATS achieves 92.7% pass@1 on HumanEval with GPT-4 and 75.9 on WebShop with GPT-3.5.

**Strengths and Limitations.** Search methods provide the strongest reasoning guarantees on hard problems through systematic exploration. However, they are compute-intensive (LATS requires many LLM calls per problem) and assume that states can be evaluated and that backtracking is feasible. In real-world environments with irreversible actions (web navigation, code deployment), pure search is often impractical.

**When to Use.** Search methods are ideal for well-defined problems with clear evaluation criteria, where individual steps can be scored, and where the solution space is large but tractable -- competitive mathematics, code generation, and puzzle solving.

### 3.2 Decomposition

**Core Mechanism.** Complex tasks are broken into smaller sub-tasks, each within the reasoning capacity of a single LLM call. A higher-level planner sequences these sub-tasks and manages dependencies.

**Representative Methods.**

| Method | Venue | Key Result |
|--------|-------|------------|
| Plan-and-Act | arXiv 2025 | 57.58% WebArena-Lite (SOTA); +34pp over ReAct |
| HiPlan | 2025 | Substantial gains via global milestones + local hints |
| ReAcTree | arXiv 2025 | 61% success vs. 31% for ReAct on WAH-NL |

**Plan-and-Act** [@planandact2025] separates reasoning into a **Planner LLM** that generates step-by-step plans and an **Executor LLM** that translates plan steps into environment-specific actions. This clean separation of concerns yields 57.58% on WebArena-Lite, a 34 percentage point improvement over reactive ReAct baselines, with dynamic replanning when the environment state deviates from expectations.

**HiPlan** combines global milestone guides (coarse-grained strategic waypoints) with local stepwise hints (fine-grained guidance) and a retrieval-augmented milestone library. This two-level hierarchy enables the agent to maintain strategic coherence while adapting tactically to local conditions.

**ReAcTree** [@reactree2025] dynamically constructs an agent tree where each subgoal is handled by an LLM agent node capable of reasoning, acting, and further tree expansion. It integrates control flow primitives (conditionals, loops) into the tree structure, achieving 61% success on the Watch-And-Help benchmark compared to 31% for standard ReAct.

**Strengths and Limitations.** Decomposition dramatically reduces per-step complexity and error propagation. Plan-then-execute consistently outperforms reactive approaches for long horizons. However, the quality of decomposition depends on the planner's ability to anticipate sub-task structure, and rigid decompositions may fail when tasks require dynamic restructuring.

**When to Use.** Decomposition is the default recommendation for multi-step agentic tasks -- web navigation, software engineering, research workflows -- especially when sub-tasks are relatively independent and the overall structure can be anticipated.

### 3.3 Reflection and Self-Correction

**Core Mechanism.** The agent evaluates its own outputs, generates verbal critiques, and uses these reflections to improve subsequent attempts. This creates a feedback loop without external training signal.

**Representative Methods.**

| Method | Venue | Key Result |
|--------|-------|------------|
| Reflexion | NeurIPS 2023 | AlfWorld: 130/134 tasks (vs. ReAct baseline) |
| Self-Refine | NeurIPS 2023 | Consistent improvement across tasks via self-feedback loops |
| Multi-round Thinking | arXiv 2025 | QwQ-32B: 80.3% to 82.1% on AIME 2024 |

**Reflexion** [@shinn2023reflexion] converts environment feedback into linguistic self-reflection provided as context in the next episode. The cycle -- define task, generate trajectory, evaluate, reflect, retry -- creates a form of verbal reinforcement learning. On AlfWorld, Reflexion completes 130 of 134 tasks, dramatically outperforming ReAct. Key components include self-evaluation, self-reflection, and episodic memory that persists across attempts.

**Self-Refine** [@madaan2023selfrefine] implements a single-model iterative loop: generate output, critique it, and refine. The model serves as both generator and critic, requiring no external feedback. Renze and Guven [@renze2024] showed that richer reflections (combining instructions, explanations, and solutions) significantly outperform limited reflections like simple retries (p < 0.001).

**Strengths and Limitations.** Reflection offers arguably the highest return on investment among all techniques: moderate compute overhead with consistent improvements across benchmarks. However, models can exhibit systematic blind spots, generating plausible but incorrect self-critiques. Self-correction without external grounding (environment feedback, verification) can sometimes degrade performance.

**When to Use.** Reflection should be a default component of any long-horizon agent. It is particularly effective when combined with environment feedback (Reflexion) or external verification (evaluator-optimizer pattern).

### 3.4 Memory and Context Management

**Core Mechanism.** Structured memory systems manage information flow across long interaction histories, preventing context overflow and ensuring that critical information remains accessible.

**Representative Methods.**

| Method | Venue | Key Result |
|--------|-------|------------|
| MEM1 | ICLR 2026 | 3.5x performance gain, 3.7x memory reduction vs. Qwen2.5-14B |
| MemGPT | 2023 | Virtual context management via OS-inspired paging |
| Context Folding | arXiv 2025 | Branch into sub-trajectory, collapse upon completion |
| COMPASS | arXiv 2025 | Hierarchical framework: Main Agent + Meta-Thinker + Context Manager |

**MEM1** [@mem1_2025] trains an agent to maintain a **constant-size internal state** that jointly supports memory consolidation and reasoning. At each turn, the agent updates a compact shared state integrating prior memory with new observations while discarding irrelevant information. MEM1-7B achieves a 3.5x performance gain with 3.7x memory reduction compared to Qwen2.5-14B, and crucially generalizes beyond its training horizon -- a key property for long-horizon tasks.

**MemGPT** [@packer2023memgpt] draws an analogy to operating system virtual memory: the LLM's context window serves as RAM, while external storage serves as disk. The model manages paging -- deciding what to load into and evict from active context -- enabling unbounded interaction histories within a fixed context window.

**Context Folding** [@contextfolding2025] allows an agent to branch into a sub-trajectory for a subtask, then "fold" (collapse) intermediate steps into a concise summary upon completion. This preserves the essential outcome while dramatically reducing context length.

**Strengths and Limitations.** Memory management is essential for any long-horizon system -- without it, agents overlook critical evidence or become distracted by irrelevant information as histories grow. The limitation lies in information loss during compression: aggressive summarization may discard details needed for later steps.

**When to Use.** Memory management is a prerequisite, not an option, for long-horizon agents. The specific strategy should match the task: MEM1-style constant memory for tasks with predictable information needs, MemGPT-style paging for tasks requiring selective recall, and context folding for tasks with clear hierarchical sub-task structure.

### 3.5 Multi-Agent Methods

**Core Mechanism.** Multiple LLM instances collaborate -- through debate, layered refinement, or parallel exploration -- to produce outputs superior to any single instance.

**Representative Methods.**

| Method | Venue | Key Result |
|--------|-------|------------|
| Multi-Agent Debate (MAD) | ICML 2024 | Improved factuality; arithmetic improves with agents/rounds |
| Mixture-of-Agents (MoA) | ICLR 2025 | 65.1% AlpacaEval 2.0 (vs. 57.5% GPT-4o) using open-source models |
| Diversity of Thought in MAD | arXiv 2024 | 4 rounds of diverse medium models beat GPT-4 on GSM-8K (91%) |

**Mixture-of-Agents** [@moa2024] employs a layered architecture where each layer contains multiple LLM agents, each receiving all outputs from the previous layer as auxiliary information. This exploits a "collaborativeness" phenomenon: models produce better responses when given access to other models' outputs, even lower-quality ones. MoA achieves 65.1% on AlpacaEval 2.0 using only open-source models, surpassing GPT-4o's 57.5%.

**Multi-Agent Debate** [@du2024mad] has multiple LLM instances propose solutions, critique each other, and converge over rounds. The finding that diverse trained models outperform homogeneous ensembles -- with four rounds of diverse medium models beating GPT-4 on GSM-8K at 91% -- underscores that **diversity matters more than individual model strength**.

However, Google Research's analysis of scaling agent systems [@google2025scaling] reveals important nuances: centralized coordination improves performance by +80.9% over single agents for parallelizable tasks (e.g., financial reasoning), but every multi-agent variant **degraded performance by 39--70%** on strict sequential reasoning tasks. Independent agents amplify errors 17.2x while centralized coordination contains amplification to 4.4x.

**Strengths and Limitations.** Multi-agent methods excel at tasks requiring diverse perspectives, verification, and factual reasoning. They are less effective for strict sequential reasoning where coordination overhead outweighs benefits. Compute costs scale linearly or super-linearly with agent count.

**When to Use.** Multi-agent approaches are best suited for tasks where multiple perspectives genuinely help: fact verification, creative generation, and parallelizable analysis. Avoid for tightly sequential reasoning where a single well-orchestrated agent is superior.

### 3.6 Compute Allocation

**Core Mechanism.** Rather than applying uniform compute to all problems or steps, adaptive methods estimate difficulty and allocate resources accordingly.

**Representative Methods.**

| Method | Venue | Key Result |
|--------|-------|------------|
| SelfBudgeter | arXiv 2025 | 61% response length compression maintaining accuracy |
| Input-Adaptive Computation | OpenReview 2025 | Up to 50% compute reduction at no quality cost |
| Sonata | 2025 | 20--60% thinking token reduction; or +2% accuracy at same cost |
| Best-of-Majority (BoM) | arXiv 2025 | Minimax-optimal strategy for Pass@k scaling |

**SelfBudgeter** [@selfbudgeter2025] trains models to self-estimate the reasoning budget required per query, then applies budget-guided RL. This achieves 61% average response length compression on math reasoning while maintaining accuracy -- the model learns to think briefly on easy problems and extensively on hard ones.

**Best-of-Majority** [@bom2025] addresses a subtle failure mode: neither majority voting nor best-of-N sampling scales monotonically (performance can degrade with more samples). BoM restricts candidates to high-frequency responses before selecting via reward model, achieving minimax-optimal Pass@k scaling.

The key insight across this category is that **problem difficulty should determine strategy**: easy problems benefit from iterative refinement of a single answer, while hard problems require broader search across diverse approaches [@thinktwice2025]. Multi-round thinking improves QwQ-32B from 80.3% to 82.1% on AIME 2024 through iterative refinement, while tree search provides larger gains on problems where the initial approach is fundamentally wrong.

**Strengths and Limitations.** Adaptive compute allocation improves efficiency dramatically, often by 50% or more. The challenge is accurately estimating difficulty before solving the problem -- a bootstrapping problem that current methods address through self-consistency signals, learned estimators, or iterative deepening.

**When to Use.** Compute allocation should be layered on top of any other technique. It is particularly valuable in production systems where cost efficiency matters and workloads contain a mix of easy and hard problems.

### 3.7 Cognitive Architectures

**Core Mechanism.** Full system designs that integrate multiple techniques (memory, planning, reflection, tool use) into a coherent architecture with defined control flow.

**Representative Methods.**

| Architecture | Source | Key Innovation |
|-------------|--------|----------------|
| CoALA | TMLR 2024 | Unifying framework: modular memory + structured actions + decision loop |
| MAP (Modular Agentic Planner) | Nature Comms. 2025 | Brain-inspired modules: conflict monitoring, state prediction, evaluation, decomposition, coordination |
| OpenHands | ICLR 2025 | Full agent platform: event stream + sandboxed runtime + agent hub |

**CoALA** [@sumers2024coala] does not prescribe a single architecture but provides a design space within which specific architectures can be described and compared. Its value for long-horizon reasoning lies in making explicit the role of each memory type (working, episodic, semantic, procedural) and each action type (internal, external) in supporting extended reasoning.

**MAP (Modular Agentic Planner)** [@map2025] draws directly from neuroscience, implementing specialized LLM modules for conflict monitoring, state prediction, state evaluation, task decomposition, and task coordination. This mirrors findings that humans use hierarchical nesting and meta-cognitive monitoring for complex planning, while current LLMs default to shallow forward chaining -- a divergence most pronounced on ill-structured problems.

**OpenHands** [@openhands2025] provides a production-grade architecture with an event stream tracking full interaction history (enabling backtracking and reflection), sandboxed runtime (enabling safe exploration), unified code action space via CodeAct, and a multi-agent coordination layer. Countdown mechanisms enforce bounded computation budgets.

**Strengths and Limitations.** Cognitive architectures provide the most complete solutions for long-horizon reasoning but are complex to implement and tune. The interaction effects between components (e.g., how memory management affects planning quality) are not yet well understood theoretically.

**When to Use.** Full cognitive architectures are appropriate for production agent systems that must handle diverse, open-ended tasks over extended time horizons. For narrower problems, individual techniques from categories 3.1--3.6 may suffice.

## 4. Technique Interactions and Compositions

The techniques described above are not mutually exclusive; indeed, the most effective systems compose multiple approaches. Several successful compositions have been documented:

**LATS = MCTS + Reflection + Value Functions.** LATS demonstrates that tree search becomes substantially more effective when augmented with verbal self-reflection on failure and LLM-powered value functions for node evaluation. The self-reflection component converts failed search branches into linguistic guidance for future exploration.

**COMPASS = Planning + Context Management + Meta-Cognition.** The COMPASS framework [@compass2025] separates the agent into three components: a Main Agent for tactical execution, a Meta-Thinker for strategic oversight, and a Context Manager for organizing structured context. This three-way decomposition addresses both the planning and memory challenges of long horizons simultaneously.

**RAT + MCTS = RARE.** The RARE framework [@rare2025] combines retrieval-augmented reasoning with Monte Carlo Tree Search, using retrieval actions within the MCTS loop and a factuality scorer to prioritize high-factuality reasoning paths. This grounds search in external knowledge, reducing hallucination during exploration.

**Plan-and-Act + Retrieval = Retrieval-Augmented World Models.** Grounding plan predictions in retrieved tutorials and procedures yields +25.3% gains in OSWorld and +18.1% in WebArena [@rawm2025], demonstrating that decomposition benefits substantially from external knowledge grounding.

The **composition challenge** remains: interactions between techniques are poorly understood, and naive combination can be counterproductive. For example, adding multi-agent debate to a task that requires strict sequential reasoning degrades performance by 39--70% [@google2025scaling]. Systematic study of technique interactions is an important open direction.

## 5. Scaling Analysis

Different technique categories exhibit qualitatively different scaling behaviors:

| Category | Scaling with Compute | Scaling Regime | Key Constraint |
|----------|---------------------|----------------|----------------|
| Search & Planning | Sub-linear to linear | Best for hard problems | Evaluation function quality |
| Decomposition | Step function (plan quality) | Best at medium horizons | Planner capability |
| Reflection | Logarithmic (diminishing returns) | 2--5 iterations optimal | Self-consistency ceiling |
| Memory & Context | Enables scaling (removes bottleneck) | Critical beyond ~10 steps | Information loss in compression |
| Multi-Agent | Sub-linear; can degrade | Diversity > quantity | Coordination overhead |
| Compute Allocation | Multiplicative (efficiency) | Universal applicability | Difficulty estimation accuracy |
| Cognitive Architectures | Synergistic (composition) | Long-horizon, open-ended | Engineering complexity |

The **difficulty-dependent optimality** insight is perhaps the most practically important finding in this literature. Snell et al. [@snell2024scaling] and subsequent work [@thinktwice2025] established that:

- For **easy problems** (model baseline > 60%): iterative refinement of a single solution is optimal. One good answer, refined 2--3 times, outperforms generating many independent samples.
- For **medium problems** (model baseline 20--60%): best-of-N with moderate N, guided by a process reward model, provides the best cost-performance trade-off.
- For **hard problems** (model baseline < 20%): tree search with backtracking is necessary to explore fundamentally different solution approaches.

The finding that a **smaller model with more inference compute can outperform a larger model** at fixed total FLOPs is profound for practical system design. It suggests that investment in inference-time infrastructure (verifiers, search algorithms, memory systems) can substitute for expensive model scaling.

## 6. Benchmark Landscape

### 6.1 Existing Benchmarks

| Benchmark | Domain | Horizon | What It Measures |
|-----------|--------|---------|-----------------|
| MATH / AIME | Mathematics | Short-Medium | Multi-step mathematical reasoning |
| HumanEval | Code Generation | Short | Single-function code synthesis |
| SWE-bench | Software Engineering | Long | End-to-end bug fixing in real repositories |
| WebArena | Web Navigation | Long | Multi-step web task completion |
| AlfWorld | Embodied Tasks | Medium | Household task completion in text environments |
| GSM-8K | Grade School Math | Short | Basic arithmetic reasoning chains |
| OSWorld | Desktop Interaction | Long | Operating system task completion |
| HotPotQA | Question Answering | Medium | Multi-hop reasoning over documents |
| OdysseyBench | Office Applications | Long | Complex office application tasks |

### 6.2 Coverage Gaps

Current benchmarks exhibit systematic gaps for evaluating long-horizon reasoning systems:

1. **Horizon length**: Most benchmarks (MATH, HumanEval, GSM-8K) require fewer than 10 reasoning steps. SWE-bench and WebArena push to 20--50 steps, but real-world agent tasks can extend to hundreds.
2. **Partial observability**: Few benchmarks model realistic information gathering, where the agent must decide what to observe.
3. **Irreversibility**: Most benchmarks allow unlimited retries, while real-world actions (sending emails, deploying code) are irreversible.
4. **Composition**: No benchmark systematically evaluates the interaction between techniques (e.g., does reflection help more when combined with decomposition?).
5. **Dynamic environments**: Benchmarks are typically static; real environments change during task execution.

The field urgently needs **long-horizon agentic benchmarks** that combine extended step counts (100+), partial observability, irreversible actions, and dynamic environments. OdysseyBench [@odysseybench2025] and WebArena represent steps in this direction, but significant gaps remain.

## 7. Practical Recommendations

### 7.1 Decision Framework

The following decision framework maps task characteristics to recommended techniques:

| Task Property | Primary Technique | Secondary Technique |
|--------------|-------------------|-------------------|
| Many sequential steps (>20) | Decomposition (Plan-and-Act) | Memory management (Context Folding) |
| Hard individual steps | Search (ToT, LATS) | PRM-guided verification |
| Requires factual accuracy | Retrieval augmentation (RAT) | Multi-agent debate |
| Long interaction history | Memory (MEM1, MemGPT) | Context summarization |
| Mix of easy and hard steps | Adaptive compute allocation | Routing to specialized models |
| Open-ended, diverse tasks | Cognitive architecture (CoALA) | Reflection + episodic memory |
| Irreversible actions | Planning with world models | Conservative verification |
| Parallelizable sub-tasks | Multi-agent (Orchestrator-Workers) | Centralized coordination |

### 7.2 Recommended Architecture for Long-Horizon Agents

Based on the evidence surveyed, we recommend a layered architecture:

1. **Foundation Layer**: Augmented LLM with tool use, retrieval, and structured output.
2. **Planning Layer**: Planner-executor separation (Plan-and-Act pattern) with dynamic replanning triggered by state deviation.
3. **Memory Layer**: Hierarchical memory (working + episodic + semantic) with context folding for completed sub-tasks and retrieval for long-term knowledge.
4. **Verification Layer**: Process reward model for step-level evaluation, enabling both guided search on hard sub-tasks and early error detection.
5. **Meta-Cognitive Layer**: Reflection after failures, adaptive compute allocation based on difficulty estimation, and budget-aware resource management.

### 7.3 Quick Wins vs. Heavy Investment

**Quick Wins (hours to implement, immediate impact):**
- Add self-reflection after failures (Reflexion pattern) -- consistent improvement across tasks
- Implement basic plan-then-execute with replanning -- +34pp on web tasks
- Add majority voting or best-of-N for critical decisions -- straightforward scaling
- Implement context summarization for long histories -- prevents context overflow

**Medium Investment (days to implement, significant impact):**
- Deploy process reward model for step-level verification
- Implement hierarchical memory with context folding
- Add retrieval augmentation at reasoning steps (+13--43% across domains)
- Implement adaptive compute allocation based on difficulty

**Heavy Investment (weeks+, maximum impact):**
- Full cognitive architecture with all memory types
- MCTS-based search with custom value functions
- Multi-agent system with specialized agents and orchestration
- Custom benchmark development for specific domain

## 8. Open Problems and Future Directions

### 8.1 Top Five Open Problems

1. **Composition Theory.** How do techniques interact when combined? Current understanding is empirical and fragmented. We lack a theory predicting when combinations will be synergistic versus antagonistic. The finding that multi-agent debate degrades sequential reasoning by 39--70% while improving parallelizable tasks by 80.9% illustrates the specificity of these interactions.

2. **Difficulty Estimation.** Adaptive compute allocation requires accurate difficulty estimation before solving the problem. Current approaches (self-consistency, learned estimators) are imperfect. Better difficulty estimation would unlock substantial efficiency gains across all technique categories.

3. **Information-Preserving Compression.** Memory management requires compressing interaction histories without losing critical information. Current summarization methods lack formal guarantees about what is preserved. How to determine which details will matter for future steps remains unsolved.

4. **Verification at Scale.** PRMs enable step-level verification, but current verifiers have limited coverage and can be fooled by plausible-sounding but incorrect reasoning. Scaling verification to open-ended, creative, and subjective tasks is an open frontier.

5. **Irreversibility-Aware Planning.** Real-world environments contain irreversible actions (sending emails, deploying code, physical manipulation). Current planning methods do not adequately distinguish between reversible exploration and irreversible commitment, leading to either excessive caution or costly mistakes.

### 8.2 Most Promising Future Directions

**Automated Architecture Search (ADAS).** Hu et al. [@adas2024] demonstrated that a meta-agent can program increasingly better agent designs in code, automatically discovering novel architectures. This points toward a future where inference-time systems are themselves designed by AI, potentially discovering compositions that human researchers would not consider.

**Recursive and Self-Improving Architectures.** The emerging paradigm of recursive language models -- where models recursively process and compress their own context -- may enable indefinite-horizon reasoning. Combined with procedural memory (learning reusable skills), agents may develop the ability to improve their own reasoning strategies during deployment.

**Input-Time Scaling.** A complementary paradigm to inference-time scaling invests compute in enriching and transforming the input before reasoning begins [@inputtime2025]. Combining input-time and inference-time scaling may yield multiplicative improvements.

**Provable Guarantees.** The existence of provable test-time scaling laws [@provable2024] opens the door to systems with formal reasoning guarantees -- something traditionally associated only with symbolic AI. Extending these guarantees to more complex, compositional reasoning tasks is a promising theoretical direction.

**Agentic Meta-Synthesis.** Scaling reasoning through meta-level orchestration -- reasoning about which reasoning strategy to apply -- represents a form of cognitive metacognition. Early work [@metasynthesis2026] suggests this can yield improvements beyond any fixed strategy, but the approach is nascent.

## 9. Conclusion

This survey has mapped the rapidly evolving landscape of inference-time and system-design approaches to long-horizon reasoning in LLM agents. The evidence is clear: without changing a single model weight, system-level techniques can extend reasoning horizons by orders of magnitude. A compute-optimal test-time strategy outperforms a 14x larger model. Planner-executor separation yields 34 percentage point improvements on web tasks. Structured memory enables 3.5x performance gains at 3.7x lower memory cost. Tree search transforms 4% success to 74% on combinatorial puzzles.

The field is converging on several design principles: separate planning from execution, manage memory hierarchically, verify intermediate steps, allocate compute adaptively, and reflect on failures. These principles, instantiated in concrete systems like LATS, Plan-and-Act, MEM1, and COMPASS, represent a new paradigm where the intelligence of an AI system is determined not only by its model weights but by the sophistication of the cognitive architecture surrounding it.

The most important open challenge is understanding how techniques compose. As the field matures, we expect the emergence of principled frameworks for selecting, combining, and adapting inference-time techniques to specific tasks and resource constraints. The ultimate goal -- agents capable of coherent reasoning across arbitrarily long horizons -- remains distant but is increasingly within the scope of system design rather than fundamental capability limitations.

## References

- [@snell2024scaling] Snell, C. et al. "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters." NeurIPS 2024 / ICLR 2025. arXiv:2408.03314.
- [@provable2024] Li, Y. et al. "Provable Scaling Laws for the Test-Time Compute of Large Language Models." NeurIPS 2025. arXiv:2411.19477.
- [@inferscaling2024] Brown, B. et al. "Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving." ICLR 2025. arXiv:2408.00724.
- [@msresearch2025] Microsoft Research. "Inference-Time Scaling for Complex Tasks: Where We Stand and What Lies Ahead." Technical Report, 2025.
- [@ttssurvey2025] "What, How, Where, and How Well? A Survey on Test-Time Scaling in LLMs." 2025. testtimescaling.github.io.
- [@agentttc2025] "Scaling Test-time Compute for LLM Agents." 2025. arXiv:2506.12928.
- [@yao2023tot] Yao, S. et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." NeurIPS 2023. arXiv:2305.10601.
- [@demystify2024] "Demystifying Chains, Trees, and Graphs of Thoughts." 2024. arXiv:2401.14295.
- [@tms2025] "Thought Management System for Long-Horizon, Goal-Driven LLM Agents." ScienceDirect, 2025.
- [@zhou2024lats] Zhou, A. et al. "Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models." ICML 2024. arXiv:2310.04406.
- [@mcts2024] "Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning." 2024. arXiv:2405.00451.
- [@rekg2025] "ReKG-MCTS: Reinforcing LLM Reasoning on Knowledge Graphs via Training-Free MCTS." ACL Findings 2025.
- [@contrastive2024] "Interpretable Contrastive Monte Carlo Tree Search Reasoning." 2024. arXiv:2410.01707.
- [@planandact2025] "PLAN-AND-ACT: Improving Planning of Agents for Long-Horizon Tasks." 2025. arXiv:2503.09572.
- [@compass2025] "COMPASS: Enhancing Agent Long-Horizon Reasoning with Evolving Context." 2025. arXiv:2510.08790.
- [@contextfolding2025] "Scaling Long-Horizon LLM Agent via Context-Folding." 2025. arXiv:2510.11967.
- [@hiplan2025] Li, X. et al. "HiPlan: Global Milestone Guides + Local Stepwise Hints." 2025.
- [@reactree2025] "ReAcTree: Hierarchical LLM Agent Trees with Control Flow for Long-Horizon Task Planning." 2025. arXiv:2511.02424.
- [@shinn2023reflexion] Shinn, N. et al. "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023.
- [@madaan2023selfrefine] Madaan, A. et al. "Self-Refine: Iterative Refinement with Self-Feedback." NeurIPS 2023. arXiv:2303.17651.
- [@renze2024] Renze, M. and Guven, E. "Self-Reflection in LLM Agents: Effects on Problem-Solving Performance." 2024. arXiv:2405.06682.
- [@thinktwice2025] "Think Twice: Enhancing LLM Reasoning by Scaling Multi-round Test-time Thinking." 2025. arXiv:2503.19855.
- [@thinkoptimal2025] "Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning." 2025. arXiv:2502.18080.
- [@pot2026] "Policy of Thoughts: Scaling LLM Reasoning via Test-time Policy Evolution." 2026. arXiv:2601.20379.
- [@mem1_2025] "MEM1: Constant-Memory Long-Horizon Agent via RL-Trained Memory Consolidation." ICLR 2026. arXiv:2506.15841.
- [@agemem2026] "AgeMem: Agentic Memory via Unified LT/ST Memory Management." 2026. arXiv:2601.01885.
- [@packer2023memgpt] Packer, C. et al. "MemGPT: Towards LLMs as Operating Systems." 2023.
- [@agentprog2025] "AgentProg: Empowering Long-Horizon GUI Agents with Program-Guided Context Management." 2025. arXiv:2512.10371.
- [@du2024mad] Du, Y. et al. "Improving Factuality and Reasoning in Language Models through Multiagent Debate." ICML 2024. arXiv:2305.14325.
- [@moa2024] "Mixture-of-Agents." ICLR 2025. arXiv:2406.04692.
- [@diversemad2024] "Diversity of Thought Improves Reasoning Abilities of LLMs in Multi-Agent Debate." 2024. arXiv:2410.12853.
- [@google2025scaling] "Towards a Science of Scaling Agent Systems." Google Research, 2025. arXiv:2512.08296.
- [@selfbudgeter2025] "SelfBudgeter: Model Self-Estimates Reasoning Budget." 2025. arXiv:2505.11274.
- [@bom2025] "Best-of-Majority: Minimax-Optimal Strategy for Pass@k Inference Scaling." 2025. arXiv:2510.03199.
- [@tale2025] Han, C. et al. "Token-Budget-Aware LLM Reasoning." ACL Findings 2025.
- [@bats2025] "Budget-Aware Tool-Use Enables Effective Agent Scaling." 2025. arXiv:2511.17006.
- [@planbudget2025] "Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs." OpenReview, 2025.
- [@budgetsurvey2025] "Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs." 2025. arXiv:2507.02076.
- [@selfcertainty2025] "Scalable Best-of-N Selection for LLMs via Self-Certainty." 2025. arXiv:2502.18581.
- [@sumers2024coala] Sumers, T. R., Yao, S., Narasimhan, K., and Griffiths, T. L. "Cognitive Architectures for Language Agents." TMLR, 2024. arXiv:2309.02427.
- [@map2025] "A Brain-Inspired Agentic Architecture to Improve Planning with LLMs." Nature Communications, 2025.
- [@cogdesign2025] "Applying Cognitive Design Patterns to General LLM Agents." 2025. arXiv:2505.07087.
- [@openhands2025] "OpenHands: An Open Platform for AI Software Development Agents." ICLR 2025. arXiv:2407.16741.
- [@agentictax2026] "Agentic AI: Architectures, Taxonomies, and Evaluation." 2026. arXiv:2601.12560.
- [@thinkprm2025] Mukhal, M. et al. "Process Reward Models That Think." 2025. arXiv:2504.16828.
- [@prm2024scaling] "Scaling Automated Process Verifiers for LLM Reasoning." 2024. arXiv:2410.08146.
- [@adas2024] Hu, S. et al. "Automated Design of Agentic Systems." 2024. OpenReview.
- [@yao2023react] Yao, S. et al. "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023.
- [@rat2024] "Retrieval Augmented Thoughts Elicit Context-Aware Reasoning in Long-Horizon Generation." 2024. arXiv:2403.05313.
- [@rare2025] "RARE: Retrieval-Augmented Reasoning Enhancement." ACL 2025.
- [@rawm2025] "Retrieval-Augmented World Models for Web and OS Agents." 2025.
- [@coa2024] "Chain-of-Agents: Large Language Models Collaborating on Long-Context Tasks." 2024. arXiv:2406.02818.
- [@webdreamer2024] "WebDreamer: LLM as World Model for Web Agent Planning." 2024. arXiv:2411.06559.
- [@selfbacktrack2025] "Self-Backtracking: Internalizing Backtrack in LLMs." 2025. arXiv:2502.04404.
- [@inputtime2025] "Input Time Scaling." 2025. arXiv:2508.13654.
- [@longcot2025] "Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning LLMs." 2025. arXiv:2503.09567.
- [@metasynthesis2026] "Scaling the Scaling Logic: Agentic Meta-Synthesis of Logic Reasoning." 2026. arXiv:2602.13218.
- [@anthropic2024agents] Anthropic. "Building Effective Agents." December 2024.
- [@odysseybench2025] "ODYSSEYBENCH: Evaluating LLM Agents on Long-Horizon Complex Office Application." 2025. arXiv:2508.09124.
- [@widerdeeper2025] "Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search." NeurIPS 2025.
