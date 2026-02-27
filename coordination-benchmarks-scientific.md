# Coordination Benchmarks for Scientific and Technical Multi-Agent Systems: A Comprehensive Survey

**Research Date**: 2026-02-26
**Scope**: 529 papers from 10 top venues (2023–2026), 90 filtered, 12 deeply read
**Prior Work**: Builds on multi-gpu-agent-tasks and massive-multi-agent-benchmarks surveys

---

## Executive Summary

This survey systematically maps the landscape of benchmarks for evaluating multi-agent coordination in scientific and technical domains. After screening 529 papers across NeurIPS, ICLR, ICML, ACL, and AAAI (2023–2026), we identify **the critical finding**: the intersection of "meaningful inter-agent coordination" and "scientific/technical task domains" is essentially empty. No existing benchmark simultaneously tests (a) how well agents coordinate with each other and (b) whether that coordination produces better science, mathematics, or engineering outcomes.

We analyze 12 key benchmarks in depth, spanning LLM multi-agent coordination (MARBLE, SwarmBench), scientific discovery (AstaBench, ScienceAgentBench, BixBench, HeurekaBench), cooperative MARL (SMACv2, Melting Pot), human-agent collaboration (Collaborative Gym), system-level evaluation (MAESTRO), meta-evaluation (HAL), and multi-agent research (LiveResearchBench). We propose a two-dimensional taxonomy, identify 10 specific gaps, and outline concrete opportunities for new benchmark design.

---

## 1. Background and Motivation

### 1.1 The Promise of Multi-Agent Scientific Systems

Recent systems demonstrate that multiple AI agents can collaborate on complex tasks: MetaGPT and ChatDev for software development (ICLR 2024, ACL 2024), CellAgent for single-cell analysis (ICLR 2026), ResearchTown for simulating research communities (ICML 2025), and AI-Researcher for autonomous scientific innovation (NeurIPS 2025 Spotlight). However, these systems define their own evaluation protocols, making cross-system comparison impossible.

### 1.2 Why Coordination Benchmarks Matter

The distinction between "parallel" and "coordinated" multi-agent systems is fundamental:
- **Parallel**: N agents work independently; best result is selected (e.g., pass@K theorem proving, BoN sampling). Scales predictably but misses synergies.
- **Coordinated**: Agents communicate, share information, divide labor, and jointly plan. Can achieve outcomes impossible for any single agent but introduces coordination overhead.

For scientific tasks specifically, coordination matters because real research is inherently collaborative: scientists specialize (wet lab, computational, theoretical), share intermediate results, peer review, and iterate. Evaluating whether AI agents can replicate this coordination requires dedicated benchmarks.

### 1.3 Scope of This Survey

Continuing from two prior surveys — "Multi-GPU Agent Tasks" (993→65 papers, focused on parallel search patterns) and "Massive Multi-Agent Benchmarks" (595→66 papers, focused on scale) — this survey focuses specifically on:
- **Benchmarks** (not systems or algorithms)
- **Coordination** (not mere parallelism)
- **Scientific/technical domains** (math, biology, software, physics, chemistry, algorithms)

---

## 2. Taxonomy: Two Dimensions of Agent Benchmarks

### 2.1 Coordination Complexity Spectrum

| Level | Description | Representative Benchmarks |
|-------|-------------|--------------------------|
| **L0: No Coordination** | Single agent, no interaction | ScienceAgentBench, AstaBench, BixBench, HeurekaBench |
| **L1: Parallel Search** | N independent agents, best-of-N | miniF2F pass@K, SWE-bench parallel, PutnamBench |
| **L2: Implicit Coordination** | Shared environment, no explicit messaging | SMACv2 (basic), Overcooked (simple layouts) |
| **L3: Explicit Communication** | Agents exchange natural language or structured messages | MARBLE (graph), SwarmBench (local broadcast), Collaborative Gym |
| **L4: Structured Orchestration** | Hierarchical roles, SOPs, topology-based planning | MARBLE (tree/star), MetaGPT, MAESTRO systems |

### 2.2 Task Domain Categories

| Category | Benchmarks | Multi-Agent? |
|----------|-----------|-------------|
| **A. Scientific Discovery** | AstaBench (ICLR'26 Oral), ScienceAgentBench (ICLR'25), BixBench, HeurekaBench (ICLR'26), ScienceBoard, NewtonBench, InnovatorBench, LLM-SRBench (ICML'25 Oral) | **All single-agent** |
| **B. Mathematics/Formal** | miniF2F, PutnamBench, ProofNet, LeanWorkbook, Putnam-AXIOM, GeoBench, FATE, UGMathBench | Parallel search only |
| **C. Software Engineering** | SWE-bench, ACE-Bench, RECODE, Terminal-Bench, ITBench (ICML'25 Oral) | Mostly single-agent |
| **D. Game/Grid-World** | SMACv2 (NeurIPS'23), Melting Pot (NeurIPS'24), SwarmBench, GoBigger, FightLadder | **Yes** |
| **E. General Multi-Agent** | MARBLE (ACL'25), Collaborative Gym (ICLR'26), MAESTRO, OrchestrationBench (ICLR'26) | **Yes** |
| **F. Research Automation** | LiveResearchBench (ICLR'26), ResearchTown (ICML'25) | LiveResearchBench: Yes |

**The critical empty quadrant**: No benchmark exists at the intersection of L3–L4 coordination and Category A/B/C domains.

---

## 3. Deep Analysis of Key Benchmarks

### 3.1 Multi-Agent Coordination Benchmarks

#### MARBLE / MultiAgentBench (ACL 2025)
The first comprehensive LLM multi-agent coordination benchmark. Tests 6 scenarios (research collaboration, Minecraft building, database analysis, coding, Werewolf, bargaining) across 4 communication topologies (star, chain, tree, graph).

**Key quantitative findings**:
- Graph topology outperforms all others in task performance and token efficiency
- Tree topology is worst: highest token consumption, lowest scores
- Cognitive self-evolving planning beats group discussion (3% milestone improvement)
- **Scaling ceiling at 3 agents**: KPI improves from 1→3 agents, then degrades
- Performance peaks at 7 iterations, then drops sharply at 10
- GPT-4o-mini: 84.13% on research tasks; Minecraft hardest at 0.21-33.60%

**Coordination metrics**: Communication Score + Planning Score (5-point scales). Human evaluation validates automated scoring.
**Open source**: https://github.com/MultiagentBench/MARBLE

#### SwarmBench (2025, withdrawn from ICLR 2026)
Tests decentralized swarm coordination with 5 tasks (pursuit, synchronization, foraging, flocking, transport) in 2D grid. Agents have 5×5 local perception, 120-char anonymous local broadcast.

**Key quantitative findings**:
- 13 LLMs tested; o4-mini leads (9.60 pursuit score)
- Transport task: only o4-mini and deepseek-r1 achieve non-zero scores
- **Physical coordination predicts success far better than semantic communication** — mimicking natural swarms
- Increasing perception from 3×3 to 5×5 helps; 7×7 shows diminishing returns
- Communication convergence can *harm* complex tasks (agents copy poor strategies)

**Open source**: https://github.com/RUC-GSAI/YuLan-SwarmIntell

#### Collaborative Gym (ICLR 2026)
Evaluates human-agent collaboration via POMDP formulation. 3 tasks (travel planning, related work writing, tabular data analysis). Asynchronous, non-turn-taking interaction.

**Key quantitative findings**:
- Collaborative agents win 86% (travel), 74% (data), 66% (writing) over autonomous
- **65% of cases show communication failures; 40% show situational awareness failures**
- Initiative Entropy measures balance of control between human and agent
- Situational Planning Agent outperforms Fully Autonomous and Basic Collaborative

**Open source**: https://github.com/SALT-NLP/collaborative-gym

### 3.2 Scientific Discovery Benchmarks (Single-Agent)

#### AstaBench (ICLR 2026 Oral)
Most comprehensive scientific agent benchmark. 11 sub-benchmarks across 4 categories: literature understanding (4), coding/execution (3), data analysis (1), end-to-end discovery (2).

**Key quantitative findings**:
- Best overall: Asta v0 at 53.0%
- Literature QA: ~85-90% (approaching practical)
- Code execution: <25% average
- Data analysis: highest only 34% (hardest category)
- **E2E discovery: <1% completion rate** — the fundamental challenge
- Open-source models: only 11.1% (huge gap vs closed-source)

**Multi-agent potential**: 4 categories naturally map to specialized agents; E2E-Bench's <1% success strongly suggests single agents are insufficient.

#### ScienceAgentBench (ICLR 2025)
102 tasks from 44 publications across bioinformatics, computational chemistry, GIS, psychology.

**Key quantitative findings**:
- SOTA: o1 + Self-Debug at 42.2% success rate
- Claude 3.5 Sonnet + Self-Debug: 32.4% at 17× lower cost than OpenHands
- Self-Debug consistently outperforms OpenHands CodeAct
- 75%+ of solved tasks have <58.6 lines of reference code
- Primary failures: data processing and domain-specific tool usage

#### BixBench (arXiv 2025, preprint)
53 capsules with 296 questions in computational biology. Tests multi-step bioinformatics analysis.

**Key quantitative findings**:
- Claude 3.5 Sonnet: 17% open-answer accuracy
- GPT-4o: 9% accuracy
- MCQ performance: **no better than random** with refusal option
- Suggests models rely on memorization, not genuine analytical capability

#### HeurekaBench (ICLR 2026)
AI co-scientist evaluation framework. Triplet structure (D, Q, A) grounded in publications + code. Instantiated as sc-HeurekaBench (single-cell biology, 50 OEQs + 50 MCQs).

**Key quantitative findings**:
- Best agent (Claude-4-Sonnet planner): 2.58/5.0 OEQ correctness
- End-critic improves by +22% on weak answers; plan-critic degrades performance
- Retriever improves from 1.56 to 2.15 on tool-usage questions
- Open-source models: 2.08 (GPT-OSS-120B), closed gap with end-critic to 2.49

### 3.3 Cooperative MARL Benchmarks

#### SMACv2 (NeurIPS 2023)
Gold standard for cooperative MARL. Fixes original SMAC's flaws: open-loop policies no longer work; procedural generation prevents memorization.

**Key innovations**:
- EPO (Extended Partial Observability): parameter p controls information asymmetry. At p=0, only discovering agent sees each enemy — creates genuine communication requirements
- 18 scenario configurations: 5v5 to 20v23 across 3 races
- QMIX > MAPPO across most scenarios

#### Melting Pot (ICML 2021 / NeurIPS 2024)
50+ substrates, 256+ test scenarios, 2-16 agents. "Largest open-source benchmark for general-sum Markov games."

**Key innovations**:
- Focal/background population protocol: tests social generalization to unseen partners
- Resident (focal majority) + Visitor (focal minority) evaluation modes
- Mixed-motive design: neither purely cooperative nor competitive
- NeurIPS 2023 contest: 826 participants, only 11% beat baselines

### 3.4 System-Level and Meta-Evaluation

#### MAESTRO (arXiv 2025, preprint)
Evaluates 12 MAS across 4 frameworks (LangGraph, AutoGen, ADK, MCP-Agent). Framework-agnostic traces with 40+ attributes.

**Key finding**: **Architecture dominates over model choice** — upgrading LLMs does not reliably improve MAS performance. 75.17% of failures are silent semantic errors (no exception thrown).

#### HAL — Holistic Agent Leaderboard (ICLR 2026)
Unifies 9 benchmarks across 4 domains. 21,730 agent rollouts, ~$40K total cost, 2.5B tokens.

**Key findings**:
- Scaffold selection is as important as model selection
- More reasoning effort decreases accuracy in 21/36 configurations
- Costs range $13 (ScienceAgentBench) to $450+ (Online Mind2Web)
- Agents can be 100× more expensive but only 1% better

#### LiveResearchBench (ICLR 2026)
100 expert-curated deep research tasks. Evaluates 17 systems including single-agent, deep research, and multi-agent modes.

**Key finding**: Multi-agent average score 69.5 > single-agent Deep Research 66.4 > single-agent Web Search 62.8. Multi-agent wins on citation association (61.9 vs 52.9) but loses on consistency. Even best systems have 13-92 unsupported citation claims per report.

---

## 4. Cross-Cutting Findings

### 4.1 The Coordination Ceiling

Converging evidence from multiple sources establishes a coordination ceiling for current LLM agents:

| Source | Finding | Ceiling |
|--------|---------|---------|
| MARBLE (ACL 2025) | KPI degrades beyond 3 agents | ~3 agents |
| Google 2025 (180 configs) | Performance drops beyond 5-7 agents | ~5-7 agents |
| MacNet (ICLR 2025) | S-curve: logistic growth then plateau | Architecture-dependent |
| MARBLE iteration study | Performance drops at 10 iterations | ~7 iterations |
| LiveResearchBench | Multi-agent only ~7 points above single | Marginal improvement |
| Melting Pot contest | Only 11% beat baselines with novel partners | Very low |

**Synthesis**: Current LLM agents hit a coordination ceiling at 3-7 agents. Beyond that, communication overhead, message redundancy, and conflicting plans dominate task improvement. This is fundamentally different from MARL (which scales to 27+ agents in SMACv2) because each LLM interaction is orders of magnitude more expensive.

### 4.2 Architecture Matters More Than Model Choice

| Evidence Source | Finding |
|----------------|---------|
| MAESTRO | Architecture is dominant driver of performance; model upgrade doesn't reliably help |
| HAL | Scaffold selection is as important as model selection |
| HeurekaBench | Critic placement (end vs plan) matters more than model choice |
| MARBLE | Topology (graph > tree) matters more than model (GPT-4o-mini vs Llama-3.3) |
| ScienceAgentBench | Self-Debug > OpenHands CodeAct regardless of model |

### 4.3 Physical Coordination vs Semantic Communication

| Benchmark | Finding |
|-----------|---------|
| SwarmBench | Position-based metrics predict success far better than message content |
| Collaborative Gym | 65% of interaction failures are communication failures |
| MARBLE | Group discussion (more communication) performs worst |
| Melting Pot | Behavioral cues matter more than explicit communication |

**Synthesis**: More communication does not mean better coordination. Effective multi-agent systems coordinate through structured behavior patterns, not verbose dialogue. This has direct implications for benchmark design — metrics should capture behavioral coordination, not just message exchange.

### 4.4 Scientific Tasks Remain Extremely Hard for Single Agents

| Benchmark | Best Single-Agent Performance |
|-----------|------------------------------|
| AstaBench E2E | <1% completion |
| ScienceAgentBench | 42.2% (o1 + Self-Debug) |
| BixBench | 17% (Claude 3.5 Sonnet) |
| HeurekaBench | 2.58/5.0 (Claude-4-Sonnet) |

These baselines matter for coordination benchmark design: if single agents can barely solve the tasks, multi-agent coordination is premature. The benchmark must either (a) use simpler scientific tasks where single agents succeed >50%, or (b) evaluate whether coordination enables solving tasks that no single agent can.

---

## 5. Gap Analysis

### Gap 1 [CRITICAL]: No Coordination Benchmark for Scientific Discovery
All 7+ scientific discovery benchmarks evaluate single agents only. The intersection of "meaningful coordination" and "scientific domain" is empty.

### Gap 2 [CRITICAL]: No Standard Coordination Efficiency Metric
No benchmark measures the ratio of coordination overhead to task improvement. Proposed: Coordination Overhead Ratio (COR) = (multi-agent cost − N × single-agent cost) / task improvement.

### Gap 3 [HIGH]: No Social Generalization Testing for LLM Agents
Melting Pot's focal/background protocol tests adaptation to unknown partners. No LLM benchmark uses this approach.

### Gap 4 [HIGH]: No Bridge Between MARL and LLM Multi-Agent Benchmarks
Two communities with no shared evaluation. Missing comparative understanding of learned vs emergent coordination.

### Gap 5 [HIGH]: Scale Gap — No O(100+) Coordinating LLM Agent Benchmark
Current max: 16 agents (SwarmBench). Real deployed systems run thousands.

### Gap 6 [MEDIUM]: No Partial Information Benchmark for Scientific Tasks
SMACv2's EPO elegantly creates information asymmetry. No scientific benchmark does this.

### Gap 7 [MEDIUM]: No Temporal/Long-Horizon Coordination Benchmark
All benchmarks evaluate single-session coordination. No multi-day iterative collaboration.

### Gap 8 [MEDIUM]: No Cross-Domain Scientific Coordination Benchmark
AstaBench has 4 categories but doesn't test cross-category agent teams.

### Gap 9 [MEDIUM]: No Adversarial Robustness Testing for Coordination
No benchmark tests coordination with adversarial or unreliable agents.

### Gap 10 [MEDIUM]: No Human Scientist + AI Agent Collaboration Benchmark
Collaborative Gym tests human-agent collaboration but only for general tasks.

---

## 6. Opportunity: Designing a Scientific Coordination Benchmark

Based on the analysis, a benchmark filling the critical gap should:

### 6.1 Design Principles
1. **Build on existing scientific benchmarks** — Use AstaBench/ScienceAgentBench tasks as building blocks, not create from scratch
2. **Require genuine coordination** — Tasks must be unsolvable by any single agent (information asymmetry, skill specialization, or scale)
3. **Measure coordination quality** — Not just task outcomes, but COR, ISE, and RSI metrics
4. **Support multiple coordination structures** — Allow topology comparison (as MARBLE does)
5. **Include human-in-the-loop mode** — Following Collaborative Gym's model

### 6.2 Candidate Task Designs

| Task | Coordination Type | Building Blocks | Domain |
|------|------------------|----------------|--------|
| **Distributed Literature Review** | Information aggregation | AstaBench PaperFindingBench + LiveResearchBench | CS/Bio/Mixed |
| **Hypothesis-Data-Analysis Pipeline** | Sequential handoff | HeurekaBench questions + ScienceAgentBench analysis tasks | Biology |
| **Parallel Experiment + Synthesis** | Fork-join | ScienceAgentBench tasks + shared reporting | Multi-domain |
| **Cross-Domain Discovery** | Specialist coordination | AstaBench sub-benchmarks as role assignments | Multi-domain |
| **Peer Review Simulation** | Adversarial/cooperative | InnovatorBench + multi-agent review + revision | CS/ML |
| **Coordinated Theorem Proving** | Shared proof tree | miniF2F lemmas + proof decomposition | Mathematics |

### 6.3 Proposed Metrics

| Metric | Formula | Measures |
|--------|---------|---------|
| **Coordination Overhead Ratio** | (C_multi − N·C_single) / ΔPerformance | Cost of coordination per unit improvement |
| **Information Sharing Efficiency** | Useful_info_transmitted / Total_info_transmitted | Communication quality |
| **Role Specialization Index** | Var(agent_profiles) / Max_var | Degree of task division |
| **Coordination Scaling Factor** | Performance(N) / (N · Performance(1)) | Super/sub-linear scaling |
| **Robustness to Dropout** | Performance(N-1) / Performance(N) | Fault tolerance |

---

## 7. Open-Source Ecosystem

### 7.1 Benchmark Repositories

| Benchmark | GitHub Stars | Language | License | HuggingFace Dataset | Leaderboard |
|-----------|------------|---------|---------|--------------------|----|
| MARBLE | ~130 | Python | Apache 2.0 | — | — |
| SwarmBench | ~28 | Python | MIT | 6cf/swarmbench | — |
| AstaBench | ~200+ | Python | Apache 2.0 | allenai/asta-bench (gated) | HF Spaces |
| ScienceAgentBench | ~124 | Python | MIT | osunlp/ScienceAgentBench | Via OpenHands |
| BixBench | ~50+ | Python | MIT | futurehouse/BixBench | — |
| Collaborative Gym | ~70+ | Python/TS | MIT | — | — |
| MAESTRO | ~7 | Python | — | maestro-eval (HF) | — |
| HeurekaBench | ~6 | Python | — | Google Drive (44GB) | — |
| SMACv2 | ~295 | Python | MIT/Apache | — | — |
| Melting Pot | ~793 | Python | Apache 2.0 | — | — |
| HAL | ~228 | Python | — | — | hal.cs.princeton.edu |
| LiveResearchBench | ~111 | Python | — | Salesforce/LiveResearchBench | — |

### 7.2 Meta-Evaluation Frameworks

| Framework | Stars | Function | Best For |
|-----------|-------|----------|----------|
| **InspectAI** | 1,783 | UK AISI evaluation framework; 100+ evals | AstaBench's foundation; extensible |
| **AgentBench** | 3,181 | 8-environment LLM agent evaluation | Single-agent comprehensive testing |
| **HAL Harness** | 228 | Unified 8+ benchmark CLI + leaderboard | Cross-benchmark comparison |
| **OpenHands Benchmarks** | 49 | 15+ benchmarks, SWE focus | Code agent evaluation |

---

## 8. Conclusions

### 8.1 Key Takeaways

1. **The coordination × science gap is the primary opportunity**: No benchmark evaluates multi-agent coordination on scientific tasks. This is both the most important gap and the most tractable to address using existing building blocks.

2. **Current coordination benchmarks are nascent**: MARBLE (ACL 2025) and SwarmBench (2025) are the first dedicated LLM coordination benchmarks, both <1 year old. The field is still defining what "coordination evaluation" means.

3. **Coordination has a ceiling at 3-7 LLM agents**: Multiple independent studies converge on this finding. Benchmark design must account for this ceiling rather than assuming arbitrary scaling.

4. **Architecture dominates model choice**: How agents are organized matters more than which LLM powers them. Benchmarks should test topologies and frameworks, not just models.

5. **Physical/behavioral coordination outperforms verbal communication**: Effective agent teams coordinate through structured action patterns, not verbose dialogue. Metrics should capture this.

6. **Scientific tasks remain extremely hard for single agents**: E2E scientific discovery has <1% completion rate. Multi-agent benchmarks for science must either use simpler sub-tasks or evaluate the unique value of coordination on otherwise-unsolvable problems.

### 8.2 Recommended Actions

For **benchmark designers**:
- Combine AstaBench sub-benchmarks with MARBLE coordination topologies to create the first scientific coordination benchmark
- Adopt Melting Pot's focal/background protocol for testing LLM agent social generalization
- Define and standardize coordination efficiency metrics (COR, ISE, RSI)

For **system builders**:
- Use MAESTRO + HAL for system-level evaluation alongside task-level benchmarks
- Test coordination strategies at the 3-7 agent scale before attempting larger scales
- Prioritize graph/decentralized topologies over hierarchical ones

For **researchers**:
- Investigate why coordination overhead grows super-linearly with agent count
- Study the relationship between physical/behavioral coordination and verbal communication
- Bridge MARL and LLM multi-agent evaluation to understand learned vs emergent coordination

---

## References

### Peer-Reviewed (Conference Papers)

[@zhu2025multiagentbench] Zhu, K. et al. "MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents." ACL 2025.

[@ellis2023smacv2] Ellis, B. et al. "SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning." NeurIPS 2023.

[@trivedi2024meltingpot] Trivedi, R. et al. "Melting Pot Contest: Charting the Future of Generalized Cooperative Intelligence." NeurIPS 2024.

[@bragg2026astabench] Bragg, J. et al. "AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite." ICLR 2026.

[@chen2025scienceagentbench] Chen, Z. et al. "ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery." ICLR 2025.

[@shao2026collaborativegym] Shao, Y. et al. "Collaborative Gym: A Framework for Enabling and Evaluating Human-Agent Collaboration." ICLR 2026.

[@panigrahi2026heurekabench] Panigrahi, S. et al. "HeurekaBench: A Benchmarking Framework for AI Co-scientist." ICLR 2026.

[@gur2026orchestrationbench] "OrchestrationBench: LLM-Driven Agentic Planning and Tool Use in Multi-Domain Scenarios." ICLR 2026.

[@chan2026hal] "Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation." ICLR 2026.

[@chiang2026liveresearchbench] "LiveResearchBench: Benchmarking Single- and Multi-Agent Systems for Citation-Grounded Deep Research." ICLR 2026.

[@hong2024metagpt] Hong, S. et al. "MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework." ICLR 2024.

[@liu2024agentbench] Liu, X. et al. "AgentBench: Evaluating LLMs as Agents." ICLR 2024.

[@jimenez2024swebench] Jiménez, C. et al. "SWE-bench: Can Language Models Resolve Real-world Github Issues?" ICLR 2024.

[@qian2024chatdev] Qian, C. et al. "ChatDev: Communicative Agents for Software Development." ACL 2024.

### Preprints

[@ruan2025swarmbench] Ruan, K. et al. "SwarmBench: Benchmarking LLMs' Swarm Intelligence." arXiv:2505.04364. (preprint)

[@ma2026maestro] Ma, T. et al. "MAESTRO: Multi-Agent Evaluation Suite for Testing, Reliability, and Observability." arXiv:2601.00481. (preprint)

[@mitchener2025bixbench] Mitchener, L. et al. "BixBench: A Comprehensive Benchmark for LLM-based Agents in Computational Biology." arXiv:2503.00096. (preprint)

[@survey2025agent] "Evaluation and Benchmarking of LLM Agents: A Survey." arXiv:2507.21504. (preprint)
