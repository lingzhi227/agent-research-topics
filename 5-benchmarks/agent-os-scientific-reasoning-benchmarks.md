# Agent OS Methods for Scientific Reasoning Benchmarks: A Quantitative Survey

**Date**: 2026-02-25
**Paper Database**: 562 papers (472 conference + 90 supplementary), 75 core papers deep-analyzed
**Venues**: ICLR 2024–2026, NeurIPS 2024–2025, ICML 2024–2025, ACL 2025, EMNLP 2025
**Scope**: How Agent OS sub-methods (context engineering, memory management, workflow optimization, multi-agent coordination, MCTS reasoning, test-time compute scaling) quantitatively improve scientific reasoning benchmark performance

---

## Executive Summary

This report maps **Agent OS methods → benchmark → quantitative improvement** across 30+ methods and 20+ benchmarks, with a focus on scientific reasoning tasks. Our central finding: **only 4 out of 14 major Agent OS methods have been evaluated on scientific reasoning benchmarks**, despite these methods showing +10pp to +70pp improvements on coding and math tasks. This represents the single largest research gap in the field.

Key quantitative findings:
- **Scaffold design dominates model choice**: Same model with different scaffolds yields 2.7x–11x performance gaps (BixBench, MLE-bench)
- **Scientific reasoning is the hardest frontier**: Best systems score 42–48% on scientific benchmarks vs 80–95% on math/coding
- **MCTS is becoming the universal scaffold**: Applied to reasoning (+22pp GSM8K), tool use (+10% avg), workflow search (4.55% cost match), and training (5.7x GPU savings)
- **Compounding potential untapped**: No study combines context engineering + memory + workflow optimization + multi-agent coordination for scientific tasks

---

## 1. Agent OS Method Taxonomy

We organize Agent OS methods into 7 categories based on the system component they optimize:

```
Agent OS Methods (no/minimal model weight changes, improvement via system design)
│
├── A. Reasoning Scaffolds — structure the reasoning process
│   ├── A1. Chain-of-Thought & variants (CoT, Self-Consistency, Elastic Reasoning)
│   ├── A2. Tree/Graph Search (ToT, AGoT, GoT)
│   ├── A3. MCTS Reasoning (MCTSr, LATS, CMCTS, DeepSearch)
│   ├── A4. Self-Reflection (Reflexion, CyclicReflex)
│   └── A5. Cognitive (System-1.5, A*-Thought)
│
├── B. Workflow & Planning Optimization — optimize task execution pipelines
│   ├── B1. MCTS Workflow (AFlow, ScoreFlow)
│   ├── B2. Trained Scaffold (AgentFlow/Flow-GRPO)
│   ├── B3. Tool Planning (ToolTree)
│   ├── B4. Auto MAS (MAS-GPT, MetaAgent)
│   └── B5. Memory-Guided (FlowSearcher)
│
├── C. Context Engineering — manage information in the prompt
│   ├── C1. Context Accumulation (ACE)
│   ├── C2. Context Compression (PAACE, ACON)
│   ├── C3. Prompt Engineering (anti-early-stop)
│   └── C4. Tool Augmentation (+16.4pp HLE with tools)
│
├── D. Memory Management — retain knowledge across turns/sessions
│   ├── D1. Virtual Context (MemGPT)
│   ├── D2. Hierarchical (MemoryOS)
│   ├── D3. Self-Organizing (A-MEM)
│   ├── D4. RL-Based (Memory-R1, MemAgent)
│   └── D5. Graph (Mem0)
│
├── E. Multi-Agent Coordination — distribute reasoning across agents
│   ├── E1. Scientific (Eigen-1)
│   ├── E2. Topological (MacNet, EvoMAC)
│   ├── E3. Debate (GroupDebate)
│   └── E4. Test-time ensemble (TUMIX)
│
├── F. Test-Time Compute Scaling — allocate more inference compute
│   ├── F1. Compute-Optimal (Snell et al.)
│   ├── F2. Budget-Constrained (s1, Elastic Reasoning)
│   ├── F3. Parallel Scaling (Best-of-N, Self-Consistency)
│   ├── F4. Sequential Scaling (revision, reflection)
│   └── F5. Adaptive (AB-MCTS, Strategic Scaling)
│
└── G. Specialized Agent Design — domain-specific optimization
    ├── G1. Science (Edison, CP-Agent)
    ├── G2. Code (AIDE, OpenHands)
    ├── G3. Research (self-debug scaffold, IterativeAgent)
    └── G4. MLE (MLE-STAR, PiEvolve)
```

---

## 2. Master Comparison Tables: Method × Benchmark × Improvement

### 2.1 Reasoning Scaffold Methods (Category A)

| Method | Venue | Retrain? | Benchmark | Baseline | Result | Δ |
|--------|-------|----------|-----------|----------|--------|---|
| CoT (Wei et al.) | NeurIPS 2022 | No | GSM8K (PaLM-540B) | 56.5% | 74.4% | **+17.9pp** |
| Self-Consistency | ICLR 2023 | No | GSM8K | CoT baseline | +17.9% | **+17.9%** |
| Self-Consistency | ICLR 2023 | No | AQuA | CoT baseline | +12.2% | **+12.2%** |
| Tree-of-Thought | NeurIPS 2023 | No | Game of 24 (GPT-4) | 4% | 74% | **+70pp** |
| AGoT | preprint | No | GPQA-Diamond (GPT-4o) | 39.4% | 57.6% | **+18.2pp** |
| Reflexion | NeurIPS 2023 | No | HumanEval (GPT-4) | 80% | 91% | **+11pp** |
| Reflexion | NeurIPS 2023 | No | AlfWorld | ~55% | ~77% | **+22pp** |
| Reflexion | NeurIPS 2023 | No | HotPotQA | ~47% | ~67% | **+20pp** |
| LATS | ICML 2024 | No | HumanEval (GPT-4) | ~67% | 92.7% | **+25.7pp** |
| LATS | ICML 2024 | No | WebShop | — | 75.9 | Matches fine-tuning |
| MCTSr (8-roll) | preprint | No | GSM8K (LLaMA-3 8B) | 74.1% | 96.7% | **+22.6pp** |
| MCTSr (8-roll) | preprint | No | MATH L1/L5 | 57.2%/7.1% | 90.2%/34.1% | **+33pp/+27pp** |
| CMCTS | preprint | No | GSM8K | 89.0% | 91.8% | **+2.8%** |
| CMCTS | preprint | No | Math-500 | 45.2% | 51.2% | **+6.0%** |
| CARTS | ICLR 2025 | No | miniF2F-test | — | 49.6% pass@1 | Theorem proving SOTA |
| DeepSearch | ICLR 2026 | Yes (+50 steps) | Math avg (1.5B) | 49.5% (R1-Distill) | 62.95% | **+13.5pp** |
| Elastic Reasoning | ICLR 2026 | Yes (GRPO) | AIME/MATH500 | Standard model | Robust under budget | Adaptive |
| AB-MCTS | NeurIPS 2025 Spotlight | No | Coding/engineering | Sampling + MCTS | Better than both | Adaptive |

### 2.2 Agent OS Infrastructure Methods (Categories B–E)

| Method | Venue | Category | Retrain? | Benchmark | Baseline | Result | Δ |
|--------|-------|----------|----------|-----------|----------|--------|---|
| ACE | ICLR 2026 | Context Eng. | No | AppWorld | 43.0% | 60.0% | **+17.0pp** |
| ACE | ICLR 2026 | Context Eng. | No | AppWorld (hard) | CUGA baseline | +8.4pp | **+8.4pp** |
| PAACE | preprint | Context Compress. | 4B distill | AppWorld | — | 59.0% SOTA | 10x cost↓ |
| ACON | preprint | Context Compress. | No | Small model tasks | — | +46% | -26–54% tokens |
| CER | ACL 2025 | Context Replay | No | WebArena (GPT-4o) | 24.3% | 36.7% | **+12.4pp** |
| MemGPT | ICLR 2024 | Memory | No | Deep retrieval | 32% | 92.5% | **+60.5pp** |
| MemoryOS | EMNLP 2025 Oral | Memory | No | LoCoMo F1 | — | +49.11% | **+49.11%** |
| A-MEM | NeurIPS 2025 | Memory | No | Multi-hop QA | 1x | 2x | **2x, 1/7 tokens** |
| Memory-R1 | — | Memory | Yes (RL) | vs Mem0 | Mem0 baseline | +68.9% F1 | **+68.9%** |
| MemAgent | ICLR 2026 | Memory | Yes (RL) | Long-context tasks | — | — | RL-based memory |
| AFlow | ICLR 2025 Oral | Workflow | No | 6 benchmarks avg | GPT-4o | GPT-4o-mini match | **4.55% cost** |
| ScoreFlow | 2025 | Workflow | No | 6 benchmarks | AFlow baseline | +8.2% | **+8.2%** |
| AgentFlow | ICLR 2026 Oral | Workflow+RL | Yes | Search tasks (7B) | GPT-4o | 57.3% vs 49.1% | **+8.2pp** |
| AgentFlow | ICLR 2026 Oral | Workflow+RL | Yes | Math tasks (7B) | GPT-4o 35.1% | 51.5% | **+16.4pp** |
| AgentFlow | ICLR 2026 Oral | Workflow+RL | Yes | Scientific tasks (7B) | GPT-4o 45.5% | 63.5% | **+18.0pp** |
| AgentFlow | ICLR 2026 Oral | Workflow+RL | Yes | GAIA (7B) | GPT-4o 17.3% | 33.1% | **+15.8pp** |
| ToolTree | ICLR 2026 | Tool Planning | No | GTA F1 | SOTA | 66.95 | **~+10%** |
| ToolTree | ICLR 2026 | Tool Planning | No | ToolBench Pass Rate | SOTA | 69.04 | **~+10%** |
| FlowSearcher | ICLR 2026 | Memory-Guided | No | GAIA/BrowseComp/GPQA | RLHF systems | Matches/exceeds | No retrain |
| Eigen-1 | ICLR 2026 | Multi-Agent | No | HLE Bio/Chem Gold | 34.9% (SciMaster) | 48.3% | **+13.4pp** |
| Eigen-1 | ICLR 2026 | Multi-Agent | No | SuperGPQA Hard | — | 69.57% | Robust |
| Eigen-1 | ICLR 2026 | Multi-Agent | No | Token efficiency | 483.6K tokens | 218.9K | **-53.5%** |
| MAS-GPT | ICML 2025 | Auto MAS | Yes | 8 benchmarks avg | — | +3.89% | **+3.89%** |

### 2.3 Test-Time Compute Scaling (Category F)

| Method | Venue | Retrain? | Key Result | Efficiency |
|--------|-------|----------|------------|------------|
| Snell et al. | ICLR 2025 Oral | No | Small model > 14x larger with optimal TTC | **4x over Best-of-N** |
| o1 (RL CoT) | OpenAI 2024 | Yes | AIME 2024: 74.3% | — |
| o3 | OpenAI 2025 | Yes | AIME 2024: 96.7% | — |
| DeepSeek-R1-Zero | 2025 | Yes (GRPO) | AIME 2024: 71.0% | Emergent reasoning |
| s1 (budget forcing) | EMNLP 2025 | Yes (1K SFT) | Exceeds o1-preview by 27% | Minimal training |
| DeepSearch | ICLR 2026 | Yes (+50 steps) | 62.95% math avg (1.5B SOTA) | **5.7x fewer GPU hrs** |
| GPT-5 | OpenAI 2025 | Yes | AIME 2025: 94.6% | — |
| GPT-5.2 Thinking | OpenAI 2026 | Yes | FrontierMath: 40.3% | — |

### 2.4 Specialized Scientific Agent Systems (Category G)

| System | Type | Benchmark | Baseline | Result | Δ |
|--------|------|-----------|----------|--------|---|
| **Eigen-1** | Multi-agent + RAG | HLE Bio/Chem Gold | 30.2% (LLM) / 34.9% (agent) | 48.3% | **+13.4pp/+18.1pp** |
| **Edison** | Specialized scaffold | BixBench | 17% (Claude 3.5) | 46% | **+29pp** |
| **o1 + self-debug** | TTC + Reflection | ScienceAgentBench | 21.6% (OpenHands) | 42.2% | **+20.6pp** |
| **Anti-early-stop prompt** | Context Eng. | PaperBench | 13.2% (basic scaffold) | 24.4% | **+11.2pp** |
| **AIDE** | Tree-search coding | MLE-bench Lite | 0.8% (MLAB) | 16.9% | **11x gap** |
| **PiEvolve** | Specialized | MLE-bench Lite | 16.9% (AIDE) | 80.3% | **+63.4pp** |
| **CP-Agent** | Multimodal agent | Cellular profiling | — | F1=0.896 | Domain-specific |
| **Asta v0** | Mixture of LLMs | AstaBench Overall | 43.3% (ReAct+GPT-5) | 53.0% | **+9.7pp** |

---

## 3. Cross-Domain Benchmark Performance Summary

### 3.1 Scientific Benchmarks: Current SOTA

| Benchmark | Domain | Best System | Score | Without Scaffold | Scaffold Δ |
|-----------|--------|-------------|-------|-----------------|-------------|
| **HLE Bio/Chem Gold** | Biology/Chemistry | Eigen-1 (multi-agent) | 48.3% | 30.2% (LLM) | **+18.1pp** |
| **BixBench** | Bioinformatics | Edison (specialized) | 46% | 17% (Claude 3.5) | **+29pp** |
| **ScienceAgentBench** | Data-driven science | o1 + self-debug | 42.2% | 21.6% (OpenHands) | **+20.6pp** |
| **AstaBench** | Full research pipeline | Asta v0 (multi-LLM) | 53.0% | 43.3% (ReAct+GPT-5) | **+9.7pp** |
| **PaperBench** | Paper replication | o1 + enhanced prompt | 24.4% | 4.1% (GPT-4o) | **+20.3pp** |
| **MLE-bench Lite** | ML engineering | PiEvolve | 80.3% | 8.7% (AIDE/GPT-4o) | **Specialized** |
| **FrontierMath** | Research-level math | GPT-5.2 Thinking | 40.3% | <2% (2024) | **+38pp** |
| **HLE (full)** | Expert-level QA | Claude Opus 4.6 (tools) | 53.1% | 36.7% (no tools) | **+16.4pp** |

### 3.2 Coding & Math Benchmarks: Near Saturation

| Benchmark | SOTA | Status |
|-----------|------|--------|
| GSM8K | ~99% | Saturated |
| MATH-500 | ~99% | Saturated |
| GPQA-Diamond | 94.1% | Near saturation |
| AIME 2024 | 96.7% | Near saturation |
| HumanEval | 92.7% | Near saturation |
| SWE-bench Verified | 79.2% | Active frontier |

### 3.3 New 2026 Benchmarks (No Agent OS Evaluations Yet)

| Benchmark | Venue | Domain | Challenge |
|-----------|-------|--------|-----------|
| **AstaBench** | ICLR 2026 Oral | Scientific research suite | Lit review + experiment + data analysis + discovery |
| **NewtonBench** | ICLR 2026 | Scientific law discovery | Interactive experimentation, OOD generalization |
| **EarthSE** | ICLR 2026 | Earth science | Open-ended scientific exploration |
| **Gravity-Bench** | ICML 2025 | Gravitational physics | Physics discovery from dynamic environments |
| **HeurekaBench** | ICLR 2026 | AI Co-scientist | Hypothesis generation and testing |
| **LLM-SRBench** | ICML 2025 Oral | Equation discovery | Scientific equation fitting |
| **EXP-Bench** | ICLR 2026 | AI experiment conduct | Can AI run experiments? |
| **RECODE** | ICLR 2026 | Research code | Code development with human feedback |

---

## 4. Improvement Magnitude Rankings

### 4.1 By Absolute Improvement (pp)

| Rank | Method | Benchmark | Δ | Category |
|------|--------|-----------|---|----------|
| 1 | ToT | Game of 24 | +70pp | Reasoning Scaffold |
| 2 | MemGPT | Deep retrieval | +60.5pp | Memory |
| 3 | MCTSr (8-roll) | MATH L1 | +33pp | MCTS Reasoning |
| 4 | Edison | BixBench | +29pp | Specialized Agent |
| 5 | LATS | HumanEval | +25.7pp | MCTS Agent |
| 6 | MCTSr (8-roll) | GSM8K | +22.6pp | MCTS Reasoning |
| 7 | Reflexion | AlfWorld | +22pp | Self-Reflection |
| 8 | o1 + self-debug | ScienceAgentBench | +20.6pp | TTC + Reflection |
| 9 | Reflexion | HotPotQA | +20pp | Self-Reflection |
| 10 | AGoT | GPQA-Diamond | +18.2pp | Tree Search |
| 11 | Eigen-1 | HLE Bio/Chem | +18.1pp | Multi-Agent |
| 12 | CoT/SC | GSM8K | +17.9pp | CoT |
| 13 | ACE | AppWorld | +17.0pp | Context Engineering |
| 14 | AgentFlow (7B) | Math avg | +16.4pp | Workflow+RL |
| 15 | Eigen-1 | HLE Bio/Chem (vs agent) | +13.4pp | Multi-Agent |

### 4.2 By Cost-Effectiveness (Best Value)

| Rank | Method | Gain | Cost | Retrain? | Efficiency Metric |
|------|--------|------|------|----------|-------------------|
| 1 | AFlow | Match GPT-4o | 4.55% cost | No | **22x cost reduction** |
| 2 | ACE | +17pp | No extra inference | No | **"Free" improvement** |
| 3 | A-MEM | 2x multi-hop | 1/7 tokens | No | **7x token reduction** |
| 4 | Eigen-1 | +13.4pp | -53.5% tokens | No | **Improvement + savings** |
| 5 | DeepSearch | 62.95% SOTA | +50 RLVR steps | Yes | **5.7x fewer GPU hrs** |
| 6 | Anti-early-stop | +11.2pp | Prompt change only | No | **Near-zero cost** |

### 4.3 By Scientific Relevance

| Rank | Method | Benchmark | Δ | Why It Matters |
|------|--------|-----------|---|----------------|
| 1 | Edison | BixBench | +29pp | Highest gain on scientific benchmark |
| 2 | o1 + self-debug | ScienceAgentBench | +20.6pp | Data-driven discovery |
| 3 | Anti-early-stop | PaperBench | +20.3pp | Paper replication (vs GPT-4o) |
| 4 | Eigen-1 | HLE Bio/Chem | +18.1pp | Multi-agent scientific reasoning |
| 5 | AgentFlow | Scientific tasks | +18.0pp (vs GPT-4o) | 7B beats GPT-4o on science |
| 6 | Asta v0 | AstaBench | +9.7pp | Full research pipeline |
| 7 | AgentFlow | Scientific tasks | +4.1% (vs GPT-4o direct) | Smallest gain among task types |

---

## 5. Key Patterns

### Pattern 1: Scaffold Design Dominates Model Choice

Multiple independent studies confirm that **how you scaffold the model matters more than which model you use**:

| Evidence | Same Model | Scaffold A vs B | Performance Gap |
|----------|-----------|----------------|-----------------|
| MLE-bench | GPT-4o | AIDE vs MLAB | **11x** (8.7% vs 0.8%) |
| BixBench | Claude 3.5 Sonnet | Edison vs direct | **2.7x** (46% vs 17%) |
| AgentFlow | Qwen-2.5-7B | Flow-GRPO vs SFT | **2.9x** (55.7% vs 19.5% avg) |
| AFlow | GPT-4o-mini | MCTS workflow vs none | **Matches GPT-4o** at 4.55% cost |
| AgentFlow | 7B model | Trained scaffold | **Beats GPT-4o** across 4 categories |
| LATS | GPT-4 | MCTS vs direct | +25.7pp HumanEval |
| Eigen-1 | DeepSeek V3.1 | Multi-agent + RAG | +18.1pp over frontier LLMs |
| MCTSr | LLaMA-3 8B | 8-rollout MCTS | +22.6pp GSM8K |
| SWE-bench | Claude Opus 4.5 | Live-SWE-agent v2.0 | ~79% (scaffold matters) |

### Pattern 2: Scientific Reasoning Shows Slower Progress

While math and coding benchmarks are near saturation, scientific reasoning benchmarks remain far from solved:

| Benchmark Category | 2024 Best | 2026 Best | Annual Gain | Status |
|-------------------|-----------|-----------|-------------|--------|
| GSM8K | ~90% | ~99% | +4.5pp/yr | **Saturated** |
| MATH-500 | ~80% | ~99% | +9.5pp/yr | **Saturated** |
| AIME 2024 | ~12% | 96.7% | +42pp/yr | **Nearly saturated** |
| HumanEval | ~67% | 92.7% | +12.8pp/yr | **Nearly saturated** |
| BixBench | 17% | 46% | +29pp/yr | **Active frontier** |
| PaperBench | 4.1% | 26.6% | +22.5pp/yr | **Active frontier** |
| ScienceAgentBench | 21.6% | 42.2% | +20.6pp/yr | **Active frontier** |
| FrontierMath | <2% | 40.3% | +38pp/yr | **Active frontier** |
| HLE | ~3% | 53.1% | +50pp/yr | **Rapidly improving** |

### Pattern 3: MCTS Is the Universal Scaffold

MCTS-based methods now appear in every major category:

| Application | Method | Result |
|-------------|--------|--------|
| Reasoning | MCTSr | +22.6pp GSM8K, +33pp MATH |
| Code generation | LATS | 92.7% HumanEval |
| Tool planning | ToolTree | +10% avg across 4 benchmarks |
| Workflow search | AFlow | 4.55% cost to match GPT-4o |
| Training integration | DeepSearch | 62.95% math avg, 5.7x fewer GPU hrs |
| Adaptive exploration | AB-MCTS | Better than sampling + standard MCTS |
| Agent RL | Tree-GRPO | Outperforms chain RL with 1/4 rollout budget |

### Pattern 4: Scientific Tasks Benefit Less from Current Methods

AgentFlow results reveal a hierarchy of task difficulty:

| Task Type | AgentFlow Δ vs GPT-4o | Interpretation |
|-----------|----------------------|----------------|
| Search tasks | +14.9% | Tool + planning → high gains |
| Agentic tasks | +14.0% | Multi-turn execution → high gains |
| Math tasks | +14.5% | Structured reasoning → high gains |
| **Scientific tasks** | **+4.1%** | **Domain knowledge needed → low gains** |

This gap suggests scientific reasoning requires capabilities beyond what current Agent OS methods optimize — likely domain-specific knowledge retrieval and structured scientific memory.

---

## 6. Gap Analysis

### Gap 1: Agent OS Methods Untested on Scientific Benchmarks ★★★★★

Of 14 major Agent OS methods surveyed, **only 4** have been evaluated on scientific reasoning benchmarks:

| Method | Category | Tested on Science? | Best Non-Science Result | Expected Impact |
|--------|----------|-------------------|------------------------|-----------------|
| ACE | Context Eng. | ❌ | +17pp AppWorld | High |
| PAACE | Context Compress. | ❌ | 59.0% AppWorld SOTA | High |
| MemoryOS | Memory | ❌ | +49.11% F1 LoCoMo | Very High |
| A-MEM | Memory | ❌ | 2x multi-hop QA | High |
| AFlow | MCTS Workflow | ❌ | +5.7% avg 6 benchmarks | High |
| LATS | MCTS Agent | ❌ | 92.7% HumanEval | Medium |
| ToolTree | MCTS Tool | ❌ | +10% avg 4 benchmarks | High |
| AB-MCTS | Adaptive MCTS | ❌ | > sampling + MCTS | Medium |
| Reflexion | Self-Reflection | ❌ | +11pp HumanEval | High |
| Snell TTC | Compute Scaling | ❌ | 4x efficiency MATH | Medium |
| Eigen-1 | Multi-Agent | ✅ | +13.4pp HLE Bio/Chem | **Demonstrated** |
| Edison | Specialized | ✅ | +29pp BixBench | **Demonstrated** |
| Self-debug | Scaffold | ✅ | +10pp ScienceAgentBench | **Demonstrated** |
| Anti-early-stop | Context Eng. | ✅ | +11.2pp PaperBench | **Demonstrated** |

### Gap 2: No Unified Agent OS Stack for Science ★★★★★

Each method operates in isolation. No study combines:
- Context engineering (ACE) + Memory management (MemoryOS) + Workflow optimization (AFlow) + Multi-agent coordination (Eigen-1)

Proposed unified stack:
```
Scientific Agent OS Stack:
├── Layer 5: AFlow workflow optimizer → discovers optimal experimental workflows
├── Layer 4: ACE context engineer → accumulates domain-specific playbooks
├── Layer 3: MemoryOS memory manager → retains cross-experiment knowledge
├── Layer 2: Eigen-1 multi-agent → structured refinement + verification
└── Layer 1: AB-MCTS adaptive search → optimal compute allocation
```

### Gap 3: New 2026 Benchmarks Lack Agent OS Evaluation ★★★★

8+ new scientific benchmarks from ICLR 2026 have zero Agent OS method evaluations:
AstaBench, NewtonBench, EarthSE, Gravity-Bench, HeurekaBench, LLM-SRBench, EXP-Bench, RECODE.

### Gap 4: Memory Management for Scientific Knowledge ★★★★

Current memory systems (MemoryOS, A-MEM, Mem0) designed for conversational memory, not structured scientific knowledge requiring:
- Formula/equation storage and retrieval
- Experimental parameter tracking across trials
- Statistical result accumulation
- Cross-paper knowledge synthesis

### Gap 5: Context Engineering for Domain Experts ★★★★

ACE creates evolving playbooks from task experience, but has not been tested on domain-specific scientific tasks. Scientific reasoning requires specialized context (chemistry formulas, biology protocols, physics laws).

### Gap 6: Test-Time Compute for Multi-Step Science ★★★

TTC scaling studied mainly on single-answer math/QA. Scientific tasks involve multi-step experimental design, data collection + analysis pipelines, and iterative hypothesis refinement over 10–50+ steps.

### Gap 7: No Multi-Dimensional Evaluation Framework ★★★

No benchmark simultaneously measures: task accuracy, context efficiency, memory utilization, coordination overhead, inference cost, and process quality.

---

## 7. Research Directions & Prioritization

### Direction 1: ACE + MemoryOS for ScienceAgentBench ★★★★★ (P1)
- **Rationale**: ACE alone gives +17pp on AppWorld. MemoryOS gives +49% F1 on memory tasks. ScienceAgentBench requires both context management and knowledge retention. Current SOTA is only 42.2%.
- **Predicted impact**: +10–20pp improvement over current SOTA
- **Feasibility**: Both have open-source implementations. ScienceAgentBench is publicly available.

### Direction 2: MCTS Workflow for Scientific Experiments ★★★★★ (P1)
- **Rationale**: AFlow discovers optimal workflows automatically. Scientific experiments have well-defined step structures. No one has applied MCTS workflow search to ScienceAgentBench or PaperBench.
- **Predicted impact**: +5–15pp
- **Feasibility**: AFlow code is available (436 stars, MIT license). Need to adapt workflow space definition.

### Direction 3: Unified Scientific Agent OS ★★★★★ (P1, long-term)
- **Rationale**: The biggest gap. No study combines context engineering + memory + workflow + multi-agent for science.
- **Predicted impact**: Potentially +20–30pp over direct prompting (based on composition of individual improvements)
- **Feasibility**: Harder — requires integrating multiple systems. But each component exists.

### Direction 4: First Agent OS Results on 2026 Benchmarks ★★★★ (P1, quick win)
- **Rationale**: NewtonBench, AstaBench, Gravity-Bench have no Agent OS evaluations. Being first gives automatic novelty.
- **Predicted impact**: Unknown — brand new benchmarks
- **Feasibility**: High — benchmarks are public, Agent OS methods are available.

### Direction 5: Scientific Memory System ★★★★ (P2)
- **Rationale**: No memory system designed for scientific knowledge (equations, experimental parameters, statistical results).
- **Predicted impact**: Could transform multi-step scientific reasoning.
- **Feasibility**: Medium — requires domain-specific design.

### Direction 6: Process-Aware Evaluation ★★★ (P2)
- **Rationale**: Current evaluation only measures outcomes, not process quality.
- **Predicted impact**: Could change how we evaluate scientific agents.
- **Feasibility**: High — can extend frameworks like TRAJECT-Bench.

### Prioritization Matrix

| Direction | Impact | Novelty | Feasibility | Priority |
|-----------|--------|---------|-------------|----------|
| ACE+MemoryOS for ScienceAgentBench | ★★★★★ | ★★★★ | ★★★★ | **P1** |
| MCTS Workflow for Science | ★★★★★ | ★★★★★ | ★★★★ | **P1** |
| Unified Scientific Agent OS | ★★★★★ | ★★★★★ | ★★★ | **P1 (long-term)** |
| First results on 2026 benchmarks | ★★★★ | ★★★★★ | ★★★★★ | **P1 (quick win)** |
| Scientific Memory System | ★★★★ | ★★★★★ | ★★★ | **P2** |
| Process-Aware Evaluation | ★★★ | ★★★★ | ★★★★★ | **P2** |

---

## 8. Open-Source Implementations

### Tier 1: Must-Integrate (Scientific Reasoning Focus)

| Repo | Stars | License | Conference | Relevance |
|------|-------|---------|------------|-----------|
| [Eigen-1](https://github.com/tangxiangru/Eigen-1) | 89 | — | ICLR 2026 | Scientific reasoning SOTA |
| [AstaBench](https://github.com/allenai/asta-bench) | 76 | Apache-2.0 | ICLR 2026 Oral | Most comprehensive scientific benchmark |
| [PaperBench](https://github.com/openai/frontier-evals) | 1,090 | MIT | ICML 2025 | Paper replication evaluation |
| [ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench) | 124 | MIT | ICLR 2025 | Data-driven discovery |

### Tier 2: Core Components

| Repo | Stars | License | Conference | Relevance |
|------|-------|---------|------------|-----------|
| [AgentFlow](https://github.com/lupantech/AgentFlow) | 1,578 | MIT | ICLR 2026 Oral | Trainable agentic framework |
| [TreeQuest/AB-MCTS](https://github.com/SakanaAI/treequest) | 522 | Apache-2.0 | NeurIPS 2025 Spotlight | Plug-and-play tree search library |
| [Tree-GRPO](https://github.com/AMAP-ML/Tree-GRPO) | 292 | Apache-2.0 | ICLR 2026 | Efficient RL training for agents |
| [BixBench](https://github.com/Future-House/BixBench) | 72 | Apache-2.0 | — | Bioinformatics benchmark |

### Tier 3: Foundational

| Repo | Stars | License | Conference | Relevance |
|------|-------|---------|------------|-----------|
| [OpenHands](https://github.com/OpenHands/OpenHands) | 68,192 | Custom | — | Infrastructure backbone |
| [Reflexion](https://github.com/noahshinn/reflexion) | 3,074 | MIT | NeurIPS 2023 | Foundational self-reflection |
| [AIDE](https://github.com/WecoAI/aideml) | 1,140 | MIT | — | Tree-search coding agent |
| [LATS](https://github.com/lapisrocks/LanguageAgentTreeSearch) | 817 | MIT | ICML 2024 | Canonical MCTS agent |
| [AFlow](https://github.com/FoundationAgents/AFlow) | 436 | MIT | ICLR 2025 Oral | MCTS workflow search |
| [DeepSearch](https://github.com/smiles724/DeepSearch) | 24 | Apache-2.0 | ICLR 2026 | MCTS in RLVR training |

### Not Yet Released
- **ToolTree** (ICLR 2026): No public repo despite strong paper results
- **FlowSearcher** (ICLR 2026): No public repo

---

## 9. Conclusion

The Agent OS approach — improving AI performance through system design rather than model retraining — has produced dramatic improvements on coding (+25pp HumanEval), math (+22pp GSM8K), and tool-use benchmarks (+10% avg). However, **systematic application to scientific reasoning benchmarks remains the central unexplored opportunity**.

The evidence strongly suggests this gap is actionable:
1. Methods like ACE (+17pp), MemoryOS (+49% F1), and AFlow (22x cost reduction) have proven their value on non-scientific tasks
2. The few scientific evaluations (Eigen-1 +13.4pp, Edison +29pp) demonstrate Agent OS methods work for science
3. 8+ new scientific benchmarks from ICLR 2026 provide immediate evaluation targets
4. All key components have open-source implementations

The highest-impact research direction is building a **unified Scientific Agent OS** that combines context engineering, memory management, workflow optimization, and multi-agent coordination — evaluated on the new generation of scientific benchmarks. Our analysis suggests this could yield +20–30pp improvements over direct prompting on benchmarks like ScienceAgentBench, BixBench, and PaperBench.

---

## References

See `references.bib` for full bibliography (562 entries).

### Key Citations

- **AgentFlow**: Li et al., "In-The-Flow Agentic System Optimization for Effective Planning and Tool Use," ICLR 2026 Oral
- **Eigen-1**: Tang et al., "Scientific Reasoning through Adaptive Multi-Agent Refinement and Monitor-based RAG," ICLR 2026
- **AFlow**: — , "AFlow: Automating Agentic Workflow Generation," ICLR 2025 Oral
- **LATS**: Zhou et al., "Language Agent Tree Search Unifies Reasoning, Acting, and Planning," ICML 2024
- **ACE**: Kapoor et al., "ACE: Accumulative Context Engineering," ICLR 2026
- **MemoryOS**: — , "MemoryOS: Hierarchical Memory Management for LLM Agents," EMNLP 2025 Oral
- **ToolTree**: — , "ToolTree: Efficient LLM Tool Planning via Dual-Feedback MCTS and Bidirectional Pruning," ICLR 2026
- **Reflexion**: Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning," NeurIPS 2023
- **DeepSearch**: — , "DeepSearch: Overcome the Bottleneck of RLVR via MCTS," ICLR 2026
- **AB-MCTS**: — , "Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search," NeurIPS 2025 Spotlight
- **Snell et al.**: Snell et al., "Scaling LLM Test-Time Compute Optimally," ICLR 2025 Oral
- **AstaBench**: Allen AI, "AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite," ICLR 2026 Oral
- **PaperBench**: OpenAI, "Evaluating AI's Ability to Replicate AI Research," ICML 2025
- **NewtonBench**: — , "Benchmarking Generalizable Scientific Law Discovery in LLM Agents," ICLR 2026
- **Edison**: FutureHouse, "Edison: A Specialized Agent for Computational Biology," proprietary
- **ScienceAgentBench**: OSU-NLP, "ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery," ICLR 2025
- **AgentFlow (Flow-GRPO)**: Li et al., ICLR 2026 Oral — 7B backbone outperforms GPT-4o
- **MCTSr**: — , "Monte Carlo Tree Self-Refine for LLM Math Reasoning," preprint
- **MemGPT**: Packer et al., "MemGPT: Towards LLMs as Operating Systems," ICLR 2024
- **A-MEM**: — , "A-MEM: Agentic Memory for LLM Agents," NeurIPS 2025
- **ScoreFlow**: — , "ScoreFlow: Score-Based Optimization of AFlow," 2025
- **Tree-GRPO**: — , "Tree-Search Rollouts for LLM Agent RL," ICLR 2026
- **Elastic Reasoning**: — , "Scalable Chain-of-Thought via Elastic Reasoning," ICLR 2026
- **Self-Consistency**: Wang et al., "Self-Consistency Improves Chain of Thought Reasoning," ICLR 2023
- **CoT**: Wei et al., "Chain-of-Thought Prompting Elicits Reasoning," NeurIPS 2022
