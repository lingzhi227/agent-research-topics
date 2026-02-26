# LLM Agent Skills: Generation, Management, and Integration with Agent-OS

## A Deep Research Report (2025 July – 2026 February)

**Generated**: 2026-02-26
**Scope**: Agent skill generation, management, reasoning enhancement, token/context optimization, skill suites, Agent-OS integration
**Papers surveyed**: 1,319 (database), 62 curated, 12 deep-read
**Focus**: Post-July 2025 research frontier

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Skill Generation Methods](#2-skill-generation-methods)
3. [Skill Management & Organization](#3-skill-management--organization)
4. [Memory Operating Systems](#4-memory-operating-systems)
5. [Context Engineering & Token Efficiency](#5-context-engineering--token-efficiency)
6. [Enhancing LLM Reasoning via Skills](#6-enhancing-llm-reasoning-via-skills)
7. [Agent Communication Protocols](#7-agent-communication-protocols)
8. [Integrating Skills with Agent-OS](#8-integrating-skills-with-agent-os)
9. [Domain Applications](#9-domain-applications)
10. [Research Gaps & Future Directions](#10-research-gaps--future-directions)
11. [References](#11-references)

---

## 1. Executive Summary

The field of LLM agent skills has undergone a paradigm shift since mid-2025. **The monolithic agent is dead; the modular, skill-equipped agent is the new norm.** This report identifies seven major research threads and their convergence toward an integrated Agent-OS architecture.

### Key Findings

1. **RL-based skill generation** (SAGE, SkillRL) outperforms prompting-based approaches by 3× in task completion while reducing token usage by up to 59%
2. **Skills are a form of compression** — ProcMEM achieves 98% memory compression by distilling trajectories into reusable procedures
3. **Memory OS design** (MemOS, EverMemOS) treats agent memory as a first-class OS resource with lifecycle management
4. **Context-Folding** enables 10× context reduction via RL-trained branch/fold/return mechanisms
5. **Phase transition** in skill selection: libraries cannot grow indefinitely without hierarchical organization
6. **MCP + A2A** emerge as the de facto protocol stack for agent-tool and agent-agent communication
7. **Small models (8B-30B) benefit most** from skill paradigms, approaching large model performance
8. **26.1% of community skills contain vulnerabilities** — security is a critical concern

### The Agent-OS Vision

```
┌─────────────────────────────────────────────────┐
│                   Agent OS                       │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │
│  │ Skill  │ │ Memory │ │Context │ │Protocol│   │
│  │Manager │ │Manager │ │Manager │ │ Stack  │   │
│  └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘   │
│       └──────────┼──────────┼──────────┘        │
│           ┌──────▼──────────▼──────┐            │
│           │  Unified RL Runtime     │            │
│           │  (GRPO-based training)  │            │
│           └─────────────────────────┘            │
└─────────────────────────────────────────────────┘
```

---

## 2. Skill Generation Methods

### 2.1 The Evolution from Prompting to RL

The skill generation paradigm has evolved through three waves:

| Wave | Era | Method | Example | Limitation |
|------|-----|--------|---------|------------|
| **Prompt-based** | 2023-24 | LLM generates code/instructions | VOYAGER | Inconsistent quality; no feedback signal |
| **SFT-based** | 2024-25 | Fine-tune on expert demonstrations | CycleQD | Static; no self-improvement |
| **RL-based** | 2025-26 | Train with outcome rewards | SAGE, SkillRL | State of the art |

**The critical evidence**: ARTIST ablation shows prompt-based tool access yields only +4.7% improvement, while RL training yields +22% on the same benchmark. RL is not optional — it's essential for learning when/how to use skills.

### 2.2 SAGE: The First RL Skill Library

SAGE [@wang2025sage] introduces two key innovations:

1. **Sequential Rollout**: Deploys agents across task chains where skills from task N accumulate for task N+1
2. **Skill-integrated Reward**: Rewards skill generation quality, not just task outcome

Results on AppWorld: +8.9% goal completion, -26% interaction steps, **-59% tokens**

The reward function:
```
R = r_outcome + 𝟙[r₁=1] · 𝟙[r₂=1] · 𝟙_skill(q₂|q₁)
```
This jointly rewards task success AND downstream skill utility.

### 2.3 SkillRL: Recursive Co-Evolution

SkillRL [@xia2026skillrl] advances SAGE with:

1. **SkillBank**: Two-tier hierarchy — general skills (always loaded) + task-specific (retrieved by similarity)
2. **Experience distillation**: Success → generalizable patterns; Failure → counterfactual lessons (10-20× compression)
3. **Recursive evolution**: After each validation epoch, analyze failures → generate new skills → update SkillBank

Critical finding: **Raw memory injection harms RL** (Mem0+GRPO: 54.7% vs pure GRPO: 77.6%). Abstraction is the key differentiator.

### 2.4 ProcMEM: Non-Parametric Skill Learning

ProcMEM [@mi2026procmem] formalizes skills mathematically:

**Skill-MDP**: Each skill is a triple (activation condition, execution process, termination condition)

**Non-Parametric PPO**: Uses "semantic gradients" (natural language feedback) instead of weight updates, with a **PPO Gate** that validates skills before storage.

Achieves **98% memory compression** (816 vs 40,510 tokens) with maintained performance.

### 2.5 FLEX: Experience Inheritance

FLEX [@cai2025flex] discovers a **scaling law of experiential growth**: performance follows a power-law relationship with experience volume.

Breakthrough: **Experience inherits across agents** — Claude-Sonnet-4.5 experience transferred to Gemini-2.5-Pro yields +11pp improvement. Even weak→strong transfer works (+6.7pp).

Three-tier experience library: Principles (abstract) → Patterns (templates) → Facts (specific)

---

## 3. Skill Management & Organization

### 3.1 The SKILL.md Specification

The Agent Skills Survey [@xu2026survey] proposes SKILL.md as a standard format:

```yaml
---
name: protein-blast-analysis
version: 1.2.0
triggers: ["blast", "sequence alignment", "protein search"]
tools: [blast+, samtools]
permissions: [file_read, network_access]
---

# Protein BLAST Analysis

## Instructions
[Step-by-step procedure...]

## Resources
[Reference documentation, example outputs...]
```

Three-layer progressive disclosure: **Metadata** (loaded always) → **Instructions** (loaded on activation) → **Resources** (loaded on demand). This minimizes context window consumption.

### 3.2 Phase Transition in Skill Selection

Li [@li2026single] discovers a **fundamental scaling limit**: skill selection accuracy remains stable until a critical library size κ, then drops sharply (non-linear phase transition).

| Model | Critical Threshold κ |
|-------|---------------------|
| GPT-4o-mini | ~91.8 skills |
| GPT-4o | ~83.5 skills |
| Note: larger model → LOWER threshold |

Root cause: **Semantic similarity between skills**, not raw count. Two similar skills create disambiguation confusion.

Solution: **Hierarchical organization** (+37-40% accuracy recovery via two-stage domain→skill selection).

### 3.3 Security Concerns

Analysis of 42,447 community skills reveals [@xu2026survey]:
- **26.1% contain vulnerabilities** (14 vulnerability patterns identified)
- **157 confirmed malicious skills** with 632 total vulnerabilities
- Script-type skills have **2.12× higher risk** than instruction-only skills

Proposed: **4-tier governance framework**
- G1: Static analysis → G2: LLM semantic classification → G3: Sandboxed execution → G4: Permission manifest verification
- T1 (untrusted community) → T2 (scanned) → T3 (vendor-certified) → T4 (platform-verified)

---

## 4. Memory Operating Systems

### 4.1 MemOS: The Reference Architecture

MemOS [@li2025memos] establishes the memory-as-OS paradigm:

**Three-layer architecture:**
1. **Interface Layer**: MemReader, Memory API (provenance, update, query)
2. **Operation Layer**: MemScheduler, MemLifecycle, MemOperator
3. **Infrastructure Layer**: MemGovernance, MemVault, MemStore

**MemCube** — the fundamental abstraction, packaging memory content + metadata:
- Descriptive identifiers (timestamp, origin, semantic type)
- Governance attributes (ACL, TTL, priority)
- Behavioral indicators (access patterns, version chain)

**Five-stage lifecycle**: Generated → Activated → Merged → Archived → Expired

Benchmarks: LoCoMo 75.8 (SOTA), LongMemEval +40%, 35% token savings.

### 4.2 Memory Taxonomy

The Memory Survey [@hu2025memory] provides the definitive taxonomy:

| Dimension | Categories | Description |
|-----------|-----------|-------------|
| **Forms** | Token-level, Parametric, Latent | How memory is stored |
| **Functions** | Factual, Experiential, Working | What memory is used for |
| **Dynamics** | Formation, Evolution, Retrieval | How memory changes over time |

**Skills map to experiential memory** — distilled procedural knowledge from past interactions.

### 4.3 DeepSeek Engram: Architecture-Level Memory

DeepSeek Engram introduces a **conditional memory axis** for sparse LLMs:

**Sparsity Allocation Law**: 75% of sparse parameters → computation ("Brain"), 25% → memory ("Book")

O(1) lookup for static patterns, freeing compute for dynamic reasoning. NIAH: 84% → 97%.

---

## 5. Context Engineering & Token Efficiency

### 5.1 Context-Folding: Active Context Management

Context-Folding [@sun2025folding] introduces three operations:
- **branch(description, prompt)**: Create sub-trajectory for subtask
- **fold**: Collapse intermediate steps upon completion
- **return(message)**: Provide concise summary to main trajectory

**FoldGRPO** training with three process rewards:
1. Unfolded Token Penalty (Q = -1 when context exceeds 50% of limit)
2. Out-of-Scope Penalty (Q = -0.2 for off-topic branch actions)
3. Failure Penalty (Q = -1 for failed tools, +0.2 for successful)

Result: **10× smaller active context** while matching/exceeding baseline performance.

### 5.2 ACON: Compression Guideline Optimization

ACON [@kang2025acon] takes a different approach — using LLMs to optimize compression guidelines in natural language:

1. Collect paired trajectories (full context succeeds, compressed fails)
2. LLM analyzes failure causes
3. Update compression guidelines accordingly
4. Distill into smaller models (95%+ accuracy retention)

26-54% peak token reduction across AppWorld, OfficeBench, Multi-objective QA.

### 5.3 Skills as Compression

A meta-finding: **skills and context compression serve the same purpose**:

| System | Compression Mechanism | Reduction |
|--------|----------------------|-----------|
| SAGE | Skill reuse vs re-derivation | -59% tokens |
| ProcMEM | Trajectory → procedure distillation | -98% (816 vs 40,510 tokens) |
| SkillRL | Experience → abstract heuristics | 10-20× |
| Context-Folding | Branch/fold/return | 10× context |
| ACON | Optimized guidelines | 26-54% |

**Implication for Agent-OS**: The context manager and skill manager should be co-designed, as both optimize the same resource (context window).

---

## 6. Enhancing LLM Reasoning via Skills

### 6.1 ARTIST: RL for Tool Use

ARTIST [@singh2025artist] proves that **RL is essential for learning tool use**:

| Approach | MATH-500 | AIME | AMC |
|----------|----------|------|-----|
| Base model | 62% | 4% | 35% |
| + Prompt tool access | 66.7% (+4.7%) | - | - |
| + ARTIST (RL) | 67.6% (+5.6%) | 15.6% (+11.6%) | 47% (+12%) |

Key finding: prompt-based tool access yields minimal gains; RL training is essential.

### 6.2 Emergent Capabilities

RL-trained agents develop capabilities without explicit supervision:
- **Self-refinement**: Iterative strategy adjustment
- **Self-correction**: Diagnosis and recovery from tool failures
- **Self-reflection**: Validation through repeated computation

### 6.3 Math + Science Skills

FLEX demonstrates skill-based reasoning in scientific domains:
- AIME25: Claude-Sonnet-4 40.0% → 63.3% (+23.3pp)
- USPTO50k (chemistry): 20% → 30%
- ProteinGym (biology): Spearman's ρ 0.460 → 0.597

---

## 7. Agent Communication Protocols

### 7.1 The Protocol Stack

| Layer | Protocol | Function | Status |
|-------|----------|----------|--------|
| 1 | **MCP** | Agent ↔ Tool invocation | Universal (2025) |
| 2 | **ACP** | Structured messaging | Emerging |
| 3 | **A2A** | Agent ↔ Agent collaboration | 50+ backers |
| 4 | **ANP** | Decentralized marketplace | Research |

**MCP** (Anthropic, 2024): JSON-RPC interface for secure tool invocation. Near-universal adoption by end of 2025.

**A2A** (Google, Apr 2025): Agent Cards for capability-based discovery; peer-to-peer task delegation.

**Complementarity**: MCP provides tools (like a plugin system), A2A enables collaboration (like a network). Both are needed.

### 7.2 Skills in the Protocol Stack

Skills bridge MCP and A2A:
- Individual skills expose tools via **MCP** (e.g., "protein-blast" skill registers as MCP tool)
- Skill suites compose via **A2A** (e.g., bioinformatics pipeline agent delegates to BLAST agent, then DESeq2 agent)

---

## 8. Integrating Skills with Agent-OS

### 8.1 The Agent-OS Architecture

Based on the research synthesis, the ideal Agent-OS comprises:

```
┌───────────────────────────────────────────────────────┐
│                    Agent OS                             │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐   │
│  │   Skill     │  │   Memory    │  │   Context    │   │
│  │   Manager   │  │   Manager   │  │   Manager    │   │
│  │             │  │             │  │              │   │
│  │ • Generation│  │ • MemCube   │  │ • Folding    │   │
│  │ • SkillBank │  │ • Lifecycle │  │ • Compression│   │
│  │ • Retrieval │  │ • Retrieval │  │ • Budget     │   │
│  │ • Evolution │  │ • Evolution │  │ • Allocation │   │
│  │ • Security  │  │ • GC        │  │              │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘   │
│         │                │                │             │
│  ┌──────▼────────────────▼────────────────▼──────────┐ │
│  │            Session Manager                         │ │
│  │  • Working memory (current task state)             │ │
│  │  • Long-term memory (persistent knowledge)         │ │
│  │  • Episodic buffer (interaction history)            │ │
│  └────────────────────┬──────────────────────────────┘ │
│                       │                                 │
│  ┌────────────────────▼──────────────────────────────┐ │
│  │            Agent Coordinator                       │ │
│  │  • A2A for inter-agent communication               │ │
│  │  • Task delegation and result aggregation          │ │
│  │  • Shared skill marketplace                        │ │
│  └────────────────────┬──────────────────────────────┘ │
│                       │                                 │
│  ┌────────────────────▼──────────────────────────────┐ │
│  │            Protocol Stack                          │ │
│  │  • MCP: tool invocation                           │ │
│  │  • A2A: agent collaboration                       │ │
│  │  • ACP: structured messaging                      │ │
│  └───────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
```

### 8.2 Key Design Principles (from Research)

1. **Skills as MemCubes**: Store skills using MemOS's MemCube abstraction with metadata, versioning, governance
2. **Hierarchical SkillBank**: Two-tier (general + task-specific) to avoid phase transition
3. **Joint context-skill optimization**: Context Manager and Skill Manager share budget — compress more when skills can compensate
4. **RL-trained runtime**: GRPO for skill generation, FoldGRPO for context management, outcome-based rewards
5. **Non-parametric updates**: Skills evolve without weight changes (ProcMEM approach), enabling safe online learning
6. **Protocol-native**: Skills expose via MCP, compose via A2A
7. **Security-by-default**: 4-tier governance for all skills

### 8.3 Dynamic Notebook Memory (Mapping to Research)

The "dynamic notebook memory" concept maps to:
- **Working memory**: Current task context (Context-Folding's active window)
- **Short-term memory**: Session-level episodic buffer (MemRL's episodic memory)
- **Long-term memory**: Persistent skill library + factual memory (SkillBank + MemOS)
- **Dynamic**: Skills evolve (SkillRL recursive evolution), memories consolidate (EverMemOS MemScenes), outdated entries are pruned (ProcMEM score-based maintenance)

---

## 9. Domain Applications

### 9.1 Bioinformatics

| System | Type | Key Capability |
|--------|------|----------------|
| BioAgents (Nature, 2025) | Multi-agent SLMs + RAG | Local, privacy-preserving bio analysis |
| BioMANIA | API integration | Automates bio workflows via LLM |
| BIA | Proteomic analysis | Non-expert accessible |

**Gap**: No formal skill library for bio tools. Opportunity: SKILL.md-wrapped bioinformatics tools with validation.

### 9.2 Mathematics

ARTIST demonstrates RL-trained math tool use (+22% on challenging benchmarks). FLEX achieves +23.3pp on AIME25 via experience distillation. SymPy/NumPy integration is key.

### 9.3 Python & Code

SWE-Bench agents (Context-Folding: 58% verified) show skill-based code agents are production-ready. SAGE and AppWorld demonstrate multi-step code generation with tool calls.

### 9.4 Science (Chemistry, Proteins)

FLEX: USPTO50k retrosynthesis +10pp, ProteinGym fitness prediction +14pp. Experience from one scientific domain transfers to others.

---

## 10. Research Gaps & Future Directions

### Top 5 Open Problems

1. **Unified Skill-Memory Framework**: Skills should be first-class objects in Memory OS
2. **Joint Context-Skill Optimization**: Context compression and skill generation should be co-optimized
3. **Skill Composition Semantics**: Formal framework for composing skills into workflows
4. **Skill Selection at Scale**: MoE-style routing for 10K+ skill libraries
5. **Cross-Platform Portability**: Standard skill format working across Claude, GPT, Gemini, etc.

### Emerging Research Frontiers

- **Recursive Language Models** (Prime Intellect, 2026): LLMs that recursively manage their own context
- **MemAgents Workshop** (ICLR 2026): Dedicated venue for agent memory research
- **Lifelong Agents Workshop** (ICLR 2026): Learning, aligning, evolving agents
- **Agent Protocol Stack**: MCP + A2A + A2UI as "TCP/IP for agents"

---

## 11. References

### Core Papers (Deep-Read)

[@xu2026survey] Xu, R. & Yan, Y. (2026). Agent Skills for Large Language Models: Architecture, Acquisition, Security, and the Path Forward. arXiv:2602.12430

[@wang2025sage] Wang, J. et al. (2025). Reinforcement Learning for Self-Improving Agent with Skill Library. arXiv:2512.17102

[@xia2026skillrl] Xia, P. et al. (2026). SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning. arXiv:2602.08234

[@mi2026procmem] Mi, Q. et al. (2026). ProcMEM: Learning Reusable Procedural Memory from Experience via Non-Parametric PPO for LLM Agents. arXiv:2602.01869

[@cai2025flex] Cai, Z. et al. (2025). FLEX: Continuous Agent Evolution via Forward Learning from Experience. arXiv:2511.06449

[@li2025memos] Li, Z. et al. (2025). MemOS: A Memory OS for AI System. arXiv:2507.03724

[@hu2025memory] Hu, Y. et al. (2025). Memory in the Age of AI Agents. arXiv:2512.13564

[@sun2025folding] Sun, W. et al. (2025). Scaling Long-Horizon LLM Agent via Context-Folding. arXiv:2510.11967

[@kang2025acon] Kang, M. et al. (2025). ACON: Optimizing Context Compression for Long-horizon LLM Agents. arXiv:2510.00615

[@singh2025artist] Singh, J. et al. (2025). Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning. arXiv:2505.01441

[@li2026single] Li, X. (2026). When Single-Agent with Skills Replace Multi-Agent Systems and When They Fail. arXiv:2601.04748

[@ehtesham2025protocols] Ehtesham, A. et al. (2025). A Survey of Agent Interoperability Protocols. arXiv:2505.02279

### Additional Key References

[@zhang2025ace] Zhang, Q. et al. (2025). Agentic Context Engineering. arXiv:2510.04618

[@hu2026evermemos] Hu, C. et al. (2026). EverMemOS. arXiv:2601.02163

[@magma2026] MAGMA: Multi-Graph based Agentic Memory Architecture. arXiv:2601.03236

[@zhang2026memrl] Zhang, S. et al. (2026). MemRL. arXiv:2601.03192

[@xu2025amem] Xu, W. et al. (2025). A-MEM: Agentic Memory for LLM Agents. arXiv:2502.12110 (NeurIPS 2025)

[@carta2025herakles] Carta, T. et al. (2025). HERAKLES. arXiv:2508.14751

[@fang2025memp] Fang, R. et al. (2025). Mem^p: Exploring Agent Procedural Memory. arXiv:2508.06433

[@cao2025reme] Cao, Z. et al. (2025). Remember Me, Refine Me. arXiv:2512.10696

[@forouzandeh2025macla] Forouzandeh, S. et al. (2025). MACLA. arXiv:2512.18950 (AAMAS 2026)

[@han2025legomem] Han, D. et al. (2025). LEGOMem. arXiv:2510.04851

[@li2025hep] Li, S. et al. (2025). H-EPM: Hybrid Episodic-Procedural Memory. arXiv:2512.07287

[@liu2026generalization] Liu, Z. et al. (2026). Paying Less Generalization Tax. arXiv:2601.18217

[@jiang2026anatomy] Jiang, D. et al. (2026). Anatomy of Agentic Memory. arXiv:2602.19320

[@xu2026slm] Xu, Y. et al. (2026). Agent Skill Framework for SLMs. arXiv:2602.16653

[@wang2025text2mem] Wang, F. et al. (2025). Text2Mem. arXiv:2509.11145

[@ren2025anemoi] Ren, X. et al. (2025). Anemoi. arXiv:2508.17068

### Domain-Specific

[@bioagents2025] BioAgents. Nature Scientific Reports, 2025.

[@biomania2025] BioMANIA. Briefings in Bioinformatics, 2025.

### Protocols & Standards

[MCP] Model Context Protocol. Anthropic, 2024. https://modelcontextprotocol.io

[A2A] Agent2Agent Protocol. Google, 2025. https://github.com/a2aproject/A2A

---

*Report generated through 6-phase deep research methodology. Full paper database (1,319 papers) and detailed notes available in the output directory.*
