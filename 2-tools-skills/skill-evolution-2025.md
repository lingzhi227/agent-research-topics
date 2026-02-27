# Agentic Skill Evolution: A Focused Survey

## Abstract

This report surveys the emerging field of **agentic skill evolution** — how autonomous agents grow, refine, compose, and transfer external skill libraries without modifying the LLM's weights. Distinguished from RL-based skill acquisition (which trains the model itself), agentic skill evolution treats skills as **external artifacts** (code, procedures, knowledge) that evolve through interaction. We analyzed 104 papers (15 deeply), identifying 5 evolution mechanisms, 6 skill representations, and 10 critical research gaps. The field has undergone rapid growth from Voyager (2023) to a diverse ecosystem including tool creation (KTCE, ToolMaker), programmatic skill networks (PSN), polymorphic abstractions (PolySkill), and open-ended self-evolution (Darwin Godel Machine). Key findings: (1) self-reflection outperforms external search for skill improvement; (2) failure-driven evolution is more effective than success accumulation; (3) cross-agent skill transfer works surprisingly well; (4) evolution reliably erodes safety alignment without safeguards.

---

## 1. Introduction

### 1.1 Scope and Distinction

**Agentic Skill Evolution** (this survey):
- Skills stored **externally** (code libraries, memory banks, tool collections)
- No weight updates to the LLM
- Growth through reflection, abstraction, experience accumulation
- Skills transferable across models and domains

**RL-Based Skill Acquisition** (excluded):
- Model weights updated via GRPO/PPO/DPO (SAGE, SkillRL, ARTIST)
- Skills are **internal** to the model
- Requires GPU training infrastructure

### 1.2 Historical Context

The field originates from three threads:
1. **Options framework** (Sutton et al., 1999) — temporal abstractions in RL
2. **Program synthesis** (Liang et al., 2010) — learning reusable program components
3. **LLM-as-agent** (Yao et al., 2023, ReAct) — LLMs as autonomous actors

**Voyager** (Wang et al., 2023) unified these into the modern paradigm: an LLM agent that writes executable code skills, stores them in a library, and retrieves them for future tasks.

---

## 2. Taxonomy

### 2.1 Five Evolution Mechanisms

| Mechanism | Key Papers | Description |
|-----------|-----------|-------------|
| **Accumulation** | Voyager, ODYSSEY, FLEX, OS-Copilot | Skills added after successful execution; simplest mechanism |
| **Refinement** | MemSkill, PSN, SkillWeaver, LearnAct | Existing skills improved through failure analysis or testing |
| **Evolutionary Search** | KTCE, CycleQD, DGM | Population-based optimization with selection, mutation, crossover |
| **Compilation** | HERAKLES, PSN | High-level reasoning compiled into efficient low-level execution |
| **Polymorphic Abstraction** | PolySkill | Skills structured as abstract interfaces with context-specific implementations |

### 2.2 Six Skill Representations

| Type | Example | Portability | Precision | Composability |
|------|---------|-------------|-----------|---------------|
| Code (Python/JS) | SkillWeaver, Voyager | Medium | High | High |
| Programmatic DAG | PSN | Low | High | Very High |
| Neural Policy | HERAKLES | Low | High | Low |
| Natural Language | FLEX, MemSkill | Very High | Low | Medium |
| OOP Hierarchy | PolySkill | High | High | High |
| Meta-Skill | CASCADE | N/A | N/A | N/A |

---

## 3. Core Papers by Cluster

### 3.1 Embodied Skill Library Growth

**Voyager** (NeurIPS 2023, 1299 citations) established the paradigm: automatic curriculum → code generation → skill library → retrieval. In Minecraft, it continuously discovers new skills without human intervention.

**ODYSSEY** (IJCAI 2025) extended this with a curated library of 40 primitive + 183 compositional skills, showing that structured organization + semantic retrieval achieves near-GPT-4 performance with open-source models (LLaMA-3 MineMA).

**HERAKLES** (2025) introduced **skill compilation**: the LLM decomposes goals into skill sequences, then these sequences are compiled into lightweight neural policies. This achieves quasi-linear scaling with task difficulty (vs. exponential for baselines).

**Evolving Programmatic Skill Networks (PSN)** (2026) organizes skills as a **directed acyclic graph** of executable programs. Three mechanisms — REFLECT (credit assignment), maturity-aware gating, and online restructuring — prevent catastrophic forgetting (SRR > 0.9) while enabling structured growth.

### 3.2 Tool Creation & Evolution

**LATM** (ICLR 2024) and **CREATOR** (EMNLP 2023) established tool creation: LLMs generate reusable Python functions. LATM introduced the tool-maker/tool-user division of labor.

**KTCE** (AAAI 2025) applied **evolutionary search** to tool populations. A three-layer knowledge tree generates initial tools; then selection, mutation, and crossover diversify and optimize the toolset. Results: +6-18% on math/science benchmarks.

**ToolMaker** (2025) automates the pipeline from GitHub repository → installed tool → LLM-callable function, achieving 80% success (vs. 20% for baselines).

**Self-Tooling Agent** (2025) trains via RL the **meta-decision** of when to create a new tool vs. reuse an existing one, showing this arbitration itself is a learnable skill.

### 3.3 Self-Evolving Agent Systems

**Darwin Godel Machine** (2025) is the most ambitious: an agent that rewrites its own source code, maintaining an **archive tree** of diverse variants. Open-ended exploration (not greedy optimization) drives improvement: SWE-bench 20% → 50%, but at ~$22K cost per run.

**CASCADE** (2025) evolves **meta-skills** — the ability to acquire new skills through continuous learning (web search + code extraction) and self-reflection (code introspection + debugging). Key finding: self-reflection contributes more than external search.

### 3.4 Experience-Driven Skill Evolution

**FLEX** (2025) accumulates a **3-tier experience library** (strategic principles → methodological patterns → factual instances) through forward learning (no gradient updates). Discovers power-law scaling of experience value and demonstrates cross-model inheritance (Claude → Gemini).

**MemSkill** (2026) reframes memory operations as **evolvable skills**. A Designer module periodically reviews hard cases, clusters failures, and proposes skill modifications. Removing the Designer drops performance by 33.4%.

### 3.5 GUI/Web Agent Skills

**SkillWeaver** (2025) discovers and refines web agent skills through: Skill Proposal → Skill Synthesis (trajectory → code API) → Skill Honing (auto-test generation + debugging). Cross-agent transfer: strong agent's skills boost weak agent by up to 54.3%.

**PolySkill** (ICLR 2026) introduces **polymorphic skill abstraction**: abstract classes define domain interfaces, concrete classes provide site-specific implementations. Achieves 73% skill reuse on unseen websites (vs. <31% prior).

**PAE** (ICML 2025) exploits the asymmetry that proposing/evaluating tasks is easier than executing them, creating an autonomous skill discovery loop that enables a 7B model to match 72B performance.

### 3.6 Safety of Skill Evolution

**"Your Agent May Misevolve"** (ICLR 2026) systematically studies risks across four evolution paths:
- **Model self-training**: Safety refusal drops >70%
- **Memory accumulation**: Attack success rises 0.6% → 20.6%
- **Tool evolution**: Harmful tool invocation increases
- **Workflow optimization**: Refusal rate drops 46.3% → 6.3%

Root cause: Post-training alignment (RLHF/DPO) is **shallow** and easily undermined by continued evolution.

---

## 4. Key Findings

### Finding 1: Self-Reflection > External Search
CASCADE's ablation shows removing self-reflection hurts more than removing web search. The agent's ability to understand its own failures is more valuable than finding external examples.

### Finding 2: Failure-Driven Evolution > Success Accumulation
MemSkill's Designer (failure-driven) is the most critical component (33.4% drop without it). PSN's REFLECT mechanism localizes errors to specific subskills. Both outperform simple success-based accumulation (Voyager approach).

### Finding 3: Cross-Agent Skill Transfer Works
SkillWeaver demonstrates up to 54.3% improvement when transferring skills from a strong agent (GPT-4o) to a weak one. FLEX shows cross-model experience inheritance with predictable power-law scaling.

### Finding 4: Polymorphic Abstraction Solves Cross-Context Transfer
PolySkill's OOP-inspired hierarchy achieves 73% skill reuse on unseen websites, dramatically outperforming prior approaches (<31%). The abstract/concrete separation provides a principled way to generalize skills.

### Finding 5: Open-Ended > Greedy Evolution
DGM's archive tree with diverse variants outperforms greedy best-first optimization. "Stepping stones" — intermediate solutions that temporarily decrease performance — enable future breakthroughs.

### Finding 6: Evolution Erodes Safety Without Safeguards
The Misevolve paper demonstrates that all four evolution paths systematically undermine alignment. No existing system has adequate safety mechanisms for skill evolution.

### Finding 7: Structured Credit Assignment Prevents Forgetting
PSN's REFLECT mechanism (analogous to backpropagation in neural networks) achieves Skill Retention Rate > 0.9, meaning almost no catastrophic forgetting when updating the skill DAG.

---

## 5. Research Gaps

### Critical (should be addressed now)
1. **Complete Skill Lifecycle**: No paper handles creation + refinement + composition + retirement + transfer
2. **Safety-Aware Evolution**: Problem identified, no solution; need "constitutional skill evolution"
3. **Unified Skill Representation**: Every paper invents its own format; need interoperable standard

### Important (should be addressed soon)
4. **Scaling Beyond 1K Skills**: Phase transition at scale unsolved
5. **Formal Evolution Theory**: No mathematical framework for growth dynamics
6. **Real-Time Online Evolution**: All methods episodic/offline

### Emerging (future directions)
7. **Multi-Agent Skill Ecosystems**: Skill marketplace, reputation systems
8. **Skill Conflict Resolution**: When skills contradict each other
9. **Long-Horizon Skill Drift**: Skills become outdated
10. **Cross-Modality Transfer**: Code ↔ NL ↔ Policy translations

---

## 6. Research Roadmap

### Near-Term (0-6 months)
- Combine PolySkill's abstraction with SkillWeaver's test-driven refinement
- Add MemSkill-style rollback to code skill libraries
- Implement PSN's credit assignment for non-game domains

### Medium-Term (6-18 months)
- Develop safety-aware evolution with constitutional constraints
- Build unified SKILL.md format compatible with MCP tool protocol
- Scale skill retrieval to 10K+ with hierarchical indexing

### Long-Term (1-3 years)
- Formal theory of skill evolution (growth laws, convergence)
- Real-time online skill evolution with safety guarantees
- Multi-agent skill ecosystems with cross-boundary transfer
- Self-evolving agent OS with integrated skill lifecycle management

---

## 7. Conclusion

Agentic skill evolution has rapidly matured from Voyager's simple accumulation (2023) to a diverse ecosystem of approaches including evolutionary search (KTCE), programmatic networks (PSN), polymorphic abstractions (PolySkill), and open-ended self-improvement (DGM). The field's key tension is between **evolution speed** and **safety**: the same mechanisms that enable rapid improvement also erode alignment. The most promising direction combines multiple approaches: polymorphic abstractions for structure, failure-driven refinement for quality, evolutionary search for diversity, and constitutional constraints for safety.

---

## References

See `references.bib` for full bibliography. Key papers:

### Embodied Skill Libraries
- [@wang2023voyager] Voyager: An Open-Ended Embodied Agent with Large Language Models. NeurIPS 2023.
- [@liu2024odyssey] ODYSSEY: Empowering Minecraft Agents with Open-World Skills. IJCAI 2025.
- [@carta2025herakles] HERAKLES: Hierarchical Skill Compilation for Open-ended LLM Agents. 2025.
- [@shi2026psn] Evolving Programmatic Skill Networks. 2026.

### Tool Creation
- [@cai2023latm] Large Language Models as Tool Makers. ICLR 2024.
- [@qian2023creator] CREATOR: Tool Creation for Disentangling Abstract and Concrete Reasoning. EMNLP 2023.
- [@ma2025ktce] Automated Creation of Reusable and Diverse Toolsets. AAAI 2025.
- [@wolflein2025toolmaker] ToolMaker: LLM Agents Making Agent Tools. 2025.

### Self-Evolving Agents
- [@zhang2025dgm] Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents. 2025.
- [@cascade2025] CASCADE: Cumulative Agentic Skill Creation through Autonomous Development and Evolution. 2025.
- [@shao2026misevolve] Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents. ICLR 2026.

### Experience-Driven Evolution
- [@flex2025] FLEX: Continuous Agent Evolution via Forward Learning from Experience. 2025.
- [@zhang2026memskill] MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents. 2026.

### GUI/Web Agent Skills
- [@skillweaver2025] SkillWeaver: Web Agents can Self-Improve by Discovering and Honing Skills. 2025.
- [@polyskill2026] One Skill, Many Websites: Learning Generalizable Skills Through Polymorphic Abstraction. ICLR 2026.
- [@pae2025] Proposer-Agent-Evaluator: Autonomous Skill Discovery For Foundation Model Internet Agents. ICML 2025.

### Surveys
- [@fang2025survey] A Comprehensive Survey of Self-Evolving AI Agents. 2025.
- [@survey2025what] A Survey of Self-Evolving Agents: What, When, How, and Where to Evolve. 2025.
- [@zheng2025lifelong] Lifelong Learning of Large Language Model based Agents: A Roadmap. TPAMI 2026.

### Paradigm
- [@du2025rethinking] Rethinking Agent Design: From Top-Down Workflows to Bottom-Up Skill Evolution. 2025.
