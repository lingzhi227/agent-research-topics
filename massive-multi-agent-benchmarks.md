# Tasks Requiring 10s-100s of LLM Agents: A Systematic Survey of Workloads and Benchmarks

## Abstract

This report surveys task types that require tens to hundreds (or more) of LLM agents, along with their associated benchmarks. We analyze 66 papers from top venues (NeurIPS, ICLR, ICML 2024-2026) and identify three fundamental dimensions of large-scale agent workloads: (1) many unique agents in shared environments (social simulation), (2) many parallel instances of the same agent (test-time scaling, batch evaluation), and (3) many concurrent agent programs (infrastructure serving). We find that only social simulation genuinely requires 100+ unique, coordinating agents; other multi-GPU scenarios involve parallel copies or concurrent independent programs. We catalog existing benchmarks, identify critical gaps—most notably the absence of a unified agent-workload inference benchmark—and connect to the companion research on scientific/technical multi-GPU agent tasks.

---

## 1. Introduction

The rapid adoption of LLM-based agents has raised a fundamental question: **which tasks actually require tens to hundreds of agents, and how should we benchmark them?**

Prior work at the intersection of multi-agent systems and LLM inference has grown rapidly, but the landscape is fragmented. Social simulation papers (OASIS, AgentSociety) focus on emergent behavior with millions of agents. Multi-agent collaboration papers (MARBLE, SwarmBench) study small groups of 3-16 agents. Test-time compute scaling papers (BoN, PaCoRe) spawn hundreds of parallel instances. Infrastructure papers (Autellix, ScaleSim) optimize serving for concurrent agent programs. Each community uses different metrics, benchmarks, and definitions of "scale."

This report provides a unified view across all these dimensions.

### Scope
- **Source**: 595 unique papers from paper_finder (conference proceedings), Semantic Scholar, and arXiv
- **Filtered**: 66 papers across 6 thematic categories
- **Coverage**: NeurIPS 2024-2025, ICLR 2024-2026, ICML 2024-2025, plus key arXiv preprints
- **Deep dives**: 12 papers analyzed in detail across simulation, scaling, and infrastructure themes

---

## 2. Taxonomy: Three Dimensions of Scale

### 2.1 Dimension 1 — Many Unique Agents (Social Simulation)

Each agent has a distinct personality, memory, and social connections. Scale ranges from 25 (Smallville) to 1M (OASIS).

| System | Agent Count | Infrastructure | GPU Requirement |
|--------|-------------|---------------|-----------------|
| OASIS | up to 1M | vLLM + async events | 5-27× A100-80G |
| AgentSociety | 10K-30K | Ray + MQTT | API-based |
| AgentTorch | up to 8.4M | LLM archetypes | Commodity (archetype batching) |
| AgentScope | 1M in 12 min | Actor-based distribution | 4× nodes (8× A100-80G each) |
| Project Sid | 1,000+ | PIANO + Minecraft | GPT-4o API |
| Smallville | 25 | Single-node | Minimal |

**Key insight**: AgentTorch's archetype model is orders of magnitude more efficient than per-agent LLM calls, reducing millions of LLM invocations to thousands by grouping agents with similar demographics.

### 2.2 Dimension 2 — Many Parallel Instances (Test-Time Scaling)

N copies of the same agent run independently on the same problem. No inter-agent coordination.

| Task Type | Parallel Scale | Coordination | GPU Scaling |
|-----------|---------------|-------------|-------------|
| Best-of-N sampling | 100-1000+ | None | Linear |
| Batch code evaluation | 500-8K | None | Linear |
| Parallel tree search | 100-65K | Shared search tree | Near-linear |
| Evolutionary search | 15-200+ | Shared database | Sublinear |

**Key insight**: These workloads benefit from standard LLM inference scaling (more GPUs = more throughput). They do not require agent-specific infrastructure beyond prefix caching and good batching.

### 2.3 Dimension 3 — Many Concurrent Programs (Infrastructure)

A serving system multiplexes many independent agent programs, each making sequential multi-turn LLM calls.

| System | Concurrent Programs | Key Optimization | Public Code |
|--------|-------------------|------------------|-------------|
| Autellix | Thousands | Program-level scheduling | No |
| ScaleSim | 1000+ | Invocation-distance KV management | No |
| AIOS | 250-2000 | Time-multiplexing | Partial |
| vLLM | General | PagedAttention, continuous batching | Yes |
| SGLang | General | RadixAttention, prefix caching | Yes |

---

## 3. Multi-Agent Scaling Laws

### 3.1 Collaboration Degrades at Scale

Multiple studies confirm that increasing the number of collaborating LLM agents shows diminishing or negative returns:

- **Google (2025)**: Tested 180 configurations across 4 benchmarks. Performance often degrades beyond 5-7 agents. Error amplification reaches 17.2× in independent architectures.
- **MacNet (ICLR 2025)**: Logistic growth law — performance follows S-curve, plateauing after initial gains. Tested up to 1000+ agents in DAG topology.
- **SwarmBench**: LLMs fundamentally struggle with decentralized coordination. Tested 13 models on 5 swarm tasks with 8-16 agents.
- **Majority voting**: Plateaus at approximately 8 agents (Google study).

### 3.2 Exceptions

- **Social simulation**: More agents = richer emergent phenomena (the goal is not task performance).
- **Best-of-N**: More samples = higher probability of finding correct solution (statistical guarantee).
- **Evolutionary search**: Larger island populations explore more of the solution space.

---

## 4. Benchmark Catalog

### 4.1 Mature Benchmarks (Standardized, Leaderboards, Reproducible)

| Benchmark | Domain | Scale | Status |
|-----------|--------|-------|--------|
| SWE-bench Verified/Pro | Software engineering | 500-1,865 tasks | Standard, active leaderboard |
| MATH / AIME | Mathematics | Thousands of problems | Standard, widely used |
| miniF2F | Formal theorem proving | 488 problems | Standard, active |
| PutnamBench | Competition mathematics | 1,692 problems | Newer, growing |

### 4.2 Emerging Benchmarks (Standardized, Limited Adoption)

| Benchmark | Domain | Scale | Status |
|-----------|--------|-------|--------|
| MARBLE / MultiAgentBench | Multi-agent collaboration | Variable (star/chain/tree/graph) | ACL 2025, active |
| SwarmBench | Swarm coordination | 5 tasks, 8-16 agents | Published, active |
| OrchestrationBench | Agent planning + tool use | ICLR 2026 | Recent |
| A2ASecBench | Agent-to-agent security | ICLR 2026 | Recent |

### 4.3 System-Specific Metrics (No Cross-System Standard)

| System | Metrics | Limitation |
|--------|---------|------------|
| OASIS | NRMSE vs real Twitter, opinion dynamics | OASIS-only |
| AgentSociety | Polarization, crisis response | AgentSociety-only |
| AgentTorch | Labor participation accuracy | AgentTorch-only |

### 4.4 Missing Benchmarks

| Domain | What's Missing | Priority |
|--------|---------------|----------|
| **Agent inference serving** | Throughput/latency under agentic workloads | CRITICAL |
| **Cross-system social sim** | Unified metrics across platforms | HIGH |
| **Evolutionary search** | Unified problem suite with budget normalization | HIGH |

---

## 5. Architecture Patterns

Four infrastructure patterns emerged from the deep dives:

### Pattern A: Actor-Based Distribution (AgentScope)
- Each agent = actor with mailbox, distributed across machines
- 1M agents in 12 min on 4 machines (32× A100-80G total)
- Best for: Social simulation with many unique agents

### Pattern B: Event-Driven Async (OASIS)
- Global time engine controls simulation ticks
- Agents process events asynchronously, multi-threading gives 13× speedup
- Best for: Synchronized simulation with shared environment

### Pattern C: Program-Level Scheduling (Autellix)
- Agent programs (not individual requests) are the scheduling unit
- DAG-based execution tracking, 4-15× throughput by exploiting program-level context
- Best for: Serving many concurrent agent programs

### Pattern D: Prefix Caching (KVFlow, SGLang)
- Agents sharing system prompts benefit from KV cache reuse
- RadixAttention automatically detects shared prefixes
- Best for: Parallel instances, BoN sampling, batch evaluation

---

## 6. Key Tensions

### Tension 1: Unique Agents vs Parallel Instances
Social simulation needs agents with **distinct identities** (expensive: each agent has unique context). Test-time scaling needs **identical instances** (efficient: shared prefix, independent execution). Infrastructure optimizations differ fundamentally between these two patterns.

### Tension 2: Scale vs Quality
More agents often degrades quality for collaboration (Google study). But for social simulation, more agents = richer emergent phenomena. And for BoN sampling, more samples = higher probability of finding a correct solution. The relationship between scale and quality is domain-dependent.

### Tension 3: Open vs Closed Infrastructure
The 3 most impactful infrastructure papers (Autellix, ScaleSim, KVFlow) have no public code. Open alternatives (vLLM, SGLang) are general-purpose, not agent-optimized. This creates a reproducibility crisis and blocks community progress.

---

## 7. Research Gaps

1. **No Unified Agent-Workload Inference Benchmark** [CRITICAL] — No public benchmark measures throughput/latency/GPU-utilization under realistic agentic workload patterns.

2. **No Cross-System Social Simulation Benchmark** [HIGH] — OASIS, AgentSociety, AgentTorch, AgentScope each define their own metrics. No way to compare quality across platforms.

3. **No Standardized Evolutionary Search Benchmark** [HIGH] — FunSearch, AlphaEvolve, ReEvo, EoH each use different problems with incomparable compute budgets.

4. **Infrastructure Papers Unreproducible** [HIGH] — Autellix (4-15×), ScaleSim (1.31-1.74×), KVFlow (1.83-2.19×) cannot be verified or built upon.

5. **Scaling Laws Not Validated Across Domains** [MODERATE] — Google's study: 4 benchmarks, max 9 agents. MacNet: coding tasks only. Neither tested scientific/technical domains.

---

## 8. Connection to Scientific/Technical Tasks

This report covers the **general landscape** of large-scale multi-agent systems. The companion research (`multi-gpu-agent-tasks/`) focuses specifically on **scientific and technical domains** (mathematics, theorem proving, algorithm design, scientific discovery, bioinformatics, software engineering).

Key finding from the companion research: Scientific multi-GPU agent tasks are fundamentally **search problems with automated verification**, organized into three architectures:

1. **Evolutionary islands** — FunSearch/AlphaEvolve: Population-based search with LLM-generated mutations. Sublinear GPU scaling due to shared evaluation database. 15-200+ agents.

2. **Parallel tree search** — STP, AB-MCTS: Systematic exploration of proof/solution trees. Near-linear to adaptive GPU scaling. 100-65K parallel samples.

3. **Embarrassingly parallel sampling** — BoN, SWE-bench: Independent runs with final selection. Linear GPU scaling. 100-8K parallel instances.

The key distinction: scientific tasks achieve multi-GPU scale through **parallel search**, not through agent **collaboration**. Coordination is minimal (shared database or proof tree at most), and the primary GPU cost driver is inference throughput.

---

## 9. Conclusions

1. **Only social simulation genuinely needs 100+ unique, coordinating agents.** All other multi-GPU scenarios involve parallel instances (test-time scaling, batch evaluation) or concurrent independent programs (serving infrastructure).

2. **Multi-agent collaboration shows diminishing returns beyond 5-9 agents** across multiple studies. Current LLMs struggle with decentralized coordination.

3. **The most critical gap is the absence of an agent-workload inference benchmark.** Without standardized measurement of agent-serving efficiency, infrastructure progress remains unverifiable.

4. **The field is bifurcated**: social simulation (unique agents) and compute scaling (parallel instances) have fundamentally different infrastructure needs. Prefix caching helps parallel instances enormously but offers little benefit for social simulation with unique agent contexts.

5. **Infrastructure research is bottlenecked by closed code.** The three most promising agent-serving optimizations (Autellix, ScaleSim, KVFlow) have no public implementations.

---

## References

### Social Simulation
- OASIS: Open Agent Social Interaction Simulations with One Million Agents (2024). arXiv:2411.11581
- AgentSociety: Large-Scale Simulation with LLM Agents (WWW 2025). arXiv:2502.17581
- AgentTorch: Scaling Simulations with LLM Archetypes (AAMAS 2025). arXiv:2306.02978
- Very Large-Scale Multi-Agent Simulation in AgentScope (2024). arXiv:2407.17789
- Project Sid: 1000-Agent Minecraft Civilization (2024). arXiv:2411.00114

### Scaling & Benchmarks
- Towards a Science of Scaling Agent Systems (Google, 2025). arXiv:2503.xxxxx
- MacNet: Collaborative Scaling Law for LLM Agents (ICLR 2025). arXiv:2406.07155
- Collective Behaviour of Hundreds of LLM Agents (2026). arXiv:2503.04367
- SwarmBench: Evaluating LLMs on Swarm Intelligence (2025). arXiv:2504.01941
- MARBLE: Multi-Agent Benchmark (ACL 2025). arXiv:2504.01559
- MAS²: Self-Generative Multi-Agent Systems (ICLR 2026). arXiv:2502.xxxxx

### Infrastructure
- Autellix: Program-Level LLM Serving (2025). arXiv:2502.13965
- ScaleSim: Multi-Agent Simulation Serving (2026). arXiv:2601.21473
- KVFlow: Prefix Caching for Multi-Agent Inference (2025)
- vLLM: Easy, Fast, and Cheap LLM Serving. https://github.com/vllm-project/vllm
- SGLang: Efficient Execution for Structured Language Model Programs. https://github.com/sgl-project/sglang

### Test-Time Scaling & Scientific Tasks
- Scaling LLM Test-Time Compute (2024). arXiv:2408.03314
- FunSearch: Making New Discoveries with LLMs (Nature, 2024)
- AlphaEvolve: A Coding Agent for Scientific Discovery (Google DeepMind, 2025)
- STP: Self-play Theorem Prover (2025). arXiv:2502.xxxxx
- SWE-bench: Can Language Models Resolve Real-World GitHub Issues? (ICLR 2024)
