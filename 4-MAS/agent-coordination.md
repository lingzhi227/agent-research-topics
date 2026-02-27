# Inference-Time Multi-Agent Coordination: A Survey

## 1. Introduction

As LLM-based agents mature from single-agent tool-use to complex multi-agent systems (MAS), a critical question emerges: **how should multiple LLM agents coordinate at inference time — without retraining the underlying models?**

This survey synthesizes 75+ papers from top AI conferences (NeurIPS 2024-2025, ICML 2024-2025, ICLR 2025-2026, ACL 2025, AAAI 2025, COLM 2025) and leading industry developments (MCP, A2A, AAIF), with emphasis on work from 2025 H2 onward.

We organize inference-time agent coordination into **two paradigms** and **six dimensions**:

- **Paradigm A: Agent OS** — Centralized runtime providing scheduling, memory management, and security (AIOS, MemOS)
- **Paradigm B: Dynamic Orchestration** — Learned controllers that adaptively assemble agents, topologies, and communication strategies per input (G-Designer, MasRouter, MASS)

The six dimensions — **Topology, Communication, Memory, Routing, Security, and Protocol** — are orthogonal and composable, forming a layered stack for MAS design.

---

## 2. Background: From Chains to Graphs

The evolution of multi-agent coordination follows a clear trajectory:

- **2023**: Foundation frameworks (AutoGen, MetaGPT, CrewAI) introduce fixed multi-agent patterns — chains, trees, group chats
- **2024**: LangGraph introduces graph-based workflows; MCP standardizes agent-tool integration; Mixture-of-Agents [@wang2024mixture] shows iterative refinement works
- **2025 H1**: A2A protocol establishes agent-to-agent communication standards; MacNet [@qian2025scaling] discovers logistic scaling laws; G-Designer [@yue2025gdesigner] automates topology optimization
- **2025 H2**: Sub-symbolic communication breakthrough (ThoughtComm, KVComm); memory systems mature (G-Memory, A-Mem); AAIF governance founded; framework consolidation (Microsoft Agent Framework merge)
- **2026 H1**: Theoretical foundations proven (communication limits); security formalized (A2ASecBench); topology optimization reaches ICLR 2026 with MASS, CARD

---

## 3. Taxonomy of Approaches

### 3.1 Topology: How Agents Connect

The communication graph structure between agents is the **highest-impact design dimension** — MASS [@wang2026mass] shows topology optimization alone contributes +10.1% performance improvement, exceeding prompt optimization.

**Key methods**:
- **G-Designer** [@yue2025gdesigner] (ICML 2025 Spotlight): A variational graph autoencoder (VGAE) generates input-dependent communication topologies. A "task virtual node" connects to all agents, enabling task-adaptive graph generation. Token reduction: up to 95.33% vs. fully connected graphs.
- **MacNet** [@qian2025scaling] (ICLR 2025): Organizes agents in a DAG with topological sort. Discovers the **logistic scaling law**: performance = L / (1 + e^{-k(x - x0)}), with emergence at 16-32 agents. Key finding: **divergent structures > convergent; irregular > regular topologies**.
- **MASS** [@wang2026mass] (ICLR 2026): Three-stage joint optimization of prompts + topology + models. Demonstrates topology is the dominant factor.
- **AgentDropout** [@wang2025agentdropout] (ACL 2025): Per-round dynamic elimination of redundant agents and edges using policy gradient + nuclear norm regularization. 29.8% token reduction with +2.19 accuracy gain.

**Consensus**: Static, one-size-fits-all topologies are suboptimal. Task-adaptive, learned topologies consistently outperform fixed structures.

### 3.2 Communication: What Agents Exchange

The **biggest 2025 H2 breakthrough** is moving beyond natural language to sub-symbolic communication.

**Text-based communication** (consensus, established):
- Multi-agent debate and iterative refinement [@du2023debate; @wang2024mixture]
- Sparse debate (ICLR 2026): Not all-to-all; sparse graphs match dense quality at lower cost
- Implicit consensus generation (AAAI 2025): Reduce communication rounds

**Sub-symbolic communication** (frontier, 2025 H2+):
- **Thought Communication** [@zheng2025thought] (NeurIPS 2025 Spotlight): Agents share latent "thought vectors" extracted via sparse autoencoders. The ThoughtComm protocol: (1) extract thoughts from hidden states, (2) route thoughts via dependency structure, (3) inject via prefix adaptation. Result: +67% avg over single-agent, +113% on math (Qwen 3-1.7B: 43.6% → 93.0%). Only **1 prefix token** needed.
- **KVComm** [@liu2026kvcomm] (ICLR 2026): Selectively shares KV-cache entries between agents. Layer selection via attention importance + Gaussian prior. 30% of layers matches full baselines; 2.5-6x FLOPs reduction. **No fine-tuning required** — 1 calibration sample suffices.
- **Communicating Activations** (ICML 2025): Direct hidden-activation sharing between agents.

**Theoretical foundation**:
- **Benefits and Limitations of Communication** [@rizvi2026benefits] (ICLR 2026): Formal complexity analysis using Unique Hard Attention Transformers. Proves communication reduces depth from O(N) to O(log w + N/w) for **state tracking** tasks, but **cannot help** for recall tasks. This explains why multi-agent debate has diminishing returns on certain task types.

**Critical limitation**: Sub-symbolic methods require open-weight models with compatible architectures. Cross-architecture communication (Llama ↔ Qwen) remains unsolved.

### 3.3 Memory: What Agents Remember

Memory has evolved from an afterthought to **core infrastructure** for MAS.

- **G-Memory** [@wang2025gmemory] (NeurIPS 2025 Spotlight): Three-layer hierarchical graph memory:
  - **Interaction layer**: Utterance-level agent dialogue logs
  - **Query layer**: Task metadata and semantic relationships
  - **Insight layer**: High-level generalizable knowledge
  - Bidirectional traversal: upward for strategy, downward for details. LLM-based graph sparsification for compression. +20.89% success rate on ALFWorld. **Framework-agnostic plugin** — works with AutoGen, DyLAN, MacNet.

- **A-Mem** (NeurIPS 2025): Zettelkasten-inspired dynamic indexing with autonomous memory organization.

- **AIOS** [@mei2025aios] (COLM 2025): OS-level context management with snapshot/restore (BLEU=1.0 fidelity). Enables agent "hibernation" — pause generation mid-stream, restore exactly later. Also provides per-agent memory isolation, FIFO/round-robin scheduling (35% latency reduction), and cross-framework compatibility.

- **MemOS** (arXiv 2025.07): Unifies plaintext, activation, and parameter memory into a single OS abstraction with Redis Streams scheduling.

**Direction**: Memory is becoming the central organizing principle — agents are ephemeral; memory is persistent. This mirrors the shift from stateful servers to stateless microservices + external state stores in cloud architecture.

### 3.4 Routing: Which Agent Handles What

- **MasRouter** [@yue2025masrouter] (ACL 2025): Cascaded controller network with three levels:
  (1) Collaboration mode selector (variational latent model: Chain/Tree/Debate)
  (2) Role assigner (structured probabilistic cascade from 26-role pool)
  (3) LLM router (multinomial distribution over model pool: GPT-4o-mini, Claude-3.5-Haiku, Gemini-1.5-Flash, Llama-3.1-70B, DeepSeek-v3)
  End-to-end policy gradient training with cost-aware reward. Avg 80.66% across 6 benchmarks; up to 52% cost reduction.

- **Optima** [@chen2025optima] (ACL 2025 Findings): Generate-rank-select-train framework with MCTS-inspired DPO. 2.8x performance with only 10% of baseline tokens (88-90% reduction).

- **KABB** (ICML 2025): Bayesian multi-armed bandits for exploration-exploitation in agent selection.

### 3.5 Security: Safeguarding Multi-Agent Systems

- **GUARDIAN** [@zhou2025guardian] (NeurIPS 2025): Models MAS collaboration as a discrete-time temporal attributed graph. Unsupervised encoder-decoder with Information Bottleneck theory detects anomalies (hallucinations, injected errors). 94.74% detection rate; 60-75% reduction in API calls.

- **A2ASecBench** (ICLR 2026): First protocol-aware security benchmark for A2A-based systems.

- **Resilience to Faulty Agents** (ICML 2025): Hierarchical structures degrade only 5.5% with faulty agents — most resilient topology.

- **Agents Under Siege** (ACL 2025): Systematic attack taxonomy showing MAS amplifies single-agent vulnerabilities.

### 3.6 Protocol: Interoperability Standards

The protocol landscape has converged rapidly:

| Protocol | Scope | Governance | Status |
|----------|-------|------------|--------|
| **MCP** | Agent ↔ Tool/Data | AAIF (Linux Foundation) | De facto standard; 97M+ monthly SDK downloads |
| **A2A** | Agent ↔ Agent | AAIF | v0.3; 150+ organizations; absorbed IBM's ACP |
| **ANP** | Decentralized agent networking | Open source | Early stage |
| **AGENTS.md** | Agent ↔ Codebase | OpenAI | 60K+ repos |

**Milestone**: In December 2025, Anthropic, Google, OpenAI, and Microsoft jointly founded the **Agentic AI Foundation (AAIF)** under the Linux Foundation, donating MCP, A2A, and AGENTS.md. This represents unprecedented cooperation among competitors on agent infrastructure standards.

---

## 4. Framework Landscape

| Framework | Stars | Architecture | Protocol Support | Key Differentiator |
|-----------|-------|-------------|-----------------|-------------------|
| AutoGen/Microsoft Agent Framework | ~45K | Event-driven group chats/workflows | MCP + A2A + OpenAPI | Most complete protocol support; enterprise-grade |
| LangGraph | ~25K | Directed graph state machine | MCP | Graph-first design; persistent checkpointing |
| CrewAI | ~32K | Role-based agents | MCP | Fastest prototyping; role/goal/backstory abstraction |
| MetaGPT/Foundation Agents | ~64K | SOP-driven workflows | Custom → MCP | AFlow (ICLR 2025 Oral); automated workflow generation |
| OpenAI Agents SDK | ~19K | Agents-as-tools + handoffs | MCP (built-in) | Simplest multi-agent API; replaced Swarm |
| CAMEL/OWL | ~16K/~11K | Communicative agent pairs | Custom | GAIA leaderboard #1 open-source; scaling law research |

**Key observation**: No framework natively supports sub-symbolic communication (ThoughtComm, KVComm) or learned topology optimization (G-Designer, MASS). This is the primary **research-to-production gap**.

---

## 5. Open Problems and Future Directions

### 5.1 Solved (Consensus)
1. **MCP** is the standard for agent-tool integration
2. Multi-agent collaboration follows **logistic scaling laws** — diminishing returns are predictable
3. **Dynamic > static** across all dimensions (topology, routing, pruning, memory)
4. Default multi-agent communication wastes **70-95% of tokens** — smart selection is essential
5. **Hierarchical structures** are most resilient to faulty/adversarial agents

### 5.2 Active Frontiers
1. **Sub-symbolic communication**: ThoughtComm, KVComm — bypassing text bottleneck
2. **Automated MAS design**: MASS, MAS², AFlow — one-shot system generation from task description
3. **Memory as infrastructure**: G-Memory, MemOS, AIOS — standardizing agent memory
4. **Formal communication theory**: Provable limits on when MAS helps
5. **Protocol-level security**: A2ASecBench, GUARDIAN integration

### 5.3 Open Problems
1. **Cross-architecture sub-symbolic communication**: KVComm only works within same model family
2. **Runtime topology adaptation**: No method adapts topology mid-conversation
3. **Sub-symbolic security**: No defense against adversarial KV-cache injection
4. **Decentralized long-horizon coordination**: Current LLMs lack the capability (SwarmBench)
5. **Unified cross-framework memory API**: No standard exists
6. **Evaluation standards**: No comprehensive MAS benchmark covering all dimensions

### 5.4 Most Promising Directions
1. **Hybrid symbolic + sub-symbolic protocols**: Text for cross-model, sub-symbolic for same-model (HIGH POTENTIAL, no paper yet)
2. **End-to-end automated MAS design**: Task → optimal agents, topology, prompts, communication in one shot (HIGH POTENTIAL, converging approaches)
3. **Memory-centric architecture**: Agents are ephemeral; memory is persistent (MEDIUM-HIGH, analogous to cloud microservices + databases)
4. **Cross-framework agent runtime**: AIOS + AAIF → Kubernetes for agents (HIGH POTENTIAL, industry momentum)

---

## 6. Industry vs. Academia Gap

The most significant tension in the field:

| Dimension | Industry | Academia | Gap |
|-----------|----------|----------|-----|
| Communication | Text/JSON (MCP, A2A) | Sub-symbolic (KVComm, ThoughtComm) | Cannot deploy research findings with API-only models |
| Topology | Manual graph definition | Learned optimization (G-Designer, MASS) | No framework API for pluggable topology optimizers |
| Memory | Framework-specific | Framework-agnostic plugins (G-Memory) | No standard memory API |
| Security | Auth/permissions | Behavioral anomaly detection | Neither alone is sufficient |
| Evaluation | Latency/cost/reliability | Accuracy on benchmarks | Misaligned optimization targets |

**Bridging mechanisms**: AAIF governance (2025.12) is the most promising vehicle. If model providers expose KV-cache via API, and frameworks support topology optimizer APIs, the gap narrows significantly.

---

## 7. Conclusion

Inference-time multi-agent coordination has matured from ad-hoc prompting (2023) to a structured discipline with:

1. **A standard protocol stack** (MCP + A2A + AGENTS.md) under neutral governance (AAIF)
2. **A composable 3-layer architecture**: Topology → Communication → Memory
3. **A theoretical framework** for when coordination helps and when it doesn't
4. **A scaling law** predicting diminishing returns from adding agents
5. **Efficiency methods** reducing overhead by 30-95% without quality loss
6. **Security primitives** for production deployment

The field's defining tension is between academic breakthroughs (sub-symbolic communication showing +67% improvement, learned topologies showing +10% gains) and industry constraints (closed API models, framework fragmentation, reliability requirements). The next 12 months will be defined by whether the AAIF governance structure can bridge this gap.

For practitioners, the current best practice is:
- **Use MCP + A2A** for agent-tool and agent-agent integration
- **Choose dynamic routing** (MasRouter-style) over static agent assignment
- **Invest in memory infrastructure** (G-Memory-style hierarchical memory)
- **Profile communication redundancy** — most MAS waste 70-95% of tokens
- **Plan for sub-symbolic communication** — it will arrive in production as open-weight model deployment grows

---

## References

[@chen2025optima]: Chen et al. "Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System." ACL 2025 Findings.

[@du2023debate]: Du et al. "Improving Factuality and Reasoning in Language Models through Multiagent Debate." 2023.

[@liu2026kvcomm]: Liu et al. "KVComm: Enabling Efficient LLM Communication through Selective KV Sharing." ICLR 2026.

[@mei2025aios]: Mei et al. "AIOS: LLM Agent Operating System." COLM 2025.

[@qian2025scaling]: Qian et al. "Scaling Large Language Model-based Multi-Agent Collaboration." ICLR 2025.

[@rizvi2026benefits]: Rizvi et al. "Benefits and Limitations of Communication in Multi-Agent Reasoning." ICLR 2026.

[@wang2024mixture]: Wang et al. "Mixture-of-Agents Enhances Large Language Model Capabilities." ICLR 2025 Spotlight.

[@wang2025agentdropout]: Wang et al. "AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration." ACL 2025 Long.

[@wang2025gmemory]: Wang et al. "G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems." NeurIPS 2025 Spotlight.

[@wang2026mass]: Wang et al. "Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies." ICLR 2026.

[@yue2025gdesigner]: Yue et al. "G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks." ICML 2025 Spotlight.

[@yue2025masrouter]: Yue et al. "MasRouter: Learning to Route LLMs for Multi-Agent Systems." ACL 2025 Long.

[@zheng2025thought]: Zheng et al. "Thought Communication in Multiagent Collaboration." NeurIPS 2025 Spotlight.

[@zhou2025guardian]: Zhou et al. "GUARDIAN: Safeguarding LLM Multi-Agent Collaborations with Temporal Graph Modeling." NeurIPS 2025.

### Additional Key References

- A-Mem: Agentic Memory for LLM Agents. NeurIPS 2025.
- AFlow: Automating Agentic Workflow Generation. ICLR 2025 Oral.
- CARD: Conditional Design of Multi-agent Topological Structures. ICLR 2026.
- MAS^2: Self-Generative, Self-Configuring, Self-Rectifying Multi-Agent Systems. ICLR 2026.
- MAS-GPT: Training LLMs to Build LLM-based Multi-Agent Systems. ICML 2025.
- MemOS: A Memory Operating System for AI. arXiv 2025.
- MemoryOS: Memory Operating System for Personalized AI Agents. EMNLP 2025 Oral.
- Multi-agent Architecture Search via Agentic Supernet. ICML 2025 Oral.
- A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP. arXiv 2505.02279.
- AgentNet: Decentralized Evolutionary Coordination for LLM-Based MAS. NeurIPS 2025.
- On the Resilience of LLM-Based MAS with Faulty Agents. ICML 2025.
- Which Agent Causes Task Failures and When? ICML 2025.
- SwarmBench: Benchmarking LLMs' Swarm Intelligence. arXiv 2025.
- A2ASecBench: Protocol-Aware Security Benchmark. ICLR 2026.
- Agents Under Siege: Breaking Pragmatic Multi-Agent LLM Systems. ACL 2025 Long.
- OWL: Optimized Workforce Learning. NeurIPS 2025.
