# Multi-GPU Agent Orchestration: Benchmark Collection Report

**Date**: 2026-02-26
**Context**: Benchmarks for designing a cross-GPU-node Agent OS layer on top of vLLM
**Hardware**: 1-20 nodes × 4× A100 80GB SXM4 each
**Base System**: vLLM server running Qwen Agent
**Focus**: High-level orchestration (NOT low-level model architecture changes)

---

## Executive Summary

This report collects benchmarks for evaluating a multi-GPU agent orchestration system, based on systematic literature review of 298 papers (83 high-relevance), deep dive of 9 key papers, and analysis of 30 code repositories.

### Key Findings

1. **Architecture matters as much as the model.** On SWE-bench, the same GPT-4 ranges from 2.7% to 28.3% depending on scaffold design (10x). Adaptive topology routing adds +6.9 to +32.8pp over fixed topologies (AdaptOrch, AMAS).

2. **No standard benchmark exists for multi-agent infrastructure.** Each of the 6 agent serving systems uses different custom workloads. Creating a unified benchmark is itself a contribution.

3. **All agent serving systems with best results (4-15x) have NO public code.** Autellix, Kairos, Tokencake, KVFlow, ScaleSim are all paper-only. Our system would be the first open-source, reproducible multi-GPU agent orchestration system.

4. **Multi-agent value shifts with model strength.** At SWE-bench 80%+, multi-agent value moves from "accuracy improvement" to "throughput/cost/reliability." But weaker models still benefit hugely (7x on LLaMA-3 8B).

5. **Framework overhead is the dominant cost.** MAFBench shows 2-117x latency overhead from framework choice alone. The Agent OS layer must be lightweight (<3x target).

6. **Adaptive topology selection is the single most impactful architecture change.** AdaptOrch (+9.8pp SWE-bench), AMAS (+32.8pp Crosswords), both with <50ms overhead.

---

## Phase 1: Frontier Landscape

### Benchmark Ecosystem (Feb 2026)

| Benchmark | Type | Top Score | Domain | Saturated? |
|-----------|------|-----------|--------|-----------|
| SWE-bench Verified | Code agent | 80.9% (Claude Opus 4.5) | Software engineering | Near ceiling |
| SWE-bench Pro | Code agent (strict) | 55.6% (GPT-5.2) | Multi-language SE | No, high ceiling |
| GAIA L1-L3 | General agent | 87.8% (Genspark) | Multi-tool reasoning | No |
| WebArena | Web agent | 64.2% (Narada) | Web browsing | No (human 78%) |
| OSWorld | Computer use | 72.7% (Claude 4.6) | OS tasks | Just hit human |
| BFCL v4 | Function calling | 70.4% (Claude Opus 4.1) | Tool use accuracy | No |
| tau-bench airline | Tool-agent | 56.0% (o4-mini) | Domain reliability | No |
| MultiAgentBench | Topology | — | Multi-agent coordination | New |
| MAFBench | Framework | — | Orchestration overhead | New |

### Agent Serving Systems

| System | Year | Key Innovation | Best Result | Code? |
|--------|------|---------------|-------------|-------|
| Autellix | 2025-02 | Program-level scheduling | 4-15x vs vLLM | No |
| Kairos | 2025-08 | Workflow discovery + priority | -17.8 to -72.8% latency | No |
| Tokencake | 2025-10 | Space/Time KV dual-scheduler | >47% E2E latency reduction | No |
| KVFlow | NeurIPS 2025 | Steps-to-execution cache | 1.83-2.19x throughput | No |
| ScaleSim | 2026-01 | Invocation distance eviction | 1.31-1.74x @ 1000 agents | No |
| AIOS | COLM 2025 | OS-level agent primitives | 2.1x throughput | Partial |

---

## Phase 2: Survey (83 High-Relevance Papers)

### Paper Distribution by Theme

| Theme | Papers | Key Insight |
|-------|--------|------------|
| Agent serving infrastructure | 56 | Disaggregated prefill/decode is consensus; CXL emerging for KV expansion |
| Scaling & distributed | 41 | Horizontal scaling near-linear; MoE-specific optimizations critical |
| Architecture design | 38 | Adaptive beats static; graph/mesh > tree/hierarchical |
| Benchmarks & evaluation | 26 | Gap: no joint task+infra benchmark |
| Multi-agent systems | 22 | MAFBench: framework choice = 117x latency difference |
| Tool calling | 19 | Reliability is the bottleneck, not reasoning |
| KV cache management | 16 | Agent workloads are KV-cache-bound, not compute-bound |
| Coordination | 10 | Topology determines coordination, more compute cannot fix wrong topology |
| Memory | 9 | Retrieval-first > accumulation (+25.8 on AR task) |

### 6 Research Gaps Identified

1. **No joint task+infrastructure benchmark** (Critical)
2. **No standard agent serving workload** (High)
3. **No multi-node agent coordination benchmark** (High)
4. **No agent × GPU scaling laws** (Medium-High)
5. **Agent serving code reproducibility** (Medium)
6. **Tool-calling overhead at scale** (Medium)

---

## Phase 3: Deep Dive (9 Papers)

### Benchmark Data Extracted

#### A. Framework Overhead (MAFBench, 2602.03128)

| Framework | Architecture | Latency (p50) | vs Direct LLM |
|-----------|-------------|--------------|---------------|
| OpenAI SDK | Role-based | ~0.4s | **2.2x** |
| LangGraph | Graph-based | ~1.2s | **6.7x** |
| CrewAI | Role-based | ~2.5s | **14x** |
| AutoGen | Role-based | ~3.5s | **19x** |
| Concordia | GABM | ~14.47s | **117x** |
| Direct LLM | — | ~0.18s | 1x (baseline) |

**Implication**: Our Agent OS layer must target <3x overhead to be competitive with the lightest frameworks.

#### B. Adaptive Topology Impact (AdaptOrch, 2602.16873)

| Benchmark | Best Fixed Topology | AdaptOrch (Adaptive) | Delta |
|-----------|-------------------|---------------------|-------|
| SWE-bench Verified | 35.6% | **45.4%** | **+9.8pp** |
| GPQA Diamond | 52.8% | **59.7%** | **+6.9pp** |
| HotpotQA F1 | 64.2 | **72.3** | **+8.1** |

Topology distribution per domain: SWE-bench uses 62% Hybrid, GPQA uses 41% Parallel, HotpotQA uses 52% Sequential.

#### C. Communication Topology Optimization (AMAS, EMNLP 2025)

| Model | Method | Game-of-24 | Crosswords | MMLU | HumanEval |
|-------|--------|-----------|-----------|------|-----------|
| LLaMA-3 8B | Best fixed | 2.0 | 23.2 | 38.6 | 40.0 |
| LLaMA-3 8B | **AMAS** | **14.0** | **56.0** | **43.3** | **44.5** |
| LLaMA-3 70B | Best fixed | 32.0 | 41.6 | 68.0 | 71.0 |
| LLaMA-3 70B | **AMAS** | **52.0** | **60.0** | **72.1** | **77.5** |

#### D. Tool-Agent Reliability (tau-bench, ICLR 2025)

| Model | tau-airline pass^1 | Key Failure Mode |
|-------|-------------------|-----------------|
| o4-mini | 56.0% | Policy violations |
| GPT-4.1 | 54.8% | Reasoning failures |
| Claude 3.7 Sonnet | 53.2% | Tool argument errors |
| GPT-4o | 49.4% | Policy violations |

Even the best model fails 44% of the time on airline tasks. Reliability (pass^k) drops rapidly: 70% single-run → 24% pass^3.

#### E. Agentic GPU Scheduling (HexGen-Flow, 2505.05286)

| Metric | vs vLLM |
|--------|---------|
| 95th-percentile SLO | **1.41x** improvement |
| Throughput | **1.57-1.75x** |
| DAG-aware scheduling overhead | <5ms per request |

Two-layer scheduler: global load-balanced dispatch + local urgency-priority queue. Decomposes agentic DAG into stages with per-stage SLO budgets.

#### F. Test-Time Scaling Limits (General AgentBench, 2602.18998)

| Scaling Strategy | Finding | Bottleneck |
|-----------------|---------|-----------|
| Sequential (more turns) | Degrades after ~128K tokens | **Context ceiling** |
| Parallel (K runs) | Needs reliable verifier | **Verification gap** |
| Optimal K | ~4 runs | Diminishing returns |

Tool definitions alone consume ~64K tokens. Context window, not reasoning, is the agent bottleneck.

#### G. Architecture Search (MaAS / Agentic Supernet, ICML 2025 Oral)

- Uses **6-45% inference cost** of handcrafted multi-agent systems
- Surpasses handcrafted by **0.54-11.82%** accuracy
- Key innovation: search over distribution of architectures, not single architecture

---

## Phase 4: Code & Tools

### Reproducibility Summary

| Category | Open Code | Benchmark Reproducible |
|----------|-----------|----------------------|
| Agent serving systems | 1/6 (AIOS partial) | **0/6** |
| Inference infrastructure | 6/6 | 5/6 |
| Multi-agent frameworks | 6/6 | 3/6 |
| Benchmark leaderboards | 5/5 | 5/5 |
| Deep dive papers | 5/7 | 4/7 |

### Key Repos for Our System

| Repo | Stars | Use For |
|------|-------|---------|
| vLLM | 71.3k | Base serving engine |
| SGLang | 23.8k | Alternative base / RadixAttention comparison |
| SWE-bench | 4.4k | Task accuracy evaluation |
| tau-bench | 1.1k | Reliability evaluation (pass^k) |
| MAFBench | 1 | Framework overhead evaluation |
| MaAS | 241 | Architecture search methodology |
| General-AgentBench | 8 | Test-time scaling evaluation |
| BFCL | 12.7k | Function calling evaluation |
| MultiAgentBench/MARBLE | 227 | Topology comparison |
| AIOS | 5.2k | Agent OS comparison baseline |

---

## Phase 5: Synthesis

### Recommended Benchmark Suite

#### Tier 1: Comparable Benchmarks (must report, existing leaderboards)

| Benchmark | What It Tests | Comparison Points |
|-----------|-------------|------------------|
| **SWE-bench Verified** | Code agent accuracy | Qwen3-Coder 67.0%, Claude 4.6 80.8% |
| **GAIA L1-L3** | Multi-tool reasoning | Genspark 87.8%, OpenAI DR 67.9% |
| **tau-bench (airline)** | Tool-calling reliability | o4-mini 56.0%, GPT-4o 49.4% |
| **BFCL v4** | Function calling accuracy | Claude Opus 4.1 70.4% |

#### Tier 2: Architecture Differentiation (show orchestration value)

| Benchmark | What It Tests | Comparison Points |
|-----------|-------------|------------------|
| **MAFBench overhead** | Framework latency | OpenAI SDK 2.2x, LangGraph 6.7x |
| **Topology adaptation** | Adaptive vs fixed | AdaptOrch +9.8pp, AMAS +32.8pp |
| **Agent serving throughput** | GPU efficiency | Autellix 4-15x, Tokencake 86% util |
| **Multi-node scaling** | Scaling efficiency | DistServe linear, Autellix linear |

#### Tier 3: Novel Contributions (fill research gaps)

| Benchmark | What's New | Fills Gap |
|-----------|-----------|----------|
| **Joint task+infra** | First accuracy × throughput evaluation | Gap 1 |
| **Standard agent workloads** | Reproducible multi-agent serving workloads | Gap 2 |
| **Multi-node coordination** | Cross-node agent orchestration benchmark | Gap 3 |
| **Scaling laws** | {agents} × {GPUs} × {topology} study | Gap 4 |

### Performance Targets

#### Task Accuracy (Qwen3-Coder, 67.0% SWE-bench baseline)

| Benchmark | Target | Justification |
|-----------|--------|--------------|
| SWE-bench Verified | **72-75%** | AdaptOrch: +9.8pp with topology adaptation |
| GAIA | **+15-25%** vs baseline | KGoT: 2x with architecture |
| tau-bench airline | **+5-10pp** | Error recovery + retry |

#### Infrastructure (per node: 4×A100 80GB, 7-14B model)

| Metric | vLLM Baseline | Our Target |
|--------|-------------|-----------|
| Throughput | ~3,300 tok/s @50 | **5,000-6,000 tok/s** |
| TTFT | 300ms @50 | **<150ms** |
| GPU utilization | ~70% | **85%+** |
| KV cache waste | ~18.5% | **<5%** |
| Framework overhead | N/A | **<3x** vs direct LLM |
| Concurrent agents | ~100 | **200-400** |

#### Multi-Node (10 nodes, 40×A100)

| Metric | Target |
|--------|--------|
| Scaling efficiency | **>90%** |
| Concurrent agents | **2,000-4,000** |
| Cross-node latency | **<30ms** |
| Coordination overhead | **<50ms/task** |

---

## References

### Deep-Dived Papers (Phase 3)
1. MAFBench — Understanding Multi-Agent LLM Frameworks (arXiv:2602.03128, 2026)
2. tau-bench — Tool-Agent-User Interaction Benchmark (arXiv:2406.12045, ICLR 2025)
3. HexGen-Flow — Agentic Text-to-SQL Scheduling (arXiv:2505.05286, 2025)
4. Adaptive GPU Resource Allocation for Multi-Agent (arXiv:2512.22149, 2025)
5. MaAS — Multi-agent Architecture Search via Agentic Supernet (arXiv:2502.04180, ICML 2025 Oral)
6. General AgentBench — Test-Time Scaling (arXiv:2602.18998, 2026)
7. MAS-Orchestra — Holistic Multi-Agent Orchestration (arXiv:2601.14652, 2026)
8. AdaptOrch — Task-Adaptive Orchestration (arXiv:2602.16873, 2026)
9. AMAS — Adaptive Communication Topology (arXiv:2510.01617, EMNLP 2025)

### Agent Serving Systems (Prior Research + Phase 2)
10. Autellix (arXiv:2502.13965, 2025) — Program-level scheduling
11. Kairos (arXiv:2508.06948, 2025) — Workflow-aware scheduling
12. AIOS (arXiv:2403.16971, COLM 2025) — Agent OS
13. Tokencake (arXiv:2510.18586, 2025) — KV-cache-centric serving
14. KVFlow (arXiv:2507.07400, NeurIPS 2025) — Workflow-aware cache
15. ScaleSim (arXiv:2601.21473, 2026) — Large-scale simulation memory

### Inference Infrastructure (Phase 4)
16. DistServe (OSDI'24) — Prefill-decode disaggregation
17. Mooncake (FAST'25 Best Paper) — KV cache pool
18. Splitwise (ISCA'24 Best Paper) — Phase splitting
19. FlashInfer (arXiv:2501.01005) — Attention kernels

### Architecture Impact Studies (Phase 1)
20. "More Agents Is All You Need" (TMLR 2024) — Agent scaling
21. "Towards a Science of Scaling Agent Systems" (Google, arXiv:2512.08296) — 180 experiments
22. MASS (Google, arXiv:2502.02533) — Topology optimization
23. MetaGPT (ICLR 2024) — Multi-agent collaboration
24. Reflexion (NeurIPS 2023) — Verbal self-reflection
25. Self-Refine (NeurIPS 2023) — Iterative refinement
26. SkillsBench (arXiv:2602.12670) — Skill library impact
27. MemArt (OpenReview) — KV-cache-centric memory

### Benchmarks
28. SWE-bench (swebench.com)
29. GAIA (HuggingFace)
30. BFCL v4 (gorilla.cs.berkeley.edu)
31. tau-bench (github.com/sierra-research/tau-bench)
32. MultiAgentBench / MARBLE (ACL 2025)
33. WebArena
34. OSWorld
35. AgentBench (ICLR 2024)
