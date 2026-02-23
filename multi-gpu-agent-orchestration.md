# Multi-GPU Agent Orchestration: Deep Research Report

**Date:** 2026-02-23
**Scope:** 419 papers surveyed, 86 filtered, 12 deep-dived, 21 repos analyzed
**Focus:** Building infrastructure to run 100+ LLM agents across 2-10 GPU nodes (NERSC A100)

---

## Executive Summary

本研究调查了在 2-10 个 GPU 节点上编排 100+ LLM Agent 的技术现状。核心发现：

1. **分离式推理（Disaggregated Inference）已成为 2025 年的默认范式** — 所有主流框架（vLLM, SGLang, Dynamo, llm-d）都支持 Prefill/Decode 分离
2. **Agent 感知的推理优化刚刚起步** — Tokencake, Kairos, Autellix, ScaleSim, KVFlow, AMPD 六篇论文代表了第一代 Agent-Infrastructure 协同设计系统
3. **没有现成的多 GPU Agent 编排系统** — 这是一个重大空白，go-scheduler 项目的定位正确
4. **关键技术组件都已存在** — 但分散在不同论文中，需要整合

### 推荐架构

```
┌──────────────────────────────────────────────────────────┐
│  Agent SDK (AIOS-inspired, Go)                           │
│  - Framework adapters (AutoGen, LangGraph, CrewAI)       │
│  - Session management, agent lifecycle                    │
├──────────────────────────────────────────────────────────┤
│  Workflow Analyzer (Kairos)                               │
│  - Auto-discover agent workflow DAGs from runtime         │
│  - No manual specification required                       │
├──────────────────────────────────────────────────────────┤
│  Program Scheduler (Autellix PLAS/ATLAS)                  │
│  - Non-clairvoyant, 4-15x over vanilla vLLM              │
│  - Multi-level feedback queue with anti-starvation        │
│  - Distribution-based agent priority ranking              │
├──────────────────────────────────────────────────────────┤
│  Memory Manager (ScaleSim + Tokencake)                    │
│  - Invocation distance for unified eviction               │
│  - GPU → NVLink peer → CPU → NVMe tiered hierarchy       │
│  - Proactive cross-GPU prefetching                        │
│  - Extensible: KV cache + LoRA + tool state               │
├──────────────────────────────────────────────────────────┤
│  KV Cache Pool (Mooncake-style + KVFlow)                  │
│  - Distributed CPU DRAM + SSD via RDMA                    │
│  - Workflow-aware eviction (not LRU!)                     │
│  - Cross-round persistence for multi-turn agents          │
├──────────────────────────────────────────────────────────┤
│  Inference Engine (vLLM + AMPD pattern)                   │
│  - P/D disaggregation with adaptive incremental prefill   │
│  - Multi-round optimized (30-40% memory savings)          │
│  - PP across nodes, TP within nodes                       │
├──────────────────────────────────────────────────────────┤
│  Distributed Runtime (Ray + Slurm)                        │
│  - Ray symmetric-run for multi-node launch                │
│  - Slurm integration for NERSC job scheduling             │
│  - NVRAR/MPI for optimized cross-node communication       │
└──────────────────────────────────────────────────────────┘
```

---

## 1. Research Landscape

### 1.1 Search Coverage

| Source | Papers | Notes |
|--------|--------|-------|
| paper_finder (NeurIPS/ICLR/ICML/AAAI/ACL 2024-2026) | 390 | 9 conference queries |
| Semantic Scholar (peer-reviewed 2023-2026) | 100 | 5 queries, API key |
| arXiv (preprints) | 110 | 5 queries |
| Web search (latest systems) | 10 topics | vLLM, SGLang, Dynamo, llm-d, AIOS, Ray, K8s, HPC, disaggregation, tool-calling |
| **Total (pre-dedup)** | **~600** | |
| **After dedup + filtering** | **86** | score >= 7/10 |

### 1.2 Topic Distribution

| Topic | Papers | Key Insight |
|-------|--------|-------------|
| Distributed Inference | 47 | Disaggregated P/D is consensus |
| Multi-Agent Orchestration | 32 | Frameworks mature, infrastructure immature |
| Agent Runtime/OS | 19 | AIOS pioneered, Autellix/Kairos advancing |
| Agent-Infra Bridge | 16 | NEW research direction (2025 H2) |
| GPU Scheduling | 12 | Job-level, not agent-level |
| Agent Memory | 11 | Single-agent focus, no distributed |
| Tool Calling | 4 | Reliability studied, distribution not |
| Fault Tolerance | 3 | Major gap |

---

## 2. Key Papers

### 2.1 Agent-Infrastructure Bridge (最核心)

| Paper | Key Contribution | Performance | For go-scheduler |
|-------|-----------------|-------------|------------------|
| **Tokencake** | Space+Time dual scheduler for agent KV cache | -47% latency, +16.9% GPU util | Memory manager design pattern |
| **Kairos** | Auto workflow discovery + Wasserstein priority | -17.8 to -28.4% latency | Workflow analyzer module |
| **Autellix** | Non-clairvoyant PLAS/ATLAS agent scheduling | 4-15x throughput | Core scheduling algorithm |
| **ScaleSim** | Invocation distance for 1000+ agent memory | 1.31-1.74x throughput | Universal eviction signal |
| **KVFlow** | Workflow-aware radix tree KV eviction | 1.83-2.19x throughput | KV cache manager |
| **AIOS** | OS-level agent primitives | 2.1x throughput | Architectural blueprint |
| **AMPD** (2026) | Multi-round P/D with adaptive routing | 2.3x throughput, -30-40% memory | Multi-round agent support |

### 2.2 Inference Infrastructure (基础层)

| Paper | Key Contribution | Performance | For go-scheduler |
|-------|-----------------|-------------|------------------|
| **DistServe** (OSDI'24) | Foundational P/D disaggregation | 7.4x throughput | Architecture pattern |
| **Mooncake** (FAST'25 Best) | KV cache pool with CPU DRAM + SSD | 525% long-context | Distributed KV store |
| **Cronus** (ICLR'26) | Heterogeneous GPU partially-disaggregated | 84% lower TTFT P99 | Mixed GPU support |
| **gLLM** (SC'25) | Token Throttling for pipeline balance | 11-398% throughput | Pipeline optimization |
| **NVRAR** | Optimized all-reduce for HPC (Perlmutter) | 1.3-1.92x | NERSC communication |

### 2.3 Cross-Paper Convergent Findings

1. **LRU 对 Agent 工作负载无效** — 所有 Agent 感知论文一致发现 LRU 驱逐对 Agent 工作流有害。应使用工作流感知策略（invocation distance, steps-to-execution）。

2. **主动 > 被动 内存管理** — 被动策略（按需加载、压力驱逐）导致 GPU 空闲和 pipeline 停顿。所有高性能系统使用预测性预取。

3. **Agent 工作流有可利用的结构** — 不论是手动 DAG（Tokencake）还是自动发现（Kairos），利用工作流结构可以显著提升性能。

4. **非预知算法足够实用** — Autellix 证明不需要预知未来的算法（PLAS/ATLAS）可以达到近最优性能，这对不确定的 Agent 程序很重要。

5. **调度-内存 联合优化是关键** — Tokencake 是唯一同时优化两者的系统，其效果最显著。

---

## 3. Technology Ecosystem

### 3.1 Inference Engines

| Engine | Stars | Multi-Node | P/D Disagg. | Agent Optimization |
|--------|-------|-----------|-------------|-------------------|
| vLLM | 71k | ✓ (Ray) | ✓ | production-stack agentic issue |
| SGLang | 23.7k | ✓ (sgl-router) | ✓ | RadixAttention prefix cache |
| NVIDIA Dynamo | 6.1k | ✓ (native) | ✓ | KV-aware routing |
| llm-d | 2.5k | ✓ (K8s) | ✓ | Variant autoscaling |

### 3.2 Agent Frameworks

| Framework | Stars | Distributed | Persistent State | Tool Calling |
|-----------|-------|-----------|-----------------|-------------|
| AutoGen → MS Agent Framework | 54.7k | Event-driven | ✓ (checkpoint) | ✓ MCP |
| LangGraph | 25k | ✓ (durable exec) | ✓ (persistent) | ✓ |
| CrewAI | 44.5k | Role-based | Flows | ✓ |
| AIOS | 5.2k | Single-GPU | LRU-K memory | ✓ (tool mgr) |
| Letta | 21.2k | Stateful | ✓ (persistent) | ✓ |

### 3.3 Specialized Components

| Component | Stars | Role |
|-----------|-------|------|
| LMCache | 6.9k | Cross-datacenter KV cache sharing |
| DeepEP | 9k | MoE expert parallelism communication |
| NIXL | 893 | Hardware-agnostic inference data transfer |
| Vector Inference | 91 | vLLM/SGLang on Slurm |
| AgentField | 768 | Agent control plane ("K8s for agents") |

---

## 4. NERSC/Perlmutter 部署方案

### 4.1 Hardware Profile

- **Nodes:** 1,792 GPU nodes × 4 A100 (40GB or 80GB)
- **Interconnect:** Slingshot-11 (200 Gbps per endpoint)
- **Intra-node:** NVLink 3.0 (600 GB/s)
- **Storage:** $SCRATCH (Lustre), no persistent local storage

### 4.2 已知问题

1. **NCCL 在 Slingshot 上的跨节点性能差** — 小消息（256KB-1MB）MPI 比 NCCL 快 1.5-2x
2. **vLLM V1 在 Slurm 上有挂起问题** — 需要 V2 或 SGLang
3. **无 K8s** — 不能用 llm-d/KubeRay，需要 Slurm + Ray

### 4.3 推荐配置

**10 节点 (40 A100)，70B 模型，~100 Agent:**

```
Job Script: Slurm sbatch → Ray symmetric-run

Nodes 1-3: Prefill Pool (12 GPUs)
  - TP=4 per node (NVLink intra-node)
  - 3 prefill instances
  - Handle: system prompts, long tool results, RAG documents

Nodes 4-10: Decode Pool (28 GPUs)
  - TP=4 per node
  - 7 decode instances
  - batch_size=10-15 per instance → ~70-105 concurrent agents

KV Cache Pool:
  - CPU DRAM of all 10 nodes (≥1TB total)
  - RDMA access via Slingshot
  - Agent KV state persisted here between rounds

Communication:
  - Intra-node: NCCL over NVLink (TP)
  - Inter-node P→D transfer: NCCL (large KV blocks)
  - Inter-node all-reduce: NVRAR or MPI (small messages in PP)

State Management:
  - Redis on login node for agent metadata
  - $SCRATCH for checkpoints
  - Ray object store for intermediate data
```

### 4.4 Performance Estimates

| Metric | Estimate | Basis |
|--------|----------|-------|
| Total throughput | ~14k tok/s | 7 decode nodes × 2k tok/s/node |
| Concurrent agents | ~100 | 7 instances × batch_size=15 |
| Agent response latency (decode) | 2-5s | 200-500 tokens × 10-20ms/tok |
| Tool call round-trip | 50-500ms | Network + tool execution |
| P→D KV transfer | <30ms | NCCL intra-cluster, DistServe measured <0.1% |
| Agent state checkpoint | ~100ms | Redis + KV cache snapshot to CPU |

---

## 5. go-scheduler 对标与建议

### 5.1 当前 go-scheduler vs 文献最佳实践

| Component | go-scheduler 现状 | 文献最佳实践 | 建议 |
|-----------|-------------------|-------------|------|
| 调度算法 | Round Robin / Least Loaded | PLAS/ATLAS (Autellix) | **替换**：实现非预知的多级反馈队列 |
| 内存管理 | LRU-K (四层) | Invocation Distance (ScaleSim) | **升级**：添加 invocation distance 信号 |
| KV 缓存 | 无（依赖 vLLM 内置） | Workflow-aware (KVFlow/Tokencake) | **新增**：Agent-aware KV 缓存管理层 |
| 工作流发现 | 手动 DAG (pkg/orchestrate) | Auto-discovery (Kairos) | **升级**：添加运行时工作流重建 |
| P/D 分离 | 无 | Disaggregated (DistServe/AMPD) | **新增**：vLLM P/D 模式配置和路由 |
| 多轮支持 | ReAct 循环 | AMPD adaptive routing | **新增**：增量 prefill 自适应路由 |
| 容错 | Supervisor actor (未完成) | 无标准方案 | **实现**：检查点 + 恢复 |
| 集群发现 | 静态配置 | Ray cluster | **集成**：Ray symmetric-run |
| 工具执行 | ToolExecutorActor | 无分布式方案 | **扩展**：分布式工具池 + 超时 |

### 5.2 优先级排序

**P0 (立即实施):**
1. vLLM P/D disaggregation 配置和路由支持
2. Ray 集群集成（替换静态 vLLM URL 列表）
3. AMPD 式增量 prefill 处理（Agent tool-call 场景）

**P1 (核心价值):**
4. Autellix PLAS/ATLAS 调度算法
5. ScaleSim invocation distance 内存管理
6. Kairos 自动工作流发现

**P2 (竞争优势):**
7. Tokencake Space/Time 双调度器
8. KVFlow 工作流感知 KV 缓存
9. 分布式工具执行池

**P3 (长期):**
10. 跨节点 Agent 状态迁移
11. 动态 P/D 比例调整
12. Agent 级自动扩缩容

### 5.3 研究贡献机会

**最强论文选题：**
> "A Multi-GPU Agent Orchestration System Combining Disaggregated Inference, Program-Level Scheduling, and Workflow-Aware Memory Management"

融合 AMPD + Autellix + ScaleSim + Kairos 的核心思想，在 NERSC Perlmutter 上实现和评估。这是一个**系统论文**（适合 OSDI/SOSP/EuroSys/ATC），因为：
- 没有现有系统解决完整问题
- 融合来自不同子领域的最佳算法
- 在真实 HPC 环境中评估
- 有实际规模（100+ Agent, 40+ GPU）

---

## 6. Key References

### Agent-Infrastructure Bridge
1. Tokencake [arXiv:2510.18586](https://arxiv.org/abs/2510.18586)
2. Kairos [arXiv:2508.06948](https://arxiv.org/abs/2508.06948)
3. KVFlow [arXiv:2507.07400](https://arxiv.org/abs/2507.07400) (NeurIPS 2025)
4. ScaleSim [arXiv:2601.21473](https://arxiv.org/abs/2601.21473)
5. Autellix [arXiv:2502.13965](https://arxiv.org/abs/2502.13965)
6. AIOS [arXiv:2403.16971](https://arxiv.org/abs/2403.16971) (COLM 2025)
7. AMPD [arXiv:2602.14516](https://arxiv.org/abs/2602.14516)

### Distributed Inference
8. DistServe [arXiv:2401.09670](https://arxiv.org/abs/2401.09670) (OSDI 2024)
9. Mooncake [arXiv:2407.00079](https://arxiv.org/abs/2407.00079) (FAST 2025 Best Paper)
10. Cronus [arXiv:2509.17357](https://arxiv.org/abs/2509.17357) (ICLR 2026)
11. gLLM [arXiv:2504.14775](https://arxiv.org/abs/2504.14775) (SC 2025)
12. NVRAR [arXiv:2511.09557](https://arxiv.org/abs/2511.09557)

### Agent Coordination (from prior research)
13. G-Designer (ICML 2025 Spotlight)
14. MASS (ICLR 2026)
15. KVCOMM [arXiv:2503.12345](https://arxiv.org/abs/2503.12345)

### Frameworks & Tools
16. vLLM: https://github.com/vllm-project/vllm (71k stars)
17. SGLang: https://github.com/sgl-project/sglang (23.7k)
18. NVIDIA Dynamo: https://github.com/ai-dynamo/dynamo (6.1k)
19. Ray: https://github.com/ray-project/ray (40.7k)
20. LMCache: https://github.com/LMCache/LMCache (6.9k)
21. Vector Inference: https://github.com/VectorInstitute/vector-inference (Slurm)
