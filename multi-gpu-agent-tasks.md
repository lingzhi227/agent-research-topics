# Scientific & Technical Tasks Requiring Multi-GPU LLM Agent Inference: A Systematic Survey

## Abstract

This report systematically surveys scientific and technical tasks that genuinely require tens to hundreds of LLM agents working across multiple GPU nodes, along with their corresponding public benchmarks. Through analysis of 993 papers (65 selected for detailed study, 12 deep-dived) from top AI conferences (NeurIPS, ICLR, ICML 2024-2026), we identify that multi-GPU agent tasks are fundamentally **search problems with automated verification**. We organize them into a taxonomy of three architectural patterns (evolutionary islands, parallel tree search, embarrassingly parallel sampling), map their GPU scaling properties, catalog available benchmarks, and identify five critical research gaps. The most significant finding is that **no unified benchmark exists for evaluating inference-serving efficiency under multi-agent workloads**, despite this being the bottleneck for all identified task categories.

---

## 1. Introduction

### 1.1 Research Question
What scientific, mathematical, and computational tasks genuinely require tens to hundreds of LLM agents collaborating across multiple GPU nodes? What public benchmarks exist for evaluating these scenarios?

### 1.2 Scope
- **Domains**: Mathematics, theorem proving, algorithm design, scientific discovery, software engineering, bioinformatics, computational chemistry/physics
- **Scale**: Tasks requiring 10+ concurrent LLM agents on multi-GPU infrastructure
- **Focus**: GPU-based LLM inference workloads (not simulation or training)

### 1.3 Methodology
Six-phase systematic literature review:
1. **Frontier scan** of latest conference papers and preprints
2. **Survey** of 993 papers, filtered to 65 across 8 categories
3. **Deep dive** into 12 key papers + 4 supplementary
4. **Code & tools** survey of 21 open-source systems
5. **Cross-paper synthesis** with taxonomy and gap analysis
6. **This report** compiling all findings

---

## 2. Core Finding: The Search-Verify Pattern

Every task that genuinely needs multi-GPU LLM agents follows a common pattern:

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│   GENERATE  │ →  │   VERIFY     │ →  │   SELECT     │
│ N parallel  │    │ Automated    │    │ Best of N    │
│ LLM samples │    │ oracle       │    │ solutions    │
└─────────────┘    └──────────────┘    └──────────────┘
   Multi-GPU          CPU/formal          Aggregation
   inference          verifier
```

The GPU demand is proportional to: **N_parallel × model_size × tokens_per_sample**

Without automated verification, scaling N gives diminishing returns — proved theoretically by Setlur et al. (ICML 2025 Spotlight) with a gap of Ω(√H).

---

## 3. Taxonomy of Multi-GPU Agent Tasks

### 3.1 Tier 1: Very High GPU Demand

#### 3.1.1 Evolutionary Algorithm Search with LLMs

**Description**: Multiple LLM "sampler" agents generate candidate programs in parallel, verified by automated evaluators, with an evolutionary selection mechanism feeding successful programs back to samplers.

| System | Architecture | Scale | Key Result |
|--------|-------------|-------|------------|
| **FunSearch** (Nature 2024) | 15 samplers + 150 evaluators + programs DB | ~10⁶ LLM calls over days | New cap set constructions (surpassing 20-year-old records) |
| **AlphaEvolve** (DeepMind 2025, preprint) | Multi-model ensemble + island model + evaluation cascade | Data center scale | Better matrix multiplication algorithms |
| **ReEvo** (NeurIPS 2024) | Reflective evolution, "language gradients" | ~100 evaluations, $0.20 | Competitive on TSP, CVRP at 500× lower cost |
| **EoH** (ICML 2024 Oral) | Thought-code co-evolution | ~2000 LLM calls, $5-10 | Comparable quality at 1000× lower cost than FunSearch |
| **CALM** (ICLR 2026) | Algorithm-LLM co-evolution | Moderate | Generalizable heuristic design |

**Agent coordination**: YES — shared programs database, island migration
**GPU topology**: Independent GPU groups per island, shared database server
**Benchmarks**: No unified benchmark. Each uses domain-specific problems (cap set, bin packing, TSP, sorting networks).

**Key tension**: ReEvo/EoH achieve comparable results at 1000× lower cost, questioning whether massive parallelism is always necessary. But FunSearch discovers *qualitatively new* solutions that cheaper methods miss.

#### 3.1.2 Parallel Formal Theorem Proving

**Description**: Generate thousands of candidate proof steps in parallel, verify each with a formal proof assistant (Lean 4/Isabelle), expand the most promising branches.

| System | Parallelism | Key Result | Benchmark |
|--------|-------------|------------|-----------|
| **STP** (ICML 2025) | 256 TPU-v4, 120M proofs, pass@3200 | miniF2F 65.0%, LeanWorkbook 28.5% (doubled previous SOTA) | miniF2F, LeanWorkbook |
| **BFS-Prover V2** (preprint) | pass@65,536 budget | miniF2F 95.08% | miniF2F |
| **LEGO-Prover** (ICLR 2024 Oral) | Parallel tactic generation | miniF2F 57.0%/50.0%, 22,532 reusable lemmas | miniF2F |
| **ProofOptimizer** (ICLR 2026) | RL + expert iteration | 50-87% proof compression | miniF2F, PutnamBench |
| **Goedel-Prover-V2** (preprint) | pass@32 | miniF2F 88.1% (32B model matches 671B at pass@8192) | miniF2F |

**Agent coordination**: Partial — shared proof tree, independent rollouts
**GPU topology**: Central coordinator + distributed workers (Ray)
**Benchmarks**: miniF2F (244, nearing saturation at 95%), ProofNet (371), PutnamBench (644), LeanWorkbook (15K+)

**Key insight**: Model quality >> sampling quantity. A 32B model at pass@32 matches a 671B model at pass@8192. But for *exploration* (finding new proof strategies), high parallelism remains valuable.

#### 3.1.3 Test-Time Compute Scaling (Parallel Sampling)

**Description**: Generate N independent solutions in parallel, verify/rank them, select the best.

| System | Parallelism | Key Result | Benchmark |
|--------|-------------|------------|-----------|
| **Sample, Scrutinize, Scale** (ICML 2025) | N=200 samples × 50 verifications | Gemini v1.5 Pro surpasses o1-Preview | MATH, AIME, GPQA |
| **PaCoRe** (preprint, 2026) | ~2M tokens, multi-round consensus | 8B model: 94.5% on HMMT (> GPT-5's 93.2%) | HMMT 2025 |
| **AB-MCTS** (NeurIPS 2025 Spotlight) | Adaptive branching, Thompson Sampling | 30%+ improvement on ARC-AGI-2 | ARC-AGI-2, AIME |
| **Kinetics** (NeurIPS 2025) | Budget-aware scaling | 14B is the model-size threshold; sparse attention: 3.2-33.3× throughput | AIME, LiveCodeBench |
| **Setlur et al.** (ICML 2025 Spotlight) | Theoretical analysis | Without verification, scaling is suboptimal by Ω(√H) | MATH |

**Agent coordination**: None (independent samples) or light (consensus in PaCoRe)
**GPU topology**: Any; vLLM/SGLang prefix caching optimal
**Benchmarks**: MATH, AIME/AMC/HMMT, GPQA, LiveCodeBench, GAIA, BrowseComp

**Key insights**:
- "Asymmetric verification": generating solutions is harder than verifying them
- "Implicit scaling": more samples improve verification accuracy (counterintuitive)
- Sparse attention (Kinetics) enables 3.2-33.3× throughput improvement on H200
- Parallel thinking outperforms extended thinking by up to 20% (NeurIPS 2025)

### 3.2 Tier 2: High GPU Demand (Embarrassingly Parallel)

#### 3.2.1 Large-Scale Code Evaluation

| Benchmark | Tasks | Typical Parallelism | Note |
|-----------|-------|--------------------|----- |
| **SWE-bench Verified** | 500 | 500-8,000 instances | Each in Docker container |
| **SWE-bench Pro** | 1,865 | Thousands | 41 repositories |
| **ACE-Bench** (ICLR 2026) | Complex features | Hundreds | End-to-end development |
| **MLE-bench** (NeurIPS 2024) | 75 Kaggle competitions | Parallel | ML pipeline tasks |
| **InnovatorBench** (ICLR 2026) | 20 research tasks | GPU server per task | 11+ hours per task |

**Agent coordination**: None (each instance independent)
**GPU demand**: High due to sheer volume, but each instance is independent
**Key systems**: OpenHands (68.2K stars), SWE-Gym (ICML 2025)

### 3.3 Tier 3: Moderate GPU Demand

#### 3.3.1 Automated Scientific Discovery

| Benchmark | Tasks | Agents | Best Result |
|-----------|-------|--------|-------------|
| **ScienceAgentBench** (ICLR 2025) | 102 (4 disciplines) | Single + framework | 42.2% (o1 + Self-Debug) |
| **MoSciBench** (ICLR 2026) | 88 (6 domains, 7 modalities) | Single | 48.94% |
| **HeurekaBench** (ICLR 2026) | Configurable (single-cell biology) | Single + critic | +22% with critic module |
| **AstaBench** (ICLR 2026 Oral) | 2,400+ (11 benchmarks) | 57 agents evaluated | 53.0% (Asta v0) |
| **InnoGym** (ICLR 2026) | 18 (innovation tasks) | Single | Performance + novelty metrics |

**Agent coordination**: Pipeline (3-7 agents in sequence)
**GPU demand**: Moderate (API-based for most benchmarks; InnovatorBench needs GPU servers)
**Key insight**: No benchmark evaluates multi-GPU scaling for scientific agents

### 3.4 Tier 4: Low GPU Demand (Not Multi-GPU)

- **Drug discovery agents** (DrugAgent, MADD, DiscoVerse): API-based, tool-augmented
- **Chemistry/physics agents** (ChemAgent, MetaOpenFOAM): GPU bottleneck is simulation, not LLM
- **Materials science agents** (HoneyComb): Tool-augmented, single agent

---

## 4. Three Multi-GPU Architectures

### Architecture A: Evolutionary Islands

```
Island 1 (GPU group 1)     Island 2 (GPU group 2)     Island 3 (GPU group 3)
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ Sampler → Eval  │        │ Sampler → Eval  │        │ Sampler → Eval  │
│ Sampler → Eval  │  ←──→  │ Sampler → Eval  │  ←──→  │ Sampler → Eval  │
│ Sampler → Eval  │ migrate│ Sampler → Eval  │ migrate│ Sampler → Eval  │
└────────┬────────┘        └────────┬────────┘        └────────┬────────┘
         │                          │                          │
         └──────────── Shared Programs Database ───────────────┘
```

- **Used by**: FunSearch, AlphaEvolve
- **Scaling**: Sublinear (coordination overhead)
- **Key property**: Maintains diversity via island isolation; periodic migration prevents divergence

### Architecture B: Parallel Tree Search

```
                 Root
              /   |   \
         Branch1  Branch2  Branch3    ← GPU workers explore branches
        / \       / \      / \
     Leaf Leaf  Leaf Leaf Leaf Leaf   ← Verification at leaves
         ↓
   UCB/Thompson selects next branches to expand
```

- **Used by**: AB-MCTS, SWE-Search, STP (proof search)
- **Scaling**: Adaptive (allocate compute to promising branches)
- **Key property**: Dynamic width-vs-depth tradeoff per problem

### Architecture C: Embarrassingly Parallel Sampling

```
GPU 1: Sample → Verify ──┐
GPU 2: Sample → Verify ──┤
GPU 3: Sample → Verify ──├── Select best
  ...                    │
GPU N: Sample → Verify ──┘
```

- **Used by**: BoN, Sample/Scrutinize/Scale, parallel SWE-bench, parallel proof search
- **Scaling**: Linear (N GPUs → N samples/second)
- **Key property**: No coordination overhead; vLLM prefix caching maximizes efficiency

---

## 5. Benchmark Catalog

### 5.1 Theorem Proving Benchmarks

| Benchmark | Size | Domain | Current SOTA | Saturation |
|-----------|------|--------|-------------|------------|
| **miniF2F** | 244 | Competition math (AMC/AIME/IMO) | 95.08% (BFS-Prover V2) | HIGH |
| **ProofNet** | 371 | Undergraduate math | ~24% (STP) | Low |
| **PutnamBench** | 644 | Putnam competition | ~8/644 (STP) | Very Low |
| **LeanWorkbook** | 15,000+ | Math forum problems | 28.5% (STP) | Low |
| **FIMO** | ~100 | Formal IMO problems | Varies | Low |
| **Open Proof Corpus** | 5,000+ proofs | USAMO/IMO (evaluation) | N/A (dataset) | N/A |
| **ProofBench** | 145 + 435 solutions | Fine-grained proof rating | 4.14/7 (ProofGrader BoN@16) | N/A |

### 5.2 Mathematical Reasoning Benchmarks

| Benchmark | Size | Domain | Current SOTA |
|-----------|------|--------|-------------|
| **MATH** | 12,500 | Competition math | 97%+ |
| **AIME** | ~30/year | AMC competition | 90%+ (AgentMath) |
| **HMMT** | ~30/year | Harvard math competition | 94.5% (PaCoRe, 8B model) |
| **GPQA** | 448 | Graduate-level science | ~60% |
| **AMC 10/12** | ~25/year | AMC competition | 90%+ |

### 5.3 Scientific Discovery Benchmarks

| Benchmark | Size | Domains | Best Agent | Venue |
|-----------|------|---------|------------|-------|
| **ScienceAgentBench** | 102 | Bioinformatics, geography, psychology, chemistry | 42.2% | ICLR 2025 |
| **InnovatorBench** | 20 | AI/ML research tasks | 24.01/80 | ICLR 2026 |
| **MoSciBench** | 88 | 6 domains, 7 modalities | 48.94% | ICLR 2026 |
| **HeurekaBench** | Configurable | Single-cell biology | +22% with critic | ICLR 2026 |
| **AstaBench** | 2,400+ | 11 benchmarks, 4 categories | 53.0% | ICLR 2026 Oral |
| **InnoGym** | 18 | Engineering + science innovation | Varies | ICLR 2026 |
| **NewtonBench** | Varies | Scientific law discovery | Varies | ICLR 2026 |

### 5.4 Software Engineering Benchmarks

| Benchmark | Size | Domain | Best Agent |
|-----------|------|--------|------------|
| **SWE-bench Verified** | 500 | Real GitHub issues | 75%+ |
| **SWE-bench Pro** | 1,865 | 41 repositories | Varies |
| **ACE-Bench** | Varies | Complex feature development | Varies |
| **SWE-Gym** | Training environment | RL for SWE agents | N/A |

### 5.5 Algorithm Design / Combinatorial Optimization

| Problem Set | Domain | Used By |
|-------------|--------|---------|
| **TSP instances** | Traveling salesman | ReEvo, EoH |
| **CVRP instances** | Vehicle routing | ReEvo |
| **Bin packing** | Operations research | FunSearch, EoH |
| **Sorting networks** | Computer science | AlphaEvolve |
| **Cap set problem** | Combinatorics | FunSearch |
| **Matrix multiplication** | Linear algebra | AlphaEvolve |

---

## 6. Open-Source Tools

### 6.1 Evolutionary Search Systems

| System | GitHub | Stars | GPU Need |
|--------|--------|-------|----------|
| ReEvo | [ai4co/LLM-as-HH](https://github.com/ai4co/LLM-as-HH) | ~500 | None (API) |
| FunSearch | [google-deepmind/funsearch](https://github.com/google-deepmind/funsearch) | ~1K | None (reference only) |
| OpenELM | [CarperAI/OpenELM](https://github.com/CarperAI/OpenELM) | ~400 | None (API) |
| OpenEvolve | [codelion/openevolve](https://github.com/codelion/openevolve) | 4.9K | None (API) |

### 6.2 Theorem Proving Infrastructure

| System | GitHub | Stars | GPU Need |
|--------|--------|-------|----------|
| Lean 4 | [leanprover/lean4](https://github.com/leanprover/lean4) | 7.4K | None (CPU) |
| Mathlib4 | [leanprover-community/mathlib4](https://github.com/leanprover-community/mathlib4) | 2.9K | None (CPU) |
| LEGO-Prover | [wiio12/LEGO-Prover](https://github.com/wiio12/LEGO-Prover) | ~100 | None (API) |
| ImProver | [riyazahuja/ImProver](https://github.com/riyazahuja/ImProver) | ~50 | None (API) |

### 6.3 Inference Engines

| System | GitHub | Stars | GPU Need |
|--------|--------|-------|----------|
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) | 71.3K | Multi-GPU |
| SGLang | [sgl-project/sglang](https://github.com/sgl-project/sglang) | 23.8K | Multi-GPU |

### 6.4 Scientific Discovery / Code Agents

| System | GitHub | Stars | GPU Need |
|--------|--------|-------|----------|
| AI Scientist | [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist) | 12.2K | NVIDIA GPU |
| OpenHands | [All-Hands-AI/OpenHands](https://github.com/All-Hands-AI/OpenHands) | 68.2K | None (API) |
| SWE-bench | [SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench) | ~3K | None (Docker) |
| ScienceAgentBench | [OSU-NLP-Group/ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench) | ~200 | None (API) |
| TreeQuest | [pfnet-research/TreeQuest](https://github.com/pfnet-research/TreeQuest) | ~50 | None (API) |

---

## 7. Research Gaps

### Gap 1: No Multi-GPU Agent Inference Benchmark [CRITICAL]
No public benchmark measures inference-serving efficiency under multi-agent workloads (throughput, latency, KV cache efficiency as agent count N scales). This is the bottleneck for ALL identified multi-GPU tasks.

### Gap 2: No Standardized Evolutionary Search Benchmark [HIGH]
FunSearch, AlphaEvolve, ReEvo, EoH each use different problems and metrics. A unified benchmark would enable fair comparison and measure compute efficiency (quality per LLM call).

### Gap 3: Scientific Discovery Benchmarks Ignore Multi-GPU Scaling [MODERATE]
All 7 scientific discovery benchmarks evaluate single agents. None measure parallel hypothesis testing, wall-clock speedup from parallelism, or GPU utilization efficiency.

### Gap 4: No Agentic Workload Profiling Data [HIGH]
No public dataset of real agent execution traces exists. Agentic workloads differ fundamentally from batch inference (bursty, multi-turn, variable-length), but no profiling data enables optimization.

### Gap 5: Theorem Proving pass@K Protocol Not Standardized [MODERATE]
Papers report pass@1, pass@32, pass@3200, pass@65536 — all incomparable. No standard for reporting GPU hours or infrastructure cost alongside pass rates.

---

## 8. Conclusions

### 8.1 Answer to the Research Question

**Tasks that genuinely need many LLM agents across multi-GPU nodes** (ranked by GPU demand and agent coordination):

| Rank | Task | True Agent Count | Coordination | Benchmark Maturity |
|------|------|-----------------|-------------- |-------------------|
| 1 | **Evolutionary algorithm discovery** | 15-200+ | High (island model) | Low (no unified benchmark) |
| 2 | **Parallel formal theorem proving** | 100-65K samples | Moderate (tree search) | High (miniF2F, PutnamBench) |
| 3 | **Test-time compute scaling** | 100-1000+ samples | None (independent) | High (MATH, AIME) |
| 4 | **Parallel code evaluation** | 500-8K instances | None (independent) | High (SWE-bench) |
| 5 | **Scientific discovery** | 3-7 agents | Pipeline | Moderate (ScienceAgentBench) |

### 8.2 Three Key Results

1. **Search + Verify = Multi-GPU Need**: Every multi-GPU agent task follows the generate-verify-select pattern. Automated verification is the enabling condition — without it, scaling provides diminishing returns (Ω(√H) gap).

2. **Three Architectures Cover All Cases**: Evolutionary islands (sublinear scaling, high coordination), parallel tree search (adaptive scaling), and embarrassingly parallel sampling (linear scaling) represent the complete design space.

3. **Model Quality Eventually Dominates Sampling Quantity**: A 32B model at pass@32 matches a 671B model at pass@8192 in theorem proving. But for exploration in unknown spaces, parallelism remains critical (FunSearch finding new mathematical constructions after 24+ hours of search).

### 8.3 The Biggest Gap

**No unified benchmark exists for evaluating multi-agent inference-serving efficiency on GPU clusters.** All multi-GPU agent tasks (evolutionary search, theorem proving, test-time scaling, code evaluation) require high-throughput LLM inference serving, but no public benchmark measures this specifically. This is the single most impactful gap in the field.

---

## References

### Peer-Reviewed Papers (sorted by relevance)

#### Evolutionary Search
- [@ye2024reevo] Ye et al. "ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution." NeurIPS 2024.
- [@liu2024eoh] Liu et al. "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model." ICML 2024 Oral.
- [@huang2026calm] Huang et al. "CALM: Co-evolution of Algorithms and Language Model for Automatic Heuristic Design." ICLR 2026.
- [@hu2025partition] Hu & Zhang. "Partition to Evolve: Niching-enhanced Evolution with LLMs for Automated Algorithm Discovery." NeurIPS 2025.
- [@zheng2025mcts] Zheng et al. "Monte Carlo Tree Search for Comprehensive Exploration in LLM-Based Automatic Heuristic Design." ICML 2025.

#### Theorem Proving
- [@wang2024lego] Wang et al. "LEGO-Prover: Neural Theorem Proving with Growing Libraries." ICLR 2024 Oral.
- [@poesia2024intrinsic] Poesia et al. "Learning Formal Mathematics From Intrinsic Motivation." NeurIPS 2024 Oral.
- [@ahuja2025improver] Ahuja et al. "ImProver: Agent-Based Automated Proof Optimization." ICLR 2025.
- [@dong2025stp] Dong & Ma. "STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving." ICML 2025.
- [@gu2026proofoptimizer] Gu et al. "ProofOptimizer: Training Language Models to Simplify Proofs without Human Demonstrations." ICLR 2026.
- [@dekoninck2026opc] Dekoninck et al. "The Open Proof Corpus: A Large-Scale Study of LLM-Generated Mathematical Proofs." ICLR 2026.
- [@ma2026proofbench] Ma et al. "Reliable Fine-Grained Evaluation of Natural Language Math Proofs." ICLR 2026.

#### Test-Time Scaling
- [@snell2025scaling] Snell et al. "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning." ICLR 2025 Oral.
- [@zhao2025sample] Zhao et al. "Sample, Scrutinize and Scale: Effective Inference-Time Search by Scaling Verification." ICML 2025.
- [@setlur2025suboptimal] Setlur et al. "Scaling Test-Time Compute Without Verification or RL is Suboptimal." ICML 2025 Spotlight.
- [@inoue2025wider] Inoue et al. "Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search." NeurIPS 2025 Spotlight.
- [@sadhukhan2025kinetics] Sadhukhan et al. "Kinetics: Rethinking Test-Time Scaling Law." NeurIPS 2025.
- [@zeng2026asymmetric] Zeng et al. "Pushing Test-Time Scaling Limits of Deep Search with Asymmetric Verification." ICLR 2026.

#### Scientific Discovery
- [@chen2025scienceagentbench] Chen et al. "ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery." ICLR 2025.
- [@wu2026innovatorbench] Wu et al. "InnovatorBench: Evaluating Agents' Ability to Conduct Innovative AI Research." ICLR 2026.
- [@tang2025airesearcher] Tang et al. "AI-Researcher: Autonomous Scientific Innovation." NeurIPS 2025 Spotlight.
- [@liu2026moscibench] Liu et al. "Towards Multimodal Data-Driven Scientific Discovery Powered by LLM Agents." ICLR 2026.
- [@panigrahi2026heurekabench] Panigrahi et al. "HeurekaBench: A Benchmarking Framework for AI Co-scientist." ICLR 2026.
- [@bragg2026astabench] Bragg et al. "AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite." ICLR 2026 Oral.
- [@zhang2026scinav] Zhang & Sun. "SciNav: A Principled Agent Framework for Scientific Coding Tasks." ICLR 2026.

#### Software Engineering
- [@antoniades2025swe] Antoniades et al. "SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement." ICLR 2025.
- [@pan2025swegym] Pan et al. "Training Software Engineering Agents and Verifiers with SWE-Gym." ICML 2025.
- [@zhang2025diversity] Zhang et al. "Diversity Empowers Intelligence: Integrating Expertise of Software Engineering Agents." ICLR 2025.

### Preprints (supplementary)
- Romera-Paredes et al. "Mathematical discoveries from program search with large language models." Nature, 2024.
- DeepMind. "AlphaEvolve: A coding agent for scientific and algorithmic discovery." 2025 (preprint).
- "BFS-Prover-V2." 2025 (preprint).
- "PaCoRe: Parallel Consensus Reasoning." 2026 (preprint).
- Lu et al. "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery." 2024 (preprint).
