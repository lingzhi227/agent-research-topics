# Mixed-Representation Reasoning for LLMs: A Comprehensive Survey

> **Date**: 2026-02-24
> **Scope**: Systematic survey of ~46 curated papers (642 total screened) across 6 thematic clusters
> **Period**: 2022–2026, with emphasis on 2024–2026
> **Database**: 70 filtered peer-reviewed papers + survey coverage

---

## Abstract

Large language models (LLMs) default to natural language (NL) as their reasoning medium, yet NL is demonstrably suboptimal for many reasoning tasks — it is verbose, sequential, and lacks formal verification guarantees. A rapidly growing body of research (2022–2026) explores alternative reasoning representations: executable code, pseudocode plans, latent continuous embeddings, graph structures, formal mathematical proofs, and domain-specific scientific notations. This survey provides the first unified perspective on **mixed-representation reasoning** — the idea that LLMs should dynamically select the most appropriate representational format for each reasoning step. We organize ~50 papers into a four-level taxonomy (human-readable structured, LLM-native latent, hybrid, and dynamic selection), present a comparative analysis across seven representative methods, identify seven cross-paper insights, and propose a five-direction research agenda. Our central finding is that the field is converging toward adaptive multi-format systems, but critical gaps remain: no unified framework, no multi-format benchmark, and no learned per-step representation selection mechanism.

---

## 1. Introduction

### 1.1 Motivation

Chain-of-thought (CoT) prompting [Wei et al., 2022] established that LLMs reason better when they "think step by step" in natural language. However, natural language is a general-purpose communication medium, not an optimized reasoning substrate. Several lines of evidence suggest NL is suboptimal:

- **Verbosity**: LightThinker [@zhang2025lightthinker] shows that 70% of reasoning tokens are redundant — models maintain 99% accuracy with only 30% of tokens.
- **Sequential bottleneck**: COCONUT [@hao2025coconut] demonstrates that continuous latent representations enable BFS-like exploration of multiple reasoning paths simultaneously, which sequential NL text fundamentally cannot express.
- **Noise injection**: Token Assorted [@su2025token_assorted] shows that replacing scaffolding tokens with latent codes *improves* accuracy by 4–13%, suggesting NL filler tokens degrade reasoning quality.
- **Lack of verification**: NL reasoning claims cannot be automatically verified, whereas code can be executed and formal proofs can be checked.

These observations motivate the core question of this survey: **What is the optimal reasoning representation for each class of problems, and can LLMs learn to dynamically select it?**

### 1.2 Scope and Contributions

This survey covers research published between 2022 and early 2026, drawing from top venues (ICLR, ICML, NeurIPS, ACL, EMNLP, AAAI, Nature, Nature Machine Intelligence) and key arXiv preprints. We make the following contributions:

1. **Unified taxonomy**: A four-level, 20+ subcategory taxonomy organizing all mixed-representation reasoning approaches (Section 2).
2. **Comparative analysis**: Quantitative comparison of seven representative methods across eight dimensions (Section 3).
3. **Representation spectrum**: A two-dimensional framework (information density × expressiveness) that positions all methods and identifies the research frontier (Section 4).
4. **Cross-paper insights**: Seven insights synthesized across 46+ papers, including the "verbosity tax," the training-free/training-heavy divide, and the missing "universal router" (Section 5).
5. **Gap analysis and research agenda**: Five critical gaps and a five-direction research agenda toward adaptive multi-representation reasoning (Section 6).
6. **Open-source ecosystem map**: Catalog of 23+ repositories, 11+ benchmarks, and their limitations (Section 7).

### 1.3 Related Surveys

Three recent surveys cover adjacent territory:
- **"Code to Think, Think to Code"** (EMNLP 2025) [@wang2025code_to_think] — surveys code↔reasoning bidirectionally but does not cover latent or formal math representations.
- **"Reasoning Beyond Language"** (arXiv 2025) [@chen2025reasoning_beyond_language] — the first latent CoT survey, covering ~50 papers on continuous/compressed reasoning but not code or domain-specific formats.
- **"Stop Overthinking"** (TMLR 2025) [@sui2025stop_overthinking] — surveys efficient reasoning methods but focuses on token reduction rather than representation selection.

**Gap**: No existing survey covers the unified perspective of optimal representation selection across all formats — code, latent, formal math, graphs, and domain-specific notations. This survey fills that gap.

---

## 2. Taxonomy of Approaches

We organize mixed-representation reasoning into four top-level categories with 20+ subcategories:

### Category A: Human-Readable Structured Representations

These preserve human interpretability while providing structural constraints superior to free-form NL.

**A1. Executable Code** (PAL, PoT, CodeAct, CodeIO). The most mature subcategory. PAL [@gao2023pal] achieves +15% over CoT on GSM8K by generating Python programs as reasoning steps. CodeAct [@wang2024codeact] unifies agent actions as executable code (+20% success rates). CodeIO [@li2025codeio] (ICML 2025 Oral) condenses code reasoning into input-output prediction patterns. Key advantage: automatic verification through runtime execution.

**A2. Pseudocode / Code-form Plans** (CodePlan, CREATOR). CodePlan [@wen2025codeplan] demonstrates +25.1% relative improvement across 13 benchmarks using pseudocode as a reasoning blueprint — leveraging control flow (conditionals, loops) and decomposition (function calls) without requiring execution.

**A3. Mathematical Notation** (AlphaProof, DeepSeek-Prover-V2). Formal proof languages provide the highest verification guarantee. AlphaProof [@alphaproof2025] achieved IMO silver medal level (28/42 points) in Lean. DeepSeek-Prover-V2 [@deepseek2025proverv2] reached 88.9% on MiniF2F-test in Lean 4.

**A4. Chemical/Molecular Formulas** (ChemCrow, MolRAG, CleanMol). SMILES notation and molecular fingerprints provide domain-specific compressed representations. ChemCrow [@bran2024chemcrow] integrates 18 chemistry tools operating on these representations.

**A5. Graph/Diagram Structures** (Graph of Thoughts, DAG-Math). GoT [@besta2024got] models reasoning as an arbitrary graph, achieving +62% quality improvement over Tree-of-Thoughts on sorting tasks.

**A6. Formal Logic** (ProverGen, FLD). ProverGen [@provergen2025] integrates symbolic provers (Prover9) into FOL reasoning, and FLD [@fld2024] shows that training on synthetic formal logic corpora substantially enhances reasoning.

### Category B: LLM-Native Latent Representations

These exist in the model's internal representation space and are not directly human-interpretable.

**B1. Continuous Thoughts** — COCONUT [@hao2025coconut] feeds the last hidden state directly back as the next input embedding, creating "continuous thoughts" that can encode multiple simultaneous reasoning branches (97% on ProsQA).

**B2. Gist Tokens** — LightThinker [@zhang2025lightthinker] trains LLMs to compress intermediate reasoning into gist tokens during generation, achieving 70% peak token reduction with ~1% accuracy loss.

**B3. VQ-VAE Latent Codes** — Token Assorted [@su2025token_assorted] uses a VQ-VAE codebook to produce discrete latent tokens, achieving +4–13% accuracy with 17% trace length reduction.

**B4. Soft Thought Tokens** — SoftCoT [@xu2025softcot] uses a lightweight assistant model to generate continuous intermediate representations projected into the main LLM's embedding space.

**B5. Activation Spaces** — ActivationReasoning (ICLR 2026) performs logical reasoning directly in intermediate transformer layer activations.

**B6. Diffusion-Based** — Diffusion of Thought [@ye2024diffusion_of_thought] replaces autoregressive reasoning with a diffusion process enabling self-correction and parallel refinement.

**B7. Pause/Filler Tokens** — Pause tokens [@goyal2024pause] provide extra computation yielding +18% EM on SQuAD without linguistic content.

### Category C: Hybrid Representations

These combine two or more representation types within a single reasoning process.

- **C1. Latent + Text Mixing**: Token Assorted interleaves VQ-VAE codes with text — early scaffolding steps tolerate compression while critical derivation steps require text.
- **C2. Code + NL Interleaving**: MathCoder [@wang2024mathcoder] seamlessly switches between NL reasoning, code generation, and execution result interpretation.
- **C3. NL + Tool Outputs**: ChemCrow mixes NL reasoning with 18 chemistry tools producing domain-specific outputs (molecular properties, reaction predictions).
- **C4. Formal + Informal Math**: DeepSeek-Prover-V2 uses NL proof sketches from DeepSeek-V3, then 7B model conducts formal Lean 4 proof search for each subgoal.
- **C5. Multi-Modal**: ThinkMorph (ICLR 2026) interleaves text and visual reasoning in unified CoT.
- **C6. Semi-Latent**: LightThinker creates a hybrid where some reasoning exists as full text, some as compressed gist tokens.

### Category D: Dynamic Format Selection

These select or adapt the reasoning format based on problem properties.

- **D1. Cognitive Paradigm Routing**: Sketch-of-Thought [@aytes2025sot] defines three cognitive paradigms and uses a lightweight router for per-problem selection, achieving up to 84% token reduction.
- **D2. Complexity-Aware Allocation**: Learning How Hard to Think [@damani2025hard_to_think] routes between cheap/fast and expensive/accurate decoding strategies.
- **D3. Format Generation & Selection**: Format-Adapter [@format_adapter2025] auto-generates suitable reasoning formats per task.
- **D4. Adaptive CoT Length**: Ada-R1 (NeurIPS 2025) uses bi-level optimization to adjust reasoning depth.

---

## 3. Comparative Analysis

We compare seven representative methods across eight dimensions:

| Dimension | NL CoT | Code (PAL/PoT) | Pseudocode (CodePlan) | Latent (COCONUT) | Hybrid (Token Assorted) | Compressed (LightThinker) | Adaptive (SoT) |
|---|---|---|---|---|---|---|---|
| **Token Efficiency** | Low | Medium | Medium-High | Very High | High (-17%) | Very High (-70%) | High (-84%) |
| **Accuracy (Math)** | Baseline | +12–15% | +25.1% relative | Variable | +4–13% | Baseline -1% | Baseline ±1% |
| **Interpretability** | Very High | High | High | None | Low-Medium | Low-Medium | Medium-High |
| **Verifiability** | Low | Very High (execution) | Medium | None | Low | Low | Medium |
| **Training Complexity** | None | Low | Medium (2M examples) | High (curriculum) | High (VQ-VAE + FT) | High | None (prompting) |
| **Generality** | Very High | Medium | High (13 benchmarks) | Medium | Medium | Medium-High | High (18 benchmarks) |
| **Scalability** | Poor (linear growth) | Good | Good | Excellent (constant) | Good | Excellent | Good |

### Key Tradeoffs

1. **Interpretability vs. Efficiency**: Clear inverse relationship. NL CoT is maximally interpretable but least efficient; COCONUT is the reverse. Hybrid approaches (Token Assorted, LightThinker) occupy the middle.

2. **Training Cost vs. Inference Savings**: Training-free methods (SoT, PAL) achieve 2–3× token reduction and 5–25% accuracy improvement. Training-heavy methods (COCONUT, Token Assorted) achieve 3–10× reduction with comparable or better accuracy.

3. **Verifiability vs. Compression**: Executable code offers the best verifiability; latent methods offer the best compression. No method yet achieves both. This motivates **hybrid verification** — latent exploration with periodic code/proof checkpoints.

4. **Generality vs. Specialization**: NL CoT and SoT are the most general. Code-based methods excel on computational tasks but struggle with subjective reasoning. Latent methods have primarily been evaluated on math/logic. No method demonstrates strong performance across ALL task categories.

---

## 4. The Representation Spectrum

### Conceptual Framework

All reasoning representations can be organized along two axes:

- **X-axis: Information Density** — tokens per reasoning step. NL CoT uses many tokens per step (low density); COCONUT uses a single continuous vector (maximum density).
- **Y-axis: Expressiveness / Generality** — breadth of tasks the representation handles. Domain-specific notations (SMILES, Lean) are narrow; NL and latent representations are general.

```
                    Expressiveness / Generality
                    ^
              High  |  NL CoT                                COCONUT
                    |         SoT (adaptive)        Token Assorted
                    |                CodePlan
                    |   GoT              LightThinker
           Medium   |    PAL/PoT (code)        SoftCoT
                    |         MathCoder              CCoT
              Low   |  SMILES/chemical                Pause Tokens
                    |  Lean/Isabelle proofs      Activation Reasoning
                    +────────────────────────────────────────────────>
                   Low                                           High
                   (many tokens/step)              (few tokens/step)
                              Information Density
```

**The Frontier Line**: The most impactful recent papers (Token Assorted, SoT, LightThinker, CodePlan) all sit along a diagonal from upper-left to upper-right, representing progress toward representations that are both general and dense.

**The Ideal System** would:
1. Operate in the upper-right quadrant (latent/compressed, general) for most reasoning
2. Switch to the lower-left (formal proofs, executable code) for domain-specific verification
3. Switch to the upper-left (NL explanations) for communication
4. Use dynamic routing (SoT-style) to select representations per subproblem

---

## 5. Cross-Paper Insights

### Insight 1: The Verbosity Tax

Multiple papers independently demonstrate that NL verbosity is not just wasteful but sometimes harmful. Token Assorted shows replacing scaffolding tokens with latent codes *improves* accuracy by 4–13%. COCONUT shows continuous thoughts enable BFS exploration impossible in sequential text. LightThinker shows models maintain 99% accuracy with 30% of tokens. The community's default assumption that "more CoT tokens = better reasoning" is being overturned.

### Insight 2: The Training-Free vs. Training-Heavy Divide

Training-free methods (SoT, PAL/PoT, GoT) offer immediate deployment with 2–3× efficiency gains. Training-heavy methods (COCONUT, Token Assorted, LightThinker, CodePlan) offer 3–10× gains. A practical two-phase deployment strategy: start with prompting-based routing, invest in latent/compressed training for high-volume workloads.

### Insight 3: Cognitive Hierarchy Parallel

The progression from NL CoT → structured → compressed → latent parallels human expertise development: novice (verbose, explicit) → intermediate (structured shorthand) → expert (compressed, intuitive). SoT explicitly cites this cognitive parallel; LightThinker draws from human chunking theory.

### Insight 4: Verification Remains Unsolved

Code-based reasoning is fully verifiable via execution; formal proofs are verified by proof checkers; NL is human-auditable. But latent methods (COCONUT, Token Assorted, LightThinker) are completely opaque. No paper implements a full latent-with-checkpoints verification system.

### Insight 5: The Missing Universal Router

SoT routes among 3 NL-based paradigms. Learning How Hard to Think routes between fast/slow strategies. Ada-R1 adapts CoT length. No published work implements a router that selects among the full range: NL, code, pseudocode, latent tokens, formal math, graph structures, and domain-specific notations.

### Insight 6: Representation > Scale

CodePlan with a smaller model outperforms larger models with direct generation. Token Assorted with Llama-3.2-1B + latent tokens approaches Llama-3.2-3B with standard CoT. Representation optimization is complementary to and potentially more cost-effective than scaling.

### Insight 7: Field Fragmentation

As of early 2026: 3 published surveys, ~50 papers across 6 top venues, 20+ GitHub repos — but no unified benchmark, no unified framework, no standard evaluation protocol. Each paper evaluates on different benchmarks with different base models.

---

## 6. Open Problems and Research Agenda

### 6.1 Critical Gaps

**Gap 1: No Unified Multi-Representation Framework.** The closest works are Token Assorted (2 modalities: latent + text) and MathCoder (3 modalities: NL + code + execution). Neither incorporates formal proofs, graph structures, or domain-specific notations.

**Gap 2: No Multi-Format Reasoning Benchmark.** Existing benchmarks evaluate reasoning within a single paradigm (GSM8K for math-CoT, MiniF2F for formal proofs, BIG-Bench Hard for text-CoT). No benchmark evaluates whether a model can *choose* the right format.

**Gap 3: No Learned Per-Step Format Selection.** SoT selects per-problem (not per-step) among text-only paradigms. No system learns: "At step 3, switch to Python; at step 5, switch to Lean; at step 7, switch to latent tokens."

**Gap 4: Latent Reasoning Limited to Small Models.** COCONUT was demonstrated on GPT-2 scale; Token Assorted caps at 8B parameters. Scaling behavior at 70B+ is uncharacterized.

**Gap 5: Bioinformatics/Genomics Unexplored.** While chemistry has ChemCrow (18 tools), bioinformatics has no system combining sequence alignment notation (FASTA/SAM/BAM), statistical models, pathway graphs (KEGG/Reactome), and code-based pipelines.

### 6.2 Research Agenda

**Direction 1: Adaptive Multi-Representation Controller (AMRC).** Define a representation action space {NL, Python, Lean 4, latent tokens, graph, domain notation}. Train a lightweight router that classifies each reasoning sub-step, with representation-specific modules (CodePlan for code, COCONUT for latent, DeepSeek-Prover-V2 for formal math). Key technical innovation: a shared embedding space hosting text, code, latent, and formal tokens (extending Token Assorted's VQ-VAE from 2 to 6+ modalities).

**Direction 2: Multi-Format Reasoning Benchmark (MFRB).** Task categories with known optimal representations: arithmetic (code), combinatorial search (latent), theorem verification (formal proof), entity tracking (graph), molecular prediction (SMILES + code). Evaluation metrics: answer accuracy, representation efficiency, format selection accuracy, cross-format coherence score.

**Direction 3: Scaling Latent Reasoning to Frontier Models.** Replicate COCONUT on 70B+ models. Evaluate on complex real-world tasks (GPQA Diamond, FrontierMath). Test whether latent BFS advantage grows with problem difficulty.

**Direction 4: Verifiable Latent Reasoning.** Explore in latent space (COCONUT-style BFS), translate candidates to verifiable format (code or Lean), verify and select. Combines the search efficiency of latent reasoning with the guarantees of formal verification.

**Direction 5: Scientific Multi-Format Reasoning.** Build end-to-end systems for drug discovery (SMILES + protein structures + docking code + ADMET statistics + pathway graphs) and bioinformatics (FASTA + SAM/BAM + KEGG pathways + statistical models + Python pipelines).

---

## 7. Open-Source Ecosystem

### Key Repositories

| Category | Project | Stars | Venue | Status |
|---|---|---|---|---|
| Code-as-Reasoning | PAL | ~510 | ICML 2023 | Stable |
| | CodeAct | ~1,600 | ICML 2024 | Active (OpenHands) |
| | CodePlan | ~50+ | ICLR 2025 | Active |
| Latent Reasoning | COCONUT | ~1,200 | ICLR 2025 | Active (Meta FAIR) |
| | SoftCoT | ~74 | ACL 2025 | Active |
| | Diffusion of Thought | ~100+ | NeurIPS 2024 | Research |
| Structured | Graph of Thoughts | ~2,400 | AAAI 2024 | Stable |
| Formal Math | DeepSeek-Prover-V2 | ~1,200 | 2025 | Active |
| | AlphaGeometry | ~4,000 | DeepMind | Stable |
| Scientific | ChemCrow | ~878 | NMI 2024 | Active |
| Efficiency | LightThinker | ~123 | EMNLP 2025 | Active |
| | Sketch-of-Thought | ~100+ | EMNLP 2025 | Active |

### Key Benchmarks

| Benchmark | Domain | Format | Size |
|---|---|---|---|
| GSM8K | Math | NL/Code | 8.5K problems |
| MATH | Competition math | LaTeX | 12.5K problems |
| MiniF2F | Formal math | Lean/Isabelle | 488 statements |
| PutnamBench | Undergrad math | Lean/Isabelle/Coq | 1,724 formalizations |
| BIG-Bench Hard | Diverse reasoning | NL | 23 tasks |
| GPQA | Graduate science | NL (MCQ) | 448 questions |
| SciCode | Scientific code | Python | 16 subdomains |
| ProsQA | Graph traversal | Custom | Synthetic |

### Notable Gaps in the Ecosystem

1. **Token Assorted** (ICML 2025) has no public code repository.
2. **Format-Adapter** (the most directly relevant work on format selection) has no public code.
3. **AlphaProof** remains closed-source; only output artifacts are mirrored.
4. **No unified multi-format reasoning benchmark** exists.

---

## 8. Evolution Timeline

```
2022  ── PAL, PoT ───────────────── Code replaces NL for math
        │
2023  ── GoT, Pause Tokens, CoC ── Graphs + first latent signals
        │
2024  ── CodeAct, DoT, MathCoder ─ Code actions + diffusion reasoning
        │      COCONUT ──────────── Continuous latent space (landmark)
        │
2025  ── Token Assorted ─────────── Hybrid latent+text (landmark)
        │  LightThinker ──────────── Gist token compression
        │  SoT ───────────────────── Dynamic format selection
        │  CodePlan ──────────────── Pseudocode as blueprint
        │  AlphaProof, DSP-V2 ────── Formal math breakthroughs
        │  3 Surveys ─────────────── Field consolidation begins
        │
2026  ── ActivationReasoning ────── Full latent activation space
        │  Unified frameworks ────── Convergence begins
        │  ??? ───────────────────── Adaptive multi-format system?
```

---

## 9. Conclusion

The field of mixed-representation reasoning for LLMs has transitioned from isolated single-format approaches (code, graphs, formal proofs) to hybrid systems (latent + text, code + NL + execution) and nascent dynamic selection mechanisms. The evidence is now clear that **natural language is suboptimal for LLM reasoning** — alternative representations offer 2–10× efficiency gains and sometimes improve accuracy.

However, the field remains fragmented. No system combines the planning capability of pseudocode (CodePlan), the search efficiency of latent reasoning (COCONUT), the verification guarantee of formal proofs (DeepSeek-Prover-V2), the domain expertise of scientific tools (ChemCrow), and the dynamic routing of adaptive selection (Sketch-of-Thought) into a single coherent framework.

The convergence toward such a system is the most significant trend we observe. We expect the next 1–2 years to bring:
- **Multi-format benchmarks** that evaluate representation selection alongside answer correctness
- **Universal routers** that select among code, latent, formal, and domain-specific representations per reasoning step
- **Verified latent reasoning** that combines the efficiency of opaque representations with the guarantees of executable code and formal proofs
- **Scientific applications** in bioinformatics, drug discovery, and materials science that inherently require multi-format reasoning

The ultimate vision — an LLM that "thinks" in the most appropriate format for each problem, seamlessly switching between mathematical notation for proofs, Python for computation, latent tokens for search, and domain-specific notation for scientific reasoning — remains an open challenge and the most exciting frontier in LLM reasoning research.

---

## References

See `references.bib` for full bibliography. Key references organized by cluster:

**Code-as-Reasoning**: PAL (ICML 2023), PoT (TMLR 2023), CodeAct (ICML 2024), CodePlan (ICLR 2025), MathCoder (ICLR 2024), CodeIO (ICML 2025 Oral)

**Latent/Compressed**: COCONUT (ICLR 2025), Token Assorted (ICML 2025), SoftCoT (ACL 2025), Diffusion of Thought (NeurIPS 2024), CCoT (arXiv 2024, preprint), Pause Tokens (ICLR 2024)

**Structured/Symbolic**: Graph of Thoughts (AAAI 2024), ProverGen (ICLR 2025), FLD (NeurIPS 2024)

**Formal Math**: AlphaProof (Nature 2025), DeepSeek-Prover-V2 (2025), LEGO-Prover (ICLR 2024 Oral), FunSearch (Nature 2024)

**Scientific**: ChemCrow (Nature Machine Intelligence 2024), MolRAG (ACL 2025), CleanMol (EMNLP 2025)

**Efficiency**: LightThinker (EMNLP 2025 Oral), Sketch-of-Thought (EMNLP 2025), Learning How Hard to Think (ICLR 2025), Stop Overthinking Survey (TMLR 2025)

**Surveys**: Code to Think (EMNLP 2025), Reasoning Beyond Language (arXiv 2025, preprint), Stop Overthinking (TMLR 2025)
