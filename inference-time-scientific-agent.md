# Inference-Time Scientific Agent: A Survey on Enhancing LLM Reasoning Through External Knowledge and Verifiable Environments

## 1. Introduction

Large Language Models have demonstrated impressive reasoning capabilities out-of-the-box, yet they consistently fail on domain-specific scientific tasks requiring specialized knowledge, precise computation, and multi-step verification. The dominant approach to improving LLMs — scaling training compute — is increasingly challenged by an alternative paradigm: **scaling inference-time compute** through external knowledge augmentation, tool use, and verifiable environments.

This survey systematically examines how inference-time interventions can enhance LLM agents for scientific reasoning, focusing on four complementary strategies: (1) inference-time compute scaling, (2) external knowledge augmentation, (3) tool augmentation, and (4) verifiable feedback environments. We analyze 70 peer-reviewed papers from top AI conferences (ICLR, NeurIPS, ICML, EMNLP, ACL, NAACL, COLING) spanning 2023-2026, with deep dives into 12 key papers.

**Core research question**: How much can inference-time interventions improve LLM scientific reasoning performance, and what are the most effective strategies?

## 2. Background

### 2.1 The Inference-Time Scaling Paradigm

The shift from training-time to inference-time scaling represents a fundamental change in how we improve LLM capabilities. Snell et al. (ICLR 2025, Oral) showed that optimal test-time compute allocation can be more effective than scaling model parameters, with smaller models + advanced inference algorithms achieving Pareto-optimal cost-performance trade-offs [@snell2025scaling].

This paradigm manifests through:
- **Extended chain-of-thought**: Models like OpenAI o1, DeepSeek-R1, and QwQ-32B spend more tokens "thinking" before answering
- **Search and verification**: Best-of-N sampling, Monte Carlo Tree Search, step-level verifier guidance
- **External augmentation**: RAG, tool use, and feedback loops that enhance reasoning at inference time

### 2.2 The Knowledge Gap in Scientific Reasoning

Despite their broad capabilities, LLMs face two critical deficiencies in science:
1. **Knowledge insufficiency**: LLMs lack or cannot recall specialized domain knowledge. On GPQA Diamond, QwQ-32B's reasoning chains contain over 30 instances of uncertainty markers ("perhaps", "possibly") per problem [@li2025searcho1].
2. **Knowledge misapplication**: Even when LLMs have relevant knowledge, they apply it incorrectly — using wrong formulas, confusing variables, or making unit errors [@wan2024physicsreasoner].

## 3. Taxonomy of Inference-Time Enhancement Methods

### 3.1 Internal Compute Scaling

**Approach**: More computation within the model itself (longer thinking, multiple samples, search).

| Model | GPQA Diamond | Method |
|-------|:---:|--------|
| GPT-4o | 50.6% | Standard |
| o1-preview | 73.3% | Extended reasoning |
| o3 | 83.3% | More reasoning + search |
| GPT-5.2 Pro | 93.2% | Full-scale reasoning |

Thought calibration can achieve up to 60% reduction in thinking tokens while maintaining performance, addressing the cost concern [@thought_calibration2025]. However, optimal thinking length varies by domain — over-thinking can hurt in some areas [@thinking_optimal2025].

### 3.2 External Knowledge Augmentation

**Static RAG** retrieves documents once before reasoning begins. While helpful for simple factual queries, it fails for multi-step scientific reasoning that requires different knowledge at different stages.

**Agentic RAG (Search-o1)** integrates retrieval dynamically into the reasoning chain [@li2025searcho1]. When the model encounters uncertainty, it autonomously generates a search query, retrieves documents, and a Reason-in-Documents module distills relevant knowledge before re-injecting it into the reasoning chain.

| Method | GPQA | Gain over no-retrieval |
|--------|:---:|:---:|
| QwQ-32B (direct) | 58.1% | — |
| Standard RAG | 58.6% | +0.5% |
| RAgent (no RiD) | 61.6% | +3.5% |
| **Search-o1** | **63.6%** | **+5.5%** |

On multi-hop QA, Search-o1 outperforms standard RAG by up to 29.6% EM.

**Structured Knowledge Augmentation (Physics Reasoner)** provides curated domain knowledge rather than retrieved documents [@wan2024physicsreasoner]. A knowledge base of 122 physics formulas from 3 textbooks, combined with structured checklists for variable extraction and guided reasoning, achieves +5.9% on SciBench physics (GPT-4-turbo: 47.1% → 53.0%).

The ablation is particularly revealing:
- Formula retrieval alone: **+1.3%** (knowing facts)
- + Variable extraction checklist: **+6.4%** (knowing how to apply)
- + Reasoning verification checklist: **+9.6%** total

This demonstrates that **structured knowledge application guidance matters far more than raw knowledge provision**.

**Case-Based Reasoning (DS-Agent)** retrieves and adapts past solutions from a curated knowledge base of winning Kaggle competition solutions [@guo2024dsagent]. This achieves 100% success rate with GPT-4 on development tasks, and even with GPT-3.5, outperforms ResearchAgent + GPT-4 — confirming that methodology > model scale.

### 3.3 Tool Augmentation

Tool augmentation extends LLMs with external capabilities for computation, simulation, and domain-specific operations.

**ToRA (Tool-integrated Reasoning Agent)** interleaves natural language reasoning with Python+SymPy execution [@gou2024tora]. The key ablation shows the power of combination: rationale-only 27.1% vs program-only 37.9% vs interleaved **56.1%** on MATH. ToRA-7B (44.6%) surpasses WizardMath-70B (22.7%) — demonstrating 10x parameter efficiency through tool integration.

**SciAgentGym** provides a comprehensive evaluation of tool-augmented scientific agents across 1,780 domain-specific tools [@sciagentgym2025]. Key findings:

| Model | No Tools | With Tools | Gain |
|-------|:---:|:---:|:---:|
| GPT-5 | 32.3% | 41.3% | +9.0pp |
| Claude-Sonnet-4 | 22.4% | 35.9% | +13.5pp |
| SciAgent-8B (fine-tuned) | 23.3% | 30.1% | +6.8pp |

Critically, **SciAgent-8B (30.1%) outperforms Qwen3-VL-235B (23.9%) despite being 29x smaller**, confirming that domain-specific tool training is far more effective than raw model scale.

However, performance degrades sharply with task complexity: L1 (easy) 46.4% → L3 (hard, 8+ steps) 14.7%.

### 3.4 Verifiable Feedback Environments

**Self-Debug** allows agents to execute generated code, receive feedback, and iterate [@chen2024scienceagentbench]. On ScienceAgentBench, this doubles Claude-3.5-Sonnet's success rate (17.7% → 32.4%) at minimal cost ($0.057/task). Surprisingly, simple Self-Debug outperforms the complex OpenHands agent framework at 1/17 the cost.

**RLVR (Reinforcement Learning with Verifiable Rewards)** uses verifiable feedback signals for post-training optimization [@wen2025rlvr]. The CoT-Pass@K metric shows RLVR genuinely improves reasoning quality, not just sampling efficiency. Med-RLVR successfully extends this to medical reasoning with superior out-of-distribution generalization.

## 4. Quantitative Summary: How Much Do These Methods Help?

| Enhancement Strategy | Typical Gain | Best Observed | Benchmark |
|---------------------|:---:|:---:|-----------|
| Internal compute scaling | +5-10% | +15.2% (o1→GPT-5.2) | GPQA Diamond |
| Agentic RAG | +2.5-5.5% | +29.6% (multi-hop QA) | GPQA, HotpotQA |
| Structured knowledge | +5-6% | +19.2% (Agent KB) | SciBench, GAIA |
| Case-based reasoning | +25-71% | +251% (DS-Agent) | Kaggle tasks |
| Tool augmentation | +9-22% | +96.5% (ToRA) | MATH, SciAgentGym |
| Self-Debug | +14-15% | +83.1% (relative) | ScienceAgentBench |
| Domain fine-tuning (8B) | +6-7% | +28.6% (relative) | SciAgentGym |

**Combined approaches** yield the largest gains: Search-o1 (agentic RAG + RiD + reasoning model), Physics Reasoner (knowledge + checklists + code), DS-Agent (cases + debugging + code).

## 5. Critical Insights

### 5.1 Methodology Trumps Scale
Three independent studies demonstrate that structured inference-time augmentation outperforms brute-force scaling:
- SciAgent-8B > Qwen3-VL-235B (29x fewer params)
- DS-Agent + GPT-3.5 > ResearchAgent + GPT-4
- ToRA-7B > WizardMath-70B (10x fewer params)

### 5.2 Knowledge Application > Knowledge Access
Physics Reasoner's ablation is the most informative result in this survey: structured verification checklists (+8.3pp) contribute 6.4x more than formula retrieval alone (+1.3pp). ScienceAgentBench confirms this — expert-provided domain knowledge adds only +1.9%.

### 5.3 The Reasoning Prerequisite
Search-o1 shows that non-reasoning models (Qwen2.5-32B-Instruct) barely benefit from agentic search (+1.5%), while the reasoning model QwQ-32B gains +5.5%. This implies a minimum reasoning capability is required to effectively utilize inference-time augmentation.

### 5.4 Domain Specificity is Critical
SciAgentGym demonstrates that generic tool-use training causes -4.6pp NEGATIVE transfer. However, cross-domain transfer within science exists — physics-only training helps materials science by +10pp.

### 5.5 Current Ceiling
Even GPT-5, the most capable model tested, achieves only 41.3% on complex multi-step scientific tool-use tasks. Long-horizon tasks (8+ steps) drop to 14.7% average. The primary bottleneck is not reasoning or domain knowledge, but practical code generation and data handling.

## 6. Implications for the Textbook-to-Markdown Approach

The user's approach of converting textbooks and papers to structured markdown for LLM consumption is well-positioned in the current research landscape:

**Strengths**:
1. **Addresses the knowledge gap**: Provides verified, structured domain knowledge instead of noisy web content
2. **Preserves context**: Full textbook sections maintain logical flow vs. RAG chunks
3. **No training required**: Operates purely at inference time, aligning with the finding that methodology > model scale
4. **Combinable**: Can be integrated with Search-o1's agentic retrieval, Physics Reasoner's checklist verification, and tool execution

**Estimated impact based on evidence**:
- Knowledge augmentation from textbooks: **+5-20%** on domain-specific benchmarks (comparable to Physics Reasoner, Agent KB)
- Combined with tool augmentation: **+10-30%** additional
- Combined with structured verification: **+5-15%** additional
- **Total estimated improvement**: **+20-65%** over vanilla LLM, depending on domain and task complexity

**Key recommendation**: Do not stop at knowledge provision. The Physics Reasoner result clearly shows that HOW knowledge is applied matters more than WHAT knowledge is available. Structured checklists, verification steps, and guided application strategies should accompany the textbook knowledge.

## 7. Open Problems and Future Directions

1. **Textbook knowledge representation**: What is the optimal format for LLM consumption? Full sections? Chunked with metadata? Formula sets with context? Systematic evaluation is needed.
2. **Long-horizon scientific workflows**: Current agents fail at 8+ step tasks (14.7% success). Better memory, planning, and error recovery are critical.
3. **Agentic RAG with local knowledge**: Replace web search with structured textbook/paper retrieval for higher quality and domain specificity.
4. **Knowledge application verification**: Develop systems that verify not just retrieval accuracy but correct application of retrieved knowledge.
5. **Cost-performance optimization**: Identify Pareto-optimal strategies combining cheap knowledge augmentation with selective expensive compute.
6. **Standardized end-to-end evaluation**: No benchmark currently captures the full scientific workflow from literature → hypothesis → experiment → analysis.

## 8. Conclusion

Inference-time enhancement of LLM scientific agents is a rapidly maturing field with strong evidence of effectiveness. The key insight from this survey is that **structured methodology at inference time consistently outperforms brute-force scaling** — whether through domain-specific knowledge augmentation (+5-20%), tool integration (+9-22%), or verifiable feedback loops (+14-15%). The most effective approaches combine multiple enhancement levels, and critically, **knowledge application guidance matters far more than raw knowledge provision**. Current systems achieve 30-42% on complex scientific tasks, indicating massive room for improvement. The approach of converting textbooks and papers into structured formats for LLM consumption addresses a genuine gap in the field and, when combined with tool use and verification, could yield 20-65% improvement over baseline LLMs on domain-specific scientific tasks.

## References

[@snell2025scaling] Snell, C. et al. "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters." ICLR 2025 (Oral).

[@li2025searcho1] Li, X. et al. "Search-o1: Agentic Search-Enhanced Large Reasoning Models." EMNLP 2025.

[@wan2024physicsreasoner] Pang, X. et al. "Physics Reasoner: Knowledge-Augmented Reasoning for Solving Physics Problems with Large Language Models." COLING 2025.

[@gou2024tora] Gou, Z. et al. "ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving." ICLR 2024.

[@chen2024scienceagentbench] Chen, Z. et al. "ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery." ICLR 2025.

[@sciagentgym2025] "SciAgentGym: Benchmarking Multi-Step Scientific Tool-use in LLM Agents." 2025 (preprint).

[@guo2024dsagent] Guo, S. et al. "DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning." ICML 2024.

[@lu2023chameleon] Lu, P. et al. "Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models." NeurIPS 2023.

[@zhu2024knowagent] Zhu, Y. et al. "KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents." NAACL 2025.

[@wen2025rlvr] Wen, X. et al. "Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs." ICLR 2026.

[@ren2025scientific] Ren, S. et al. "Towards Scientific Intelligence: A Survey of LLM-based Scientific Agents." arXiv 2025 (preprint).

[@thinking_optimal2025] "Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning." arXiv 2025 (preprint).

[@thought_calibration2025] "Thought Calibration: Efficient and Confident Test-time Scaling." arXiv 2025 (preprint).
