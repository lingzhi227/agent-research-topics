# Agent Research Topics

Systematic literature surveys on LLM agent capabilities, generated via a 6-phase deep research pipeline (frontier scan → broad survey → deep dive → code/tools mapping → cross-paper synthesis → final report).

## Reports

| Report | Papers | Focus |
|--------|--------|-------|
| [Agent Coordination](agent-coordination.md) | 75+ | Inference-time multi-agent coordination — topology, communication, memory, routing, security, and protocol standards (MCP, A2A) across two paradigms (Agent OS vs. Dynamic Orchestration) |
| [Agent OS: Context Engineering](agent-os-context-engineering.md) | 85 | Agent OS paradigm for LLM benchmark enhancement — context engineering, memory management (MemoryOS +49% F1), multi-agent coordination, and inference-time compute allocation without model retraining |
| [Agent OS: Scientific Reasoning Benchmarks](agent-os-scientific-reasoning-benchmarks.md) | 75 | Quantitative mapping of Agent OS methods to scientific reasoning benchmarks — scaffold design yields 2.7x–11x performance gaps; MCTS as universal scaffold; identifies that only 4/14 major methods evaluated on scientific tasks |
| [Inference-Time Scientific Agent](inference-time-scientific-agent.md) | 70 | Enhancing LLM scientific reasoning via inference-time compute scaling, external knowledge augmentation, tool use, and verifiable feedback environments |
| [LLM Agent Action Chain Optimization](llm-agent-action-chain-optimization.md) | 70 | Action chain formal representation, verification, and efficiency optimization — FSM workflow modeling, LTL/model checking, constrained decoding, and code-as-action |
| [LLM Agent Bioinformatics Mapping](llm-agent-bioinformatics-mapping.md) | 48 | Systematic mapping of 14 concept pairs between bioinformatics and LLM agent infrastructure — workflow orchestration, provenance, knowledge representation, tool ecosystems, and a cross-pollination roadmap |
| [LLM Agent Formal Knowledge Representation](llm-agent-formal-knowledge-representation.md) | 74 | Using code, DSLs, and formal languages (instead of natural language) to represent agent knowledge — covering programmatic policies, learned reward functions, tool/API abstractions, and the PRAL research line |
| [LLM Agent Paper Writing](llm-agent-paper-writing.md) | 23 projects | Open-source frameworks for LLM-driven automated scientific paper writing — from literature review to experiments, figures, LaTeX generation, and peer review |
| [LLM Agentic Benchmarks & Long Context](llm-agentic-benchmarks-long-context.md) | 15 deep-read | LLM agent benchmark survey — long-context reasoning, scientific reasoning, tool calling, multi-step agent tasks, and evaluation methodology (dynamic benchmarks, token efficiency, process metrics) |
| [LLM Bio Tools & Workflow Compatibility](llm-bio-tools-workflow.md) | 65 | Tool I/O incompatibility in LLM agent bioinformatics workflows — protocol standardization (MCP), knowledge-graph routing, semantic type systems, and a proposed three-layer architecture |
| [LLM Knowledge Distillation & Reasoning](llm-knowledge-distillation-reasoning.md) | 70 | Enhancing LLM domain reasoning (math, biology, genomics, CS, statistics) via automatic prompt optimization, knowledge distillation, long-context retrieval, and document-grounded skill generation |
| [Long-Horizon Reasoning (2025H2)](long-horizon-reasoning-2025h2.md) | 72 | The 2025H2 reasoning revolution — latent reasoning in continuous space, efficiency methods (40–73% token reduction), CoT faithfulness crisis (19–41%), and the shift from "think more" to "think smarter" |
| [Long-Horizon Reasoning Agent](long-horizon-reasoning-agent.md) | 60+ | Inference-time system design for long-horizon LLM agents — search/planning, decomposition, reflection, memory management, multi-agent methods, compute allocation, and cognitive architectures |
| [Mixed-Representation Reasoning](mixed-representation-reasoning.md) | 46 | Mixed-representation reasoning for LLMs — code, latent embeddings, graph structures, formal proofs, and dynamic per-step format selection as alternatives to natural language chain-of-thought |
| [Multi-GPU Agent Orchestration](multi-gpu-agent-orchestration.md) | 86 | Infrastructure for orchestrating 100+ LLM agents across multi-GPU nodes — disaggregated inference, agent-aware scheduling, KV-cache management, and workflow DAG optimization |
| [PDF-to-LLM Conversion Tools](pdf-to-llm-conversion-tools.md) | 12 tools | Converting academic textbooks (PDF) to LLM-readable formats — benchmarking MinerU, Docling, Marker, GOT-OCR, and others across math, biology, and CS domains |
| [Prompt Hacking & Attention Architecture](prompt-hacking-attention-architecture.md) | 662 searched, 7 deep-read | Architecture-aware prompt hacking — exploiting softmax normalization and causal attention masks for prompt optimization, attention sink manipulation, positional bias, and context engineering |

## Methodology

Each report follows the same pipeline:

1. **Phase 1 — Frontier**: Scan NeurIPS, ICLR, ICML, ACL 2024–2026 for latest breakthroughs
2. **Phase 2 — Survey**: Broad search across 2022–2026, curate peer-reviewed paper database
3. **Phase 3 — Deep Dive**: Read 8–15 key papers in full, extract structured notes
4. **Phase 4 — Code & Tools**: Map open-source implementations and GitHub repos
5. **Phase 5 — Synthesis**: Cross-paper taxonomy, comparative tables, gap analysis
6. **Phase 6 — Report**: Final compilation with citations and actionable recommendations

Peer-reviewed conference papers are prioritized over arXiv preprints throughout.

## License

These reports are provided for research reference purposes.
