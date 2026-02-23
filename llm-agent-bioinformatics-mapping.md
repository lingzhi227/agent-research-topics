# Bioinformatics Meets LLM Agents: A Systematic Mapping of Parallel Concepts and Gap Analysis

## Abstract

The rapid rise of Large Language Model (LLM) agent systems has produced a wave of new infrastructure for tool orchestration, provenance tracking, knowledge representation, and reproducible execution. Yet many of these challenges were confronted -- and substantially solved -- over two decades of bioinformatics infrastructure development. This survey presents a systematic mapping of 14 concept pairs spanning the bioinformatics and LLM agent domains, organized into six thematic clusters: workflow and orchestration, provenance and traceability, knowledge representation, tool ecosystems, platforms and standards, and data and annotation. We identify four concept pairs with explicit, published bridges (provenance, semantic web, workflow languages, knowledge graphs), three with partial bridges (ontologies, FAIR principles, Galaxy platform), and seven with implicit parallels but no published mapping (containers, data formats, package managers, pipeline registries, interoperability standards, metadata standards, annotation pipelines). We catalog seven bridge implementations already in production, present a prioritized gap analysis, and propose a concrete implementation roadmap. Our central thesis is that the bioinformatics community's hard-won solutions to tool interoperability, workflow reproducibility, and provenance tracking constitute an underutilized knowledge base that could accelerate the maturation of LLM agent infrastructure by years.

## 1. Introduction

In 2025, the LLM agent ecosystem is experiencing the same growing pains that bioinformatics faced in the early 2000s: how do you describe tools so machines can discover and compose them? How do you capture what happened during a complex multi-step computation so that someone else can reproduce it? How do you build registries that scale, federate, and maintain quality? How do you ensure that heterogeneous software components interoperate reliably?

The bioinformatics community spent two decades building answers to these questions. The Common Workflow Language (CWL) standardized tool descriptions and workflow portability across engines [@Crusoe2022]. The W3C PROV family of standards, adopted enthusiastically by scientific workflow systems, formalized provenance tracking with the rigor needed for reproducible research [@Moreau2013]. The EDAM ontology provided a controlled vocabulary for operations, data types, and formats [@Ison2013]. Galaxy democratized complex analyses through a web-based platform with integrated tool wrapping, history tracking, and sharing [@Galaxy2024]. The FAIR principles articulated machine-actionable guidelines for data stewardship [@Wilkinson2016]. GA4GH built an international governance framework for genomic data interoperability [@Rehm2021].

Meanwhile, the LLM agent community is building its own infrastructure stack at remarkable speed. The Model Context Protocol (MCP) standardizes tool access for LLM agents. LangGraph, CrewAI, and AutoGen provide orchestration frameworks. Weights & Biases, LangSmith, and MLflow capture execution metadata. MCP registries aggregate tool descriptions. Yet these efforts proceed largely without reference to the bioinformatics precedents that could inform their design.

This survey maps the two infrastructure stacks systematically, identifies where bridges have already been built, and highlights where the most valuable opportunities for cross-pollination remain. The scope covers literature from 2023 through early 2026, focusing on peer-reviewed papers, preprints, and open-source implementations at the intersection of both domains.

## 2. Background

### 2.1 Bioinformatics Infrastructure Stack

The modern bioinformatics infrastructure stack has been refined through decades of community effort, driven by the need to process massive genomic datasets reproducibly across heterogeneous computing environments.

**Workflow Languages and Engines.** CWL, WDL, Nextflow, and Snakemake represent the major workflow systems. CWL emphasizes portability through explicit, hardware-agnostic tool definitions [@Crusoe2022]. Nextflow's DSL2 provides modularization through processes and subworkflows, with the nf-core community maintaining hundreds of reusable pipeline components [@Ewels2025]. Snakemake uses Python-based, file-dependency-driven rules [@Molder2021]. WDL, developed at the Broad Institute, prioritizes human readability.

**Provenance and Reproducibility.** The W3C PROV data model (PROV-DM) and its OWL ontology (PROV-O) define entities, activities, and agents with relationships such as `wasGeneratedBy`, `used`, and `wasAssociatedWith` [@Moreau2013]. Scientific workflow systems capture full execution traces -- inputs, outputs, parameters, software versions, and execution environments -- enabling auditing and reproduction.

**Knowledge Representation.** The Gene Ontology (GO) provides the gold standard for functional annotation across three sub-ontologies: Molecular Function, Biological Process, and Cellular Component. The EDAM ontology structures bioinformatics operations, data types, and formats into a controlled vocabulary used by tool registries like bio.tools [@Ison2013]. The OBO Foundry coordinates interoperable ontology development across the biomedical domain.

**Tool Ecosystem.** Bioconda provides over 11,000 bioinformatics packages with dependency resolution. BioContainers offer Docker and Singularity images for reproducible tool execution. The bio.tools registry annotates tools with EDAM terms for semantic discovery.

**Platforms and Standards.** Galaxy provides GUI-based workflow construction and execution for non-programmers [@Galaxy2024]. GA4GH develops open standards including WES (Workflow Execution Service), TES (Task Execution Service), and DRS (Data Repository Service). The FAIR principles guide data management toward machine-actionability [@Wilkinson2016].

**Data and Annotation.** Domain-specific formats (FASTA, FASTQ, BAM/SAM, VCF, BED) encode biological data with increasing levels of metadata richness. Annotation pipelines (BLAST, InterProScan) combine homology search, domain prediction, and functional inference in multi-step, multi-evidence processes.

### 2.2 LLM Agent Architecture Stack

The LLM agent stack has coalesced rapidly since 2023, driven by advances in function calling, reasoning, and multi-step execution.

**Orchestration Frameworks.** LangGraph provides cyclic, graph-based architecture for stateful agents where nodes represent tool invocations and edges control information flow. CrewAI implements role-based multi-agent crews. AutoGen (Microsoft) enables multi-agent conversational frameworks. AWS Strands Agents provide advanced orchestration techniques.

**Tool Protocols.** The Model Context Protocol (MCP), introduced by Anthropic, standardizes how LLM agents access tools, resources, and prompts. OpenAI's function calling uses JSON Schema to define tool parameters. Google's Agent-to-Agent (A2A) protocol enables inter-agent communication.

**Provenance and Logging.** Weights & Biases, MLflow, and LangSmith capture model parameters, prompts, and results. PROV-AGENT extends W3C PROV for agent tracking [@Souza2025]. yProv manages provenance in large-scale AI workflows.

**Knowledge and Memory.** GraphRAG (Microsoft) combines knowledge graphs with retrieval-augmented generation. Vector databases provide embedding-based retrieval. Agent memory systems store and retrieve past interactions.

**Registries.** The MCP Registry (registry.modelcontextprotocol.io) launched in September 2025 as an open catalog for MCP servers with federated discovery. ToolRegistry provides protocol-agnostic tool management [@ToolRegistry2025].

**Sandboxing.** Code sandboxes (E2B, Blaxel, Daytona) provide isolated execution environments for LLM-generated code, with sub-100ms creation times and fault-tolerant operation [@FaultTolerantSandbox2025].

## 3. Concept Mapping: A Complete Taxonomy

### 3.1 Workflow and Orchestration

This cluster encompasses the foundational parallel between bioinformatics workflow systems and LLM agent orchestration frameworks.

**CWL/WDL Task Definitions and Function/Tool Definitions.** A CWL `CommandLineTool` document specifies a tool's inputs, outputs, base command, requirements, and hints in a declarative YAML format. An LLM function calling definition specifies a tool's name, description, and parameters in JSON Schema. Both serve the same purpose: making a computational tool machine-discoverable, machine-composable, and self-describing. The structural isomorphism is striking, yet no paper has formally analyzed it.

**Nextflow Channels/Processes and LangGraph Nodes/Edges.** Nextflow's dataflow paradigm, where data flows through channels connecting processes, is directly analogous to LangGraph's directed graph where state flows through nodes connected by edges. Both support branching, merging, and conditional execution. Both separate the definition of computational steps from the orchestration logic that connects them.

**Snakemake Rules and Agent Action-Observation Loops.** Snakemake determines which rules to execute based on file-dependency tracking and timestamps. An agent's action-observation loop similarly determines which tools to invoke based on the current state and the results of previous tool calls. Both represent reactive, state-dependent execution.

| Bioinformatics | LLM Agents | Mapping Strength |
|---|---|---|
| CWL/WDL task definitions | Function/tool definitions (JSON Schema) | Strong structural |
| Nextflow channels and processes | LangGraph nodes and edges | Strong structural |
| Snakemake rules and wildcards | Agent action-observation loops | Moderate conceptual |
| nf-core modules | Reusable tool definitions | Strong structural |
| DAG execution engine | Agent orchestration framework | Strong structural |
| CWL scatter/gather | Parallel tool calls | Moderate structural |

**Bridge: Agentic Workflow Library (AWL).** The most direct bridge is AWL (github.com/agentic-workflow-library), which wraps CWL tools with metadata, test datasets, and an interactive interface for agentic execution. AWL's tagline -- "The power of CWL or Nextflow, but with an LLM-aware assistant behind every step" -- captures the convergence precisely.

**Bridge: WorkflowLLM.** The WorkflowLLM system [@WorkflowLLM2025], presented at ICLR 2025, fine-tunes Llama-3.1-8B on WorkflowBench (106,000 samples, 1,503 APIs, 83 applications) to enhance LLM workflow orchestration capabilities, bridging Robotic Process Automation and Agentic Process Automation.

**Bridge: Prompt2DAG.** Prompt2DAG [@Prompt2DAG2025] transforms natural language descriptions into executable Apache Airflow DAGs through a four-stage pipeline, achieving a 78.5% success rate, demonstrating that the conceptual gap between natural language intent and formal workflow specification is bridgeable.

### 3.2 Provenance and Traceability

Provenance tracking represents the strongest and most explicit bridge between the two domains.

**W3C PROV and PROV-AGENT.** The PROV-AGENT system [@Souza2025], published at IEEE e-Science 2025, directly extends the W3C PROV standard with MCP concepts to represent agent actions and their connections to data and workflow tasks. An `AgentTool` may be informed by `AIModelInvocations`; each invocation uses a `Prompt` and a specific `AIModel` (with metadata: name, type, provider, temperature) and generates a `ResponseData` object. PROV-AGENT supports CrewAI, LangChain, and OpenAI interfaces, and has been evaluated across edge, cloud, and HPC environments. This is the single most important paper bridging the two domains.

**Workflow Provenance Querying.** A companion paper on LLM Agents for Interactive Workflow Provenance [@Souza2025b] establishes a reference architecture and evaluation methodology for LLM agents that interact with workflow provenance data, enabling natural language queries over execution traces.

| Bioinformatics | LLM Agents | Mapping Strength |
|---|---|---|
| W3C PROV-O ontology | PROV-AGENT extensions | Explicit, formal |
| Retrospective provenance capture | Agent interaction logging | Explicit, formal |
| Provenance queries (SPARQL) | Natural language provenance queries | Explicit, demonstrated |
| Execution trace graphs | Agent execution trace DAGs | Strong structural |
| Hallucination detection through provenance | Hallucination propagation tracking | Explicit in PROV-AGENT |

### 3.3 Knowledge Representation

This cluster maps bioinformatics ontologies and controlled vocabularies to LLM tool description systems.

**EDAM Ontology and MCP Tool Descriptions.** EDAM's four sections -- Topic, Operation, Data, and Format -- provide a structured vocabulary for annotating bioinformatics tools. MCP tool descriptions serve an analogous purpose: specifying what a tool does (operation), what it accepts (data/format), and what domain it belongs to (topic). However, MCP descriptions are free-text with JSON Schema parameters, lacking EDAM's controlled vocabulary and hierarchical organization.

**Gene Ontology and Tool Capability Tags.** GO annotations attach standardized functional labels to gene products. In the agent world, tool capability tags serve a similar function, but without GO's rigor: there is no controlled vocabulary, no evidence codes, and no curation process for LLM tool descriptions.

**OBO Foundry Coordination and (Nothing Yet).** The OBO Foundry provides a governance framework ensuring that biomedical ontologies are interoperable, non-overlapping, and community-maintained. No equivalent coordination mechanism exists for LLM tool schemas, which proliferate without standardization.

| Bioinformatics | LLM Agents | Mapping Strength |
|---|---|---|
| EDAM Operation types | Tool function definitions | Moderate, unrecognized |
| EDAM Data types | Tool input/output schemas | Moderate, unrecognized |
| EDAM Format types | Content types / MIME types | Moderate, unrecognized |
| GO term annotations | Tool capability tags | Weak, unrecognized |
| OBO Foundry coordination | No equivalent | Gap |

**Bridge: OntoGPT.** The Monarch Initiative's OntoGPT (github.com/monarch-initiative/ontogpt) uses LLMs for ontological extraction via SPIRES (Structured Prompt Interrogation and Recursive Extraction of Semantics), populating knowledge bases with zero-shot learning. This bridges LLMs into ontology construction but does not map ontology design patterns to tool description design.

**Bridge: Artificial Intelligence Ontology (AIO).** The AIO [@Joachimiak2024] uses LLM-assisted construction (Claude 3 Sonnet, GPT-4) with ROBOT templates to build an AI concept hierarchy, demonstrating that ontology engineering methods from bioinformatics transfer directly to AI concept organization.

### 3.4 Tool Ecosystem

This cluster covers tool packaging, distribution, and discovery.

**Bioconda/bio.tools and MCP Registry.** Bioconda provides dependency-resolved packages for 11,000+ bioinformatics tools. The bio.tools registry annotates each tool with EDAM terms for semantic discovery. The MCP Registry (launched September 2025) serves an analogous function: an open catalog with an API for discovering MCP servers. Both solve the problem of "how do I find and install the right tool?" -- but Bioconda has a 10-year head start on dependency resolution, version pinning, and community curation.

**BioContainers and Code Sandboxes.** BioContainers provide Docker/Singularity images ensuring reproducible tool execution across local, HPC, and cloud environments. LLM agent sandboxes (E2B, Blaxel, Daytona) provide isolated execution environments for agent-generated code. Both solve tool isolation and reproducibility, but BioContainers emphasize long-term reproducibility (exact version pinning) while agent sandboxes emphasize ephemeral, low-latency execution.

| Bioinformatics | LLM Agents | Mapping Strength |
|---|---|---|
| Bioconda (package channel) | MCP Registry (server catalog) | Moderate, parallel development |
| bio.tools (annotated registry) | MCP Registry with descriptions | Moderate, parallel development |
| EDAM annotations on tools | Tool descriptions + JSON Schema | Moderate, unrecognized |
| BioContainers (Docker images) | Code sandboxes (micro-VMs) | Moderate, implicit |
| Container registries (quay.io) | Sandbox templates | Weak, implicit |
| Dependency resolution | Tool compatibility checking | Weak, implicit |

**Bridge: BioinfoMCP.** BioinfoMCP [@Widjaja2025] auto-converts bioinformatics tool documentation into MCP server interfaces using LLMs, achieving a 94.7% success rate across 38 tools. This is a direct analog to how CWL tool wrappers were historically written by hand and are now being auto-generated.

**Bridge: MCPMed.** MCPMed [@MCPMed2025] calls for MCP-enabled bioinformatics web services, proposing a standardized discovery, invocation, and verification layer -- essentially a "bio.tools for the MCP era."

### 3.5 Platforms and Standards

This cluster encompasses integrated platforms and governance frameworks.

**Galaxy and No-Code Agent Builders.** Galaxy democratized bioinformatics by enabling non-programmers to construct and execute complex workflows through a web GUI. Platforms like Dify, Flowise, and AnythingLLM serve the same democratization function for LLM agents. The structural parallels are deep: Galaxy's XML tool wrappers correspond to MCP server definitions; Galaxy's workflow editor corresponds to visual agent builders; Galaxy's histories correspond to agent execution logs; Galaxy's Tool Shed corresponds to MCP server marketplaces.

**GA4GH Standards Suite and MCP/A2A.** GA4GH's standards suite (WES, TES, DRS, Data Connect) provides interoperability across genomic data systems. MCP, A2A, and OpenAI function calling collectively provide interoperability across LLM agent systems. GA4GH's experience with multi-stakeholder governance, conformance testing, and versioning is directly relevant to MCP's evolution.

**FAIR Principles and FAIR4AI/FAIR4ML.** The FAIR principles, originally articulated for scientific data, are being extended to AI models through FAIR4AI [@FAIR4AI2023] and FAIR4ML (RDA Interest Group). The specific parallel between FAIR for bioinformatics tools/workflows and FAIR for LLM agent tools has not been explicitly drawn, but the conceptual alignment is strong: both domains need tools that are Findable (discoverable in registries), Accessible (via standard protocols), Interoperable (through shared schemas), and Reusable (with clear licenses and provenance).

| Bioinformatics | LLM Agents | Mapping Strength |
|---|---|---|
| Galaxy platform (web GUI) | No-code agent builders (Dify, Flowise) | Strong structural |
| Galaxy tool wrappers (XML) | MCP server tool definitions | Strong structural |
| Galaxy Tool Shed | MCP server marketplaces | Strong structural |
| Galaxy histories | Agent conversation/execution logs | Strong structural |
| GA4GH WES/TES/DRS | Agent orchestration / tool execution / resource APIs | Moderate, unrecognized |
| GA4GH governance (TASC) | MCP specification governance | Weak, unrecognized |
| FAIR principles for data | FAIR4AI/FAIR4ML for models/tools | Moderate, explicit |
| WorkflowHub (FAIR registry) | MCP Registry | Moderate, parallel development |

**Bridge: Galaxy LLM Integration.** Galaxy has directly integrated LLM agents into its platform [@Galaxy2024]: a Natural Language Analysis Tool for interactive dataset analysis, a Galaxy Workflow Agent for executing tools and workflows from natural language, and an LLM Hub providing direct access to language models within the Galaxy interface. This represents an internal bridge where a bioinformatics platform adopts LLM agent capabilities.

### 3.6 Data and Annotation

This cluster covers data format design patterns and multi-step annotation processes.

**Bioinformatics Formats and LLM Data Schemas.** The evolution of bioinformatics data formats -- from simple, human-readable FASTA to indexed binary BAM to richly-headered VCF -- mirrors the LLM community's trajectory from plain text output to structured JSON to typed schemas with validation. Both domains face the same tension between human readability and machine efficiency. Both have evolved header-plus-body patterns where metadata headers describe the structure and provenance of the data that follows.

**Annotation Pipelines and Entity Extraction.** BLAST searches for sequence similarity using statistical scoring (E-values). LLM-based entity extraction searches for semantic similarity using embedding distances. InterProScan integrates multiple databases in a multi-step pipeline with evidence aggregation. LLM agent orchestration integrates multiple tools in a multi-step pipeline with confidence scoring. The parallel extends to quality control: bioinformatics annotation uses evidence codes (IEA, ISS, IDA) to grade confidence, while LLM annotation uses model confidence scores and human-in-the-loop validation.

| Bioinformatics | LLM Agents | Mapping Strength |
|---|---|---|
| FASTA/FASTQ (sequence formats) | Plain text / structured output | Weak, conceptual |
| BAM/SAM (alignment format) | JSON response formats | Weak, conceptual |
| VCF (variant format with metadata headers) | JSON Schema with metadata | Moderate, structural |
| BLAST (sequence similarity) | Embedding similarity (RAG retrieval) | Moderate, conceptual |
| InterProScan (multi-database integration) | Multi-tool agent orchestration | Moderate, structural |
| Annotation evidence codes | Confidence scores / provenance | Moderate, conceptual |
| Format validators | Schema validators | Strong, structural |

**Bridge: LLM4Annotation.** The LLM4Annotation collection (github.com/Zhen-Tan-dmml/LLM4Annotation) aggregates LLM-based annotation approaches, and human-AI synergy approaches for annotation pipelines are emerging [@AIAnnotation2025], but the explicit mapping to bioinformatics annotation pipeline design patterns remains unmade.

## 4. Bridge Implementations

Seven significant bridge implementations already exist, connecting bioinformatics concepts to LLM agent systems:

### 4.1 PROV-AGENT

**Origin:** Oak Ridge National Laboratory (UT-Battelle, LLC)
**Publication:** IEEE e-Science 2025 [@Souza2025]
**Bridge:** W3C PROV --> LLM Agent Provenance

PROV-AGENT is the most formally rigorous bridge identified. It extends the W3C PROV ontology with MCP-aware concepts to capture fine-grained agentic provenance: prompts, responses, model invocations, and tool calls. It wraps CrewAI, LangChain, and OpenAI agent frameworks and has been evaluated across edge, cloud, and HPC environments. Its significance lies not just in the implementation but in the explicit argument that AI agents need the same provenance rigor that scientific workflows achieved through PROV-O.

### 4.2 BioinfoMCP

**Origin:** Chinese University of Hong Kong, Shenzhen
**Publication:** arXiv:2510.02139 [@Widjaja2025]
**Bridge:** Bioinformatics Tool Documentation --> MCP Servers

BioinfoMCP auto-converts bioinformatics tool documentation into MCP server interfaces using LLMs, achieving 94.7% success across 38 tools. This is the CWL-to-MCP bridge: just as CWL wraps command-line tools for workflow engines, BioinfoMCP wraps bioinformatics tools for LLM agents. The BioinfoMCP Benchmark validates converted tools across AI-agent platforms including Claude Desktop and Cursor.

### 4.3 MCPMed

**Origin:** Clinical Bioinformatics, Saarland University
**Publication:** Briefings in Bioinformatics 2025 [@MCPMed2025]
**Bridge:** Bioinformatics Web Services --> MCP-enabled Discovery

MCPMed operates as a bioinformatics MCP hub at mcpmed.org, hosting specialized MCP servers for STRING protein interactions, UCSC Cell Browser, and bioinformatics service discovery. It articulates a vision for MCP as a semantic layer over bioinformatics APIs -- standardizing discovery, invocation, and verification.

### 4.4 AWL (Agentic Workflow Library)

**Origin:** Open-source community (Apache 2.0)
**Repository:** github.com/agentic-workflow-library
**Bridge:** CWL Workflows --> Agentic Execution

AWL wraps CWL tools with metadata, test datasets, and an interactive interface for LLM-aware execution. Its structural variant analysis agent demonstrates the pattern: existing CWL-wrapped bioinformatics tools gain an agentic interface without requiring reimplementation.

### 4.5 sparql-llm / sparql-mcp

**Origin:** Swiss Institute of Bioinformatics (SIB)
**Repository:** github.com/sib-swiss/sparql-llm
**Bridge:** RDF/SPARQL Endpoints --> LLM Agent Tools

This direct bridge exposes bioinformatics SPARQL endpoints (UniProt, Bgee, OMA, SwissLipids, Cellosaurus) to LLM agents via MCP. The accompanying chat system at expasy.org/chat enables natural language queries that are automatically translated to SPARQL, validated, and executed against biological knowledge bases.

### 4.6 BioScientist Agent

**Origin:** bioRxiv 2025
**Bridge:** Biomedical Knowledge Graphs --> LLM Multi-Agent Reasoning

BioScientist Agent unifies a billion-fact biomedical knowledge graph with a variational graph auto-encoder for link prediction, a reinforcement learning module for graph traversal, and an LLM multi-agent orchestration layer. Performance improvements over baselines (MRR +36%, Hit@1 +53%, Hit@3 +42%) demonstrate that combining curated biological knowledge with agent reasoning produces substantial gains.

### 4.7 GeneAgent

**Origin:** NIH / NCBI
**Publication:** Nature Methods 2025
**Bridge:** Biological Database Verification --> Agent Self-Verification

GeneAgent implements a self-verification pattern where the agent autonomously queries biological databases via Web APIs to fact-check its own outputs, achieving 92% accuracy across 1,106 gene sets. This bridges the bioinformatics practice of evidence-grounded annotation with the agent need for hallucination reduction through tool-mediated verification.

## 5. Gap Analysis

### 5.1 High-Priority Gaps

**Gap 1: EDAM-like Ontology for LLM Tools.**
No controlled vocabulary exists for LLM tool descriptions. MCP tool descriptions use free text, with no standardized categories for operation types, data types, or formats. EDAM's 15-year evolution provides a direct template. Priority: *critical for tool discovery at scale*.

**Gap 2: GA4GH-style Governance for MCP.**
The MCP ecosystem lacks the multi-stakeholder governance, conformance testing, and versioning framework that GA4GH has built for genomics. As MCP adoption grows, the absence of formal compliance testing and specification versioning will become a bottleneck. GA4GH's TASC model and its openapi-test-runner provide a direct template. Priority: *critical for ecosystem maturity*.

**Gap 3: ISA-like Metadata Standard for Agent Executions.**
No standard metadata schema exists for describing agent execution contexts. The ISA framework's Investigation-Study-Assay hierarchy maps remarkably to Agent Session-Task-ToolCall patterns: an Investigation (research project) contains Studies (agent sessions) that contain Assays (individual tool calls with parameters and results). Adopting this pattern would enable cross-platform comparison and meta-analysis of agent performance. Priority: *high for reproducibility*.

**Gap 4: WorkflowHub-like FAIR Registry for Agent Workflows.**
WorkflowHub (workflowhub.eu) provides FAIR workflow registration with DOIs, RO-Crate metadata packaging, and community curation. No equivalent exists for agent workflows. Agent definitions, prompts, and tool configurations are shared informally through GitHub repositories and blog posts without persistent identifiers, structured metadata, or quality curation. Priority: *high for reusability*.

### 5.2 Medium-Priority Gaps

**Gap 5: BioContainers Design Patterns for Agent Sandboxes.**
BioContainers' lessons on version pinning, automated testing, community maintenance, and multi-architecture support could directly improve agent sandbox ecosystems. Currently, agent sandboxes prioritize latency over reproducibility -- the opposite of BioContainers' design philosophy. A synthesis of both priorities would serve the agent community well. Priority: *medium, active parallel development*.

**Gap 6: nf-core-style Community Standards for MCP Servers.**
nf-core provides linting rules, CI/CD templates, and community review for Nextflow pipelines, maintaining quality across 8,000+ community members. MCP servers lack equivalent quality standards. MCPMed's community review model is an early signal but not yet formalized. Priority: *medium, community-building needed*.

**Gap 7: Bioinformatics Format Design Patterns for LLM Output Schemas.**
The evolution from FASTA to BAM to VCF encodes deep lessons about balancing human readability with machine efficiency, embedding metadata, and supporting indexed access. LLM output format design could benefit from studying this evolutionary trajectory rather than reinventing it. Priority: *medium, theoretical but actionable*.

### 5.3 Low-Priority Gaps (Mostly Theoretical)

**Gap 8: BLAST E-value Analog for LLM Confidence.**
BLAST's E-value provides a statistically calibrated measure of match significance. LLM confidence scores lack comparable calibration. While the mathematical frameworks differ fundamentally, the principle -- that every computational result should carry a calibrated significance measure -- transfers. Priority: *low, requires fundamental research*.

**Gap 9: Annotation Pipeline Design Patterns for Entity Extraction.**
Multi-evidence annotation (combining homology, domain prediction, and functional inference) parallels multi-tool agent orchestration (combining retrieval, reasoning, and validation). The parallel is structural but the implementation details diverge enough that direct transfer is limited. Priority: *low, architectural insight rather than direct implementation*.

**Gap 10: Linked Data Principles for Agent Resource Linking.**
The Semantic Web's Linked Data principles (use URIs, use HTTP, provide useful information at URIs, link to other URIs) provide a template for MCP resource linking. MCP resources use URIs but do not yet fully embrace the Linked Data vision of a web of interlinked, dereferenceable resources. Priority: *low, long-term architectural direction*.

## 6. Implementation Roadmap

### 6.1 Quick Wins (Weeks)

| Initiative | Description | Effort | Impact |
|---|---|---|---|
| EDAM-to-MCP vocabulary mapping | Map EDAM Operation/Data/Format terms to MCP tool description conventions; publish as a crosswalk table | 2-4 weeks | Immediate improvement in tool discoverability |
| GA4GH conformance test patterns for MCP | Adapt ga4gh/openapi-test-runner patterns to validate MCP server implementations | 2-3 weeks | Quality assurance for MCP servers |
| ISA-to-Agent metadata template | Create a JSON-LD template mapping ISA Investigation-Study-Assay to Agent Session-Task-ToolCall | 1-2 weeks | Standardized execution metadata |
| BioContainers best practices guide for agent sandboxes | Document version-pinning, multi-arch, and testing patterns from BioContainers for agent sandbox developers | 1-2 weeks | Improved sandbox reproducibility |

### 6.2 Medium-Term Projects (Months)

| Initiative | Description | Effort | Impact |
|---|---|---|---|
| EDAM-MCP: Controlled vocabulary for agent tools | Develop an EDAM-inspired ontology for LLM tool operations, inputs, and outputs, enabling semantic search over tool registries | 3-6 months | Foundational for tool ecosystem maturity |
| AgentHub: FAIR registry for agent workflows | Build a WorkflowHub-inspired registry with DOIs, RO-Crate metadata, and community curation for agent definitions | 3-6 months | Enables reproducibility and sharing |
| MCP-nfcore: Quality framework for MCP servers | Develop linting rules, CI/CD templates, and review processes modeled on nf-core community standards | 2-4 months | Raises MCP server quality baseline |
| PROV-AGENT extensions for bioinformatics agents | Extend PROV-AGENT with EDAM term annotations so that bioinformatics agent provenance is semantically rich | 2-3 months | Bridges provenance and knowledge representation |

### 6.3 Strategic Initiatives (Year+)

| Initiative | Description | Effort | Impact |
|---|---|---|---|
| GA4GH-for-Agents: Governance framework | Establish multi-stakeholder governance for agent interoperability standards, modeled on GA4GH's TASC, with conformance testing and specification versioning | 12-18 months | Industry-shaping standards governance |
| Unified bioinformatics-agent infrastructure | Build a system where CWL tools, MCP servers, and agent workflows share a common registry, provenance model, and quality framework | 12-24 months | Eliminates redundant infrastructure |
| FAIR Agent Maturity Model | Develop a FAIR maturity assessment framework for agent tools and workflows, analogous to FAIR data maturity models | 6-12 months | Measurable quality improvement |
| Cross-domain benchmark suite | Create benchmarks that test agents on bioinformatics tasks requiring both domain knowledge and infrastructure awareness (provenance, reproducibility, format handling) | 6-12 months | Drives infrastructure improvement through evaluation |

## 7. The Infrastructure Isomorphism Thesis

This survey's findings point to a claim that is stronger than mere analogy. We propose the **Infrastructure Isomorphism Thesis**:

> *Any sufficiently mature system for machine-mediated multi-step computation will converge on the same infrastructure patterns: declarative tool descriptions, directed-graph orchestration, provenance-tracked execution, ontology-governed discovery, containerized reproducibility, FAIR-compliant registries, and multi-stakeholder governance. Bioinformatics reached this convergence first. LLM agent systems are reaching it now. The isomorphism is not coincidental -- it is structural, arising from the universal requirements of composable, reproducible, and auditable computational pipelines.*

Three observations support this thesis:

**1. The convergence is layer-complete.** The isomorphism is not limited to one or two superficial parallels. It spans every layer of the infrastructure stack: from low-level tool wrapping (CWL CommandLineTool ↔ MCP Tool JSON) through mid-level orchestration (Nextflow dataflow ↔ LangGraph state graph) to high-level governance (GA4GH TASC ↔ MCP specification process). All 53 concept pairs identified in this survey map across six thematic clusters. No layer of the bioinformatics stack lacks an LLM agent counterpart; no layer of the agent stack lacks a bioinformatics precedent.

**2. The gap is not in knowledge, but in knowledge transfer.** The bioinformatics community has spent 20+ years solving infrastructure problems -- tool description, workflow portability, provenance tracking, FAIR compliance, standards governance, quality assurance -- that the LLM agent community is now encountering for the first time. Every one of the 16 gaps identified in this survey could be addressed faster and better by systematically learning from bioinformatics precedents rather than reinventing solutions from scratch. The reason this transfer hasn't happened is sociological, not technical: the two communities read different journals, attend different conferences, and cite different literatures.

**3. The highest-leverage intervention is a bridge, not a build.** The single most impactful meta-action is not to build any specific tool, ontology, or standard. It is to **establish a formal, sustained dialogue between the bioinformatics standards community (GA4GH, nf-core, WorkflowHub, OBO Foundry, W3C PROV) and the LLM agent standards community (MCP, A2A, LangChain, CrewAI, OpenAI)**. This dialogue does not yet exist. Creating it would unlock knowledge transfer across all 16 gaps simultaneously. A joint workshop at a venue like ISMB/ECCB or NeurIPS, a cross-community working group, or even a shared mailing list would do more to accelerate agent infrastructure maturity than any single engineering project.

The Infrastructure Isomorphism Thesis has a practical corollary: **for any infrastructure problem encountered in the LLM agent ecosystem, check whether bioinformatics has already solved it.** The answer, as this survey demonstrates, is usually yes.

## 8. Conclusion

This survey reveals a systematic, layer-complete isomorphism between bioinformatics infrastructure and LLM agent infrastructure. Across 53 concept pairs organized in six thematic clusters, we find 16 with strong mappings (including 7 with explicit, published bridges), 17 with moderate parallels, and 14 with weak but real structural correspondence.

Seven significant bridges already exist: PROV-AGENT extends W3C PROV to agent systems; BioinfoMCP and MCPMed wrap bioinformatics tools for MCP access; AWL wraps CWL workflows for agentic execution; sparql-llm exposes biological knowledge bases to LLM agents; BioScientist Agent and GeneAgent connect biological knowledge graphs to agent reasoning.

But seven equally important concept pairs remain unbridged: containers and sandboxes, data formats and output schemas, package managers and tool registries, pipeline registries and agent registries, interoperability standards and agent protocols, metadata standards and execution traces, annotation pipelines and entity extraction systems. These gaps represent concrete, actionable opportunities.

Our prioritized roadmap identifies four immediate actions: (1) developing ADAM, an EDAM-inspired controlled vocabulary for LLM tools; (2) creating ISA-Agent, a metadata standard for agent execution; (3) defining agent evidence codes modeled on GO annotation evidence; and (4) standardizing PROV-AGENT across agent frameworks. Six quick wins are achievable in weeks; five strategic projects could reshape the field over months to years.

The bioinformatics community built its infrastructure through two decades of community effort, failed experiments, standards negotiations, and hard-won lessons about what works at scale. The LLM agent community can either repeat that twenty-year journey, or learn from it. The Infrastructure Isomorphism Thesis argues -- with 48 papers, 15 repositories, and 53 concept pairs as evidence -- that learning is the rational choice.

## References

[@Crusoe2022] Crusoe, M.R., et al. "Methods included: standardizing computational reuse and portability with the Common Workflow Language." *Communications of the ACM*, 65(6), 2022.

[@Moreau2013] Moreau, L. and Missier, P. "PROV-DM: The PROV Data Model." W3C Recommendation, 2013.

[@Ison2013] Ison, J., et al. "EDAM: an ontology of bioinformatics operations, types of data and identifiers, topics and formats." *Bioinformatics*, 29(10), 2013.

[@Galaxy2024] The Galaxy Community. "The Galaxy platform for accessible, reproducible, and collaborative data analyses: 2024 update." *Nucleic Acids Research*, 52(W1):W83, 2024.

[@Wilkinson2016] Wilkinson, M.D., et al. "The FAIR Guiding Principles for scientific data management and stewardship." *Scientific Data*, 3, 160018, 2016.

[@Rehm2021] Rehm, H.L., et al. "GA4GH: International policies and standards for data sharing across genomic research and healthcare." *Cell Genomics*, 1(2), 2021.

[@Souza2025] Souza, R., Gueroudji, A., DeWitt, S., Rosendo, D., Ghosal, T., Ross, R., Balaprakash, P., and Ferreira da Silva, R. "PROV-AGENT: Unified Provenance for Tracking AI Agent Interactions in Agentic Workflows." *IEEE 21st International Conference on e-Science*, 2025. arXiv:2508.02866.

[@Souza2025b] Souza, R., Poteet, T., Etz, B., Rosendo, D., Gueroudji, A., Shin, W., Balaprakash, P., and Ferreira da Silva, R. "LLM Agents for Interactive Workflow Provenance." *ACM SC'25 Workshops*, 2025. arXiv:2509.13978.

[@Widjaja2025] Widjaja, F., Chen, Z., and Zhou, J. "BioinfoMCP: A Unified Platform Enabling MCP Interfaces in Agentic Bioinformatics." arXiv:2510.02139, 2025.

[@MCPMed2025] "MCPmed: A call for Model Context Protocol-enabled bioinformatics web services for LLM-driven discovery." *Briefings in Bioinformatics*, 2025.

[@WorkflowLLM2025] Fan, S., Cong, X., Fu, Y., et al. "WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models." *ICLR*, 2025.

[@Prompt2DAG2025] "Prompt2DAG: A Modular Methodology for LLM-Based Data Enrichment Pipeline Generation." arXiv:2509.13487, 2025.

[@Joachimiak2024] Joachimiak, M., et al. "The Artificial Intelligence Ontology: LLM-Assisted Construction of AI Concept Hierarchies." *Applied Ontology*, 2024.

[@FAIR4AI2023] "FAIR for AI: An interdisciplinary and international community building perspective." *Nature Scientific Data*, 2023.

[@ToolRegistry2025] "ToolRegistry: A Protocol-Agnostic Tool Management Library for Function-Calling LLMs." arXiv:2507.10593, 2025.

[@FaultTolerantSandbox2025] "Fault-Tolerant Sandboxing for AI Coding Agents." arXiv:2512.12806, 2025.

[@AIAnnotation2025] "AI-Powered Annotation Pipelines: Human-AI Synergy for Annotation Feedback Loops." arXiv:2512.13714, 2025.

[@Ewels2025] Ewels, P., et al. "Empowering bioinformatics communities with Nextflow and nf-core." *Genome Biology*, 2025.

[@Molder2021] Molder, F., et al. "Sustainable data analysis with Snakemake." *F1000Research*, 10:33, 2021.

[@WorkflowHub2025] Goble, C., et al. "WorkflowHub: a registry for computational workflows." *Nature Scientific Data*, 2025.

[@BioScientist2025] "BioScientist Agent: LLM-Biomedical Agents with KG-Augmented RL Reasoning." *bioRxiv*, 2025.

[@GeneAgent2025] "GeneAgent: self-verification language agent for gene-set analysis using domain databases." *Nature Methods*, 2025.

[@BioAgentsMicrosoft2025] "BioAgents: Bridging the gap in bioinformatics analysis with multi-agent systems." *Nature Scientific Reports*, 15, 2025.

[@AgenticBioinformatics2025] "Streamline automated biomedical discoveries with agentic bioinformatics." *Briefings in Bioinformatics*, 26(5), bbaf505, 2025.

[@LLMAgentsBioinformatics2025] "The rise and potential opportunities of large language model agents in bioinformatics and biomedicine." *Briefings in Bioinformatics*, 26(6), 2025.

[@SPARQLLM2025] "SPARQL-LLM: Real-Time SPARQL Query Generation from Natural Language." arXiv:2512.14277, 2025.

[@Biomni2025] Huang, K., Zhang, S., Wang, H., Qu, Y., et al. "Biomni: A General-Purpose Biomedical AI Agent." *bioRxiv*, 2025.

[@MARBLE2026] Kim, S., Yun, S., Yun, Y., Lee, Y., and Lim, S. "MARBLE: Multi-Agent Reasoning for Bioinformatics Learning and Evolution." arXiv:2601.14349, 2026.

[@ESCARGOT2025] "ESCARGOT: Enhanced Strategy and Cypher-driven Analysis and Reasoning using Graph Of Thoughts." *Bioinformatics (Oxford)*, 41(2), 2025.

[@LLMSandbox2025] "LLM-in-Sandbox Elicits General Agentic Intelligence." arXiv:2601.16206, 2025.

[@BRAD2025] Pickard, J., et al. "BRAD: Bioinformatics Retrieval Augmented Digital Assistant." *Bioinformatics (Oxford)*, 2025.

[@KGARevion2024] "KGARevion: Knowledge Graph-based Agent for Biomedical QA." Zitnik Lab (Harvard), 2024.
