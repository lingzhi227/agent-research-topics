# LLM Agent配备生物信息学工具的工作流兼容性问题：系统文献综述

**作者**: 系统性文献综述（自动生成）
**日期**: 2026年2月23日
**研究主题**: LLM Agent使用生物信息学工具时，工作流中不同节点的工具Input/Output/Arguments差异导致的不兼容问题，以及通过预先设计工具集解决该问题的方法

---

## 摘要

本文对2022-2026年间LLM Agent工具兼容性问题的相关文献进行了系统综述，特别聚焦于生物信息学工作流中如何通过**预先设计工具模式（pre-defined tool schemas）**来解决工具I/O不兼容问题。我们共整理了65篇核心论文，对12篇关键论文进行深度阅读，并识别出六类解决方案范式：(1) 协议级标准化（MCP）、(2) 知识图谱驱动路由、(3) RAG驱动工具文档检索、(4) 类型化消息传递、(5) 语义类型系统、(6) 层级工具封装。研究发现，工具I/O兼容性存在三个层次：语法兼容性、语义兼容性和语用兼容性，而现有系统大多仅解决了语法层面的问题。我们提出了一个结合MCP协议（语法层）、知识图谱（关系层）和生物医学语义类型系统（语义层）的三层理想架构。

---

## 1. 研究背景与问题定义

### 1.1 问题的根本原因

生物信息学工具生态系统具有高度碎片化的特点：每个工具作为独立应用程序存在，具有独特的输入输出格式、命令行接口和计算需求。这种碎片化在LLM Agent驱动的工作流中造成了系统性障碍。

以全基因组测序（WGS）分析流水线为例，典型工具链为：
```
fastp（质控）→ Bowtie2（比对）→ samtools sort（排序）→ samtools index（索引）
→ GATK HaplotypeCaller（变异识别）→ bcftools view（过滤）→ bcftools stats（统计）
```

每个节点的输入输出格式不同：
- fastp输入FASTQ，输出FASTQ + JSON质控报告
- Bowtie2输入FASTQ，输出SAM/BAM
- GATK HaplotypeCaller输入BAM，输出VCF
- bcftools输入VCF，输出VCF/BCF或统计文本

**关键挑战**：不同工具的参数名称不统一（`-i` vs `--input`），输出格式不兼容（有些工具默认输出CRAM而非BAM），且工具更新频繁导致接口变化。

### 1.2 现有研究的核心假设差异

| 假设 | 对应方案 | 代表系统 |
|------|----------|----------|
| LLM可以实时生成代码修复不兼容 | Coding Agent方式 | 通用LLM + Python |
| 工具文档足以让LLM推断正确用法 | RAG方式 | BioMaster, BIA |
| **工具集应提前设计好供LLM使用** | **预定义模式方式** | **BioinfoMCP, SciToolAgent, Playbook WB** |

本综述聚焦于第三种假设，即**不依赖LLM在运行时修复工具接口，而是预先将工具封装为标准化可调用接口**。

---

## 2. 解决方案分类综述

### 2.1 第一类：协议级标准化（MCP生态）

#### 2.1.1 Model Context Protocol (MCP)

**背景**: Anthropic于2024年11月发布MCP协议，迅速成为工具-AI通信的事实标准。MCP定义了：
- 工具发现（tool discovery）
- 工具调用（tool invocation）
- 结构化输入输出（typed I/O schema）

**MCP工具清单格式**：
```json
{
  "name": "samtools_sort",
  "description": "Sort BAM/SAM files by genomic coordinate",
  "inputSchema": {
    "type": "object",
    "properties": {
      "input_file": {
        "type": "string",
        "description": "Path to input BAM/SAM file"
      },
      "output_file": {
        "type": "string",
        "description": "Path to output sorted BAM file"
      },
      "threads": {
        "type": "integer",
        "default": 4,
        "description": "Number of threads"
      }
    },
    "required": ["input_file", "output_file"]
  }
}
```

#### 2.1.2 BioinfoMCP（arXiv 2510.02139，2025年10月）

**机构**: 香港中文大学（深圳）周珏晓研究组

**问题**：生信工具的接口设计面向人类用户而非AI程序，导致"不兼容接口、异构输入输出格式、不一致的参数命名规范"。

**解决方案**：BioinfoMCP分两个组件：

**① BioinfoMCP Converter（核心创新）**:
```
工具文档（man page / --help输出）
    ↓（LLM阅读并提取参数信息）
MCP服务器代码（含完整类型化参数schema）
    ↓
可运行的MCP服务器（包装原始CLI工具）
```

转换流程：平均每个工具40秒，复杂工具（如bcftools, cellranger）不超过2分钟。

**② BioinfoMCP Benchmark**:
- 38个生信工具已转换
- 在3个AI平台（Claude、GPT等）上验证
- **94.7%的工具成功执行复杂工作流**

**验证案例**——WGS pipeline：
```
fastp → Bowtie2 align → Samtools sort → Samtools index
→ GATK HaplotypeCaller → Freebayes → bcftools view
→ samtools flagstat → bcftools stats → 输出报告
```
所有步骤通过MCP自动串联，AI Agent无需理解底层CLI语法。

**局限性**：
- 当前需要人工获取工具帮助文档
- 内存密集型工具（如STAR）因GPU限制无法完全自动化
- 工具版本更新时需要重新生成schema

**GitHub**: https://github.com/florensiawidjaja/BioinfoMCP

#### 2.1.3 MCPmed（arXiv 2507.08055，2025年7月）

**机构**: 萨尔大学临床生物信息学系

**问题**：生信数据库（GEO、STRING、UCSC）面向人类用户设计，LLM无法机器可读地访问。

**方案**：将MCP作为标准语义合约层覆盖在现有API之上，将API端点与科学概念+版本化元数据关联。

**FAIR vs GA4GH TRS vs MCP 比较**：

| | FAIR | GA4GH TRS | MCP |
|--|------|-----------|-----|
| 机器可操作性 | 建议但不强制 | 结构化API schema | **最高优先级** |
| 范围 | 通用数据管理 | 基因组学工具 | **AI/LLM集成** |
| 元数据 | 概念框架 | JSON schema | **JSON协议规范** |

**轻量级迁移路径**——HTML breadcrumbs:
```html
<meta name="mcp:tool" content="mirTargetLink">
<meta name="mcp:input_type" content="miRNA_ID">
<meta name="mcp:output_type" content="gene_interaction_network">
```
现有服务无需完全重写即可增量标准化。

#### 2.1.4 bio-mcp GitHub组织

由社区维护的独立MCP服务器集合，每个服务器包装单个生信工具：
- `bio-mcp-blast` — NCBI BLAST序列相似性搜索
- `bio-mcp-samtools` — SAM/BAM文件操作
- `bio-mcp-bcftools` — 变异调用
- `bio-mcp-fastqc` — 质量控制
- `bio-mcp-interpro` — 蛋白质域分析
- **GitHub**: https://github.com/bio-mcp

---

### 2.2 第二类：知识图谱驱动路由

#### 2.2.1 SciToolAgent（Nature Computational Science 2025）

**机构**: 浙江大学HICAI研究组

**问题**：现有Agent框架依赖简单上下文学习，"无法考虑大量工具之间的内在依赖关系"，导致多步工作流中次优解和效率降低。

**核心创新：科学工具知识图谱（SciToolKG）**

```
节点（每个工具）：
  - 工具ID、名称、领域（生物/化学/材料）
  - 输入格式（类型+示例）
  - 输出格式（类型+示例）
  - 功能描述
  - 安全级别
  - 先决工具（prerequisite）
  - 兼容后续工具（compatible successors）

边（工具关系）：
  - 顺序依赖（工具A输出→工具B输入）
  - 可选替代（工具A OR 工具B）
  - 并行关系（工具A和B可同时执行）
```

**Agent架构**：
```
用户查询
  ↓
Planner（基于SciToolKG-RAG生成"工具链"）
  ↓
chain-of-tools: [tool1 → tool2 → tool3 → ...]
  ↓
Executor（逐步执行，每步输出传入下一步）
  ↓
Safety Check（伦理安全验证）
  ↓
Summarizer（综合结果，必要时触发重规划）
```

**蛋白质设计案例**（多步工具链）：
```
设计CATH类别蛋白质序列 → 计算展开力/能量 → 预测3D结构 → 计算振动频率 → 分析二级结构
对应工具链：
design_protein_CATH → calculate_unfold_force_energy → ESMfold
→ calculate_protein_ANM → analyze_protein_structure
```
知识图谱确保每步输出类型与下一步输入类型匹配（蛋白质序列→结构→频率数据→结构元素）。

**性能**：
- SciToolEval基准531题，整体准确率**94%**
- 超过最先进基线**+10%**
- 500+工具（生物、化学、材料科学）

**GitHub**: https://github.com/HICAI-ZJU/SciToolAgent

---

### 2.3 第三类：RAG驱动工具文档检索

#### 2.3.1 BioMaster（bioRxiv 2025.01.23）

**机构**: 中科院/中科大核基因组研究组

**架构**：
```
用户输入
  ↓
Plan Agent + Plan RAG（检索类似工作流模式）
  ↓
工作流计划（步骤序列）
  ↓
Task Agent + Tool RAG（检索工具使用文档）
  ↓
每步可执行脚本
  ↓
执行 + 错误检测 + 自动调试
```

**关键洞察**：Tool RAG不仅包含单个工具文档，还包含**完整工作流示例**，展示工具链接模式（如RNA-seq完整流程）。这为I/O兼容性提供隐式指导。

**覆盖范围**：102个工具，18种组学形态，49个基准任务。

**与其他系统对比**：
- BioMaster vs CRISPR-GPT: BioMaster不局限于单一工具
- BioMaster vs AutoBA: BioMaster通过RAG动态适应新工具
- BioMaster vs BIA: BioMaster包含Plan和Execute两级RAG

**GitHub**: https://github.com/ai4nucleome/BioMaster

#### 2.3.2 BIA（bioRxiv 2024.05.22）

**贡献：静态/动态工作流分类法**

BIA是第一个系统性地将生信工作流分类为：
- **静态工作流**：成熟标准流程（如标准RNA-seq分析），可预先定义工具序列
- **动态工作流**：探索性分析，工具序列取决于数据特性

这一分类对于理解何时可以用预定义schema（静态）、何时需要动态适应（动态）非常重要。

**工具调用架构**：
```
任务描述 → RAG检索（工具包+代码片段+使用场景）→ 工具调用代码生成
```

---

### 2.4 第四类：类型化消息传递

#### 2.4.1 GenoMAS（arXiv 2507.21035，2025年7月）

**机构**: 伊利诺伊大学香槟分校 + 加州大学圣地亚哥分校

**核心问题**：基因组数据分析中，通用自治Agent会产生严重错误。以Manus AI失败案例为例：
```python
# 错误代码（Manus AI生成）
expression_df = process_expression_file(series_matrix_path)
# 返回探针ID作为列名（如'7892501', '7892502'）——科学上无用

# 正确代码（人类专家）
gene_annotation = get_gene_annotation(soft_file_path)
mapping_data = get_gene_mapping(gene_annotation, 'ID', 'gene_assignment')
gene_data = apply_gene_mapping(expression_df=genetic_data, mapping_df=mapping_data)
gene_data_normalized = normalize_gene_symbols_in_index(gene_data)
```

这说明**工具I/O兼容性不仅是格式问题，还需要领域知识**：理解需要探针ID到基因符号的映射，并知道使用哪个注释文件。

**解决方案：类型化消息传递 + 引导规划框架**

六个专业Agent通过类型化消息协议通信：
- Bioinformatics Expert（领域知识验证）
- Data Processor（格式转换）
- Statistician（统计分析）
- Programmer（代码生成执行）
- Reviewer（质量控制）
- Coordinator（任务协调）

消息类型编码数据的生物学意义（如`GeneExpressionMatrix` vs `ProbeIDMatrix`），强制显式的数据转换步骤。

**引导规划框架（Action Units）**：
每个决策点可以：进行（Advance）、修改（Revise）、跳过（Bypass）或回溯（Backtrack）。

**性能**：
- 预处理复合相似度相关性：**89.13%**（vs基线78.52%，+10.61%）
- 基因识别F1：**60.48%**（vs基线43.63%，+16.85%）

**GitHub**: https://github.com/Liu-Hy/GenoMAS

---

### 2.5 第五类：语义类型系统

#### 2.5.1 Playbook Workflow Builder（PLoS Comput. Biol. 2025）

**机构**: 西奈山伊坎医学院Ma'ayan实验室（NIH Common Fund资助）

**核心创新：知识解析图（KRG）**

KRG实现了迄今最强的类型安全保证：

**语义类型（节点）**：
```
Gene（基因符号、Entrez ID）
GeneSet（基因集合）
Variant（SNP、插入缺失）
Drug（ChEMBL ID）
Disease（MeSH ID）
Metabolite（PubChem CID）
GeneExpression（矩阵格式）
Pathway（通路信息）
...
```

**三种Metanode类型**：
- **Prompt**：void → Type（接受用户输入，产生语义类型）
- **Resolver**：Type(s) → Type（转换类型，如Gene → GeneSet）
- **View**：Type → void（可视化，不产生新类型）

**类型安全机制**：
系统动态计算当前数据类型下可用的工具，用户界面只显示与当前类型兼容的后续工具。不兼容的连接**在类型系统层面即不可能发生**。

**API标准化要求**：
> "APIs must be documented with OpenAPI and deposited into the SmartAPI repository."

所有工具必须提供标准化OpenAPI规范，这是加入KRG的前提条件。

**示例合法工作流**：
```
用户输入 → Gene → [Enrich Gene Set (Resolver)] → GeneSet
→ [GSEA分析 (Resolver)] → EnrichmentResults → [火山图 (View)]
```

**GitHub**: https://github.com/MaayanLab/Playbook-Workflow-Builder

---

### 2.6 第六类：协议无关工具注册表

#### 2.6.1 ToolRegistry（arXiv 2507.10593，2025年7月）

**机构**: 芝加哥大学

**问题**：工具集成生态碎片化——OpenAI function calling、MCP、OpenAPI、LangChain各自有不同的schema格式和执行模式。

**解决方案**：统一工具注册表，一个接口管理所有来源的工具：
```python
registry = ToolRegistry()

# 注册Python函数（自动从类型注解生成JSON Schema）
@registry.register
def align_reads(fastq_path: str, genome_ref: str,
                threads: int = 4) -> str:
    """Align reads to reference genome using Bowtie2"""
    ...

# 注册MCP服务器工具（如BioinfoMCP提供的）
registry.register_from_mcp("https://bioinformatics-mcp.server")

# 注册OpenAPI工具
registry.register_from_openapi(config, openapi_spec)

# 获取统一的OpenAI兼容JSON schema
tools_json = registry.get_tools_json()
```

**关键特性**：
- **自动Schema生成**：从Python类型注解自动提取JSON Schema，无需手动编写
- **并发执行**：最高3.1x性能提升
- 60-80%代码减少
- 100%兼容OpenAI function calling标准

**生物信息学意义**：ToolRegistry可以统一管理：
- BioinfoMCP提供的CLI工具（作为MCP工具）
- Python生信库（biopython、scanpy等，作为native函数）
- Web服务API（作为OpenAPI工具）

---

## 3. 理论框架：工具I/O兼容性的三个层次

基于文献综述，提出生信LLM工作流中工具兼容性的三层理论框架：

### 第一层：语法兼容性（Syntactic Compatibility）
**定义**：工具A的输出可以被工具B的输入解析（格式匹配）。

**示例**：
- BAM文件从samtools sort输出 → GATK接受BAM输入 ✓
- FASTQ文件从fastp输出 → STAR接受FASTQ输入 ✓
- 文本字符串 → 任何接受字符串的工具 ✓

**解决工具**：JSON Schema（MCP）、Python类型注解（ToolRegistry）

### 第二层：语义兼容性（Semantic Compatibility）
**定义**：工具A的输出具有工具B所期望的**生物学意义**。

**反例（GenoMAS失败案例）**：
- 探针ID矩阵 ≠ 基因符号矩阵（数据格式相同但生物意义不同）
- 原始计数 ≠ 归一化TPM（都是数值矩阵，但统计意义不同）

**解决工具**：SciToolKG（显式工具依赖）、Playbook KRG（语义类型系统）

### 第三层：语用兼容性（Pragmatic Compatibility）
**定义**：在给定上下文中，先调用工具A再调用工具B是**科学合理的**。

**示例**：
- 质量控制 → 比对 → 定量（顺序不可颠倒）
- 细胞类型注释必须在聚类之后，不能在之前

**解决工具**：BioMaster的Plan RAG（编码工作流模式）、GenoMAS的领域专家Agent

**结论**：现有系统大多解决了第一层（语法兼容性），第二和第三层仍是主要挑战。

---

## 4. 系统对比表

### 4.1 按核心方法分类

| 系统 | 年份 | 方法 | 工具规模 | 类型安全 | 自动化程度 | 同行评审 |
|------|------|------|----------|----------|------------|----------|
| BioinfoMCP | 2025 | MCP自动转换 | 38工具 | 语法（JSON Schema） | 高（LLM从文档） | 否（预印本） |
| SciToolAgent | 2025 | 知识图谱 | 500+工具 | 语义（显式关系） | 中（手动KG） | 是（Nature CS） |
| GenoMAS | 2025 | 类型化消息传递 | N/A | 语义+语用 | 中 | 否（预印本） |
| MCPmed | 2025 | MCP数据库层 | 主要数据库 | 语法（JSON） | 中 | 否（预印本） |
| Playbook WB | 2025 | 语义类型系统 | 若干百个 | **最强（类型理论）** | 低（手动注释） | 是（PLoS CB） |
| WorkflowLLM | 2025 | 数据驱动fine-tune | 1503 API | Python类型 | 高（数据驱动） | 是（ICLR 2025） |
| ToolRegistry | 2025 | 协议无关注册表 | 任意 | 语法 | 高（类型注解） | 否（预印本） |
| BioMaster | 2025 | 双级RAG | 102工具 | 无形式化 | 高（RAG） | 否（预印本） |
| BIA | 2024 | 工具调用+RAG | 若干 | 无形式化 | 中 | 否（预印本） |
| Biomni | 2025 | 统一动作空间 | 150工具 | 无形式化 | 高（文献挖掘） | 否（预印本） |
| CellAgent | 2025/2026 | 多Agent | 单细胞工具 | 通过Agent角色 | 中 | 是（ICLR 2026） |

### 4.2 按生物信息学覆盖范围

| 系统 | RNA-seq | scRNA-seq | ChIP-seq | 变异分析 | 蛋白质 | 化学 | 多组学 |
|------|---------|-----------|----------|----------|--------|------|--------|
| BioinfoMCP | ✓ | ✓ | ✓ | ✓ | 部分 | - | - |
| SciToolAgent | ✓ | - | - | - | ✓ | ✓ | 部分 |
| BioMaster | ✓ | ✓ | ✓ | - | - | - | 部分（Hi-C） |
| GenoMAS | ✓ | - | - | - | - | - | - |
| Biomni | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## 5. 关键发现

### 发现1：MCP协议是生信工具标准化的主流方向

自2024年11月MCP发布后，短短一年内已出现：BioinfoMCP（38工具）、MCPmed（主要数据库）、bio-mcp组织（9+服务器）、scmcphub（单细胞组学）。这一趋势将持续加速。

**对研究者的建议**：设计生信Agent工具集时，优先考虑MCP格式封装。

### 发现2：知识图谱是实现语义I/O兼容性的最有效方法

SciToolAgent通过SciToolKG明确编码工具间的输入输出依赖关系，使LLM能够生成科学有效的工具链，准确率达94%（发表于Nature Computational Science）。

**关键实现细节**：SciToolKG中每个工具节点包含：
- 输入格式规范（含示例）
- 输出格式规范（含示例）
- 先决工具列表
- 兼容后续工具列表

### 发现3：语义类型系统提供最强类型安全但成本最高

Playbook Workflow Builder的KRG方法在理论上最严格，通过语义类型系统使不兼容连接在系统层面不可能发生。但这需要：
- 预定义所有合法生物医学概念类型
- 所有工具提供OpenAPI规范
- 人工标注每个工具的输入/输出类型

**适用场景**：对类型安全要求极高的临床/监管应用场景。

### 发现4：RAG方法在实践中最灵活但不保证类型安全

BioMaster（Plan RAG + Execute RAG）和BIA展示了RAG在实践中的强大灵活性。然而，RAG方法依赖LLM正确推断格式要求，无法在执行前静态验证工具链的I/O兼容性。

**GenoMAS的关键警示**：即使给Manus AI提供了详细提示，它仍然无法正确处理探针ID到基因符号的映射，产生科学上无用的输出。这表明**生物信息学I/O兼容性不能仅靠RAG解决——需要编码领域知识**。

### 发现5：域知识是不可消除的兼容性组件

所有审查的系统都显示，纯粹的格式/协议标准化不足以解决生信工具链中的兼容性问题。以下领域知识必须被系统编码：
- 文件格式语义（探针ID ≠ 基因符号）
- 数据类型规范化（TPM vs RPKM vs raw counts）
- 平台特异性处理（Affymetrix微阵列 vs Illumina短读）
- 质量阈值（在哪个步骤过滤低质量数据）

---

## 6. 推荐架构：三层预设工具系统

基于综述发现，提出以下用于"提前设计好一套工具让LLM Agent使用"的理想三层架构：

```
┌─────────────────────────────────────────────────────────┐
│              Layer 3: 语用层（Pragmatic）                 │
│                 工作流模板库（Plan RAG）                   │
│   RNA-seq模板 | scRNA-seq模板 | ChIP-seq模板 | WGS模板    │
│              专家级工作流模式编码                          │
└─────────────────────────────────────────────────────────┘
                          ↑
┌─────────────────────────────────────────────────────────┐
│              Layer 2: 语义层（Semantic）                  │
│              工具关系知识图谱（Tool KG）                   │
│                                                         │
│  工具节点: { input_type, output_type, prerequisites,    │
│             compatible_successors, domain_context }     │
│                                                         │
│  生物类型系统: FASTQ → BAM/SAM → VCF                    │
│              RawCounts → TPM → DEGList                  │
│              ProbeID → GeneSymbol → NormalizedGene      │
└─────────────────────────────────────────────────────────┘
                          ↑
┌─────────────────────────────────────────────────────────┐
│              Layer 1: 语法层（Syntactic）                 │
│              MCP工具模式注册表（Tool Registry）            │
│                                                         │
│  每个工具: {                                            │
│    name: "samtools_sort",                               │
│    description: "Sort BAM by coordinate",              │
│    inputSchema: { input_bam: "string", ... },          │
│    outputSchema: { output_bam: "string" },             │
│    examples: [...]                                      │
│  }                                                      │
│                                                         │
│  实现: BioinfoMCP（自动生成MCP服务器）                    │
│        ToolRegistry（统一管理多协议工具）                  │
└─────────────────────────────────────────────────────────┘
```

**构建策略**：
1. **Layer 1（自动）**: 用BioinfoMCP自动从工具文档生成MCP服务器
2. **Layer 2（半自动）**: 领域专家构建工具KG；使用LLM辅助提取工具关系
3. **Layer 3（人工+LLM）**: 从现有工作流文档（Nextflow/Snakemake流程）提取模板

---

## 7. 研究空白与开放问题

### 7.1 主要研究空白

**空白1: 生信专用类型系统的缺失**

现有系统使用语法类型（字符串、整数）或高层概念类型（基因、疾病），但缺乏覆盖全生信数据谱的正式类型系统：
```
序列格式: FASTQ → BAM/SAM → VCF/BCF → BED → GTF/GFF3
矩阵类型: 原始计数 → TPM/RPKM → 差异表达基因
分子类型: 蛋白质序列 → 3D结构 → 结合位点
单细胞: 细胞×基因矩阵 → UMAP嵌入 → 细胞类型标签
```

**研究机会**: 定义涵盖主要生信数据类型的**BioType System**，含子类型关系（如`NormalizedExpressionMatrix`是`ExpressionMatrix`的子类型）。

**空白2: 工具链的静态验证**

目前没有工具能在执行前**静态验证**工具链的I/O兼容性。类似TypeScript类型检查器的"BioTool Type Checker"尚不存在。

**空白3: 工具版本化管理**

生信工具频繁更新，但现有系统没有自动跟踪schema变化和版本兼容矩阵的机制。

**空白4: 多组学集成的类型桥接**

不同组学层（基因组/转录组/蛋白质组/代谢组）的数据结构、测量单位和生物实体根本不同，目前没有系统定义这些层之间的标准转换接口。

### 7.2 次要研究空白

- **可重现性标准**: LLM编排的生信工作流缺乏确保结果可重现的标准
- **隐私保护**: 工具schema不编码临床数据的隐私级别要求
- **成本/性能权衡**: 工具schema不包含性能特征以指导工具选择
- **新工具发现**: 所有系统要求提前知道工具集，对未知工具无法处理

---

## 8. 结论

### 8.1 主要结论

1. **MCP是语法兼容性的解决方案**: BioinfoMCP展示了从文档自动生成标准化工具接口的可行性（38工具，94.7%成功率），这是"提前设计工具模式"的基础设施。

2. **知识图谱是语义兼容性的关键**: SciToolAgent证明，只有明确编码工具间的输入输出类型依赖（而非让LLM自行推断），才能可靠地生成正确的多步工具链（发表于Nature Computational Science，+10%基线）。

3. **域知识不可替代**: GenoMAS的对比实验清楚显示，探针ID→基因符号映射等生物学变换需要领域知识，纯粹的协议标准化无法替代。这意味着预设工具模式必须包含生物学语义，而不仅仅是参数类型。

4. **三层架构是理想方案**: 语法层（MCP）+ 语义层（工具KG + 生物类型系统）+ 语用层（工作流模板库）共同构成完整的预设工具兼容性保证。

5. **现有方案各有优劣**: 没有任何单一现有系统同时解决了三个兼容性层次。BioinfoMCP解决了语法层，SciToolAgent解决了部分语义层，Playbook WB解决了语义层（但仅限Web服务），BioMaster/GenoMAS解决了语用层。

### 8.2 对实践者的建议

**设计预设生信工具集时**：

1. **首选MCP封装所有工具**（利用BioinfoMCP自动化转换或bio-mcp组织的模板）
2. **构建工具依赖知识图谱**（参考SciToolKG方法）：明确每个工具的前置和后置工具
3. **定义生物类型词汇表**（参考Playbook WB的KRG方法）：至少涵盖FASTQ、BAM、VCF、表达矩阵、基因集等核心类型
4. **收集工作流模板**（参考BioMaster的Plan RAG）：为RNA-seq、scRNA-seq、ChIP-seq等主流流程提供验证过的工具链示例
5. **为工具提供丰富示例**（参考SciAgent的工具schema格式）：每个工具schema应包含输入输出示例，帮助LLM正确推断格式

---

## 参考文献

### 核心论文（深度阅读）

[1] Widjaja F, Chen Z, Zhou J. "BioinfoMCP: A Unified Platform Enabling MCP Interfaces in Agentic Bioinformatics." arXiv:2510.02139, 2025. (preprint)

[2] Ding K, Yu J, Huang J, Yang Y, Zhang Q, Chen H. "SciToolAgent: a knowledge-graph-driven scientific agent for multitool integration." *Nature Computational Science*, 5:962-972, 2025. (peer-reviewed)

[3] Liu H, Li Y, Wang H. "GenoMAS: A Multi-Agent Framework for Scientific Discovery via Code-Driven Gene Expression Analysis." arXiv:2507.21035, 2025. (preprint)

[4] Flotho M, et al. "MCPmed: A Call for MCP-Enabled Bioinformatics Web Services for LLM-Driven Discovery." arXiv:2507.08055, 2025. (preprint)

[5] Fan S, Cong X, Fu Y, et al. "WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models." *ICLR 2025*. (peer-reviewed)

[6] Ding P. "ToolRegistry: A Protocol-Agnostic Tool Management Library for Function-Calling LLMs." arXiv:2507.10593, 2025. (preprint)

[7] (Authors). "BioInformatics Agent (BIA): Unleashing the Power of Large Language Models to Reshape Bioinformatics Workflow." *bioRxiv*, 2024. (preprint)

[8] (Authors, Stanford/Genentech). "Biomni: A General-Purpose Biomedical AI Agent." *bioRxiv*, 2025. (preprint)

[9] Clarke DJ, Evangelista JE, Xie Z, et al. "Playbook workflow builder: Interactive construction of bioinformatics workflows." *PLoS Computational Biology*, 21(4):e1012901, 2025. (peer-reviewed)

[10] Su H, et al. "BioMaster: Multi-agent System for Automated Bioinformatics Analysis Workflow." *bioRxiv*, 2025. (preprint)

### 生物信息学Agent系统

[11] (Authors). "CellAgent: LLM-Driven Multi-Agent Framework for Natural Language-Based Single-Cell Analysis." *ICLR 2026*. (peer-reviewed)

[12] (Authors). "scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery." *NeurIPS 2025*. (peer-reviewed)

[13] Wu S, et al. "BioAgents: Democratizing Bioinformatics Analysis with Multi-Agent Systems." *arXiv:2501.06314*, 2025. (preprint, also Scientific Reports)

[14] (Authors). "BioDiscoveryAgent: An AI Agent for Designing Genetic Perturbation Experiments." *ICLR 2025*. (peer-reviewed)

[15] (Authors). "MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Research." *ICLR 2026 Oral*. (peer-reviewed)

[16] (Authors). "FlowAgent: A Modular Agent-Based System for Automated Workflow Management and Data Interpretation." *bioRxiv*, 2025. (preprint)

[17] (Authors). "GenoMAS/GenoTEX: An LLM Agent Benchmark for Automated Gene Expression Data Analysis." arXiv:2406.15341, MLCB 2025 Oral. (peer-reviewed)

### 通用工具使用框架

[18] (Authors). "ToolGen: Unified Tool Retrieval and Calling via Generation." *ICLR 2025*. (peer-reviewed)

[19] (Authors). "ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models." *ICLR 2026*. (peer-reviewed)

[20] (Authors). "ToolACE: Winning the Points of LLM Function Calling." *ICLR 2025*. (peer-reviewed)

[21] (Authors). "ToolTree: Efficient LLM Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning." *ICLR 2026*. (peer-reviewed)

[22] (Authors). "Flow: Modularized Agentic Workflow Automation." *ICLR 2025*. (peer-reviewed)

[23] (Authors). "DyFlow: Dynamic Workflow Framework for Agentic Reasoning." *NeurIPS 2025*. (peer-reviewed)

[24] Wang Y, et al. "AvaTaR: Optimizing LLM Agents for Tool Usage via Contrastive Reasoning." *NeurIPS 2024*. (peer-reviewed)

[25] (Authors). "Large Language Models as Tool Makers." *ICLR 2024*. (peer-reviewed)

[26] (Authors). "CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets." *ICLR 2024*. (peer-reviewed)

### 科学发现Agent

[27] Ma R, et al. "SciAgent: Tool-augmented Language Models for Scientific Reasoning." *EMNLP 2024*. (peer-reviewed)

[28] (Authors). "Expert-level protocol translation for self-driving labs." *NeurIPS 2024*. (peer-reviewed)

[29] (Authors). "Hierarchically Encapsulated Representation for Protocol Design in Self-Driving Labs." *ICLR 2025*. (peer-reviewed)

[30] (Authors). "ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery." *ICLR 2025*. (peer-reviewed)

[31] (Authors). "AI-Researcher: Autonomous Scientific Innovation." *NeurIPS 2025 Spotlight*. (peer-reviewed)

[32] (Authors). "ScienceBoard: Evaluating Multimodal Autonomous Agents in Realistic Scientific Workflows." *ICLR 2026*. (peer-reviewed)

[33] (Authors). "OrchestrationBench: LLM-Driven Agentic Planning and Tool Use in Multi-Domain Scenarios." *ICLR 2026*. (peer-reviewed)

---

## 附录：关键工具和代码资源

| 资源 | URL | 描述 |
|------|-----|------|
| BioinfoMCP | https://github.com/florensiawidjaja/BioinfoMCP | 生信工具MCP转换平台 |
| bio-mcp组织 | https://github.com/bio-mcp | MCP服务器集合（samtools、bcftools等）|
| SciToolAgent | https://github.com/HICAI-ZJU/SciToolAgent | 知识图谱驱动科学Agent |
| BioMaster | https://github.com/ai4nucleome/BioMaster | 多Agent生信工作流自动化 |
| GenoMAS | https://github.com/Liu-Hy/GenoMAS | 基因表达分析多Agent框架 |
| Playbook WB | https://github.com/MaayanLab/Playbook-Workflow-Builder | 语义类型生信工作流构建器 |
| FlowAgent | https://github.com/EnteloBio/flowagent | 模块化工作流Agent |
| WorkflowLLM | https://github.com/OpenBMB/WorkflowLLM | 工作流编排LLM训练框架 |
| ToolRegistry | https://github.com/Oaklight/ToolRegistry | 协议无关工具管理库 |
| Biomni | https://github.com/snap-stanford/Biomni | Stanford通用生物医学Agent |
| MCPmed | https://github.com/MCPmed | MCP生信数据库接口 |
| awesome-bioagent | https://github.com/aristoteleo/awesome-bioagent-papers | 生物Agent论文列表 |
