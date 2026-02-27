# PDF-to-LLM Conversion Tools for Academic Textbooks: A Comprehensive Survey

## 1. Introduction

随着大语言模型（LLM）Agent 的兴起，将 PDF 格式教材系统地转化为 LLM 可直接阅读的格式（Markdown、JSON、结构化文本）成为关键需求。本报告系统调研了开源免费的 PDF 转换工具，重点关注四类教材的特殊需求：数学教材（公式）、生物教材（图表、树状图、通路图）、信息学教材（代码、伪代码）和计算机系统教材（架构图、状态图）。

The field has undergone rapid evolution from 2023 to 2025, transitioning from rule-based text extraction to multi-model pipelines, and now to vision-language models (VLMs) that process entire pages end-to-end. This survey covers **12 major tools**, evaluated across **4 textbook domains**, with data from the OmniDocBench benchmark (CVPR 2025) and other standardized evaluations.

---

## 2. Background

### 2.1 The LLM-Readable Format Challenge

LLM Agents need structured text input that preserves:
- **Semantic structure**: headings, paragraphs, lists
- **Mathematical notation**: LaTeX for formulas and equations
- **Tabular data**: HTML or Markdown tables
- **Code content**: fenced code blocks with language tags
- **Reading order**: sequential, natural text flow
- **Visual content**: images with text descriptions or structured representations

PDF is a display-oriented format that lacks this semantic structure, making conversion non-trivial.

### 2.2 Key Evaluation Metrics

| Metric | What It Measures | Used For |
|--------|-----------------|----------|
| **Edit Distance** | Character-level text accuracy | Text extraction quality |
| **CDM** (Character Detection Matching) | Formula recognition accuracy | Math formula quality |
| **TEDS** (Tree Edit Distance-based Similarity) | Table structure + content accuracy | Table extraction quality |
| **F1-score** | Precision × Recall balance | Layout detection, OCR accuracy |
| **BLEU** | N-gram overlap with reference | Formula/text generation quality |

### 2.3 Benchmark: OmniDocBench (CVPR 2025) [@ouyang2024omnidocbench]

The most comprehensive benchmark for document parsing, featuring 1,355 PDF pages across 9 document types (academic papers, textbooks, financial reports, etc.), with 15 block-level and 4 span-level annotations. Evaluation covers text, tables, formulas, and reading order.

---

## 3. Taxonomy of Approaches

### 3.1 End-to-End Vision-Language Models (VLMs)

These models process page images directly and output structured text.

**Large VLMs (>1B parameters)**:
- **olmOCR** [@clo2025olmocr] (preprint): Allen AI, 7B VLM (Qwen2.5-VL-7B base), document-anchoring technique + GRPO reinforcement learning. Optimized for large-scale batch processing at ~$439 per million pages. Top on its own benchmark (OLMOCR-BENCH). English-focused.
- **MinerU 2.5** (preprint): OpenDataLab, 1.2B VLM with decoupled coarse-to-fine architecture. Stage I: global layout on downsampled images; Stage II: native-resolution content recognition. OmniDocBench score: **90.67**. Best accuracy-speed tradeoff.
- **dots.ocr** (preprint): RedNote (小红书), 1.7B LLM backbone. SOTA multilingual support (100+ languages). OmniDocBench score: 88.41.

**Compact VLMs (<1B parameters)**:
- **GOT-OCR2.0** [@wei2024general] (preprint): StepFun, 580M parameters. Most versatile — handles text, math formulas, molecular structures (SMILES), tables, charts, sheet music (kern format), and geometric shapes (TikZ). F1-scores: math 0.865, tables 0.878, geometry 0.884.
- **Nougat** [@blecher2023nougat]: Meta, ~250M, Donut-based. Trained on arXiv. Pioneered academic PDF→Markdown. Strong math but poor generalization to non-academic content.
- **Granite-Docling** (preprint): IBM, 258M (Idefics3 + SigLIP2 + Granite LLM). Apache 2.0 license. Code recognition F1: 0.988. Equation F1: 0.968.

### 3.2 Pipeline-Based (Multi-Model Ensemble)

These combine specialized models for layout detection, OCR, formula recognition, and table extraction.

- **MinerU (pipeline)** [@wang2024mineru] (preprint): OpenDataLab. LayoutLMv3/DocLayout-YOLO → UniMERNet (formulas) + TableMaster/StructEqTable (tables) + PaddleOCR (text). Top pipeline tool on OmniDocBench. 54.8K GitHub stars.
- **Marker** [@paruchuri2024marker] (preprint): Datalab. Surya OCR suite (5 models). Key differentiator: LLM integration (Gemini, Ollama, OpenAI, Claude). With LLM: table TEDS 0.907. 31.8K stars. GPL-3.0 license.
- **Docling** [@auer2024docling]: IBM → Linux Foundation. RT-DETR layout + TableFormer. **MIT license**. Broadest input format support (16+ formats). Best table extraction (TEDS 97.9%). LangChain/LlamaIndex integration. 53.9K stars. AAAI 2025.
- **PaddleOCR / PP-StructureV3**: Baidu. Most mature ecosystem. 100+ languages. Apache 2.0. PP-OCRv5 (2025). 50K+ stars.

### 3.3 LLM-Based / Hybrid

- **Marker + LLM**: Pipeline extraction + LLM refinement (Gemini/Ollama). Best when highest accuracy is needed.
- **Zerox**: Zero-shot PDF OCR using GPT-4o-mini. No training needed, but depends on proprietary API.
- **MinerU hybrid-auto-engine**: Default since v2.5+. Combines pipeline reliability with VLM understanding.

### 3.4 Specialized Components

- **UniMERNet** (OpenDataLab): SOTA formula recognition. 1.06M training pairs. CDM metric. Three model sizes (441MB-1.3GB).
- **Pix2Text** (Breezedeus): Free Mathpix alternative. MFR 1.5 model. 80+ languages. Lightweight, offline-capable.
- **PDF-Extract-Kit** (OpenDataLab): Chemical equation/diagram recognition. Formula detection + recognition. Layout detection.
- **LaTeX-OCR** (pix2tex): ViT-based equation → LaTeX. MIT license. 13K+ stars.

---

## 4. Detailed Comparative Analysis

### 4.1 Overall Benchmark (OmniDocBench v1.5)

| Rank | Tool | Score | Type | Params |
|------|------|-------|------|--------|
| 1 | MonkeyOCR v1.5 | 93.01 | VLM | — |
| 2 | **MinerU 2.5** | **90.67** | VLM | 1.2B |
| 3 | MonkeyOCR-Pro-3B | 88.85 | VLM | 3B |
| 4 | **dots.ocr** | 88.41 | VLM | 1.7B |
| 5 | Gemini-2.5 Pro | 88.03 | VLM (API) | — |

### 4.2 Per-Component Accuracy

| Component | Best Open-Source Tool | Score |
|-----------|----------------------|-------|
| Text (English) | MinerU 2.5 | Edit Distance 0.047 |
| Math Formulas | MinerU 2.5 (UniMERNet) | CDM 88.46 |
| Tables | Docling (TableFormer) | TEDS 97.9% |
| Reading Order | MinerU 2.5 | Edit Distance 0.044 |
| Code Recognition | Granite-Docling | F1 0.988 |
| Chart Understanding | GOT-OCR2.0 | AP@strict 0.747 |
| Chemical Diagrams | PDF-Extract-Kit | Supported (unique) |

### 4.3 Speed & Cost

| Tool | Hardware | Speed | Cost per 1M pages |
|------|----------|-------|--------------------|
| olmOCR | H100 | 3,400 tok/s | ~$439 |
| MinerU 2.5 | A100 80GB | 2.12 pages/sec | Compute cost |
| MinerU 2.5 | RTX 4090 | 1.70 pages/sec | Compute cost |
| Marker | H100 (batch) | ~122 pages/sec | Compute cost |
| Docling | M3 Max | 1.27 pages/sec | Compute cost |
| Docling | RTX 5090 | 7.9 pages/sec | Compute cost |

### 4.4 License Comparison

| License | Tools | Commercial Use | Key Restriction |
|---------|-------|---------------|-----------------|
| **MIT** | Docling, Nougat, pdfplumber, LaTeX-OCR | Freely | None |
| **Apache 2.0** | olmOCR, PaddleOCR, Granite-Docling, PDF-Extract-Kit, UniMERNet | Freely | Patent clause |
| **GPL-3.0** | Marker | Restricted | Must open-source derivatives |
| **AGPL-3.0** | MinerU, PyMuPDF | Network copyleft | Server-side use triggers copyleft |

---

## 5. Domain-Specific Recommendations

### 5.1 数学教材 (Math Textbooks)

**核心挑战**: 行内公式 vs 行间公式识别、多行方程、方程编号、手写公式

**推荐方案 (按优先级)**:

| 优先级 | 工具 | 原因 | 公式 CDM |
|--------|------|------|----------|
| 1 | **MinerU 2.5** (VLM backend) | 综合最优，公式 CDM 88.46，支持行内/行间 | 88.46 |
| 2 | **Pix2Text** (MFR 1.5) | 轻量级，免费 Mathpix 替代，离线可用 | SOTA (lightweight) |
| 3 | **GOT-OCR2.0** | 紧凑模型(580M)，公式 F1 0.865 | F1 0.865 |
| 4 | **Nougat** | 学术论文专用，但泛化性差 | Moderate |
| ❌ | **Marker** | 公式能力弱 (CDM 18.4%)，不推荐用于数学教材 | 18.4% |

**工作流**: `PDF → MinerU 2.5 (VLM) → Markdown + LaTeX → KaTeX 验证 → LLM 可读文本`

### 5.2 生物教材 (Biology Textbooks)

**核心挑战**: 系统发育树、代谢通路图、分子结构、复杂表格、带标签的图片

**推荐方案**:

| 优先级 | 工具 | 原因 |
|--------|------|------|
| 1 | **Docling** | 表格提取最佳 (TEDS 97.9%)，MIT 许可证 |
| 2 | **MinerU** | 综合准确度最高 |
| 3 | **PDF-Extract-Kit** | 唯一支持化学方程/图表识别的工具 |
| 补充 | **GPT-4o/Claude Vision** | 对提取的图片进行描述（非开源，但目前唯一方案） |

**工作流**: `PDF → Docling → Markdown + HTML 表格 + 图片提取 → 图片 → LLM Vision 描述 → 合并为 LLM 可读文本`

**重大缺口**: 没有开源工具能将系统发育树转为 Newick 格式，或将代谢通路转为 SBML 格式。

### 5.3 信息学/CS 教材 (Informatics/CS Textbooks)

**核心挑战**: 语法高亮的代码、伪代码、算法框、程序输出、脚本语言

**推荐方案**:

| 优先级 | 工具 | 原因 |
|--------|------|------|
| 1 | **Marker + LLM** | LLM 后处理提升代码/伪代码识别 |
| 2 | **MinerU 2.5** (VLM backend) | 支持 `code` 和 `algorithm` 块类型 |
| 3 | **Granite-Docling** | 代码识别 F1 0.988（最高） |
| 4 | **Docling** | 支持 LaTeX 输入，可处理 LaTeX 源文件 |

**工作流**: `PDF → Marker + LLM (Gemini/Ollama) → Markdown (含代码块) → LLM 可读文本`

### 5.4 计算机系统教材 (Computer Systems Textbooks)

**核心挑战**: 架构图、状态图、流程图、电路图、存储层次图

**推荐方案**:

| 优先级 | 工具 | 原因 |
|--------|------|------|
| 1 | **MinerU 2.5** | 最佳文本准确度和阅读顺序 |
| 2 | **Docling** | 复杂布局处理、表格提取 |
| 补充 | **GPT-4o/Claude Vision** | 图表描述（非开源） |

**工作流**: `PDF → MinerU 2.5 → Markdown + 图片提取 → 图片 → LLM Vision 描述 → 合并`

**重大缺口**: 架构图、状态机、流程图的结构化提取完全没有开源方案。所有工具只能将其作为图片提取。

---

## 6. Practical Implementation Guide

### 6.1 Hardware Requirements

| 方案 | 最低 VRAM | 推荐 GPU | CPU 可用? |
|------|-----------|----------|-----------|
| MinerU 2.5 (VLM) | 10 GB | RTX 4090 / A100 | 不推荐 |
| MinerU (pipeline) | 6 GB | RTX 3060+ | 可以但慢 |
| Marker | 3.5 GB | 任何 CUDA GPU | 可以 (~16s/页) |
| Marker + LLM | 3.5 GB + API | 任何 CUDA GPU | 可以 |
| Docling | 6 GB | 任何 CUDA GPU | 可以 (~3s/页) |
| Pix2Text | < 2 GB | 可选 | 可以 |
| GOT-OCR2.0 | ~4 GB | CUDA 11.8+ | 不推荐 |
| olmOCR | 12-20 GB | RTX 4090 / A100 / H100 | 不支持 |

### 6.2 Quick Start Commands

```bash
# === 方案 A: 数学教材 (推荐) ===
pip install uv && uv pip install -U "mineru[all]"
mineru -p textbook.pdf -o output/ -b hybrid-auto-engine

# === 方案 B: CS/信息学教材 ===
pip install marker-pdf[full]
marker_single textbook.pdf --output_format markdown --use_llm --gemini_api_key YOUR_KEY

# === 方案 C: 生物教材 (MIT 许可证) ===
pip install docling
python -c "
from docling.document_converter import DocumentConverter
converter = DocumentConverter()
result = converter.convert('textbook.pdf')
print(result.document.export_to_markdown())
"

# === 方案 D: 轻量级数学公式提取 ===
pip install pix2text
p2t predict -l en --file-type pdf -i textbook.pdf -o output/
```

### 6.3 Batch Processing at Scale

对于大量教材的批量转换：

1. **olmOCR** — 最经济的大规模方案 ($439/百万页)
2. **MinerU 2.5 + vLLM** — A100 上 2.12 页/秒
3. **Marker batch mode** — `marker /path/to/pdfs/ --output_format markdown`
4. **Docling docling-serve** — API 部署用于生产环境

---

## 7. Open Problems and Future Directions

### 7.1 重大未解决问题

1. **图表结构化提取**: 流程图、架构图、状态图、系统发育树等无法转为机器可读格式。这是当前最大的缺口。
2. **伪代码结构化**: 算法框中的伪代码无法准确转为结构化算法表示。
3. **跨页内容**: 跨页表格、方程、段落的连续性处理仍不完善。
4. **手写内容**: 教材中的手写示例和注释识别准确度很低。
5. **多语言学术内容**: 非英语学术内容的准确度仍有较大差距。

### 7.2 预期发展方向

1. **VLM + Pipeline 混合架构** 将成为主流（MinerU hybrid-auto-engine 已是默认）
2. **领域特定微调** — 针对不同学科教材的专用模型
3. **更小更快的模型** — 从 Granite-Docling (258M) 的成功可见
4. **图表理解 VLM** — 训练 VLM 将图表转为 Mermaid/PlantUML/TikZ
5. **统一基准** — 扩展 OmniDocBench 覆盖教材特定元素

---

## 8. Conclusion

### 总结推荐

| 教材类型 | 首选工具 | 备选工具 | 关键注意事项 |
|---------|---------|---------|------------|
| **数学** | MinerU 2.5 | Pix2Text | 公式转 LaTeX 质量最关键 |
| **生物** | Docling + PDF-Extract-Kit | MinerU | 图表只能提取为图片 |
| **信息学/CS** | Marker + LLM | MinerU 2.5 (VLM) / Granite-Docling | 代码/伪代码识别质量关键 |
| **计算机系统** | MinerU 2.5 | Docling | 架构图无结构化方案 |
| **通用/全科** | MinerU 2.5 | Docling (MIT 许可证) | 综合最优 |

### 如果只选一个工具

**MinerU 2.5** (hybrid-auto-engine) 是目前综合能力最强的开源 PDF-to-LLM 转换工具：
- OmniDocBench 排名第二（仅次于闭源 MonkeyOCR）
- 公式、文本、阅读顺序全面领先
- 支持代码/算法块（VLM 模式）
- 速度快（A100 上 2.12 页/秒）
- 活跃开发（54.8K GitHub stars）

**但需注意**:
- AGPL-3.0 许可证有商业限制（如需 MIT 许可证，选 Docling）
- 图表/diagrams 仍只能提取为图片
- 需要 GPU（10GB+ VRAM）获得最佳效果

### 实际工作流建议

对于系统地转换大量教材 PDF 为 LLM 可读格式：

```
Step 1: MinerU 2.5 批量转换 → Markdown + LaTeX + HTML tables + 图片
Step 2: 提取的图片 → GPT-4o/Claude Vision 生成文字描述
Step 3: 合并 Step 1 + Step 2 → 完整的 LLM 可读文档
Step 4: 质量检查 (抽样人工验证公式、表格、阅读顺序)
```

---

## References

See `references.bib` for full bibliography (70 entries).

Key references:
- OmniDocBench (CVPR 2025): Comprehensive document parsing benchmark [@ouyang2024omnidocbench]
- Docling (AAAI 2025): Efficient open-source toolkit [@auer2024docling]
- Nougat: Academic document understanding via VLM [@blecher2023nougat]
- olmOCR: Large-scale PDF processing with VLM (preprint) [@clo2025olmocr]
- GOT-OCR2.0: General OCR theory via unified model (preprint) [@wei2024general]
- MinerU: High-quality PDF extraction toolkit (preprint) [@wang2024mineru]
