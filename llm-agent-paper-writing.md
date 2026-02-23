# LLM Agent 自动论文写作框架：开源生态调研报告

> **调研日期**: 2026-02-19
> **调研目标**: 梳理LLM Agent自动写论文、规划code base、做实验、画图、LaTeX写论文的开源GitHub实现，为半自动/全自动化论文写作选型提供依据

---

## 一、背景与现状

2024-2025年，LLM Agent驱动的自动化科研写作经历了从概念验证到可用框架的快速演进。核心里程碑包括：

- **2024.08**: Sakana AI 发布 AI Scientist v1，首个端到端全自动科研系统
- **2025.01**: Agent Laboratory 发布，引入 Copilot Mode 实现人机协作
- **2025.04**: AI Scientist v2 生成首篇通过同行评审的AI论文（ICLR 2025 Workshop）
- **2025.05**: AI-Researcher 被接收为 NeurIPS 2025 Spotlight 论文

当前共收集到 **23个核心开源项目**，涵盖从文献综述到论文生成的完整流程。

---

## 二、核心框架详解

### 2.1 端到端系统

#### AI Scientist v2 [@sakana2025aiscientistv2]
- **仓库**: [SakanaAI/AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) | ⭐ 2.1k
- **状态**: ICLR 2025 Workshop（首个通过同行评审的AI论文）
- **架构**: Agentic Tree Search + 4阶段实验管理 + VLM图表反馈
- **成本**: ~$15/论文 | **亮点**: 无需人工代码模板，多模型协作

#### AI-Researcher [@tang2025airesearcher]
- **仓库**: [HKUDS/AI-Researcher](https://github.com/HKUDS/AI-Researcher) | ⭐ 4.5k
- **状态**: **NeurIPS 2025 Spotlight**
- **架构**: 7个专用Agent + 原子组件分解 + 发散-收敛创意框架
- **输入**: 10-15篇参考论文 → **输出**: 完整学术论文+代码
- **亮点**: 数学-代码双向映射、Docker容器化、领域模板、93.8%实现完整率
- **生产版**: [novix.science](https://novix.science/chat)

#### Agent Laboratory [@schmidgall2025agentlab]
- **仓库**: [SamuelSchmidgall/AgentLaboratory](https://github.com/SamuelSchmidgall/AgentLaboratory) | ⭐ 5.3k
- **状态**: EMNLP 2025 Findings
- **架构**: 5种Agent角色(PhD/Postdoc/ML Engineer/Professor/Reviewer×3)
- **三阶段**: Literature Review → Experimentation → Report Writing
- **核心特色**: **Copilot Mode** — 人类可在每个阶段检查点介入
- **成本**: $2.33-$13.10/次 | MLE-Bench 4枚奖牌(最佳)

#### data-to-paper [@kishony2024datatopaper]
- **仓库**: [Technion-Kishony-lab/data-to-paper](https://github.com/Technion-Kishony-lab/data-to-paper) | ⭐ 756
- **状态**: NEJM AI
- **架构**: 16阶段精细流水线（从数据探索到PDF编译）
- **核心创新**: **Backward Traceability** — 论文中任何数值可点击追溯到具体代码行
- **准确率**: 简单目标80-90%可复现同行评审结果

---

### 2.2 文献综述与知识生成

#### STORM [@shao2024storm]
- **仓库**: [stanford-oval/storm](https://github.com/stanford-oval/storm) | ⭐ **27.9k**（最受欢迎）
- **状态**: NAACL 2024
- **架构**: 多视角对话模拟（N=5视角, M=5轮）→ 大纲生成 → 分节写作
- **引用质量**: 精确率85.18%, 召回率84.83%
- **扩展**: Co-STORM支持人机协作

#### AutoSurvey [@yang2024autosurvey]
- **仓库**: [AutoSurveys/AutoSurvey](https://github.com/AutoSurveys/AutoSurvey) | ⭐ 458
- **状态**: NeurIPS 2024
- **功能**: 并行多LLM生成子节 + RAG实时更新 → 自动综述论文

#### GPT-Researcher
- **仓库**: [assafelovic/gpt-researcher](https://github.com/assafelovic/gpt-researcher) | ⭐ **25.4k**
- **功能**: 深度研究任意主题 → 2K+字带引用报告
- **亮点**: 递归深度研究、多种检索器支持

---

### 2.3 论文→代码

#### Paper2Code / PaperCoder [@paper2code2025]
- **仓库**: [going-doer/Paper2Code](https://github.com/going-doer/Paper2Code) | ⭐ 4.2k
- **状态**: ICLR 2026
- **架构**: Planning(总体计划+UML图+依赖图+配置) → Analysis → Coding
- **关键指标**: PaperBench 44.26%, 人评77%最佳, 仅需修改0.48%代码行

---

### 2.4 科学可视化/画图

#### MatPlotAgent [@yang2024matplotagent]
- **仓库**: [thunlp/MatPlotAgent](https://github.com/thunlp/MatPlotAgent) | ⭐ 105
- **架构**: 查询扩展 → 代码生成(自修复×3) → **GPT-4V视觉反馈** → 迭代优化
- **关键指标**: GPT-4从48.86→61.16(+12.3), 6.7B模型使用后超过GPT-4直接生成

#### LIDA (Microsoft Research)
- **网站**: [microsoft.github.io/lida](https://microsoft.github.io/lida/)
- **功能**: Grammar-agnostic可视化 + 信息图生成（4模块系统）

---

### 2.5 编码Agent（实验代码编写）

#### OpenHands
- **仓库**: [OpenHands/OpenHands](https://github.com/OpenHands/OpenHands) | ⭐ **68.0k**
- **状态**: ICLR 2025 | 最通用AI编码平台

#### Aider
- **仓库**: [Aider-AI/aider](https://github.com/Aider-AI/aider) | ⭐ **40.8k**
- **功能**: 终端AI配对编程，支持Claude/GPT/DeepSeek/本地模型

#### SWE-agent
- **仓库**: [SWE-agent/SWE-agent](https://github.com/SWE-agent/SWE-agent) | ⭐ 18.5k
- **状态**: NeurIPS 2024 | 自动修复GitHub Issue

---

### 2.6 审稿与想法生成

| 工具 | 功能 | 状态 |
|------|------|------|
| ChatReviewer | 自动论文审稿+回复审稿意见 | 开源 |
| CycleResearcher | 研究+审稿循环训练 | ICLR 2025 |
| SciMON | 文献启发→新颖性优化→想法生成 | ACL 2024 |
| ResearchAgent | 基于学术图谱的迭代想法生成 | (preprint) |

---

## 三、核心对比表

| 框架 | Stars | 会议 | 输入 | 人机交互 | 论文质量 | 成本 |
|------|-------|------|------|---------|---------|------|
| AI Scientist v2 | 2.1k | ICLR WS | 主题 | 无 | Workshop级 | $15 |
| AI-Researcher | 4.5k | NeurIPS Spotlight | 论文列表 | 无 | 15-79%可比人类 | - |
| Agent Laboratory | 5.3k | EMNLP | 想法 | **Copilot** | 3.5-4.4/10 | $2-13 |
| data-to-paper | 756 | NEJM AI | 数据 | 副驾驶 | 80-90%复现 | - |
| STORM | 27.9k | NAACL | 主题 | Co-STORM | 高质量综述 | - |

---

## 四、推荐方案

### 方案A：模块化半自动流水线 ⭐推荐

最灵活、最可控的方案。按需组合：

| 阶段 | 工具 | 自动化程度 |
|------|------|-----------|
| 文献综述/Related Work | **STORM** | 高 |
| 代码实现 | **Aider** / Claude Code | 高 |
| 运行实验 | 手动 + Python脚本 | 低-中 |
| 生成图表 | **MatPlotAgent** | 中-高 |
| 写LaTeX论文 | **Agent Laboratory (Copilot)** | 中 |
| 自查审稿 | ChatReviewer | 中 |

### 方案B：端到端自动

适合快速原型或Workshop级别投稿：
- **AI-Researcher**: 输入参考论文→完整论文（最完整）
- **AI Scientist v2**: 输入主题→自动探索+实验+论文（最自主）

### 方案C：数据驱动研究

适合从数据出发的实证研究（生物医学/社科等）：
- **data-to-paper**: 数据→可回溯论文

---

## 五、下一步建议

1. **短期（1-2周）**:
   - Clone并试用 Agent Laboratory (Copilot Mode)
   - 集成 STORM 到现有文献综述工作流
   - 试用 MatPlotAgent 生成论文图表

2. **中期（1-3个月）**:
   - 构建自定义模块化流水线
   - 集成 实验结果→LaTeX表格 的自动转换
   - 基于 AI-Researcher 的模板思路构建领域特定模板

3. **长期**:
   - 关注 ICLR 2026 新接收的研究Agent论文
   - 构建自己的论文写作Agent（基于现有框架组合+定制）

---

## 六、参考资源

### 核心GitHub仓库
| 仓库 | URL | Stars |
|------|-----|-------|
| AI Scientist v2 | github.com/SakanaAI/AI-Scientist-v2 | 2.1k |
| AI-Researcher | github.com/HKUDS/AI-Researcher | 4.5k |
| Agent Laboratory | github.com/SamuelSchmidgall/AgentLaboratory | 5.3k |
| data-to-paper | github.com/Technion-Kishony-lab/data-to-paper | 756 |
| STORM | github.com/stanford-oval/storm | 27.9k |
| AutoSurvey | github.com/AutoSurveys/AutoSurvey | 458 |
| Paper2Code | github.com/going-doer/Paper2Code | 4.2k |
| MatPlotAgent | github.com/thunlp/MatPlotAgent | 105 |
| OpenHands | github.com/OpenHands/OpenHands | 68.0k |
| Aider | github.com/Aider-AI/aider | 40.8k |
| GPT-Researcher | github.com/assafelovic/gpt-researcher | 25.4k |

### Survey & Awesome Lists
- [Awesome-LLM-Scientific-Discovery](https://github.com/HKUST-KnowComp/Awesome-LLM-Scientific-Discovery) (EMNLP 2025)
- [LLM4SR](https://github.com/du-nlp-lab/LLM4SR) — LLM for Scientific Research Survey
- [Awesome-Agent-Papers](https://github.com/luo-junyu/Awesome-Agent-Papers)

### 关键论文
1. AI Scientist v2: arXiv:2504.08066
2. AI-Researcher: arXiv:2505.18705
3. Agent Laboratory: arXiv:2501.04227
4. data-to-paper: NEJM AI (2024)
5. STORM: arXiv:2402.14207
6. Paper2Code: arXiv:2504.17192
7. MatPlotAgent: arXiv:2402.11453
8. ScienceAgentBench: arXiv:2410.05080
9. "From Automation to Autonomy" Survey: EMNLP 2025
