# Architecture-Aware Prompt Hacking: A Deep Research Report

## 研究概述

本报告系统性调研了利用 LLM 架构特性（attention mechanism, transformer sparsity, MoE, sampling 等）进行 prompt 优化和 context 管理的最新研究（2023-2026）。基于 662 篇论文（415 篇同行评审）的搜索、7 篇核心论文的深度阅读、5 个代码仓库/工具生态系统的代码级分析，提出了一个架构感知的 prompt 设计框架。

**核心结论**：几乎所有可利用的 prompt hacking 模式都可追溯到两个架构选择——softmax 归一化和因果注意力掩码。这些不是 bug，而是 Transformer 架构的数学必然，可以被系统性地利用。

---

## 一、根因分析：Softmax + 因果注意力 = 所有可利用的注意力偏差

### 1.1 Softmax 创造零和注意力预算

每个 token 必须将恰好 1.0 的注意力分配给所有可见 token。当 query token 没有强语义匹配时，多余的注意力被"倾倒"到初始 token 上——这就是 attention sink。

**因果证据链**（Gu et al., ICLR 2025, sail-sg/Attention-Sink 代码库）：

| 操作 | Sink 指标 | 含义 |
|------|----------|------|
| 标准 Softmax | 45.11% | 基线 |
| Sigmoid + 归一化 | 18.18% | 减半，但归一化仍导致 sink |
| **Sigmoid 无归一化** | **2.46%** | 几乎消除 — 证明归一化是根因 |
| 添加可学习 Key 偏置 | 位置0: 18.18%, 偏置位置: **74.12%** | Sink 完全转移到偏置 — sink 的本质就是 key 偏置 |

**关键洞察**：Sink token 的 value 向量 L2 范数≈0。它们吸收注意力但不向输出贡献任何信息——是 softmax 的"压力释放阀"。

### 1.2 因果掩码创造指数级首因偏差

因果注意力可建模为有向图。定理 4.1（Wu et al., ICML 2025）证明：随层数增加，所有 token attend to 第一个 token 的概率**指数趋近于 1**。

从位置 $j$ 到位置 $i$，经过 $t$ 层的单调路径数为 $\binom{t+i-j}{i-j}$。早期位置拥有指数级更多的路径——这是**纯图结构性质**，与训练数据无关。

**与位置编码的竞争**（临界距离公式）：

$$x^* = \frac{t}{e^m - 1}$$

- 增大衰减 $m$ → 偏向近期 token（recency）
- 增加深度 $t$ → 偏向早期 token（primacy）
- **更深的模型中，因果掩码总是胜出** — 这解释了为什么大模型的位置偏差更强

### 1.3 二者共同产生的注意力预算模型

```
Position 0-3:   SINK ZONE (~50% 深层注意力)
                Key 范数小但余弦相似度高；Value 范数≈0
                ── StreamingLLM: start_size=4; 移除后 PPL 从 5.40 暴增至 5,158.07

Position 4-20:  PRIMACY ZONE (高注意力 + 高信息效用)
                受累积路径效应保护（定理 4.1）
                ── ACT 发现此区域也存在"有益 sink"

Position 20..N-L: MIDDLE ZONE (~10% 共享注意力，最易被压缩)
                SnapKV 的 top-k 选择一致丢弃此区域
                LLMLingua 的 PPL 评分中此区域最可预测
                ── Lost in Middle: 准确率从 75.8% 降至 53.8%（-22pp）

Position N-L..N: RECENCY ZONE (~40% 注意力)
                SnapKV 的 observation window (最后 32-64 token)
                驱动哪些 KV 条目在压缩后存活
```

---

## 二、十大架构感知 Prompt Hacking 技术

### 2.1 位置优化：利用 U 型注意力

**Lost in the Middle**（Liu et al., TACL 2024）：GPT-3.5-Turbo 在 20 篇文档设置下，开头 75.8% → 中间 53.8% → 结尾 63.2%。扩展 context window **不能解决此问题**。

**Hack**：将关键信息放在 sink 区之后（位置 5-20）和末尾。中间放可压缩的背景信息。

### 2.2 Attention Sink 保持：前 4 个 Token 的结构性角色

**StreamingLLM**（Xiao et al., ICLR 2024）：保留 4 个初始 token + 滑动窗口实现 400 万 token 稳定流式推理。

**代码级验证**（`mit-han-lab/streaming-llm`）：

```python
# 核心机制：StartRecentKVCache (~90 行代码)
# 保留 [sink_0..3, recent_N-512..N]，丢弃所有中间 token
# RoPE 位置重编码：key 存储时不带 RoPE，每次前向传播用连续位置重新应用
key_position_ids = torch.arange(kv_seq_len, device=device).unsqueeze(0)
key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
```

**"传送"效应**：缓存中的 token 被重编号为 [0,1,2,3,4,5,...,519]。第 5000 步的 recent token 被"传送"到紧邻 sink token 的位置。模型无法区分对话的第 2 轮和第 200 轮。

**MPT 例外**：使用 ALiBi 的模型不需要位置重编码——ALiBi 在注意力计算时从相对距离生成偏置，连续缓存位置自动产生正确的相对距离。

**生产部署**：`pip install attention-sinks`（tomaarsen/attention_sinks），支持 Llama/Mistral/Falcon/MPT/GPT-NeoX/Qwen/Yi，drop-in 替换。

### 2.3 Prompt Repetition：零延迟成本的双向注意力

**论文**（Leviathan et al., Google, 2025）：7 模型 × 7 基准 = 70 组测试，**47 胜 0 负 23 平**。

**最惊人案例**：Gemini 2.0 Flash-Lite, NameIndex 任务：21.33% → **97.33%**（+76pp）

**机制**：因果注意力下，`<Q><Q>` 中第二份的每个 token 可以 attend to 第一份中的所有 token。这等效于 encoder-decoder 架构的双向 encoder。

**关键细节**：
- **不增加延迟**：重复发生在 prefill 阶段（并行处理），不影响 decode 阶段（顺序处理，决定延迟）
- **推理模型无效**：CoT 天然重复用户请求，已隐式获得此效果（5 胜 1 负 22 平）
- **用句号填充同等长度无效**：确认是内容而非长度带来的提升
- **x3 优于 x2**：在 NameIndex/MiddleMatch 上进一步提升，未观察到明显收益递减

### 2.4 SinkTrack：将 Attention Sink 从被动现象变为主动机制

**论文**（Liu et al., ICLR 2026）：既然 LLM 天然持续关注 BOS token，将关键上下文注入 BOS 的表示，模型在整个生成中自动"锚定"于该信息。

**双轨交叉注意力**：
- Track 1（注入轨）：BOS 作为 Query，上下文 token 作为 Key/Value → 更新 BOS 表示
- Track 2（原始轨）：标准因果自注意力处理其余 token
- 递减调度：早期层注入更强，深层逐渐减弱

**结果**：QuAC +18.9%（Llama3.1-8B），M3CoT +23.0%（Qwen2.5-VL-7B）。训练无关、即插即用、零额外推理开销。

**研究演进**：发现（StreamingLLM, 2024）→ 理解（Sink Emergence, 2025）→ **利用**（SinkTrack, 2026）

### 2.5 Prompt Compression：两种截然不同的重要性定义

#### LLMLingua：困惑度 = 重要性

**核心代码**（`prompt_compressor.py`）：

```python
# get_ppl(): 用小型 LM 计算每个 token 的交叉熵损失
loss = CrossEntropyLoss(reduction="none")(active_logits, active_labels)
# 高损失 = 难预测 = 信息量大 = 保留
# 低损失 = 易预测 = 冗余 = 修剪
```

- **LLMLingua v1**：GPT-2/LLaMA-7B 计算 PPL，200 token 滑动窗口迭代压缩
- **LLMLingua-2**：XLM-RoBERTa 二分类器（GPT-4 蒸馏训练），比 v1 快 3-6x
- **结构化压缩**：`<llmlingua, compress=False>` 标签实现细粒度控制
- **SecurityLingua**：同一框架，训练为提取混淆/越狱 prompt 的真实意图

#### SnapKV：注意力 = 重要性

**核心代码**（`snapkv_utils.py`）：

```python
# 仅用最后 32-64 token 的 query 对所有 prefix key 投票
attn_weights = torch.matmul(
    query_states[..., -window_size:, :],  # 观察窗口
    key_states.transpose(2, 3)             # 所有 key
) / math.sqrt(head_dim)
# Softmax → 跨观察窗口求和 → per-head 独立 top-k 选择
# 1D avg_pool1d(kernel_size=5, stride=1) 平滑保留局部邻域
```

**两者的关系**：

| 维度 | LLMLingua (PPL) | SnapKV (注意力) |
|------|-----------------|----------------|
| 评分模型 | 外部小 LM | 目标 LLM 自身的注意力 |
| 查询依赖 | 否 (v1) / 部分 (LongLLMLingua) | 是 — 查询驱动观察窗口 |
| 压缩阶段 | 推理前（文本级） | 推理中（KV 级） |
| 位置偏差 | 最小 | 强（继承 U 型注意力） |
| 什么存活 | 罕见/意外的 token | 查询相关的 token |

**未被研究的交互**：一个 token 可以是"意外但不相关"（LLMLingua 保留，SnapKV 丢弃）或"可预测但查询相关"（LLMLingua 丢弃，SnapKV 保留）。最优管线应结合两者，但目前无此工具。

### 2.6 KV Cache Compression 生态：NVIDIA kvpress 的 30+ 方法

**NVIDIA kvpress** 是目前最完整的 KV 压缩框架：

```python
from kvpress import SnapKVPress, AdaKVPress, pipeline

# 一行代码集成
press = AdaKVPress(SnapKVPress(compression_ratio=0.5))  # 自适应 per-head SnapKV
pipe = pipeline("kv-press-text-generation", model=model, press=press)
answer = pipe(context, question=question)["answer"]
```

**关键方法分类**：

| 类别 | 代表方法 | 策略 |
|------|---------|------|
| Scorer-based | SnapKVPress, ExpectedAttentionPress, StreamingLLMPress, TOVAPress, KnormPress | 按重要性评分裁剪 KV |
| 层自适应 | SimLayerKVPress, DuoAttentionPress | 识别"懒惰层"选择性压缩 |
| 维度压缩 | ThinKPress | 通道注意力压缩 key 维度 |
| 组合式 | AdaKVPress, ComposedPress, PrefillDecodingPress | 包装任意方法，添加自适应性 |

**RoPE 校正**：`KeyRerotationPress` 提供 StreamingLLM 相同的位置重编码功能，作为可组合组件。

### 2.7 ACT：Per-Head Sink 画像与校准

**GATECH-EIC/ACT**（ICML 2024）发现 attention sink 是**逐头现象**——有些头的 sink 有益（提高准确率），有些有害。

**两阶段方法**：
1. **离线搜索**（`calibration.sh`）：识别哪些 head 需要校准
2. **在线应用**（`evaluation.sh`）：推理时自适应注意力分布优化

**结果**：Llama-30B 上准确率提升 **+7.3%**；MT-Bench 上缩小模型间差距 **33%**。无需修改权重。

**对 prompt hacking 的启示**：不同 head 关注 prompt 的不同位置。理论上可以设计 prompt 使关键信息被"好 head"而非"坏 head"捕获。

### 2.8 Min-p Sampling：替代 Top-P 的最佳策略

**ICLR 2025 Oral #18**（Nguyen et al.）：

```python
# 核心算法 (已内置于 vLLM, SGLang, llama.cpp)
p_threshold = min_p * max(probs)  # 动态阈值 = min_p × 最大概率
probs[probs < p_threshold] = 0    # 移除低于阈值的 token
```

**上下文自适应**：模型确信时（高 p_max），阈值严格，保持连贯；模型不确定时（低 p_max），阈值放松，允许多样性。

**实践推荐**：min_p = 0.05-0.1，替代 Top-K 和 Top-P。所有主流推理引擎原生支持。

### 2.9 XGrammar：<40μs 的受限解码

**核心突破**：将词表分为**上下文无关 token**（~99%，可预计算位掩码）和**上下文依赖 token**（~1%，运行时检查）。

```
运行时流程：
1. PDA 解析先前输出 → 获取当前栈状态
2. 用栈顶索引预计算位掩码 → 覆盖 99% 词表
3. 对 ~1% 上下文依赖 token → 运行 PDA 全栈检查
4. 合并掩码 → 应用到 logits → 采样
```

**性能**：每 token < 40μs，JSON schema 任务比替代方案快 3.5x，通用 CFG 快 10x+。vLLM 和 SGLang 的默认结构化生成后端。

### 2.10 位置偏差是架构层面不可避免的

**Wu et al., ICML 2025** 的图论框架提供了最深刻的统一理解：

- **纯因果掩码**只能捕获首因偏差（向序列开头偏移）
- 要学习"U 型"偏差，**必须依赖位置编码**（RoPE/ALiBi）
- 窗口注意力可以减缓但**不能消除**偏差
- 更深的模型 → 更强的首因偏差（与语义无关）

**攻防启示**：攻击者将恶意指令放在序列开头获得架构层面放大；防御者确保安全指令占据首位同样获得放大。

---

## 三、针对用户问题的直接回答

### Q1: 长 context 首尾注意力高、中间 recall 差？

**确认**。"Lost in the Middle"（TACL 2024）记录了 U 型曲线：开头 75.8% → 中间 53.8% → 结尾 63.2%。

**根因已完全理清**：
- **首因偏差** = 因果掩码的累积路径效应（定理 4.1，指数收敛）
- **近因偏差** = RoPE/ALiBi 的距离衰减
- **中间低谷** = 既不享受首因路径优势，也不被近因衰减偏爱
- **扩展 context window 不解决此问题** — GPT-3.5-Turbo 4K vs 16K 表现几乎重叠

**代码验证**：SnapKV 的观察窗口投票机制一致丢弃中间位置 token。KV 压缩将位置偏差从"注意力低"升级为"物理删除"。

### Q2: 重复两遍 prompt 提高准确率？

**确认**。47 胜 0 负 23 平（7 模型 × 7 基准）。

**机制解析**：因果注意力下，`<QUERY><QUERY>` 的第二份中每个 token 都能 attend to 第一份所有 token — 等效实现 prompt 内的双向注意力。这正是 encoder-decoder 架构中双向 encoder 的效果。

**关键限制**：
- 推理模型无效（CoT 已隐式提供此效果）
- 填充相同长度的句号无效（是内容而非长度）
- x3 在部分任务上优于 x2
- **零延迟成本**：重复在 prefill 阶段并行处理

### Q3: 锁住开头 token 后超出 context 质量降幅不明显？

**确认**。StreamingLLM 证明保留 4 个初始 token + 滑动窗口即可 400 万 token 稳定运行。

**代码级解释**：
1. `StartRecentKVCache` 只有 ~90 行代码
2. 默认 `start_size=4`（保留 BOS + 前 3 token）
3. **RoPE 位置重编码**：key 存储时**不带 RoPE**，每次前向传播用连续位置 [0,1,2,...] 重新应用
4. 效果：evict 后 cache = [sink_0-3, recent_N-512..N]，位置重编为 [0,1,2,3,4,...,519]
5. 模型认为 recent token 紧邻 sink token — "传送"效应

**为什么 content 不重要**：替换前 4 token 为 `\n\n\n\n` → PPL 5.60 vs 原始 5.40。完全移除 → PPL **5,158.07**。差 1000 倍。Sink token 是 softmax 的压力阀，不是信息载体。

### Q4: 还有哪些对绝大多数模型无差别的 hacking 方式？

除了你提到的三种外，基于本研究发现的 **8 种额外通用技术**：

| # | 技术 | 机制 | 适用性 | 实现难度 |
|---|------|------|--------|---------|
| 1 | **Min-p Sampling** | 动态阈值 = min_p × max(probs) | 所有模型家族 (ICLR 2025 Oral) | `SamplingParams(min_p=0.05)` |
| 2 | **SnapKV 观察窗口** | 指令放末尾驱动 KV 压缩选择 | 所有 decoder-only 模型 | `kvpress.SnapKVPress` |
| 3 | **SinkTrack 上下文锚定** | 注入关键信息到 BOS 表示 | 所有有 attention sink 的模型 | Training-free |
| 4 | **ACT 注意力校准** | Per-head 有益/有害 sink 识别 | 所有 Transformer | Offline calibration |
| 5 | **书签式信息放置** | Sink 区之后 + 序列末尾 | 所有因果注意力模型 | 免费 — 重排内容 |
| 6 | **Prefix caching** | 稳定内容前置，复用 KV | 所有支持 KV cache 的模型 | vLLM/SGLang 原生 |
| 7 | **Top-n-sigma** | Logit 空间高斯噪声消除 | 所有开源模型 | 3 行代码 |
| 8 | **XGrammar 受限解码** | PDA-based CFG logit masking | vLLM/SGLang | <40μs/token |

### Q5: 超出 context window 的管理机制？

最新研究表明最佳策略是**分层管理**，而非被动截断：

```
Layer 1: Active Context (context window 内)
├── Sink Zone (Token 0-3):
│   保留为注意力锚定。可通过 SinkTrack 注入关键上下文。
├── Primacy Zone (Token 4-20):
│   放置最关键的指令。享受累积路径效应放大。
├── Compressed Context:
│   LLMLingua-2 压缩后的关键信息（去除冗余 token）。
│   聚集相关信息（SnapKV 的 avg_pool1d 保留局部邻域）。
└── Query/Instruction (末尾):
    驱动 SnapKV 的观察窗口。放置最终指令/查询。

Layer 2: Retrievable Store (context window 外)
├── 被 SnapKV 裁剪的 KV 条目 → 语义索引，按需检索
├── 被 LLMLingua 修剪的文本 → 完整版存储，按需恢复
├── 历史对话摘要 → 带时间戳的语义检索
└── 代码库索引 → AST/依赖图组织的按需检索

Layer 3: Persistent Memory
├── 长期知识 → 向量数据库 (embedding retrieval)
└── 结构化数据 → SQL/图数据库
```

**实现工具链**：
1. 文本级压缩：`pip install llmlingua` → `PromptCompressor`
2. KV 级压缩：`pip install kvpress` → `SnapKVPress / StreamingLLMPress`
3. 推理引擎：vLLM/SGLang（prefix caching + min-p + XGrammar 原生）
4. 流式对话：`pip install attention-sinks`（drop-in StreamingLLM）

### Q6: 代码库应该怎么处理？

你的直觉完全正确——**不应该把大片代码塞到 context window**。基于论文和代码分析：

**1. 分类存储**
- 按模块/功能/依赖关系组织（Tree-sitter AST 解析）
- 每个代码块附带摘要（函数签名 + docstring + 调用关系）

**2. 按需检索**
- 只拉取当前任务相关的代码片段
- 关键函数放在 prompt 末尾（驱动 SnapKV 观察窗口）
- 辅助函数/类型定义放在开头（享受 primacy zone）

**3. 压缩展示**
- 对于需要全局了解的代码，用 AST 摘要替代完整代码
- LLMLingua 的 `structured_compress_prompt()` 可对代码注释和空白压缩而保留关键 token

**4. 位置优化**
- 最相关的代码放在首尾（bookend pattern）
- 参考代码放中间（最易被 SnapKV 裁剪，但有冗余保护）

**5. 冗余标记**
- 关键函数签名在 context 开头（类型摘要）和代码块处重复出现
- 增加被 SnapKV per-head 投票保留的概率

**6. 增量管理**
- 随着对话推进，压缩旧代码上下文，为新代码腾出空间
- 已讨论的代码存入 Layer 2（可检索存储）
- 当前焦点代码保持在 Active Context

**目前的研究空白**：没有学术工作专门研究代码的注意力模式（代码有跨位置的结构性依赖——函数调用可能引用千 token 之外的定义）。代码 token 在 SnapKV 压缩下是否有特殊行为？代码结构 token（大括号、分号）是否充当 attention sink？这些都是开放问题。

---

## 四、实用工具生态系统

### 生产部署流水线

```
用户输入
  │
  ▼
[LLMLingua-2] ─── 文本级压缩（可选）
  │                pip install llmlingua
  │                PromptCompressor(use_llmlingua2=True)
  ▼
[vLLM / SGLang] ── 推理引擎
  │                Min-p sampling（原生支持）
  │                XGrammar 受限解码（默认后端）
  │                Prefix caching（原生支持）
  ▼
[kvpress] ──────── KV Cache 压缩
  │                pip install kvpress
  │                SnapKVPress / ExpectedAttentionPress
  │                AdaKVPress 自适应 per-head 预算
  │                DecodingPress 生成阶段压缩
  ▼
模型输出
```

### 研究/分析工具箱

| 工具 | 用途 | 安装/使用 |
|------|------|-----------|
| BertViz / Inspectus | 注意力可视化 | `pip install inspectus` |
| sail-sg/Attention-Sink | Sink 指标计算（30+ 模型） | GitHub clone |
| GATECH-EIC/ACT | Per-head sink 画像与校准 | `bash scripts/calibration.sh` |
| nanoGCG / BrokenHill | 梯度对抗后缀生成 | GitHub clone |
| DSPy MIPRO | 贝叶斯 prompt 优化 | `pip install dspy` |
| Needle-in-a-Haystack | 位置偏差评估 | `gkamradt/LLMTest_NeedleInAHaystack` |
| RobustKV | KV Cache 越狱防御 | `TanqiuJiang/RobustKV` |

### 关键参数推荐

| 参数 | 推荐值 | 适用场景 |
|------|--------|---------|
| StreamingLLM `start_size` | 4 | 所有流式场景 |
| SnapKV `window_size` | 32-64 | KV 压缩 |
| SnapKV `kernel_size` | 5 | 局部邻域保留 |
| SnapKV `compression_ratio` | 0.5 (起步) | 按需调整 |
| LLMLingua `rate` | 0.3-0.5 | 文本压缩 |
| Min-p | 0.05-0.1 | 所有开源模型 |
| Top-n-sigma `n` | 0.5-1.0 | 替代 temperature 调优 |
| XGrammar | JSON schema | 结构化输出 |

---

## 五、2025 下半年至 2026 年最新进展

> 补充搜索覆盖 2025 年 7 月 – 2026 年 2 月的最新研究。核心发现：架构层面的根本性变革正在发生。

### 5.1 最新模型的位置偏差表现

| 模型 | Context | 位置偏差状态 | 关键技术 |
|------|---------|------------|---------|
| **GPT-5.2** (OpenAI, 2025.12) | 400K | 4-needle MRCR **98%**；8-needle **70%** | 未公开架构细节 |
| **Llama 4 Scout** (Meta) | **10M** | 大幅缓解（iRoPE 重新设计位置编码） | iRoPE: 3:1 RoPE+chunked : NoPE 层 |
| **Qwen3-Next** (Alibaba) | 256K-1M | **消除 attention sink** | Gated Attention (NeurIPS 2025 Best Paper) |
| **Kimi Linear** (Moonshot) | 1M | KV cache -75%, 6x 解码加速 | 3:1 线性注意力:MLA 混合 |
| **Gemini 3 Pro** (Google) | 1M | 缺少专项评测 | 多模态 |
| **Claude Opus 4.6** (Anthropic) | 200K | 缺少公开位置偏差评测 | 未公开 |

**关键判断**：GPT-5.2 在 4-needle 测试达到 98% 说明 Lost-in-the-Middle **被大幅缓解**。但 8-needle 降到 70% 说明**没有消除**——当需要同时追踪更多跨位置信息时，中间位置仍然吃亏。

### 5.2 Gated Attention：消除 Attention Sink 的架构突破

**NeurIPS 2025 最佳论文** — Qiu et al.

在 scaled dot-product attention 之后加入 query-dependent、head-specific 的 **sigmoid 门控**：

- 系统比较了 **30+ 门控变体**（15B MoE + 1.7B dense，3.5T tokens）
- **消除 attention sink 和 massive activation** — 我们综述第一章识别的根因被架构层面解决
- 仅训练门控机制即可实现长上下文外推
- **已集成到 Qwen3-Next 官方架构**

**对我们框架的影响**：
- Sink 利用技术（StreamingLLM, SinkTrack, ACT）在 Gated Attention 模型上**不再必要**
- 但因果掩码导致的累积路径偏差（定理 4.1）**不受门控影响**——这是图结构性质
- **Softmax 模型仍然是绝大多数部署模型**（GPT-5.2, Claude, Gemini）—— sink 技术仍然高度相关

### 5.3 iRoPE：Llama 4 的位置编码革命

```
每 4 层的布局：
  层 1-3: RoPE + 分块注意力（8192 token 块）
  层 4:   NoPE（无位置编码）+ 全因果掩码 + scaled softmax
```

- NoPE 层可以无距离衰减地访问**全部上下文**
- Scaled softmax 实现从 256K 训练到 **10M 推理**的泛化
- 分块注意力限制了定理 4.1 的累积路径效应在 8192 token 块内

**影响**：RoPE 层的 recency bias 被限制在块内；NoPE 层提供全局无偏差访问。U 型曲线应被大幅平坦化，但尚无独立的 Lost-in-the-Middle 复现评测。

### 5.4 混合线性/Softmax 注意力

**Kimi Linear** (Moonshot, 2025.10):
- 3 层 KDA（线性注意力）: 1 层 MLA（softmax）
- 线性层无 softmax → **无 attention sink**
- 48B/3B active, 1M context, AIME 2025 + LiveCodeBench 超越同规模全注意力模型

**趋势**：线性注意力层比例上升 → softmax 相关的 sink/偏差问题自然减少 → KV cache 需求大幅降低。

### 5.5 位置偏差的最新理论

**2026 年 2 月预印本**（arXiv:2602.16837）修正了我们框架的一个关键预测：

- **问题**：定理 4.1 预测 depth→∞ 时注意力完全坍缩到 token 0。但实际模型不会。
- **修正**：**残差连接**防止了完全坍缩。有限深度下，因果掩码 + 残差连接产生 **U 形偏差**（而非坍缩）。
- 即使打乱词序，位置偏差仍然存在——确认纯架构属性。

### 5.6 KV Cache 压缩最新进展（ICLR 2026）

| 方法 | 创新点 | 效果 | 来源 |
|------|--------|------|------|
| **GVote** | Query 采样+投票自动计算最优 cache 预算 | 消除人工预算调优 | ICLR 2026 |
| **ThinKV** | 将 CoT 分解为推理/执行/过渡段，分段压缩 | 推理链保真度更高 | ICLR 2026 |
| **xKV** | 跨层 SVD 压缩（对齐奇异向量） | 后训练跨层压缩 | ICLR 2026 |
| **HEX** | Expander 图骨架 + heavy-hitters | 频谱连通性保证 | ICLR 2026 |
| **CompressKV** | 语义检索头识别 | **0.7% KV → 90% NIAH** | Under review |
| **ChunkKV** | 语义 chunk 级（非 token 级）压缩 | +8.7% 精度, +26.5% 吞吐 | Under review |

**CompressKV 的 0.7% 结果**意味着：在 128K context 中，只保留 ~900 个 KV 条目即可在 Needle-in-a-Haystack 达到 90%。这验证了我们的"注意力预算极度集中"分析。

### 5.7 Ms-PoE：训练无关的位置偏差缓解

**NeurIPS 2024** — 为不同注意力头分配不同的位置缩放比率：
- 即插即用，无需训练
- Zero-SCROLLS: 平均 +3.8，**中间位置准确率 +20-40%**
- `github.com/VITA-Group/Ms-PoE`

### 5.8 更新后的判断：什么仍然有效，什么正在过时

| 本报告的发现 | 2026 年状态 | 说明 |
|------------|-----------|------|
| Softmax 创造 attention sink | **GPT/Claude/Gemini: 仍然有效** | Qwen3-Next 已消除 |
| U 型 recall (Lost in Middle) | **大幅缓解但未消除** | GPT-5.2: 4-needle 98%, 8-needle 70% |
| 位置偏差是数学必然 | **修正**: 有限深度 U 形，残差连接防止坍缩 | 2026 年新理论 |
| Prompt repetition | **仍然有效** | 已测试 GPT-4o, Claude 3.7, Gemini 2.0, DeepSeek V3 |
| SnapKV 观察窗口 | **仍然有效，持续进化** | GVote 自动预算, ThinKV CoT 感知 |
| StreamingLLM sink 保持 | **对 softmax 模型有效；将逐步过时** | Gated attention 模型不需要 |
| LLMLingua 压缩 | **仍然有效** | 与注意力架构正交 |
| Min-p sampling | **仍然有效** | 架构无关 |
| KV 压缩仍然关键 | **部分被混合线性注意力替代** | Kimi Linear 原生 -75% KV |

**底线**：本报告的 Softmax-Causal Nexus 框架对 GPT-5.2、Claude Opus 4.6、Gemini 等主流部署模型**完全成立**。但 Qwen3-Next 和 Kimi Linear 代表了一个新的架构分支，在该分支上 sink 利用技术将逐步失效。未来 2-3 年是过渡期。

---

## 六、研究空白与未来方向

### 高优先级（有明确工具/数据可立即研究）

1. **架构感知 Prompt 优化器**：DSPy MIPRO 不考虑模型架构（RoPE vs ALiBi vs iRoPE、head 数、深度、是否 gated attention）。结合 ACT 的 per-head 画像 + kvpress 的压缩模拟 + DSPy 的优化框架 = 可立即构建。

2. **PPL × 注意力重要性交互**：LLMLingua（PPL 评分）和 SnapKV（注意力评分）测量不同维度。二者如何交互？最优管线配置？两个 pip 包都已就绪。

3. **代码特定的 Context 管理**：无学术工作研究代码在 LLM 中的注意力模式。Tree-sitter AST + LLMLingua + kvpress 都已可用。

### 中优先级

4. **Gated Attention 下的 Prompt 设计范式**：Sink 消失后，最优 prompt 结构是什么？位置偏差如何变化？需要新的实证研究。

5. **iRoPE 模型的位置偏差评测**：Llama 4 的 Lost-in-the-Middle 复现评测尚未发表。NoPE 层是否真的消除了 U 型曲线？

6. **多轮对话的 Context 管理**：StreamingLLM 只测试单程流。在 gated attention / iRoPE 模型上，多轮对话是否有不同的最优策略？

7. **MoE 感知 Prompt 设计**：DeepSeek-V3（256 专家）中 prompt 措辞如何影响专家路由？无系统研究。

### 长期方向

8. **动态 Context Orchestration 框架**：统一文本压缩 + KV 压缩 + 外部检索 + sink/gating 感知的实时管理框架。

9. **KV Cache 安全威胁模型**：多租户 KV 共享（vLLM prefix caching）创造侧信道。

10. **Prompt Repetition 在最新模型上的验证**：已测 GPT-4o/Claude 3.7/Gemini 2.0——需要在 GPT-5.2、Qwen3、Llama 4 上重新评估。

---

## 七、核心论文参考

### 必读论文（7 篇深度阅读 + 3 篇 2025H2-2026 关键论文）

1. **Lost in the Middle** — Liu et al., TACL 2024 [@liu2024lost]
2. **StreamingLLM (Attention Sinks)** — Xiao et al., ICLR 2024 [@xiao2024efficient]
3. **When Attention Sink Emerges** — Gu et al., ICLR 2025 Spotlight [@gu2025attention]
4. **On the Emergence of Position Bias** — Wu et al., ICML 2025 [@wu2025emergence]
5. **Prompt Repetition** — Leviathan et al., Google, 2025 [@leviathan2025prompt]
6. **SnapKV** — Li et al., NeurIPS 2024 [@li2024snapkv]
7. **SinkTrack** — Liu et al., ICLR 2026 [@liu2026sinktrack]
8. **Gated Attention** — Qiu et al., NeurIPS 2025 **Best Paper** [@qiu2025gated]
9. **Sigmoid Self-Attention Theory** — Ramapuram et al., ICLR 2025 [@ramapuram2025sigmoid]
10. **Residual-Aware Position Bias** — 2026 preprint [@residualaware2026]

### 代码级研究的仓库（5 个深度分析）

| 仓库 | Stars | 分析文件 |
|------|-------|---------|
| mit-han-lab/streaming-llm | 7.2k | `phase4_code/streaming_llm_deep_dive.md` |
| microsoft/LLMLingua | 5.8k | `phase4_code/llmlingua_analysis.md` |
| FasterDecoding/SnapKV + NVIDIA/kvpress | - | `phase4_code/snapkv_deep_dive.md` |
| sail-sg/Attention-Sink + GATECH-EIC/ACT + kvpress | - | `phase4_code/code_repos.md` |
| vLLM/SGLang min-p + DSPy + XGrammar | - | `phase4_code/tools_deep_dive.md` |

### 重要补充论文

11. **Attention Sinks and Compression Valleys** — Queipo-de-Llano et al., ICLR 2026 [@queipodellano2026attention]
12. **LLMLingua-2** — Jiang et al., ACL 2024 [@jiang2024llmlingua2]
13. **Found in the Middle** — Hsieh et al., ACL 2024 [@hsieh2024found]
14. **Eliminating Position Bias (PINE)** — Wang et al., ICLR 2025 [@wang2025eliminating]
15. **Min-p Sampling** — Nguyen et al., ICLR 2025 Oral [@nguyen2025minp]
16. **Top-n-sigma** — Tang et al., ACL 2025 [@tang2025topnsigma]
17. **Hidden Attention Sinks (ACT)** — Yu et al., ICML 2024 [@yu2024hidden]
18. **Catch-Tag-Release** — Zhang et al., NeurIPS 2025 [@zhang2025catchtagrelease]
19. **Ms-PoE** — Chi et al., NeurIPS 2024 [@chi2024mspoe]
20. **Kimi Linear** — Moonshot AI, 2025 [@moonshot2025kimi]

---

*报告生成日期: 2026-02-26*
*论文数据库: 662 篇论文 + 2025H2-2026 补充搜索*
*深度阅读: 7 篇核心论文 + 3 篇最新关键论文*
*代码级分析: 5 个工具生态系统*
