# Hello Agents 第九章：上下文工程（详细版）

> **本章核心思想**：让 Agent 在有限的"注意力预算"中，精准选择并组织信息，从"信息过载"变成"精准决策"。

---

## 📖 目录

- [1. 什么是上下文工程？](#1-什么是上下文工程)
- [2. 为什么需要上下文工程？](#2-为什么需要上下文工程)
- [3. 上下文工程的四大策略](#3-上下文工程的四大策略)
- [4. HelloAgents 中的实践：ContextBuilder](#4-helloagents-中的实践contextbuilder)
- [5. 实战工具：NoteTool 和 TerminalTool](#5-实战工具notetool-和-terminaltool)
- [6. 本章总结](#6-本章总结)

---

## 1. 什么是上下文工程？

### 🤔 从提示工程到上下文工程

想象你是一位学生准备考试，你有两种学习方式：

**方式A：提示工程**（传统方式）
```
把所有教科书、笔记、习题册全部摊在桌子上
希望考试时能想起所有内容
结果：信息太多，脑子一团糟 😵
```

**方式B：上下文工程**（新方式）
```
考试前：
1️⃣ 筛选重点章节（Select）
2️⃣ 整理关键知识点（Structure）
3️⃣ 压缩成复习提纲（Compress）
4️⃣ 考试时只看提纲（Context Window）
结果：思路清晰，答题准确 ✅
```

### 💡 核心定义

**上下文工程**是一门系统化的学科，研究如何在每次模型调用前，以**可复用、可度量、可演进**的方式，拼装并优化输入上下文，从而提升：

- ✅ **正确性**：提供准确、相关的信息
- ✅ **鲁棒性**：避免信息污染和冲突
- ✅ **效率**：减少 Token 消耗和延迟

> 💭 **类比**：就像操作系统管理 CPU 的 RAM 一样，上下文工程管理 LLM 的"工作记忆"

### 📊 对比表格

| 维度 | 提示工程 | 上下文工程 |
|------|---------|-----------|
| **关注点** | 如何写好提示词 | 如何构建最优上下文 |
| **适用场景** | 单轮对话、简单任务 | 多轮交互、复杂任务 |
| **核心问题** | "问什么问题？" | "提供什么信息？" |
| **包含内容** | 系统提示 + 用户输入 | 提示 + 历史 + 工具 + 知识 + 记忆 |
| **优化目标** | 更好的指令 | 更优的上下文窗口 |

### 🎯 上下文的组成

一个完整的上下文包含三大类信息：

```
┌──────────────────────────────────────┐
│         LLM 上下文窗口                │
├──────────────────────────────────────┤
│ 📜 Instructions（指令上下文）         │
│  ├─ 系统提示词                        │
│  ├─ 用户指令                          │
│  ├─ 少样本示例                        │
│  └─ 输出格式要求                      │
├──────────────────────────────────────┤
│ 📚 Knowledge（知识上下文）            │
│  ├─ RAG 检索结果                      │
│  ├─ 记忆系统数据                      │
│  ├─ 领域知识库                        │
│  └─ 实时数据                          │
├──────────────────────────────────────┤
│ 🔧 Tools（工具上下文）                │
│  ├─ 工具描述                          │
│  ├─ 工具调用结果                      │
│  ├─ 执行历史                          │
│  └─ 错误反馈                          │
└──────────────────────────────────────┘
```

---

## 2. 为什么需要上下文工程？

### 😱 问题一：上下文腐蚀（Context Rot）

**现象**：上下文越长，模型表现越差

想象你在图书馆找一本书：

```
短上下文（10本书）：
  📚 一眼就能找到目标 → 准确率 95%

长上下文（10,000本书）：
  📚📚📚...（太多了）
  → 找不到目标书 → 准确率 60% ❌
```

**实验数据**：

| 上下文长度 | 准确率 | 模型表现 |
|-----------|--------|---------|
| 1K tokens | 92% | 优秀 ✅ |
| 10K tokens | 85% | 良好 👍 |
| 50K tokens | 72% | 一般 😐 |
| 100K tokens | 58% | 较差 ❌ |

> 💡 **原因**：Transformer 的注意力机制是 O(n²)，上下文越长，每个 token 分配到的"注意力"越少

### 😵 问题二：上下文四大挑战

#### 1️⃣ 上下文污染（Context Poisoning）

**场景**：错误信息被反复引用

```python
# 第1轮：模型幻觉
Agent: "北京的人口是 5000 万"  # ❌ 实际是 2170 万

# 第2轮：错误被固化
User: "那上海呢？"
Agent: "上海 2500 万，比北京的 5000 万少很多"  # ❌ 错上加错

# 结果：整个对话基于错误前提
```

**解决方案**：验证工具输出，及时纠错

#### 2️⃣ 上下文分散（Context Distraction）

**场景**：信息过多导致"失焦"

```
任务：写一篇关于 Python 的博客

上下文：
- Python 基础语法（10K tokens）
- Python 高级特性（15K tokens）
- Python 历史（5K tokens）
- Python 社区（8K tokens）
- ... 还有 20 篇文档

结果：Agent 不知道从哪开始，反复重复相同内容 ❌
```

**解决方案**：只提供任务相关的精简信息

#### 3️⃣ 上下文混淆（Context Confusion）

**场景**：无关信息干扰决策

```python
# 任务：计算 2 + 3
# 上下文包含：
tools = [
    "calculator",        # ✅ 相关
    "search",           # ❌ 无关
    "weather",          # ❌ 无关
    "translate",        # ❌ 无关
    # ... 还有 50 个工具
]

# 结果：Agent 可能选错工具，或者陷入"选择困难症"
```

**解决方案**：动态过滤无关工具和信息

#### 4️⃣ 上下文冲突（Context Clash）

**场景**：信息相互矛盾

```
知识库A：巴黎是法国首都
知识库B：巴黎是德国首都（错误数据）

Agent 收到冲突信息 → 产生困惑 → 输出不一致 ❌
```

**解决方案**：数据源优先级 + 一致性检查

### 💰 问题三：成本与延迟

**Token 成本对比**：

| 上下文策略 | Tokens | 成本（GPT-4） | 延迟 |
|-----------|--------|--------------|------|
| ❌ 无优化 | 100K | $1.00 | 5s |
| ✅ 压缩 50% | 50K | $0.50 | 3s |
| ✅ 精选 20% | 20K | $0.20 | 1s |

> 💡 **结论**：有效的上下文工程能节省 **80% 成本**和 **80% 延迟**

---

## 3. 上下文工程的四大策略

上下文工程的核心是四个动作：**写入（Write）、选择（Select）、压缩（Compress）、隔离（Isolate）**

### 📝 策略一：写入上下文（Write）

**目标**：将信息持久化到上下文窗口之外

#### 1️⃣ 临时笔记板（Scratchpad）

**场景**：记录推理过程

```python
# Agent 的思考过程
scratchpad = """
Step 1: 用户想去北京旅游
Step 2: 需要查询天气 → 调用 weather_api
Step 3: 北京明天有雨 → 建议带伞
Step 4: 查询景点 → 故宫周一闭馆
Step 5: 调整行程 → 推荐周二参观
"""

# 后续推理时，读取 scratchpad 保持连贯性
```

**优点**：
- ✅ 思考过程可见、可追溯
- ✅ 多步推理不会"忘记"中间结果

#### 2️⃣ 记忆系统（Memory）

**场景**：跨会话的持久化记忆

```python
# 短期记忆（Short-term）
conversation_history = [
    "User: 我叫小明",
    "Agent: 你好小明！",
    "User: 我喜欢爬山",
]

# 长期记忆（Long-term）
user_profile = {
    "name": "小明",
    "interests": ["爬山", "摄影"],
    "last_trip": "北京",
    "preferences": "喜欢历史文化"
}

# 下次对话时自动加载记忆
Agent: "小明，你上次去了北京，这次要不要试试西安？那里也有很多历史遗迹。"
```

### 🎯 策略二：选择上下文（Select）

**目标**：在海量信息中挑选最相关的部分

#### 选择算法：相关性 + 新近性

```python
# 综合评分公式
score = relevance_weight × relevance_score + recency_weight × recency_score

# 示例配置
relevance_weight = 0.7  # 相关性权重 70%
recency_weight = 0.3    # 新近性权重 30%
```

**实战案例**：

```python
# 用户查询："今天北京天气怎么样？"

候选信息：
1. "北京今天晴天，25°C"          → 相关性 0.9, 新鲜度 1.0 → 得分 0.93 ✅
2. "上海今天多云，20°C"          → 相关性 0.3, 新鲜度 1.0 → 得分 0.51
3. "北京去年天气统计"            → 相关性 0.6, 新鲜度 0.1 → 得分 0.45
4. "天气API使用文档"             → 相关性 0.2, 新鲜度 0.5 → 得分 0.29

# 最终选择：信息1（得分最高）
```

#### 过滤策略

```python
# 设置最低相关性阈值
min_relevance = 0.1

# 过滤低质量信息
selected = [p for p in packets if p.relevance_score >= min_relevance]
```

### 🗜️ 策略三：压缩上下文（Compress）

**目标**：在不丢失关键信息的前提下，减少 Token 数量

#### 1️⃣ 对话摘要（Conversation Summary）

**原始对话**（500 tokens）：
```
User: 你好
Agent: 你好！有什么可以帮你的？
User: 我想去北京旅游
Agent: 好的，你打算什么时候去？
User: 下周
Agent: 下周北京天气不错，你想去哪些景点？
User: 故宫和长城
Agent: 很好的选择！我来帮你规划一下...
```

**压缩后**（100 tokens）：
```
用户计划下周去北京旅游，目标景点：故宫、长城。天气预报良好。
```

**压缩率**：80%（节省 400 tokens）

#### 2️⃣ 工具输出摘要

**原始工具输出**（2000 tokens）：
```json
{
  "status": 200,
  "timestamp": "2025-01-29T10:30:00Z",
  "request_id": "abc123...",
  "debug_info": {...},
  "metadata": {...},
  "result": {
    "city": "北京",
    "temperature": 25,
    "condition": "晴天",
    "humidity": 45,
    "wind_speed": 10,
    "air_quality": 85,
    "forecast_7days": [...]
  }
}
```

**摘要后**（50 tokens）：
```
北京今天晴天，25°C，空气质量良好。
```

**压缩率**：97.5%（节省 1950 tokens）

#### 3️⃣ 上下文修剪（Context Pruning）

**基于规则的修剪**：

```python
# 规则1：删除旧对话（保留最近10轮）
conversation_history = conversation_history[-10:]

# 规则2：删除已完成的任务记录
active_tasks = [t for t in tasks if t.status != "completed"]

# 规则3：删除过期数据（超过24小时）
from datetime import datetime, timedelta
cutoff_time = datetime.now() - timedelta(hours=24)
recent_data = [d for d in data if d.timestamp > cutoff_time]
```

**智能修剪**：

```python
# 基于相关性动态修剪
def smart_prune(packets, user_query, max_tokens):
    # 1. 计算每个 packet 的相关性分数
    scored = [(calculate_relevance(p, user_query), p) for p in packets]

    # 2. 按分数排序
    scored.sort(reverse=True)

    # 3. 贪心选择，直到达到 token 上限
    selected = []
    total_tokens = 0
    for score, packet in scored:
        if total_tokens + packet.token_count <= max_tokens:
            selected.append(packet)
            total_tokens += packet.token_count

    return selected
```

### 🔒 策略四：隔离上下文（Isolate）

**目标**：通过分离关注点，避免信息污染

#### 1️⃣ 多 Agent 架构

**场景**：复杂研究任务

```
┌──────────────────────────────────────┐
│         主 Agent（总指挥）             │
│  任务：撰写 AI 行业报告                │
│  上下文：5000 tokens（轻量）           │
└────────┬─────────────────────────────┘
         │
         ├─► 子Agent 1（技术研究）
         │   检索论文 → 分析趋势
         │   上下文：20000 tokens
         │   输出摘要：1000 tokens ✅
         │
         ├─► 子Agent 2（市场调研）
         │   分析报告 → 统计数据
         │   上下文：15000 tokens
         │   输出摘要：800 tokens ✅
         │
         └─► 子Agent 3（案例分析）
             收集案例 → 总结经验
             上下文：18000 tokens
             输出摘要：1200 tokens ✅

主 Agent 接收到的总上下文：
  = 5000（自身）+ 1000 + 800 + 1200
  = 8000 tokens（远小于 53000 tokens）
```

**优势**：
- ✅ 关注点分离（每个 Agent 专注一个领域）
- ✅ 并行处理（多个子 Agent 同时工作）
- ✅ 上下文隔离（主 Agent 不被细节淹没）

#### 2️⃣ 执行环境隔离

**场景**：代码执行与 LLM 分离

```
┌─────────────────────────────────────┐
│  LLM（决策层）                       │
│  上下文：轻量化指令 + 工具描述       │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  处理层（过滤 + 摘要）               │
│  - 过滤调试信息                      │
│  - 提取关键结果                      │
│  - 格式化输出                        │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  执行环境（沙箱）                    │
│  - 工具实际运行                      │
│  - 产生原始输出（可能很大）          │
└─────────────────────────────────────┘
```

**示例**：

```python
# ❌ 不好的做法：直接把原始输出给 LLM
tool_output = execute_code(script)  # 10000 tokens 的日志
llm.invoke(tool_output)  # LLM 被大量日志淹没

# ✅ 好的做法：先过滤再传递
raw_output = execute_code(script)  # 10000 tokens
summary = filter_and_summarize(raw_output)  # 200 tokens
llm.invoke(summary)  # LLM 只看关键信息
```

### 📊 四大策略总结

| 策略 | 解决的问题 | 核心方法 | 适用场景 |
|-----|-----------|---------|---------|
| **写入（Write）** | 信息丢失 | 临时笔记板 + 记忆系统 | 长时程任务、多步推理 |
| **选择（Select）** | 信息冗余 | 相关性评分 + 过滤 | 海量数据、RAG 检索 |
| **压缩（Compress）** | 信息过载 | 摘要 + 修剪 | 超长对话、大型工具输出 |
| **隔离（Isolate）** | 信息污染/冲突 | 多 Agent + 环境隔离 | 复杂系统、并行任务 |

---

## 4. HelloAgents 中的实践：ContextBuilder

### 🎯 ContextBuilder 简介

**ContextBuilder** 是 HelloAgents 框架中实现上下文工程的核心组件，它将复杂的上下文管理抽象为一个简洁的 **GSSC 流水线**。

```
GSSC 流水线
├─ G (Gather)    → 汇集多源信息
├─ S (Select)    → 智能选择相关信息
├─ S (Structure) → 结构化组织
└─ C (Compress)  → 压缩超限内容
```

### 📦 核心数据结构

#### 1️⃣ ContextPacket：信息包

```python
@dataclass
class ContextPacket:
    """候选信息包"""
    content: str              # 内容
    timestamp: datetime       # 时间戳
    token_count: int          # Token 数量
    relevance_score: float    # 相关性分数（0.0-1.0）
    metadata: Dict[str, Any]  # 元数据
```

**示例**：

```python
# 创建一个信息包
packet = ContextPacket(
    content="北京今天晴天，25°C",
    timestamp=datetime.now(),
    token_count=10,
    relevance_score=0.9,
    metadata={"type": "weather", "source": "api"}
)
```

#### 2️⃣ ContextConfig：配置管理

```python
@dataclass
class ContextConfig:
    """上下文配置"""
    max_tokens: int = 3000        # 最大 token 数
    reserve_ratio: float = 0.2    # 系统指令预留比例
    min_relevance: float = 0.1    # 最低相关性阈值
    enable_compression: bool = True  # 启用压缩
    recency_weight: float = 0.3   # 新近性权重
    relevance_weight: float = 0.7  # 相关性权重
```

### 🔄 GSSC 流水线详解

#### 阶段一：Gather（汇集）

**目标**：从多个数据源汇集候选信息

```python
def _gather(self, user_query, conversation_history, system_instructions):
    packets = []

    # 1. 添加系统指令（最高优先级）
    if system_instructions:
        packets.append(ContextPacket(
            content=system_instructions,
            relevance_score=1.0,  # 始终保留
            metadata={"type": "system_instruction"}
        ))

    # 2. 从记忆系统检索
    if self.memory_tool:
        memory_results = self.memory_tool.search(user_query)
        packets.extend(self._parse_memory_results(memory_results))

    # 3. 从 RAG 系统检索
    if self.rag_tool:
        rag_results = self.rag_tool.search(user_query)
        packets.extend(self._parse_rag_results(rag_results))

    # 4. 添加对话历史（最近5轮）
    if conversation_history:
        recent_history = conversation_history[-5:]
        packets.extend(self._convert_to_packets(recent_history))

    return packets
```

**容错机制**：

```python
# 每个数据源都有异常处理
try:
    memory_results = self.memory_tool.search(user_query)
except Exception as e:
    print(f"[WARNING] 记忆检索失败: {e}")
    # 继续处理其他数据源，不影响整体流程
```

#### 阶段二：Select（选择）

**目标**：根据评分选择最有价值的信息

```python
def _select(self, packets, user_query, available_tokens):
    # 1. 分离系统指令和其他信息
    system_packets = [p for p in packets if p.metadata.get("type") == "system_instruction"]
    other_packets = [p for p in packets if p.metadata.get("type") != "system_instruction"]

    # 2. 计算系统指令占用的 token
    system_tokens = sum(p.token_count for p in system_packets)
    remaining_tokens = available_tokens - system_tokens

    # 3. 为其他信息计算综合分数
    scored_packets = []
    for packet in other_packets:
        # 相关性分数
        relevance = self._calculate_relevance(packet.content, user_query)

        # 新近性分数
        recency = self._calculate_recency(packet.timestamp)

        # 综合分数
        combined_score = (
            self.config.relevance_weight * relevance +
            self.config.recency_weight * recency
        )

        # 过滤低分信息
        if relevance >= self.config.min_relevance:
            scored_packets.append((combined_score, packet))

    # 4. 按分数排序
    scored_packets.sort(key=lambda x: x[0], reverse=True)

    # 5. 贪心选择（填满 token 预算）
    selected = system_packets.copy()
    current_tokens = system_tokens

    for score, packet in scored_packets:
        if current_tokens + packet.token_count <= available_tokens:
            selected.append(packet)
            current_tokens += packet.token_count
        else:
            break  # Token 预算已满

    return selected
```

**评分算法**：

```python
# 相关性：Jaccard 相似度
def _calculate_relevance(self, content, query):
    content_words = set(content.lower().split())
    query_words = set(query.lower().split())

    intersection = content_words & query_words
    union = content_words | query_words

    return len(intersection) / len(union) if union else 0.0

# 新近性：指数衰减
def _calculate_recency(self, timestamp):
    import math
    age_hours = (datetime.now() - timestamp).total_seconds() / 3600
    decay_factor = 0.1
    recency_score = math.exp(-decay_factor * age_hours / 24)
    return max(0.1, min(1.0, recency_score))
```

#### 阶段三：Structure（结构化）

**目标**：组织成分区清晰的模板

```python
def _structure(self, selected_packets, user_query):
    # 按类型分组
    system_instructions = []
    evidence = []
    context = []

    for packet in selected_packets:
        packet_type = packet.metadata.get("type", "general")

        if packet_type == "system_instruction":
            system_instructions.append(packet.content)
        elif packet_type in ["rag_result", "knowledge"]:
            evidence.append(packet.content)
        else:
            context.append(packet.content)

    # 构建模板
    sections = []

    if system_instructions:
        sections.append("[Role & Policies]\n" + "\n".join(system_instructions))

    sections.append(f"[Task]\n{user_query}")

    if evidence:
        sections.append("[Evidence]\n" + "\n---\n".join(evidence))

    if context:
        sections.append("[Context]\n" + "\n".join(context))

    sections.append("[Output]\n请基于以上信息，提供准确、有据的回答。")

    return "\n\n".join(sections)
```

**输出示例**：

```
[Role & Policies]
你是一个专业的旅游规划助手。

[Task]
帮我规划北京三日游

[Evidence]
故宫：周一闭馆，门票60元
长城：八达岭段最受欢迎，建议早上前往

[Context]
User: 我喜欢历史文化
Agent: 那北京非常适合你

[Output]
请基于以上信息，提供准确、有据的回答。
```

#### 阶段四：Compress（压缩）

**目标**：当超限时进行智能压缩

```python
def _compress(self, context, max_tokens):
    current_tokens = self._count_tokens(context)

    if current_tokens <= max_tokens:
        return context  # 无需压缩

    # 使用 LLM 进行智能压缩
    compression_prompt = f"""
请将以下内容压缩到 {max_tokens} tokens 以内，保留关键信息：

{context}
"""

    compressed = self.llm.invoke(compression_prompt)
    return compressed
```

### 💡 完整使用示例

```python
from hello_agents import ContextBuilder, ContextConfig
from hello_agents.tools import MemoryTool, RAGTool

# 1. 配置
config = ContextConfig(
    max_tokens=3000,
    reserve_ratio=0.2,
    min_relevance=0.1,
    recency_weight=0.3,
    relevance_weight=0.7
)

# 2. 创建工具
memory_tool = MemoryTool()
rag_tool = RAGTool()

# 3. 创建 ContextBuilder
builder = ContextBuilder(
    config=config,
    memory_tool=memory_tool,
    rag_tool=rag_tool
)

# 4. 构建上下文
context = builder.build(
    user_query="帮我规划北京三日游",
    conversation_history=[
        Message(role="user", content="我喜欢历史"),
        Message(role="assistant", content="好的，记住了")
    ],
    system_instructions="你是旅游规划专家"
)

# 5. 使用上下文调用 LLM
response = llm.invoke(context)
```

---

## 5. 实战工具：NoteTool 和 TerminalTool

### 📝 NoteTool：结构化笔记工具

**用途**：让 Agent 维护持久化的"笔记本"

#### 核心功能

```python
from hello_agents.tools import NoteTool

# 创建笔记工具
note_tool = NoteTool(workspace="./notes")

# 1. 创建笔记
note_tool.execute("create", {
    "title": "北京旅游计划",
    "content": """
    ## 第一天：故宫 + 天安门
    - 上午：天安门广场
    - 下午：故宫博物院

    ## 第二天：长城
    - 八达岭长城（建议早上7点出发）

    ## 第三天：颐和园
    """,
    "tags": ["旅游", "北京", "计划"]
})

# 2. 搜索笔记
results = note_tool.execute("search", {
    "query": "长城",
    "tags": ["旅游"]
})

# 3. 更新笔记
note_tool.execute("update", {
    "note_id": "abc123",
    "content": "增加：需要提前预约门票"
})

# 4. 列出所有笔记
all_notes = note_tool.execute("list", {
    "tags": ["旅游"]
})
```

#### 实战场景：长期项目管理

```python
# Agent 在多天任务中使用 NoteTool

# Day 1: 创建项目笔记
note_tool.create("项目TODO", """
- [ ] 设计数据库schema
- [ ] 实现用户认证
- [ ] 编写API文档
""")

# Day 2: 更新进度
note_tool.update("项目TODO", """
- [x] 设计数据库schema ✅
- [ ] 实现用户认证（进行中）
- [ ] 编写API文档
""")

# Day 3: 查询历史
history = note_tool.search("数据库schema")
# Agent 能回忆起 Day 1 的设计决策
```

### 💻 TerminalTool：文件系统导航工具

**用途**：让 Agent 能像人类一样浏览文件系统

#### 核心功能

```python
from hello_agents.tools import TerminalTool

terminal = TerminalTool()

# 1. 列出文件
files = terminal.execute("ls", {"path": "./project"})
# 输出：["src/", "tests/", "README.md", "requirements.txt"]

# 2. 查看文件内容（前10行）
content = terminal.execute("head", {
    "file": "./project/README.md",
    "lines": 10
})

# 3. 搜索文件内容
matches = terminal.execute("grep", {
    "pattern": "TODO",
    "path": "./project/src"
})

# 4. 查找文件
found = terminal.execute("find", {
    "pattern": "*.py",
    "path": "./project"
})
```

#### 实战场景：代码库维护

```python
# Agent 探索代码库的过程

# Step 1: 查看项目结构
terminal.execute("tree", {"path": "./project", "max_depth": 2})

# Step 2: 搜索 TODO 标记
todos = terminal.execute("grep", {"pattern": "# TODO", "path": "./project/src"})

# Step 3: 查看具体文件
for file in todos:
    content = terminal.execute("cat", {"file": file})
    # Agent 分析代码并记录到 NoteTool
    note_tool.create(f"TODO-{file}", content)

# Step 4: 生成报告
report = note_tool.search("TODO")
```

### 🔄 NoteTool + TerminalTool 联动

**完整工作流示例**：

```python
# 任务：为代码库生成文档

# 1. 使用 TerminalTool 探索代码
project_structure = terminal.execute("tree", {"path": "./src"})

# 2. 记录到 NoteTool
note_tool.create("项目结构", project_structure)

# 3. 逐个分析文件
python_files = terminal.execute("find", {"pattern": "*.py", "path": "./src"})

for file in python_files:
    # 读取文件
    code = terminal.execute("cat", {"file": file})

    # 用 LLM 分析
    analysis = llm.invoke(f"分析这段代码的功能：\n{code}")

    # 记录分析结果
    note_tool.create(f"代码分析-{file}", analysis)

# 4. 生成最终文档
all_analyses = note_tool.list({"tags": ["代码分析"]})
final_doc = llm.invoke(f"根据以下分析，生成项目文档：\n{all_analyses}")
```

---

## 6. 本章总结

### 🎯 核心要点

#### 1. 上下文工程的本质

```
上下文工程 = 操作系统管理 RAM 的方式
          ↓
在有限的"注意力预算"中，精准选择和组织信息
```

**关键认知**：
- ✅ 上下文是**稀缺资源**，需要精心管理
- ✅ 长上下文 ≠ 好效果（上下文腐蚀）
- ✅ 提示工程是上下文工程的**子集**

#### 2. 四大策略

| 策略 | 核心思想 | 实现方法 |
|-----|---------|---------|
| **写入** | 信息持久化 | Scratchpad + Memory |
| **选择** | 精准过滤 | 相关性 + 新近性评分 |
| **压缩** | 减少冗余 | 摘要 + 修剪 |
| **隔离** | 关注点分离 | 多 Agent + 环境隔离 |

#### 3. GSSC 流水线

```
Gather（汇集）
   ↓
Select（选择）
   ↓
Structure（结构化）
   ↓
Compress（压缩）
   ↓
最优上下文 ✅
```

### 📊 对比：传统 RAG vs 上下文工程

| 维度 | 传统 RAG | 上下文工程（高级 RAG） |
|------|---------|---------------------|
| **检索** | 简单向量搜索 | 混合检索 + 重排序 |
| **过滤** | 基本/缺失 | 智能过滤 + 去重 |
| **排序** | 单一相似度 | 多维度评分（相关性+新近性+重要性） |
| **压缩** | 无 | 摘要 + 修剪 |
| **评估** | 缺失 | 系统化评估 + 迭代优化 |

### 💡 实践建议

#### 对于初学者

1. ✅ **先理解概念**：上下文工程 vs 提示工程
2. ✅ **体验 ContextBuilder**：运行示例代码，观察输出
3. ✅ **尝试调参**：修改 `relevance_weight`、`recency_weight`，观察效果变化

```python
# 实验：对比不同权重配置
config_a = ContextConfig(relevance_weight=0.9, recency_weight=0.1)  # 重视相关性
config_b = ContextConfig(relevance_weight=0.5, recency_weight=0.5)  # 平衡
config_c = ContextConfig(relevance_weight=0.1, recency_weight=0.9)  # 重视新鲜度
```

#### 对于进阶者

1. ✅ **优化评分算法**：用 Embedding 相似度替换 Jaccard
2. ✅ **实现自适应压缩**：根据任务类型动态调整压缩策略
3. ✅ **构建评估体系**：建立 Golden Dataset 评估上下文质量

```python
# 高级：使用 Embedding 计算相关性
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_relevance_advanced(content, query):
    embeddings = model.encode([content, query])
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    return similarity
```

#### 对于专业开发者

1. ✅ **生产化部署**：添加监控、日志、缓存
2. ✅ **A/B 测试**：对比不同上下文策略的效果
3. ✅ **性能优化**：并行检索、批量处理、结果缓存

```python
# 生产级：监控上下文质量
import logging

class ProductionContextBuilder(ContextBuilder):
    def build(self, user_query, **kwargs):
        start_time = time.time()

        # 构建上下文
        context = super().build(user_query, **kwargs)

        # 记录指标
        metrics = {
            "latency": time.time() - start_time,
            "context_tokens": self._count_tokens(context),
            "packets_gathered": len(self._last_gathered),
            "packets_selected": len(self._last_selected),
        }

        logging.info(f"Context build metrics: {metrics}")

        return context
```

### 🚀 未来展望

**上下文工程的演进方向**：

1. **自适应上下文**：根据任务类型自动调整策略
2. **多模态上下文**：整合文本、图像、音频等多种模态
3. **上下文缓存**：智能缓存高频查询的上下文
4. **上下文可视化**：可视化工具帮助调试和优化

### 🔗 相关资源

**论文与博客**：
- [Context Engineering for Agents](https://blog.langchain.com/context-engineering-for-agents/) - LangChain 官方博客
- [RAG is Dead, Context Engineering is King](https://www.latent.space/p/chroma) - Chroma CEO Jeff Huber 访谈
- [How Long Contexts Fail](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html) - Drew Breunig

**代码仓库**：
- [Hello-Agents GitHub](https://github.com/datawhalechina/hello-agents) - 官方代码仓库
- 本章代码：`code/chapter9/` 目录

**在线文档**：
- [Hello-Agents 在线教程](https://datawhalechina.github.io/hello-agents/)

---

## 📝 快速参考

### 安装

```bash
pip install "hello-agents[all]==0.2.7"
```

### 最小示例

```python
from hello_agents import ContextBuilder, ContextConfig, HelloAgentsLLM

# 1. 配置
config = ContextConfig(max_tokens=3000)

# 2. 创建 Builder
builder = ContextBuilder(config=config)

# 3. 构建上下文
context = builder.build(
    user_query="帮我规划北京三日游",
    system_instructions="你是旅游专家"
)

# 4. 调用 LLM
llm = HelloAgentsLLM()
response = llm.invoke(context)
print(response)
```

### 配置参数速查

```python
ContextConfig(
    max_tokens=3000,         # 最大token数（建议：2000-4000）
    reserve_ratio=0.2,       # 系统指令预留（建议：0.1-0.3）
    min_relevance=0.1,       # 最低相关性（建议：0.05-0.2）
    recency_weight=0.3,      # 新近性权重（建议：0.2-0.4）
    relevance_weight=0.7,    # 相关性权重（建议：0.6-0.8）
    enable_compression=True  # 启用压缩（建议：True）
)
```

---

## 🎓 章节习题提示

1. **概念理解**：解释"上下文腐蚀"现象及其原因
2. **算法分析**：对比 Jaccard 相似度和 Embedding 相似度的优缺点
3. **系统设计**：设计一个支持多模态（文本+图像）的 ContextBuilder
4. **性能优化**：如何减少上下文构建的延迟？
5. **实战应用**：为你的项目设计上下文工程策略

---

## 📌 核心要点回顾

```
🎯 什么是上下文工程？
   → 管理 LLM "工作记忆"的系统化学科

😱 为什么需要？
   → 上下文腐蚀 + 四大挑战（污染/分散/混淆/冲突）

🔧 四大策略
   → Write（写入）+ Select（选择）+ Compress（压缩）+ Isolate（隔离）

🏗️ GSSC 流水线
   → Gather → Select → Structure → Compress

🔨 实战工具
   → ContextBuilder + NoteTool + TerminalTool
```

---

**下一章预告**：第十章将深入探讨**智能体协议（MCP）**，学习如何让多个 Agent 高效协作、通信与编排！

**Happy Context Engineering! 🚀**
